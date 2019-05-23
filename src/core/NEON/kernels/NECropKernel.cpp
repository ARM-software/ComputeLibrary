/*
 * Copyright (c) 2019 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/core/NEON/kernels/NECropKernel.h"

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Window.h"

#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/helpers/bit_ops.h"
#include "arm_compute/core/utils/helpers/tensor_transform.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include <map>

namespace arm_compute
{
namespace
{
template <typename T>
inline float32x4_t load_as_f32(T *ptr)
{
    ARM_COMPUTE_UNUSED(ptr);
    ARM_COMPUTE_ERROR("Type not supported.");
}

template <>
inline float32x4_t load_as_f32(float *ptr)
{
    return wrapper::vloadq(ptr);
}

template <>
inline float32x4_t load_as_f32(int32_t *ptr)
{
    return vcvtq_f32_s32(wrapper::vloadq(ptr));
}

template <>
inline float32x4_t load_as_f32(uint32_t *ptr)
{
    return vcvtq_f32_u32(wrapper::vloadq(ptr));
}

template <>
inline float32x4_t load_as_f32(int16_t *ptr)
{
    return vcvtq_f32_s32(vmovl_s16(wrapper::vload(ptr)));
}

template <>
inline float32x4_t load_as_f32(uint16_t *ptr)
{
    return vcvtq_f32_u32(vmovl_u16(wrapper::vload(ptr)));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <>
inline float32x4_t load_as_f32(float16_t *ptr)
{
    return vcvt_f32_f16(wrapper::vload(ptr));
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

template <typename T, bool input_has_single_channel, bool is_width_flipped>
inline void in_bounds_crop_window(const ITensor *input, const ITensor *output, float *output_ptr, Coordinates input_offset,
                                  int32_t window_step_x, int32_t output_width_start, int32_t output_width_limit)
{
    // Reverse elements if width flipped.
    if(is_width_flipped)
    {
        // Collapse first dimension if possible.
        if(input_has_single_channel)
        {
            int32_t     x = output_width_start;
            Coordinates negative_offset(input_offset);
            negative_offset.set(1, negative_offset[1] - window_step_x + 1);
            for(; x <= output_width_limit - window_step_x; x += window_step_x, negative_offset[1] -= window_step_x)
            {
                auto in = load_as_f32(reinterpret_cast<T *>(input->ptr_to_element(negative_offset)));

                in = wrapper::vrev64(in);
                in = wrapper::vcombine(wrapper::vgethigh(in), wrapper::vgetlow(in));

                wrapper::vstore(output_ptr + x, in);
            }
            input_offset[1] = negative_offset[1] + window_step_x - 1;
            for(; x < output_width_limit; ++x, --input_offset[1])
            {
                *(output_ptr + x) = static_cast<float>(*reinterpret_cast<T *>(input->ptr_to_element(input_offset)));
            }
        }
        else
        {
            for(int32_t x = output_width_start; x < output_width_limit; ++x, --input_offset[1])
            {
                input_offset.set(0, 0);
                int32_t c = 0;
                for(; c <= static_cast<int32_t>(input->info()->dimension(0)) - window_step_x; c += window_step_x, input_offset[0] += window_step_x)
                {
                    auto in = load_as_f32(reinterpret_cast<T *>(input->ptr_to_element(input_offset)));
                    wrapper::vstore(output_ptr + x * output->info()->dimension(0) + c, in);
                }
                for(; c < static_cast<int32_t>(input->info()->dimension(0)); ++c, ++input_offset[0])
                {
                    *(output_ptr + x * output->info()->dimension(0) + c) = static_cast<float>(*reinterpret_cast<T *>(input->ptr_to_element(input_offset)));
                }
            }
        }
    }
    else
    {
        // Use memcpy if the elements don't need converting to float.
        if(std::is_same<T, float>::value)
        {
            memcpy(static_cast<void *>(output_ptr + output_width_start * output->info()->dimension(0)),
                   reinterpret_cast<const void *>(input->ptr_to_element(input_offset)),
                   (output_width_limit - output_width_start) * output->info()->dimension(0) * output->info()->element_size());
        }
        else
        {
            int32_t x                = 0;
            int32_t limit            = (output_width_limit - output_width_start) * static_cast<int32_t>(output->info()->dimension(0));
            float *output_start_ptr = output_ptr + output_width_start * output->info()->dimension(0);
            for(; x <= limit - window_step_x; x += window_step_x, input_offset[0] += window_step_x)
            {
                auto in = load_as_f32(reinterpret_cast<T *>(input->ptr_to_element(input_offset)));
                wrapper::vstore(output_start_ptr + x, in);
            }
            for(; x < limit; ++x, ++input_offset[0])
            {
                *(output_start_ptr + x) = static_cast<float>(*reinterpret_cast<T *>(input->ptr_to_element(input_offset)));
            }
        }
    }
}

inline void out_of_bounds_crop_window(const ITensor *output, float *output_ptr, float extrapolation_value,
                                      int32_t window_step_x, int32_t output_width_start, int32_t output_width_limit)
{
    auto    in               = wrapper::vdup_n(extrapolation_value, wrapper::traits::vector_128_tag());
    int32_t x                = 0;
    int32_t limit            = (output_width_limit - output_width_start) * static_cast<int32_t>(output->info()->dimension(0));
    float *output_start_ptr = output_ptr + output_width_start * output->info()->dimension(0);
    for(; x <= limit - window_step_x; x += window_step_x)
    {
        wrapper::vstore(output_start_ptr + x, in);
    }
    for(; x < limit; ++x)
    {
        *(output_start_ptr + x) = extrapolation_value;
    }
}

template <bool is_height_flipped, bool has_cols_in_bounds, bool has_cols_out_of_bounds_before, bool has_cols_out_of_bounds_after>
inline void execute_window(const ITensor *input, const ITensor *output, Coordinates input_offset, float extrapolation_value,
                           const std::array<uint32_t, 2> &rows_out_of_bounds, const std::array<uint32_t, 2> &cols_out_of_bounds, NECropKernel::InBoundsCropFunction *in_bounds_crop_function)
{
    // Output is always float.
    const int window_step_x = 16 / sizeof(float);
    auto     *output_ptr    = reinterpret_cast<float *>(output->buffer());
    //  Output window:
    //  --------------------------------
    //  |          Out of bounds       |
    //  |          rows before         |
    //  |------------------------------|
    //  | Out of | In         | Out of |
    //  | bounds | bounds     | bounds |
    //  | cols   | elements   | cols   |
    //  | before | copied     | after  |
    //  |        | from input |        |
    //  --------------------------------
    //  |        Out of bounds         |
    //  |        rows after            |
    //  |------------------------------|
    // Fill all output rows that have no elements that are within the input bounds with the extrapolation value.
    // First for the rows before the in bounds rows.
    out_of_bounds_crop_window(output, output_ptr, extrapolation_value, window_step_x, 0, rows_out_of_bounds[0] * output->info()->dimension(1));
    output_ptr += rows_out_of_bounds[0] * output->info()->dimension(1) * output->info()->dimension(0);
    // Iterate through each row that has any elements within the input bounds.
    for(uint32_t row = rows_out_of_bounds[0]; static_cast<int32_t>(row) < static_cast<int32_t>(output->info()->dimension(2) - rows_out_of_bounds[1]);
        ++row, is_height_flipped ? --input_offset[2] : ++input_offset[2])
    {
        // Fill all elements in the row that are out of bounds with the extrapolation value.
        // First for the elements before the in bounds elements.
        if(has_cols_out_of_bounds_before)
        {
            out_of_bounds_crop_window(output, output_ptr, extrapolation_value, window_step_x, 0, cols_out_of_bounds[0]);
        }
        // Copy all elements within the input bounds from the input tensor.
        if(has_cols_in_bounds)
        {
            (*in_bounds_crop_function)(input, output, output_ptr, input_offset, window_step_x, cols_out_of_bounds[0], output->info()->dimension(1) - cols_out_of_bounds[1]);
        }
        // Fill all elements after the in bounds elements with the extrapolation value.
        if(has_cols_out_of_bounds_after)
        {
            out_of_bounds_crop_window(output, output_ptr, extrapolation_value, window_step_x, output->info()->dimension(1) - cols_out_of_bounds[1], output->info()->dimension(1));
        }
        output_ptr += output->info()->dimension(1) * output->info()->dimension(0);
    }
    // Fill all rows after the in bounds elements with the extrapolation value.
    out_of_bounds_crop_window(output, output_ptr, extrapolation_value, window_step_x, 0, rows_out_of_bounds[1] * output->info()->dimension(1));
}
} // namespace

NECropKernel::NECropKernel()
    : _input(nullptr), _crop_boxes(nullptr), _box_ind(nullptr), _output(nullptr), _start(), _end(), _crop_box_ind(0), _extrapolation_value(0), _rows_out_of_bounds(), _cols_out_of_bounds(),
      _in_bounds_crop_functions(), _in_bounds_crop_function(nullptr), _crop_function(nullptr)
{
}

void NECropKernel::configure(const ITensor *input, const ITensor *crop_boxes, const ITensor *box_ind, ITensor *output, uint32_t crop_box_ind, float extrapolation_value)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate(input->info(), crop_boxes->info(), box_ind->info(), output->info(), crop_box_ind, extrapolation_value));

    _input               = input;
    _crop_boxes          = crop_boxes;
    _box_ind             = box_ind;
    _output              = output;
    _crop_box_ind        = crop_box_ind;
    _extrapolation_value = extrapolation_value;

    const static std::map<std::pair<DataType, bool>, std::pair<NECropKernel::InBoundsCropFunction *, NECropKernel::InBoundsCropFunction *>> in_map_function =
    {
        { { DataType::F32, false }, { &in_bounds_crop_window<float, false, false>, &in_bounds_crop_window<float, false, true> } },
        { { DataType::F32, true }, { &in_bounds_crop_window<float, true, false>, &in_bounds_crop_window<float, true, true> } },
        { { DataType::U16, false }, { &in_bounds_crop_window<uint16_t, false, false>, &in_bounds_crop_window<uint16_t, false, true> } },
        { { DataType::U16, true }, { &in_bounds_crop_window<uint16_t, true, false>, &in_bounds_crop_window<uint16_t, true, true> } },
        { { DataType::S16, false }, { &in_bounds_crop_window<int16_t, false, false>, &in_bounds_crop_window<int16_t, false, true> } },
        { { DataType::S16, true }, { &in_bounds_crop_window<int16_t, true, false>, &in_bounds_crop_window<int16_t, true, true> } },
        { { DataType::U32, false }, { &in_bounds_crop_window<uint32_t, false, false>, &in_bounds_crop_window<uint32_t, false, true> } },
        { { DataType::U32, true }, { &in_bounds_crop_window<uint32_t, true, false>, &in_bounds_crop_window<uint32_t, true, true> } },
        { { DataType::S32, false }, { &in_bounds_crop_window<int32_t, false, false>, &in_bounds_crop_window<int32_t, false, true> } },
        { { DataType::S32, true }, { &in_bounds_crop_window<int32_t, true, false>, &in_bounds_crop_window<int32_t, true, true> } },
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        { { DataType::F16, false }, { &in_bounds_crop_window<float16_t, false, false>, &in_bounds_crop_window<float16_t, false, true> } },
        { { DataType::F16, false }, { &in_bounds_crop_window<float16_t, true, false>, &in_bounds_crop_window<float16_t, true, true> } }
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    };

    auto in_it = in_map_function.find({ input->info()->data_type(), input->info()->dimension(0) == 1 });

    if(in_it != in_map_function.end())
    {
        _in_bounds_crop_functions = in_it->second;
    }
}

Status NECropKernel::validate(const ITensorInfo *input, const ITensorInfo *crop_boxes, const ITensorInfo *box_ind, const ITensorInfo *output, uint32_t crop_box_ind, float extrapolation_value)
{
    ARM_COMPUTE_UNUSED(extrapolation_value);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U16, DataType::S16, DataType::F16, DataType::U32, DataType::S32, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(input, DataLayout::NHWC);
    ARM_COMPUTE_RETURN_ERROR_ON(input->tensor_shape().num_dimensions() > 4);
    ARM_COMPUTE_RETURN_ERROR_ON(crop_boxes->tensor_shape()[0] != 4);
    ARM_COMPUTE_RETURN_ERROR_ON(crop_boxes->tensor_shape()[1] != box_ind->tensor_shape()[0]);
    ARM_COMPUTE_RETURN_ERROR_ON(crop_boxes->tensor_shape()[1] <= crop_box_ind);
    ARM_COMPUTE_RETURN_ERROR_ON(box_ind->tensor_shape()[0] <= crop_box_ind);
    if(output->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(output, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON(output->num_dimensions() != 3);
        ARM_COMPUTE_RETURN_ERROR_ON(output->has_padding());
    }
    return Status{};
}

void NECropKernel::configure_output_shape()
{
    // _crop_box_ind is used to index _crop_boxes and retrieve the appropriate crop box.
    // The crop box is specified by normalized coordinates [y0, x0, y1, x1].
    const float x0 = *reinterpret_cast<const float *>(_crop_boxes->ptr_to_element(Coordinates(1, _crop_box_ind)));
    const float y0 = *reinterpret_cast<const float *>(_crop_boxes->ptr_to_element(Coordinates(0, _crop_box_ind)));
    const float x1 = *reinterpret_cast<const float *>(_crop_boxes->ptr_to_element(Coordinates(3, _crop_box_ind)));
    const float y1 = *reinterpret_cast<const float *>(_crop_boxes->ptr_to_element(Coordinates(2, _crop_box_ind)));
    // The normalized coordiantes are scaled to retrieve the floating point image coordinates which are rounded to integers.
    _start = Coordinates(std::floor(x0 * (_input->info()->tensor_shape()[1] - 1) + 0.5f),
                         std::floor(y0 * (_input->info()->tensor_shape()[2] - 1) + 0.5f));
    _end = Coordinates(std::floor(x1 * (_input->info()->tensor_shape()[1] - 1) + 0.5f),
                       std::floor(y1 * (_input->info()->tensor_shape()[2] - 1) + 0.5f));
    const TensorShape out_shape(_input->info()->tensor_shape()[0], abs(_end[0] - _start[0]) + 1, abs(_end[1] - _start[1]) + 1);
    _output->info()->set_tensor_shape(out_shape);

    _in_bounds_crop_function = _start[0] <= _end[0] ? _in_bounds_crop_functions.first : _in_bounds_crop_functions.second;

    bool is_width_flipped  = _end[0] < _start[0];
    bool is_height_flipped = _end[1] < _start[1];
    if(is_height_flipped)
    {
        _rows_out_of_bounds[0] = _start[1] >= static_cast<int32_t>(_input->info()->dimension(2)) ? std::min(static_cast<uint32_t>(_start[1] - _input->info()->dimension(2) + 1),
                                                                                                            static_cast<uint32_t>(_output->info()->dimension(2))) :
                                 0;
        _rows_out_of_bounds[1] = _end[1] < 0 ? std::min(static_cast<uint32_t>(-_end[1]),
                                                        static_cast<uint32_t>(_output->info()->dimension(2))) :
                                 0;
    }
    else
    {
        _rows_out_of_bounds[0] = _start[1] < 0 ? std::min(static_cast<uint32_t>(-_start[1]),
                                                          static_cast<uint32_t>(_output->info()->dimension(2))) :
                                 0;
        _rows_out_of_bounds[1] = _end[1] >= static_cast<int32_t>(_input->info()->dimension(2)) ? std::min(static_cast<uint32_t>(_end[1] - _input->info()->dimension(2) + 1),
                                                                                                          static_cast<uint32_t>(_output->info()->dimension(2))) :
                                 0;
    }
    if(is_width_flipped)
    {
        _cols_out_of_bounds[0] = _start[0] >= static_cast<int32_t>(_input->info()->dimension(1)) ? std::min(static_cast<uint32_t>(_start[0] - _input->info()->dimension(1) + 1),
                                                                                                            static_cast<uint32_t>(_output->info()->dimension(1))) :
                                 0;
        _cols_out_of_bounds[1] = _end[0] < 0 ? std::min(static_cast<uint32_t>(-_end[0]),
                                                        static_cast<uint32_t>(_output->info()->dimension(1))) :
                                 0;
    }
    else
    {
        _cols_out_of_bounds[0] = _start[0] < 0 ? std::min(static_cast<uint32_t>(-_start[0]),
                                                          static_cast<uint32_t>(_output->info()->dimension(1))) :
                                 0;
        _cols_out_of_bounds[1] = _end[0] >= static_cast<int32_t>(_input->info()->dimension(1)) ? std::min(static_cast<uint32_t>(_end[0] - _input->info()->dimension(1) + 1),
                                                                                                          static_cast<uint32_t>(_output->info()->dimension(1))) :
                                 0;
    }

    const static std::map<std::tuple<bool, bool, bool, bool>, NECropKernel::CropFunction *> map_function =
    {
        { std::make_tuple(false, false, false, false), &execute_window<false, false, false, false> },
        { std::make_tuple(false, false, false, true), &execute_window<false, false, false, true> },
        { std::make_tuple(false, false, true, false), &execute_window<false, false, true, false> },
        { std::make_tuple(false, false, true, true), &execute_window<false, false, true, true> },
        { std::make_tuple(false, true, false, false), &execute_window<false, true, false, false> },
        { std::make_tuple(false, true, false, true), &execute_window<false, true, false, true> },
        { std::make_tuple(false, true, true, false), &execute_window<false, true, true, false> },
        { std::make_tuple(false, true, true, true), &execute_window<false, true, true, true> },
        { std::make_tuple(true, false, false, false), &execute_window<true, false, false, false> },
        { std::make_tuple(true, false, false, true), &execute_window<true, false, false, true> },
        { std::make_tuple(true, false, true, false), &execute_window<true, false, true, false> },
        { std::make_tuple(true, false, true, true), &execute_window<true, false, true, true> },
        { std::make_tuple(true, true, false, false), &execute_window<true, true, false, false> },
        { std::make_tuple(true, true, false, true), &execute_window<true, true, false, true> },
        { std::make_tuple(true, true, true, false), &execute_window<true, true, true, false> },
        { std::make_tuple(true, true, true, true), &execute_window<true, true, true, true> },
    };

    auto it = map_function.find(std::make_tuple(is_height_flipped,
                                                _cols_out_of_bounds[0] + _cols_out_of_bounds[1] < _output->info()->dimension(1),
                                                _cols_out_of_bounds[0] > 0,
                                                _cols_out_of_bounds[1] > 0));

    if(it != map_function.end())
    {
        _crop_function = it->second;
    }

    INEKernel::configure(calculate_max_window(*_output->info()));
}

void NECropKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(window, info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    ARM_COMPUTE_ERROR_ON(_input->info()->has_padding());
    ARM_COMPUTE_ERROR_ON(_output->info()->has_padding());

    uint32_t    batch_index = *(reinterpret_cast<int32_t *>(_box_ind->ptr_to_element(Coordinates(_crop_box_ind))));
    Coordinates input_offset(0, _end[0] < _start[0] ? _start[0] - _cols_out_of_bounds[0] : _start[0] + _cols_out_of_bounds[0],
                             _end[1] < _start[1] ? _start[1] - _rows_out_of_bounds[0] : _start[1] + _rows_out_of_bounds[0], batch_index);
    (*_crop_function)(_input, _output, input_offset, _extrapolation_value, _rows_out_of_bounds, _cols_out_of_bounds, _in_bounds_crop_function);
}
} // namespace arm_compute
