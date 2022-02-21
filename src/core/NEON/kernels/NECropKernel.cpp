/*
 * Copyright (c) 2019-2022 Arm Limited.
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
#include "src/core/NEON/kernels/NECropKernel.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/helpers/tensor_transform.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/CPP/Validate.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/common/Registrars.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/utils/helpers/bit_ops.h"
#include "src/cpu/kernels/crop/list.h"

namespace arm_compute
{
namespace
{
struct CropSelectorData
{
    DataType dt;
};

using CropSelectorPtr = std::add_pointer<bool(const CropSelectorData &data)>::type;
using CropUKernelPtr  = std::add_pointer<void(const ITensor *, const ITensor *, float *, Coordinates, int32_t, int32_t, int32_t, bool, bool)>::type;

struct CropUKernel
{
    const char           *name;
    const CropSelectorPtr is_selected;
    CropUKernelPtr        ukernel;
};

static const CropUKernel available_kernels[] =
{
    {
        "fp16_neon_crop",
        [](const CropSelectorData & data) { return data.dt == DataType::F16; },
        REGISTER_FP16_NEON(arm_compute::cpu::fp16_in_bounds_crop_window)
    },
    {
        "f32_neon_crop",
        [](const CropSelectorData & data) { return data.dt == DataType::F32; },
        REGISTER_FP32_NEON(arm_compute::cpu::fp32_in_bounds_crop_window)
    },
    {
        "u8_neon_crop",
        [](const CropSelectorData & data) { return data.dt == DataType::U8; },
        REGISTER_INTEGER_NEON(arm_compute::cpu::u8_in_bounds_crop_window)
    },
    {
        "u16_neon_crop",
        [](const CropSelectorData & data) { return data.dt == DataType::U16; },
        REGISTER_INTEGER_NEON(arm_compute::cpu::u16_in_bounds_crop_window)
    },
    {
        "u32_neon_crop",
        [](const CropSelectorData & data) { return data.dt == DataType::U32; },
        REGISTER_INTEGER_NEON(arm_compute::cpu::u32_in_bounds_crop_window)
    },
    {
        "s8_neon_crop",
        [](const CropSelectorData & data) { return data.dt == DataType::S8; },
        REGISTER_INTEGER_NEON(arm_compute::cpu::s8_in_bounds_crop_window)
    },
    {
        "s16_neon_crop",
        [](const CropSelectorData & data) { return data.dt == DataType::S16; },
        REGISTER_INTEGER_NEON(arm_compute::cpu::s16_in_bounds_crop_window)
    },
    {
        "s32_neon_crop",
        [](const CropSelectorData & data) { return data.dt == DataType::S32; },
        REGISTER_INTEGER_NEON(arm_compute::cpu::s32_in_bounds_crop_window)
    },
};

/** Micro-kernel selector
 *
 * @param[in] data Selection data passed to help pick the appropriate micro-kernel
 *
 * @return A matching micro-kernel else nullptr
 */
const CropUKernel *get_implementation(const CropSelectorData &data)
{
    for(const auto &uk : available_kernels)
    {
        if(uk.is_selected(data))
        {
            return &uk;
        }
    }

    return nullptr;
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

inline void execute_window(const ITensor *input, const ITensor *output, Coordinates input_offset, float extrapolation_value,
                           const std::array<uint32_t, 2> &rows_out_of_bounds, const std::array<uint32_t, 2> &cols_out_of_bounds, NECropKernel::InBoundsCropFunction *in_bounds_crop_function,
                           bool is_height_flipped, bool has_cols_in_bounds, bool has_cols_out_of_bounds_before, bool has_cols_out_of_bounds_after, bool input_has_single_channel, bool is_width_flipped)
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
            (*in_bounds_crop_function)(input, output, output_ptr, input_offset, window_step_x, cols_out_of_bounds[0],
                                       output->info()->dimension(1) - cols_out_of_bounds[1], input_has_single_channel, is_width_flipped);
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
    : _input(nullptr), _crop_boxes(nullptr), _box_ind(nullptr), _output(nullptr), _start(), _end(), _crop_box_ind(0), _extrapolation_value(0), _rows_out_of_bounds(), _cols_out_of_bounds()
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
}

Status NECropKernel::validate(const ITensorInfo *input, const ITensorInfo *crop_boxes, const ITensorInfo *box_ind, const ITensorInfo *output, uint32_t crop_box_ind, float extrapolation_value)
{
    ARM_COMPUTE_UNUSED(extrapolation_value);
    const auto *uk = get_implementation(CropSelectorData{ input->data_type() });
    ARM_COMPUTE_RETURN_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::U16, DataType::S16, DataType::F16, DataType::U32, DataType::S32, DataType::F32);
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

    INEKernel::configure(calculate_max_window(*_output->info()));
}

void NECropKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(window, info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    ARM_COMPUTE_ERROR_ON(_input->info()->has_padding());
    ARM_COMPUTE_ERROR_ON(_output->info()->has_padding());

    const auto *uk = get_implementation(CropSelectorData{ _input->info()->data_type() });

    uint32_t    batch_index = *(reinterpret_cast<int32_t *>(_box_ind->ptr_to_element(Coordinates(_crop_box_ind))));
    Coordinates input_offset(0, _end[0] < _start[0] ? _start[0] - _cols_out_of_bounds[0] : _start[0] + _cols_out_of_bounds[0],
                             _end[1] < _start[1] ? _start[1] - _rows_out_of_bounds[0] : _start[1] + _rows_out_of_bounds[0], batch_index);
    execute_window(_input, _output, input_offset, _extrapolation_value, _rows_out_of_bounds, _cols_out_of_bounds, uk->ukernel, _end[1] < _start[1],
                   _cols_out_of_bounds[0] + _cols_out_of_bounds[1] < _output->info()->dimension(1), _cols_out_of_bounds[0] > 0, _cols_out_of_bounds[1] > 0,
                   _start[0] <= _end[0], _end[0] < _start[0]);
}
} // namespace arm_compute
