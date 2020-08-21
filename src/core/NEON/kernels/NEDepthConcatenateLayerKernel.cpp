/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/core/NEON/kernels/NEDepthConcatenateLayerKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEAsymm.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <cstdint>

namespace arm_compute
{
namespace
{
template <typename T>
void depth_concat(const ITensor *in, ITensor *out, unsigned int depth_offset, const Window &window)
{
    // Offset input
    uint8_t *input_ptr = in->buffer() + in->info()->offset_first_element_in_bytes();

    // Offset output
    uint8_t *output_ptr = out->buffer() + out->info()->offset_first_element_in_bytes() + depth_offset * out->info()->strides_in_bytes()[2];

    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());
    const int  window_step_x  = 16 / out->info()->element_size();

    Window win{ window };
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    win.set(Window::DimZ, Window::Dimension(0, in->info()->tensor_shape().z(), 1));

    Iterator input(in, win);
    Iterator output(out, win);

    const DataType                dt           = in->info()->data_type();
    const UniformQuantizationInfo input_qinfo  = in->info()->quantization_info().uniform();
    const UniformQuantizationInfo output_qinfo = out->info()->quantization_info().uniform();
    if(dt == DataType::QASYMM8 && input_qinfo != output_qinfo)
    {
        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto in_ptr  = reinterpret_cast<const uint8_t *>(input_ptr + input.offset());
            const auto out_ptr = reinterpret_cast<uint8_t *>(output_ptr + output.offset());
            int        x       = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                wrapper::vstore(out_ptr + x, vquantize(vdequantize(wrapper::vloadq(in_ptr + x), input_qinfo), output_qinfo));
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                *(out_ptr + x) = quantize_qasymm8(dequantize_qasymm8(*(in_ptr + x), input_qinfo), output_qinfo);
            }
        },
        input, output);
    }
    else if(dt == DataType::QASYMM8_SIGNED && input_qinfo != output_qinfo)
    {
        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto in_ptr  = reinterpret_cast<const int8_t *>(input_ptr + input.offset());
            const auto out_ptr = reinterpret_cast<int8_t *>(output_ptr + output.offset());
            int        x       = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                wrapper::vstore(out_ptr + x, vquantize_signed(vdequantize(wrapper::vloadq(in_ptr + x), input_qinfo), output_qinfo));
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                *(out_ptr + x) = quantize_qasymm8_signed(dequantize_qasymm8_signed(*(in_ptr + x), input_qinfo), output_qinfo);
            }
        },
        input, output);
    }
    else
    {
        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto in_ptr  = reinterpret_cast<const T *>(input_ptr + input.offset());
            const auto out_ptr = reinterpret_cast<T *>(output_ptr + output.offset());
            int        x       = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                wrapper::vstore(out_ptr + x, wrapper::vloadq(in_ptr + x));
            }
            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                *(out_ptr + x) = *(in_ptr + x);
            }
        },
        input, output);
    }
}

Status validate_arguments(const ITensorInfo *input, unsigned int depth_offset, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    //Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input) is not needed here as this kernel doesn't use NEON FP16 instructions.
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(Window::DimX) != output->dimension(Window::DimX));
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(Window::DimY) != output->dimension(Window::DimY));
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(2) + depth_offset > output->dimension(2));
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(3, input, output);

    return Status{};
}
} // namespace

NEDepthConcatenateLayerKernel::NEDepthConcatenateLayerKernel()
    : _func(nullptr), _depth_offset(0)
{
}

void NEDepthConcatenateLayerKernel::configure(const ITensorInfo *input, unsigned int depth_offset, ITensorInfo *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input, depth_offset, output));

    _func         = nullptr;
    _depth_offset = depth_offset;

    switch(input->data_type())
    {
        case DataType::QASYMM8:
            _func = &depth_concat<uint8_t>;
            break;
        case DataType::QASYMM8_SIGNED:
            _func = &depth_concat<int8_t>;
            break;
        case DataType::F16:
            _func = &depth_concat<uint16_t>;
            break;
        case DataType::F32:
            _func = &depth_concat<uint32_t>;
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data type.");
    }

    // Configure kernel window
    Window      win = calculate_max_window(*output, Steps());
    Coordinates coord;
    coord.set_num_dimensions(output->num_dimensions());

    output->set_valid_region(ValidRegion(coord, output->tensor_shape()));
    INEKernel::configure(win);
}

Status NEDepthConcatenateLayerKernel::validate(const arm_compute::ITensorInfo *input,
                                               unsigned int                    depth_offset,
                                               const arm_compute::ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, depth_offset, output));
    return Status{};
}

void NEDepthConcatenateLayerKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (*_func)(tensors.get_const_tensor(TensorType::ACL_SRC),
             tensors.get_tensor(TensorType::ACL_DST),
             _depth_offset,
             window);
}
} // namespace arm_compute
