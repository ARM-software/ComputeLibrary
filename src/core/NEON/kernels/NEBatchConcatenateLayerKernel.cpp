/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#include "src/core/NEON/kernels/NEBatchConcatenateLayerKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "src/core/NEON/NEAsymm.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
namespace
{
template <typename T>
void batch_concat(const ITensor *in, ITensor *out, unsigned int batch_offset, const Window &window)
{
    // Offset input
    uint8_t *input_ptr = in->buffer() + in->info()->offset_first_element_in_bytes();

    // Offset output
    uint8_t *output_ptr = out->buffer() + out->info()->offset_first_element_in_bytes() + batch_offset * out->info()->strides_in_bytes()[3];

    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());
    const int  window_step_x  = 16 / out->info()->element_size();

    Window win{ window };
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    win.set(3, Window::Dimension(0, in->info()->tensor_shape()[3], 1));

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

            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                wrapper::vstore(out_ptr, vquantize(vdequantize(wrapper::vloadq(in_ptr), input_qinfo), output_qinfo));
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
                wrapper::vstore(out_ptr, vquantize_signed(vdequantize(wrapper::vloadq(in_ptr), input_qinfo), output_qinfo));
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

            int x = window_start_x;
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

Status validate_arguments(const ITensorInfo *input, unsigned int batch_offset, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    //Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input) is not needed here as this kernel doesn't use NEON FP16 instructions.
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(Window::DimX) != output->dimension(Window::DimX));
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(Window::DimY) != output->dimension(Window::DimY));
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(Window::DimZ) != output->dimension(Window::DimZ));
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(3) + batch_offset > output->dimension(3));
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(4, input, output);

    return Status{};
}
} // namespace

NEBatchConcatenateLayerKernel::NEBatchConcatenateLayerKernel()
    : _func(nullptr), _batch_offset(0)
{
}

void NEBatchConcatenateLayerKernel::configure(const ITensorInfo *input, unsigned int batch_offset, ITensorInfo *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input, batch_offset, output));

    _func         = nullptr;
    _batch_offset = batch_offset;

    switch(input->data_type())
    {
        case DataType::S8:
        case DataType::U8:
        case DataType::QASYMM8:
        case DataType::QASYMM8_SIGNED:
            _func = &batch_concat<uint8_t>;
            break;
        case DataType::S16:
        case DataType::U16:
        case DataType::F16:
            _func = &batch_concat<uint16_t>;
            break;
        case DataType::S32:
        case DataType::U32:
        case DataType::F32:
            _func = &batch_concat<uint32_t>;
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

Status NEBatchConcatenateLayerKernel::validate(const arm_compute::ITensorInfo *input,
                                               unsigned int                    batch_offset,
                                               const arm_compute::ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, batch_offset, output));
    return Status{};
}

void NEBatchConcatenateLayerKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (*_func)(tensors.get_const_tensor(TensorType::ACL_SRC),
             tensors.get_tensor(TensorType::ACL_DST),
             _batch_offset,
             window);
}
} // namespace arm_compute
