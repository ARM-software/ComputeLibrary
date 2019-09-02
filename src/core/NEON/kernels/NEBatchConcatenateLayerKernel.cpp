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
#include "arm_compute/core/NEON/kernels/NEBatchConcatenateLayerKernel.h"

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

using namespace arm_compute;

namespace
{
template <typename T>
void batch_concat(const ITensor *in, ITensor *out, int batch_offset, const Window &window)
{
    // Offset input
    uint8_t *input_ptr = in->buffer() + in->info()->offset_first_element_in_bytes();

    // Offset output
    uint8_t *output_ptr = out->buffer() + out->info()->offset_first_element_in_bytes() + batch_offset * out->info()->strides_in_bytes()[3];

    Iterator input(in, window);
    Iterator output(out, window);

    const DataType                dt           = in->info()->data_type();
    const UniformQuantizationInfo input_qinfo  = in->info()->quantization_info().uniform();
    const UniformQuantizationInfo output_qinfo = out->info()->quantization_info().uniform();
    if(dt == DataType::QASYMM8 && input_qinfo != output_qinfo)
    {
        execute_window_loop(window, [&](const Coordinates &)
        {
            const auto in_ptr  = reinterpret_cast<const uint8_t *>(input_ptr + input.offset());
            const auto out_ptr = reinterpret_cast<uint8_t *>(output_ptr + output.offset());
            vst1q_u8(out_ptr, vquantize(vdequantize(vld1q_u8(in_ptr), input_qinfo), output_qinfo));
        },
        input, output);
    }
    else
    {
        execute_window_loop(window, [&](const Coordinates &)
        {
            const auto in_ptr  = reinterpret_cast<const T *>(input_ptr + input.offset());
            const auto out_ptr = reinterpret_cast<T *>(output_ptr + output.offset());

            wrapper::vstore(out_ptr, wrapper::vloadq(in_ptr));
        },
        input, output);
    }
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, unsigned int batch_offset, ITensorInfo *output)
{
    ARM_COMPUTE_UNUSED(batch_offset);

    const unsigned int num_elems_processed_per_iteration = 16 / input->element_size();

    // The window needs to be based on input as we copy all the batchs of input
    Window win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration));
    win.set(3, Window::Dimension(0, input->tensor_shape()[3], 1));

    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
    bool                   window_changed = update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

Status validate_arguments(const ITensorInfo *input, unsigned int batch_offset, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    //Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input) is not needed here as this kernel doesn't use NEON FP16 instructions.
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S8, DataType::QASYMM8,
                                                         DataType::U16, DataType::S16,
                                                         DataType::U32, DataType::S32,
                                                         DataType::F16, DataType::F32);
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
    : _func(nullptr), _input(nullptr), _output(nullptr), _batch_offset(0)
{
}

void NEBatchConcatenateLayerKernel::configure(const ITensor *input, unsigned int batch_offset, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), batch_offset, output->info()));

    _func         = nullptr;
    _input        = input;
    _output       = output;
    _batch_offset = batch_offset;

    switch(input->info()->data_type())
    {
        case DataType::S8:
        case DataType::U8:
        case DataType::QASYMM8:
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
    auto win_config = validate_and_configure_window(input->info(), batch_offset, output->info());
    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

    INEKernel::configure(std::get<1>(win_config));

    // Set output valid region
    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));
}

Status NEBatchConcatenateLayerKernel::validate(const arm_compute::ITensorInfo *input,
                                               unsigned int                    batch_offset,
                                               const arm_compute::ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, batch_offset, output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), batch_offset, output->clone().get()).first);
    return Status{};
}

void NEBatchConcatenateLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (*_func)(_input, _output, _batch_offset, window);
}
