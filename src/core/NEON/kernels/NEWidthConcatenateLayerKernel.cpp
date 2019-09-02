/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEWidthConcatenateLayerKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEAsymm.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <cstdint>

using namespace arm_compute;

namespace
{
std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, unsigned int width_offset, ITensorInfo *output)
{
    const unsigned int num_elems_processed_per_iteration = 16 / output->element_size();

    // The window needs to be based on input as we copy all the widths of input
    Window                 win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output, width_offset, num_elems_processed_per_iteration);
    bool                   window_changed = update_window_and_padding(win, input_access, output_access);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

Status validate_arguments(const ITensorInfo *input, unsigned int width_offset, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    // Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input) is not needed here as this kernel doesn't use NEON FP16 instructions.
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1,
                                                         DataType::U8, DataType::S8, DataType::QASYMM8,
                                                         DataType::U16, DataType::S16, DataType::F16,
                                                         DataType::U32, DataType::S32, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(0) + width_offset > output->dimension(0));

    for(size_t i = 1; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(i) != output->dimension(i));
    }

    return Status{};
}
} // namespace

NEWidthConcatenateLayerKernel::NEWidthConcatenateLayerKernel()
    : _input(nullptr), _output(nullptr), _width_offset(0)
{
}

void NEWidthConcatenateLayerKernel::configure(const ITensor *input, unsigned int width_offset, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), width_offset, output->info()));

    _input        = input;
    _output       = output;
    _width_offset = width_offset;

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), width_offset, output->info());
    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

    INEKernel::configure(std::get<1>(win_config));

    // Set output valid region
    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));
}

Status NEWidthConcatenateLayerKernel::validate(const ITensorInfo *input, unsigned int width_offset, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, width_offset, output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), width_offset, output->clone().get()).first);
    return Status{};
}

void NEWidthConcatenateLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    // Offset output pointer to the correct position
    uint8_t *output_ptr = _output->buffer() + _output->info()->offset_first_element_in_bytes() + _width_offset * _output->info()->strides_in_bytes()[0];

    // Create iterators
    Iterator                       input(_input, window);
    Iterator                       output(_output, window);
    const DataType                 dt           = _input->info()->data_type();
    const UniformQuantizationInfo &input_qinfo  = _input->info()->quantization_info().uniform();
    const UniformQuantizationInfo &output_qinfo = _output->info()->quantization_info().uniform();
    if(dt == DataType::QASYMM8 && input_qinfo != output_qinfo)
    {
        execute_window_loop(window, [&](const Coordinates &)
        {
            vst1q_u8(output_ptr + output.offset(), vquantize(vdequantize(vld1q_u8(input.ptr()), input_qinfo), output_qinfo));
        },
        input, output);
    }
    else
    {
        execute_window_loop(window, [&](const Coordinates &)
        {
            const auto in_ptr  = input.ptr();
            const auto out_ptr = output_ptr + output.offset();

            wrapper::vstore(out_ptr, wrapper::vloadq(in_ptr));
        },
        input, output);
    }
}
