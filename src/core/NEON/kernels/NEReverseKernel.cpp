/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "src/core/NEON/kernels/NEReverseKernel.h"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *axis)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output, axis);
    //Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input) is not needed here as this kernel doesn't use Neon FP16 instructions.
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(axis, 1, DataType::U32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis->num_dimensions() > 1, "Axis must be a 1D tensor");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis->dimension(0) > 4, "Only up to 4 dimensions can be reversed");

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    return Status{};
}
} // namespace

NEReverseKernel::NEReverseKernel()
    : _input(nullptr), _output(nullptr), _axis(nullptr)
{
}

void NEReverseKernel::configure(const ITensor *input, ITensor *output, const ITensor *axis)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, axis);

    _input  = input;
    _output = output;
    _axis   = axis;

    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), *input->info()->clone());

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), axis->info()));

    // Configure kernel window
    INEKernel::configure(calculate_max_window(*output->info()));
}

Status NEReverseKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *axis)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, axis));

    return Status{};
}

template <typename T>
void run_reverse(const Window &window, const ITensor *input, const ITensor *axis, ITensor *output)
{
    int axis_bit = 0;
    for(unsigned int i = 0; i < axis->info()->dimension(0); ++i)
    {
        const int axis_i = *(reinterpret_cast<const int *>(axis->buffer()) + i);
        axis_bit |= 1 << axis_i;
    }

    // Check if we need a left-over loop for the y dimension
    const int window_step_x  = 16 / input->info()->element_size();
    const int window_start_x = window.x().start();
    const int window_end_x   = window.x().end();

    Window win(window);
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input_it(input, win);
    execute_window_loop(win, [&](const Coordinates & id)
    {
        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            auto in = wrapper::vloadq(reinterpret_cast<T *>(input_it.ptr()) + x);

            // Reverse 0 axis
            if(axis_bit & 0x1)
            {
                in = wrapper::vrev64(in);
                in = wrapper::vcombine(wrapper::vgethigh(in), wrapper::vgetlow(in));
            }

            const int offset_x = (axis_bit & 0x1) ? output->info()->dimension(0) - x - window_step_x : x;
            const int offset_y = (axis_bit & 0x2) ? output->info()->dimension(1) - id.y() - 1 : id.y();
            const int offset_z = (axis_bit & 0x4) ? output->info()->dimension(2) - id.z() - 1 : id.z();
            const int offset_w = (axis_bit & 0x8) ? output->info()->dimension(3) - id[3] - 1 : id[3];

            auto out_ptr = reinterpret_cast<T *>(output->ptr_to_element(Coordinates(offset_x, offset_y, offset_z, offset_w)));
            wrapper::vstore(out_ptr, in);
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            const auto in = *(reinterpret_cast<T *>(input_it.ptr()) + x);

            const int offset_x = (axis_bit & 0x1) ? output->info()->dimension(0) - x - 1 : x;
            const int offset_y = (axis_bit & 0x2) ? output->info()->dimension(1) - id.y() - 1 : id.y();
            const int offset_z = (axis_bit & 0x4) ? output->info()->dimension(2) - id.z() - 1 : id.z();
            const int offset_w = (axis_bit & 0x8) ? output->info()->dimension(3) - id[3] - 1 : id[3];

            *reinterpret_cast<T *>(output->ptr_to_element(Coordinates(offset_x, offset_y, offset_z, offset_w))) = in;
        }
    },
    input_it);
}

void NEReverseKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    switch(_input->info()->element_size())
    {
        case 4:
            run_reverse<uint32_t>(window, _input, _axis, _output);
            break;
        case 2:
            run_reverse<uint16_t>(window, _input, _axis, _output);
            break;
        case 1:
            run_reverse<uint8_t>(window, _input, _axis, _output);
            break;
        default:
            ARM_COMPUTE_ERROR("Element size not supported");
    }
}
} // namespace arm_compute
