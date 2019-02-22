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
#include "arm_compute/core/NEON/kernels/NEReverseKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEAsymm.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "arm_compute/core/QAsymm8.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>
#include <array>
#include <cmath>
#include <map>

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *axis)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output, axis);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S8, DataType::QASYMM8,
                                                         DataType::U16, DataType::S16,
                                                         DataType::U32, DataType::S32,
                                                         DataType::F16, DataType::F32);
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
    const int window_step_x            = 16 / input->info()->element_size();
    const int window_start_x           = window.x().start();
    const int window_end_x             = std::min(window.x().end(), static_cast<int>(input->info()->dimension(0)));
    const int window_end_x_multiple_of = ((window_end_x - window_start_x) / window_step_x) * window_step_x;
    bool      left_over_loop_x         = (((window_end_x - window_start_x) % window_step_x) != 0);

    Window slice = window.first_slice_window_4D();

    if(left_over_loop_x)
    {
        // Check if window_end_y_multiple_of is greater than window_start_y
        if(window_end_x_multiple_of > window_start_x)
        {
            slice.set(Window::DimX, Window::Dimension(window_start_x, window_end_x_multiple_of, window_step_x));
        }
        else
        {
            slice.set(Window::DimX, Window::Dimension(0, 0, 1));
        }
    }

    do
    {
        Iterator input_it(input, slice);
        execute_window_loop(slice, [&](const Coordinates & id)
        {
            auto in = wrapper::vloadq(reinterpret_cast<T *>(input_it.ptr()));

            // Reverse 0 axis
            if(axis_bit & 0x1)
            {
                in = wrapper::vrev64(in);
                in = wrapper::vcombine(wrapper::vgethigh(in), wrapper::vgetlow(in));
            }

            const int offset_x = (axis_bit & 0x1) ? output->info()->dimension(0) - id.x() - window_step_x : id.x();
            const int offset_y = (axis_bit & 0x2) ? output->info()->dimension(1) - id.y() - 1 : id.y();
            const int offset_z = (axis_bit & 0x4) ? output->info()->dimension(2) - id.z() - 1 : id.z();
            const int offset_w = (axis_bit & 0x8) ? output->info()->dimension(3) - id[3] - 1 : id[3];

            auto out_ptr = reinterpret_cast<T *>(output->ptr_to_element(Coordinates(offset_x, offset_y, offset_z, offset_w)));
            wrapper::vstore(out_ptr, in);
        },
        input_it);

        if(left_over_loop_x)
        {
            slice.set(Window::DimX, Window::Dimension(window_end_x_multiple_of, window_end_x, 1));

            Iterator input_it(input, slice);

            // Compute left-over elements along the y dimension (1x1)
            execute_window_loop(slice, [&](const Coordinates & id)
            {
                const auto in = *reinterpret_cast<T *>(input_it.ptr());

                const int offset_x = (axis_bit & 0x1) ? output->info()->dimension(0) - id.x() - 1 : id.x();
                const int offset_y = (axis_bit & 0x2) ? output->info()->dimension(1) - id.y() - 1 : id.y();
                const int offset_z = (axis_bit & 0x4) ? output->info()->dimension(2) - id.z() - 1 : id.z();
                const int offset_w = (axis_bit & 0x8) ? output->info()->dimension(3) - id[3] - 1 : id[3];

                *reinterpret_cast<T *>(output->ptr_to_element(Coordinates(offset_x, offset_y, offset_z, offset_w))) = in;
            },
            input_it);
        }

    }
    while(window.slide_window_slice_4D(slice));
}

void NEReverseKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    switch(_input->info()->data_type())
    {
        case DataType::F32:
            run_reverse<float>(window, _input, _axis, _output);
            break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            run_reverse<float16_t>(window, _input, _axis, _output);
            break;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::U32:
            run_reverse<uint32_t>(window, _input, _axis, _output);
            break;
        case DataType::S32:
            run_reverse<int32_t>(window, _input, _axis, _output);
            break;
        case DataType::S16:
            run_reverse<int16_t>(window, _input, _axis, _output);
            break;
        case DataType::U16:
            run_reverse<uint16_t>(window, _input, _axis, _output);
            break;
        case DataType::QASYMM8:
        case DataType::U8:
            run_reverse<uint8_t>(window, _input, _axis, _output);
            break;
        case DataType::S8:
            run_reverse<int8_t>(window, _input, _axis, _output);
            break;
        default:
            ARM_COMPUTE_ERROR("Data type not supported");
    }
}
} // namespace arm_compute