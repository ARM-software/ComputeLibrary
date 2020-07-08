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
#include "arm_compute/core/NEON/kernels/NEFloorKernel.h"

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/Validate.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);

    // Validate in case of configured output
    if(output->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
    }

    return Status{};
}
} // namespace

void NEFloorKernel::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Auto initialize output
    auto_init_if_empty(*output->info(), input->info()->tensor_shape(), 1, input->info()->data_type());

    // Validate
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info()));

    _input  = input;
    _output = output;

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps());

    Coordinates coord;
    coord.set_num_dimensions(output->info()->num_dimensions());
    output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));

    INEKernel::configure(win);
}

Status NEFloorKernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output));
    return Status{};
}

void NEFloorKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const DataType data_type = _input->info()->data_type();

    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());
    const int  window_step_x  = 16 / _input->info()->element_size();

    Window win{ window };
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    Iterator input(_input, win);
    Iterator output(_output, win);

    if(data_type == DataType::F32)
    {
        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto input_ptr  = reinterpret_cast<const float *>(input.ptr());
            const auto output_ptr = reinterpret_cast<float *>(output.ptr());

            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const float32x4_t res = vfloorq_f32(vld1q_f32(input_ptr + x));
                vst1q_f32(output_ptr + x, res);
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                *(output_ptr + x) = std::floor(*(input_ptr + x));
            }
        },
        input, output);
    }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    else if(data_type == DataType::F16)
    {
        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto input_ptr  = reinterpret_cast<const float16_t *>(input.ptr());
            const auto output_ptr = reinterpret_cast<float16_t *>(output.ptr());

            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const float16x8_t res = vfloorq_f16(vld1q_f16(input_ptr + x));
                vst1q_f16(output_ptr + x, res);
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                *(output_ptr + x) = std::floor(*(input_ptr + x));
            }
        },
        input, output);
    }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    else
    {
        ARM_COMPUTE_ERROR("Invalid data type!");
    }
}
} // namespace arm_compute
