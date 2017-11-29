/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEGEMMMatrixVectorMultiplyKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>
#include <tuple>

using namespace arm_compute;

NEGEMMMatrixVectorMultiplyKernel::NEGEMMMatrixVectorMultiplyKernel()
    : _input0(nullptr), _input1(nullptr), _output(nullptr)
{
}

void NEGEMMMatrixVectorMultiplyKernel::configure(const ITensor *input0, const ITensor *input1, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input0, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input0, input1, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input0, input1, output);
    ARM_COMPUTE_ERROR_ON(input0->info()->dimension(2) != input1->info()->dimension(1));

    _input0 = input0;
    _input1 = input1;
    _output = output;

    // Configure kernel window
    const unsigned int num_elems_read_per_iteration = 4;

    Window win = calculate_max_window(*input0->info(), Steps(num_elems_read_per_iteration));

    AccessWindowHorizontal input0_access(input0->info(), 0, num_elems_read_per_iteration);
    AccessWindowHorizontal input1_access(input1->info(), 0, num_elems_read_per_iteration);
    AccessWindowStatic     output_access(output->info(), 0, 0, output->info()->dimension(0), output->info()->dimension(1));

    update_window_and_padding(win, input0_access, input1_access, output_access);

    _output->info()->set_valid_region(ValidRegion(Coordinates(), _output->info()->tensor_shape()));

    INEKernel::configure(win);
}

void NEGEMMMatrixVectorMultiplyKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);

    Window window_slice = window.first_slice_window_3D();

    Window window_in(window);
    Window window_weights(window_slice);
    Window window_out(window);

    // Setup input0 slice
    window_in.set(Window::DimX, Window::Dimension(0, _input0->info()->dimension(0), _input0->info()->dimension(0)));
    window_in.set(Window::DimY, Window::Dimension(0, _input0->info()->dimension(1), 1));
    window_in.set(Window::DimZ, Window::Dimension(0, _input0->info()->dimension(2), 1));

    // Setup input1 and output slice. Their dimensions are increased in the kernel.
    window_weights.set(Window::DimX, Window::Dimension(0, 0, 0));
    window_weights.set(Window::DimY, Window::Dimension(0, 0, 0));
    window_weights.set(Window::DimZ, Window::Dimension(0, 0, 0));

    window_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    window_out.set(Window::DimY, Window::Dimension(0, 0, 0));
    window_out.set(Window::DimZ, Window::Dimension(0, 0, 0));

    Iterator in(_input0, window_in);
    Iterator in2(_input1, window_weights);
    Iterator out(_output, window_out);

    const int input_w          = _input0->info()->dimension(0);
    const int input_h          = _input0->info()->dimension(1);
    const int input_stride_x   = _input0->info()->strides_in_bytes().x();
    const int weights_stride_x = _input1->info()->strides_in_bytes().x();
    const int weights_stride_y = _input1->info()->strides_in_bytes().y();
    const int output_stride_x  = _output->info()->strides_in_bytes().x();

    execute_window_loop(window_in, [&](const Coordinates & id)
    {
        // Get pointers
        const uint8_t *const input_ptr   = in.ptr();
        const uint8_t *const weights_ptr = in2.ptr() + id.z() * weights_stride_y;
        auto                 output_ptr  = reinterpret_cast<float *>(out.ptr() + (id.y() + id.z() * input_h) * output_stride_x);

        float32x4_t row_dot = vdupq_n_f32(0.f);
        for(int i = 0; i < input_w; i += 4)
        {
            const auto input   = vld1q_f32(reinterpret_cast<const float *>(input_ptr + i * input_stride_x));
            const auto weights = vld1q_f32(reinterpret_cast<const float *>(weights_ptr + i * weights_stride_x));
            row_dot            = vaddq_f32(row_dot, vmulq_f32(input, weights));
        }

        auto temp = vadd_f32(vget_high_f32(row_dot), vget_low_f32(row_dot));
        temp      = vpadd_f32(temp, temp);

        *output_ptr = vget_lane_f32(temp, 0);
    },
    in, in2, out);
}
