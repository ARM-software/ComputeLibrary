/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NELocallyConnectedMatrixMultiplyKernel.h"

#include "arm_compute/core/AccessWindowTranspose.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>
#include <tuple>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

namespace
{
void vector_matrix_multiply_f32(const ITensor *input0, const ITensor *input1, ITensor *output, const Window &window)
{
    const auto width_matrix_b  = static_cast<int>(output->info()->dimension(0));
    const auto in_b_stride     = static_cast<int>(input1->info()->strides_in_bytes()[1] / data_size_from_type(input1->info()->data_type()));
    const auto num_elems_vec_a = static_cast<int>(input0->info()->dimension(0));

    // The implementation computes 16 elements per iteration
    const int window_start_x = 16 * window.thread_id();
    const int window_step_x  = 16 * window.num_threads();
    // Make sure (window_end_x - window_start_x) is a multiple of window_step_x
    const int window_end_x = ceil_to_multiple(width_matrix_b - window_start_x, window_step_x) + window_start_x;

    Window win_out(window);
    win_out.set(Window::DimX, Window::Dimension(window_start_x, window_end_x, window_step_x));

    Window win_a(window);
    win_a.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator ina(input0, win_a);
    Iterator out(output, win_out);

    execute_window_loop(win_out, [&](const Coordinates & id)
    {
        if(id.x() > width_matrix_b)
        {
            return;
        }

        float32x4_t acc0 = vdupq_n_f32(0.f);
        float32x4_t acc1 = vdupq_n_f32(0.f);
        float32x4_t acc2 = vdupq_n_f32(0.f);
        float32x4_t acc3 = vdupq_n_f32(0.f);

        auto vec_a    = reinterpret_cast<const float *>(ina.ptr());
        auto matrix_b = reinterpret_cast<const float *>(input1->ptr_to_element(Coordinates(id[0], 0, id[1])));

#if __arm__
        asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(vec_a)));
        asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(matrix_b)));
        asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(matrix_b + in_b_stride)));
#endif

        const float *vec_a_end_addr = vec_a + num_elems_vec_a;

        for(; vec_a <= (vec_a_end_addr - 4);)
        {
            float32x2_t a0l = vld1_f32(vec_a);

            float32x4_t b00 = vld1q_f32(matrix_b + 0 + 0 * in_b_stride);
            float32x4_t b01 = vld1q_f32(matrix_b + 4 + 0 * in_b_stride);
            float32x4_t b02 = vld1q_f32(matrix_b + 8 + 0 * in_b_stride);
            float32x4_t b03 = vld1q_f32(matrix_b + 12 + 0 * in_b_stride);

            float32x4_t b10 = vld1q_f32(matrix_b + 0 + 1 * in_b_stride);
            float32x4_t b11 = vld1q_f32(matrix_b + 4 + 1 * in_b_stride);
            float32x4_t b12 = vld1q_f32(matrix_b + 8 + 1 * in_b_stride);
            float32x4_t b13 = vld1q_f32(matrix_b + 12 + 1 * in_b_stride);

#if __arm__
            asm volatile("PLD [%0, #128*4]" ::"r"(reinterpret_cast<const uint8_t *>(vec_a)));
            asm volatile("PLD [%0, #128*1]" ::"r"(reinterpret_cast<const uint8_t *>(matrix_b + 1 * in_b_stride)));
            asm volatile("PLD [%0, #128*1]" ::"r"(reinterpret_cast<const uint8_t *>(matrix_b + 2 * in_b_stride)));
            asm volatile("PLD [%0, #128*1]" ::"r"(reinterpret_cast<const uint8_t *>(matrix_b + 3 * in_b_stride)));
            asm volatile("PLD [%0, #128*1]" ::"r"(reinterpret_cast<const uint8_t *>(matrix_b + 4 * in_b_stride)));
#endif

            acc0 = vmlaq_lane_f32(acc0, b00, a0l, 0);
            acc1 = vmlaq_lane_f32(acc1, b01, a0l, 0);
            acc2 = vmlaq_lane_f32(acc2, b02, a0l, 0);
            acc3 = vmlaq_lane_f32(acc3, b03, a0l, 0);

            acc0 = vmlaq_lane_f32(acc0, b10, a0l, 1);
            acc1 = vmlaq_lane_f32(acc1, b11, a0l, 1);
            acc2 = vmlaq_lane_f32(acc2, b12, a0l, 1);
            acc3 = vmlaq_lane_f32(acc3, b13, a0l, 1);

            vec_a += 2;
            matrix_b += 2 * in_b_stride;

            a0l = vld1_f32(vec_a);

            b00 = vld1q_f32(matrix_b + 0 + 0 * in_b_stride);
            b01 = vld1q_f32(matrix_b + 4 + 0 * in_b_stride);
            b02 = vld1q_f32(matrix_b + 8 + 0 * in_b_stride);
            b03 = vld1q_f32(matrix_b + 12 + 0 * in_b_stride);

            b10 = vld1q_f32(matrix_b + 0 + 1 * in_b_stride);
            b11 = vld1q_f32(matrix_b + 4 + 1 * in_b_stride);
            b12 = vld1q_f32(matrix_b + 8 + 1 * in_b_stride);
            b13 = vld1q_f32(matrix_b + 12 + 1 * in_b_stride);

            acc0 = vmlaq_lane_f32(acc0, b00, a0l, 0);
            acc1 = vmlaq_lane_f32(acc1, b01, a0l, 0);
            acc2 = vmlaq_lane_f32(acc2, b02, a0l, 0);
            acc3 = vmlaq_lane_f32(acc3, b03, a0l, 0);

            acc0 = vmlaq_lane_f32(acc0, b10, a0l, 1);
            acc1 = vmlaq_lane_f32(acc1, b11, a0l, 1);
            acc2 = vmlaq_lane_f32(acc2, b12, a0l, 1);
            acc3 = vmlaq_lane_f32(acc3, b13, a0l, 1);

            vec_a += 2;
            matrix_b += 2 * in_b_stride;
        }

        for(; vec_a < vec_a_end_addr;)
        {
            const float a0 = *vec_a;

            const float32x4_t b00 = vld1q_f32(matrix_b + 0 + 0 * in_b_stride);
            const float32x4_t b01 = vld1q_f32(matrix_b + 4 + 0 * in_b_stride);
            const float32x4_t b02 = vld1q_f32(matrix_b + 8 + 0 * in_b_stride);
            const float32x4_t b03 = vld1q_f32(matrix_b + 12 + 0 * in_b_stride);

            acc0 = vmlaq_n_f32(acc0, b00, a0);
            acc1 = vmlaq_n_f32(acc1, b01, a0);
            acc2 = vmlaq_n_f32(acc2, b02, a0);
            acc3 = vmlaq_n_f32(acc3, b03, a0);

            vec_a += 1;
            matrix_b += in_b_stride;
        }

        const auto vec_out = reinterpret_cast<float *>(out.ptr());

        vst1q_f32(vec_out + 0, acc0);
        vst1q_f32(vec_out + 4, acc1);
        vst1q_f32(vec_out + 8, acc2);
        vst1q_f32(vec_out + 12, acc3);
    },
    ina, out);
}
} // namespace

NELocallyConnectedMatrixMultiplyKernel::NELocallyConnectedMatrixMultiplyKernel()
    : _input0(nullptr), _input1(nullptr), _output(nullptr)
{
}

void NELocallyConnectedMatrixMultiplyKernel::configure(const ITensor *input0, const ITensor *input1, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input0, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input1, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON(input0->info()->dimension(0) != input1->info()->dimension(1));

    _input0 = input0;
    _input1 = input1;
    _output = output;

    unsigned int num_elems_processed_per_iteration_x = 16;

    // Configure kernel window
    Window win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration_x));

    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration_x);

    update_window_and_padding(win,
                              AccessWindowHorizontal(input0->info(), 0, num_elems_processed_per_iteration_x),
                              AccessWindowHorizontal(input1->info(), 0, num_elems_processed_per_iteration_x),
                              output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(0, 0), output->info()->tensor_shape()));

    INEKernel::configure(win);
}

void NELocallyConnectedMatrixMultiplyKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    vector_matrix_multiply_f32(_input0, _input1, _output, window);
}
