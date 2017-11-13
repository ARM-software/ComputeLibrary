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
#include "arm_compute/core/NEON/kernels/NEGEMMMatrixAdditionKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

#include <arm_neon.h>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

namespace
{
void matrix_addition_f32(const ITensor *input, ITensor *output, const Window &window, float beta)
{
    const float32x4_t beta_f32 = vdupq_n_f32(beta);

    Iterator in(input, window);
    Iterator out(output, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto in_ptr  = reinterpret_cast<const float *>(in.ptr());
        const auto out_ptr = reinterpret_cast<float *>(out.ptr());

        float32x4x4_t       alpha_ab = vld4q_f32(out_ptr);
        const float32x4x4_t c        = vld4q_f32(in_ptr);

        // Multiply matrix C by its weight and accumulate
        alpha_ab.val[0] = vmlaq_f32(alpha_ab.val[0], c.val[0], beta_f32);
        alpha_ab.val[1] = vmlaq_f32(alpha_ab.val[1], c.val[1], beta_f32);
        alpha_ab.val[2] = vmlaq_f32(alpha_ab.val[2], c.val[2], beta_f32);
        alpha_ab.val[3] = vmlaq_f32(alpha_ab.val[3], c.val[3], beta_f32);

        vst4q_f32(out_ptr, alpha_ab);
    },
    in, out);
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
void matrix_addition_f16(const ITensor *input, ITensor *output, const Window &window, float beta)
{
    const float16x8_t beta_f16 = vdupq_n_f16(beta);

    Iterator in(input, window);
    Iterator out(output, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto in_ptr  = reinterpret_cast<const float16_t *>(in.ptr());
        const auto out_ptr = reinterpret_cast<float16_t *>(out.ptr());

        float16x8x2_t       alpha_ab = vld2q_f16(out_ptr);
        const float16x8x2_t c        = vld2q_f16(in_ptr);
        // Multiply matrix C by its weight and accumulate
        alpha_ab.val[0] = vaddq_f16(alpha_ab.val[0], vmulq_f16(c.val[0], beta_f16));
        alpha_ab.val[1] = vaddq_f16(alpha_ab.val[1], vmulq_f16(c.val[1], beta_f16));

        vst2q_f16(out_ptr + 0, alpha_ab);
    },
    in, out);
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

void matrix_addition_qs8(const ITensor *input, ITensor *output, const Window &window, float beta)
{
    const int        fixed_point_position = input->info()->fixed_point_position();
    const qint8x16_t beta_qs8             = vdupq_n_qs8(sqcvt_qs8_f32(beta, fixed_point_position));

    Iterator in(input, window);
    Iterator out(output, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto in_ptr  = reinterpret_cast<const qint8_t *>(in.ptr());
        const auto out_ptr = reinterpret_cast<qint8_t *>(out.ptr());

        qint8x16_t       alpha_ab = vld1q_qs8(out_ptr);
        const qint8x16_t c        = vld1q_qs8(in_ptr);

        // Multiply matrix C by its weight and accumulate
        alpha_ab = vqmlaq_qs8(alpha_ab, c, beta_qs8, fixed_point_position);

        vst1q_qs8(out_ptr, alpha_ab);
    },
    in, out);
}

void matrix_addition_qs16(const ITensor *input, ITensor *output, const Window &window, float beta)
{
    const int        fixed_point_position = input->info()->fixed_point_position();
    const qint16x8_t beta_qs16            = vdupq_n_qs16(sqcvt_qs16_f32(beta, fixed_point_position));

    Iterator in(input, window);
    Iterator out(output, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto in_ptr  = reinterpret_cast<const qint16_t *>(in.ptr());
        const auto out_ptr = reinterpret_cast<qint16_t *>(out.ptr());

        qint16x8x2_t       alpha_ab = vld2q_s16(out_ptr);
        const qint16x8x2_t c        = vld2q_s16(in_ptr);

        // Multiply matrix C by its weight and accumulate
        alpha_ab.val[0] = vqmlaq_qs16(alpha_ab.val[0], c.val[0], beta_qs16, fixed_point_position);
        alpha_ab.val[1] = vqmlaq_qs16(alpha_ab.val[1], c.val[1], beta_qs16, fixed_point_position);

        vst2q_s16(out_ptr, alpha_ab);
    },
    in, out);
}
} // namespace

NEGEMMMatrixAdditionKernel::NEGEMMMatrixAdditionKernel()
    : INESimpleKernel(), _func(nullptr), _beta(0.0f)
{
}

void NEGEMMMatrixAdditionKernel::configure(const ITensor *input, ITensor *output, float beta)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(0) != output->info()->dimension(0));
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(1) != output->info()->dimension(1));

    switch(input->info()->data_type())
    {
        case DataType::F32:
            _func = &matrix_addition_f32;
            break;
        case DataType::QS8:
            _func = &matrix_addition_qs8;
            break;
        case DataType::QS16:
            _func = &matrix_addition_qs16;
            break;
        case DataType::F16:
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            _func = &matrix_addition_f16;
            break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        default:
            ARM_COMPUTE_ERROR("Data type not supported");
            break;
    }

    constexpr unsigned int num_elems_processed_per_iteration = 16;

    INESimpleKernel::configure(input, output, num_elems_processed_per_iteration);

    _beta = beta;
}

void NEGEMMMatrixAdditionKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INESimpleKernel::window(), window);

    if(_beta != 0.0f)
    {
        (*_func)(_input, _output, window, _beta);
    }
}
