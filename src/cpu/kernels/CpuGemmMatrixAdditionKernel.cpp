/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#include "src/cpu/kernels/CpuGemmMatrixAdditionKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "src/core/CPP/Validate.h"
#include "src/core/NEON/NEFixedPoint.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
void matrix_addition_f32(const ITensor *src, ITensor *dst, const Window &window, float beta)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    const float32x4_t beta_f32 = vdupq_n_f32(beta);

    constexpr int window_step_x  = 16;
    const auto    window_start_x = static_cast<int>(window.x().start());
    const auto    window_end_x   = static_cast<int>(window.x().end());

    Window win = window.collapse_if_possible(window, Window::DimZ);
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator in(src, win);
    Iterator out(dst, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto in_ptr  = reinterpret_cast<const float *>(in.ptr());
        const auto out_ptr = reinterpret_cast<float *>(out.ptr());

        int x = window_start_x;
        for(; x < (window_end_x - window_step_x); x += window_step_x)
        {
            float32x4x4_t       alpha_ab = vld4q_f32(out_ptr + x);
            const float32x4x4_t c        = vld4q_f32(in_ptr + x);

            // Multiply matrix C by its weight and accumulate
            alpha_ab.val[0] = vmlaq_f32(alpha_ab.val[0], c.val[0], beta_f32);
            alpha_ab.val[1] = vmlaq_f32(alpha_ab.val[1], c.val[1], beta_f32);
            alpha_ab.val[2] = vmlaq_f32(alpha_ab.val[2], c.val[2], beta_f32);
            alpha_ab.val[3] = vmlaq_f32(alpha_ab.val[3], c.val[3], beta_f32);

            vst4q_f32(out_ptr + x, alpha_ab);
        }

        // Left-over loop
        for(; x < window_end_x; ++x)
        {
            *(out_ptr + x) += *(in_ptr + x) * beta;
        }
    },
    in, out);
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
void matrix_addition_f16(const ITensor *src, ITensor *dst, const Window &window, float beta)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    const float16x8_t beta_f16 = vdupq_n_f16(beta);

    constexpr int window_step_x  = 16;
    const auto    window_start_x = static_cast<int>(window.x().start());
    const auto    window_end_x   = static_cast<int>(window.x().end());

    Window win = window.collapse_if_possible(window, Window::DimZ);
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator in(src, win);
    Iterator out(dst, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto in_ptr  = reinterpret_cast<const float16_t *>(in.ptr());
        const auto out_ptr = reinterpret_cast<float16_t *>(out.ptr());

        int x = window_start_x;
        for(; x < (window_end_x - window_step_x); x += window_step_x)
        {
            float16x8x2_t       alpha_ab = vld2q_f16(out_ptr + x);
            const float16x8x2_t c        = vld2q_f16(in_ptr + x);
            // Multiply matrix C by its weight and accumulate
            alpha_ab.val[0] = vaddq_f16(alpha_ab.val[0], vmulq_f16(c.val[0], beta_f16));
            alpha_ab.val[1] = vaddq_f16(alpha_ab.val[1], vmulq_f16(c.val[1], beta_f16));

            vst2q_f16(out_ptr + x, alpha_ab);
        }

        // Left-over loop
        for(; x < window_end_x; ++x)
        {
            *(out_ptr + x) += *(in_ptr + x) * static_cast<float16_t>(beta);
        }
    },
    in, out);
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

} // namespace

void CpuGemmMatrixAdditionKernel::configure(const ITensorInfo *src, ITensorInfo *dst, float beta)
{
    ARM_COMPUTE_UNUSED(dst);
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(CpuGemmMatrixAdditionKernel::validate(src, dst, beta));

    _beta = beta;
    switch(src->data_type())
    {
        case DataType::F32:
            _func = &matrix_addition_f32;
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

    // Configure kernel window
    Window win = calculate_max_window(*src, Steps());
    ICPPKernel::configure(win);
}

Status CpuGemmMatrixAdditionKernel::validate(const ITensorInfo *src, const ITensorInfo *dst, float beta)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_UNUSED(beta);

    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F16, DataType::F32);

    if(dst->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, dst);
    }
    return Status{};
}

void CpuGemmMatrixAdditionKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(tensors.empty());

    const ITensor *src = tensors.get_const_tensor(TensorType::ACL_SRC);
    ITensor       *dst = tensors.get_tensor(TensorType::ACL_DST);

    if(_beta != 0.0f)
    {
        (*_func)(src, dst, window, _beta);
    }
}

const char *CpuGemmMatrixAdditionKernel::name() const
{
    return "CpuGemmMatrixAdditionKernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
