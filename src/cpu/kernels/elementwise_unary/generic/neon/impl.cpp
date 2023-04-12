/*
 * Copyright (c) 2018-2023 Arm Limited.
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
#include "src/cpu/kernels/elementwise_unary/generic/neon/impl.h"
#include "arm_compute/core/Helpers.h"
#include "src/core/NEON/NEAsymm.h"

namespace arm_compute
{
namespace cpu
{
template <typename ScalarType>
inline ScalarType elementwise_op_scalar_imp(ElementWiseUnary op, const ScalarType &a)
{
    switch(op)
    {
        case ElementWiseUnary::RSQRT:
            return 1 / sqrt(a);
        case ElementWiseUnary::EXP:
            return std::exp(a);
        case ElementWiseUnary::NEG:
            return -a;
        case ElementWiseUnary::LOG:
            return std::log(a);
        case ElementWiseUnary::ABS:
            return std::abs(a);
        case ElementWiseUnary::ROUND:
            return support::cpp11::nearbyint(a);
        case ElementWiseUnary::SIN:
            return std::sin(a);
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }
}

template <typename ScalarType, typename VectorType>
inline VectorType elementwise_op_imp(ElementWiseUnary op, const VectorType &a)
{
    switch(op)
    {
        case ElementWiseUnary::RSQRT:
            return wrapper::vinvsqrt(a);
        case ElementWiseUnary::EXP:
            return wrapper::vexpq(a);
        case ElementWiseUnary::NEG:
            return wrapper::vneg(a);
        case ElementWiseUnary::LOG:
            return wrapper::vlog(a);
        case ElementWiseUnary::ABS:
            return wrapper::vabs(a);
        case ElementWiseUnary::ROUND:
            return wrapper::vround(a);
        case ElementWiseUnary::SIN:
            return wrapper::vsin(a);
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }
}

template <typename ScalarType>
void elementwise_op(const ITensor *in, ITensor *out, const Window &window, ElementWiseUnary op)
{
    const int  window_step_x  = 16 / sizeof(ScalarType);
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(in, win);
    Iterator output(out, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        auto       output_ptr = reinterpret_cast<ScalarType *>(output.ptr());
        const auto input_ptr  = reinterpret_cast<const ScalarType *>(input.ptr());

        int x = window_start_x;
        for(; x <= window_end_x - window_step_x; x += window_step_x)
        {
            wrapper::vstore(output_ptr + x, elementwise_op_imp<ScalarType>(op, wrapper::vloadq(input_ptr + x)));
        }
        for(; x < window_end_x; ++x)
        {
            *(output_ptr + x) = elementwise_op_scalar_imp(op, *(input_ptr + x));
        }
    },
    input, output);
}
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)
template void elementwise_op<__fp16>(const ITensor *in, ITensor *out, const Window &window, ElementWiseUnary op);
#endif //defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)
template void elementwise_op<float>(const ITensor *in, ITensor *out, const Window &window, ElementWiseUnary op);
template void elementwise_op<int32_t>(const ITensor *in, ITensor *out, const Window &window, ElementWiseUnary op);

template <>
void elementwise_op<int8_t>(const ITensor *in, ITensor *out, const Window &window, ElementWiseUnary op)
{
    const int                     window_step_x     = 16;
    const auto                    window_start_x    = static_cast<int>(window.x().start());
    const auto                    window_end_x      = static_cast<int>(window.x().end());
    const UniformQuantizationInfo qi_in             = in->info()->quantization_info().uniform();
    const UniformQuantizationInfo qi_out            = out->info()->quantization_info().uniform();
    const auto                    min_clamped_value = vdupq_n_f32((-128 - qi_out.offset) * qi_out.scale);
    const auto                    max_clamped_value = vdupq_n_f32((127 - qi_out.offset) * qi_out.scale);
    Window                        win               = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(in, win);
    Iterator output(out, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        int8x16_t  vout;
        auto       output_ptr    = reinterpret_cast<int8_t *>(output.ptr());
        const auto input_ptr     = reinterpret_cast<const int8_t *>(input.ptr());
        const auto vconst_0_f32  = vdupq_n_f32(0);
        auto       clamped_value = (op == ElementWiseUnary::LOG) ? min_clamped_value : max_clamped_value;

        int x = window_start_x;
        for(; x <= window_end_x - window_step_x; x += window_step_x)
        {
            const auto vin = wrapper::vloadq(input_ptr + x);

            // De-quantize
            const auto vin_deq = vdequantize(vin, qi_in);

            // Perform activation
            float32x4x4_t vtmp_deq =
            {
                {
                    elementwise_op_imp<float>(op, vin_deq.val[0]),
                    elementwise_op_imp<float>(op, vin_deq.val[1]),
                    elementwise_op_imp<float>(op, vin_deq.val[2]),
                    elementwise_op_imp<float>(op, vin_deq.val[3]),
                }
            };

            if((op == ElementWiseUnary::LOG) || (op == ElementWiseUnary::RSQRT))
            {
                vtmp_deq.val[0] = vbslq_f32(vcleq_f32(vin_deq.val[0], vconst_0_f32), clamped_value, vtmp_deq.val[0]);
                vtmp_deq.val[1] = vbslq_f32(vcleq_f32(vin_deq.val[1], vconst_0_f32), clamped_value, vtmp_deq.val[1]);
                vtmp_deq.val[2] = vbslq_f32(vcleq_f32(vin_deq.val[2], vconst_0_f32), clamped_value, vtmp_deq.val[2]);
                vtmp_deq.val[3] = vbslq_f32(vcleq_f32(vin_deq.val[3], vconst_0_f32), clamped_value, vtmp_deq.val[3]);
            }

            // Re-quantize to new output space
            vout = vquantize_signed(vtmp_deq, qi_out);
            wrapper::vstore(output_ptr + x, vout);
        }
        for(; x < window_end_x; ++x)
        {
            qasymm8_signed_t in    = *(reinterpret_cast<const qasymm8_signed_t *>(input_ptr + x));
            qasymm8_signed_t tmp   = 0;
            float            tmp_f = dequantize_qasymm8_signed(in, qi_in);
            if(tmp_f <= 0.0)
            {
                if(op == ElementWiseUnary::LOG)
                {
                    tmp_f = (-128 - qi_out.offset) * qi_out.scale;
                }
                else if(op == ElementWiseUnary::RSQRT)
                {
                    tmp_f = (127 - qi_out.offset) * qi_out.scale;
                }
                else
                {
                    tmp_f = elementwise_op_scalar_imp<float>(op, tmp_f);
                }
            }
            else
            {
                tmp_f = elementwise_op_scalar_imp<float>(op, tmp_f);
            }
            tmp = quantize_qasymm8_signed(tmp_f, qi_out, RoundingPolicy::TO_ZERO); // Set rounding policy TO_ZERO to be compatible with vquantize_signed() used above that follow same policy for armv7a.
            // For aarch64 LUT is used and rounding to nearest is used
            *(output_ptr + x) = tmp;
        }
    },
    input, output);
}
template <>
void elementwise_op<uint8_t>(const ITensor *in, ITensor *out, const Window &window, ElementWiseUnary op)
{
    const int                     window_step_x     = 16;
    const auto                    window_start_x    = static_cast<int>(window.x().start());
    const auto                    window_end_x      = static_cast<int>(window.x().end());
    const UniformQuantizationInfo qi_in             = in->info()->quantization_info().uniform();
    const UniformQuantizationInfo qi_out            = out->info()->quantization_info().uniform();
    const auto                    vconst_0_f32      = vdupq_n_f32(0);
    const auto                    min_clamped_value = vdupq_n_f32((0 - qi_out.offset) * qi_out.scale);
    const auto                    max_clamped_value = vdupq_n_f32((255 - qi_out.offset) * qi_out.scale);
    Window                        win               = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(in, win);
    Iterator output(out, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        uint8x16_t vout;
        auto       clamped_value = (op == ElementWiseUnary::LOG) ? min_clamped_value : max_clamped_value;
        auto       output_ptr    = reinterpret_cast<uint8_t *>(output.ptr());
        const auto input_ptr     = reinterpret_cast<const uint8_t *>(input.ptr());
        int        x             = window_start_x;
        for(; x <= window_end_x - window_step_x; x += window_step_x)
        {
            const auto vin = wrapper::vloadq(input_ptr + x);

            // De-quantize
            const auto vin_deq = vdequantize(vin, qi_in);

            // Perform activation
            float32x4x4_t vtmp_deq =
            {
                {
                    elementwise_op_imp<float>(op, vin_deq.val[0]),
                    elementwise_op_imp<float>(op, vin_deq.val[1]),
                    elementwise_op_imp<float>(op, vin_deq.val[2]),
                    elementwise_op_imp<float>(op, vin_deq.val[3]),
                }
            };
            if((op == ElementWiseUnary::LOG) || (op == ElementWiseUnary::RSQRT))
            {
                vtmp_deq.val[0] = vbslq_f32(vcleq_f32(vin_deq.val[0], vconst_0_f32), clamped_value, vtmp_deq.val[0]);
                vtmp_deq.val[1] = vbslq_f32(vcleq_f32(vin_deq.val[1], vconst_0_f32), clamped_value, vtmp_deq.val[1]);
                vtmp_deq.val[2] = vbslq_f32(vcleq_f32(vin_deq.val[2], vconst_0_f32), clamped_value, vtmp_deq.val[2]);
                vtmp_deq.val[3] = vbslq_f32(vcleq_f32(vin_deq.val[3], vconst_0_f32), clamped_value, vtmp_deq.val[3]);
            }

            // Re-quantize to new output space
            vout = vquantize(vtmp_deq, qi_out);
            wrapper::vstore(output_ptr + x, vout);
        }
        for(; x < window_end_x; ++x)
        {
            qasymm8_t in    = *(reinterpret_cast<const qasymm8_t *>(input_ptr + x));
            qasymm8_t tmp   = 0;
            float     tmp_f = dequantize_qasymm8(in, qi_in);
            if(tmp_f <= 0.0)
            {
                if(op == ElementWiseUnary::LOG)
                {
                    tmp_f = (0 - qi_out.offset) * qi_out.scale;
                }
                else if(op == ElementWiseUnary::RSQRT)
                {
                    tmp_f = (255 - qi_out.offset) * qi_out.scale;
                }
                else
                {
                    tmp_f = elementwise_op_scalar_imp<float>(op, tmp_f);
                }
            }
            else
            {
                tmp_f = elementwise_op_scalar_imp<float>(op, tmp_f);
            }
            tmp               = quantize_qasymm8(tmp_f, qi_out, RoundingPolicy::TO_ZERO);
            *(output_ptr + x) = tmp;
        }
    },
    input, output);
}
} // namespace cpu
} // namespace arm_compute
