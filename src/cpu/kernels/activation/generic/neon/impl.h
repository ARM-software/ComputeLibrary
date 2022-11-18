/*
 * Copyright (c) 2020-2022 Arm Limited.
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
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Window.h"
#include "src/core/NEON/wrapper/wrapper.h"
namespace arm_compute
{
namespace cpu
{
/** Constant parameters needed by the activation implementation.
 *  These parameters differ for each floating type
 *
 * @note This are passed as a struct as C++ does not allow float as a template parameter until C++20
 **/
struct ActFpImplParams
{
    float delta;  /**< Minimum delta needed to avoid NaN on corner-cases of elementary functions */
    int   step_x; /**< Window step at the x dimension */
};

#ifndef __aarch64__
inline float32x4_t mask_float_vector(const float32x4_t &in, const uint32x4_t &mask)
{
    auto int_in = vreinterpretq_u32_f32(in);
    return vreinterpretq_f32_u32(wrapper::vand(int_in, mask));
}
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)
inline float16x8_t mask_float_vector(const float16x8_t &in, const uint16x8_t &mask)
{
    auto int_in = vreinterpretq_u16_f16(in);
    return vreinterpretq_f16_u16(wrapper::vand(int_in, mask));
}
#endif //defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)
#endif /* __aarch64__ */

template <typename T, const ActFpImplParams &P>
void fp_neon_activation_impl(const ITensor *src, ITensor *dst, const ActivationLayerInfo &act_info, const Window &window)
{
    /** SIMD vector tag type. */
    using ExactTagType                                           = typename arm_compute::wrapper::traits::neon_bitvector_tag_t<T, wrapper::traits::BitWidth::W128>;
    constexpr int                                 window_step_x  = P.step_x;
    const auto                                    window_start_x = static_cast<int>(window.x().start());
    const auto                                    window_end_x   = static_cast<int>(window.x().end());
    const ActivationLayerInfo::ActivationFunction act            = act_info.activation();
    Window                                        win_collapsed  = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));
    Iterator input(src, win_collapsed);
    Iterator output(dst, win_collapsed);
    // In case of non-aarch64, a small delta value is added to the input
    // to prevent NAN values caused by zeros in inputs to SQRT.
    // In case of aarh64, we call vsqrt directly, so we don't use delta.
#ifndef __aarch64__
    const auto delta = wrapper::vdup_n(static_cast<T>(P.delta), ExactTagType {});
#else  /* #ifndef __aarch64__ */
    const auto const_inv_2      = wrapper::vdup_n(static_cast<T>(0.5f), ExactTagType {});
    const auto const_inv_sqrt_2 = wrapper::vdup_n(static_cast<T>(0.70710678118f), ExactTagType{});
#endif /* __aarch64__ */
    const auto      const_1           = wrapper::vdup_n(static_cast<T>(1.f), ExactTagType {});
    const auto      const_0           = wrapper::vdup_n(static_cast<T>(0.f), ExactTagType{});
    const auto      const_6           = wrapper::vdup_n(static_cast<T>(6.f), ExactTagType{});
    const auto      const_3           = wrapper::vdup_n(static_cast<T>(3.f), ExactTagType{});
    const auto      const_inv_6       = wrapper::vdup_n(static_cast<T>(0.166666667f), ExactTagType{});
    constexpr float soft_relu_thresh  = 12.f;
    const auto      vsoft_relu_thresh = wrapper::vdup_n(static_cast<T>(soft_relu_thresh), ExactTagType{});
    const auto      va                = wrapper::vdup_n(static_cast<T>(act_info.a()), ExactTagType{});
    const auto      vb                = wrapper::vdup_n(static_cast<T>(act_info.b()), ExactTagType{});
    const auto      a                 = static_cast<T>(act_info.a());
    const auto      b                 = static_cast<T>(act_info.b());
    execute_window_loop(win_collapsed, [&](const Coordinates &)
    {
        const auto input_ptr  = reinterpret_cast<const T *>(input.ptr());
        const auto output_ptr = reinterpret_cast<T *>(output.ptr());
        wrapper::traits::neon_bitvector_t<T, wrapper::traits::BitWidth::W128> tmp;
        // Compute S elements per iteration
        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const auto vin = wrapper::vloadq(input_ptr + x);
            switch(act)
            {
                case ActivationLayerInfo::ActivationFunction::ABS:
                    tmp = wrapper::vabs(vin);
                    break;
                case ActivationLayerInfo::ActivationFunction::LINEAR:
                    tmp = wrapper::vmla(vb, va, vin);
                    break;
                case ActivationLayerInfo::ActivationFunction::LOGISTIC:
                    tmp = wrapper::vinv(wrapper::vadd(const_1, wrapper::vexpq(wrapper::vneg(vin))));
                    break;
                case ActivationLayerInfo::ActivationFunction::RELU:
                    tmp = wrapper::vmax(const_0, vin);
                    break;
                case ActivationLayerInfo::ActivationFunction::BOUNDED_RELU:
                    tmp = wrapper::vmin(va, wrapper::vmax(const_0, vin));
                    break;
                case ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU:
                    tmp = wrapper::vmin(va, wrapper::vmax(vb, vin));
                    break;
                case ActivationLayerInfo::ActivationFunction::LEAKY_RELU:
                    tmp = wrapper::vbsl(wrapper::vcgt(vin, const_0), vin, wrapper::vmul(va, vin));
                    break;
                case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
                    tmp = wrapper::vbsl(wrapper::vcgt(vin, vsoft_relu_thresh), vin, wrapper::vlog(wrapper::vadd(const_1, wrapper::vexpq(vin))));
                    break;
                case ActivationLayerInfo::ActivationFunction::ELU:
                    tmp = wrapper::vbsl(wrapper::vcge(vin, const_0), vin, wrapper::vmul(va, wrapper::vsub(wrapper::vexpq(vin), const_1)));
                    break;
                case ActivationLayerInfo::ActivationFunction::SQRT:
#ifdef __aarch64__
                    tmp = wrapper::vsqrt(vin);
#else  /* __aarch64__ */
                    {
                        const auto bitmask = wrapper::vceq(vin, wrapper::vdup_n(0.f, ExactTagType{}));
                        tmp                 = wrapper::vinv(wrapper::vinvsqrt(wrapper::vadd(vin, mask_float_vector(delta, bitmask))));
                        tmp                 = mask_float_vector(tmp, wrapper::vnot(bitmask));
                    }
#endif /* __aarch64__ */
                    break;
                case ActivationLayerInfo::ActivationFunction::SQUARE:
                    tmp = wrapper::vmul(vin, vin);
                    break;
                case ActivationLayerInfo::ActivationFunction::TANH:
                    tmp = wrapper::vmul(va, wrapper::vtanh(wrapper::vmul(vb, vin)));
                    break;
                case ActivationLayerInfo::ActivationFunction::IDENTITY:
                    tmp = vin;
                    break;
                case ActivationLayerInfo::ActivationFunction::HARD_SWISH:
                    tmp = wrapper::vmul(vin, wrapper::vmul(const_inv_6, wrapper::vmin(const_6, wrapper::vmax(const_0, wrapper::vadd(vin, const_3)))));
                    break;
                case ActivationLayerInfo::ActivationFunction::SWISH:
                    tmp = wrapper::vmul(vin, wrapper::vinv(wrapper::vadd(const_1, wrapper::vexpq(wrapper::vneg(wrapper::vmul(va, vin))))));
                    break;
#ifdef __aarch64__
                case ActivationLayerInfo::ActivationFunction::GELU:
                    tmp = wrapper::vmul(vin, wrapper::vmul(const_inv_2, wrapper::vadd(const_1, wrapper::verf(wrapper::vmul(vin, const_inv_sqrt_2)))));
                    break;
#endif /* __aarch64__ */
                default:
                    ARM_COMPUTE_ERROR("Unsupported activation function");
            }
            wrapper::vstore(output_ptr + x, tmp);
        }
        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            const T in = *(reinterpret_cast<const T *>(input_ptr + x));
            T       tmp;
            switch(act)
            {
                case ActivationLayerInfo::ActivationFunction::ABS:
                    tmp = std::abs(in);
                    break;
                case ActivationLayerInfo::ActivationFunction::LINEAR:
                    tmp = a * in + b;
                    break;
                case ActivationLayerInfo::ActivationFunction::LOGISTIC:
                    tmp = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-in));
                    break;
                case ActivationLayerInfo::ActivationFunction::RELU:
                    tmp = std::max<T>(static_cast<T>(0), in);
                    break;
                case ActivationLayerInfo::ActivationFunction::BOUNDED_RELU:
                    tmp = std::min<T>(a, std::max(static_cast<T>(0), in));
                    break;
                case ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU:
                    tmp = std::min<T>(a, std::max<T>(b, in));
                    break;
                case ActivationLayerInfo::ActivationFunction::LEAKY_RELU:
                    tmp = (in > 0) ? in : a * in;
                    break;
                case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
                    tmp = (in > soft_relu_thresh) ? in : std::log(static_cast<T>(1) + std::exp(in));
                    break;
                case ActivationLayerInfo::ActivationFunction::ELU:
                    tmp = (in >= 0) ? in : a * (std::exp(in) - 1);
                    break;
                case ActivationLayerInfo::ActivationFunction::SQRT:
                    tmp = std::sqrt(in);
                    break;
                case ActivationLayerInfo::ActivationFunction::SQUARE:
                    tmp = in * in;
                    break;
                case ActivationLayerInfo::ActivationFunction::TANH:
                    tmp = a * std::tanh(b * in);
                    break;
                case ActivationLayerInfo::ActivationFunction::IDENTITY:
                    tmp = in;
                    break;
                case ActivationLayerInfo::ActivationFunction::HARD_SWISH:
                    tmp = in * ((std::min(std::max((in + 3), 0.0f), 6.0f)) * 0.166666667f);
                    break;
                case ActivationLayerInfo::ActivationFunction::SWISH:
                    tmp = in / (static_cast<T>(1) + std::exp(-a*in));
                    break;
                case ActivationLayerInfo::ActivationFunction::GELU:
                    tmp = in * static_cast<T>(0.5f * (1.0f + erff(static_cast<float>(in) / 1.41421356237f)));
                    break;
                default:
                    ARM_COMPUTE_ERROR("Unsupported activation function");
            }
            *(output_ptr + x) = tmp;
        }
    },
    input, output);
}
} // namespace cpu
} // namespace arm_compute
