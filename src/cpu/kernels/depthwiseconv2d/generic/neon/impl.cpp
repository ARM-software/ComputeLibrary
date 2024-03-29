/*
 * Copyright (c) 2019-2023 Arm Limited.
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
#include "src/cpu/kernels/depthwiseconv2d/generic/neon/impl.h"

#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/function_info/ConvolutionInfo.h"

#include "src/core/NEON/wrapper/wrapper.h"

namespace arm_compute
{
namespace cpu
{
inline int32x4_t saturating_doubling_high_mul(const int32x4_t &a, const int32_t &b)
{
    return vqrdmulhq_n_s32(a, b);
}

inline int32_t saturating_doubling_high_mul(const int32_t &a, const int32_t &b)
{
    return vget_lane_s32(vqrdmulh_n_s32(vdup_n_s32(a), b), 0);
}

inline int32x4_t rounding_divide_by_exp2(const int32x4_t &x, const int exponent)
{
    const int32x4_t shift = vdupq_n_s32(-exponent);
    const int32x4_t fixup = vshrq_n_s32(vandq_s32(x, shift), 31);
    const int32x4_t fixed = vqaddq_s32(x, fixup);
    return vrshlq_s32(fixed, shift);
}

inline int32x2_t rounding_divide_by_exp2(const int32x2_t &x, const int exponent)
{
    const int32x2_t shift = vdup_n_s32(-exponent);
    const int32x2_t fixup = vshr_n_s32(vand_s32(x, shift), 31);
    const int32x2_t fixed = vqadd_s32(x, fixup);
    return vrshl_s32(fixed, shift);
}

inline int32_t rounding_divide_by_exp2(const int32_t &x, const int exponent)
{
    const int32x2_t xs = vdup_n_s32(x);
    return vget_lane_s32(rounding_divide_by_exp2(xs, exponent), 0);
}

namespace
{
template <typename T, typename TW>
void depthwise_loop_multiplier1_quantized(const ITensor       *src,
                                          const ITensor       *weights,
                                          const ITensor       *biases,
                                          ITensor             *dst,
                                          const PadStrideInfo &conv_info,
                                          const Size2D        &dilation,
                                          std::vector<int>     output_multiplier,
                                          std::vector<int>     output_shift,
                                          const Window        &window,
                                          bool                 has_biases) // NOLINT
{
    ARM_COMPUTE_UNUSED(output_multiplier, output_shift);
    constexpr auto element_per_vector = vector_size / sizeof(T);
    using VectorType                  = typename wrapper::traits::neon_vector<T, element_per_vector>::type;
    using TagType                     = typename wrapper::traits::neon_vector<T, element_per_vector>::tag_type;
    using AccType                     = int32_t;
    using AccArrayType                = std::array<AccType, element_per_vector>;

    const auto out_of_bound_value =
        PixelValue(static_cast<uint64_t>(0), src->info()->data_type(), src->info()->quantization_info()).get<T>();
    const auto out_of_bound_vector = wrapper::vdup_n(static_cast<T>(out_of_bound_value), TagType{});

    const auto run_info = DepthwiseConvolutionRunInfo(*src->info(), *weights->info(), conv_info, window);

    const int32_t input_qoffset   = src->info()->quantization_info().uniform().offset;
    const int32_t weights_qoffset = weights->info()->quantization_info().uniform().offset;
    const int32_t output_qoffset  = dst->info()->quantization_info().uniform().offset;
    const int32_t k_offset        = run_info.weights_width * run_info.weights_height * input_qoffset * weights_qoffset;

    Window execution_window = window;
    execution_window.set(Window::DimX, dim_single_unit_step);

    Window win_input = window;
    win_input.set(Window::DimX, dim_manual_loop);
    win_input.set(Window::DimY, dim_manual_loop);
    win_input.set(Window::DimZ, dim_manual_loop);

    Window win_weights = win_input;
    win_weights.set(Window::DimW, dim_manual_loop);

    Window win_output = window;
    win_output.set(Window::DimX, dim_manual_loop);

    Iterator input_it(src, win_input);
    Iterator weights_it(weights, win_weights);
    Iterator output_it(dst, win_output);
    Iterator biases_it{};

    if (has_biases)
    {
        biases_it = Iterator(biases, win_weights);
    }

    execute_window_loop(
        execution_window,
        [&](const Coordinates &id)
        {
            const int32_t input_y           = id.y() * run_info.conv_stride_x - run_info.conv_pad_left;
            const int32_t input_z           = id.z() * run_info.conv_stride_y - run_info.conv_pad_top;
            const int64_t base_input_offset = input_y * run_info.input_stride_y + input_z * run_info.input_stride_z;
            auto const    base_weights_ptr  = weights_it.ptr();
            size_t        x                 = run_info.x_start;

            for (; x < run_info.x_leftover_start; x += run_info.x_step)
            {
                AccArrayType acc{};
                AccArrayType in_sum{};
                AccArrayType we_sum{};

                auto weights_ptr  = base_weights_ptr;
                auto input_offset = base_input_offset;

                for (size_t h = 0; h < run_info.weights_height; ++h)
                {
                    int64_t offs = input_offset + x * sizeof(T);
                    for (size_t w = 0; w < run_info.weights_width; ++w)
                    {
                        const bool is_valid_region = is_valid_input_region(input_y, input_z, w, h, run_info, dilation);
                        const auto input_vals =
                            is_valid_region
                                ? wrapper::vload(reinterpret_cast<T *>(
                                      input_it.ptr() + std::min(static_cast<size_t>(offs), run_info.input_max_offset)))
                                : out_of_bound_vector;
                        const auto weights_vals =
                            wrapper::vload(reinterpret_cast<TW *>(weights_ptr + w * run_info.weights_stride_y) + x);

                        for (size_t i = 0; i < element_per_vector; ++i)
                        {
                            acc.at(i) += input_vals[i] * weights_vals[i];
                            in_sum.at(i) += input_vals[i];
                            we_sum.at(i) += weights_vals[i];
                        }

                        offs += dilation.x() * run_info.input_stride_y;
                    }

                    weights_ptr += run_info.weights_stride_z;
                    input_offset += dilation.y() * run_info.input_stride_z;
                }

                VectorType out_vals = wrapper::vdup_n(static_cast<T>(0), TagType{});
                for (size_t i = 0; i < element_per_vector; ++i)
                {
                    acc.at(i) -= in_sum.at(i) * weights_qoffset;
                    acc.at(i) -= we_sum.at(i) * input_qoffset;
                    acc.at(i) += k_offset;

                    if (has_biases)
                    {
                        acc.at(i) += *(reinterpret_cast<int32_t *>(biases_it.ptr() + i * sizeof(int32_t)) + x);
                    }

                    const int32_t out_mul   = output_multiplier.at(x + i);
                    const int32_t out_shift = output_shift.at(x + i);
                    if (out_shift < 0)
                    {
                        acc.at(i) =
                            saturating_doubling_high_mul(acc.at(i) * (1 << (-out_shift)), out_mul) + output_qoffset;
                    }
                    else
                    {
                        acc.at(i) =
                            rounding_divide_by_exp2(saturating_doubling_high_mul(acc.at(i), out_mul), out_shift) +
                            output_qoffset;
                    }
                    out_vals[i] = static_cast<T>(utility::clamp<AccType, T>(acc.at(i)));
                }

                wrapper::vstore(reinterpret_cast<T *>(output_it.ptr()) + x, out_vals);
            }

            // left-over
            for (; x < run_info.x_end; ++x)
            {
                AccType acc    = 0;
                AccType in_sum = 0;
                AccType we_sum = 0;

                auto weights_ptr  = base_weights_ptr;
                auto input_offset = base_input_offset;

                for (size_t h = 0; h < run_info.weights_height; ++h)
                {
                    int64_t offs = input_offset + x * sizeof(T);
                    for (size_t w = 0; w < run_info.weights_width; ++w)
                    {
                        const bool is_valid_region = is_valid_input_region(input_y, input_z, w, h, run_info, dilation);
                        const auto input_val =
                            is_valid_region
                                ? *reinterpret_cast<T *>(input_it.ptr() +
                                                         std::min(static_cast<size_t>(offs), run_info.input_max_offset))
                                : out_of_bound_value;
                        const auto weights_val =
                            *(reinterpret_cast<TW *>(weights_ptr + w * run_info.weights_stride_y) + x);

                        acc += input_val * weights_val;
                        in_sum += input_val;
                        we_sum += weights_val;

                        offs += dilation.x() * run_info.input_stride_y;
                    }

                    weights_ptr += run_info.weights_stride_z;
                    input_offset += dilation.y() * run_info.input_stride_z;
                }

                T out_vals{0};

                acc -= in_sum * weights_qoffset;
                acc -= we_sum * input_qoffset;
                acc += k_offset;

                if (has_biases)
                {
                    acc += *(reinterpret_cast<int32_t *>(biases_it.ptr()) + x);
                }

                const int32_t out_mul   = output_multiplier.at(x);
                const int32_t out_shift = output_shift.at(x);

                if (out_shift < 0)
                {
                    acc = saturating_doubling_high_mul(acc * (1 << (-out_shift)), out_mul) + output_qoffset;
                }
                else
                {
                    acc =
                        rounding_divide_by_exp2(saturating_doubling_high_mul(acc, out_mul), out_shift) + output_qoffset;
                }

                out_vals                                      = static_cast<T>(utility::clamp<AccType, T>(acc));
                *(reinterpret_cast<T *>(output_it.ptr()) + x) = out_vals;
            }
        },
        input_it, weights_it, biases_it, output_it);
}

template <typename T, typename TW>
void depthwise_loop_generic_quantized(const ITensor       *src,
                                      const ITensor       *weights,
                                      const ITensor       *biases,
                                      ITensor             *dst,
                                      const PadStrideInfo &conv_info,
                                      const Size2D        &dilation,
                                      unsigned int         depth_multiplier,
                                      std::vector<int>     output_multiplier,
                                      std::vector<int>     output_shift,
                                      const Window        &window,
                                      bool                 has_biases) // NOLINT
{
    using AccType = int32_t;

    const auto run_info =
        DepthwiseConvolutionRunInfo(*src->info(), *weights->info(), conv_info, window, depth_multiplier);

    const auto out_of_bound_value =
        PixelValue(static_cast<uint64_t>(0), src->info()->data_type(), src->info()->quantization_info()).get<T>();

    const int32_t input_qoffset   = src->info()->quantization_info().uniform().offset;
    const int32_t weights_qoffset = weights->info()->quantization_info().uniform().offset;
    const int32_t output_qoffset  = dst->info()->quantization_info().uniform().offset;
    const int32_t k_offset        = run_info.weights_width * run_info.weights_height * input_qoffset * weights_qoffset;

    Window execution_window = window;
    execution_window.set(Window::DimX, Window::Dimension(0, run_info.input_depth, 1));

    Window win_input = execution_window;
    win_input.set(Window::DimY, dim_manual_loop);
    win_input.set(Window::DimZ, dim_manual_loop);

    Window win_weights = window;
    win_weights.set_dimension_step(Window::DimX, run_info.x_step);
    win_weights.set(Window::DimY, dim_manual_loop);
    win_weights.set(Window::DimZ, dim_manual_loop);
    win_weights.set(Window::DimW, dim_manual_loop);

    Window win_output = window;
    win_output.set_dimension_step(Window::DimX, run_info.x_step);

    Iterator input_it(src, win_input);
    Iterator weights_it(weights, win_weights);
    Iterator output_it(dst, win_output);
    Iterator biases_it{};

    if (has_biases)
    {
        biases_it = Iterator(biases, win_weights);
    }

    execute_window_loop(
        execution_window,
        [&](const Coordinates &id)
        {
            std::vector<AccType> acc(depth_multiplier, 0);
            std::vector<AccType> we_sum(depth_multiplier, 0);
            AccType              in_sum = 0;

            const int32_t input_y      = id.y() * run_info.conv_stride_x - run_info.conv_pad_left;
            const int32_t input_z      = id.z() * run_info.conv_stride_y - run_info.conv_pad_top;
            int64_t       input_offset = input_y * run_info.input_stride_y + input_z * run_info.input_stride_z;

            auto weights_ptr = weights_it.ptr();
            for (size_t h = 0; h < run_info.weights_height; ++h)
            {
                int offs = input_offset;
                for (size_t w = 0; w < run_info.weights_width; ++w)
                {
                    const bool is_valid_region = is_valid_input_region(input_y, input_z, w, h, run_info, dilation);
                    const auto input_val =
                        is_valid_region ? *(reinterpret_cast<T *>(input_it.ptr() + std::min(static_cast<size_t>(offs),
                                                                                            run_info.input_max_offset)))
                                        : out_of_bound_value;

                    for (size_t m = 0; m < depth_multiplier; ++m)
                    {
                        const auto weights_val =
                            *(reinterpret_cast<TW *>(weights_ptr + m * sizeof(T) + w * run_info.weights_stride_y));
                        acc.at(m) += input_val * weights_val;

                        we_sum.at(m) += weights_val;
                    }

                    offs += dilation.x() * run_info.input_stride_y;
                    in_sum += input_val;
                }

                weights_ptr += run_info.weights_stride_z;
                input_offset += dilation.y() * run_info.input_stride_z;
            }

            for (size_t m = 0; m < depth_multiplier; ++m)
            {
                acc.at(m) -= in_sum * weights_qoffset;
                acc.at(m) -= we_sum.at(m) * input_qoffset;
                acc.at(m) += k_offset;

                if (has_biases)
                {
                    acc.at(m) += *(reinterpret_cast<int32_t *>(biases_it.ptr() + m * sizeof(int32_t)));
                }

                const int32_t out_mul   = output_multiplier.at(id.x() * depth_multiplier + m);
                const int32_t out_shift = output_shift.at(id.x() * depth_multiplier + m);
                if (out_shift < 0)
                {
                    acc.at(m) = saturating_doubling_high_mul(acc.at(m) * (1 << (-out_shift)), out_mul) + output_qoffset;
                }
                else
                {
                    acc.at(m) = rounding_divide_by_exp2(saturating_doubling_high_mul(acc.at(m), out_mul), out_shift) +
                                output_qoffset;
                }
                *(reinterpret_cast<T *>(output_it.ptr() + m * sizeof(T))) =
                    static_cast<T>(utility::clamp<AccType, T>(acc.at(m)));
            }
        },
        input_it, weights_it, biases_it, output_it);
}

template <typename T, typename TW>
void depthwise_loop_pow2_quantized_per_tensor(const ITensor       *src,
                                              const ITensor       *weights,
                                              const ITensor       *biases,
                                              ITensor             *dst,
                                              const PadStrideInfo &conv_info,
                                              const Size2D        &dilation,
                                              unsigned int         depth_multiplier,
                                              std::vector<int>     output_multiplier,
                                              std::vector<int>     output_shift,
                                              const Window        &window,
                                              bool                 has_biases) // NOLINT
{
    constexpr int half_vec = vector_size / 2;

    using AccType          = int32_t;
    using AccVectorType    = typename wrapper::traits::neon_vector<AccType, half_vec>::type;
    using AccVectorTagType = typename wrapper::traits::neon_vector<AccType, half_vec>::tag_type;
    using TagType          = typename wrapper::traits::neon_vector<T, vector_size>::tag_type;

    const auto run_info =
        DepthwiseConvolutionRunInfo(*src->info(), *weights->info(), conv_info, window, depth_multiplier);

    const auto input_qoffset_vec = wrapper::vreinterpret(
        wrapper::vmovl(wrapper::vdup_n(static_cast<T>(src->info()->quantization_info().uniform().offset), TagType{})));
    const auto weights_qoffset_vec = wrapper::vreinterpret(wrapper::vmovl(
        wrapper::vdup_n(static_cast<TW>(weights->info()->quantization_info().uniform().offset), TagType{})));
    const auto output_qoffset_vec  = wrapper::vdup_n(dst->info()->quantization_info().uniform().offset,
                                                     arm_compute::wrapper::traits::vector_128_tag{});

    const auto lower = wrapper::vdup_n(static_cast<AccType>(std::numeric_limits<T>::lowest()), AccVectorTagType{});
    const auto upper = wrapper::vdup_n(static_cast<AccType>(std::numeric_limits<T>::max()), AccVectorTagType{});
    const auto zero  = wrapper::vdup_n(static_cast<AccType>(0), AccVectorTagType{});

    const auto out_mul   = output_multiplier.at(0);
    const auto out_shift = output_shift.at(0);

    Window execution_window = window;
    execution_window.set(Window::DimX, Window::Dimension(0, run_info.input_depth, 1));

    Window win_input = execution_window;
    win_input.set(Window::DimY, dim_manual_loop);
    win_input.set(Window::DimZ, dim_manual_loop);

    Window win_weights = window;
    win_weights.set_dimension_step(Window::DimX, run_info.x_step);
    win_weights.set(Window::DimY, dim_manual_loop);
    win_weights.set(Window::DimZ, dim_manual_loop);
    win_weights.set(Window::DimW, dim_manual_loop);

    Window win_output = window;
    win_output.set_dimension_step(Window::DimX, run_info.x_step);

    Iterator input_it(src, win_input);
    Iterator weights_it(weights, win_weights);
    Iterator output_it(dst, win_output);
    Iterator biases_it{};

    if (has_biases)
    {
        biases_it = Iterator(biases, win_weights);
    }

    std::vector<AccVectorType> acc0(depth_multiplier / vector_size);
    std::vector<AccVectorType> acc1(depth_multiplier / vector_size);

    execute_window_loop(
        execution_window,
        [&](const Coordinates &id)
        {
            std::fill(begin(acc0), end(acc0), zero);
            std::fill(begin(acc1), end(acc1), zero);

            const int32_t input_y      = id.y() * run_info.conv_stride_x - run_info.conv_pad_left;
            const int32_t input_z      = id.z() * run_info.conv_stride_y - run_info.conv_pad_top;
            int64_t       input_offset = input_y * run_info.input_stride_y + input_z * run_info.input_stride_z;

            auto weights_ptr = weights_it.ptr();
            for (size_t h = 0; h < run_info.weights_height; ++h)
            {
                const int32_t current_h = input_z + h * dilation.y();
                if (current_h >= 0 && current_h < static_cast<int32_t>(run_info.input_height))
                {
                    int offs = input_offset;
                    for (size_t w = 0; w < run_info.weights_width; ++w)
                    {
                        const int32_t current_w = input_y + w * dilation.x();
                        if (current_w >= 0 && current_w < static_cast<int32_t>(run_info.input_width))
                        {
                            const auto input_8x8 = wrapper::vdup_n(
                                *(reinterpret_cast<T *>(
                                    input_it.ptr() + std::min(static_cast<size_t>(offs), run_info.input_max_offset))),
                                TagType{});
                            const auto input_s16x8   = wrapper::vreinterpret(wrapper::vmovl(input_8x8));
                            const auto input_no_offs = wrapper::vsub(input_s16x8, input_qoffset_vec);

                            for (size_t m = 0, i = 0; m < depth_multiplier; m += vector_size, ++i)
                            {
                                const auto weights_8x8     = wrapper::vload(reinterpret_cast<TW *>(
                                    weights_ptr + m * sizeof(T) + w * run_info.weights_stride_y));
                                const auto weights_s16x8   = wrapper::vreinterpret(wrapper::vmovl(weights_8x8));
                                const auto weights_no_offs = wrapper::vsub(weights_s16x8, weights_qoffset_vec);

                                acc0.at(i) = wrapper::vmlal(acc0.at(i), wrapper::vgetlow(input_no_offs),
                                                            wrapper::vgetlow(weights_no_offs));
                                acc1.at(i) = wrapper::vmlal(acc1.at(i), wrapper::vgethigh(input_no_offs),
                                                            wrapper::vgethigh(weights_no_offs));
                            }
                        }

                        offs += dilation.x() * run_info.input_stride_y;
                    }
                }

                weights_ptr += run_info.weights_stride_z;
                input_offset += dilation.y() * run_info.input_stride_z;
            }

            for (size_t m = 0, i = 0; m < depth_multiplier; m += vector_size, ++i)
            {
                if (has_biases)
                {
                    const auto bias_val0 =
                        wrapper::vloadq(reinterpret_cast<int32_t *>(biases_it.ptr() + m * sizeof(int32_t)));
                    const auto bias_val1 = wrapper::vloadq(
                        reinterpret_cast<int32_t *>(biases_it.ptr() + (m + half_vec) * sizeof(int32_t)));

                    acc0.at(i) = wrapper::vadd(acc0.at(i), bias_val0);
                    acc1.at(i) = wrapper::vadd(acc1.at(i), bias_val1);
                }

                if (out_shift < 0)
                {
                    acc0.at(i) = wrapper::vadd(saturating_doubling_high_mul(acc0.at(i) * (1 << (-out_shift)), out_mul),
                                               output_qoffset_vec);
                    acc1.at(i) = wrapper::vadd(saturating_doubling_high_mul(acc1.at(i) * (1 << (-out_shift)), out_mul),
                                               output_qoffset_vec);
                }
                else
                {
                    acc0.at(i) = wrapper::vadd(
                        rounding_divide_by_exp2(saturating_doubling_high_mul(acc0.at(i), out_mul), out_shift),
                        output_qoffset_vec);
                    acc1.at(i) = wrapper::vadd(
                        rounding_divide_by_exp2(saturating_doubling_high_mul(acc1.at(i), out_mul), out_shift),
                        output_qoffset_vec);
                }

                acc0.at(i) = wrapper::vmin(wrapper::vmax(acc0.at(i), lower), upper);
                acc1.at(i) = wrapper::vmin(wrapper::vmax(acc1.at(i), lower), upper);

                const auto out_val = wrapper::vcombine(wrapper::vmovn(acc0.at(i)), wrapper::vmovn(acc1.at(i)));

                if (std::is_same<T, uint8_t>::value)
                {
                    wrapper::vstore(reinterpret_cast<uint8_t *>(output_it.ptr() + m * sizeof(uint8_t)),
                                    wrapper::vqmovn(vreinterpretq_u16_s16(out_val)));
                }
                else
                {
                    wrapper::vstore(reinterpret_cast<int8_t *>(output_it.ptr() + m * sizeof(int8_t)),
                                    wrapper::vqmovn(out_val));
                }
            }
        },
        input_it, weights_it, biases_it, output_it);
}
} // namespace

template <typename T, typename TW>
void run_depthwise_quanitized8bit(const ITensor         *src,
                                  const ITensor         *weights,
                                  const ITensor         *biases,
                                  ITensor               *dst,
                                  const Window          &window,
                                  bool                   has_biases,
                                  const ConvolutionInfo &info)
{
    PadStrideInfo    conv_info        = info.pad_stride_info;
    unsigned int     depth_multiplier = info.depth_multiplier;
    Size2D           dilation         = info.dilation;
    std::vector<int> output_multiplier;
    std::vector<int> output_shift;

    const auto input_scale   = src->info()->quantization_info().uniform().scale;
    const auto output_scale  = dst->info()->quantization_info().uniform().scale;
    auto       weights_scale = weights->info()->quantization_info().scale();

    if (!is_data_type_quantized_per_channel(weights->info()->data_type()))
    {
        for (size_t i = 1; i < weights->info()->dimension(channel_idx); ++i)
        {
            weights_scale.push_back(weights_scale.front());
        }
    }

    for (const auto &s : weights_scale)
    {
        int32_t     out_mult   = 0;
        int32_t     out_shift  = 0;
        const float multiplier = input_scale * s / output_scale;
        arm_compute::quantization::calculate_quantized_multiplier(multiplier, &out_mult, &out_shift);

        output_multiplier.push_back(out_mult);
        output_shift.push_back(out_shift);
    }

    if (depth_multiplier == 1)
    {
        depthwise_loop_multiplier1_quantized<T, TW>(src, weights, biases, dst, conv_info, dilation, output_multiplier,
                                                    output_shift, window, has_biases);
    }
    else
    {
        const bool is_pow2                 = ((depth_multiplier & (depth_multiplier - 1)) == 0);
        const bool is_quantized_per_tensor = !(is_data_type_quantized_per_channel(weights->info()->data_type()));

        if (is_pow2 && is_quantized_per_tensor && depth_multiplier >= 8)
        {
            depthwise_loop_pow2_quantized_per_tensor<T, TW>(src, weights, biases, dst, conv_info, dilation,
                                                            depth_multiplier, output_multiplier, output_shift, window,
                                                            has_biases);
        }
        else
        {
            depthwise_loop_generic_quantized<T, TW>(src, weights, biases, dst, conv_info, dilation, depth_multiplier,
                                                    output_multiplier, output_shift, window, has_biases);
        }
    }
}
template void run_depthwise_quanitized8bit<uint8_t, uint8_t>(const ITensor         *src,
                                                             const ITensor         *weights,
                                                             const ITensor         *biases,
                                                             ITensor               *dst,
                                                             const Window          &window,
                                                             bool                   has_biases,
                                                             const ConvolutionInfo &info);
template void run_depthwise_quanitized8bit<int8_t, int8_t>(const ITensor         *src,
                                                           const ITensor         *weights,
                                                           const ITensor         *biases,
                                                           ITensor               *dst,
                                                           const Window          &window,
                                                           bool                   has_biases,
                                                           const ConvolutionInfo &info);
template void run_depthwise_quanitized8bit<uint8_t, int8_t>(const ITensor         *src,
                                                            const ITensor         *weights,
                                                            const ITensor         *biases,
                                                            ITensor               *dst,
                                                            const Window          &window,
                                                            bool                   has_biases,
                                                            const ConvolutionInfo &info);
} // namespace cpu
} // namespace arm_compute
