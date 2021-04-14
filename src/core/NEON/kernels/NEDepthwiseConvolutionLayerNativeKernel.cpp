/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#include "src/core/NEON/kernels/NEDepthwiseConvolutionLayerNativeKernel.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "src/core/CPP/Validate.h"
#include "src/core/NEON/kernels/convolution/depthwise/impl_qa8_qa8.hpp"
#include "src/core/NEON/wrapper/traits.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
namespace
{
constexpr auto data_layout = DataLayout::NHWC;
const size_t   width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
const size_t   height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
const size_t   channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

constexpr auto   dim_manual_loop      = Window::Dimension(0, 0, 0);
constexpr auto   dim_single_unit_step = Window::Dimension(0, 1, 1);
constexpr size_t vector_size          = 8;

struct DepthwiseConvolutionRunInfo
{
    const size_t   num_read_elements_per_iteration;
    const uint32_t x_start;
    const uint32_t x_end;
    const uint32_t x_step;
    const uint32_t x_leftover_start;
    const size_t   input_stride_y;
    const size_t   input_stride_z;
    const size_t   input_max_offset;
    const size_t   weights_width;
    const size_t   weights_height;
    const size_t   weights_stride_y;
    const size_t   weights_stride_z;
    const size_t   conv_stride_x;
    const size_t   conv_stride_y;
    const size_t   conv_pad_left;
    const size_t   conv_pad_top;
    const size_t   input_height;
    const size_t   input_width;
    const size_t   input_depth;

    DepthwiseConvolutionRunInfo(const ITensorInfo &input, const ITensorInfo &weights, const PadStrideInfo &conv_info, const Window &w, uint32_t depth_multiplier = 1)
        : num_read_elements_per_iteration((depth_multiplier == 1 ? (vector_size / element_size_from_data_type(input.data_type())) : 1)),
          x_start(w.x().start()),
          x_end(w.x().end()),
          x_step(static_cast<uint32_t>(num_read_elements_per_iteration * depth_multiplier)),
          x_leftover_start(std::max(static_cast<int32_t>(w.x().end()) - static_cast<int32_t>(x_step) + 1, int32_t(0))),
          input_stride_y(input.strides_in_bytes().y()),
          input_stride_z(input.strides_in_bytes().z()),
          input_max_offset(input.strides_in_bytes().z() * input.dimension(height_idx) - (input.padding().bottom + input.padding().top) * input.strides_in_bytes().y()),
          weights_width(weights.dimension(width_idx)),
          weights_height(weights.dimension(height_idx)),
          weights_stride_y(weights.strides_in_bytes().y()),
          weights_stride_z(weights.strides_in_bytes().z()),
          conv_stride_x(conv_info.stride().first),
          conv_stride_y(conv_info.stride().second),
          conv_pad_left(conv_info.pad_left()),
          conv_pad_top(conv_info.pad_top()),
          input_height(input.dimension(height_idx)),
          input_width(input.dimension(width_idx)),
          input_depth(input.dimension(channel_idx))
    {
    }
};

inline bool is_valid_input_region(int32_t base_w, uint32_t base_h, uint32_t w, uint32_t h, const DepthwiseConvolutionRunInfo &run_info, const Size2D &dilation)
{
    const int32_t current_h  = base_h + h * dilation.y();
    const bool    is_valid_h = current_h >= 0 && current_h < static_cast<int32_t>(run_info.input_height);

    const int32_t current_w  = base_w + w * dilation.x();
    const bool    is_valid_w = current_w >= 0 && current_w < static_cast<int32_t>(run_info.input_width);

    return is_valid_h && is_valid_w;
}

template <typename T>
void depthwise_loop_multiplier1_fp(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info,
                                   const Size2D &dilation, const Window &window, bool has_biases)
{
    constexpr auto element_per_vector = vector_size / sizeof(T);
    using VectorType                  = typename wrapper::traits::neon_vector<T, element_per_vector>::type;
    using TagType                     = typename wrapper::traits::neon_vector<T, element_per_vector>::tag_type;

    const auto run_info = DepthwiseConvolutionRunInfo(*input->info(), *weights->info(), conv_info, window);

    const VectorType zero_vector = wrapper::vdup_n(static_cast<T>(0), TagType{});

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

    Iterator input_it(input, win_input);
    Iterator weights_it(weights, win_weights);
    Iterator output_it(output, win_output);
    Iterator biases_it{};

    if(has_biases)
    {
        biases_it = Iterator(biases, win_weights);
    }

    execute_window_loop(execution_window, [&](const Coordinates & id)
    {
        const int32_t input_y           = id.y() * run_info.conv_stride_x - run_info.conv_pad_left;
        const int32_t input_z           = id.z() * run_info.conv_stride_y - run_info.conv_pad_top;
        const int64_t base_input_offset = input_y * run_info.input_stride_y + input_z * run_info.input_stride_z;

        auto const base_weights_ptr = weights_it.ptr();
        uint32_t   x                = run_info.x_start;

        for(; x < run_info.x_leftover_start; x += run_info.x_step)
        {
            VectorType acc          = zero_vector;
            auto       weights_ptr  = base_weights_ptr;
            int64_t    input_offset = base_input_offset;

            for(uint32_t h = 0; h < run_info.weights_height; ++h)
            {
                int64_t offs = input_offset + x * sizeof(T);
                for(uint32_t w = 0; w < run_info.weights_width; ++w)
                {
                    const bool is_valid_region = is_valid_input_region(input_y, input_z, w, h, run_info, dilation);
                    const auto input_vals      = is_valid_region ?
                                                 wrapper::vload(reinterpret_cast<T *>(input_it.ptr() + std::min(static_cast<size_t>(offs), run_info.input_max_offset))) :
                                                 zero_vector;
                    const auto weights_vals = wrapper::vload(reinterpret_cast<T *>(weights_ptr + w * run_info.weights_stride_y) + x);
                    acc                     = wrapper::vmla(acc, weights_vals, input_vals);

                    offs += dilation.x() * run_info.input_stride_y;
                }

                weights_ptr += run_info.weights_stride_z;
                input_offset += dilation.y() * run_info.input_stride_z;
            }

            if(has_biases)
            {
                const auto biases_vals = wrapper::vload(reinterpret_cast<T *>(biases_it.ptr()) + x);
                acc                    = wrapper::vadd(acc, biases_vals);
            }

            wrapper::vstore(reinterpret_cast<T *>(output_it.ptr()) + x, acc);
        }

        for(; x < run_info.x_end; ++x)
        {
            auto    acc_scalar   = T{ 0 };
            auto    weights_ptr  = base_weights_ptr;
            int64_t input_offset = base_input_offset;

            for(size_t h = 0; h < run_info.weights_height; ++h)
            {
                int64_t offs = input_offset + x * sizeof(T);
                for(size_t w = 0; w < run_info.weights_width; ++w)
                {
                    const bool is_valid_region = is_valid_input_region(input_y, input_z, w, h, run_info, dilation);
                    const auto input_vals      = is_valid_region ? *reinterpret_cast<T *>(input_it.ptr() + std::min(static_cast<size_t>(offs), run_info.input_max_offset)) : 0;
                    const auto weights_vals    = *(reinterpret_cast<T *>(weights_ptr + w * run_info.weights_stride_y) + x);

                    acc_scalar += (input_vals * weights_vals);

                    offs += dilation.x() * run_info.input_stride_y;
                }

                weights_ptr += run_info.weights_stride_z;
                input_offset += dilation.y() * run_info.input_stride_z;
            }

            if(has_biases)
            {
                const auto biases_vals = *(reinterpret_cast<T *>(biases_it.ptr()) + x);
                acc_scalar += biases_vals;
            }
            *(reinterpret_cast<T *>(output_it.ptr()) + x) = acc_scalar;
        }
    },
    input_it, weights_it, biases_it, output_it);
}

template <typename T>
void depthwise_loop_generic_fp(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info,
                               const Size2D &dilation, unsigned int depth_multiplier, const Window &window, bool has_biases)
{
    const auto run_info = DepthwiseConvolutionRunInfo(*input->info(), *weights->info(), conv_info, window, depth_multiplier);

    Window execution_window = window;
    execution_window.set(Window::DimX, Window::Dimension(0, run_info.input_depth, 1));

    Window win_input = execution_window;
    win_input.set(Window::DimX, Window::Dimension(0, run_info.input_depth, 1));
    win_input.set(Window::DimY, dim_manual_loop);
    win_input.set(Window::DimZ, dim_manual_loop);

    Window win_weights = window;
    win_weights.set_dimension_step(Window::DimX, run_info.x_step);
    win_weights.set(Window::DimY, dim_manual_loop);
    win_weights.set(Window::DimZ, dim_manual_loop);
    win_weights.set(Window::DimW, dim_manual_loop);

    Window win_output = window;
    win_output.set_dimension_step(Window::DimX, run_info.x_step);

    Iterator input_it(input, win_input);
    Iterator weights_it(weights, win_weights);
    Iterator output_it(output, win_output);
    Iterator biases_it{};

    if(has_biases)
    {
        biases_it = Iterator(biases, win_weights);
    }

    execute_window_loop(execution_window, [&](const Coordinates & id)
    {
        std::vector<T> acc(depth_multiplier, static_cast<T>(0));

        const int input_y      = id.y() * run_info.conv_stride_x - run_info.conv_pad_left;
        const int input_z      = id.z() * run_info.conv_stride_y - run_info.conv_pad_top;
        int       input_offset = input_y * run_info.input_stride_y + input_z * run_info.input_stride_z;

        auto weights_ptr = weights_it.ptr();
        for(size_t h = 0; h < run_info.weights_height; ++h)
        {
            int offs = input_offset;
            for(size_t w = 0; w < run_info.weights_width; ++w)
            {
                const bool is_valid_region = is_valid_input_region(input_y, input_z, w, h, run_info, dilation);
                const auto input_val       = is_valid_region ? *(reinterpret_cast<T *>(input_it.ptr() + std::min(static_cast<size_t>(offs), run_info.input_max_offset))) : T(0);

                for(size_t m = 0; m < depth_multiplier; ++m)
                {
                    const auto weights_val = *(reinterpret_cast<T *>(weights_ptr + m * sizeof(T) + w * run_info.weights_stride_y));
                    acc.at(m)              = support::cpp11::fma(weights_val, input_val, acc.at(m));
                }

                offs += dilation.x() * run_info.input_stride_y;
            }

            weights_ptr += run_info.weights_stride_z;
            input_offset += dilation.y() * run_info.input_stride_z;
        }

        if(has_biases)
        {
            for(size_t m = 0; m < depth_multiplier; ++m)
            {
                const auto biases_val                                     = *(reinterpret_cast<T *>(biases_it.ptr() + m * sizeof(T)));
                *(reinterpret_cast<T *>(output_it.ptr() + m * sizeof(T))) = acc.at(m) + biases_val;
            }
        }
        else
        {
            for(size_t m = 0; m < depth_multiplier; ++m)
            {
                *(reinterpret_cast<T *>(output_it.ptr() + m * sizeof(T))) = acc.at(m);
            }
        }
    },
    input_it, weights_it, biases_it, output_it);
}

template <typename T, typename TW>
void depthwise_loop_multiplier1_quantized(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info,
                                          const Size2D &dilation, std::vector<int> output_multiplier, std::vector<int> output_shift, const Window &window, bool has_biases)
{
    constexpr auto element_per_vector = vector_size / sizeof(T);
    using VectorType                  = typename wrapper::traits::neon_vector<T, element_per_vector>::type;
    using TagType                     = typename wrapper::traits::neon_vector<T, element_per_vector>::tag_type;
    using AccType                     = int32_t;
    using AccArrayType                = std::array<AccType, element_per_vector>;

    const auto out_of_bound_value  = PixelValue(static_cast<uint64_t>(0), input->info()->data_type(), input->info()->quantization_info()).get<T>();
    const auto out_of_bound_vector = wrapper::vdup_n(static_cast<T>(out_of_bound_value), TagType{});

    const auto run_info = DepthwiseConvolutionRunInfo(*input->info(), *weights->info(), conv_info, window);

    const int32_t input_qoffset   = input->info()->quantization_info().uniform().offset;
    const int32_t weights_qoffset = weights->info()->quantization_info().uniform().offset;
    const int32_t output_qoffset  = output->info()->quantization_info().uniform().offset;
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

    Iterator input_it(input, win_input);
    Iterator weights_it(weights, win_weights);
    Iterator output_it(output, win_output);
    Iterator biases_it{};

    if(has_biases)
    {
        biases_it = Iterator(biases, win_weights);
    }

    execute_window_loop(execution_window, [&](const Coordinates & id)
    {
        const int32_t input_y           = id.y() * run_info.conv_stride_x - run_info.conv_pad_left;
        const int32_t input_z           = id.z() * run_info.conv_stride_y - run_info.conv_pad_top;
        const int64_t base_input_offset = input_y * run_info.input_stride_y + input_z * run_info.input_stride_z;
        auto const    base_weights_ptr  = weights_it.ptr();
        size_t        x                 = run_info.x_start;

        for(; x < run_info.x_leftover_start; x += run_info.x_step)
        {
            AccArrayType acc{};
            AccArrayType in_sum{};
            AccArrayType we_sum{};

            auto weights_ptr  = base_weights_ptr;
            auto input_offset = base_input_offset;

            for(size_t h = 0; h < run_info.weights_height; ++h)
            {
                int64_t offs = input_offset + x * sizeof(T);
                for(size_t w = 0; w < run_info.weights_width; ++w)
                {
                    const bool is_valid_region = is_valid_input_region(input_y, input_z, w, h, run_info, dilation);
                    const auto input_vals      = is_valid_region ?
                                                 wrapper::vload(reinterpret_cast<T *>(input_it.ptr() + std::min(static_cast<size_t>(offs), run_info.input_max_offset))) :
                                                 out_of_bound_vector;
                    const auto weights_vals = wrapper::vload(reinterpret_cast<TW *>(weights_ptr + w * run_info.weights_stride_y) + x);

                    for(size_t i = 0; i < element_per_vector; ++i)
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
            for(size_t i = 0; i < element_per_vector; ++i)
            {
                acc.at(i) -= in_sum.at(i) * weights_qoffset;
                acc.at(i) -= we_sum.at(i) * input_qoffset;
                acc.at(i) += k_offset;

                if(has_biases)
                {
                    acc.at(i) += *(reinterpret_cast<int32_t *>(biases_it.ptr() + i * sizeof(int32_t)) + x);
                }

                const int32_t out_mul   = output_multiplier.at(x + i);
                const int32_t out_shift = output_shift.at(x + i);
                if(out_shift < 0)
                {
                    acc.at(i) = saturating_doubling_high_mul(acc.at(i) * (1 << (-out_shift)), out_mul) + output_qoffset;
                }
                else
                {
                    acc.at(i) = rounding_divide_by_exp2(saturating_doubling_high_mul(acc.at(i), out_mul), out_shift) + output_qoffset;
                }
                out_vals[i] = static_cast<T>(utility::clamp<AccType, T>(acc.at(i)));
            }

            wrapper::vstore(reinterpret_cast<T *>(output_it.ptr()) + x, out_vals);
        }

        // left-over
        for(; x < run_info.x_end; ++x)
        {
            AccType acc    = 0;
            AccType in_sum = 0;
            AccType we_sum = 0;

            auto weights_ptr  = base_weights_ptr;
            auto input_offset = base_input_offset;

            for(size_t h = 0; h < run_info.weights_height; ++h)
            {
                int64_t offs = input_offset + x * sizeof(T);
                for(size_t w = 0; w < run_info.weights_width; ++w)
                {
                    const bool is_valid_region = is_valid_input_region(input_y, input_z, w, h, run_info, dilation);
                    const auto input_val       = is_valid_region ?
                                                 *reinterpret_cast<T *>(input_it.ptr() + std::min(static_cast<size_t>(offs), run_info.input_max_offset)) :
                                                 out_of_bound_value;
                    const auto weights_val = *(reinterpret_cast<TW *>(weights_ptr + w * run_info.weights_stride_y) + x);

                    acc += input_val * weights_val;
                    in_sum += input_val;
                    we_sum += weights_val;

                    offs += dilation.x() * run_info.input_stride_y;
                }

                weights_ptr += run_info.weights_stride_z;
                input_offset += dilation.y() * run_info.input_stride_z;
            }

            T out_vals{ 0 };

            acc -= in_sum * weights_qoffset;
            acc -= we_sum * input_qoffset;
            acc += k_offset;

            if(has_biases)
            {
                acc += *(reinterpret_cast<int32_t *>(biases_it.ptr()) + x);
            }

            const int32_t out_mul   = output_multiplier.at(x);
            const int32_t out_shift = output_shift.at(x);

            if(out_shift < 0)
            {
                acc = saturating_doubling_high_mul(acc * (1 << (-out_shift)), out_mul) + output_qoffset;
            }
            else
            {
                acc = rounding_divide_by_exp2(saturating_doubling_high_mul(acc, out_mul), out_shift) + output_qoffset;
            }

            out_vals                                      = static_cast<T>(utility::clamp<AccType, T>(acc));
            *(reinterpret_cast<T *>(output_it.ptr()) + x) = out_vals;
        }
    },
    input_it, weights_it, biases_it, output_it);
}

template <typename T, typename TW>
void depthwise_loop_generic_quantized(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info,
                                      const Size2D &dilation, unsigned int depth_multiplier, std::vector<int> output_multiplier, std::vector<int> output_shift, const Window &window, bool has_biases)
{
    using AccType = int32_t;

    const auto run_info = DepthwiseConvolutionRunInfo(*input->info(), *weights->info(), conv_info, window, depth_multiplier);

    const auto out_of_bound_value = PixelValue(static_cast<uint64_t>(0), input->info()->data_type(), input->info()->quantization_info()).get<T>();

    const int32_t input_qoffset   = input->info()->quantization_info().uniform().offset;
    const int32_t weights_qoffset = weights->info()->quantization_info().uniform().offset;
    const int32_t output_qoffset  = output->info()->quantization_info().uniform().offset;
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

    Iterator input_it(input, win_input);
    Iterator weights_it(weights, win_weights);
    Iterator output_it(output, win_output);
    Iterator biases_it{};

    if(has_biases)
    {
        biases_it = Iterator(biases, win_weights);
    }

    execute_window_loop(execution_window, [&](const Coordinates & id)
    {
        std::vector<AccType> acc(depth_multiplier, 0);
        std::vector<AccType> we_sum(depth_multiplier, 0);
        AccType              in_sum = 0;

        const int32_t input_y      = id.y() * run_info.conv_stride_x - run_info.conv_pad_left;
        const int32_t input_z      = id.z() * run_info.conv_stride_y - run_info.conv_pad_top;
        int64_t       input_offset = input_y * run_info.input_stride_y + input_z * run_info.input_stride_z;

        auto weights_ptr = weights_it.ptr();
        for(size_t h = 0; h < run_info.weights_height; ++h)
        {
            int offs = input_offset;
            for(size_t w = 0; w < run_info.weights_width; ++w)
            {
                const bool is_valid_region = is_valid_input_region(input_y, input_z, w, h, run_info, dilation);
                const auto input_val       = is_valid_region ? *(reinterpret_cast<T *>(input_it.ptr() + std::min(static_cast<size_t>(offs), run_info.input_max_offset))) : out_of_bound_value;

                for(size_t m = 0; m < depth_multiplier; ++m)
                {
                    const auto weights_val = *(reinterpret_cast<TW *>(weights_ptr + m * sizeof(T) + w * run_info.weights_stride_y));
                    acc.at(m) += input_val * weights_val;

                    we_sum.at(m) += weights_val;
                }

                offs += dilation.x() * run_info.input_stride_y;
                in_sum += input_val;
            }

            weights_ptr += run_info.weights_stride_z;
            input_offset += dilation.y() * run_info.input_stride_z;
        }

        for(size_t m = 0; m < depth_multiplier; ++m)
        {
            acc.at(m) -= in_sum * weights_qoffset;
            acc.at(m) -= we_sum.at(m) * input_qoffset;
            acc.at(m) += k_offset;

            if(has_biases)
            {
                acc.at(m) += *(reinterpret_cast<int32_t *>(biases_it.ptr() + m * sizeof(int32_t)));
            }

            const int32_t out_mul   = output_multiplier.at(id.x() * depth_multiplier + m);
            const int32_t out_shift = output_shift.at(id.x() * depth_multiplier + m);
            if(out_shift < 0)
            {
                acc.at(m) = saturating_doubling_high_mul(acc.at(m) * (1 << (-out_shift)), out_mul) + output_qoffset;
            }
            else
            {
                acc.at(m) = rounding_divide_by_exp2(saturating_doubling_high_mul(acc.at(m), out_mul), out_shift) + output_qoffset;
            }
            *(reinterpret_cast<T *>(output_it.ptr() + m * sizeof(T))) = static_cast<T>(utility::clamp<AccType, T>(acc.at(m)));
        }
    },
    input_it, weights_it, biases_it, output_it);
}

template <typename T, typename TW>
void depthwise_loop_pow2_quantized_per_tensor(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info,
                                              const Size2D &dilation, unsigned int depth_multiplier, std::vector<int> output_multiplier, std::vector<int> output_shift, const Window &window, bool has_biases)
{
    constexpr int half_vec = vector_size / 2;

    using AccType          = int32_t;
    using AccVectorType    = typename wrapper::traits::neon_vector<AccType, half_vec>::type;
    using AccVectorTagType = typename wrapper::traits::neon_vector<AccType, half_vec>::tag_type;
    using TagType          = typename wrapper::traits::neon_vector<T, vector_size>::tag_type;

    const auto run_info = DepthwiseConvolutionRunInfo(*input->info(), *weights->info(), conv_info, window, depth_multiplier);

    const auto input_qoffset_vec   = wrapper::vreinterpret(wrapper::vmovl(wrapper::vdup_n(static_cast<T>(input->info()->quantization_info().uniform().offset), TagType{})));
    const auto weights_qoffset_vec = wrapper::vreinterpret(wrapper::vmovl(wrapper::vdup_n(static_cast<TW>(weights->info()->quantization_info().uniform().offset), TagType{})));
    const auto output_qoffset_vec  = wrapper::vdup_n(output->info()->quantization_info().uniform().offset, arm_compute::wrapper::traits::vector_128_tag{});

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

    Iterator input_it(input, win_input);
    Iterator weights_it(weights, win_weights);
    Iterator output_it(output, win_output);
    Iterator biases_it{};

    if(has_biases)
    {
        biases_it = Iterator(biases, win_weights);
    }

    std::vector<AccVectorType> acc0(depth_multiplier / vector_size);
    std::vector<AccVectorType> acc1(depth_multiplier / vector_size);

    execute_window_loop(execution_window, [&](const Coordinates & id)
    {
        std::fill(begin(acc0), end(acc0), zero);
        std::fill(begin(acc1), end(acc1), zero);

        const int32_t input_y      = id.y() * run_info.conv_stride_x - run_info.conv_pad_left;
        const int32_t input_z      = id.z() * run_info.conv_stride_y - run_info.conv_pad_top;
        int64_t       input_offset = input_y * run_info.input_stride_y + input_z * run_info.input_stride_z;

        auto weights_ptr = weights_it.ptr();
        for(size_t h = 0; h < run_info.weights_height; ++h)
        {
            const int32_t current_h = input_z + h * dilation.y();
            if(current_h >= 0 && current_h < static_cast<int32_t>(run_info.input_height))
            {
                int offs = input_offset;
                for(size_t w = 0; w < run_info.weights_width; ++w)
                {
                    const int32_t current_w = input_y + w * dilation.x();
                    if(current_w >= 0 && current_w < static_cast<int32_t>(run_info.input_width))
                    {
                        const auto input_8x8     = wrapper::vdup_n(*(reinterpret_cast<T *>(input_it.ptr() + std::min(static_cast<size_t>(offs), run_info.input_max_offset))), TagType{});
                        const auto input_s16x8   = wrapper::vreinterpret(wrapper::vmovl(input_8x8));
                        const auto input_no_offs = wrapper::vsub(input_s16x8, input_qoffset_vec);

                        for(size_t m = 0, i = 0; m < depth_multiplier; m += vector_size, ++i)
                        {
                            const auto weights_8x8     = wrapper::vload(reinterpret_cast<TW *>(weights_ptr + m * sizeof(T) + w * run_info.weights_stride_y));
                            const auto weights_s16x8   = wrapper::vreinterpret(wrapper::vmovl(weights_8x8));
                            const auto weights_no_offs = wrapper::vsub(weights_s16x8, weights_qoffset_vec);

                            acc0.at(i) = wrapper::vmlal(acc0.at(i), wrapper::vgetlow(input_no_offs), wrapper::vgetlow(weights_no_offs));
                            acc1.at(i) = wrapper::vmlal(acc1.at(i), wrapper::vgethigh(input_no_offs), wrapper::vgethigh(weights_no_offs));
                        }
                    }

                    offs += dilation.x() * run_info.input_stride_y;
                }
            }

            weights_ptr += run_info.weights_stride_z;
            input_offset += dilation.y() * run_info.input_stride_z;
        }

        for(size_t m = 0, i = 0; m < depth_multiplier; m += vector_size, ++i)
        {
            if(has_biases)
            {
                const auto bias_val0 = wrapper::vloadq(reinterpret_cast<int32_t *>(biases_it.ptr() + m * sizeof(int32_t)));
                const auto bias_val1 = wrapper::vloadq(reinterpret_cast<int32_t *>(biases_it.ptr() + (m + half_vec) * sizeof(int32_t)));

                acc0.at(i) = wrapper::vadd(acc0.at(i), bias_val0);
                acc1.at(i) = wrapper::vadd(acc1.at(i), bias_val1);
            }

            if(out_shift < 0)
            {
                acc0.at(i) = wrapper::vadd(saturating_doubling_high_mul(acc0.at(i) * (1 << (-out_shift)), out_mul), output_qoffset_vec);
                acc1.at(i) = wrapper::vadd(saturating_doubling_high_mul(acc1.at(i) * (1 << (-out_shift)), out_mul), output_qoffset_vec);
            }
            else
            {
                acc0.at(i) = wrapper::vadd(rounding_divide_by_exp2(saturating_doubling_high_mul(acc0.at(i), out_mul), out_shift), output_qoffset_vec);
                acc1.at(i) = wrapper::vadd(rounding_divide_by_exp2(saturating_doubling_high_mul(acc1.at(i), out_mul), out_shift), output_qoffset_vec);
            }

            acc0.at(i) = wrapper::vmin(wrapper::vmax(acc0.at(i), lower), upper);
            acc1.at(i) = wrapper::vmin(wrapper::vmax(acc1.at(i), lower), upper);

            const auto out_val = wrapper::vcombine(wrapper::vmovn(acc0.at(i)),
                                                   wrapper::vmovn(acc1.at(i)));

            if(std::is_same<T, uint8_t>::value)
            {
                wrapper::vstore(reinterpret_cast<uint8_t *>(output_it.ptr() + m * sizeof(uint8_t)), wrapper::vqmovn(vreinterpretq_u16_s16(out_val)));
            }
            else
            {
                wrapper::vstore(reinterpret_cast<int8_t *>(output_it.ptr() + m * sizeof(int8_t)), wrapper::vqmovn(out_val));
            }
        }
    },
    input_it, weights_it, biases_it, output_it);
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier,
                          const Size2D &dilation)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_layout() == DataLayout::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(depth_multiplier == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(1) + (weights->dimension(1) - 1) * (dilation.x() - 1) > input->dimension(1) + conv_info.pad_left() + conv_info.pad_right());
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(2) + (weights->dimension(2) - 1) * (dilation.y() - 1) > input->dimension(2) + conv_info.pad_top() + conv_info.pad_bottom());
    ARM_COMPUTE_RETURN_ERROR_ON((input->dimension(0) * depth_multiplier) != weights->dimension(0));
    ARM_COMPUTE_RETURN_ERROR_ON((dilation.x() < 1) || (dilation.y() < 1));
    ARM_COMPUTE_RETURN_ERROR_ON((conv_info.stride().first < 1) || (conv_info.stride().second < 1));

    if(is_data_type_quantized_per_channel(weights->data_type()))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::QSYMM8_PER_CHANNEL);
        ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(0) != weights->quantization_info().scale().size());
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    }

    if(biases != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != weights->dimension(0));

        if(is_data_type_quantized_asymmetric(input->data_type()))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::S32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, biases);
        }
    }

    if(output->total_size() != 0)
    {
        const TensorShape output_shape = misc::shape_calculator::compute_depthwise_convolution_shape(*input, *weights, conv_info, depth_multiplier, dilation);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}
} // namespace

NEDepthwiseConvolutionLayerNativeKernel::NEDepthwiseConvolutionLayerNativeKernel()
    : _func(), _input(), _weights(), _biases(), _output(), _conv_info(), _depth_multiplier(1), _dilation(), _output_multiplier(), _output_shift(), _has_biases()
{
}

void NEDepthwiseConvolutionLayerNativeKernel::configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output,
                                                        const PadStrideInfo &conv_info, unsigned int depth_multiplier, const Size2D &dilation)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), weights->info(), (biases != nullptr) ? biases->info() : nullptr, output->info(), conv_info, depth_multiplier, dilation));

    _input            = input;
    _weights          = weights;
    _biases           = biases;
    _output           = output;
    _conv_info        = conv_info;
    _depth_multiplier = depth_multiplier;
    _dilation         = dilation;
    _has_biases       = (biases != nullptr);

    if(is_data_type_quantized(_input->info()->data_type()))
    {
        const auto input_scale  = input->info()->quantization_info().uniform().scale;
        const auto output_scale = output->info()->quantization_info().uniform().scale;

        auto weights_scale = weights->info()->quantization_info().scale();
        if(!is_data_type_quantized_per_channel(_weights->info()->data_type()))
        {
            for(size_t i = 1; i < _weights->info()->dimension(channel_idx); ++i)
            {
                weights_scale.push_back(weights_scale.front());
            }
        }

        for(const auto &s : weights_scale)
        {
            int32_t     out_mult   = 0;
            int32_t     out_shift  = 0;
            const float multiplier = input_scale * s / output_scale;
            arm_compute::quantization::calculate_quantized_multiplier(multiplier, &out_mult, &out_shift);

            _output_multiplier.push_back(out_mult);
            _output_shift.push_back(out_shift);
        }
    }

    switch(_weights->info()->data_type())
    {
        case DataType::QASYMM8:
            _func = &NEDepthwiseConvolutionLayerNativeKernel::run_depthwise<uint8_t, uint8_t>;
            break;
        case DataType::QASYMM8_SIGNED:
            _func = &NEDepthwiseConvolutionLayerNativeKernel::run_depthwise<int8_t, int8_t>;
            break;
        case DataType::QSYMM8_PER_CHANNEL:
            if(_input->info()->data_type() == DataType::QASYMM8)
            {
                _func = &NEDepthwiseConvolutionLayerNativeKernel::run_depthwise<uint8_t, int8_t>;
            }
            else
            {
                _func = &NEDepthwiseConvolutionLayerNativeKernel::run_depthwise<int8_t, int8_t>;
            }
            break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            _func = &NEDepthwiseConvolutionLayerNativeKernel::run_depthwise<float16_t, float16_t>;
            break;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F32:
            _func = &NEDepthwiseConvolutionLayerNativeKernel::run_depthwise<float, float>;
            break;
        default:
            ARM_COMPUTE_ERROR("Data type not supported");
            break;
    }

    const TensorShape output_shape = misc::shape_calculator::compute_depthwise_convolution_shape(*input->info(), *weights->info(), conv_info, depth_multiplier, dilation);
    auto_init_if_empty(*output->info(), input->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(output_shape).set_quantization_info(output->info()->quantization_info()));

    Window win = calculate_max_window(*output->info(), Steps());
    INEKernel::configure(win);
}

Status NEDepthwiseConvolutionLayerNativeKernel::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                                         unsigned int  depth_multiplier,
                                                         const Size2D &dilation)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, weights, biases, output, conv_info, depth_multiplier, dilation));
    return Status{};
}

void NEDepthwiseConvolutionLayerNativeKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    (this->*_func)(window, _has_biases);
}

template <typename T, typename TW, NEDepthwiseConvolutionLayerNativeKernel::FloatEnalber<T>>
void NEDepthwiseConvolutionLayerNativeKernel::run_depthwise(const Window &window, bool has_biases)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    if(_depth_multiplier == 1)
    {
        depthwise_loop_multiplier1_fp<T>(_input, _weights, _biases, _output, _conv_info, _dilation, window, has_biases);
    }
    else
    {
        depthwise_loop_generic_fp<T>(_input, _weights, _biases, _output, _conv_info, _dilation, _depth_multiplier, window, has_biases);
    }
}

template <typename T, typename TW, NEDepthwiseConvolutionLayerNativeKernel::Quantized8bitEnalber<T>>
void NEDepthwiseConvolutionLayerNativeKernel::run_depthwise(const Window &window, bool has_biases)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    if(_depth_multiplier == 1)
    {
        depthwise_loop_multiplier1_quantized<T, TW>(_input, _weights, _biases, _output, _conv_info, _dilation, _output_multiplier, _output_shift, window, has_biases);
    }
    else
    {
        const bool is_pow2                 = ((_depth_multiplier & (_depth_multiplier - 1)) == 0);
        const bool is_quantized_per_tensor = !(is_data_type_quantized_per_channel(_weights->info()->data_type()));

        if(is_pow2 && is_quantized_per_tensor && _depth_multiplier >= 8)
        {
            depthwise_loop_pow2_quantized_per_tensor<T, TW>(_input, _weights, _biases, _output, _conv_info, _dilation, _depth_multiplier, _output_multiplier, _output_shift, window, has_biases);
        }
        else
        {
            depthwise_loop_generic_quantized<T, TW>(_input, _weights, _biases, _output, _conv_info, _dilation, _depth_multiplier, _output_multiplier, _output_shift, window, has_biases);
        }
    }
}
} // namespace arm_compute
