/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#ifndef SRC_CORE_KERNELS_DEPTWISECONV2DNATIVE_IMPL_H
#define SRC_CORE_KERNELS_DEPTWISECONV2DNATIVE_IMPL_H
#include "arm_compute/core/Helpers.h"
#include "src/core/NEON/wrapper/wrapper.h"

namespace arm_compute
{
struct ConvolutionInfo;

namespace cpu
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

    DepthwiseConvolutionRunInfo(const ITensorInfo &input, const ITensorInfo &weights, const PadStrideInfo &conv_info, const Window &w, uint32_t depth_multiplier = 1) // NOLINT
        : num_read_elements_per_iteration((depth_multiplier == 1 ? (vector_size / element_size_from_data_type(input.data_type())) : 1)),
          x_start(w.x().start()),
          x_end(w.x().end()),
          x_step(static_cast<uint32_t>(num_read_elements_per_iteration * depth_multiplier)),
          x_leftover_start(std::max(static_cast<int32_t>(w.x().end() + 1) - static_cast<int32_t>(x_step), int32_t(0))),
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
void depthwise_loop_multiplier1_fp(const ITensor *src, const ITensor *weights, const ITensor *biases, ITensor *dst, const PadStrideInfo &conv_info,
                                   const Size2D &dilation, const Window &window, bool has_biases)
{
    constexpr auto element_per_vector = vector_size / sizeof(T);
    using VectorType                  = typename wrapper::traits::neon_vector<T, element_per_vector>::type;
    using TagType                     = typename wrapper::traits::neon_vector<T, element_per_vector>::tag_type;

    const auto run_info = DepthwiseConvolutionRunInfo(*src->info(), *weights->info(), conv_info, window);

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

    Iterator input_it(src, win_input);
    Iterator weights_it(weights, win_weights);
    Iterator output_it(dst, win_output);
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
void depthwise_loop_generic_fp(const ITensor *src, const ITensor *weights, const ITensor *biases, ITensor *dst, const PadStrideInfo &conv_info,
                               const Size2D &dilation, unsigned int depth_multiplier, const Window &window, bool has_biases)
{
    const auto run_info = DepthwiseConvolutionRunInfo(*src->info(), *weights->info(), conv_info, window, depth_multiplier);

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

    Iterator input_it(src, win_input);
    Iterator weights_it(weights, win_weights);
    Iterator output_it(dst, win_output);
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
void run_depthwise_float(const ITensor *src, const ITensor *weights, const ITensor *biases,
                         ITensor *dst, const Window &window, bool has_biases, const ConvolutionInfo &info)
{
    PadStrideInfo conv_info        = info.pad_stride_info;
    unsigned int  depth_multiplier = info.depth_multiplier;
    Size2D        dilation         = info.dilation;

    if(depth_multiplier == 1)
    {
        depthwise_loop_multiplier1_fp<T>(src, weights, biases, dst, conv_info, dilation, window, has_biases);
    }
    else
    {
        depthwise_loop_generic_fp<T>(src, weights, biases, dst, conv_info, dilation, depth_multiplier, window, has_biases);
    }
}

template <typename T, typename TW>
void run_depthwise_quanitized8bit(const ITensor *src, const ITensor *weights, const ITensor *biases,
                                  ITensor *dst, const Window &window, bool has_biases, const ConvolutionInfo &info);

} // namespace cpu
} // namespace arm_compute
#endif //define SRC_CORE_KERNELS_DEPTWISECONV2DNATIVE_IMPL_H
