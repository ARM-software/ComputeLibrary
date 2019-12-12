/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEDepthwiseConvolutionLayerNativeKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/NEON/wrapper/traits.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "src/core/NEON/kernels/convolution/depthwise/impl_qa8_qa8.hpp"

namespace arm_compute
{
namespace
{
void pad_vectors(std::vector<int> &mult, std::vector<int> &shift, int vec_size)
{
    ARM_COMPUTE_ERROR_ON(mult.size() != shift.size());
    while(mult.size() % vec_size != 0)
    {
        mult.push_back(0);
        shift.push_back(0);
    }
}

template <typename T, int S, bool has_biases>
void depthwise_loop_multiplier1_fp(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info,
                                   const Size2D &dilation, const Window &window)
{
    using VectorType = typename wrapper::traits::neon_vector<T, S>::type;
    using TagType    = typename wrapper::traits::neon_vector<T, S>::tag_type;

    const size_t input_stride_y   = input->info()->strides_in_bytes().y();
    const size_t input_stride_z   = input->info()->strides_in_bytes().z();
    const size_t input_max_offset = input->info()->strides_in_bytes().z() * input->info()->dimension(2) - (input->info()->padding().bottom + input->info()->padding().top) *
                                    input->info()->strides_in_bytes().y();
    const size_t weights_width    = weights->info()->dimension(1);
    const size_t weights_height   = weights->info()->dimension(2);
    const size_t weights_stride_y = weights->info()->strides_in_bytes().y();
    const size_t weights_stride_z = weights->info()->strides_in_bytes().z();
    const size_t conv_stride_x    = conv_info.stride().first;
    const size_t conv_stride_y    = conv_info.stride().second;
    const size_t conv_pad_left    = conv_info.pad_left();
    const size_t conv_pad_top     = conv_info.pad_top();

    Window win_input = window;
    win_input.set(Window::DimY, Window::Dimension(0, 0, 0));
    win_input.set(Window::DimZ, Window::Dimension(0, 0, 0));

    Window win_weights = win_input;
    win_weights.set(3, Window::Dimension(0, 0, 0));

    Iterator input_it(input, win_input);
    Iterator weights_it(weights, win_weights);
    Iterator output_it(output, window);
    Iterator biases_it{};

    if(has_biases)
    {
        biases_it = Iterator(biases, win_weights);
    }

    execute_window_loop(window, [&](const Coordinates & id)
    {
        VectorType acc = wrapper::vdup_n(static_cast<T>(0), TagType{});

        const int input_y      = id.y() * conv_stride_x - conv_pad_left;
        const int input_z      = id.z() * conv_stride_y - conv_pad_top;
        int       input_offset = input_y * input_stride_y + input_z * input_stride_z;

        auto weights_ptr = weights_it.ptr();
        for(size_t h = 0; h < weights_height; ++h)
        {
            int offs = input_offset;
            for(size_t w = 0; w < weights_width; ++w)
            {
                const auto input_vals   = wrapper::vload(reinterpret_cast<T *>(input_it.ptr() + std::min(static_cast<size_t>(offs), input_max_offset)));
                const auto weights_vals = wrapper::vload(reinterpret_cast<T *>(weights_ptr + w * weights_stride_y));

                acc = wrapper::vmla(acc, weights_vals, input_vals);
                offs += dilation.x() * input_stride_y;
            }

            weights_ptr += weights_stride_z;
            input_offset += dilation.y() * input_stride_z;
        }

        if(has_biases)
        {
            const auto biases_vals = wrapper::vload(reinterpret_cast<T *>(biases_it.ptr()));
            acc                    = wrapper::vadd(acc, biases_vals);
        }

        wrapper::vstore(reinterpret_cast<T *>(output_it.ptr()), acc);
    },
    input_it, weights_it, biases_it, output_it);
}

template <typename T, bool has_biases>
void depthwise_loop_generic_fp(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info,
                               const Size2D &dilation, unsigned int depth_multiplier, const Window &window)
{
    const size_t input_stride_y   = input->info()->strides_in_bytes().y();
    const size_t input_stride_z   = input->info()->strides_in_bytes().z();
    const size_t input_max_offset = input->info()->strides_in_bytes().z() * input->info()->dimension(2) - (input->info()->padding().bottom + input->info()->padding().top) *
                                    input->info()->strides_in_bytes().y();
    const size_t weights_width    = weights->info()->dimension(1);
    const size_t weights_height   = weights->info()->dimension(2);
    const size_t weights_stride_y = weights->info()->strides_in_bytes().y();
    const size_t weights_stride_z = weights->info()->strides_in_bytes().z();
    const size_t conv_stride_x    = conv_info.stride().first;
    const size_t conv_stride_y    = conv_info.stride().second;
    const size_t conv_pad_left    = conv_info.pad_left();
    const size_t conv_pad_top     = conv_info.pad_top();

    Window win_input = window;
    win_input.set(Window::DimY, Window::Dimension(0, 0, 0));
    win_input.set(Window::DimZ, Window::Dimension(0, 0, 0));

    Window win_weights = win_input;
    win_weights.set(3, Window::Dimension(0, 0, 0));

    win_input.set_dimension_step(Window::DimX, 1);

    Iterator input_it(input, win_input);
    Iterator weights_it(weights, win_weights);
    Iterator output_it(output, window);
    Iterator biases_it{};

    if(has_biases)
    {
        biases_it = Iterator(biases, win_weights);
    }

    execute_window_loop(window, [&](const Coordinates & id)
    {
        std::vector<T> acc(depth_multiplier, static_cast<T>(0));

        const int input_y      = id.y() * conv_stride_x - conv_pad_left;
        const int input_z      = id.z() * conv_stride_y - conv_pad_top;
        int       input_offset = input_y * input_stride_y + input_z * input_stride_z;

        auto weights_ptr = weights_it.ptr();
        for(size_t h = 0; h < weights_height; ++h)
        {
            int offs = input_offset;
            for(size_t w = 0; w < weights_width; ++w)
            {
                const auto input_val = *(reinterpret_cast<T *>(input_it.ptr() + std::min(static_cast<size_t>(offs), input_max_offset)));

                for(size_t m = 0; m < depth_multiplier; ++m)
                {
                    const auto weights_val = *(reinterpret_cast<T *>(weights_ptr + m * sizeof(T) + w * weights_stride_y));
                    acc.at(m)              = support::cpp11::fma(weights_val, input_val, acc.at(m));
                }

                offs += dilation.x() * input_stride_y;
            }

            weights_ptr += weights_stride_z;
            input_offset += dilation.y() * input_stride_z;
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

template <typename T, typename TW, int S, bool has_biases, bool is_per_channel>
void depthwise_loop_multiplier1_quantized(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info,
                                          const Size2D &dilation, std::vector<int> output_multiplier, std::vector<int> output_shift, const Window &window)
{
    using VectorType = typename wrapper::traits::neon_vector<T, S>::type;
    using TagType    = typename wrapper::traits::neon_vector<T, S>::tag_type;

    const size_t input_stride_y   = input->info()->strides_in_bytes().y();
    const size_t input_stride_z   = input->info()->strides_in_bytes().z();
    const size_t input_max_offset = input->info()->strides_in_bytes().z() * input->info()->dimension(2) - (input->info()->padding().bottom + input->info()->padding().top) *
                                    input->info()->strides_in_bytes().y();
    const size_t weights_width    = weights->info()->dimension(1);
    const size_t weights_height   = weights->info()->dimension(2);
    const size_t weights_stride_y = weights->info()->strides_in_bytes().y();
    const size_t weights_stride_z = weights->info()->strides_in_bytes().z();
    const size_t conv_stride_x    = conv_info.stride().first;
    const size_t conv_stride_y    = conv_info.stride().second;
    const size_t conv_pad_left    = conv_info.pad_left();
    const size_t conv_pad_top     = conv_info.pad_top();

    const int32_t input_qoffset   = input->info()->quantization_info().uniform().offset;
    const int32_t weights_qoffset = weights->info()->quantization_info().uniform().offset;
    const int32_t output_qoffset  = output->info()->quantization_info().uniform().offset;
    const int32_t k_offset        = weights_width * weights_height * input_qoffset * weights_qoffset;

    Window win_input = window;
    win_input.set(Window::DimY, Window::Dimension(0, 0, 0));
    win_input.set(Window::DimZ, Window::Dimension(0, 0, 0));

    Window win_weights = win_input;
    win_weights.set(3, Window::Dimension(0, 0, 0));

    Iterator input_it(input, win_input);
    Iterator weights_it(weights, win_weights);
    Iterator output_it(output, window);
    Iterator biases_it{};

    if(has_biases)
    {
        biases_it = Iterator(biases, win_weights);
    }

    execute_window_loop(window, [&](const Coordinates & id)
    {
        std::vector<int32_t> acc(S, 0);
        std::vector<int32_t> in_sum(S, 0);
        std::vector<int32_t> we_sum(S, 0);

        const int input_y      = id.y() * conv_stride_x - conv_pad_left;
        const int input_z      = id.z() * conv_stride_y - conv_pad_top;
        int       input_offset = input_y * input_stride_y + input_z * input_stride_z;

        auto weights_ptr = weights_it.ptr();
        for(size_t h = 0; h < weights_height; ++h)
        {
            int offs = input_offset;
            for(size_t w = 0; w < weights_width; ++w)
            {
                const auto input_vals   = wrapper::vload(reinterpret_cast<T *>(input_it.ptr() + std::min(static_cast<size_t>(offs), input_max_offset)));
                const auto weights_vals = wrapper::vload(reinterpret_cast<TW *>(weights_ptr + w * weights_stride_y));

                for(int i = 0; i < S; ++i)
                {
                    acc.at(i) += input_vals[i] * weights_vals[i];
                    in_sum.at(i) += input_vals[i];
                    we_sum.at(i) += weights_vals[i];
                }

                offs += dilation.x() * input_stride_y;
            }

            weights_ptr += weights_stride_z;
            input_offset += dilation.y() * input_stride_z;
        }

        VectorType out_vals = wrapper::vdup_n(static_cast<T>(0), TagType{});
        for(int i = 0; i < S; ++i)
        {
            acc.at(i) -= in_sum.at(i) * weights_qoffset;
            acc.at(i) -= we_sum.at(i) * input_qoffset;
            acc.at(i) += k_offset;

            if(has_biases)
            {
                acc.at(i) += *reinterpret_cast<int32_t *>(biases_it.ptr() + i * sizeof(int32_t));
            }

            acc.at(i)   = rounding_divide_by_exp2(saturating_doubling_high_mul(acc.at(i), output_multiplier.at(id.x() + i)), output_shift.at(id.x() + i)) + output_qoffset;
            out_vals[i] = static_cast<T>(utility::clamp<int32_t, uint8_t>(acc.at(i)));
        }

        wrapper::vstore(reinterpret_cast<T *>(output_it.ptr()), out_vals);
    },
    input_it, weights_it, biases_it, output_it);
}

template <typename T, typename TW, bool has_biases, bool is_per_channel>
void depthwise_loop_generic_quantized(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info,
                                      const Size2D &dilation, unsigned int depth_multiplier, std::vector<int> output_multiplier, std::vector<int> output_shift, const Window &window)
{
    const size_t input_stride_y   = input->info()->strides_in_bytes().y();
    const size_t input_stride_z   = input->info()->strides_in_bytes().z();
    const size_t input_max_offset = input->info()->strides_in_bytes().z() * input->info()->dimension(2) - (input->info()->padding().bottom + input->info()->padding().top) *
                                    input->info()->strides_in_bytes().y();
    const size_t weights_width    = weights->info()->dimension(1);
    const size_t weights_height   = weights->info()->dimension(2);
    const size_t weights_stride_y = weights->info()->strides_in_bytes().y();
    const size_t weights_stride_z = weights->info()->strides_in_bytes().z();
    const size_t conv_stride_x    = conv_info.stride().first;
    const size_t conv_stride_y    = conv_info.stride().second;
    const size_t conv_pad_left    = conv_info.pad_left();
    const size_t conv_pad_top     = conv_info.pad_top();

    const int32_t input_qoffset   = input->info()->quantization_info().uniform().offset;
    const int32_t weights_qoffset = weights->info()->quantization_info().uniform().offset;
    const int32_t output_qoffset  = output->info()->quantization_info().uniform().offset;
    const int32_t k_offset        = weights_width * weights_height * input_qoffset * weights_qoffset;

    Window win_input = window;
    win_input.set(Window::DimY, Window::Dimension(0, 0, 0));
    win_input.set(Window::DimZ, Window::Dimension(0, 0, 0));

    Window win_weights = win_input;
    win_weights.set(3, Window::Dimension(0, 0, 0));

    win_input.set_dimension_step(Window::DimX, 1);

    Iterator input_it(input, win_input);
    Iterator weights_it(weights, win_weights);
    Iterator output_it(output, window);
    Iterator biases_it{};

    if(has_biases)
    {
        biases_it = Iterator(biases, win_weights);
    }

    execute_window_loop(window, [&](const Coordinates & id)
    {
        std::vector<int32_t> acc(depth_multiplier, 0);
        std::vector<int32_t> we_sum(depth_multiplier, 0);
        int32_t              in_sum = 0;

        const int input_y      = id.y() * conv_stride_x - conv_pad_left;
        const int input_z      = id.z() * conv_stride_y - conv_pad_top;
        int       input_offset = input_y * input_stride_y + input_z * input_stride_z;

        auto weights_ptr = weights_it.ptr();
        for(size_t h = 0; h < weights_height; ++h)
        {
            int offs = input_offset;
            for(size_t w = 0; w < weights_width; ++w)
            {
                const auto input_val = *(reinterpret_cast<T *>(input_it.ptr() + std::min(static_cast<size_t>(offs), input_max_offset)));

                for(size_t m = 0; m < depth_multiplier; ++m)
                {
                    const auto weights_val = *(reinterpret_cast<TW *>(weights_ptr + m * sizeof(T) + w * weights_stride_y));
                    acc.at(m) += input_val * weights_val;

                    we_sum.at(m) += weights_val;
                }

                offs += dilation.x() * input_stride_y;
                in_sum += input_val;
            }

            weights_ptr += weights_stride_z;
            input_offset += dilation.y() * input_stride_z;
        }

        for(size_t m = 0; m < depth_multiplier; ++m)
        {
            acc.at(m) -= in_sum * weights_qoffset;
            acc.at(m) -= we_sum.at(m) * input_qoffset;
            acc.at(m) += k_offset;

            if(has_biases)
            {
                const auto biases_val = *(reinterpret_cast<int32_t *>(biases_it.ptr() + m * sizeof(int32_t)));

                int32_t out_val = acc.at(m) + biases_val;
                out_val         = rounding_divide_by_exp2(saturating_doubling_high_mul(out_val, output_multiplier.at(id.x() + m)),
                                                          output_shift.at(id.x() + m))
                                  + output_qoffset;
                *(reinterpret_cast<T *>(output_it.ptr() + m * sizeof(T))) = static_cast<T>(utility::clamp<int32_t, uint8_t>(out_val));
            }
            else
            {
                int32_t out_val = rounding_divide_by_exp2(saturating_doubling_high_mul(acc.at(m), output_multiplier.at(id.x() + m)),
                                                          output_shift.at(id.x() + m))
                                  + output_qoffset;
                *(reinterpret_cast<T *>(output_it.ptr() + m * sizeof(T))) = static_cast<T>(utility::clamp<int32_t, uint8_t>(out_val));
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
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(depth_multiplier == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(1) + (weights->dimension(1) - 1) * (dilation.x() - 1) > input->dimension(1) + conv_info.pad_left() + conv_info.pad_right());
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(2) + (weights->dimension(2) - 1) * (dilation.y() - 1) > input->dimension(2) + conv_info.pad_top() + conv_info.pad_bottom());
    ARM_COMPUTE_RETURN_ERROR_ON((input->dimension(0) * depth_multiplier) != weights->dimension(0));
    ARM_COMPUTE_RETURN_ERROR_ON((dilation.x() < 1) || (dilation.y() < 1));
    ARM_COMPUTE_RETURN_ERROR_ON((conv_info.stride().first < 1) || (conv_info.stride().second < 1));

    if(is_data_type_quantized_per_channel(weights->data_type()))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::QSYMM8_PER_CHANNEL);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(0) != weights->quantization_info().scale().size());
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights, output);
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
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *weights, ITensorInfo *biases,
                                                        ITensorInfo *output, const PadStrideInfo &conv_info,
                                                        unsigned int depth_multiplier, const Size2D &dilation)
{
    // Get convolved dimensions
    const TensorShape output_shape = misc::shape_calculator::compute_depthwise_convolution_shape(*input, *weights, conv_info, depth_multiplier, dilation);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output, input->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(output_shape).set_quantization_info(output->quantization_info()));

    // Configure kernel window (generic)
    const unsigned int num_elems_read_per_iteration    = (depth_multiplier == 1) ? 8 / element_size_from_data_type(input->data_type()) : 1;
    const unsigned int num_elems_written_per_iteration = num_elems_read_per_iteration * depth_multiplier;

    // Configure kernel window
    Window win = calculate_max_window(*output, Steps(num_elems_written_per_iteration));

    AccessWindowStatic input_access(input, 0, -conv_info.pad_left(), ceil_to_multiple(num_elems_read_per_iteration, input->dimension(0)),
                                    input->dimension(1) + std::max(std::max(conv_info.pad_right(), conv_info.pad_bottom()), conv_info.pad_top()));
    AccessWindowHorizontal weights_access(weights, 0, num_elems_written_per_iteration);
    AccessWindowHorizontal output_access(output, 0, num_elems_written_per_iteration);

    bool window_changed = update_window_and_padding(win, input_access, weights_access, output_access);

    if(biases != nullptr)
    {
        AccessWindowHorizontal biases_access(biases, 0, num_elems_written_per_iteration);
        window_changed |= update_window_and_padding(win, biases_access);
    }

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

NEDepthwiseConvolutionLayerNativeKernel::NEDepthwiseConvolutionLayerNativeKernel()
    : _func(), _border_size(0), _input(), _weights(), _biases(), _output(), _conv_info(), _depth_multiplier(1), _dilation(), _output_multiplier(), _output_shift()
{
}

BorderSize NEDepthwiseConvolutionLayerNativeKernel::border_size() const
{
    return _border_size;
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
    _border_size      = BorderSize(_conv_info.pad_left(), 0, std::max(std::max(conv_info.pad_right(), conv_info.pad_bottom()), conv_info.pad_top()), 0);
    _dilation         = dilation;

    if(is_data_type_quantized(_input->info()->data_type()))
    {
        const auto input_scale  = input->info()->quantization_info().uniform().scale;
        const auto output_scale = output->info()->quantization_info().uniform().scale;

        auto weights_scale = weights->info()->quantization_info().scale();
        if(!is_data_type_quantized_per_channel(_weights->info()->data_type()))
        {
            for(size_t i = 1; i < _weights->info()->dimension(0); ++i)
            {
                weights_scale.push_back(weights_scale.front());
            }
        }

        for(size_t i = 0; i < weights_scale.size(); ++i)
        {
            int32_t     out_mult   = 0;
            int32_t     out_shift  = 0;
            const float multiplier = input_scale * weights_scale.at(i) / output_scale;
            ARM_COMPUTE_ERROR_ON(multiplier > 1.f);
            arm_compute::quantization::calculate_quantized_multiplier_less_than_one(multiplier, &out_mult, &out_shift);

            _output_multiplier.push_back(out_mult);
            _output_shift.push_back(out_shift);
        }
    }

    switch(_weights->info()->data_type())
    {
        case DataType::QASYMM8:
            _func = (biases != nullptr) ? &NEDepthwiseConvolutionLayerNativeKernel::run_depthwise<uint8_t, uint8_t, 8, true, false> :
                    &NEDepthwiseConvolutionLayerNativeKernel::run_depthwise<uint8_t, uint8_t, 8, false, false>;
            pad_vectors(_output_multiplier, _output_shift, 8);
            break;
        case DataType::QSYMM8_PER_CHANNEL:
            _func = (biases != nullptr) ? &NEDepthwiseConvolutionLayerNativeKernel::run_depthwise<uint8_t, int8_t, 8, true, true> :
                    &NEDepthwiseConvolutionLayerNativeKernel::run_depthwise<uint8_t, int8_t, 8, false, true>;
            pad_vectors(_output_multiplier, _output_shift, 8);
            break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            _func = (biases != nullptr) ? &NEDepthwiseConvolutionLayerNativeKernel::run_depthwise<float16_t, float16_t, 4, true, false> :
                    &NEDepthwiseConvolutionLayerNativeKernel::run_depthwise<float16_t, float16_t, 4, false, false>;
            pad_vectors(_output_multiplier, _output_shift, 4);
            break;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F32:
            _func = (biases != nullptr) ? &NEDepthwiseConvolutionLayerNativeKernel::run_depthwise<float, float, 2, true, false> :
                    &NEDepthwiseConvolutionLayerNativeKernel::run_depthwise<float, float, 2, false, false>;
            pad_vectors(_output_multiplier, _output_shift, 2);
            break;
        default:
            ARM_COMPUTE_ERROR("Data type not supported");
            break;
    }

    auto win_config = validate_and_configure_window(_input->info(), _weights->info(), (biases != nullptr) ? biases->info() : nullptr, _output->info(), _conv_info, _depth_multiplier, dilation);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NEDepthwiseConvolutionLayerNativeKernel::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                                         unsigned int  depth_multiplier,
                                                         const Size2D &dilation)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, weights, biases, output, conv_info, depth_multiplier, dilation));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), weights->clone().get(), (biases != nullptr) ? biases->clone().get() : nullptr, output->clone().get(), conv_info,
                                                              depth_multiplier, dilation)
                                .first);
    return Status{};
}

void NEDepthwiseConvolutionLayerNativeKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    (this->*_func)(window);
}

template < typename T, typename TW, int S, bool has_biases, bool is_per_channel, typename std::enable_if < std::is_same<T, float>::value
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                                                                                           || std::is_same<T, float16_t>::value
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                                                                                           ,
                                                                                                           int >::type >
void NEDepthwiseConvolutionLayerNativeKernel::run_depthwise(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    if(_depth_multiplier == 1)
    {
        depthwise_loop_multiplier1_fp<T, S, has_biases>(_input, _weights, _biases, _output, _conv_info, _dilation, window);
    }
    else
    {
        depthwise_loop_generic_fp<T, has_biases>(_input, _weights, _biases, _output, _conv_info, _dilation, _depth_multiplier, window);
    }
}

template <typename T, typename TW, int S, bool has_biases, bool is_per_channel, typename std::enable_if<std::is_same<T, uint8_t>::value, int>::type>
void NEDepthwiseConvolutionLayerNativeKernel::run_depthwise(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    if(_depth_multiplier == 1)
    {
        depthwise_loop_multiplier1_quantized<T, TW, S, has_biases, is_per_channel>(_input, _weights, _biases, _output, _conv_info, _dilation, _output_multiplier, _output_shift, window);
    }
    else
    {
        depthwise_loop_generic_quantized<T, TW, has_biases, is_per_channel>(_input, _weights, _biases, _output, _conv_info, _dilation, _depth_multiplier, _output_multiplier, _output_shift, window);
    }
}
} // namespace arm_compute
