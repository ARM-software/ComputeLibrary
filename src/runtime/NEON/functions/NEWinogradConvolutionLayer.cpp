/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEWinogradConvolutionLayer.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/NEON/kernels/NEWinogradConvolutionLayerKernel.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMAssemblyDispatch.h"
#include "support/ToolchainSupport.h"

#include "arm_compute/core/NEON/kernels/convolution/winograd/winograd.hpp"

namespace arm_compute
{
namespace
{
inline Status validate_kernel_3x3(const Size2D input_dims, const ITensorInfo *input, const TensorInfo *input0, const TensorInfo *input1, const TensorInfo *batched_mm_output,
                                  const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const WinogradInfo &winograd_info, const ActivationLayerInfo &act_info)
{
    if(input_dims.width > 4 && input_dims.height > 4)
    {
        ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformInputKernel<float, 4, 4, 3, 3>::validate(input, input0, winograd_info)));
        ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformWeightsKernel<float, 4, 4, 3, 3>::validate(weights, input1, winograd_info)));
        ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformOutputKernel<float, 4, 4, 3, 3>::validate(batched_mm_output, biases, output, winograd_info)));
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformInputKernel<float, 2, 2, 3, 3>::validate(input, input0, winograd_info)));
        ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformWeightsKernel<float, 2, 2, 3, 3>::validate(weights, input1, winograd_info)));
        ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformOutputKernel<float, 2, 2, 3, 3>::validate(batched_mm_output, biases, output, winograd_info)));
    }

    if(act_info.enabled())
    {
        NEActivationLayer::validate(output, nullptr, act_info);
    }
    return Status{};
}

inline Status validate_kernel_5x5(const ITensorInfo *input, const TensorInfo *input0, const TensorInfo *input1, const TensorInfo *batched_mm_output,
                                  const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const WinogradInfo &winograd_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformInputKernel<float, 2, 2, 5, 5>::validate(input, input0, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformWeightsKernel<float, 2, 2, 5, 5>::validate(weights, input1, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformOutputKernel<float, 2, 2, 5, 5>::validate(batched_mm_output, biases, output, winograd_info)));
    if(act_info.enabled())
    {
        NEActivationLayer::validate(output, nullptr, act_info);
    }
    return Status{};
}

inline Status validate_kernel_3x1(const ITensorInfo *input, const TensorInfo *input0, const TensorInfo *input1, const TensorInfo *batched_mm_output,
                                  const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const WinogradInfo &winograd_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformInputKernel<float, 1, 6, 1, 3>::validate(input, input0, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformWeightsKernel<float, 1, 6, 1, 3>::validate(weights, input1, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformOutputKernel<float, 1, 6, 1, 3>::validate(batched_mm_output, biases, output, winograd_info)));
    if(act_info.enabled())
    {
        NEActivationLayer::validate(output, nullptr, act_info);
    }
    return Status{};
}

inline Status validate_kernel_1x3(const ITensorInfo *input, const TensorInfo *input0, const TensorInfo *input1, const TensorInfo *batched_mm_output,
                                  const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const WinogradInfo &winograd_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformInputKernel<float, 6, 1, 3, 1>::validate(input, input0, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformWeightsKernel<float, 6, 1, 3, 1>::validate(weights, input1, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformOutputKernel<float, 6, 1, 3, 1>::validate(batched_mm_output, biases, output, winograd_info)));

    if(act_info.enabled())
    {
        NEActivationLayer::validate(output, nullptr, act_info);
    }
    return Status{};
}

inline Status validate_kernel_5x1(const ITensorInfo *input, const TensorInfo *input0, const TensorInfo *input1, const TensorInfo *batched_mm_output,
                                  const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const WinogradInfo &winograd_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformInputKernel<float, 1, 4, 1, 5>::validate(input, input0, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformWeightsKernel<float, 1, 4, 1, 5>::validate(weights, input1, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformOutputKernel<float, 1, 4, 1, 5>::validate(batched_mm_output, biases, output, winograd_info)));
    if(act_info.enabled())
    {
        NEActivationLayer::validate(output, nullptr, act_info);
    }
    return Status{};
}
inline Status validate_kernel_1x5(const ITensorInfo *input, const TensorInfo *input0, const TensorInfo *input1, const TensorInfo *batched_mm_output,
                                  const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const WinogradInfo &winograd_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformInputKernel<float, 4, 1, 5, 1>::validate(input, input0, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformWeightsKernel<float, 4, 1, 5, 1>::validate(weights, input1, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformOutputKernel<float, 4, 1, 5, 1>::validate(batched_mm_output, biases, output, winograd_info)));
    if(act_info.enabled())
    {
        NEActivationLayer::validate(output, nullptr, act_info);
    }
    return Status{};
}

inline Status validate_kernel_7x1(const ITensorInfo *input, const TensorInfo *input0, const TensorInfo *input1, const TensorInfo *batched_mm_output,
                                  const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const WinogradInfo &winograd_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformInputKernel<float, 1, 2, 1, 7>::validate(input, input0, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformWeightsKernel<float, 1, 2, 1, 7>::validate(weights, input1, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformOutputKernel<float, 1, 2, 1, 7>::validate(batched_mm_output, biases, output, winograd_info)));
    if(act_info.enabled())
    {
        NEActivationLayer::validate(output, nullptr, act_info);
    }
    return Status{};
}

inline Status validate_kernel_1x7(const ITensorInfo *input, const TensorInfo *input0, const TensorInfo *input1, const TensorInfo *batched_mm_output,
                                  const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const WinogradInfo &winograd_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformInputKernel<float, 2, 1, 7, 1>::validate(input, input0, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformWeightsKernel<float, 2, 1, 7, 1>::validate(weights, input1, winograd_info)));
    ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformOutputKernel<float, 2, 1, 7, 1>::validate(batched_mm_output, biases, output, winograd_info)));

    if(act_info.enabled())
    {
        NEActivationLayer::validate(output, nullptr, act_info);
    }
    return Status{};
}

inline Tensor4DShape internal_get_input_shape(const arm_compute::ITensor *input)
{
    const DataLayout data_layout = input->info()->data_layout();
    const int        in_width    = input->info()->dimension(get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH));
    const int        in_height   = input->info()->dimension(get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT));
    const int        in_channels = input->info()->dimension(get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL));
    const int        in_batches  = input->info()->dimension(3);

    return Tensor4DShape{ in_batches, in_height, in_width, in_channels };
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_UNUSED(output);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.stride().first != 1 || conv_info.stride().second != 1, "Winograd layer only supports unit strides.");
    if(biases != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
    }
    return INEWinogradLayerTransformWeightsKernel<float>::validate(input, weights);
}

Size2D winograd_output_tile(const Size2D &input_dims, const Size2D &kernel_dims)
{
    Size2D output_tile = Size2D{};
    if(kernel_dims == Size2D(3U, 3U))
    {
        output_tile = (input_dims.width <= 4 || input_dims.height <= 4) ? Size2D(2U, 2U) : Size2D(4U, 4U);
    }
    else if(kernel_dims == Size2D(5U, 5U))
    {
        output_tile = Size2D(2U, 2U);
    }
    else if(kernel_dims == Size2D(1U, 3U))
    {
        output_tile = Size2D(1U, 6U);
    }
    else if(kernel_dims == Size2D(3U, 1U))
    {
        output_tile = Size2D(6U, 1U);
    }
    else if(kernel_dims == Size2D(1U, 5U))
    {
        output_tile = Size2D(1U, 4U);
    }
    else if(kernel_dims == Size2D(5U, 1U))
    {
        output_tile = Size2D(4U, 1U);
    }
    else if(kernel_dims == Size2D(7U, 1U))
    {
        output_tile = Size2D(2U, 1U);
    }
    else if(kernel_dims == Size2D(1U, 7U))
    {
        output_tile = Size2D(1U, 2U);
    }
    return output_tile;
}

bool check_support_fast_math(const Size2D &output_tile, const Size2D &kernel_size)
{
    // Check if we want to configure a Winograd configuration which requires fast math
    using WinogradConfiguration = std::pair<std::pair<int, int>, std::pair<int, int>>;

    const std::vector<WinogradConfiguration> fast_math_winograd =
    {
        WinogradConfiguration(std::pair<int, int>(2, 2), std::pair<int, int>(5, 5)),
        WinogradConfiguration(std::pair<int, int>(4, 4), std::pair<int, int>(5, 5))
    };

    auto p = std::make_pair(std::pair<int, int>(output_tile.width, output_tile.height),
                            std::pair<int, int>(kernel_size.width, kernel_size.height));

    return std::find(fast_math_winograd.begin(), fast_math_winograd.end(), p) != fast_math_winograd.end();
}

} //namespace

NEWinogradConvolutionLayer::NEWinogradConvolutionLayer(const std::shared_ptr<IMemoryManager> &memory_manager)
    : _memory_group(memory_manager), _gemm_function(memory_manager), _transform_input_kernel(nullptr), _transform_output_kernel(nullptr), _transform_weights_kernel(nullptr), _activationlayer_function(),
      _permute_input(), _permute_weights(), _permute_output(), _input_transformed(), _output_transformed(), _input_workspace(), _output_workspace(), _kernel_storage(), _input_nhwc(), _output_nhwc(),
      _weights_hwio(), _input(), _weights(), _output(), _is_prepared(false), _is_activationlayer_enabled(false)
{
}

void NEWinogradConvolutionLayer::configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info,
                                           bool enable_fast_math)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), weights->info(), (biases != nullptr) ? biases->info() : nullptr, output->info(), conv_info));

    // Get indices for the width and height
    const DataLayout   data_layout = input->info()->data_layout();
    const unsigned int width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const unsigned int channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    const Size2D input_dims  = Size2D(input->info()->dimension(width_idx), input->info()->dimension(height_idx));
    const Size2D kernel_size = Size2D(weights->info()->dimension(width_idx), weights->info()->dimension(height_idx));
    const Size2D output_tile = winograd_output_tile(input_dims, kernel_size);

    // Check if the Winograd configuration requires fast math
    if(!enable_fast_math)
    {
        ARM_COMPUTE_ERROR_ON_MSG(check_support_fast_math(output_tile, kernel_size), "This Winograd configuration requires enable_fast_math=true");
    }

    _weights     = weights;
    _input       = input;
    _output      = output;
    _is_prepared = false;

    std::unique_ptr<INEWinogradLayerTransformInputKernel<float>>   transform_input_kernel;
    std::unique_ptr<INEWinogradLayerTransformWeightsKernel<float>> transform_weights_kernel;
    std::unique_ptr<INEWinogradLayerTransformOutputKernel<float>>  transform_output_kernel;

    int n_gemms = 0;
    int N_BLOCK = 0; // Size of block used by GEMM.

    if(kernel_size == Size2D(3, 3))
    {
        if(input->info()->dimension(width_idx) > 4 && input->info()->dimension(height_idx) > 4)
        {
            using config             = NEWinogradLayerConfiguration<float, float, 4, 4, 3, 3>;
            transform_input_kernel   = support::cpp14::make_unique<config::TransformInputKernel>();
            transform_weights_kernel = support::cpp14::make_unique<config::TransformWeightsKernel>();
            transform_output_kernel  = support::cpp14::make_unique<config::TransformOutputKernel>();
            n_gemms                  = config::WinogradBase::N_GEMMS;
            N_BLOCK                  = config::WinogradConv::N_BLOCK;
        }
        else
        {
            using config             = NEWinogradLayerConfiguration<float, float, 2, 2, 3, 3>;
            transform_input_kernel   = support::cpp14::make_unique<config::TransformInputKernel>();
            transform_weights_kernel = support::cpp14::make_unique<config::TransformWeightsKernel>();
            transform_output_kernel  = support::cpp14::make_unique<config::TransformOutputKernel>();
            n_gemms                  = config::WinogradBase::N_GEMMS;
            N_BLOCK                  = config::WinogradConv::N_BLOCK;
        }
    }
    else if(kernel_size == Size2D(5, 5))
    {
        using config             = NEWinogradLayerConfiguration<float, float, 2, 2, 5, 5>;
        transform_input_kernel   = support::cpp14::make_unique<config::TransformInputKernel>();
        transform_weights_kernel = support::cpp14::make_unique<config::TransformWeightsKernel>();
        transform_output_kernel  = support::cpp14::make_unique<config::TransformOutputKernel>();
        n_gemms                  = config::WinogradBase::N_GEMMS;
        N_BLOCK                  = config::WinogradConv::N_BLOCK;
    }
    else if(kernel_size == Size2D(1, 3))
    {
        using config             = NEWinogradLayerConfiguration<float, float, 6, 1, 3, 1>;
        transform_input_kernel   = support::cpp14::make_unique<config::TransformInputKernel>();
        transform_weights_kernel = support::cpp14::make_unique<config::TransformWeightsKernel>();
        transform_output_kernel  = support::cpp14::make_unique<config::TransformOutputKernel>();
        n_gemms                  = config::WinogradBase::N_GEMMS;
        N_BLOCK                  = config::WinogradConv::N_BLOCK;
    }
    else if(kernel_size == Size2D(3, 1))
    {
        using config             = NEWinogradLayerConfiguration<float, float, 1, 6, 1, 3>;
        transform_input_kernel   = support::cpp14::make_unique<config::TransformInputKernel>();
        transform_weights_kernel = support::cpp14::make_unique<config::TransformWeightsKernel>();
        transform_output_kernel  = support::cpp14::make_unique<config::TransformOutputKernel>();
        n_gemms                  = config::WinogradBase::N_GEMMS;
        N_BLOCK                  = config::WinogradConv::N_BLOCK;
    }
    else if(kernel_size == Size2D(1, 5))
    {
        using config             = NEWinogradLayerConfiguration<float, float, 4, 1, 5, 1>;
        transform_input_kernel   = support::cpp14::make_unique<config::TransformInputKernel>();
        transform_weights_kernel = support::cpp14::make_unique<config::TransformWeightsKernel>();
        transform_output_kernel  = support::cpp14::make_unique<config::TransformOutputKernel>();
        n_gemms                  = config::WinogradBase::N_GEMMS;
        N_BLOCK                  = config::WinogradConv::N_BLOCK;
    }
    else if(kernel_size == Size2D(5, 1))
    {
        using config             = NEWinogradLayerConfiguration<float, float, 1, 4, 1, 5>;
        transform_input_kernel   = support::cpp14::make_unique<config::TransformInputKernel>();
        transform_weights_kernel = support::cpp14::make_unique<config::TransformWeightsKernel>();
        transform_output_kernel  = support::cpp14::make_unique<config::TransformOutputKernel>();
        n_gemms                  = config::WinogradBase::N_GEMMS;
        N_BLOCK                  = config::WinogradConv::N_BLOCK;
    }
    else if(kernel_size == Size2D(1, 7))
    {
        using config             = NEWinogradLayerConfiguration<float, float, 2, 1, 7, 1>;
        transform_input_kernel   = support::cpp14::make_unique<config::TransformInputKernel>();
        transform_weights_kernel = support::cpp14::make_unique<config::TransformWeightsKernel>();
        transform_output_kernel  = support::cpp14::make_unique<config::TransformOutputKernel>();
        n_gemms                  = config::WinogradBase::N_GEMMS;
        N_BLOCK                  = config::WinogradConv::N_BLOCK;
    }
    else if(kernel_size == Size2D(7, 1))
    {
        using config             = NEWinogradLayerConfiguration<float, float, 1, 2, 1, 7>;
        transform_input_kernel   = support::cpp14::make_unique<config::TransformInputKernel>();
        transform_weights_kernel = support::cpp14::make_unique<config::TransformWeightsKernel>();
        transform_output_kernel  = support::cpp14::make_unique<config::TransformOutputKernel>();
        n_gemms                  = config::WinogradBase::N_GEMMS;
        N_BLOCK                  = config::WinogradConv::N_BLOCK;
    }
    else
    {
        ARM_COMPUTE_ERROR("Not supported.");
    }

    const PaddingType use_padding_type = (conv_info.pad_top() != 0u || conv_info.pad_left() != 0) ? PADDING_SAME : PADDING_VALID;
    const bool        use_same_padding = use_padding_type == PADDING_SAME;

    // Get convolved dimensions
    const int in_channels  = input->info()->dimension(channel_idx);
    const int out_channels = output->info()->dimension(channel_idx);

    const Tensor4DShape in_shape(internal_get_input_shape(input));
    const DataType      data_type      = input->info()->data_type();
    const size_t        data_type_size = input->info()->element_size();
    // Get the memory required to instantiate a new Winograd operator.
    constexpr size_t storage_alignment = 64;

    // Kernel Storage
    const size_t kernel_storage_size = transform_weights_kernel->get_weight_storage_size(out_channels,
                                                                                         in_channels)
                                       * data_type_size;

    // Input storage
    const size_t input_storage_size = transform_input_kernel->get_input_storage_size(in_shape.n_batches, in_shape.n_channels, in_shape.n_rows, in_shape.n_cols,
                                                                                     use_same_padding)
                                      * data_type_size;

    // Output storage
    const size_t output_storage_size = transform_output_kernel->get_output_storage_size(in_shape.n_batches, in_shape.n_rows, in_shape.n_cols, out_channels,
                                                                                        use_same_padding)
                                       * data_type_size;
    ;
    const KernelShape kernel_shape({ out_channels, static_cast<int>(kernel_size.height), static_cast<int>(kernel_size.width), in_channels });
    const int         kernel_matrix_stride = transform_weights_kernel->get_matrix_stride(kernel_shape);

    const int  output_matrix_stride = transform_output_kernel->get_matrix_stride(kernel_shape, in_shape, use_padding_type);
    const auto output_shape(transform_output_kernel->get_output_shape(kernel_shape, in_shape, use_padding_type));

    const int input_matrix_stride = transform_input_kernel->get_matrix_stride(kernel_shape, in_shape, use_padding_type);

    // Configure GEMM
    const int tile_rows                = iceildiv(output_shape.n_rows, output_tile.height);
    const int tile_cols                = iceildiv(output_shape.n_cols, output_tile.width);
    const int m                        = in_shape.n_batches * tile_rows * tile_cols;
    const int k                        = in_shape.n_channels;
    const int n                        = out_channels;
    const int kernel_matrix_row_stride = roundup(out_channels, N_BLOCK);
    const int output_matrix_row_stride = kernel_matrix_row_stride;

    TensorShape a_shape(k, m, 1, n_gemms);
    Strides     a_strides(data_type_size);
    a_strides.set(1, a_strides[0] * k);
    //a_strides.set(2, data_type_size * input_matrix_stride / n_gemms); FIXME: This is the real batch size, but RSH's code crashes if it's not 0.
    a_strides.set(2, 0);
    a_strides.set(3, data_type_size * input_matrix_stride);

    TensorShape b_shape(n, k, n_gemms);
    Strides     b_strides(data_type_size);
    b_strides.set(1, data_type_size * kernel_matrix_row_stride);
    b_strides.set(2, data_type_size * kernel_matrix_stride);

    TensorShape d_shape(n, m, 1, n_gemms);
    Strides     d_strides(data_type_size);
    d_strides.set(1, data_type_size * output_matrix_row_stride);
    //d_strides.set(2, data_type_size * output_matrix_stride / n_gemms); FIXME: This is the real batch size, but RSH's code crashes if it's not 0.
    d_strides.set(2, 0);
    d_strides.set(3, data_type_size * output_matrix_stride);

    TensorInfo a_info{};
    TensorInfo b_info{};
    TensorInfo d_info{};
    a_info.init(a_shape, 1, data_type, a_strides, 0, input_storage_size);
    b_info.init(b_shape, 1, data_type, b_strides, 0, kernel_storage_size);
    d_info.init(d_shape, 1, data_type, d_strides, 0, output_storage_size);

    _input_transformed.allocator()->init(a_info, storage_alignment);
    _kernel_storage.allocator()->init(b_info, storage_alignment);
    _output_transformed.allocator()->init(d_info, storage_alignment);

    // configure and allocate dst tensor to be used to convert from winograd domain to spatial domain when calling to reshape_output()
    TensorInfo info(TensorShape(_output->info()->dimension(2), _output->info()->dimension(0),
                                _output->info()->dimension(1), _output->info()->dimension(3)),
                    1, _output->info()->data_type());
    _output_nhwc.allocator()->init(info);

    const ITensor     *input_to_use  = _input;
    ITensor           *output_to_use = _output;
    PermutationVector  weights_permutation_vector(3U, 0U, 1U, 2U);
    const unsigned int max_num_threads = NEScheduler::get().num_threads();

    // Configure the kernel to transform the input tensor from NCHW -> NHWC
    if(data_layout == DataLayout::NCHW)
    {
        _memory_group.manage(&_input_nhwc);
        _permute_input.configure(input, &_input_nhwc, PermutationVector(2U, 0U, 1U));
        input_to_use               = &_input_nhwc;
        weights_permutation_vector = PermutationVector(3U, 2U, 0U, 1U);
    }

    // Configure input transform kernel
    _memory_group.manage(&_input_transformed);
    _memory_group.manage(&_input_workspace);
    transform_input_kernel->configure(input_to_use, in_shape.n_batches, in_shape.n_rows, in_shape.n_cols, in_shape.n_channels, use_padding_type,
                                      &_input_transformed, input_matrix_stride, &_input_workspace);
    const size_t input_workspace_size = transform_input_kernel->get_working_space_size(max_num_threads);
    TensorInfo   input_workspace_info(TensorShape(input_workspace_size), 1, _input->info()->data_type());
    _input_workspace.allocator()->init(input_workspace_info);
    _input_workspace.allocator()->allocate();
    if(data_layout == DataLayout::NCHW)
    {
        _input_nhwc.allocator()->allocate();
    }

    // Re-order a weight tensor from [Output feature map x Input feature map x Height x Width] to [Height x Width x Input feature map x Output feature map]
    _permute_weights.configure(weights, &_weights_hwio, weights_permutation_vector);
    transform_weights_kernel->configure(&_weights_hwio, &_kernel_storage, kernel_matrix_stride, out_channels, in_channels);

    // Configure GEMM function
    _memory_group.manage(&_output_transformed);
    _gemm_function.configure(&_input_transformed, &_kernel_storage, nullptr, &_output_transformed, 1.0f, 0.f);
    _input_transformed.allocator()->allocate();

    // Configure output transform function
    // The biases tensor has not been allocated at this point in time, the output transform will add the biases to the final result in the run() method
    if(data_layout == DataLayout::NCHW)
    {
        _memory_group.manage(&_output_nhwc);
        output_to_use = &_output_nhwc;
    }
    transform_output_kernel->configure(biases, &_output_transformed,
                                       output_matrix_stride, output_to_use,
                                       in_shape.n_batches, output_shape.n_rows, output_shape.n_cols, out_channels, &_output_workspace);
    const size_t output_workspace_size = transform_output_kernel->get_working_space_size(max_num_threads);
    TensorInfo   output_workspace_info(TensorShape(output_workspace_size), 1, _output->info()->data_type());
    _output_workspace.allocator()->init(output_workspace_info);
    _output_workspace.allocator()->allocate();
    _output_transformed.allocator()->allocate();

    // Reorder the convoluted output to ACL's ordering NCHW
    if(data_layout == DataLayout::NCHW)
    {
        _permute_output.configure(&_output_nhwc, _output, PermutationVector(1U, 2U, 0U));
        _output_nhwc.allocator()->allocate();
    }

    _transform_input_kernel   = std::move(transform_input_kernel);
    _transform_weights_kernel = std::move(transform_weights_kernel);
    _transform_output_kernel  = std::move(transform_output_kernel);

    //Configure Activation Layer
    _is_activationlayer_enabled = act_info.enabled();
    if(_is_activationlayer_enabled)
    {
        _activationlayer_function.configure(_output, nullptr, act_info);
    }
}

void NEWinogradConvolutionLayer::run()
{
    const DataLayout data_layout = _input->info()->data_layout();

    prepare();

    MemoryGroupResourceScope scope_mg(_memory_group);

    if(data_layout == DataLayout::NCHW)
    {
        //Bring channels to the front as Winograd code expects the tensor to be in the format NHWC
        _permute_input.run();
    }

    // Transform input tensor to the winograd domain
    NEScheduler::get().schedule(_transform_input_kernel.get(), Window::DimX);

    //Run 16 GEMMs in multiple threads, each kernel runs one or more GEMMs
    _gemm_function.run();

    // Transform output tensor to the spatial domain
    NEScheduler::get().schedule(_transform_output_kernel.get(), Window::DimX);

    if(data_layout == DataLayout::NCHW)
    {
        // Reorder the convoluted output to ACL's ordering NCHW
        _permute_output.run();
    }

    if(_is_activationlayer_enabled)
    {
        _activationlayer_function.run();
    }
}

Status NEWinogradConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                            const ActivationLayerInfo &act_info, bool enable_fast_math)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, weights, biases, output, conv_info));

    // Get indices for the width and height
    const size_t idx_width  = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_height = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT);

    // Input shape, kernel size and output tile
    const Size2D input_dims  = Size2D(input->dimension(idx_width), input->dimension(idx_height));
    const Size2D kernel_size = Size2D(weights->dimension(idx_width), weights->dimension(idx_height));
    const Size2D output_tile = winograd_output_tile(input_dims, kernel_size);

    // Check if the Winograd configuration requires fast math
    if(!enable_fast_math)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(check_support_fast_math(output_tile, kernel_size), "This Winograd configuration requires enable_fast_math=true");
    }

    const WinogradInfo winograd_info = WinogradInfo(output_tile,
                                                    kernel_size,
                                                    input_dims,
                                                    conv_info,
                                                    input->data_layout());

    // Validate input transform
    const TensorShape input0_shape = misc::shape_calculator::compute_winograd_input_transform_shape(*input, winograd_info);
    const TensorInfo  input0       = input->clone()->set_tensor_shape(input0_shape);
    // Validate filter transform
    const TensorShape input1_shape = misc::shape_calculator::compute_winograd_filter_transform_shape(*weights, winograd_info);
    const TensorInfo  input1       = weights->clone()->set_tensor_shape(input1_shape);
    // Validate batched matrix multiply
    TensorShape batched_mm_output_shape = input0.tensor_shape();
    batched_mm_output_shape[0]          = input1.tensor_shape()[0];
    const TensorInfo batched_mm_output  = input0.clone()->set_tensor_shape(batched_mm_output_shape);

    if(kernel_size == Size2D(3, 3))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_top() != 0u && conv_info.pad_top() != 1, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_bottom() != 0u && conv_info.pad_bottom() != 1, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_left() != 0u && conv_info.pad_left() != 1, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_right() != 0u && conv_info.pad_right() != 1, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_right() != conv_info.pad_left(), "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_top() != conv_info.pad_bottom(), "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_top() != conv_info.pad_left(), "Only SAME or VALID padding supported");
        return validate_kernel_3x3(input_dims, input, &input0, &input1, &batched_mm_output, weights, biases, output, winograd_info, act_info);
    }
    else if(kernel_size == Size2D(5, 5))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_top() != 0u && conv_info.pad_top() != 2, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_left() != 0u && conv_info.pad_left() != 2, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_bottom() != 0u && conv_info.pad_bottom() != 2, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_right() != 0u && conv_info.pad_right() != 2, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_right() != conv_info.pad_left(), "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_top() != conv_info.pad_bottom(), "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_top() != conv_info.pad_left(), "Only SAME or VALID padding supported");
        return validate_kernel_5x5(input, &input0, &input1, &batched_mm_output, weights, biases, output, winograd_info, act_info);
    }
    if(kernel_size == Size2D(3, 1))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_left() != 0u && conv_info.pad_left() != 1, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_right() != 0u && conv_info.pad_right() != 1, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_top() != 0u && conv_info.pad_bottom() != 0, "Only SAME or VALID padding supported");
        return validate_kernel_3x1(input, &input0, &input1, &batched_mm_output, weights, biases, output, winograd_info, act_info);
    }
    else if(kernel_size == Size2D(1, 3))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_top() != 0u && conv_info.pad_top() != 1, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_bottom() != 0u && conv_info.pad_bottom() != 1, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_left() != 0u && conv_info.pad_right() != 0, "Only SAME or VALID padding supported");
        return validate_kernel_1x3(input, &input0, &input1, &batched_mm_output, weights, biases, output, winograd_info, act_info);
    }
    else if(kernel_size == Size2D(5, 1))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_left() != 0u && conv_info.pad_left() != 2, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_right() != 0u && conv_info.pad_right() != 2, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_top() != 0u && conv_info.pad_bottom() != 0, "Only SAME or VALID padding supported");
        return validate_kernel_5x1(input, &input0, &input1, &batched_mm_output, weights, biases, output, winograd_info, act_info);
    }
    else if(kernel_size == Size2D(1, 5))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_top() != 0u && conv_info.pad_top() != 2, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_bottom() != 0u && conv_info.pad_bottom() != 2, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_left() != 0u && conv_info.pad_right() != 0, "Only SAME or VALID padding supported");
        return validate_kernel_1x5(input, &input0, &input1, &batched_mm_output, weights, biases, output, winograd_info, act_info);
    }
    else if(kernel_size == Size2D(7, 1))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_left() != 0u && conv_info.pad_left() != 3, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_right() != 0u && conv_info.pad_right() != 3, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_top() != 0u && conv_info.pad_bottom() != 0, "Only SAME or VALID padding supported");
        return validate_kernel_7x1(input, &input0, &input1, &batched_mm_output, weights, biases, output, winograd_info, act_info);
    }
    else if(kernel_size == Size2D(1, 7))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_top() != 0u && conv_info.pad_top() != 3, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_bottom() != 0u && conv_info.pad_bottom() != 3, "Only SAME or VALID padding supported");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.pad_left() != 0u && conv_info.pad_right() != 0, "Only SAME or VALID padding supported");
        return validate_kernel_1x7(input, &input0, &input1, &batched_mm_output, weights, biases, output, winograd_info, act_info);
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_MSG("Kernel shape not supported");
    }
}

void NEWinogradConvolutionLayer::prepare()
{
    if(!_is_prepared)
    {
        // Permute weights
        _weights_hwio.allocator()->allocate();
        _permute_weights.run();
        _weights->mark_as_unused();

        // Transform weights
        _kernel_storage.allocator()->allocate();
        NEScheduler::get().schedule(_transform_weights_kernel.get(), Window::DimX);

        _weights_hwio.allocator()->free();
        _is_prepared = true;
    }
}

} // namespace arm_compute
