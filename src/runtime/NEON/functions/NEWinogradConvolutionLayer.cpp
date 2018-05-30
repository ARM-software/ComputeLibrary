/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/NEON/AssemblyHelper.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "support/ToolchainSupport.h"

#include "arm_compute/core/NEON/kernels/NEWinogradConvolutionLayerKernel.h"

#include "arm_compute/core/NEON/kernels/convolution/winograd/winograd_gemm.hpp"

namespace arm_compute
{
namespace
{
inline Tensor4DShape internal_get_input_shape(const arm_compute::ITensor *input)
{
    const DataLayout data_layout = input->info()->data_layout();
    const int        in_width    = input->info()->dimension(get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH));
    const int        in_height   = input->info()->dimension(get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT));
    const int        in_channels = input->info()->dimension(get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL));
    const int        in_batches  = input->info()->dimension(3);

    return Tensor4DShape({ in_batches, in_height, in_width, in_channels });
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info)
{
    const DataLayout   data_layout = input->data_layout();
    const unsigned int width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    ARM_COMPUTE_UNUSED(output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(width_idx) != 3 && weights->dimension(height_idx) != 5, "Only 3 and 5 kernels are supported");
    ARM_COMPUTE_RETURN_ERROR_ON(data_layout != DataLayout::NCHW); // COMPMID-1287
    ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 4);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.stride().first != 1 || conv_info.stride().second != 1, "Winograd layer only supports unit strides.");

    if(biases != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
    }

    return Status{};
}

Size2D winograd_output_tile(const Size2D &input_dims, const Size2D &kernel_dims)
{
    Size2D output_tile = Size2D{};

    if(kernel_dims == Size2D(3U, 3U))
    {
        output_tile = (input_dims.width <= 4 && input_dims.height <= 4) ? Size2D(2U, 2U) : Size2D(4U, 4U);
    }
    else if(kernel_dims == Size2D(5U, 5U))
    {
        output_tile = Size2D(2U, 2U);
    }

    return output_tile;
}

bool check_support_fast_math(const Size2D &output_tile, const Size2D &kernel_size)
{
    // Check if we want to configure a Winograd configuration which requires fast math
    using WinogradConfiguration = std::pair<std::pair<int, int>, std::pair<int, int>>;

    std::vector<WinogradConfiguration> fast_math_winograd =
    {
        WinogradConfiguration(std::pair<int, int>(2, 2), std::pair<int, int>(5, 5)),
        WinogradConfiguration(std::pair<int, int>(4, 4), std::pair<int, int>(5, 5))
    };

    auto p = std::make_pair(std::pair<int, int>(output_tile.width, output_tile.height),
                            std::pair<int, int>(kernel_size.width, kernel_size.height));

    return std::find(fast_math_winograd.begin(), fast_math_winograd.end(), p) != fast_math_winograd.end();
}

} //namespace

NEWinogradConvolutionLayer::NEWinogradConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _arm_gemm(nullptr), _gemm_kernel(nullptr), _transform_input_kernel(nullptr), _transform_output_kernel(nullptr), _transform_weights_kernel(nullptr),
      _activationlayer_function(), _permute_input(), _permute_weights(), _permute_output(), _input_workspace(), _output_workspace(), _kernel_storage(), _input_nhwc(), _output_nhwc(), _weights_hwio(),
      _workspace(), _input(), _weights(), _output(), _reshaped_kernel(false), _is_activationlayer_enabled(false)
{
} /* arm_compute */

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

    _weights = weights;
    _input   = input;
    _output  = output;

    std::unique_ptr<INEWinogradLayerTransformInputKernel<float>>   transform_input_kernel;
    std::unique_ptr<INEWinogradLayerTransformWeightsKernel<float>> transform_weights_kernel;
    std::unique_ptr<INEWinogradLayerTransformOutputKernel<float>>  transform_output_kernel;

    int n_gemms = 0;
    int N_BLOCK = 0; // Size of block used by GEMM.

    switch(kernel_size.width)
    {
        case 3:
        {
            if(input->info()->dimension(width_idx) > 4 && input->info()->dimension(height_idx) > 4)
            {
                transform_input_kernel   = support::cpp14::make_unique<NEWinogradLayerTransformInputKernel<float, 4, 4, 3, 3>>();
                transform_weights_kernel = support::cpp14::make_unique<NEWinogradLayerTransformWeightsKernel<float, 4, 4, 3, 3>>();
                transform_output_kernel  = support::cpp14::make_unique<NEWinogradLayerTransformOutputKernel<float, 4, 4, 3, 3>>();
                n_gemms                  = NEWinogradLayerBatchedGEMMKernel<float, float, 4, 4, 3, 3>::WinogradBase::N_GEMMS;
                N_BLOCK                  = NEWinogradLayerBatchedGEMMKernel<float, float, 4, 4, 3, 3>::WinogradConv::N_BLOCK;
            }
            else
            {
                transform_input_kernel   = support::cpp14::make_unique<NEWinogradLayerTransformInputKernel<float, 2, 2, 3, 3>>();
                transform_weights_kernel = support::cpp14::make_unique<NEWinogradLayerTransformWeightsKernel<float, 2, 2, 3, 3>>();
                transform_output_kernel  = support::cpp14::make_unique<NEWinogradLayerTransformOutputKernel<float, 2, 2, 3, 3>>();
                n_gemms                  = NEWinogradLayerBatchedGEMMKernel<float, float, 2, 2, 3, 3>::WinogradBase::N_GEMMS;
                N_BLOCK                  = NEWinogradLayerBatchedGEMMKernel<float, float, 2, 2, 3, 3>::WinogradConv::N_BLOCK;
            }
            break;
        }
        case 5:
        {
            transform_input_kernel   = support::cpp14::make_unique<NEWinogradLayerTransformInputKernel<float, 2, 2, 5, 5>>();
            transform_weights_kernel = support::cpp14::make_unique<NEWinogradLayerTransformWeightsKernel<float, 2, 2, 5, 5>>();
            transform_output_kernel  = support::cpp14::make_unique<NEWinogradLayerTransformOutputKernel<float, 2, 2, 5, 5>>();
            n_gemms                  = NEWinogradLayerBatchedGEMMKernel<float, float, 2, 2, 5, 5>::WinogradBase::N_GEMMS;
            N_BLOCK                  = NEWinogradLayerBatchedGEMMKernel<float, float, 2, 2, 5, 5>::WinogradConv::N_BLOCK;
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Not supported.");
            break;
        }
    }

    const PaddingType use_padding_type = (conv_info.pad_left() != 0u) ? PADDING_SAME : PADDING_VALID;
    const bool        use_same_padding = use_padding_type == PADDING_SAME;

    // Get convolved dimensions
    const int in_channels  = input->info()->dimension(channel_idx);
    const int out_channels = output->info()->dimension(channel_idx);

    const Tensor4DShape in_shape(internal_get_input_shape(input));
    const size_t        data_type_size = input->info()->element_size();
    // Get the memory required to instantiate a new Winograd operator.
    constexpr size_t storage_alignment   = 64;
    const size_t     kernel_storage_size = transform_weights_kernel->get_weight_storage_size(out_channels, in_channels) * data_type_size;
    _kernel_storage.allocator()->init(TensorInfo(TensorShape{ (kernel_storage_size + storage_alignment - 1) }, 1, DataType::U8));
    _kernel_storage.allocator()->allocate();
    // Input storage
    const size_t input_storage_size = transform_input_kernel->get_input_storage_size(in_shape.n_batches, in_shape.n_channels, in_shape.n_rows, in_shape.n_cols, use_same_padding) * data_type_size;
    _input_workspace.allocator()->init(TensorInfo(TensorShape{ (input_storage_size + storage_alignment - 1) }, 1, DataType::U8));
    _input_workspace.allocator()->allocate();

    // Output storage
    const size_t output_storage_size = transform_output_kernel->get_output_storage_size(in_shape.n_batches, in_shape.n_rows, in_shape.n_cols, out_channels, use_same_padding) * data_type_size;
    _output_workspace.allocator()->init(TensorInfo(TensorShape{ (output_storage_size + storage_alignment - 1) }, 1, DataType::U8));
    _output_workspace.allocator()->allocate();

    // configure and allocate dst tensor to be used to convert from winograd domain to spatial domain when calling to reshape_output()
    TensorInfo info(TensorShape(_output->info()->dimension(2), _output->info()->dimension(0),
                                _output->info()->dimension(1), _output->info()->dimension(3)),
                    1, _output->info()->data_type());
    _output_nhwc.allocator()->init(info);
    _output_nhwc.allocator()->allocate();

    const KernelShape kernel_shape({ out_channels, static_cast<int>(kernel_size.height), static_cast<int>(kernel_size.width), in_channels });

    // Configure the InputTransform
    const int input_matrix_stride = transform_input_kernel->get_matrix_stride(kernel_shape, in_shape, use_padding_type);

    if(data_layout == DataLayout::NCHW)
    {
        // configure the kernel to transform the input tensor from NCHW -> NHWC
        _permute_input.configure(input, &_input_nhwc, PermutationVector(2U, 0U, 1U));
        _input_nhwc.allocator()->allocate();
        transform_input_kernel->configure(&_input_nhwc, in_shape.n_batches, in_shape.n_rows, in_shape.n_cols, in_shape.n_channels, use_padding_type,
                                          reinterpret_cast<float *>(_input_workspace.buffer()), input_matrix_stride);
    }
    else
    {
        transform_input_kernel->configure(_input, in_shape.n_batches, in_shape.n_rows, in_shape.n_cols, in_shape.n_channels, use_padding_type,
                                          reinterpret_cast<float *>(_input_workspace.buffer()), input_matrix_stride);
    }

    // Configure WeightsTransform
    const int kernel_matrix_stride = transform_weights_kernel->get_matrix_stride(kernel_shape);
    if(data_layout == DataLayout::NCHW)
    {
        // Re-order a weight tensor from [Output feature map x Input feature map x Height x Width] to [Height x Width x Input feature map x Output feature map]
        _permute_weights.configure(weights, &_weights_hwio, PermutationVector(3U, 2U, 0U, 1U));

        transform_weights_kernel->configure(&_weights_hwio, reinterpret_cast<float *>(_kernel_storage.buffer()), kernel_matrix_stride, out_channels, in_channels);
    }
    else
    {
        // Re-order a weight tensor from [Output feature map x Input feature map x Height x Width] to [Height x Width x Input feature map x Output feature map]
        _permute_weights.configure(weights, &_weights_hwio, PermutationVector(3U, 0U, 1U, 2U));

        transform_weights_kernel->configure(&_weights_hwio, reinterpret_cast<float *>(_kernel_storage.buffer()), kernel_matrix_stride, out_channels, in_channels);
    }
    _weights_hwio.allocator()->allocate();

    // Configure OutputTransform
    //The biases tensor has not been allocated at this point in time, the output transform will add the biases to the final result in the run() method
    const int  output_matrix_stride = transform_output_kernel->get_matrix_stride(kernel_shape, in_shape, use_padding_type);
    const auto output_shape(transform_output_kernel->get_output_shape(kernel_shape, in_shape, use_padding_type));

    if(data_layout == DataLayout::NCHW)
    {
        transform_output_kernel->configure(biases, reinterpret_cast<float *>(_output_workspace.buffer()),
                                           output_matrix_stride, &_output_nhwc,
                                           in_shape.n_batches, output_shape.n_rows, output_shape.n_cols, out_channels);
    }
    else
    {
        transform_output_kernel->configure(biases, reinterpret_cast<float *>(_output_workspace.buffer()),
                                           output_matrix_stride, _output,
                                           in_shape.n_batches, output_shape.n_rows, output_shape.n_cols, out_channels);
    }

    // Configure GEMM
    const int    tile_rows                = iceildiv(output_shape.n_rows, output_tile.height);
    const int    tile_cols                = iceildiv(output_shape.n_cols, output_tile.width);
    const int    m                        = in_shape.n_batches * tile_rows * tile_cols;
    const int    k                        = in_shape.n_channels;
    const int    n                        = out_channels;
    const int    input_matrix_row_stride  = in_shape.n_channels;
    const int    kernel_matrix_row_stride = roundup(out_channels, N_BLOCK);
    const int    output_matrix_row_stride = kernel_matrix_row_stride;
    unsigned int num_threads              = NEScheduler::get().num_threads();

    _arm_gemm = arm_gemm::gemm<float, float>(NEScheduler::get().cpu_info(), m, n, k, 1, n_gemms, false, false, 1.f, 0.f, num_threads, false);
    _arm_gemm->set_arrays(reinterpret_cast<float *>(_input_workspace.buffer()), input_matrix_row_stride, 0, input_matrix_stride, reinterpret_cast<float *>(_kernel_storage.buffer()),
                          kernel_matrix_row_stride, kernel_matrix_stride, reinterpret_cast<float *>(_output_workspace.buffer()), output_matrix_row_stride, 0, output_matrix_stride);

    auto acl_gemm_wrapper = support::cpp14::make_unique<NEGEMMAssemblyWrapper<arm_gemm::GemmCommon<float, float>>>();
    acl_gemm_wrapper->configure(_arm_gemm.get());
    const size_t workspace_size = _arm_gemm->get_working_size();

    // Allocate workspace
    if(workspace_size > 0)
    {
        const unsigned int alignment = 4096;
        // TODO (COMPMID-1248) : Add support for memory manager in NEWinogradConvolutionLayer
        // Warning : Do not set a memory group in allocate_workspace, should be done under COMPMID-1248
        allocate_workspace(workspace_size, _workspace, nullptr, alignment, 1);
        _arm_gemm->set_working_space(reinterpret_cast<float *>(_workspace.buffer()));
    }

    const unsigned int window_size = _arm_gemm->get_window_size();
    if(window_size < num_threads)
    {
        num_threads = window_size;
        _arm_gemm->set_nthreads(num_threads);
    }

    _gemm_kernel = std::move(acl_gemm_wrapper);

    // Reorder the convoluted output to ACL's ordering NCHW
    _permute_output.configure(&_output_nhwc, _output, PermutationVector(1U, 2U, 0U));

    _transform_input_kernel   = std::move(transform_input_kernel);
    _transform_weights_kernel = std::move(transform_weights_kernel);
    _transform_output_kernel  = std::move(transform_output_kernel);

    //Configure Activation Layer
    _is_activationlayer_enabled = act_info.enabled();
    if(data_layout == DataLayout::NCHW && _is_activationlayer_enabled)
    {
        _activationlayer_function.configure(_output, nullptr, act_info);
    }
}

void NEWinogradConvolutionLayer::run()
{
    const DataLayout data_layout = _input->info()->data_layout();

    _memory_group.acquire();
    if(!_reshaped_kernel)
    {
        _reshaped_kernel = true;
        _permute_weights.run();
        NEScheduler::get().schedule(_transform_weights_kernel.get(), Window::DimX);
    }

    if(data_layout == DataLayout::NCHW)
    {
        //Bring channels to the front as Winograd code expects the tensor to be in the format NHWC
        _permute_input.run();
    }
    // Transform input tensor to the winograd domain
    NEScheduler::get().schedule(_transform_input_kernel.get(), Window::DimX);

    //Run 16 GEMMs in multiple threads, each kernel runs one or more GEMMs
    NEScheduler::get().schedule(_gemm_kernel.get(), Window::DimX);

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
    _memory_group.release();
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
    switch(weights->dimension(idx_width))
    {
        case 3:
        {
            if(input_dims.width > 4 && input_dims.height > 4)
            {
                ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformInputKernel<float, 4, 4, 3, 3>::validate(input, &input0, winograd_info)));
            }
            else
            {
                ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformInputKernel<float, 2, 2, 3, 3>::validate(input, &input0, winograd_info)));
            }
            break;
        }
        case 5:
        {
            ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformInputKernel<float, 2, 2, 5, 5>::validate(input, &input0, winograd_info)));
            break;
        }
        default:
        {
            ARM_COMPUTE_RETURN_ERROR_MSG("Only 3x3 and 5x5 kernels supported.");
            break;
        }
    }
    // Validate filter transform
    const TensorShape input1_shape = misc::shape_calculator::compute_winograd_filter_transform_shape(*weights, winograd_info);
    const TensorInfo  input1       = weights->clone()->set_tensor_shape(input1_shape);

    switch(weights->dimension(idx_width))
    {
        case 3:
        {
            if(input_dims.width > 4 && input_dims.height > 4)
            {
                ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformWeightsKernel<float, 4, 4, 3, 3>::validate(weights, &input1, winograd_info)));
            }
            else
            {
                ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformWeightsKernel<float, 2, 2, 3, 3>::validate(weights, &input1, winograd_info)));
            }
            break;
        }
        case 5:
        {
            ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformWeightsKernel<float, 2, 2, 5, 5>::validate(weights, &input1, winograd_info)));
            break;
        }
        default:
        {
            ARM_COMPUTE_RETURN_ERROR_MSG("Only 3x3 and 5x5 kernels supported.");
            break;
        }
    }
    // Validate batched matrix multiply
    TensorShape batched_mm_output_shape = input0.tensor_shape();
    batched_mm_output_shape[0]          = input1.tensor_shape()[0];
    const TensorInfo batched_mm_output  = input0.clone()->set_tensor_shape(batched_mm_output_shape);
    switch(weights->dimension(idx_width))
    {
        case 3:
        {
            if(input_dims.width > 4 && input_dims.height > 4)
            {
                // Validate output transform
                ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformOutputKernel<float, 4, 4, 3, 3>::validate(&batched_mm_output, biases, output, winograd_info)));
            }
            else
            {
                // Validate output transform
                ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformOutputKernel<float, 2, 2, 3, 3>::validate(&batched_mm_output, biases, output, winograd_info)));
            }
            break;
        }
        case 5:
        {
            // Validate output transform
            ARM_COMPUTE_RETURN_ON_ERROR((NEWinogradLayerTransformOutputKernel<float, 2, 2, 5, 5>::validate(&batched_mm_output, biases, output, winograd_info)));
            break;
        }
        default:
        {
            ARM_COMPUTE_RETURN_ERROR_MSG("Only 3x3 and 5x5 kernels supported.");
            break;
        }
    }

    // Validate Activation Layer
    if(act_info.enabled())
    {
        NEActivationLayer::validate(output, nullptr, act_info);
    }
    return Status{};
}

} // namespace arm_compute
