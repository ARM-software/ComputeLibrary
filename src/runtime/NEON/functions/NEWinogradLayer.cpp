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
#include "arm_compute/runtime/NEON/functions/NEWinogradLayer.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "support/ToolchainSupport.h"

#include "arm_compute/core/NEON/kernels/NEWinogradLayerKernel.h"

#include "arm_compute/core/NEON/kernels/convolution/winograd/winograd_gemm.hpp"

namespace
{
inline Tensor4DShape internal_get_input_shape(const arm_compute::ITensor *input)
{
    const int in_width    = input->info()->dimension(0);
    const int in_height   = input->info()->dimension(1);
    const int in_batches  = input->info()->dimension(3);
    const int in_channels = input->info()->dimension(2);
    return Tensor4DShape({ in_batches, in_height, in_width, in_channels });
}
} /* namespace */

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(0) != 3 && weights->dimension(0) != 5, "Only 3 and 5 kernels are supported");
    ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 4);

    if(biases != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
    }

    // Get parameters from conv_info
    unsigned int stride_x = 0;
    unsigned int stride_y = 0;
    std::tie(stride_x, stride_y) = conv_info.stride();
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(stride_y != 1 || stride_x != 1, "Winograd layer only supports unit strides.");

    ARM_COMPUTE_UNUSED(output);

    return Status{};
}
} //namespace

NEWinogradLayer::NEWinogradLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _batched_gemm_kernel(nullptr), _transform_input_kernel(nullptr), _transform_output_kernel(nullptr), _transform_weights_kernel(nullptr),
      _activationlayer_function(), _permute_input(), _permute_weights(), _permute_output(), _input_workspace(), _output_workspace(), _kernel_storage(), _input_nhwc(), _output_nhwc(), _weights_hwio(),
      _input(), _weights(), _output(), _reshaped_kernel(false), _is_activationlayer_enabled(false)
{
} /* arm_compute */

void NEWinogradLayer::configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_UNUSED(conv_info);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), weights->info(), (biases != nullptr) ? biases->info() : nullptr, output->info(), conv_info));

    _weights = weights;
    _input   = input;
    _output  = output;

    std::unique_ptr<INEWinogradLayerBatchedGEMMKernel<float, float>> batched_gemm_kernel;
    std::unique_ptr<INEWinogradLayerTransformInputKernel<float>>   transform_input_kernel;
    std::unique_ptr<INEWinogradLayerTransformWeightsKernel<float>> transform_weights_kernel;
    std::unique_ptr<INEWinogradLayerTransformOutputKernel<float>>  transform_output_kernel;

    switch(weights->info()->dimension(0))
    {
        case 3:
        {
            batched_gemm_kernel      = support::cpp14::make_unique<NEWinogradLayerBatchedGEMMKernel<float, float, 2, 2, 3, 3>>();
            transform_input_kernel   = support::cpp14::make_unique<NEWinogradLayerTransformInputKernel<float, 2, 2, 3, 3>>();
            transform_weights_kernel = support::cpp14::make_unique<NEWinogradLayerTransformWeightsKernel<float, 2, 2, 3, 3>>();
            transform_output_kernel  = support::cpp14::make_unique<NEWinogradLayerTransformOutputKernel<float, 2, 2, 3, 3>>();
            break;
        }
        case 5:
        {
            batched_gemm_kernel      = support::cpp14::make_unique<NEWinogradLayerBatchedGEMMKernel<float, float, 2, 2, 5, 5>>();
            transform_input_kernel   = support::cpp14::make_unique<NEWinogradLayerTransformInputKernel<float, 2, 2, 5, 5>>();
            transform_weights_kernel = support::cpp14::make_unique<NEWinogradLayerTransformWeightsKernel<float, 2, 2, 5, 5>>();
            transform_output_kernel  = support::cpp14::make_unique<NEWinogradLayerTransformOutputKernel<float, 2, 2, 5, 5>>();
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

    // Get parameters from conv_info
    unsigned int stride_x = 0;
    unsigned int stride_y = 0;
    std::tie(stride_x, stride_y) = conv_info.stride();
    ARM_COMPUTE_ERROR_ON_MSG(stride_y != 1 || stride_x != 1, "Winograd layer only supports unit strides.");

    // Get convolved dimensions
    const int in_channels  = input->info()->dimension(2);
    const int out_channels = output->info()->dimension(2);

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

    // Re-order a weight tensor from [Output feature map x Input feature map x Height x Width] to [Height x Width x Input feature map x Output feature map]
    _permute_weights.configure(weights, &_weights_hwio, PermutationVector(3U, 2U, 0U, 1U));
    _weights_hwio.allocator()->allocate();

    // configure the kernel to transform the input tensor from NCHW -> NHWC
    _permute_input.configure(input, &_input_nhwc, PermutationVector(2U, 0U, 1U));
    _input_nhwc.allocator()->allocate();

    const int         weights_width  = weights->info()->dimension(0);
    const int         weights_height = weights->info()->dimension(1);
    const KernelShape kernel_shape({ out_channels, weights_height, weights_width, in_channels });

    // Configure the InputTransform
    const int input_matrix_stride = transform_input_kernel->get_matrix_stride(kernel_shape, in_shape, use_padding_type);
    transform_input_kernel->configure(reinterpret_cast<float *>(_input_nhwc.buffer()), in_shape.n_batches, in_shape.n_rows, in_shape.n_cols, in_shape.n_channels, use_padding_type,
                                      reinterpret_cast<float *>(_input_workspace.buffer()), input_matrix_stride);

    // Configure WeightsTransform
    const int kernel_matrix_stride = transform_weights_kernel->get_matrix_stride(kernel_shape);
    transform_weights_kernel->configure(&_weights_hwio, reinterpret_cast<float *>(_kernel_storage.buffer()), kernel_matrix_stride, out_channels, in_channels);

    // Configure OutputTransform
    //The biases tensor has not been allocated at this point in time, the output transform will add the biases to the final result in the run() method
    const int  output_matrix_stride = transform_output_kernel->get_matrix_stride(kernel_shape, in_shape, use_padding_type);
    const auto output_shape(transform_output_kernel->get_output_shape(kernel_shape, in_shape, use_padding_type));

    transform_output_kernel->configure(biases, reinterpret_cast<float *>(_output_workspace.buffer()),
                                       output_matrix_stride, reinterpret_cast<float *>(_output_nhwc.buffer()),
                                       in_shape.n_batches, output_shape.n_rows, output_shape.n_cols, out_channels);

    // Configure Batched GEMMs
    const int      output_tile_rows         = batched_gemm_kernel->get_output_tile_rows();
    const int      output_tile_cols         = batched_gemm_kernel->get_output_tile_cols();
    const int      n_block                  = batched_gemm_kernel->get_number_blocks();
    const int      tile_rows                = iceildiv(output_shape.n_rows, output_tile_rows);
    const int      tile_cols                = iceildiv(output_shape.n_cols, output_tile_cols);
    const int      m                        = in_shape.n_batches * tile_rows * tile_cols;
    const int      k                        = in_shape.n_channels;
    const int      n                        = out_channels;
    const int      input_matrix_row_stride  = in_shape.n_channels;
    const int      kernel_matrix_row_stride = roundup(out_channels, n_block);
    const int      output_matrix_row_stride = kernel_matrix_row_stride;
    const unsigned n_gemms                  = batched_gemm_kernel->get_number_gemms();

    batched_gemm_kernel->configure(n_gemms, m, k, n,
                                   input_matrix_stride, input_matrix_row_stride,
                                   kernel_matrix_stride, kernel_matrix_row_stride,
                                   output_matrix_stride, output_matrix_row_stride,
                                   reinterpret_cast<float *>(_input_workspace.buffer()),
                                   reinterpret_cast<float *>(_kernel_storage.buffer()),
                                   reinterpret_cast<float *>(_output_workspace.buffer()));

    // Reorder the convoluted output to ACL's ordering NCHW
    _permute_output.configure(&_output_nhwc, _output, PermutationVector(1U, 2U, 0U));

    _transform_input_kernel   = std::move(transform_input_kernel);
    _transform_weights_kernel = std::move(transform_weights_kernel);
    _transform_output_kernel  = std::move(transform_output_kernel);
    _batched_gemm_kernel      = std::move(batched_gemm_kernel);

    //Configure Activation Layer
    _is_activationlayer_enabled = act_info.enabled();
    if(_is_activationlayer_enabled)
    {
        _activationlayer_function.configure(output, nullptr, act_info);
    }
}

void NEWinogradLayer::run()
{
    _memory_group.acquire();
    if(!_reshaped_kernel)
    {
        _reshaped_kernel = true;
        _permute_weights.run();
        NEScheduler::get().schedule(_transform_weights_kernel.get(), Window::DimX);
    }
    //Bring channels to the front as Winograd code expects the tensor to be in the format NHWC
    _permute_input.run();

    // Transform input tensor to the winograd domain
    NEScheduler::get().schedule(_transform_input_kernel.get(), Window::DimX);

    //Run 16 GEMMs in multiple threads, each kernel runs one or more GEMMs
    NEScheduler::get().schedule(_batched_gemm_kernel.get(), Window::DimX);

    // Transform output tensor to the spatial domain
    NEScheduler::get().schedule(_transform_output_kernel.get(), Window::DimX);

    // Reorder the convoluted output to ACL's ordering NCHW
    _permute_output.run();

    if(_is_activationlayer_enabled)
    {
        _activationlayer_function.run();
    }

    _memory_group.release();
}

Status NEWinogradLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_RETURN_ERROR_ON(validate_arguments(input, weights, biases, output, conv_info));

    return Status{};
}

} // namespace arm_compute
