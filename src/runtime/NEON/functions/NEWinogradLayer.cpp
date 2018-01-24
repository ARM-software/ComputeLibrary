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

#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "support/ToolchainSupport.h"

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
NEWinogradLayer::NEWinogradLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _winograd_kernel(), _permute_input(), _permute_weights(), _permute_output(), _input_workspace(), _output_workspace(), _kernel_storage(), _input_nhwc(),
      _output_nhwc(), _weights_hwio(), _input(), _weights(), _output(), _reshaped_kernel(false), _conv()
{
} /* arm_compute */

void NEWinogradLayer::configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    ARM_COMPUTE_ERROR_ON_MSG(weights->info()->dimension(1) != 3 || weights->info()->dimension(0) != 3, "Only 3x3 kernels are supported");
    ARM_COMPUTE_ERROR_ON(weights->info()->num_dimensions() > 4);

    if(biases != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        ARM_COMPUTE_ERROR_ON(biases->info()->num_dimensions() > 1);
    }

    _weights = weights;
    _input   = input;
    _output  = output;

    // Get parameters from conv_info
    unsigned int stride_x = 0;
    unsigned int stride_y = 0;
    std::tie(stride_x, stride_y) = conv_info.stride();
    ARM_COMPUTE_ERROR_ON_MSG(stride_y != 1 || stride_x != 1, "Winograd layer only supports unit strides.");

    // Get convolved dimensions
    const int in_channels  = input->info()->dimension(2);
    const int out_channels = output->info()->dimension(2);

    const Tensor4DShape in_shape(internal_get_input_shape(input));

    // Get the memory required to instantiate a new Winograd operator.
    constexpr size_t storage_alignment   = 64;
    const size_t     kernel_storage_size = NEWinogradLayerKernel::get_weight_storage_size(out_channels, in_channels) * sizeof(float);
    _kernel_storage.allocator()->init(TensorInfo(TensorShape{ (kernel_storage_size + storage_alignment - 1) }, 1, DataType::U8));
    _memory_group.manage(&_kernel_storage);
    _memory_group.manage(&_input_nhwc);
    _kernel_storage.allocator()->allocate();
    // Input storage
    const size_t input_storage_size = NEWinogradLayerKernel::get_input_storage_size(in_shape.n_batches, in_shape.n_channels, in_shape.n_rows, in_shape.n_cols, false) * sizeof(float);
    _input_workspace.allocator()->init(TensorInfo(TensorShape{ (input_storage_size + storage_alignment - 1) }, 1, DataType::U8));
    _memory_group.manage(&_input_workspace);
    _input_workspace.allocator()->allocate();

    // Output storage
    const size_t output_storage_size = NEWinogradLayerKernel::get_output_storage_size(in_shape.n_batches, in_shape.n_rows, in_shape.n_cols, out_channels, false) * sizeof(float);
    _output_workspace.allocator()->init(TensorInfo(TensorShape{ (output_storage_size + storage_alignment - 1) }, 1, DataType::U8));
    _memory_group.manage(&_output_workspace);
    _output_workspace.allocator()->allocate();

    // configure and allocate dst tensor to be used to convert from winograd domain to spatial domain when calling to reshape_output()
    TensorInfo info(TensorShape(_output->info()->dimension(2), _output->info()->dimension(0),
                                _output->info()->dimension(1), _output->info()->dimension(3)),
                    1, _output->info()->data_type());
    _output_nhwc.allocator()->init(info);

    _output_nhwc.allocator()->allocate();

    // Re-order a weight tensor from [Output feature map x Input feature map x Height x Width] to [Height x Width x Input feature map x Output feature map]
    switch(weights->info()->num_dimensions())
    {
        case 3:
        {
            _permute_weights.configure(weights, &_weights_hwio, PermutationVector(2U, 0U, 1U));
            break;
        }
        case 4:
        {
            _permute_weights.configure(weights, &_weights_hwio, PermutationVector(3U, 2U, 0U, 1U));
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Not supported.");
            break;
        }
    }

    _weights_hwio.allocator()->allocate();

    // configure the kernel to transform the input tensor from NCHW -> NHWC
    _permute_input.configure(input, &_input_nhwc, PermutationVector(2U, 0U, 1U));

    _input_nhwc.allocator()->allocate();

    // Create Winograd operator object
    _conv = support::cpp14::make_unique<Winograd3x3F32>(
                in_shape.n_batches,
                in_shape.n_channels,
                in_shape.n_rows,
                in_shape.n_cols,
                out_channels,
                false,
                reinterpret_cast<const float *>(_weights_hwio.buffer()),
                reinterpret_cast<float *>(_kernel_storage.buffer()),
                reinterpret_cast<float *>(_input_nhwc.buffer()),
                reinterpret_cast<float *>(_input_workspace.buffer()),
                reinterpret_cast<float *>(_output_nhwc.buffer()),
                reinterpret_cast<float *>(_output_workspace.buffer()));

    // Configure the kernel, padding not needed so it's safe to call configure after allocare
    _winograd_kernel.configure(_conv.get());

    // Reorder the convoluted output to ACL's ordering NCHW
    _permute_output.configure(&_output_nhwc, _output, PermutationVector(1U, 2U, 0U));

}

void NEWinogradLayer::run()
{
    _memory_group.acquire();
    if(!_reshaped_kernel)
    {
        _reshaped_kernel = true;
        _permute_weights.run();
        _conv->transform_weights();
    }
    //Bring channels to the front as Winograd code expects the tensor to be in the format NHWC
    _permute_input.run();
    // Transform input tensor to the winograd domain
    _conv->transform_input();
    //Run 16 GEMMs in multiple threads, each kernel runs one or more GEMMs
    NEScheduler::get().schedule(&_winograd_kernel, Window::DimX);
    // Transform output tensor to the spatial domain
    _conv->transform_output();
    // Reorder the convoluted output to ACL's ordering NCHW
    _permute_output.run();
    _memory_group.release();
}
} // namespace arm_compute
