/*
 * Copyright (c) 2017 ARM Limited.
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
    : _memory_group(std::move(memory_manager)), _winograd_kernel(), _weights_workspace(), _workspace(), _kernel_storage(), _input(), _weights(), _output(), _reshaped_kernel(false), _conv()
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
    auto      padding     = PADDING_VALID;
    const int in_channels = input->info()->dimension(2);

    const int out_channels   = output->info()->dimension(2);
    const int weights_width  = weights->info()->dimension(0);
    const int weights_height = weights->info()->dimension(1);

    const KernelShape   kernel_shape({ out_channels, weights_height, weights_width, in_channels });
    const Tensor4DShape in_shape(internal_get_input_shape(input));

    // Get the memory required to instantiate a new Winograd operator.
    constexpr size_t kstore_alignment          = 64;
    const size_t     kernel_storage_per_thread = NEWinogradLayerKernel::get_kernel_storage_size(kernel_shape);
    _kernel_storage.allocator()->init(TensorInfo(TensorShape{ (kernel_storage_per_thread + kstore_alignment - 1) }, 1, DataType::U8));
    _memory_group.manage(&_kernel_storage);

    // Get workbench size and allocate memory
    constexpr size_t wspace_alignment = 64;
    const size_t     ws_size          = NEWinogradLayerKernel::get_working_space_size(in_shape, kernel_shape, padding);
    _workspace.allocator()->init(TensorInfo(TensorShape{ (ws_size + wspace_alignment - 1) }, 1, DataType::U8));
    _memory_group.manage(&_workspace);

    // Workspace for weights transform
    const size_t weights_transform_size = NEWinogradLayerKernel::get_kernel_transform_working_size(kernel_shape);
    _weights_workspace.allocator()->init(TensorInfo(TensorShape{ (weights_transform_size + wspace_alignment - 1) }, 1, DataType::U8));
    _memory_group.manage(&_weights_workspace);

    _kernel_storage.allocator()->allocate();
    _workspace.allocator()->allocate();
    _weights_workspace.allocator()->allocate();

    // Create Winograd operator object
    _conv = support::cpp14::make_unique<Winograd3x3F32>(kernel_shape, in_shape, padding, _kernel_storage.buffer());

    // Configure the kernel, padding not needed so it's safe to call configure after allocare
    _winograd_kernel.configure(output, _conv.get());
}

void NEWinogradLayer::run()
{
#if defined(__aarch64__)
    _memory_group.acquire();
    if(!_reshaped_kernel)
    {
        _conv->transform_weights(reinterpret_cast<const float *>(_weights->buffer()), reinterpret_cast<float *>(_weights_workspace.buffer()));
        _reshaped_kernel = true;
    }
    const Tensor4DShape in_shape(internal_get_input_shape(_input));
    auto                padding = PADDING_VALID;

    //Bring channels to the front as Winograd code expects the tensor to be in the format NHWC
    _conv->nchw2nhwc(in_shape, padding, _workspace.buffer(), reinterpret_cast<const float *>(_input->buffer()));

    //Get ptrs into the workspace
    std::pair<void *, void *> nhwc_ptrs = _conv->get_nhwc_ptrs(in_shape, padding, _workspace.buffer());

    //Setup matrices ptrs and transfor the input tensor to the appropriate form before running GEMM.
    _conv->reshape_input(in_shape, padding, nhwc_ptrs.second, _workspace.buffer());

    //Run 16 GEMMs in multiple threads, each kernel runs one or more GEMMs
    NEScheduler::get().schedule(&_winograd_kernel, Window::DimY);

    //Transform the output to the appropriate form
    _conv->reshape_output(in_shape, padding, nhwc_ptrs.first);

    //Transform back to NCHW
    _conv->nhwc2nchw(in_shape, padding, _workspace.buffer(), reinterpret_cast<float *>(_output->buffer()));

    _memory_group.release();
#else  /* __aarch64__ */
    ARM_COMPUTE_UNUSED(_winograd_kernel);
    ARM_COMPUTE_UNUSED(_workspace);
    ARM_COMPUTE_UNUSED(_kernel_storage);
    ARM_COMPUTE_UNUSED(_input);
    ARM_COMPUTE_UNUSED(_weights);
    ARM_COMPUTE_UNUSED(_output);
    ARM_COMPUTE_UNUSED(_reshaped_kernel);
    ARM_COMPUTE_UNUSED(_conv);
    ARM_COMPUTE_ERROR("Winograd only supported for aarch64, recompile with arch=arm64-v8a.");
#endif /* __aarch64__ */
}
} // namespace arm_compute
