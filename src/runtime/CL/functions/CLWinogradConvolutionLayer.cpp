/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLWinogradConvolutionLayer.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

using namespace arm_compute;

CLWinogradConvolutionLayer::CLWinogradConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(memory_manager), _batched_mm(memory_manager), _input_transform(), _filter_transform(), _output_transform(), _activationlayer_function(), _input0(), _input1(), _batched_mm_output(),
      _is_first_run(true), _is_activationlayer_enabled(false)
{
}

void CLWinogradConvolutionLayer::configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info)
{
    // Get indices for the width and height
    const size_t idx_width  = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_height = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::HEIGHT);

    // Input shape
    const TensorShape input_shape = input->info()->tensor_shape();

    // Kernel size
    const unsigned int kernel_w = weights->info()->tensor_shape()[idx_width];
    const unsigned int kernel_h = weights->info()->tensor_shape()[idx_height];

    const WinogradInfo winograd_info = WinogradInfo(Size2D(2, 2),
                                                    Size2D(kernel_w, kernel_h),
                                                    Size2D(input_shape[idx_width], input_shape[idx_height]),
                                                    conv_info,
                                                    input->info()->data_layout());

    // Manage intermediate tensors
    _memory_group.manage(&_input0);
    _memory_group.manage(&_batched_mm_output);

    // Do not manage _input1 as it contains the weights

    // Configure input transform
    _input_transform.configure(input, &_input0, winograd_info);

    // Configure filter transform
    _filter_transform.configure(weights, &_input1, winograd_info);

    // Configure batched matrix multiply
    _batched_mm.configure(&_input0, &_input1, nullptr, &_batched_mm_output, 1.0f, 0.0f, GEMMInfo(false, false, true /* Reshape weights only for the first run*/));

    // Configure output transform
    _output_transform.configure(&_batched_mm_output, biases, output, winograd_info);

    // Configure activation layer
    _is_activationlayer_enabled = act_info.enabled();
    if(_is_activationlayer_enabled)
    {
        _activationlayer_function.configure(output, nullptr, act_info);
    }

    // Allocate temporary tensors
    _input0.allocator()->allocate();
    _input1.allocator()->allocate();
    _batched_mm_output.allocator()->allocate();
}

Status CLWinogradConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                            const ActivationLayerInfo &act_info)
{
    // Get indeces for the width and height
    const size_t idx_width  = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_height = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT);

    // Input shape
    const TensorShape input_shape = input->tensor_shape();

    // Kernel size
    const unsigned int kernel_w = weights->tensor_shape()[idx_width];
    const unsigned int kernel_h = weights->tensor_shape()[idx_height];

    const WinogradInfo winograd_info = WinogradInfo(Size2D(2, 2),
                                                    Size2D(kernel_w, kernel_h),
                                                    Size2D(input_shape[idx_width], input_shape[idx_height]),
                                                    conv_info,
                                                    input->data_layout());

    // Validate input transform
    const TensorShape input0_shape = misc::shape_calculator::compute_winograd_input_transform_shape(*input, winograd_info);
    const TensorInfo  input0       = input->clone()->set_tensor_shape(input0_shape);
    ARM_COMPUTE_RETURN_ON_ERROR(CLWinogradInputTransform::validate(input, &input0, winograd_info));

    // Validate filter transform
    const TensorShape input1_shape = misc::shape_calculator::compute_winograd_filter_transform_shape(*weights, winograd_info);
    const TensorInfo  input1       = weights->clone()->set_tensor_shape(input1_shape);
    ARM_COMPUTE_RETURN_ON_ERROR(CLWinogradFilterTransformKernel::validate(weights, &input1, winograd_info));

    // Validate batched matrix multiply
    TensorShape batched_mm_output_shape = input0.tensor_shape();
    batched_mm_output_shape[0]          = input1.tensor_shape()[0];
    const TensorInfo batched_mm_output  = input0.clone()->set_tensor_shape(batched_mm_output_shape);
    ARM_COMPUTE_RETURN_ON_ERROR(CLGEMM::validate(&input0, &input1, nullptr, &batched_mm_output, 1.0f, 0.0f, GEMMInfo(false, false, true /* Reshape weights only for the first run*/)));

    // Configure output transform
    ARM_COMPUTE_RETURN_ON_ERROR(CLWinogradOutputTransformKernel::validate(&batched_mm_output, biases, output, winograd_info));

    // Validate Activation Layer
    if(act_info.enabled())
    {
        CLActivationLayer::validate(output, nullptr, act_info);
    }

    return Status{};
}

void CLWinogradConvolutionLayer::run()
{
    if(_is_first_run)
    {
        // Run filter transform
        CLScheduler::get().enqueue(_filter_transform, false);

        _is_first_run = false;
    }

    _memory_group.acquire();

    // Run input transform
    _input_transform.run();

    // Run batched matrix multiplication
    _batched_mm.run();

    // Run output transform
    CLScheduler::get().enqueue(_output_transform);

    if(_is_activationlayer_enabled)
    {
        _activationlayer_function.run();
    }

    _memory_group.release();
}
