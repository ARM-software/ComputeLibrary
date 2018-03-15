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
    : _memory_group(memory_manager), _batched_mm(memory_manager), _input_transform(), _filter_transform(), _output_transform(), _input0(), _input1(), _batched_mm_output(), _is_first_run(true)
{
}

void CLWinogradConvolutionLayer::configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info)
{
    // TODO(COMPMID-1013): This part will be removed
    // Get indeces for the width and height
    const size_t idx_width  = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_height = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::HEIGHT);

    // Kernel size
    const unsigned int kernel_w = weights->info()->tensor_shape()[idx_width];
    const unsigned int kernel_h = weights->info()->tensor_shape()[idx_height];

    // Number of tiles along the X and Y direction
    const unsigned int num_tiles_x = std::ceil((input->info()->tensor_shape().x() - (kernel_w - 1) + conv_info.pad_left() + conv_info.pad_right()) / 2.f);
    const unsigned int num_tiles_y = std::ceil((input->info()->tensor_shape().y() - (kernel_h - 1) + conv_info.pad_top() + conv_info.pad_bottom()) / 2.f);

    // Compute output shape
    const TensorShape output_convolved_shape = misc::shape_calculator::compute_deep_convolution_shape(*input->info(), *weights->info(), conv_info);

    // Manage intermediate tensors
    _memory_group.manage(&_input0);
    _memory_group.manage(&_batched_mm_output);

    // Do not manage _input1 as it contains the weights

    // Configure input transform
    _input_transform.configure(input, &_input0, conv_info, Size2D(kernel_w, kernel_h));

    // Configure filter transform
    _filter_transform.configure(weights, &_input1, Size2D(2U, 2U));

    // Configure batched matrix multiply
    _batched_mm.configure(&_input0, &_input1, nullptr, &_batched_mm_output, 1.0f, 0.0f, GEMMInfo(false, false, true /* Reshape weights only for the first run*/));

    // Configure output transform
    _output_transform.configure(&_batched_mm_output, biases, output, Size2D(kernel_w, kernel_h), Size2D(output_convolved_shape[idx_width], output_convolved_shape[idx_height]), Size2D(num_tiles_x,
                                num_tiles_y));

    // Allocate temporary tensors
    _input0.allocator()->allocate();
    _input1.allocator()->allocate();
    _batched_mm_output.allocator()->allocate();
}

Status CLWinogradConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info)
{
    // TODO(COMPMID-1013): This part will be removed
    // Get indeces for the width and height
    const size_t idx_width  = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_height = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT);

    // Kernel size
    const unsigned int kernel_w = weights->tensor_shape()[idx_width];
    const unsigned int kernel_h = weights->tensor_shape()[idx_height];

    // Number of tiles along the X and Y direction
    const unsigned int num_tiles_x = std::ceil((input->tensor_shape().x() - (kernel_w - 1) + conv_info.pad_left() + conv_info.pad_right()) / 2.f);
    const unsigned int num_tiles_y = std::ceil((input->tensor_shape().y() - (kernel_h - 1) + conv_info.pad_top() + conv_info.pad_bottom()) / 2.f);

    // Compute output shape
    const TensorShape output_convolved_shape = misc::shape_calculator::compute_deep_convolution_shape(*input, *weights, conv_info);

    // Validate input transform
    const TensorShape input0_shape = misc::shape_calculator::compute_winograd_input_transform_shape(*input, conv_info, Size2D(kernel_w, kernel_h));
    const TensorInfo  input0       = input->clone()->set_tensor_shape(input0_shape);
    ARM_COMPUTE_RETURN_ON_ERROR(CLWinogradInputTransform::validate(input, &input0, conv_info, Size2D(kernel_w, kernel_h)));

    // Validate filter transform
    const TensorShape input1_shape = misc::shape_calculator::compute_winograd_filter_transform_shape(*weights, Size2D(2U, 2U));
    const TensorInfo  input1       = weights->clone()->set_tensor_shape(input1_shape);
    ARM_COMPUTE_RETURN_ON_ERROR(CLWinogradFilterTransformKernel::validate(weights, &input1, Size2D(2U, 2U)));

    // Configure batched matrix multiply
    TensorShape batched_mm_output_shape = input0.tensor_shape();
    batched_mm_output_shape[0]          = input1.tensor_shape()[0];
    const TensorInfo batched_mm_output  = input0.clone()->set_tensor_shape(batched_mm_output_shape);
    ARM_COMPUTE_RETURN_ON_ERROR(CLGEMM::validate(&input0, &input1, nullptr, &batched_mm_output, 1.0f, 0.0f, GEMMInfo(false, false, true /* Reshape weights only for the first run*/)));

    // Configure output transform
    ARM_COMPUTE_RETURN_ON_ERROR(CLWinogradOutputTransformKernel::validate(&batched_mm_output, biases, output, Size2D(kernel_w, kernel_h), Size2D(output_convolved_shape[idx_width],
                                                                          output_convolved_shape[idx_height]),
                                                                          Size2D(num_tiles_x, num_tiles_y)));

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

    _memory_group.release();
}
