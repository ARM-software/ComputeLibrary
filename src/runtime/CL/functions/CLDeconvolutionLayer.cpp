/*
 * Copyright (c) 2017, 2018 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLDeconvolutionLayer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include <memory>
#include <tuple>

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

CLDeconvolutionLayer::CLDeconvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager) // NOLINT
    : _memory_group(std::move(memory_manager)),
      _scale_f(),
      _conv_f(),
      _scaled_output()
{
}

Status CLDeconvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *bias, ITensorInfo *output, const PadStrideInfo &info,
                                      unsigned int inner_border_right, unsigned int inner_border_top)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(0) != weights->dimension(1));
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(0) < 1);

    const unsigned int stride_x = info.stride().first;
    const unsigned int stride_y = info.stride().second;

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(inner_border_right > stride_x - 1, "inner_border_right must be smaller than stride_x");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(inner_border_top > stride_y - 1, "inner_border_top must be smaller than stride_y");

    auto out_dims = deconvolution_output_dimensions(input->dimension(0), input->dimension(1), weights->dimension(0), weights->dimension(1),
                                                    info.pad().first, info.pad().second, inner_border_right, inner_border_top, stride_x, stride_y);

    const TensorShape output_shape = deconvolution_output_shape(out_dims, input->tensor_shape(), weights->tensor_shape());

    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output, weights, bias);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(input, output, weights, bias);

    if(bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, bias);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(input, bias);
    }

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->dimension(Window::DimX) != output_shape.x(), "Output's width is invalid.");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->dimension(Window::DimY) != output_shape.y(), "Output's height is invalid.");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->dimension(Window::DimZ) != output_shape.z(), "Output's depth is invalid.");

    TensorInfo scale_out_info(input->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(compute_deconvolution_shape(*input, stride_x, stride_y, inner_border_right, inner_border_top,
                                                                                                      info)));
    const PadStrideInfo conv_info(1, 1, 0, 0, 0, 0, DimensionRoundingType::CEIL);

    ARM_COMPUTE_RETURN_ON_ERROR(CLDeconvolutionLayerUpsample::validate(input, &scale_out_info, BorderSize(inner_border_right, inner_border_top), info));
    ARM_COMPUTE_RETURN_ON_ERROR(CLDirectConvolutionLayer::validate(&scale_out_info, weights, bias, output, conv_info));

    return Status{};
}

void CLDeconvolutionLayer::configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *bias, ICLTensor *output, const PadStrideInfo &info,
                                     unsigned int inner_border_right, unsigned int inner_border_top)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);

    const unsigned int stride_x = info.stride().first;
    const unsigned int stride_y = info.stride().second;

    auto out_dims = deconvolution_output_dimensions(input->info()->dimension(0), input->info()->dimension(1), weights->info()->dimension(0), weights->info()->dimension(1),
                                                    info.pad().first, info.pad().second, inner_border_top, inner_border_right, stride_x, stride_y);

    const TensorShape output_shape = deconvolution_output_shape(out_dims, input->info()->tensor_shape(), weights->info()->tensor_shape());

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type(), input->info()->fixed_point_position());

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(CLDeconvolutionLayer::validate(input->info(), weights->info(), bias == nullptr ? nullptr : bias->info(), output->info(), info, inner_border_right, inner_border_top));

    _memory_group.manage(&_scaled_output);

    // configure scale function
    // Init and allocate intermmidiate tensor for output, same size as input but the first two axis are the same as the output tensor
    TensorShape        scale_out_shape(input->info()->tensor_shape());
    const unsigned int out_x = input->info()->dimension(0) + (input->info()->dimension(0) - 1) * (stride_x - 1) + inner_border_right + 2 * info.pad().first;
    const unsigned int out_y = input->info()->dimension(1) + (input->info()->dimension(1) - 1) * (stride_y - 1) + inner_border_top + 2 * info.pad().second;
    scale_out_shape.set(0, out_x);
    scale_out_shape.set(1, out_y);
    TensorInfo scale_out_info(scale_out_shape, 1, input->info()->data_type(), input->info()->fixed_point_position());
    _scaled_output.allocator()->init(scale_out_info);

    _scale_f.configure(input, &_scaled_output, BorderSize(inner_border_top, inner_border_right), info);

    // setup the function to convolve the upscaled output
    const PadStrideInfo conv_info(1, 1, 0, 0, 0, 0, DimensionRoundingType::CEIL);
    _conv_f.configure(&_scaled_output, weights, bias, output, conv_info);
    _scaled_output.allocator()->allocate();
}

void CLDeconvolutionLayer::run()
{
    _memory_group.acquire();
    _scale_f.run();
    _conv_f.run();
    _memory_group.release();
}
