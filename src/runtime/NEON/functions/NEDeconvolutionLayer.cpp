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
#include "arm_compute/runtime/NEON/functions/NEDeconvolutionLayer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

NEDeconvolutionLayer::NEDeconvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager) // NOLINT
    : _memory_group(std::move(memory_manager)),
      _conv_f(),
      _upsample_f(),
      _flip_weights(),
      _scaled_output(),
      _weights_flipped(),
      _original_weights(nullptr),
      _input(nullptr),
      _info(),
      _inner_border(),
      _is_prepared(false)
{
}

Status NEDeconvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *bias, const ITensorInfo *output, const PadStrideInfo &info,
                                      unsigned int inner_border_right, unsigned int inner_border_top)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(0) != weights->dimension(1));
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(0) < 1);
    ARM_COMPUTE_RETURN_ERROR_ON(!info.padding_is_symmetric());

    const unsigned int stride_x = info.stride().first;
    const unsigned int stride_y = info.stride().second;

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(inner_border_right > stride_x - 1, "inner_border_right must be smaller than stride_x");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(inner_border_top > stride_y - 1, "inner_border_top must be smaller than stride_y");

    auto out_dims = deconvolution_output_dimensions(input->dimension(0), input->dimension(1), weights->dimension(0), weights->dimension(1),
                                                    info.pad().first, info.pad().second, stride_x, stride_y);

    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);

    if(bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, bias);
    }

    if(output->tensor_shape().total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

        const TensorShape output_shape = compute_deconvolution_output_shape(out_dims, *input, *weights);

        ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->dimension(Window::DimX) != output_shape.x(), "Output's width is invalid.");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->dimension(Window::DimY) != output_shape.y(), "Output's height is invalid.");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->dimension(Window::DimZ) != output_shape.z(), "Output's depth is invalid.");
    }

    unsigned int        padx            = 0;
    unsigned int        pady            = 0;
    const TensorShape   scale_out_shape = compute_deconvolution_upsampled_shape(*input, *weights, stride_x, stride_y, inner_border_right, inner_border_top, out_dims, padx, pady);
    TensorInfo          scale_out_info(input->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(scale_out_shape));
    const PadStrideInfo conv_info(1, 1, 0, 0, 0, 0, DimensionRoundingType::CEIL);

    for(size_t i = 2; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(i) != scale_out_info.dimension(i));
    }

    ARM_COMPUTE_RETURN_ON_ERROR(NEConvolutionLayer::validate(&scale_out_info, weights, bias, output, conv_info, WeightsInfo()));

    return Status{};
}

void NEDeconvolutionLayer::configure(ITensor *input, const ITensor *weights, const ITensor *bias, ITensor *output, const PadStrideInfo &info,
                                     unsigned int inner_border_right, unsigned int inner_border_top)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);

    _input            = input;
    _original_weights = weights;
    _info             = info;
    _inner_border     = std::make_pair(inner_border_right, inner_border_top);
    _is_prepared      = false;

    const unsigned int stride_x = info.stride().first;
    const unsigned int stride_y = info.stride().second;

    _weights_flipped.allocator()->init(TensorInfo(weights->info()->tensor_shape(), 1, weights->info()->data_type()));
    _flip_weights.configure(weights, &_weights_flipped);

    auto out_dims = deconvolution_output_dimensions(input->info()->dimension(0), input->info()->dimension(1), weights->info()->dimension(0), weights->info()->dimension(1),
                                                    info.pad().first, info.pad().second, stride_x, stride_y);

    const TensorShape output_shape = compute_deconvolution_output_shape(out_dims, *input->info(), *weights->info());
    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type(), input->info()->quantization_info());

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(NEDeconvolutionLayer::validate(input->info(), weights->info(), bias == nullptr ? nullptr : bias->info(), output->info(), info, inner_border_right, inner_border_top));

    _memory_group.manage(&_scaled_output);

    // Find the upsampled dimensions and the padding needed for the convolution with stride 1 in order to match output shape
    unsigned int      padx            = 0;
    unsigned int      pady            = 0;
    const TensorShape scale_out_shape = compute_deconvolution_upsampled_shape(*input->info(), *weights->info(), stride_x, stride_y, inner_border_right, inner_border_top, out_dims, padx, pady);

    TensorInfo scale_out_info(scale_out_shape, 1, input->info()->data_type(), input->info()->quantization_info());
    _scaled_output.allocator()->init(scale_out_info);

    const PadStrideInfo upsample_info(stride_x, stride_y, padx / 2, pady / 2);
    _upsample_f.configure(input, &_scaled_output, upsample_info, inner_border_right, inner_border_top);

    // setup the function to convolve the upscaled output
    const PadStrideInfo conv_info(1, 1, 0, 0, 0, 0, DimensionRoundingType::CEIL);
    _conv_f.configure(&_scaled_output, &_weights_flipped, bias, output, conv_info);
    _scaled_output.allocator()->allocate();
}
Status NEDeconvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *bias, const ITensorInfo *output, const PadStrideInfo &info)
{
    return NEDeconvolutionLayer::validate(input, weights, bias, output, info, 0, 0);
}

void NEDeconvolutionLayer::configure(ITensor *input, const ITensor *weights, const ITensor *bias, ITensor *output, const PadStrideInfo &info)
{
    configure(input, weights, bias, output, info, 0, 0);
}

void NEDeconvolutionLayer::run()
{
    prepare();

    _memory_group.acquire();

    _upsample_f.run();
    _conv_f.run();

    _memory_group.release();
}

void NEDeconvolutionLayer::prepare()
{
    if(!_is_prepared)
    {
        ARM_COMPUTE_ERROR_ON(!_original_weights->is_used());

        // Run weights flipping and mark original weights tensor as unused
        _weights_flipped.allocator()->allocate();
        NEScheduler::get().schedule(&_flip_weights, Window::DimZ);
        _original_weights->mark_as_unused();

        // Prepare convolution
        _conv_f.prepare();

        if(!_weights_flipped.is_used())
        {
            _weights_flipped.allocator()->free();
        }

        _is_prepared = true;
    }
}
