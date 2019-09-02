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

using namespace arm_compute::misc::shape_calculator;

namespace arm_compute
{
NEDeconvolutionLayer::NEDeconvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager) // NOLINT
    : _memory_group(std::move(memory_manager)),
      _conv_f(),
      _upsample_f(),
      _flip_weights(),
      _permute_input(),
      _permute_weights(),
      _permute_output(),
      _scaled_output(),
      _weights_flipped(),
      _permuted_input(),
      _permuted_weights(),
      _permuted_output(),
      _is_nchw(false),
      _original_weights(nullptr),
      _input(nullptr),
      _info(),
      _is_prepared(false)
{
}

Status NEDeconvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *bias, const ITensorInfo *output, const PadStrideInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32, DataType::F16, DataType::QASYMM8);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, input);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(weights, input);
    const unsigned int width_idx  = get_data_layout_dimension_index(weights->data_layout(), DataLayoutDimension::WIDTH);
    const unsigned int height_idx = get_data_layout_dimension_index(weights->data_layout(), DataLayoutDimension::HEIGHT);
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(width_idx) != weights->dimension(height_idx));
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(width_idx) < 1);
    ARM_COMPUTE_RETURN_ERROR_ON(!info.padding_is_symmetric());

    const unsigned int stride_x = info.stride().first;
    const unsigned int stride_y = info.stride().second;

    auto out_dims = deconvolution_output_dimensions(input->dimension(width_idx), input->dimension(height_idx), weights->dimension(width_idx), weights->dimension(height_idx),
                                                    info.pad().first, info.pad().second, stride_x, stride_y);

    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    if(bias != nullptr)
    {
        if(is_data_type_quantized_asymmetric(input->data_type()))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(bias, 1, DataType::S32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, bias);
        }
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
    const TensorShape   scale_out_shape = compute_deconvolution_upsampled_shape(*input, *weights, stride_x, stride_y, out_dims, padx, pady);
    TensorInfo          scale_out_info(input->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(scale_out_shape));
    const PadStrideInfo conv_info(1, 1, 0, 0, 0, 0, DimensionRoundingType::CEIL);

    const unsigned int batches_idx = get_data_layout_dimension_index(weights->data_layout(), DataLayoutDimension::BATCHES);
    const unsigned int channel_idx = get_data_layout_dimension_index(weights->data_layout(), DataLayoutDimension::CHANNEL);
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(batches_idx) != scale_out_info.dimension(batches_idx));
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(channel_idx) != scale_out_info.dimension(channel_idx));

    ARM_COMPUTE_RETURN_ON_ERROR(NEConvolutionLayer::validate(&scale_out_info, weights, bias, output, conv_info, WeightsInfo()));

    return Status{};
}

void NEDeconvolutionLayer::configure(ITensor *input, const ITensor *weights, const ITensor *bias, ITensor *output, const PadStrideInfo &info)
{
    // Perform validation step
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(NEDeconvolutionLayer::validate(input->info(), weights->info(), (bias == nullptr) ? nullptr : bias->info(), output->info(), info));

    const DataLayout data_layout = input->info()->data_layout();

    _input            = input;
    _original_weights = weights;
    _info             = info;
    _is_prepared      = false;
    _is_nchw          = data_layout == DataLayout::NCHW;

    const unsigned int stride_x = info.stride().first;
    const unsigned int stride_y = info.stride().second;

    const unsigned int width_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int height_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    auto               out_dims   = deconvolution_output_dimensions(input->info()->dimension(width_idx), input->info()->dimension(height_idx), weights->info()->dimension(width_idx),
                                                                    weights->info()->dimension(height_idx),
                                                                    info.pad().first, info.pad().second, stride_x, stride_y);

    const TensorShape output_shape = compute_deconvolution_output_shape(out_dims, *input->info(), *weights->info());
    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type(), input->info()->quantization_info());

    _memory_group.manage(&_scaled_output);

    if(!_is_nchw)
    {
        _memory_group.manage(&_permuted_input);
        _memory_group.manage(&_permuted_output);

        // Configure the function to transform the input tensor from NHWC -> NCHW
        _permuted_input.info()->set_quantization_info(input->info()->quantization_info());
        _permute_input.configure(input, &_permuted_input, PermutationVector(1U, 2U, 0U));
        _permuted_input.info()->set_data_layout(DataLayout::NCHW);

        // Configure the function to transform the weights tensor from NHWC -> NCHW
        _permuted_weights.info()->set_quantization_info(weights->info()->quantization_info());
        _permute_weights.configure(weights, &_permuted_weights, PermutationVector(1U, 2U, 0U));
        _permuted_weights.info()->set_data_layout(DataLayout::NCHW);

        // Find the upsampled dimensions and the padding needed for the convolution with stride 1 in order to match output shape
        unsigned int      padx            = 0;
        unsigned int      pady            = 0;
        const TensorShape scale_out_shape = compute_deconvolution_upsampled_shape(*_permuted_input.info(), *_permuted_weights.info(), stride_x, stride_y, out_dims, padx,
                                                                                  pady);

        TensorInfo scale_out_info(scale_out_shape, 1, _permuted_input.info()->data_type(), _permuted_input.info()->quantization_info());
        scale_out_info.set_data_layout(DataLayout::NCHW);
        _scaled_output.allocator()->init(scale_out_info);

        const PadStrideInfo upsample_info(stride_x, stride_y, padx / 2, pady / 2);
        _upsample_f.configure(&_permuted_input, &_scaled_output, upsample_info);

        _weights_flipped.allocator()->init(*_permuted_weights.info()->clone());
        _weights_flipped.info()->set_quantization_info(weights->info()->quantization_info());
        _flip_weights.configure(&_permuted_weights, &_weights_flipped);

        // setup the function to convolve the upscaled output
        const PadStrideInfo conv_info(1, 1, 0, 0, 0, 0, DimensionRoundingType::CEIL);

        _permuted_output.info()->set_quantization_info(output->info()->quantization_info());
        _conv_f.configure(&_scaled_output, &_weights_flipped, bias, &_permuted_output, conv_info);

        // Configure the function to transform the convoluted output to NHWC
        _permute_output.configure(&_permuted_output, output, PermutationVector(2U, 0U, 1U));
        _permuted_output.info()->set_data_layout(DataLayout::NCHW);

        _permuted_input.allocator()->allocate();
        _permuted_output.allocator()->allocate();
    }
    else
    {
        // Find the upsampled dimensions and the padding needed for the convolution with stride 1 in order to match output shape
        unsigned int      padx            = 0;
        unsigned int      pady            = 0;
        const TensorShape scale_out_shape = compute_deconvolution_upsampled_shape(*input->info(), *weights->info(), stride_x, stride_y, out_dims, padx, pady);

        TensorInfo scale_out_info(scale_out_shape, 1, input->info()->data_type(), input->info()->quantization_info());
        scale_out_info.set_data_layout(data_layout);
        _scaled_output.allocator()->init(scale_out_info);
        const PadStrideInfo upsample_info(stride_x, stride_y, padx / 2, pady / 2);
        _upsample_f.configure(input, &_scaled_output, upsample_info);

        _weights_flipped.allocator()->init(weights->info()->clone()->set_data_layout(data_layout));
        _flip_weights.configure(weights, &_weights_flipped);

        // setup the function to convolve the upscaled output
        const PadStrideInfo conv_info(1, 1, 0, 0, 0, 0, DimensionRoundingType::CEIL);
        _conv_f.configure(&_scaled_output, &_weights_flipped, bias, output, conv_info);
    }
    _scaled_output.allocator()->allocate();
}

void NEDeconvolutionLayer::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_memory_group);

    // Permute input
    if(!_is_nchw)
    {
        _permute_input.run();
    }

    _upsample_f.run();
    _conv_f.run();

    // Permute output
    if(!_is_nchw)
    {
        _permute_output.run();
    }
}

void NEDeconvolutionLayer::prepare()
{
    if(!_is_prepared)
    {
        ARM_COMPUTE_ERROR_ON(!_original_weights->is_used());
        // Permute weights
        if(!_is_nchw)
        {
            // Manually manage _permuted_weights
            _permuted_weights.allocator()->allocate();
            _permute_weights.run();
        }

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

        if(!_is_nchw)
        {
            // Manually manage _permuted_weights
            // Free _permuted_weights as it not used after this method (prepare)
            _permuted_weights.allocator()->free();
        }

        _is_prepared = true;
    }
}
} // namespace arm_compute
