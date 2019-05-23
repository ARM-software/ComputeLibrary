/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLDirectDeconvolutionLayer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include <memory>
#include <tuple>

namespace arm_compute
{
using namespace arm_compute::misc::shape_calculator;

CLDirectDeconvolutionLayer::CLDirectDeconvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager) // NOLINT
    : _memory_group(std::move(memory_manager)),
      _scale_f(),
      _conv_f(),
      _flip_weights(),
      _scaled_output(),
      _original_weights(nullptr),
      _weights_flipped(),
      _flip_axis(),
      _is_prepared(false)
{
}

Status CLDirectDeconvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *bias, ITensorInfo *output, const PadStrideInfo &info,
                                            const WeightsInfo &weights_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, weights);

    const DataLayout data_layout = input->data_layout();

    const size_t idx_w = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const size_t idx_h = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const size_t idx_c = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(idx_w) != weights->dimension(idx_h));
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(idx_w) < 1);
    ARM_COMPUTE_RETURN_ERROR_ON(!info.padding_is_symmetric());

    const unsigned int stride_x = info.stride().first;
    const unsigned int stride_y = info.stride().second;

    auto out_dims = deconvolution_output_dimensions(input->dimension(idx_w), input->dimension(idx_h), weights->dimension(idx_w), weights->dimension(idx_h),
                                                    info.pad().first, info.pad().second, stride_x, stride_y);

    const TensorShape output_shape = compute_deconvolution_output_shape(out_dims, *input, *weights);

    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output, weights);

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
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, bias);
    }

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->dimension(idx_w) != output_shape[idx_w], "Output's width is invalid.");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->dimension(idx_h) != output_shape[idx_h], "Output's height is invalid.");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->dimension(idx_c) != output_shape[idx_c], "Output's depth is invalid.");

    unsigned int        padx            = 0;
    unsigned int        pady            = 0;
    const TensorShape   scale_out_shape = compute_deconvolution_upsampled_shape(*input, *weights, stride_x, stride_y, 0, 0, out_dims, padx, pady);
    TensorInfo          scale_out_info(input->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(scale_out_shape).set_data_layout(data_layout));
    const PadStrideInfo conv_info(1, 1, 0, 0, 0, 0, DimensionRoundingType::CEIL);

    ARM_COMPUTE_RETURN_ON_ERROR(CLDeconvolutionLayerUpsample::validate(input, &scale_out_info, BorderSize(), info));
    ARM_COMPUTE_RETURN_ON_ERROR(CLConvolutionLayer::validate(&scale_out_info, weights, bias, output, conv_info, weights_info));

    return Status{};
}

void CLDirectDeconvolutionLayer::configure(ICLTensor *input, ICLTensor *weights, const ICLTensor *bias, ICLTensor *output, const PadStrideInfo &info,
                                           const WeightsInfo &weights_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);

    const unsigned int stride_x = info.stride().first;
    const unsigned int stride_y = info.stride().second;

    const DataLayout data_layout = input->info()->data_layout();

    const size_t idx_w = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const size_t idx_h = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    _original_weights = weights;
    _flip_axis.allocator()->init(TensorInfo(TensorShape(2U), 1, DataType::U32));
    _weights_flipped.allocator()->init(weights->info()->clone()->set_data_layout(data_layout));
    _flip_weights.configure(weights, &_weights_flipped, &_flip_axis);

    auto out_dims = deconvolution_output_dimensions(input->info()->dimension(idx_w), input->info()->dimension(idx_h), weights->info()->dimension(idx_w), weights->info()->dimension(idx_h),
                                                    info.pad().first, info.pad().second, stride_x, stride_y);

    const TensorShape output_shape = compute_deconvolution_output_shape(out_dims, *input->info(), *weights->info());

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(output_shape).set_data_layout(data_layout));

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(CLDirectDeconvolutionLayer::validate(input->info(), weights->info(), bias == nullptr ? nullptr : bias->info(), output->info(), info));

    _is_prepared = weights_info.retain_internal_weights();

    _memory_group.manage(&_scaled_output);

    // Find the upsampled dimensions and the padding needed for the convolution with stride 1 in order to match output shape
    unsigned int      padx            = 0;
    unsigned int      pady            = 0;
    const TensorShape scale_out_shape = compute_deconvolution_upsampled_shape(*input->info(), *weights->info(), stride_x, stride_y, 0, 0, out_dims, padx, pady);

    TensorInfo scale_out_info(scale_out_shape, 1, input->info()->data_type(), input->info()->quantization_info());
    scale_out_info.set_data_layout(data_layout);
    _scaled_output.allocator()->init(scale_out_info);

    // configure scale function
    const PadStrideInfo upsample_info(stride_x, stride_y, padx / 2, pady / 2);
    _scale_f.configure(input, &_scaled_output, BorderSize(), upsample_info);

    // Setup the function to convolve the upscaled output
    const PadStrideInfo conv_info(1, 1, 0, 0, 0, 0, DimensionRoundingType::CEIL);
    _conv_f.configure(&_scaled_output, &_weights_flipped, bias, output, conv_info, weights_info);
    _scaled_output.allocator()->allocate();

    // Setup flip axis data
    _flip_axis.allocator()->allocate();
    _flip_axis.map(true);
    auto axis_data = reinterpret_cast<uint32_t *>(_flip_axis.buffer());
    if(weights->info()->data_layout() == DataLayout::NHWC)
    {
        axis_data[0] = 1;
        axis_data[1] = 2;
    }
    else
    {
        axis_data[0] = 0;
        axis_data[1] = 1;
    }
    _flip_axis.unmap();
}

void CLDirectDeconvolutionLayer::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_memory_group);

    _scale_f.run();
    _conv_f.run();
}

void CLDirectDeconvolutionLayer::prepare()
{
    if(!_is_prepared)
    {
        ARM_COMPUTE_ERROR_ON(!_original_weights->is_used());

        // Run weights flipping and mark original weights tensor as unused
        _weights_flipped.allocator()->allocate();
        _flip_weights.run();
        _original_weights->mark_as_unused();

        // Prepare convolution
        _conv_f.prepare();

        // Free flipped weights
        if(!_weights_flipped.is_used())
        {
            _weights_flipped.allocator()->free();
        }

        _is_prepared = true;
    }
}
} // namespace arm_compute
