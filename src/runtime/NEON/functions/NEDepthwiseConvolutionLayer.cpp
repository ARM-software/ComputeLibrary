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
#include "arm_compute/runtime/NEON/functions/NEDepthwiseConvolutionLayer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;
using namespace arm_compute::misc;
using namespace arm_compute::misc::shape_calculator;

NEDepthwiseConvolutionLayer3x3::NEDepthwiseConvolutionLayer3x3()
    : _dwc_kernel(), _output_stage_kernel(), _border_handler(), _permute_input(), _permute_weights(), _permute_output(), _accumulator(), _permuted_input(), _permuted_weights(), _permuted_output(),
      _has_bias(false), _is_quantized(false), _is_optimized(false), _are_weights_reshaped(false), _is_nchw(true), _is_first_run(true), _permute(false)
{
}

void NEDepthwiseConvolutionLayer3x3::configure(ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);

    PixelValue zero_value(0.f);

    _is_quantized = is_data_type_quantized_asymmetric(input->info()->data_type());
    _has_bias     = biases != nullptr;
    _is_optimized = NEDepthwiseConvolutionLayer3x3Kernel::is_optimized_execution_possible(input->info()->tensor_shape(),
                                                                                          conv_info,
                                                                                          input->info()->data_type(),
                                                                                          depth_multiplier,
                                                                                          input->info()->data_layout());
    _are_weights_reshaped = false;
    _is_nchw              = input->info()->data_layout() == DataLayout::NCHW;
    _permute              = _is_optimized == _is_nchw;

    if(_is_optimized)
    {
        if(_is_nchw)
        {
            // Configure the function to transform the input tensor from NCHW -> NHWC
            _permute_input.configure(input, &_permuted_input, PermutationVector(2U, 0U, 1U));
            _permuted_input.info()->set_data_layout(DataLayout::NHWC);

            // Configure the function to transform the weights tensor from IHW -> HWI
            _permute_weights.configure(weights, &_permuted_weights, PermutationVector(2U, 0U, 1U));
            _permuted_weights.info()->set_data_layout(DataLayout::NHWC);

            // Configure optimized depthwise
            _dwc_kernel.configure(&_permuted_input, &_permuted_weights, &_permuted_output, conv_info, depth_multiplier, DataLayout::NHWC);

            // Configure the function to transform the convoluted output to ACL's native ordering format NCHW
            _permute_output.configure(&_permuted_output, output, PermutationVector(1U, 2U, 0U));
            _permuted_output.info()->set_data_layout(DataLayout::NCHW);

            // Allocate tensors
            _permuted_input.allocator()->allocate();
            _permuted_weights.allocator()->allocate();
            _permuted_output.allocator()->allocate();
        }
        else
        {
            _dwc_kernel.configure(input, weights, output, conv_info, depth_multiplier, DataLayout::NHWC);
        }
    }
    else
    {
        // Allocate the intermediate accumulator tensor in case of quantized input
        if(_is_quantized)
        {
            TensorShape accum_shape = output->info()->tensor_shape();

            if(!_is_nchw)
            {
                permute(accum_shape, PermutationVector(1U, 2U, 0U));
            }

            _accumulator.allocator()->init(TensorInfo(accum_shape, 1, DataType::S32));
            _accumulator.info()->set_quantization_info(input->info()->quantization_info());
            zero_value = PixelValue(static_cast<uint32_t>(input->info()->quantization_info().offset));
        }

        if(!_is_nchw)
        {
            // Configure the function to transform the input tensor from NHWC -> NCHW
            _permute_input.configure(input, &_permuted_input, PermutationVector(1U, 2U, 0U));
            _permuted_input.info()->set_data_layout(DataLayout::NCHW);

            // Configure the function to transform the weights tensor from HWI -> IHW
            _permute_weights.configure(weights, &_permuted_weights, PermutationVector(1U, 2U, 0U));
            _permuted_weights.info()->set_data_layout(DataLayout::NCHW);

            // Configure optimized depthwise
            _dwc_kernel.configure(&_permuted_input, &_permuted_weights, (_is_quantized) ? &_accumulator : &_permuted_output, conv_info, depth_multiplier);

            // Configure border handler
            _border_handler.configure(&_permuted_input, _dwc_kernel.border_size(), BorderMode::CONSTANT, zero_value);

            // Allocate tensors
            _permuted_input.allocator()->allocate();
            _permuted_weights.allocator()->allocate();
        }
        else
        {
            // Configure depthwise convolution kernel
            _dwc_kernel.configure(input, weights, (_is_quantized) ? &_accumulator : output, conv_info, depth_multiplier);

            // Configure border handler
            _border_handler.configure(input, _dwc_kernel.border_size(), BorderMode::CONSTANT, zero_value);
        }
    }

    // Configure biases accumulation
    if(_is_quantized)
    {
        const QuantizationInfo output_quant_info = (output->info()->total_size() == 0) ? input->info()->quantization_info() : output->info()->quantization_info();

        float multiplier = input->info()->quantization_info().scale * weights->info()->quantization_info().scale / output_quant_info.scale;
        int   output_multiplier, output_shift;
        quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift);
        _output_stage_kernel.configure(&_accumulator, biases, _is_nchw ? output : &_permuted_output, output_multiplier, output_shift, output_quant_info.offset);
        _accumulator.allocator()->allocate();
    }
    else if(_has_bias)
    {
        _output_stage_kernel.configure((_is_nchw || _is_optimized) ? output : &_permuted_output, biases);
    }

    if(!_is_optimized && !_is_nchw)
    {
        // Configure the function to transform the convoluted output to NHWC
        _permute_output.configure(&_permuted_output, output, PermutationVector(2U, 0U, 1U));
        _permuted_output.allocator()->allocate();
    }
}

Status NEDepthwiseConvolutionLayer3x3::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                                unsigned int depth_multiplier)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_layout() != DataLayout::NCHW && input->data_layout() != DataLayout::NHWC);

    if(biases != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != weights->dimension(3));
    }

    return NEDepthwiseConvolutionLayer3x3Kernel::validate(input, weights, output, conv_info, depth_multiplier);
}

void NEDepthwiseConvolutionLayer3x3::run()
{
    if(_is_first_run && _is_optimized)
    {
        _is_first_run = false;
        // Create convolver (deferred)
        _dwc_kernel.generate_convolver();
    }

    // Permute weights
    if(_permute)
    {
        if(!_are_weights_reshaped)
        {
            _are_weights_reshaped = true;
            _permute_weights.run();
        }

        _permute_input.run();
    }

    // Handle input
    if(!_is_optimized)
    {
        // Fill border
        NEScheduler::get().schedule(&_border_handler, Window::DimX);
    }

    // Execute depthwise convolution
    NEScheduler::get().schedule(&_dwc_kernel, Window::DimX);

    // Permute output
    if(_is_optimized && _is_nchw)
    {
        _permute_output.run();
    }

    // Add biases
    if(_has_bias || _is_quantized)
    {
        NEScheduler::get().schedule(&_output_stage_kernel, Window::DimX);
    }

    // Permute output
    if(!_is_optimized && !_is_nchw)
    {
        _permute_output.run();
    }
}

NEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayer()
    : _im2col_kernel(), _weights_reshape_kernel(), _v2mm_kernel(), _vector_to_tensor_kernel(), _output_stage_kernel(), _v2mm_input_fill_border(), _v2mm_weights_fill_border(), _permute_input(),
      _permute_weights(), _permute_output(), _input_reshaped(), _weights_reshaped(), _v2mm_output(), _output_reshaped(), _permuted_input(), _permuted_weights(), _permuted_output(), _is_prepared(false),
      _is_quantized(false), _is_nhwc(false), _original_weights(nullptr)
{
}

void NEDepthwiseConvolutionLayer::configure(ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier)
{
    const unsigned int channel_idx = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::CHANNEL);
    ARM_COMPUTE_UNUSED(channel_idx);

    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    ARM_COMPUTE_ERROR_ON((input->info()->dimension(channel_idx) * depth_multiplier) != weights->info()->dimension(channel_idx));

    _is_nhwc = input->info()->data_layout() == DataLayout::NHWC;

    ITensor       *input_to_use   = input;
    const ITensor *weights_to_use = weights;
    ITensor       *output_to_use  = output;

    if(_is_nhwc)
    {
        _permute_input.configure(input, &_permuted_input, PermutationVector(1U, 2U, 0U));
        _permuted_input.info()->set_data_layout(DataLayout::NCHW);
        input_to_use = &_permuted_input;

        _permute_weights.configure(weights, &_permuted_weights, PermutationVector(1U, 2U, 0U));
        _permuted_weights.info()->set_data_layout(DataLayout::NCHW);
        weights_to_use = &_permuted_weights;
    }

    const size_t weights_w = weights_to_use->info()->dimension(0);
    const size_t weights_h = weights_to_use->info()->dimension(1);
    const size_t weights_z = weights_to_use->info()->dimension(2);

    _is_quantized     = is_data_type_quantized_asymmetric(input->info()->data_type());
    _is_prepared      = false;
    _original_weights = weights_to_use;

    // Should bias be appended ?
    bool append_bias = (biases != nullptr) && !_is_quantized;

    // Calculate output shape
    TensorShape output_shape = shape_calculator::compute_depthwise_convolution_shape(*input->info(), *weights->info(), conv_info, depth_multiplier);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(output_shape));
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(output->info()->tensor_shape(), output_shape);

    if(_is_nhwc)
    {
        permute(output_shape, PermutationVector(1U, 2U, 0U));
        _permuted_output.allocator()->init(output->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(output_shape));
        _permuted_output.info()->set_data_layout(DataLayout::NCHW);
        output_to_use = &_permuted_output;
    }

    // Output width and height
    const unsigned int conv_w = output_shape.x();
    const unsigned int conv_h = output_shape.y();

    // Set up intermediate tensors
    const size_t patch_size = weights_w * weights_h + (append_bias ? 1 : 0);
    const size_t conv_size  = conv_w * conv_h;

    // Im2Col configuration
    TensorShape shape_im2col = input_to_use->info()->tensor_shape();
    shape_im2col.set(0, patch_size);
    shape_im2col.set(1, conv_size);
    shape_im2col.set(2, weights_z);
    _input_reshaped.allocator()->init(input->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(shape_im2col).set_data_layout(DataLayout::NCHW));
    _im2col_kernel.configure(input_to_use, &_input_reshaped, Size2D(weights_w, weights_h), conv_info, append_bias, depth_multiplier);

    // Weights reshape configuration
    const TensorShape shape_weights_reshape(patch_size, weights_z);
    _weights_reshaped.allocator()->init(weights->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(shape_weights_reshape).set_data_layout(DataLayout::NCHW));
    _weights_reshape_kernel.configure(weights_to_use, &_weights_reshaped, append_bias ? biases : nullptr);

    // GEMV configuration
    DataType    v2mm_dt        = (input->info()->data_type() == DataType::QASYMM8) ? DataType::S32 : input->info()->data_type();
    TensorShape shape_v2mm_out = input_to_use->info()->tensor_shape();
    shape_v2mm_out.set(0, conv_size * weights_z);
    shape_v2mm_out.set(1, 1);
    shape_v2mm_out.set(2, 1);
    _v2mm_output.allocator()->init(input->info()->clone()->set_is_resizable(true).reset_padding().set_data_type(v2mm_dt).set_tensor_shape(shape_v2mm_out).set_data_layout(DataLayout::NCHW));
    _v2mm_kernel.configure(&_input_reshaped, &_weights_reshaped, &_v2mm_output);
    _output_reshaped.allocator()->init(_v2mm_output.info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(output_shape));
    _vector_to_tensor_kernel.configure(&_v2mm_output, (_is_quantized) ? &_output_reshaped : output_to_use, conv_w, conv_h);

    // Output staged configuration
    if(_is_quantized)
    {
        const QuantizationInfo output_quant_info = output->info()->quantization_info();

        float multiplier = input->info()->quantization_info().scale * weights->info()->quantization_info().scale / output_quant_info.scale;
        int   output_multiplier, output_shift;
        quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift);
        _output_stage_kernel.configure(&_output_reshaped, biases, output_to_use, output_multiplier, output_shift, output_quant_info.offset);
        _output_reshaped.allocator()->allocate();
    }

    if(_is_nhwc)
    {
        _permute_output.configure(&_permuted_output, output, PermutationVector(2U, 0U, 1U));

        _permuted_input.allocator()->allocate();
        _permuted_weights.allocator()->allocate();
        _permuted_output.allocator()->allocate();
    }

    // Fill borders on inputs
    PixelValue zero_in(static_cast<int32_t>(0));
    PixelValue zero_w(static_cast<int32_t>(0));
    if(_is_quantized)
    {
        zero_in = PixelValue(static_cast<int32_t>(input->info()->quantization_info().offset));
        zero_w  = PixelValue(static_cast<int32_t>(weights->info()->quantization_info().offset));
    }
    BorderSize border_size = _v2mm_kernel.border_size();
    _v2mm_input_fill_border.configure(&_input_reshaped, border_size, BorderMode::CONSTANT, zero_in);

    border_size.bottom = 0;
    _v2mm_weights_fill_border.configure(&_weights_reshaped, border_size, BorderMode::CONSTANT, zero_w);

    // Allocate intermediate tensors
    _input_reshaped.allocator()->allocate();
    _v2mm_output.allocator()->allocate();
}

Status NEDepthwiseConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                             unsigned int depth_multiplier)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_layout() != DataLayout::NCHW && input->data_layout() != DataLayout::NHWC);

    // Clone output to use auto init
    auto output_clone = output->clone();

    const ITensorInfo *input_to_use   = input;
    const ITensorInfo *weights_to_use = weights;
    const ITensorInfo *output_to_use  = output_clone.get();

    TensorShape permuted_input_shape   = input->tensor_shape();
    TensorShape permuted_weights_shape = weights->tensor_shape();
    TensorInfo  permuted_input;
    TensorInfo  permuted_weights;

    if(input->data_layout() == DataLayout::NHWC)
    {
        permute(permuted_input_shape, PermutationVector(1U, 2U, 0U));
        permute(permuted_weights_shape, PermutationVector(1U, 2U, 0U));

        permuted_input   = TensorInfo(input->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(permuted_input_shape).set_data_layout(DataLayout::NCHW));
        permuted_weights = TensorInfo(weights->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(permuted_weights_shape).set_data_layout(DataLayout::NCHW));

        input_to_use   = &permuted_input;
        weights_to_use = &permuted_weights;
    }

    const bool         is_quantized = is_data_type_quantized_asymmetric(input->data_type());
    const bool         append_bias  = (biases != nullptr) && !is_quantized;
    TensorShape        output_shape = shape_calculator::compute_depthwise_convolution_shape(*input, *weights, conv_info, depth_multiplier);
    const size_t       weights_w    = weights_to_use->dimension(0);
    const size_t       weights_h    = weights_to_use->dimension(1);
    const size_t       weights_z    = weights_to_use->dimension(2);
    const unsigned int conv_w       = output_shape.x();
    const unsigned int conv_h       = output_shape.y();
    const size_t       patch_size   = weights_w * weights_h + (append_bias ? 1 : 0);
    const size_t       conv_size    = conv_w * conv_h;

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output_clone, input->clone()->set_tensor_shape(output_shape));
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), output_shape);

    TensorInfo permuted_output;
    if(input->data_layout() == DataLayout::NHWC)
    {
        permute(output_shape, PermutationVector(1U, 2U, 0U));
        permuted_output = TensorInfo(output_clone->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(output_shape).set_data_layout(DataLayout::NCHW));
        output_to_use   = &permuted_output;
    }

    // Im2Col configuration
    TensorShape shape_im2col = input_to_use->tensor_shape();
    shape_im2col.set(0, patch_size);
    shape_im2col.set(1, conv_size);
    shape_im2col.set(2, weights_z);
    TensorInfo input_reshaped(input->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(shape_im2col).set_data_layout(DataLayout::NCHW));
    ARM_COMPUTE_RETURN_ON_ERROR(NEDepthwiseIm2ColKernel::validate(input_to_use, &input_reshaped, Size2D(weights_w, weights_h), conv_info, append_bias, depth_multiplier));

    // Weights reshape configuration
    const TensorShape shape_weights_reshape(patch_size, weights_z);
    TensorInfo        weights_reshaped(weights->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(shape_weights_reshape).set_data_layout(DataLayout::NCHW));
    ARM_COMPUTE_RETURN_ON_ERROR(NEDepthwiseWeightsReshapeKernel::validate(weights_to_use, &weights_reshaped, append_bias ? biases : nullptr));

    // GEMV configuration
    DataType    v2mm_dt        = (input->data_type() == DataType::QASYMM8) ? DataType::S32 : input->data_type();
    TensorShape shape_v2mm_out = input_to_use->tensor_shape();
    shape_v2mm_out.set(0, conv_size * weights_z);
    shape_v2mm_out.set(1, 1);
    shape_v2mm_out.set(2, 1);
    TensorInfo v2mm_output(input->clone()->set_is_resizable(true).reset_padding().set_data_type(v2mm_dt).set_tensor_shape(shape_v2mm_out).set_data_layout(DataLayout::NCHW));
    ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMMatrixVectorMultiplyKernel::validate(&input_reshaped, &weights_reshaped, &v2mm_output));

    TensorInfo output_reshaped(v2mm_output.clone()->set_is_resizable(true).reset_padding().set_tensor_shape(output_to_use->tensor_shape()));
    ARM_COMPUTE_RETURN_ON_ERROR(NEDepthwiseVectorToTensorKernel::validate(&v2mm_output, (is_quantized) ? &output_reshaped : output_to_use, conv_w, conv_h));

    if(is_quantized)
    {
        ARM_COMPUTE_RETURN_ON_ERROR(NEDirectConvolutionLayerOutputStageKernel::validate(&output_reshaped, biases, output_to_use));
    }

    return Status{};
}

void NEDepthwiseConvolutionLayer::run()
{
    prepare();

    if(_is_nhwc)
    {
        _permute_input.run();
    }

    NEScheduler::get().schedule(&_im2col_kernel, Window::DimX);
    NEScheduler::get().schedule(&_v2mm_input_fill_border, Window::DimX);
    NEScheduler::get().schedule(&_v2mm_kernel, Window::DimX);
    NEScheduler::get().schedule(&_vector_to_tensor_kernel, Window::DimX);
    if(_is_quantized)
    {
        NEScheduler::get().schedule(&_output_stage_kernel, Window::DimX);
    }

    if(_is_nhwc)
    {
        _permute_output.run();
    }
}

void NEDepthwiseConvolutionLayer::prepare()
{
    if(!_is_prepared)
    {
        ARM_COMPUTE_ERROR_ON(!_original_weights->is_used());

        if(_is_nhwc)
        {
            _permute_weights.run();
        }

        // Run reshape and mark original weights as unused
        _weights_reshaped.allocator()->allocate();
        NEScheduler::get().schedule(&_weights_reshape_kernel, Window::DimX);
        NEScheduler::get().schedule(&_v2mm_weights_fill_border, Window::DimX);
        _original_weights->mark_as_unused();

        _is_prepared = true;
    }
}
