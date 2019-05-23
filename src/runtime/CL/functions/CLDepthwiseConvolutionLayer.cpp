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
#include "arm_compute/runtime/CL/functions/CLDepthwiseConvolutionLayer.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/kernels/CLDepthwiseConvolutionLayer3x3NCHWKernel.h"
#include "arm_compute/core/CL/kernels/CLDepthwiseConvolutionLayer3x3NHWCKernel.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
using namespace arm_compute::misc;
using namespace arm_compute::misc::shape_calculator;

CLDepthwiseConvolutionLayer3x3::CLDepthwiseConvolutionLayer3x3(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _kernel(nullptr), _border_handler(), _permute_input_to_nchw(), _permute_weights_to_nchw(), _permute_output_to_nhwc(), _reshape_weights(), _permuted_input(),
      _permuted_weights(), _permuted_output(), _original_weights(nullptr), _needs_permute(false), _needs_weights_reshape(false), _is_prepared(false)
{
}

void CLDepthwiseConvolutionLayer3x3::configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier,
                                               ActivationLayerInfo act_info, const Size2D &dilation)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    // idx_w and idx_h only used for validation
    const size_t idx_w = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_h = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::HEIGHT);
    ARM_COMPUTE_UNUSED(idx_w);
    ARM_COMPUTE_UNUSED(idx_h);

    ARM_COMPUTE_ERROR_ON(weights->info()->dimension(idx_w) + (weights->info()->dimension(idx_w) - 1) * (dilation.x() - 1) > input->info()->dimension(idx_w) + conv_info.pad_left() + conv_info.pad_right());
    ARM_COMPUTE_ERROR_ON(weights->info()->dimension(idx_h) + (weights->info()->dimension(idx_h) - 1) * (dilation.y() - 1) > input->info()->dimension(idx_h) + conv_info.pad_top() + conv_info.pad_bottom());

    const bool is_nhwc = input->info()->data_layout() == DataLayout::NHWC;

    _needs_permute         = is_nhwc && (depth_multiplier > 1);
    _needs_weights_reshape = is_nhwc && (depth_multiplier == 1)
                             && is_data_type_quantized_asymmetric(input->info()->data_type());
    _is_prepared      = false;
    _original_weights = weights;

    ICLTensor       *input_to_use   = input;
    const ICLTensor *weights_to_use = weights;
    ICLTensor       *output_to_use  = output;

    const bool is_stride_1            = ((conv_info.stride().first == conv_info.stride().second) && (conv_info.stride().first == 1));
    const bool is_dot8_supported      = dot8_supported(CLKernelLibrary::get().get_device());
    const bool is_stride_1_dilation_1 = (is_stride_1 && dilation.x() == 1 && dilation.y() == 1);

    DepthwiseConvolutionReshapeInfo info;
    info.c0        = 4;
    info.transpose = is_stride_1_dilation_1 && is_dot8_supported;

    if(_needs_permute)
    {
        _memory_group.manage(&_permuted_input);
        _memory_group.manage(&_permuted_output);

        // Configure the function to transform the input tensor from NHWC -> NCHW
        _permute_input_to_nchw.configure(input, &_permuted_input, PermutationVector(1U, 2U, 0U));
        _permuted_input.info()->set_data_layout(DataLayout::NCHW);

        // Configure the function to transform the weights tensor from HWI -> IHW
        _permute_weights_to_nchw.configure(weights, &_permuted_weights, PermutationVector(1U, 2U, 0U));
        _permuted_weights.info()->set_data_layout(DataLayout::NCHW);

        input_to_use   = &_permuted_input;
        weights_to_use = &_permuted_weights;
        output_to_use  = &_permuted_output;

        _kernel = arm_compute::support::cpp14::make_unique<CLDepthwiseConvolutionLayer3x3NCHWKernel>();
    }
    else if(is_nhwc)
    {
        if(_needs_weights_reshape)
        {
            _reshape_weights.configure(weights, &_permuted_weights, info);
            weights_to_use = &_permuted_weights;
        }
        _kernel = arm_compute::support::cpp14::make_unique<CLDepthwiseConvolutionLayer3x3NHWCKernel>();
    }
    else
    {
        _kernel = arm_compute::support::cpp14::make_unique<CLDepthwiseConvolutionLayer3x3NCHWKernel>();
    }

    // Configure kernel
    _kernel->set_target(CLScheduler::get().target());
    _kernel->configure(input_to_use, weights_to_use, biases, output_to_use, conv_info, depth_multiplier, act_info, dilation);

    // Permute output if needed
    if(_needs_permute)
    {
        // Configure the function to transform the convoluted output to ACL's native ordering format NCHW
        _permuted_output.info()->set_data_layout(DataLayout::NCHW);
        _permute_output_to_nhwc.configure(&_permuted_output, output, PermutationVector(2U, 0U, 1U));

        // Allocate tensors
        _permuted_input.allocator()->allocate();
        _permuted_output.allocator()->allocate();
    }
    // Configure border handler
    PixelValue &&zero_value(0.f);
    if(is_data_type_quantized_asymmetric(input->info()->data_type()))
    {
        zero_value = PixelValue(static_cast<uint8_t>(input->info()->quantization_info().offset));
    }
    _border_handler.configure(input_to_use, _kernel->border_size(), BorderMode::CONSTANT, zero_value);
}

Status CLDepthwiseConvolutionLayer3x3::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                                unsigned int depth_multiplier, ActivationLayerInfo act_info, GPUTarget gpu_target, const Size2D &dilation)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_layout() == DataLayout::UNKNOWN);

    const bool                      is_nhwc                = input->data_layout() == DataLayout::NHWC;
    const bool                      needs_permute          = is_nhwc && (depth_multiplier > 1);
    const bool                      needs_weights_reshape  = is_nhwc && (depth_multiplier == 1);
    const bool                      is_stride_1            = ((conv_info.stride().first == conv_info.stride().second) && (conv_info.stride().first == 1));
    const bool                      is_stride_1_dilation_1 = (is_stride_1 && dilation.x() == 1 && dilation.y() == 1);
    const bool                      is_dot8_supported      = dot8_supported(CLKernelLibrary::get().get_device());
    DepthwiseConvolutionReshapeInfo info;
    info.c0        = 4;
    info.transpose = is_stride_1_dilation_1 && is_dot8_supported;

    if(needs_permute)
    {
        TensorShape permuted_input_shape   = input->tensor_shape();
        TensorShape permuted_weights_shape = weights->tensor_shape();
        TensorShape permuted_output_shape  = shape_calculator::compute_depthwise_convolution_shape(*input, *weights, conv_info, depth_multiplier, dilation);

        permute(permuted_input_shape, PermutationVector(1U, 2U, 0U));
        permute(permuted_weights_shape, PermutationVector(1U, 2U, 0U));
        permute(permuted_output_shape, PermutationVector(1U, 2U, 0U));

        const TensorInfo permuted_input   = input->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(permuted_input_shape).set_data_layout(DataLayout::NCHW);
        const TensorInfo permuted_weights = weights->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(permuted_weights_shape).set_data_layout(DataLayout::NCHW);
        const TensorInfo permuted_output  = output->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(permuted_output_shape).set_data_layout(DataLayout::NCHW);

        ARM_COMPUTE_RETURN_ON_ERROR(CLDepthwiseConvolutionLayer3x3NCHWKernel::validate(&permuted_input, &permuted_weights, biases, &permuted_output, conv_info, depth_multiplier, act_info, gpu_target,
                                                                                       dilation));
    }
    else if(is_nhwc)
    {
        if(needs_weights_reshape)
        {
            auto reshaped_weights_shape = arm_compute::misc::shape_calculator::compute_reshaped_depthwise_weights_shape(*weights, info);
            ARM_COMPUTE_RETURN_ON_ERROR(CLDepthwiseConvolutionLayer3x3NHWCKernel::validate(input, &weights->clone()->set_tensor_shape(reshaped_weights_shape), biases, output, conv_info, depth_multiplier,
                                                                                           act_info, dilation));
        }
        ARM_COMPUTE_RETURN_ON_ERROR(CLDepthwiseConvolutionLayer3x3NHWCKernel::validate(input, weights, biases, output, conv_info, depth_multiplier, act_info, dilation));
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLDepthwiseConvolutionLayer3x3NCHWKernel::validate(input, weights, biases, output, conv_info, depth_multiplier, act_info, gpu_target, dilation));
    }

    return Status{};
}

void CLDepthwiseConvolutionLayer3x3::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_memory_group);

    if(_needs_permute)
    {
        _permute_input_to_nchw.run();
    }
    CLScheduler::get().enqueue(_border_handler);
    CLScheduler::get().enqueue(*_kernel);

    if(_needs_permute)
    {
        _permute_output_to_nhwc.run();
    }
}

void CLDepthwiseConvolutionLayer3x3::prepare()
{
    if(!_is_prepared)
    {
        if(_needs_permute)
        {
            ARM_COMPUTE_ERROR_ON(!_original_weights->is_used());

            _permuted_weights.allocator()->allocate();
            _permute_weights_to_nchw.run();
            _original_weights->mark_as_unused();
        }

        if(_needs_weights_reshape)
        {
            ARM_COMPUTE_ERROR_ON(_needs_permute);
            ARM_COMPUTE_ERROR_ON(!_original_weights->is_used());
            _permuted_weights.allocator()->allocate();
            CLScheduler::get().enqueue(_reshape_weights);
            _original_weights->mark_as_unused();
        }
        _is_prepared = true;
    }
}

CLDepthwiseConvolutionLayer::CLDepthwiseConvolutionLayer()
    : _im2col_kernel(), _weights_reshape_kernel(), _v2mm_kernel(), _vector_to_tensor_kernel(), _output_stage_kernel(), _activationlayer_function(), _v2mm_input_fill_border(), _v2mm_weights_fill_border(),
      _input_reshaped(), _weights_reshaped(), _v2mm_output(), _output_reshaped(), _is_prepared(false), _is_quantized(false), _is_activationlayer_enabled(false), _original_weights(nullptr),
      _optimised_function(nullptr)
{
}

void CLDepthwiseConvolutionLayer::configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                                            unsigned int depth_multiplier, const ActivationLayerInfo &act_info, const Size2D &dilation)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);

    const size_t idx_w = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_h = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::HEIGHT);

    ARM_COMPUTE_ERROR_ON(weights->info()->dimension(idx_w) + (weights->info()->dimension(idx_w) - 1) * (dilation.x() - 1) > input->info()->dimension(idx_w) + conv_info.pad_left() + conv_info.pad_right());
    ARM_COMPUTE_ERROR_ON(weights->info()->dimension(idx_h) + (weights->info()->dimension(idx_h) - 1) * (dilation.y() - 1) > input->info()->dimension(idx_h) + conv_info.pad_top() + conv_info.pad_bottom());

    const bool can_run_optimised_3x3_kernel = (weights->info()->dimension(idx_w) == 3) && (weights->info()->dimension(idx_h) == 3);

    if(bool(can_run_optimised_3x3_kernel))
    {
        auto f = arm_compute::support::cpp14::make_unique<CLDepthwiseConvolutionLayer3x3>();
        f->configure(input, weights, biases, output, conv_info, depth_multiplier, act_info, dilation);
        _optimised_function = std::move(f);
    }
    else
    {
        const size_t idx_c = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::CHANNEL);

        const size_t weights_w = weights->info()->dimension(idx_w);
        const size_t weights_h = weights->info()->dimension(idx_h);
        const size_t weights_z = weights->info()->dimension(idx_c);

        _is_prepared      = false;
        _original_weights = weights;
        _is_quantized     = is_data_type_quantized_asymmetric(input->info()->data_type());

        bool            append_bias = (biases != nullptr) && !_is_quantized;
        const GPUTarget gpu_target  = CLScheduler::get().target();

        // Calculate output shape
        TensorShape output_shape = shape_calculator::compute_depthwise_convolution_shape(*input->info(), *weights->info(), conv_info, depth_multiplier, dilation);

        // Output auto inizialitation if not yet initialized
        auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(output_shape));
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(output->info()->tensor_shape(), output_shape);

        // Output width and height
        const unsigned int conv_w = output_shape[idx_w];
        const unsigned int conv_h = output_shape[idx_h];

        // Set up intermediate tensors
        const size_t patch_size = weights_w * weights_h + ((append_bias) ? 1 : 0);
        const size_t conv_size  = conv_w * conv_h;

        // Im2Col configuration
        TensorShape shape_im2col = input->info()->tensor_shape();
        shape_im2col.set(0, patch_size);
        shape_im2col.set(1, conv_size);
        shape_im2col.set(2, weights_z);
        _input_reshaped.allocator()->init(input->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(shape_im2col));
        _im2col_kernel.set_target(gpu_target);
        _im2col_kernel.configure(input, &_input_reshaped, Size2D(weights_w, weights_h), conv_info, append_bias, depth_multiplier, dilation);
        CLScheduler::get().tune_kernel_static(_im2col_kernel);

        // Weights reshape configuration
        const TensorShape shape_weights_reshape(patch_size, weights_z);
        _weights_reshaped.allocator()->init(weights->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(shape_weights_reshape));
        _weights_reshape_kernel.configure(weights, &_weights_reshaped, append_bias ? biases : nullptr);

        // GEMV configuration
        DataType    v2mm_dt        = (input->info()->data_type() == DataType::QASYMM8) ? DataType::S32 : input->info()->data_type();
        TensorShape shape_v2mm_out = input->info()->tensor_shape();
        shape_v2mm_out.set(0, conv_size * weights_z);
        shape_v2mm_out.set(1, 1);
        shape_v2mm_out.set(2, 1);
        _v2mm_output.allocator()->init(input->info()->clone()->set_is_resizable(true).reset_padding().set_data_type(v2mm_dt).set_tensor_shape(shape_v2mm_out));
        _v2mm_kernel.set_target(gpu_target);
        _v2mm_kernel.configure(&_input_reshaped, &_weights_reshaped, &_v2mm_output);
        CLScheduler::get().tune_kernel_static(_v2mm_kernel);
        _output_reshaped.allocator()->init(_v2mm_output.info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(output_shape));
        _vector_to_tensor_kernel.configure(&_v2mm_output, (_is_quantized) ? &_output_reshaped : output, conv_w, conv_h);

        // Output staged configuration
        if(_is_quantized)
        {
            const QuantizationInfo output_quant_info = (output->info()->total_size() == 0) ? input->info()->quantization_info() : output->info()->quantization_info();

            float multiplier = input->info()->quantization_info().scale * weights->info()->quantization_info().scale / output_quant_info.scale;
            int   output_multiplier;
            int   output_shift;
            quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift);
            _output_stage_kernel.configure(&_output_reshaped, biases, output, output_multiplier, output_shift, output_quant_info.offset);
            _output_reshaped.allocator()->allocate();
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

        //Configure Activation Layer
        _is_activationlayer_enabled = act_info.enabled();

        if(_is_activationlayer_enabled)
        {
            _activationlayer_function.configure(output, nullptr, act_info);
        }
    }
}

Status CLDepthwiseConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                             unsigned int depth_multiplier, const ActivationLayerInfo &act_info, const Size2D &dilation)
{
    const size_t idx_w = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_h = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT);

    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(idx_w) + (weights->dimension(idx_w) - 1) * (dilation.x() - 1) > input->dimension(idx_w) + conv_info.pad_left() + conv_info.pad_right());
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(idx_h) + (weights->dimension(idx_h) - 1) * (dilation.y() - 1) > input->dimension(idx_h) + conv_info.pad_top() + conv_info.pad_bottom());

    const bool can_run_optimised_3x3_kernel = (weights->dimension(idx_w) == 3) && (weights->dimension(idx_h) == 3);

    if(can_run_optimised_3x3_kernel)
    {
        const size_t idx_c = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::CHANNEL);

        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
        ARM_COMPUTE_RETURN_ERROR_ON((input->dimension(idx_c) * depth_multiplier) != weights->dimension(idx_c));

        const bool         is_quantized = is_data_type_quantized_asymmetric(input->data_type());
        const bool         append_bias  = (biases != nullptr) && !is_quantized;
        const TensorShape  output_shape = shape_calculator::compute_depthwise_convolution_shape(*input, *weights, conv_info, depth_multiplier, dilation);
        const size_t       weights_w    = weights->dimension(idx_w);
        const size_t       weights_h    = weights->dimension(idx_h);
        const size_t       weights_z    = weights->dimension(idx_c);
        const unsigned int conv_w       = output_shape[idx_w];
        const unsigned int conv_h       = output_shape[idx_h];
        const size_t       patch_size   = weights_w * weights_h + ((append_bias) ? 1 : 0);
        const size_t       conv_size    = conv_w * conv_h;

        TensorShape shape_im2col = input->tensor_shape();
        shape_im2col.set(0, patch_size);
        shape_im2col.set(1, conv_size);
        shape_im2col.set(2, weights_z);
        TensorInfo input_reshaped(input->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(shape_im2col));
        ARM_COMPUTE_RETURN_ON_ERROR(CLDepthwiseIm2ColKernel::validate(input, &input_reshaped, Size2D(weights_w, weights_h), conv_info, append_bias, depth_multiplier, dilation));

        const TensorShape shape_weights_reshape(patch_size, weights_z);
        TensorInfo        weights_reshaped(weights->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(shape_weights_reshape));
        ARM_COMPUTE_RETURN_ON_ERROR(CLDepthwiseConvolutionLayerReshapeWeightsGenericKernel::validate(weights, &weights_reshaped, append_bias ? biases : nullptr));

        DataType    v2mm_dt        = (input->data_type() == DataType::QASYMM8) ? DataType::S32 : input->data_type();
        TensorShape shape_v2mm_out = input->tensor_shape();
        shape_v2mm_out.set(0, conv_size * weights_z);
        shape_v2mm_out.set(1, 1);
        shape_v2mm_out.set(2, 1);
        TensorInfo v2mm_output(input->clone()->set_is_resizable(true).reset_padding().set_data_type(v2mm_dt).set_tensor_shape(shape_v2mm_out));
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMMatrixVectorMultiplyKernel::validate(&input_reshaped, &weights_reshaped, &v2mm_output));

        TensorInfo output_reshaped(v2mm_output.clone()->set_is_resizable(true).reset_padding().set_tensor_shape(output_shape));
        ARM_COMPUTE_RETURN_ON_ERROR(CLDepthwiseVectorToTensorKernel::validate(&v2mm_output, (is_quantized) ? &output_reshaped : output, conv_w, conv_h));

        if(is_quantized)
        {
            ARM_COMPUTE_RETURN_ON_ERROR(CLDirectConvolutionLayerOutputStageKernel::validate(&output_reshaped, biases, output));
        }

        // Validate Activation Layer
        if(act_info.enabled())
        {
            ARM_COMPUTE_RETURN_ON_ERROR(CLActivationLayer::validate(output, nullptr, act_info));
        }
    }
    else
    {
        CLDepthwiseConvolutionLayer3x3::validate(input, weights, biases, output, conv_info, depth_multiplier, act_info, GPUTarget::MIDGARD, dilation);
    }
    return Status{};
}

void CLDepthwiseConvolutionLayer::run()
{
    prepare();

    if(_optimised_function != nullptr)
    {
        _optimised_function->run();
    }
    else
    {
        CLScheduler::get().enqueue(_im2col_kernel);
        CLScheduler::get().enqueue(_v2mm_input_fill_border);
        CLScheduler::get().enqueue(_v2mm_kernel);
        CLScheduler::get().enqueue(_vector_to_tensor_kernel);
        if(_is_quantized)
        {
            CLScheduler::get().enqueue(_output_stage_kernel);
        }
        if(_is_activationlayer_enabled)
        {
            _activationlayer_function.run();
        }
    }
}

void CLDepthwiseConvolutionLayer::prepare()
{
    if(_optimised_function != nullptr)
    {
        _optimised_function->prepare();
    }
    else
    {
        if(!_is_prepared)
        {
            ARM_COMPUTE_ERROR_ON(!_original_weights->is_used());

            // Run weights reshaping and mark original weights tensor as unused
            _weights_reshaped.allocator()->allocate();
            CLScheduler::get().enqueue(_weights_reshape_kernel);
            CLScheduler::get().enqueue(_v2mm_weights_fill_border);
            _original_weights->mark_as_unused();

            CLScheduler::get().queue().finish();
            _is_prepared = true;
        }
    }
}
} // namespace arm_compute
