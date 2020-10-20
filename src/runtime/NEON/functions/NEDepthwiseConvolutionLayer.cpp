/*
 * Copyright (c) 2017-2020 Arm Limited.
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

#include "arm_compute/core/utils/misc/InfoHelpers.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/core/NEON/kernels/NEDepthwiseConvolutionLayerNativeKernel.h"
#include "support/MemorySupport.h"

using namespace arm_compute::misc;
using namespace arm_compute::misc::shape_calculator;

namespace arm_compute
{
namespace
{
Status validate_arguments_optimized(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                    unsigned int depth_multiplier, const ActivationLayerInfo &act_info, const Size2D &dilation)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    if(!is_data_type_quantized_per_channel(weights->data_type()))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    }
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_layout() == DataLayout::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON(dilation.x() < 1 || dilation.y() < 1);
    const size_t idx_w = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_h = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT);
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(idx_w) + (weights->dimension(idx_w) - 1) * (dilation.x() - 1) > input->dimension(idx_w) + conv_info.pad_left() + conv_info.pad_right());
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(idx_h) + (weights->dimension(idx_h) - 1) * (dilation.y() - 1) > input->dimension(idx_h) + conv_info.pad_top() + conv_info.pad_bottom());

    if(biases != nullptr)
    {
        const unsigned int channel_idx = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::CHANNEL);
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != weights->dimension(channel_idx));
    }

    ARM_COMPUTE_RETURN_ON_ERROR(NEDepthwiseConvolutionAssemblyDispatch::validate(input, weights, biases, output, conv_info, depth_multiplier, act_info, dilation));

    //Validate Activation Layer
    if(act_info.enabled())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(NEActivationLayer::validate(output, nullptr, act_info));
    }
    return Status{};
}
} // namespace

NEDepthwiseConvolutionLayer::~NEDepthwiseConvolutionLayer() = default;

NEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerOptimizedInternal::NEDepthwiseConvolutionLayerOptimizedInternal(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(memory_manager), _dwc_optimized_func(memory_manager), _permute_input(), _permute_weights(), _permute_output(), _activationlayer_function(), _accumulator(), _permuted_input(),
      _permuted_weights(), _permuted_output(), _original_weights(nullptr), _has_bias(false), _is_quantized(false), _is_nchw(true), _permute(false), _is_activationlayer_enabled(false), _is_prepared(false)
{
}

void NEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerOptimizedInternal::configure(ITensor       *input,
                                                                                          const ITensor *weights,
                                                                                          const ITensor *biases,
                                                                                          ITensor *output, const PadStrideInfo &conv_info,
                                                                                          unsigned int               depth_multiplier,
                                                                                          const ActivationLayerInfo &act_info,
                                                                                          const Size2D              &dilation)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(NEDepthwiseConvolutionLayerOptimizedInternal::validate(input->info(), weights->info(), (biases == nullptr) ? nullptr : biases->info(),
                                                                                      output->info(), conv_info, depth_multiplier, act_info, dilation));

    _original_weights           = weights;
    _is_quantized               = is_data_type_quantized_asymmetric(input->info()->data_type());
    _has_bias                   = biases != nullptr;
    _is_nchw                    = input->info()->data_layout() == DataLayout::NCHW;
    _permute                    = _is_nchw;
    _is_prepared                = false;
    _is_activationlayer_enabled = act_info.enabled();

    // Configure pipeline
    ActivationLayerInfo act_info_to_use = ActivationLayerInfo();
    const bool          is_relu         = arm_compute::utils::info_helpers::is_relu(act_info);
    const bool          is_relu6        = arm_compute::utils::info_helpers::is_relu6(act_info);
    _is_activationlayer_enabled         = act_info.enabled() && !(is_relu || is_relu6);
    if(!_is_activationlayer_enabled)
    {
        act_info_to_use = act_info;
    }

    if(_is_nchw)
    {
        _memory_group.manage(&_permuted_input);
        _memory_group.manage(&_permuted_output);

        // Configure the function to transform the input tensor from NCHW -> NHWC
        _permute_input.configure(input, &_permuted_input, PermutationVector(2U, 0U, 1U));
        _permuted_input.info()->set_data_layout(DataLayout::NHWC);

        // Configure the function to transform the weights tensor from IHW -> HWI
        _permute_weights.configure(weights, &_permuted_weights, PermutationVector(2U, 0U, 1U));
        _permuted_weights.info()->set_data_layout(DataLayout::NHWC);

        _permuted_output.info()->set_data_layout(DataLayout::NHWC);
        _permuted_output.info()->set_quantization_info(output->info()->quantization_info());

        // Configure optimized depthwise
        _dwc_optimized_func.configure(&_permuted_input, &_permuted_weights, biases, &_permuted_output, conv_info, depth_multiplier, act_info_to_use, dilation);

        // Configure the function to transform the convoluted output to ACL's native ordering format NCHW
        _permuted_output.info()->set_data_layout(DataLayout::NHWC);
        _permute_output.configure(&_permuted_output, output, PermutationVector(1U, 2U, 0U));

        // Allocate tensors
        _permuted_input.allocator()->allocate();
        _permuted_output.allocator()->allocate();
    }
    else
    {
        _dwc_optimized_func.configure(input, weights, biases, output, conv_info, depth_multiplier, act_info_to_use, dilation);
    }

    // Configure activation
    if(_is_activationlayer_enabled)
    {
        _activationlayer_function.configure(output, nullptr, act_info);
    }
}

Status NEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerOptimizedInternal::validate(const ITensorInfo         *input,
                                                                                           const ITensorInfo         *weights,
                                                                                           const ITensorInfo         *biases,
                                                                                           const ITensorInfo         *output,
                                                                                           const PadStrideInfo       &conv_info,
                                                                                           unsigned int               depth_multiplier,
                                                                                           const ActivationLayerInfo &act_info,
                                                                                           const Size2D              &dilation)
{
    return validate_arguments_optimized(input, weights, biases, output, conv_info, depth_multiplier, act_info, dilation);
}

void NEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerOptimizedInternal::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_memory_group);

    // Permute input
    if(_permute)
    {
        _permute_input.run();
    }

    // Run assembly function
    _dwc_optimized_func.run();

    // Permute output
    if(_is_nchw)
    {
        _permute_output.run();
    }

    // Run activation
    if(_is_activationlayer_enabled)
    {
        _activationlayer_function.run();
    }
}

void NEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerOptimizedInternal::prepare()
{
    if(!_is_prepared)
    {
        // Permute weights
        if(_permute)
        {
            _permuted_weights.allocator()->allocate();
            _permute_weights.run();
            _original_weights->mark_as_unused();
        }

        // Prepare optimized function
        _dwc_optimized_func.prepare();
        if(!_permuted_weights.is_used())
        {
            _permuted_weights.allocator()->free();
        }

        _is_prepared = true;
    }
}

NEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerGeneric::NEDepthwiseConvolutionLayerGeneric()
    : _depthwise_conv_kernel(), _permute_input(), _permute_weights(), _permute_output(), _activationlayer_function(), _permuted_input(), _permuted_weights(), _permuted_output(), _is_prepared(false),
      _is_nchw(false), _is_activationlayer_enabled(false), _original_weights(nullptr)
{
}

void NEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerGeneric::configure(ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info,
                                                                                unsigned int depth_multiplier, const ActivationLayerInfo &act_info, const Size2D &dilation)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(NEDepthwiseConvolutionLayer::validate(input->info(), weights->info(), (biases == nullptr) ? nullptr : biases->info(),
                                                                     output->info(), conv_info, depth_multiplier, act_info, dilation));

    _is_nchw     = input->info()->data_layout() == DataLayout::NCHW;
    _is_prepared = !_is_nchw;

    ITensor       *input_to_use   = input;
    const ITensor *weights_to_use = weights;
    ITensor       *output_to_use  = output;
    if(_is_nchw)
    {
        _permute_input.configure(input, &_permuted_input, PermutationVector(2U, 0U, 1U));
        _permuted_input.info()->set_data_layout(DataLayout::NHWC);
        input_to_use = &_permuted_input;

        _permute_weights.configure(weights, &_permuted_weights, PermutationVector(2U, 0U, 1U));
        _permuted_weights.info()->set_data_layout(DataLayout::NHWC);
        weights_to_use = &_permuted_weights;

        _permuted_output.allocator()->init(output->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(TensorShape()));
        output_to_use = &_permuted_output;
    }
    _original_weights = weights_to_use;

    _depthwise_conv_kernel = arm_compute::support::cpp14::make_unique<NEDepthwiseConvolutionLayerNativeKernel>();
    _depthwise_conv_kernel->configure(input_to_use, weights_to_use, biases, output_to_use, conv_info, depth_multiplier, dilation);

    if(_is_nchw)
    {
        _permute_output.configure(&_permuted_output, output, PermutationVector(1U, 2U, 0U));
        _permuted_output.info()->set_data_layout(DataLayout::NHWC);

        _permuted_input.allocator()->allocate();
        _permuted_weights.allocator()->allocate();
        _permuted_output.allocator()->allocate();
    }

    //Configure Activation Layer
    _is_activationlayer_enabled = act_info.enabled();
    if(_is_activationlayer_enabled)
    {
        _activationlayer_function.configure(output, nullptr, act_info);
    }
}

Status NEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerGeneric::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
                                                                                 const PadStrideInfo &conv_info,
                                                                                 unsigned int depth_multiplier, const ActivationLayerInfo &act_info, const Size2D &dilation)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    if(input->data_layout() == DataLayout::NCHW)
    {
        TensorShape permuted_input_shape   = input->tensor_shape();
        TensorShape permuted_weights_shape = weights->tensor_shape();
        TensorShape permuted_output_shape  = misc::shape_calculator::compute_depthwise_convolution_shape(*input, *weights, conv_info, depth_multiplier, dilation);
        permute(permuted_input_shape, PermutationVector(2U, 0U, 1U));
        permute(permuted_weights_shape, PermutationVector(2U, 0U, 1U));
        permute(permuted_output_shape, PermutationVector(2U, 0U, 1U));

        const TensorInfo permuted_input   = TensorInfo(input->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(permuted_input_shape).set_data_layout(DataLayout::NHWC));
        const TensorInfo permuted_weights = TensorInfo(weights->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(permuted_weights_shape).set_data_layout(DataLayout::NHWC));
        const TensorInfo permuted_output  = TensorInfo(output->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(permuted_output_shape).set_data_layout(DataLayout::NCHW));

        ARM_COMPUTE_RETURN_ON_ERROR(NEPermute::validate(input, &permuted_input, PermutationVector(2U, 0U, 1U)));
        ARM_COMPUTE_RETURN_ON_ERROR(NEPermute::validate(weights, &permuted_weights, PermutationVector(2U, 0U, 1U)));
        ARM_COMPUTE_RETURN_ON_ERROR(NEPermute::validate(&permuted_output, output, PermutationVector(1U, 2U, 0U)));

        ARM_COMPUTE_RETURN_ON_ERROR(NEDepthwiseConvolutionLayerNativeKernel::validate(&permuted_input, &permuted_weights, biases, &permuted_output, conv_info, depth_multiplier, dilation));
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(NEDepthwiseConvolutionLayerNativeKernel::validate(input, weights, biases, output, conv_info, depth_multiplier, dilation));
    }

    // Validate Activation Layer
    if(act_info.enabled())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(NEActivationLayer::validate(output, nullptr, act_info));
    }

    return Status{};
}

void NEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerGeneric::run()
{
    if(_is_nchw)
    {
        prepare();
        _permute_input.run();
    }

    NEScheduler::get().schedule(_depthwise_conv_kernel.get(), Window::DimY);

    if(_is_nchw)
    {
        _permute_output.run();
    }

    if(_is_activationlayer_enabled)
    {
        _activationlayer_function.run();
    }
}

void NEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerGeneric::prepare()
{
    if(!_is_prepared)
    {
        ARM_COMPUTE_ERROR_ON(!_original_weights->is_used());

        _permute_weights.run();
        _original_weights->mark_as_unused();
        _is_prepared = true;
    }
}

NEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _depth_conv_func(DepthwiseConvolutionFunction::GENERIC), _func_optimized(std::move(memory_manager)), _func_generic()
{
}

void NEDepthwiseConvolutionLayer::configure(ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier,
                                            const ActivationLayerInfo &act_info, const Size2D &dilation)
{
    _depth_conv_func = get_depthwiseconvolution_function(input->info(), weights->info(), (biases != nullptr) ? biases->info() : nullptr, output->info(), conv_info, depth_multiplier, act_info, dilation);
    switch(_depth_conv_func)
    {
        case DepthwiseConvolutionFunction::OPTIMIZED:
            _func_optimized.configure(input, weights, biases, output, conv_info, depth_multiplier, act_info, dilation);
            break;
        case DepthwiseConvolutionFunction::GENERIC:
            _func_generic.configure(input, weights, biases, output, conv_info, depth_multiplier, act_info, dilation);
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported DepthwiseConvolutionFunction");
    }
}

Status NEDepthwiseConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                             unsigned int depth_multiplier, const ActivationLayerInfo &act_info, const Size2D &dilation)
{
    DepthwiseConvolutionFunction depth_conv_func = get_depthwiseconvolution_function(input, weights, biases, output, conv_info, depth_multiplier, act_info, dilation);
    switch(depth_conv_func)
    {
        case DepthwiseConvolutionFunction::OPTIMIZED:
            return NEDepthwiseConvolutionLayerOptimizedInternal::validate(input, weights, biases, output, conv_info, depth_multiplier, act_info, dilation);
            break;
        case DepthwiseConvolutionFunction::GENERIC:
            return NEDepthwiseConvolutionLayerGeneric::validate(input, weights, biases, output, conv_info, depth_multiplier, act_info, dilation);
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported DepthwiseConvolutionFunction");
    }
}

DepthwiseConvolutionFunction NEDepthwiseConvolutionLayer::get_depthwiseconvolution_function(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
                                                                                            const PadStrideInfo &conv_info,
                                                                                            unsigned int depth_multiplier, ActivationLayerInfo act_info, const Size2D &dilation)
{
    if(bool(NEDepthwiseConvolutionLayerOptimizedInternal::validate(input, weights, biases, output, conv_info, depth_multiplier, act_info, dilation)))
    {
        return DepthwiseConvolutionFunction::OPTIMIZED;
    }
    else
    {
        return DepthwiseConvolutionFunction::GENERIC;
    }
}

void NEDepthwiseConvolutionLayer::run()
{
    switch(_depth_conv_func)
    {
        case DepthwiseConvolutionFunction::OPTIMIZED:
            _func_optimized.run();
            break;
        case DepthwiseConvolutionFunction::GENERIC:
            _func_generic.run();
            break;
        default:
            ARM_COMPUTE_ERROR("DepthwiseConvolutionFunction not properly configured");
    }
}

void NEDepthwiseConvolutionLayer::prepare()
{
    switch(_depth_conv_func)
    {
        case DepthwiseConvolutionFunction::OPTIMIZED:
            _func_optimized.prepare();
            break;
        case DepthwiseConvolutionFunction::GENERIC:
            _func_generic.prepare();
            break;
        default:
            ARM_COMPUTE_ERROR("DepthwiseConvolutionFunction not properly configured");
    }
}
} // namespace arm_compute
