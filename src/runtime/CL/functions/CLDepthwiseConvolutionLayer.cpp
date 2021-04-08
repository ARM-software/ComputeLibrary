/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/core/CL/kernels/CLDepthwiseConvolutionLayer3x3NCHWKernel.h"
#include "src/core/CL/kernels/CLDepthwiseConvolutionLayer3x3NCHWKernel.h"
#include "src/core/CL/kernels/CLDepthwiseConvolutionLayer3x3NHWCKernel.h"
#include "src/core/CL/kernels/CLDepthwiseConvolutionLayer3x3NHWCKernel.h"
#include "src/core/CL/kernels/CLDepthwiseConvolutionLayerNativeKernel.h"
#include "src/core/CL/kernels/CLDepthwiseConvolutionLayerReshapeWeightsKernel.h"
#include "src/core/CL/kernels/CLFillBorderKernel.h"
#include "src/core/CL/kernels/ICLDepthwiseConvolutionLayer3x3Kernel.h"

namespace arm_compute
{
using namespace arm_compute::misc;
using namespace arm_compute::misc::shape_calculator;

namespace
{
Status validate_arguments_3x3(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                              unsigned int depth_multiplier, ActivationLayerInfo act_info, GPUTarget gpu_target, const Size2D &dilation)
{
    // This function should be removed and incorporated inside CLDepthwiseConvolutionLayerInternal3x3 once CLDepthwiseConvolutionLayer3x3 is properly removed
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_layout() == DataLayout::UNKNOWN);

    const bool                      is_quantized           = is_data_type_quantized_asymmetric(input->data_type());
    const bool                      is_nhwc                = input->data_layout() == DataLayout::NHWC;
    const bool                      needs_permute          = is_nhwc && (depth_multiplier > 1);
    const bool                      needs_weights_reshape  = is_nhwc && (depth_multiplier == 1) && is_quantized;
    const bool                      is_stride_1            = ((conv_info.stride().first == conv_info.stride().second) && (conv_info.stride().first == 1));
    const bool                      is_stride_1_dilation_1 = (is_stride_1 && dilation.x() == 1 && dilation.y() == 1);
    const bool                      is_dot8_supported      = dot8_supported(CLKernelLibrary::get().get_device());
    DepthwiseConvolutionReshapeInfo info;
    info.c0        = 4;
    info.transpose = is_stride_1_dilation_1 && is_dot8_supported;

    TensorInfo output_multipliers_shifts_info(TensorInfo(TensorShape(1U), 1, DataType::S32));
    if(is_quantized)
    {
        if(is_data_type_quantized_per_channel(weights->data_type()))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::QSYMM8_PER_CHANNEL);

            const size_t idx_c = get_data_layout_dimension_index(weights->data_layout(), DataLayoutDimension::CHANNEL);
            output_multipliers_shifts_info.set_tensor_shape(TensorShape(weights->dimension(idx_c)));
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
        }
    }

    if(needs_permute)
    {
        TensorShape           permuted_input_shape   = input->tensor_shape();
        TensorShape           permuted_weights_shape = weights->tensor_shape();
        const ConvolutionInfo info{ conv_info, depth_multiplier, ActivationLayerInfo(), dilation };
        TensorShape           permuted_output_shape = shape_calculator::compute_depthwise_convolution_shape(*input, *weights, info);

        permute(permuted_input_shape, PermutationVector(1U, 2U, 0U));
        permute(permuted_weights_shape, PermutationVector(1U, 2U, 0U));
        permute(permuted_output_shape, PermutationVector(1U, 2U, 0U));

        const TensorInfo permuted_input   = input->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(permuted_input_shape).set_data_layout(DataLayout::NCHW);
        const TensorInfo permuted_weights = weights->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(permuted_weights_shape).set_data_layout(DataLayout::NCHW);
        const TensorInfo permuted_output  = output->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(permuted_output_shape).set_data_layout(DataLayout::NCHW);

        ARM_COMPUTE_RETURN_ON_ERROR(CLDepthwiseConvolutionLayer3x3NCHWKernel::validate(&permuted_input, &permuted_weights, biases, &permuted_output,
                                                                                       conv_info, depth_multiplier, act_info, gpu_target,
                                                                                       dilation, &output_multipliers_shifts_info, &output_multipliers_shifts_info));
    }
    else if(is_nhwc)
    {
        if(needs_weights_reshape)
        {
            auto reshaped_weights_shape = arm_compute::misc::shape_calculator::compute_reshaped_depthwise_weights_shape(*weights, info);
            ARM_COMPUTE_RETURN_ON_ERROR(CLDepthwiseConvolutionLayer3x3NHWCKernel::validate(input, &weights->clone()->set_tensor_shape(reshaped_weights_shape), biases,
                                                                                           output, conv_info, depth_multiplier, act_info,
                                                                                           dilation, &output_multipliers_shifts_info, &output_multipliers_shifts_info));
        }
        else
        {
            ARM_COMPUTE_RETURN_ON_ERROR(CLDepthwiseConvolutionLayer3x3NHWCKernel::validate(input, weights, biases, output, conv_info, depth_multiplier, act_info,
                                                                                           dilation, &output_multipliers_shifts_info, &output_multipliers_shifts_info));
        }
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLDepthwiseConvolutionLayer3x3NCHWKernel::validate(input, weights, biases, output, conv_info, depth_multiplier, act_info, gpu_target,
                                                                                       dilation, &output_multipliers_shifts_info, &output_multipliers_shifts_info));
    }
    return Status{};
}
} // namespace

CLDepthwiseConvolutionLayer::CLDepthwiseConvolutionLayerGeneric::CLDepthwiseConvolutionLayerGeneric(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)),
      _dwc_native_kernel(std::make_unique<CLDepthwiseConvolutionLayerNativeKernel>()),
      _permute_input_to_nhwc(),
      _permute_weights_to_nhwc(),
      _permute_output_to_nchw(),
      _permuted_input(),
      _permuted_weights(),
      _permuted_output(),
      _output_multipliers(),
      _output_shifts(),
      _original_weights(),
      _input(),
      _output(),
      _needs_permute(false),
      _is_prepared(false),
      _is_quantized(false)
{
}

CLDepthwiseConvolutionLayer::~CLDepthwiseConvolutionLayer() = default;

void CLDepthwiseConvolutionLayer::CLDepthwiseConvolutionLayerGeneric::configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                                                                                unsigned int depth_multiplier, const ActivationLayerInfo &act_info, const Size2D &dilation)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, weights, biases, output, conv_info, depth_multiplier, act_info, dilation);
}

void CLDepthwiseConvolutionLayer::CLDepthwiseConvolutionLayerGeneric::configure(const CLCompileContext &compile_context, ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases,
                                                                                ICLTensor *output, const PadStrideInfo &conv_info,
                                                                                unsigned int depth_multiplier, const ActivationLayerInfo &act_info, const Size2D &dilation)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(CLDepthwiseConvolutionLayer::validate(input->info(),
                                                                     weights->info(),
                                                                     biases != nullptr ? biases->info() : nullptr,
                                                                     output->info(),
                                                                     conv_info,
                                                                     depth_multiplier,
                                                                     act_info,
                                                                     dilation));

    _is_quantized     = is_data_type_quantized(input->info()->data_type());
    _is_prepared      = false;
    _original_weights = weights;
    _input            = input;
    _output           = output;
    _needs_permute    = input->info()->data_layout() == DataLayout::NCHW;

    ICLTensor       *input_to_use   = input;
    const ICLTensor *weights_to_use = weights;
    ICLTensor       *output_to_use  = output;
    if(_needs_permute)
    {
        _memory_group.manage(&_permuted_input);
        _memory_group.manage(&_permuted_output);

        // Configure the function to transform the input tensor from NCHW -> NHWC
        _permute_input_to_nhwc.configure(compile_context, input, &_permuted_input, PermutationVector(2U, 0U, 1U));
        _permuted_input.info()->set_data_layout(DataLayout::NHWC);

        // Configure the function to transform the weights tensor from IHW -> HWI
        _permute_weights_to_nhwc.configure(compile_context, weights, &_permuted_weights, PermutationVector(2U, 0U, 1U));
        _permuted_weights.info()->set_data_layout(DataLayout::NHWC);

        // Set output quantization info before dwc kernel configure
        _permuted_output.info()->set_quantization_info(output->info()->quantization_info());

        input_to_use   = &_permuted_input;
        weights_to_use = &_permuted_weights;
        output_to_use  = &_permuted_output;
    }

    CLTensor *output_multipliers_to_use = nullptr;
    CLTensor *output_shifts_to_use      = nullptr;
    if(_is_quantized)
    {
        const size_t idx_c       = get_data_layout_dimension_index(weights->info()->data_layout(), DataLayoutDimension::CHANNEL);
        const size_t num_filters = (is_data_type_quantized_per_channel(weights->info()->data_type())) ? weights->info()->dimension(idx_c) : 1;

        _output_multipliers.allocator()->init(TensorInfo(TensorShape(num_filters), 1, DataType::S32));
        _output_shifts.allocator()->init(TensorInfo(TensorShape(num_filters), 1, DataType::S32));

        output_multipliers_to_use = &_output_multipliers;
        output_shifts_to_use      = &_output_shifts;
    }

    DWCWeightsKernelInfo dwc_weights_info;
    dwc_weights_info.n0 = (depth_multiplier == 1) ? 8 : 1;
    DWCKernelInfo dwc_info;
    dwc_info.activation_info = act_info;
    _dwc_native_kernel->configure(compile_context, input_to_use, weights_to_use, biases, output_to_use,
                                  dwc_weights_info, dwc_info, conv_info, depth_multiplier, dilation,
                                  output_multipliers_to_use, output_shifts_to_use);

    if(_needs_permute)
    {
        _permuted_input.allocator()->allocate();

        // Configure the function to transform the convoluted output to NCHW format
        _permuted_output.info()->set_data_layout(DataLayout::NCHW);
        _permute_output_to_nchw.configure(compile_context, &_permuted_output, output, PermutationVector(1U, 2U, 0U));
        _permuted_output.allocator()->allocate();
    }

    if(_is_quantized)
    {
        _output_multipliers.allocator()->allocate();
        _output_shifts.allocator()->allocate();
    }
}

Status CLDepthwiseConvolutionLayer::CLDepthwiseConvolutionLayerGeneric::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
                                                                                 const PadStrideInfo &conv_info,
                                                                                 unsigned int depth_multiplier, const ActivationLayerInfo &act_info, const Size2D &dilation)
{
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
    const size_t idx_w = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_h = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT);

    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(idx_w) + (weights->dimension(idx_w) - 1) * (dilation.x() - 1) > input->dimension(idx_w) + conv_info.pad_left() + conv_info.pad_right());
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(idx_h) + (weights->dimension(idx_h) - 1) * (dilation.y() - 1) > input->dimension(idx_h) + conv_info.pad_top() + conv_info.pad_bottom());

    DWCWeightsKernelInfo dwc_weights_info;
    dwc_weights_info.n0 = (depth_multiplier == 1) ? 8 : 1;
    DWCKernelInfo dwc_info;
    dwc_info.activation_info = act_info;

    const bool needs_permute = input->data_layout() == DataLayout::NCHW;

    const bool is_quantized = is_data_type_quantized(input->data_type());

    TensorInfo output_multipliers_shifts_info(TensorInfo(TensorShape(1U), 1, DataType::S32));
    if(is_quantized)
    {
        if(is_data_type_quantized_per_channel(weights->data_type()))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::QSYMM8_PER_CHANNEL);

            const size_t idx_c = get_data_layout_dimension_index(weights->data_layout(), DataLayoutDimension::CHANNEL);
            output_multipliers_shifts_info.set_tensor_shape(TensorShape(weights->dimension(idx_c)));
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
        }
    }

    if(needs_permute)
    {
        TensorShape           permuted_input_shape   = input->tensor_shape();
        TensorShape           permuted_weights_shape = weights->tensor_shape();
        const ConvolutionInfo info{ conv_info, depth_multiplier, ActivationLayerInfo(), dilation };
        TensorShape           permuted_output_shape = shape_calculator::compute_depthwise_convolution_shape(*input, *weights, info);

        permute(permuted_input_shape, PermutationVector(2U, 0U, 1U));
        permute(permuted_weights_shape, PermutationVector(2U, 0U, 1U));
        permute(permuted_output_shape, PermutationVector(2U, 0U, 1U));

        const TensorInfo permuted_input   = input->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(permuted_input_shape).set_data_layout(DataLayout::NHWC);
        const TensorInfo permuted_weights = weights->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(permuted_weights_shape).set_data_layout(DataLayout::NHWC);
        const TensorInfo permuted_output  = output->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(permuted_output_shape).set_data_layout(DataLayout::NHWC);

        ARM_COMPUTE_RETURN_ON_ERROR(CLPermute::validate(input, &permuted_input, PermutationVector(2U, 0U, 1U)));
        ARM_COMPUTE_RETURN_ON_ERROR(CLPermute::validate(weights, &permuted_weights, PermutationVector(2U, 0U, 1U)));
        ARM_COMPUTE_RETURN_ON_ERROR(CLDepthwiseConvolutionLayerNativeKernel::validate(&permuted_input, &permuted_weights, biases, &permuted_output, dwc_weights_info,
                                                                                      dwc_info, conv_info, depth_multiplier, dilation,
                                                                                      &output_multipliers_shifts_info, &output_multipliers_shifts_info));
        ARM_COMPUTE_RETURN_ON_ERROR(CLPermute::validate(&permuted_output, output, PermutationVector(1U, 2U, 0U)));
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLDepthwiseConvolutionLayerNativeKernel::validate(input, weights, biases, output, dwc_weights_info, dwc_info, conv_info, depth_multiplier,
                                                                                      dilation, &output_multipliers_shifts_info, &output_multipliers_shifts_info));
    }
    return Status{};
}

void CLDepthwiseConvolutionLayer::CLDepthwiseConvolutionLayerGeneric::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_memory_group);

    if(_needs_permute)
    {
        _permute_input_to_nhwc.run();
    }
    CLScheduler::get().enqueue(*_dwc_native_kernel);
    if(_needs_permute)
    {
        _permute_output_to_nchw.run();
    }
}

void CLDepthwiseConvolutionLayer::CLDepthwiseConvolutionLayerGeneric::prepare()
{
    if(!_is_prepared)
    {
        if(_is_quantized)
        {
            _output_multipliers.map();
            _output_shifts.map();
            const unsigned int idx_ofms = _needs_permute ? 2 : 0;
            quantization::compute_quantized_multipliers_and_shifts(_input->info(),
                                                                   _original_weights->info(),
                                                                   _output->info(),
                                                                   idx_ofms,
                                                                   reinterpret_cast<int32_t *>(_output_multipliers.ptr_to_element(Coordinates(0))),
                                                                   reinterpret_cast<int32_t *>(_output_shifts.ptr_to_element(Coordinates(0))));
            _output_multipliers.unmap();
            _output_shifts.unmap();
        }

        if(_needs_permute)
        {
            ARM_COMPUTE_ERROR_ON(!_original_weights->is_used());

            _permuted_weights.allocator()->allocate();
            _permute_weights_to_nhwc.run();
            _original_weights->mark_as_unused();
        }
        _is_prepared = true;
    }
}

CLDepthwiseConvolutionLayer::CLDepthwiseConvolutionLayerInternal3x3::CLDepthwiseConvolutionLayerInternal3x3(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)),
      _kernel(nullptr),
      _border_handler(std::make_unique<CLFillBorderKernel>()),
      _permute_input_to_nchw(),
      _permute_weights_to_nchw(),
      _permute_output_to_nhwc(),
      _reshape_weights(std::make_unique<CLDepthwiseConvolutionLayerReshapeWeightsKernel>()),
      _permuted_input(),
      _permuted_weights(),
      _permuted_output(),
      _output_multipliers(),
      _output_shifts(),
      _original_weights(nullptr),
      _input(nullptr),
      _output(nullptr),
      _needs_permute(false),
      _needs_weights_reshape(false),
      _is_prepared(false),
      _is_quantized(false),
      _is_nhwc(false)
{
}

void CLDepthwiseConvolutionLayer::CLDepthwiseConvolutionLayerInternal3x3::configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output,
                                                                                    const PadStrideInfo &conv_info, unsigned int depth_multiplier, ActivationLayerInfo act_info, const Size2D &dilation)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, weights, biases, output, conv_info, depth_multiplier, act_info, dilation);
}

void CLDepthwiseConvolutionLayer::CLDepthwiseConvolutionLayerInternal3x3::configure(const CLCompileContext &compile_context, ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases,
                                                                                    ICLTensor           *output,
                                                                                    const PadStrideInfo &conv_info, unsigned int depth_multiplier, ActivationLayerInfo act_info, const Size2D &dilation)
{
    const GPUTarget gpu_target = CLScheduler::get().target();

    // Perform validation step
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(CLDepthwiseConvolutionLayerInternal3x3::validate(input->info(),
                                                                                weights->info(),
                                                                                biases != nullptr ? biases->info() : nullptr,
                                                                                output->info(),
                                                                                conv_info,
                                                                                depth_multiplier,
                                                                                act_info,
                                                                                gpu_target,
                                                                                dilation));

    _is_nhwc               = input->info()->data_layout() == DataLayout::NHWC;
    _is_quantized          = is_data_type_quantized_asymmetric(input->info()->data_type());
    _needs_permute         = _is_nhwc && (depth_multiplier > 1);
    _needs_weights_reshape = _is_nhwc && (depth_multiplier == 1) && _is_quantized;

    _is_prepared      = false;
    _original_weights = weights;
    _input            = input;
    _output           = output;

    ICLTensor       *input_to_use   = input;
    const ICLTensor *weights_to_use = weights;
    ICLTensor       *output_to_use  = output;

    const bool is_quantized_per_channel = is_data_type_quantized_per_channel(weights->info()->data_type());
    const bool is_stride_1              = ((conv_info.stride().first == conv_info.stride().second) && (conv_info.stride().first == 1));
    const bool is_dot8_supported        = dot8_supported(CLKernelLibrary::get().get_device()) && !is_quantized_per_channel;
    const bool is_stride_1_dilation_1   = (is_stride_1 && dilation.x() == 1 && dilation.y() == 1);

    DepthwiseConvolutionReshapeInfo info;
    info.c0        = 4;
    info.transpose = is_stride_1_dilation_1 && is_dot8_supported;

    if(_needs_permute)
    {
        _memory_group.manage(&_permuted_input);
        _memory_group.manage(&_permuted_output);

        // Configure the function to transform the input tensor from NHWC -> NCHW
        _permute_input_to_nchw.configure(compile_context, input, &_permuted_input, PermutationVector(1U, 2U, 0U));
        _permuted_input.info()->set_data_layout(DataLayout::NCHW);

        // Configure the function to transform the weights tensor from HWI -> IHW
        _permute_weights_to_nchw.configure(compile_context, weights, &_permuted_weights, PermutationVector(1U, 2U, 0U));
        _permuted_weights.info()->set_data_layout(DataLayout::NCHW);
        _permuted_output.info()->set_quantization_info(output->info()->quantization_info());

        input_to_use   = &_permuted_input;
        weights_to_use = &_permuted_weights;
        output_to_use  = &_permuted_output;

        _kernel = std::make_unique<CLDepthwiseConvolutionLayer3x3NCHWKernel>();
    }
    else if(_is_nhwc)
    {
        if(_needs_weights_reshape)
        {
            _reshape_weights->configure(compile_context, weights, &_permuted_weights, info);
            weights_to_use = &_permuted_weights;
        }
        _kernel = std::make_unique<CLDepthwiseConvolutionLayer3x3NHWCKernel>();
    }
    else
    {
        _kernel = std::make_unique<CLDepthwiseConvolutionLayer3x3NCHWKernel>();
    }

    CLTensor *output_multipliers_to_use = nullptr;
    CLTensor *output_shifts_to_use      = nullptr;
    if(_is_quantized)
    {
        const size_t idx_c       = get_data_layout_dimension_index(weights->info()->data_layout(), DataLayoutDimension::CHANNEL);
        const size_t num_filters = (is_quantized_per_channel) ? weights->info()->dimension(idx_c) : 1;

        _output_multipliers.allocator()->init(TensorInfo(TensorShape(num_filters), 1, DataType::S32));
        _output_shifts.allocator()->init(TensorInfo(TensorShape(num_filters), 1, DataType::S32));

        output_multipliers_to_use = &_output_multipliers;
        output_shifts_to_use      = &_output_shifts;
    }

    // Configure kernel
    _kernel->set_target(gpu_target);
    _kernel->configure(compile_context, input_to_use, weights_to_use, biases, output_to_use, conv_info, depth_multiplier,
                       act_info, dilation, output_multipliers_to_use, output_shifts_to_use);

    if(_is_quantized)
    {
        _output_multipliers.allocator()->allocate();
        _output_shifts.allocator()->allocate();
    }

    // Permute output if needed
    if(_needs_permute)
    {
        // Configure the function to transform the convoluted output to ACL's native ordering format NCHW
        _permuted_output.info()->set_data_layout(DataLayout::NCHW);
        _permute_output_to_nhwc.configure(compile_context, &_permuted_output, output, PermutationVector(2U, 0U, 1U));

        // Allocate tensors
        _permuted_input.allocator()->allocate();
        _permuted_output.allocator()->allocate();
    }
    // Configure border handler
    PixelValue &&zero_value(0.f);
    if(is_data_type_quantized_asymmetric(input->info()->data_type()))
    {
        zero_value = PixelValue(static_cast<uint8_t>(input->info()->quantization_info().uniform().offset));
    }
    _border_handler->configure(compile_context, input_to_use, _kernel->border_size(), BorderMode::CONSTANT, zero_value);
}

Status CLDepthwiseConvolutionLayer::CLDepthwiseConvolutionLayerInternal3x3::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
                                                                                     const PadStrideInfo &conv_info, unsigned int depth_multiplier, ActivationLayerInfo act_info, GPUTarget gpu_target, const Size2D &dilation)
{
    return validate_arguments_3x3(input, weights, biases, output, conv_info, depth_multiplier, act_info, gpu_target, dilation);
}

void CLDepthwiseConvolutionLayer::CLDepthwiseConvolutionLayerInternal3x3::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_memory_group);

    if(_needs_permute)
    {
        _permute_input_to_nchw.run();
    }
    CLScheduler::get().enqueue(*_border_handler);
    CLScheduler::get().enqueue(*_kernel);

    if(_needs_permute)
    {
        _permute_output_to_nhwc.run();
    }
}

void CLDepthwiseConvolutionLayer::CLDepthwiseConvolutionLayerInternal3x3::prepare()
{
    if(!_is_prepared)
    {
        if(_is_quantized)
        {
            _output_multipliers.map();
            _output_shifts.map();
            const unsigned int idx_ofms = _is_nhwc ? 0 : 2;
            quantization::compute_quantized_multipliers_and_shifts(_input->info(),
                                                                   _original_weights->info(),
                                                                   _output->info(),
                                                                   idx_ofms,
                                                                   reinterpret_cast<int32_t *>(_output_multipliers.ptr_to_element(Coordinates(0))),
                                                                   reinterpret_cast<int32_t *>(_output_shifts.ptr_to_element(Coordinates(0))));
            _output_multipliers.unmap();
            _output_shifts.unmap();
        }

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
            CLScheduler::get().enqueue(*_reshape_weights);
            _original_weights->mark_as_unused();
        }
        _is_prepared = true;
    }
}

CLDepthwiseConvolutionLayer::CLDepthwiseConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_manager(std::move(memory_manager)), _depth_conv_func(DepthwiseConvolutionFunction::GENERIC), _func_3x3(), _func_generic()
{
}

void CLDepthwiseConvolutionLayer::configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier,
                                            ActivationLayerInfo act_info, const Size2D &dilation)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, weights, biases, output, conv_info, depth_multiplier, act_info, dilation);
}

void CLDepthwiseConvolutionLayer::configure(const CLCompileContext &compile_context, ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output,
                                            const PadStrideInfo &conv_info,
                                            unsigned int         depth_multiplier,
                                            ActivationLayerInfo act_info, const Size2D &dilation)
{
    const GPUTarget gpu_target = CLScheduler::get().target();
    _depth_conv_func           = get_depthwiseconvolution_function(input->info(), weights->info(), (biases != nullptr) ? biases->info() : nullptr, output->info(), conv_info, depth_multiplier, act_info,
                                                                   dilation, gpu_target);
    switch(_depth_conv_func)
    {
        case DepthwiseConvolutionFunction::OPTIMIZED:
            _func_3x3.set_memory_group(_memory_manager);
            _func_3x3.configure(compile_context, input, weights, biases, output, conv_info, depth_multiplier, act_info, dilation);
            break;
        case DepthwiseConvolutionFunction::GENERIC:
        {
            _func_generic.set_memory_group(_memory_manager);
            _func_generic.configure(compile_context, input, weights, biases, output, conv_info, depth_multiplier, act_info, dilation);
        }
        break;
        default:
            ARM_COMPUTE_ERROR("Unsupported DepthwiseConvolutionFunction");
    }
}

Status CLDepthwiseConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                             unsigned int depth_multiplier, ActivationLayerInfo act_info, const Size2D &dilation)
{
    const GPUTarget              gpu_target      = CLScheduler::get().target();
    DepthwiseConvolutionFunction depth_conv_func = get_depthwiseconvolution_function(input, weights, biases, output, conv_info, depth_multiplier, act_info, dilation, gpu_target);
    switch(depth_conv_func)
    {
        case DepthwiseConvolutionFunction::OPTIMIZED:
            return CLDepthwiseConvolutionLayerInternal3x3::validate(input, weights, biases, output, conv_info, depth_multiplier, act_info, gpu_target, dilation);
        case DepthwiseConvolutionFunction::GENERIC:
            return CLDepthwiseConvolutionLayerGeneric::validate(input, weights, biases, output, conv_info, depth_multiplier, act_info, dilation);
        default:
            ARM_COMPUTE_ERROR("Unsupported DepthwiseConvolutionFunction");
    }
}

DepthwiseConvolutionFunction CLDepthwiseConvolutionLayer::get_depthwiseconvolution_function(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
                                                                                            const PadStrideInfo &conv_info,
                                                                                            unsigned int depth_multiplier, ActivationLayerInfo act_info, const Size2D &dilation, GPUTarget gpu_target)
{
    if(bool(CLDepthwiseConvolutionLayerInternal3x3::validate(input, weights, biases, output, conv_info, depth_multiplier, act_info, gpu_target, dilation)) && (is_data_type_float(input->data_type())
            || get_arch_from_target(gpu_target) == GPUTarget::MIDGARD))
    {
        return DepthwiseConvolutionFunction::OPTIMIZED;
    }
    else
    {
        return DepthwiseConvolutionFunction::GENERIC;
    }
}

void CLDepthwiseConvolutionLayer::run()
{
    switch(_depth_conv_func)
    {
        case DepthwiseConvolutionFunction::OPTIMIZED:
            _func_3x3.run();
            break;
        case DepthwiseConvolutionFunction::GENERIC:
            _func_generic.run();
            break;
        default:
            ARM_COMPUTE_ERROR("DepthwiseConvolutionFunction not properly configured");
    }
}

void CLDepthwiseConvolutionLayer::prepare()
{
    switch(_depth_conv_func)
    {
        case DepthwiseConvolutionFunction::OPTIMIZED:
            _func_3x3.prepare();
            break;
        case DepthwiseConvolutionFunction::GENERIC:
            _func_generic.prepare();
            break;
        default:
            ARM_COMPUTE_ERROR("DepthwiseConvolutionFunction not properly configured");
    }
}
} // namespace arm_compute
