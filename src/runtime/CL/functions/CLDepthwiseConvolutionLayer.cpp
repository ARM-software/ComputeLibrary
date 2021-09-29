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

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/core/CL/kernels/CLDepthwiseConvolutionLayerNativeKernel.h"

#include "src/common/utils/Log.h"

namespace arm_compute
{
using namespace arm_compute::misc;
using namespace arm_compute::misc::shape_calculator;

namespace
{
bool export_weights_to_cl_image_heuristic(const ITensorInfo *weights, unsigned int depth_multiplier, GPUTarget gpu_target)
{
    if(!export_weights_to_cl_image(weights))
    {
        return false;
    }

    const size_t idx_w    = get_data_layout_dimension_index(weights->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_h    = get_data_layout_dimension_index(weights->data_layout(), DataLayoutDimension::HEIGHT);
    const size_t kernel_w = weights->tensor_shape()[idx_w];
    const size_t kernel_h = weights->tensor_shape()[idx_h];

    if((kernel_w == 1) && (kernel_h == 1))
    {
        return false;
    }

    if(depth_multiplier > 1)
    {
        return false;
    }

    if(gpu_target == GPUTarget::G71 || get_arch_from_target(gpu_target) == GPUTarget::MIDGARD)
    {
        return false;
    }

    return true;
}

void initialize_dwc_native_compute_info(DWCComputeKernelInfo &dwc_compute_info, const ITensorInfo *weights, const PadStrideInfo &conv_info, const Size2D &dilation, unsigned int depth_multiplier,
                                        GPUTarget gpu_target)
{
    if(!is_data_type_float(weights->data_type()))
    {
        dwc_compute_info.export_weights_to_cl_image = false;
        dwc_compute_info.n0                         = (depth_multiplier == 1) ? 4 : 1;
        if(conv_info.stride().first == 1 && dilation.x() == 1 && depth_multiplier == 1)
        {
            dwc_compute_info.m0 = 2;
        }
        else
        {
            dwc_compute_info.m0 = 1;
        }

        return;
    }

    // Floating point path

    // First check if we can export to cl_image.
    dwc_compute_info.export_weights_to_cl_image = export_weights_to_cl_image_heuristic(weights, depth_multiplier, gpu_target);

    // Set n0
    if(depth_multiplier == 1)
    {
        if(dwc_compute_info.export_weights_to_cl_image == false && weights->data_type() == DataType::F16)
        {
            dwc_compute_info.n0 = 8;
        }
        else
        {
            dwc_compute_info.n0 = 4;
        }
    }
    else
    {
        dwc_compute_info.n0 = 1;
    }

    dwc_compute_info.n0 = adjust_vec_size(dwc_compute_info.n0, weights->dimension(0));

    // Set m0 only if stride_x == 1 and dilation_x == 1
    if(conv_info.stride().first == 1 && dilation.x() == 1)
    {
        const size_t idx_w    = get_data_layout_dimension_index(weights->data_layout(), DataLayoutDimension::WIDTH);
        const size_t kernel_w = weights->tensor_shape()[idx_w];

        dwc_compute_info.m0 = (kernel_w >= 9) || (kernel_w == 1) ? 1 : 2;
    }
    else
    {
        dwc_compute_info.m0 = 1;
    }
    return;
}

} // namespace

CLDepthwiseConvolutionLayer::CLDepthwiseConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
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

void CLDepthwiseConvolutionLayer::configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                                            unsigned int depth_multiplier, ActivationLayerInfo act_info, const Size2D &dilation)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, weights, biases, output, conv_info, depth_multiplier, act_info, dilation);
}

void CLDepthwiseConvolutionLayer::configure(const CLCompileContext &compile_context, ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases,
                                            ICLTensor *output, const PadStrideInfo &conv_info,
                                            unsigned int depth_multiplier, ActivationLayerInfo act_info, const Size2D &dilation)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights);
    ARM_COMPUTE_ERROR_THROW_ON(CLDepthwiseConvolutionLayer::validate(input->info(),
                                                                     weights->info(),
                                                                     biases != nullptr ? biases->info() : nullptr,
                                                                     output != nullptr ? output->info() : input->info(),
                                                                     conv_info,
                                                                     depth_multiplier,
                                                                     act_info,
                                                                     dilation));
    ARM_COMPUTE_LOG_PARAMS(input, weights, biases, output, conv_info, depth_multiplier, act_info, dilation);

    _is_quantized     = is_data_type_quantized(input->info()->data_type());
    _is_prepared      = false;
    _original_weights = weights;
    _input            = input;
    _output           = output;
    _needs_permute    = input->info()->data_layout() == DataLayout::NCHW;

    const GPUTarget gpu_target = CLScheduler::get().target();

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

    DWCComputeKernelInfo dwc_native_compute_info;
    initialize_dwc_native_compute_info(dwc_native_compute_info, weights_to_use->info(), conv_info, dilation, depth_multiplier, gpu_target);

    const ConvolutionInfo conv_kernel_info{ conv_info, depth_multiplier, act_info, dilation };

    _dwc_native_kernel->configure(compile_context, input_to_use, weights_to_use, biases, output_to_use,
                                  dwc_native_compute_info, conv_kernel_info, output_multipliers_to_use, output_shifts_to_use);

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

Status CLDepthwiseConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
                                             const PadStrideInfo &conv_info,
                                             unsigned int depth_multiplier, ActivationLayerInfo act_info, const Size2D &dilation)
{
    const bool in_place = input == output || output == nullptr;
    if(in_place)
    {
        output = input;
    }
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
    const size_t idx_w = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_h = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT);

    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(idx_w) + (weights->dimension(idx_w) - 1) * (dilation.x() - 1) > input->dimension(idx_w) + conv_info.pad_left() + conv_info.pad_right());
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(idx_h) + (weights->dimension(idx_h) - 1) * (dilation.y() - 1) > input->dimension(idx_h) + conv_info.pad_top() + conv_info.pad_bottom());

    const GPUTarget gpu_target = CLScheduler::get().target();

    const ConvolutionInfo conv_kernel_info{ conv_info, depth_multiplier, act_info, dilation };

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
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(in_place, "In-place is supported only with NHWC data layout");
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

        DWCComputeKernelInfo dwc_native_compute_info;
        initialize_dwc_native_compute_info(dwc_native_compute_info, &permuted_weights, conv_info, dilation, depth_multiplier, gpu_target);

        ARM_COMPUTE_RETURN_ON_ERROR(CLDepthwiseConvolutionLayerNativeKernel::validate(&permuted_input, &permuted_weights, biases, &permuted_output,
                                                                                      dwc_native_compute_info, conv_kernel_info, &output_multipliers_shifts_info, &output_multipliers_shifts_info));
        ARM_COMPUTE_RETURN_ON_ERROR(CLPermute::validate(&permuted_output, output, PermutationVector(1U, 2U, 0U)));
    }
    else
    {
        DWCComputeKernelInfo dwc_native_compute_info;
        initialize_dwc_native_compute_info(dwc_native_compute_info, weights, conv_info, dilation, depth_multiplier, gpu_target);
        ARM_COMPUTE_RETURN_ON_ERROR(CLDepthwiseConvolutionLayerNativeKernel::validate(input, weights, biases, output, dwc_native_compute_info, conv_kernel_info, &output_multipliers_shifts_info,
                                                                                      &output_multipliers_shifts_info));
    }
    return Status{};
}

void CLDepthwiseConvolutionLayer::run()
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

void CLDepthwiseConvolutionLayer::prepare()
{
    if(!_is_prepared)
    {
        if(_is_quantized)
        {
            _output_multipliers.map();
            _output_shifts.map();
            quantization::compute_quantized_multipliers_and_shifts(_input->info(),
                                                                   _original_weights->info(),
                                                                   _output != nullptr ? _output->info() : _input->info(),
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
} // namespace arm_compute
