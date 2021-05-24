/*
 * Copyright (c) 2021 Arm Limited.
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
#include "src/runtime/cpu/operators/CpuDepthwiseConv2d.h"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/InfoHelpers.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/core/cpu/kernels/CpuDepthwiseConv2dNativeKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace
{
Status validate_arguments_optimized(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const ConvolutionInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, weights, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    if(!is_data_type_quantized_per_channel(weights->data_type()))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, weights);
    }
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_layout() == DataLayout::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON(info.dilation.x() < 1 || info.dilation.y() < 1);
    const size_t idx_w = get_data_layout_dimension_index(src->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_h = get_data_layout_dimension_index(src->data_layout(), DataLayoutDimension::HEIGHT);
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(idx_w) + (weights->dimension(idx_w) - 1) * (info.dilation.x() - 1) > src->dimension(idx_w) + info.pad_stride_info.pad_left() +
                                info.pad_stride_info.pad_right());
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(idx_h) + (weights->dimension(idx_h) - 1) * (info.dilation.y() - 1) > src->dimension(idx_h) + info.pad_stride_info.pad_top() +
                                info.pad_stride_info.pad_bottom());

    if(biases != nullptr)
    {
        const unsigned int channel_idx = get_data_layout_dimension_index(src->data_layout(), DataLayoutDimension::CHANNEL);
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != weights->dimension(channel_idx));
    }

    ARM_COMPUTE_RETURN_ON_ERROR(CpuDepthwiseConv2dAssemblyDispatch::validate(src, weights, biases, dst, info));

    //Validate Activation Layer
    if(info.act_info.enabled())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CpuActivation::validate(dst, nullptr, info.act_info));
    }
    return Status{};
}
} // namespace

CpuDepthwiseConv2d::CpuDepthwiseConv2dOptimizedInternal::CpuDepthwiseConv2dOptimizedInternal()
    : _dwc_optimized_func(nullptr), _permute_input(nullptr), _permute_weights(nullptr), _permute_output(nullptr), _activationlayer_function(nullptr), _has_bias(false), _is_quantized(false),
      _is_nchw(true), _permute(false), _is_activationlayer_enabled(false), _is_prepared(false)
{
}

void CpuDepthwiseConv2d::CpuDepthwiseConv2dOptimizedInternal::configure(ITensorInfo           *src,
                                                                        const ITensorInfo     *weights,
                                                                        const ITensorInfo     *biases,
                                                                        ITensorInfo           *dst,
                                                                        const ConvolutionInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, weights, dst);
    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(CpuDepthwiseConv2dOptimizedInternal::validate(src, weights, (biases == nullptr) ? nullptr : biases,
                                                                             dst, info));

    _is_quantized = is_data_type_quantized_asymmetric(src->data_type());
    _has_bias     = biases != nullptr;
    _is_nchw      = src->data_layout() == DataLayout::NCHW;
    _permute      = _is_nchw;
    _is_prepared  = false;

    // Configure pipeline
    ActivationLayerInfo act_info_to_use = ActivationLayerInfo();
    const bool          is_relu         = arm_compute::utils::info_helpers::is_relu(info.act_info);
    const bool          is_relu6        = arm_compute::utils::info_helpers::is_relu6(info.act_info);
    _is_activationlayer_enabled         = info.act_info.enabled() && !(is_relu || is_relu6);

    if(!_is_activationlayer_enabled)
    {
        act_info_to_use = info.act_info;
    }

    _dwc_optimized_func = std::make_unique<CpuDepthwiseConv2dAssemblyDispatch>();
    if(_is_nchw)
    {
        _permute_input   = std::make_unique<cpu::CpuPermute>();
        _permute_weights = std::make_unique<cpu::CpuPermute>();
        _permute_output  = std::make_unique<cpu::CpuPermute>();

        auto input_perm   = std::make_unique<TensorInfo>();
        auto weights_perm = std::make_unique<TensorInfo>();
        auto output_perm  = std::make_unique<TensorInfo>();

        // Configure the function to transform the input tensor from NCHW -> NHWC
        _permute_input->configure(src, input_perm.get(), PermutationVector(2U, 0U, 1U));
        input_perm->set_data_layout(DataLayout::NHWC);

        // Configure the function to transform the weights tensor from IHW -> HWI
        _permute_weights->configure(weights, weights_perm.get(), PermutationVector(2U, 0U, 1U));
        weights_perm->set_data_layout(DataLayout::NHWC);

        output_perm->set_data_layout(DataLayout::NHWC);
        output_perm->set_quantization_info(dst->quantization_info());

        // Configure optimized depthwise
        _dwc_optimized_func->configure(input_perm.get(), weights_perm.get(), biases, output_perm.get(), info);

        // Configure the function to transform the convoluted output to ACL's native ordering format NCHW
        output_perm->set_data_layout(DataLayout::NHWC);
        _permute_output->configure(output_perm.get(), dst, PermutationVector(1U, 2U, 0U));
    }
    else
    {
        _dwc_optimized_func->configure(src, weights, biases, dst, info);
    }

    // Configure activation
    if(_is_activationlayer_enabled)
    {
        _activationlayer_function = std::make_unique<cpu::CpuActivation>();
        _activationlayer_function->configure(dst, nullptr, info.act_info);
    }
}

Status CpuDepthwiseConv2d::CpuDepthwiseConv2dOptimizedInternal::validate(const ITensorInfo     *src,
                                                                         const ITensorInfo     *weights,
                                                                         const ITensorInfo     *biases,
                                                                         const ITensorInfo     *dst,
                                                                         const ConvolutionInfo &info)
{
    return validate_arguments_optimized(src, weights, biases, dst, info);
}

void CpuDepthwiseConv2d::CpuDepthwiseConv2dOptimizedInternal::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
    prepare(tensors);

    auto bias           = tensors.get_const_tensor(TensorType::ACL_SRC_2);
    auto dst            = tensors.get_tensor(TensorType::ACL_DST_0);
    auto workspace      = tensors.get_tensor(TensorType::ACL_INT_3);
    auto packed_weights = tensors.get_tensor(TensorType::ACL_INT_4);

    // Permute input
    if(_permute)
    {
        ITensorPack pack;
        auto        src      = tensors.get_const_tensor(TensorType::ACL_SRC_0);
        auto        src_perm = tensors.get_tensor(TensorType::ACL_INT_0);
        pack.add_tensor(TensorType::ACL_SRC, src);
        pack.add_tensor(TensorType::ACL_DST, src_perm);
        _permute_input->run(pack);
    }

    // Run assembly function
    if(_is_nchw)
    {
        auto src_perm     = tensors.get_tensor(TensorType::ACL_INT_0);
        auto weights_perm = tensors.get_tensor(TensorType::ACL_INT_1);
        auto dst_perm     = tensors.get_tensor(TensorType::ACL_INT_2);

        ITensorPack pack;
        pack.add_tensor(TensorType::ACL_SRC_0, src_perm);
        pack.add_tensor(TensorType::ACL_SRC_1, weights_perm);
        pack.add_tensor(TensorType::ACL_SRC_2, bias);
        pack.add_tensor(TensorType::ACL_INT_0, workspace);
        pack.add_tensor(TensorType::ACL_INT_1, packed_weights);
        pack.add_tensor(TensorType::ACL_DST, dst_perm);
        _dwc_optimized_func->run(pack);
    }
    else
    {
        auto src     = tensors.get_tensor(TensorType::ACL_SRC_0);
        auto weights = tensors.get_tensor(TensorType::ACL_SRC_1);
        auto dst     = tensors.get_tensor(TensorType::ACL_DST);

        ITensorPack pack;
        pack.add_tensor(TensorType::ACL_SRC_0, src);
        pack.add_tensor(TensorType::ACL_SRC_1, weights);
        pack.add_tensor(TensorType::ACL_SRC_2, bias);
        pack.add_tensor(TensorType::ACL_INT_0, workspace);
        pack.add_tensor(TensorType::ACL_INT_1, packed_weights);
        pack.add_tensor(TensorType::ACL_DST, dst);
        _dwc_optimized_func->run(pack);
    }

    // Permute output
    if(_is_nchw)
    {
        ITensorPack pack;
        auto        dst_perm = tensors.get_tensor(TensorType::ACL_INT_2);
        pack.add_tensor(TensorType::ACL_SRC, dst_perm);
        pack.add_tensor(TensorType::ACL_DST, dst);
        _permute_output->run(pack);
    }

    // Run activation
    if(_is_activationlayer_enabled)
    {
        ITensorPack pack;
        pack.add_tensor(TensorType::ACL_SRC, dst);
        pack.add_tensor(TensorType::ACL_DST, dst);
        _activationlayer_function->run(pack);
    }
}

void CpuDepthwiseConv2d::CpuDepthwiseConv2dOptimizedInternal::prepare(ITensorPack &tensors)
{
    if(!_is_prepared)
    {
        auto weights        = tensors.get_const_tensor(TensorType::ACL_SRC_1);
        auto bias           = tensors.get_const_tensor(TensorType::ACL_SRC_2);
        auto packed_weights = tensors.get_tensor(TensorType::ACL_INT_4);

        // Permute weights
        if(_permute)
        {
            auto permuted_weights = tensors.get_tensor(TensorType::ACL_INT_1);

            ITensorPack pack;
            pack.add_tensor(TensorType::ACL_SRC, weights);
            pack.add_tensor(TensorType::ACL_DST, permuted_weights);
            _permute_weights->run(pack);

            weights->mark_as_unused();

            ITensorPack pack_opt;
            pack_opt.add_const_tensor(TensorType::ACL_SRC_1, permuted_weights);
            pack_opt.add_tensor(TensorType::ACL_SRC_2, bias);
            pack_opt.add_tensor(TensorType::ACL_INT_1, packed_weights);

            // Prepare optimized function
            _dwc_optimized_func->prepare(pack_opt);
        }
        else
        {
            ITensorPack pack_opt;
            pack_opt.add_tensor(TensorType::ACL_SRC_1, weights);
            pack_opt.add_tensor(TensorType::ACL_SRC_2, bias);
            pack_opt.add_tensor(TensorType::ACL_INT_1, packed_weights);

            // Prepare optimized function
            _dwc_optimized_func->prepare(pack_opt);
        }

        _is_prepared = true;
    }
}

CpuDepthwiseConv2d::CpuDepthwiseConv2dGeneric::CpuDepthwiseConv2dGeneric()
    : _depthwise_conv_kernel(nullptr), _permute_input(nullptr), _permute_weights(nullptr), _permute_output(nullptr), _activationlayer_function(nullptr), _is_nchw(true), _is_prepared(false),
      _is_activationlayer_enabled(false)
{
}

void CpuDepthwiseConv2d::CpuDepthwiseConv2dGeneric::configure(ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *dst, const ConvolutionInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, weights, dst);
    ARM_COMPUTE_ERROR_THROW_ON(CpuDepthwiseConv2d::validate(src, weights, (biases == nullptr) ? nullptr : biases,
                                                            dst, info));

    _is_nchw     = src->data_layout() == DataLayout::NCHW;
    _is_prepared = !_is_nchw;

    ITensorInfo       *input_to_use   = src;
    const ITensorInfo *weights_to_use = weights;
    ITensorInfo       *output_to_use  = dst;

    auto input_perm   = std::make_unique<TensorInfo>();
    auto weights_perm = std::make_unique<TensorInfo>();
    auto output_perm  = std::make_unique<TensorInfo>(dst->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(TensorShape()));

    if(_is_nchw)
    {
        _permute_input   = std::make_unique<cpu::CpuPermute>();
        _permute_weights = std::make_unique<cpu::CpuPermute>();

        _permute_input->configure(src, input_perm.get(), PermutationVector(2U, 0U, 1U));
        input_perm->set_data_layout(DataLayout::NHWC);
        input_to_use = input_perm.get();

        _permute_weights->configure(weights, weights_perm.get(), PermutationVector(2U, 0U, 1U));
        weights_perm->set_data_layout(DataLayout::NHWC);
        weights_to_use = weights_perm.get();

        output_to_use = output_perm.get();
    }

    _depthwise_conv_kernel = std::make_unique<cpu::kernels::CpuDepthwiseConv2dNativeKernel>();
    _depthwise_conv_kernel->configure(input_to_use, weights_to_use, biases, output_to_use, info);

    if(_is_nchw)
    {
        _permute_output = std::make_unique<cpu::CpuPermute>();
        _permute_output->configure(output_perm.get(), dst, PermutationVector(1U, 2U, 0U));
        output_perm->set_data_layout(DataLayout::NHWC);
    }

    //Configure Activation Layer
    _is_activationlayer_enabled = info.act_info.enabled();
    if(_is_activationlayer_enabled)
    {
        _activationlayer_function = std::make_unique<cpu::CpuActivation>();
        _activationlayer_function->configure(dst, nullptr, info.act_info);
    }
}

Status CpuDepthwiseConv2d::CpuDepthwiseConv2dGeneric::validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst,
                                                               const ConvolutionInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, weights, dst);
    if(src->data_layout() == DataLayout::NCHW)
    {
        TensorShape permuted_input_shape   = src->tensor_shape();
        TensorShape permuted_weights_shape = weights->tensor_shape();
        TensorShape permuted_output_shape  = misc::shape_calculator::compute_depthwise_convolution_shape(*src, *weights, info);
        permute(permuted_input_shape, PermutationVector(2U, 0U, 1U));
        permute(permuted_weights_shape, PermutationVector(2U, 0U, 1U));
        permute(permuted_output_shape, PermutationVector(2U, 0U, 1U));

        const TensorInfo permuted_input   = TensorInfo(src->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(permuted_input_shape).set_data_layout(DataLayout::NHWC));
        const TensorInfo permuted_weights = TensorInfo(weights->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(permuted_weights_shape).set_data_layout(DataLayout::NHWC));
        const TensorInfo permuted_output  = TensorInfo(dst->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(permuted_output_shape).set_data_layout(DataLayout::NCHW));

        ARM_COMPUTE_RETURN_ON_ERROR(CpuPermute::validate(src, &permuted_input, PermutationVector(2U, 0U, 1U)));
        ARM_COMPUTE_RETURN_ON_ERROR(CpuPermute::validate(weights, &permuted_weights, PermutationVector(2U, 0U, 1U)));
        ARM_COMPUTE_RETURN_ON_ERROR(CpuPermute::validate(&permuted_output, dst, PermutationVector(1U, 2U, 0U)));

        ARM_COMPUTE_RETURN_ON_ERROR(cpu::kernels::CpuDepthwiseConv2dNativeKernel::validate(&permuted_input, &permuted_weights, biases, &permuted_output, info));
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(cpu::kernels::CpuDepthwiseConv2dNativeKernel::validate(src, weights, biases, dst, info));
    }

    // Validate Activation Layer
    if(info.act_info.enabled())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CpuActivation::validate(dst, nullptr, info.act_info));
    }

    return Status{};
}

void CpuDepthwiseConv2d::CpuDepthwiseConv2dGeneric::run(ITensorPack &tensors)
{
    auto src     = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    auto weights = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    auto biases  = tensors.get_const_tensor(TensorType::ACL_SRC_2);
    auto dst     = tensors.get_tensor(TensorType::ACL_DST_0);

    if(_is_nchw)
    {
        prepare(tensors);
        auto src_perm     = tensors.get_tensor(TensorType::ACL_INT_0);
        auto weights_perm = tensors.get_tensor(TensorType::ACL_INT_1);
        auto dst_perm     = tensors.get_tensor(TensorType::ACL_INT_2);

        ITensorPack pack;
        pack.add_tensor(TensorType::ACL_SRC, src);
        pack.add_tensor(TensorType::ACL_DST, src_perm);
        _permute_input->run(pack);

        ITensorPack pack_depth;
        pack_depth.add_const_tensor(TensorType::ACL_SRC_0, src_perm);
        pack_depth.add_const_tensor(TensorType::ACL_SRC_1, weights_perm);
        pack_depth.add_tensor(TensorType::ACL_SRC_2, biases);
        pack_depth.add_tensor(TensorType::ACL_DST, dst_perm);
        NEScheduler::get().schedule_op(_depthwise_conv_kernel.get(), Window::DimY, _depthwise_conv_kernel->window(), pack_depth);
    }
    else
    {
        ITensorPack pack_depth;
        pack_depth.add_tensor(TensorType::ACL_SRC_0, src);
        pack_depth.add_tensor(TensorType::ACL_SRC_1, weights);
        pack_depth.add_tensor(TensorType::ACL_SRC_2, biases);
        pack_depth.add_tensor(TensorType::ACL_DST, dst);
        NEScheduler::get().schedule_op(_depthwise_conv_kernel.get(), Window::DimY, _depthwise_conv_kernel->window(), pack_depth);
    }

    if(_is_nchw)
    {
        ITensorPack pack;
        auto        dst_perm = tensors.get_tensor(TensorType::ACL_INT_2);
        pack.add_tensor(TensorType::ACL_SRC, dst_perm);
        pack.add_tensor(TensorType::ACL_DST, dst);
        _permute_output->run(pack);
    }

    if(_is_activationlayer_enabled)
    {
        ITensorPack pack;
        pack.add_tensor(TensorType::ACL_SRC, dst);
        pack.add_tensor(TensorType::ACL_DST, dst);
        _activationlayer_function->run(pack);
    }
}

void CpuDepthwiseConv2d::CpuDepthwiseConv2dGeneric::prepare(ITensorPack &tensors)
{
    if(!_is_prepared)
    {
        auto weights      = tensors.get_const_tensor(TensorType::ACL_SRC_1);
        auto weights_perm = tensors.get_tensor(TensorType::ACL_INT_1);

        ARM_COMPUTE_ERROR_ON(!weights->is_used());

        ITensorPack pack;
        pack.add_tensor(TensorType::ACL_SRC, weights);
        pack.add_tensor(TensorType::ACL_DST, weights_perm);

        _permute_weights->run(pack);
        weights->mark_as_unused();
        _is_prepared = true;
    }
}

CpuDepthwiseConv2d::CpuDepthwiseConv2d()
    : _depth_conv_func(DepthwiseConvolutionFunction::GENERIC), _func_optimized(), _func_generic()
{
}

void CpuDepthwiseConv2d::configure(ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *dst, const ConvolutionInfo &info)
{
    _depth_conv_func = get_depthwiseconvolution_function(src, weights, (biases != nullptr) ? biases : nullptr, dst, info);
    switch(_depth_conv_func)
    {
        case DepthwiseConvolutionFunction::OPTIMIZED:
            _func_optimized.configure(src, weights, biases, dst, info);
            break;
        case DepthwiseConvolutionFunction::GENERIC:
            _func_generic.configure(src, weights, biases, dst, info);
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported DepthwiseConvolutionFunction");
    }
}

Status CpuDepthwiseConv2d::validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const ConvolutionInfo &info)
{
    DepthwiseConvolutionFunction depth_conv_func = get_depthwiseconvolution_function(src, weights, biases, dst, info);
    switch(depth_conv_func)
    {
        case DepthwiseConvolutionFunction::OPTIMIZED:
            return CpuDepthwiseConv2dOptimizedInternal::validate(src, weights, biases, dst, info);
            break;
        case DepthwiseConvolutionFunction::GENERIC:
            return CpuDepthwiseConv2dGeneric::validate(src, weights, biases, dst, info);
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported DepthwiseConvolutionFunction");
    }
}

DepthwiseConvolutionFunction CpuDepthwiseConv2d::get_depthwiseconvolution_function(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst,
                                                                                   const ConvolutionInfo &info)
{
    if(bool(CpuDepthwiseConv2dOptimizedInternal::validate(src, weights, biases, dst, info)))
    {
        return DepthwiseConvolutionFunction::OPTIMIZED;
    }
    else
    {
        return DepthwiseConvolutionFunction::GENERIC;
    }
}

void CpuDepthwiseConv2d::run(ITensorPack &tensors)
{
    switch(_depth_conv_func)
    {
        case DepthwiseConvolutionFunction::OPTIMIZED:
            _func_optimized.run(tensors);
            break;
        case DepthwiseConvolutionFunction::GENERIC:
            _func_generic.run(tensors);
            break;
        default:
            ARM_COMPUTE_ERROR("DepthwiseConvolutionFunction not properly configured");
    }
}

void CpuDepthwiseConv2d::prepare(ITensorPack &tensors)
{
    switch(_depth_conv_func)
    {
        case DepthwiseConvolutionFunction::OPTIMIZED:
            _func_optimized.prepare(tensors);
            break;
        case DepthwiseConvolutionFunction::GENERIC:
            _func_generic.prepare(tensors);
            break;
        default:
            ARM_COMPUTE_ERROR("DepthwiseConvolutionFunction not properly configured");
    }
}
} // namespace cpu
} // namespace arm_compute
