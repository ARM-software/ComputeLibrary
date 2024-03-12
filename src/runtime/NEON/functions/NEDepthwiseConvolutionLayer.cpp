/*
 * Copyright (c) 2017-2021, 2023 Arm Limited.
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

#include "src/common/utils/Log.h"
#include "src/cpu/operators/CpuDepthwiseConv2d.h"

using namespace arm_compute::misc;
using namespace arm_compute::misc::shape_calculator;

namespace arm_compute
{
NEDepthwiseConvolutionLayer::~NEDepthwiseConvolutionLayer() = default;

struct NEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerOptimizedInternal::Impl
{
    ITensor                                 *src{nullptr};       // SRC_0
    ITensor                                 *dst{nullptr};       // DST_0
    const ITensor                           *weights{nullptr};   // SRC_1
    const ITensor                           *biases{nullptr};    // SRC_2
    Tensor                                   permuted_input{};   // INT_0
    Tensor                                   permuted_weights{}; // INT_1
    Tensor                                   permuted_output{};  // INT_2
    Tensor                                   workspace{};        // INT_3
    Tensor                                   packed_weights{};   // INT_4
    std::shared_ptr<cpu::CpuDepthwiseConv2d> op{nullptr};
    bool                                     is_prepared{false};
    bool                                     permute{false};
};

NEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerOptimizedInternal::NEDepthwiseConvolutionLayerOptimizedInternal(
    std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(memory_manager), _impl(std::make_unique<Impl>())
{
}

void NEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerOptimizedInternal::configure(
    ITensor                   *input,
    const ITensor             *weights,
    const ITensor             *biases,
    ITensor                   *output,
    const PadStrideInfo       &conv_info,
    unsigned int               depth_multiplier,
    const ActivationLayerInfo &act_info,
    const Size2D              &dilation)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);

    bool is_nhwc   = input->info()->data_layout() == DataLayout::NCHW;
    _impl->src     = input;
    _impl->weights = weights;
    _impl->biases  = biases;
    _impl->dst     = output;
    _impl->permute = is_nhwc;

    _impl->op = std::make_unique<cpu::CpuDepthwiseConv2d>();
    ConvolutionInfo info{conv_info, depth_multiplier, act_info, dilation};
    _impl->op->configure(_impl->src->info(), _impl->weights->info(),
                         _impl->biases == nullptr ? nullptr : _impl->biases->info(), _impl->dst->info(), info);

    // Configure pipeline
    ActivationLayerInfo act_info_to_use            = ActivationLayerInfo();
    const bool          is_relu                    = arm_compute::utils::info_helpers::is_relu(act_info);
    const bool          is_relu6                   = arm_compute::utils::info_helpers::is_relu6(act_info);
    bool                is_activationlayer_enabled = act_info.enabled() && !(is_relu || is_relu6);

    if (!is_activationlayer_enabled)
    {
        act_info_to_use = act_info;
    }
    info = ConvolutionInfo{conv_info, depth_multiplier, act_info_to_use, dilation};

    auto dwc_optimized_func = std::make_unique<cpu::CpuDepthwiseConv2dAssemblyDispatch>();

    if (is_nhwc)
    {
        auto permute_input   = std::make_unique<cpu::CpuPermute>();
        auto permute_weights = std::make_unique<cpu::CpuPermute>();
        auto permute_output  = std::make_unique<cpu::CpuPermute>();

        _memory_group.manage(&_impl->permuted_input);
        _memory_group.manage(&_impl->permuted_weights);
        _memory_group.manage(&_impl->permuted_output);

        // Configure the function to transform the input tensor from NCHW -> NHWC
        permute_input->configure(input->info(), _impl->permuted_input.info(), PermutationVector(2U, 0U, 1U));
        _impl->permuted_input.info()->set_data_layout(DataLayout::NHWC);

        // Configure the function to transform the weights tensor from IHW -> HWI
        permute_weights->configure(weights->info(), _impl->permuted_weights.info(), PermutationVector(2U, 0U, 1U));
        _impl->permuted_weights.info()->set_data_layout(DataLayout::NHWC);

        _impl->permuted_output.info()->set_data_layout(DataLayout::NHWC);
        _impl->permuted_output.info()->set_quantization_info(output->info()->quantization_info());

        // Configure optimized depthwise
        dwc_optimized_func->configure(_impl->permuted_input.info(), _impl->permuted_weights.info(),
                                      biases == nullptr ? nullptr : biases->info(), _impl->permuted_output.info(),
                                      info);

        // Configure the function to transform the convoluted output to ACL's native ordering format NCHW
        _impl->permuted_output.info()->set_data_layout(DataLayout::NHWC);
        permute_output->configure(_impl->permuted_output.info(), output->info(), PermutationVector(1U, 2U, 0U));

        _impl->permuted_input.allocator()->allocate();
        _impl->permuted_output.allocator()->allocate();
    }
    else
    {
        dwc_optimized_func->configure(_impl->src->info(), _impl->weights->info(),
                                      biases == nullptr ? nullptr : biases->info(), _impl->dst->info(), info);
    }

    // Allocate memory based on the internal memory requirements
    experimental::MemoryRequirements mem_req = dwc_optimized_func->workspace();
    _impl->workspace.allocator()->init(TensorInfo(TensorShape{mem_req[0].size + mem_req[0].alignment}, 1, DataType::S8),
                                       mem_req[0].alignment);
    _impl->packed_weights.allocator()->init(
        TensorInfo(TensorShape{mem_req[1].size + mem_req[1].alignment}, 1, DataType::S8), mem_req[1].alignment);
    _memory_group.manage(&_impl->workspace);
    _memory_group.manage(&_impl->packed_weights);
    _impl->workspace.allocator()->allocate();
    _impl->packed_weights.allocator()->allocate();
}

Status
NEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerOptimizedInternal::validate(const ITensorInfo   *input,
                                                                                    const ITensorInfo   *weights,
                                                                                    const ITensorInfo   *biases,
                                                                                    const ITensorInfo   *output,
                                                                                    const PadStrideInfo &conv_info,
                                                                                    unsigned int depth_multiplier,
                                                                                    const ActivationLayerInfo &act_info,
                                                                                    const Size2D              &dilation)
{
    ConvolutionInfo info{conv_info, depth_multiplier, act_info, dilation};
    return cpu::CpuDepthwiseConv2d::validate(input, weights, biases, output, info);
}

void NEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerOptimizedInternal::run()
{
    prepare();
    MemoryGroupResourceScope scope_mg(_memory_group);

    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->weights);
    pack.add_tensor(TensorType::ACL_SRC_2, _impl->biases);
    pack.add_tensor(TensorType::ACL_INT_0, &_impl->permuted_input);
    pack.add_tensor(TensorType::ACL_INT_1, &_impl->permuted_weights);
    pack.add_tensor(TensorType::ACL_INT_2, &_impl->permuted_output);
    pack.add_tensor(TensorType::ACL_INT_3, &_impl->workspace);
    pack.add_tensor(TensorType::ACL_INT_4, &_impl->packed_weights);
    pack.add_tensor(TensorType::ACL_DST_0, _impl->dst);

    _impl->op->run(pack);
}

void NEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerOptimizedInternal::prepare()
{
    if (!_impl->is_prepared)
    {
        // Permute weights
        if (_impl->permute)
        {
            _impl->permuted_weights.allocator()->allocate();
        }

        if (!_impl->permuted_weights.is_used())
        {
            _impl->permuted_weights.allocator()->free();
        }

        _impl->is_prepared = true;
    }
}

struct NEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerGeneric::Impl
{
    Tensor                                   permuted_input{};
    Tensor                                   permuted_weights{};
    Tensor                                   permuted_output{};
    bool                                     is_prepared{false};
    bool                                     is_nchw{false};
    bool                                     is_activationlayer_enabled{false};
    const ITensor                           *weights{nullptr};
    const ITensor                           *biases{nullptr};
    const ITensor                           *src{nullptr};
    ITensor                                 *dst{nullptr};
    std::shared_ptr<cpu::CpuDepthwiseConv2d> op{nullptr};
};

NEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerGeneric::NEDepthwiseConvolutionLayerGeneric()
    : _impl(std::make_unique<Impl>())
{
}

void NEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerGeneric::configure(ITensor             *input,
                                                                                const ITensor       *weights,
                                                                                const ITensor       *biases,
                                                                                ITensor             *output,
                                                                                const PadStrideInfo &conv_info,
                                                                                unsigned int         depth_multiplier,
                                                                                const ActivationLayerInfo &act_info,
                                                                                const Size2D              &dilation)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);

    const ConvolutionInfo info{conv_info, depth_multiplier, act_info, dilation};
    _impl->op = std::make_unique<cpu::CpuDepthwiseConv2d>();
    _impl->op->configure(input->info(), weights->info(), biases == nullptr ? nullptr : biases->info(), output->info(),
                         info);

    _impl->src         = input;
    _impl->dst         = output;
    _impl->weights     = weights;
    _impl->biases      = biases;
    _impl->is_nchw     = input->info()->data_layout() == DataLayout::NCHW;
    _impl->is_prepared = !_impl->is_nchw;

    ITensor       *input_to_use   = input;
    const ITensor *weights_to_use = weights;
    ITensor       *output_to_use  = output;
    if (_impl->is_nchw)
    {
        auto permute_input   = std::make_unique<cpu::CpuPermute>();
        auto permute_weights = std::make_unique<cpu::CpuPermute>();

        permute_input->configure(input->info(), _impl->permuted_input.info(), PermutationVector(2U, 0U, 1U));
        _impl->permuted_input.info()->set_data_layout(DataLayout::NHWC);
        input_to_use = &_impl->permuted_input;

        permute_weights->configure(weights->info(), _impl->permuted_weights.info(), PermutationVector(2U, 0U, 1U));
        _impl->permuted_weights.info()->set_data_layout(DataLayout::NHWC);
        weights_to_use = &_impl->permuted_weights;

        _impl->permuted_output.allocator()->init(
            output->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(TensorShape()));
        output_to_use = &_impl->permuted_output;
    }

    auto depthwise_conv_kernel = std::make_unique<cpu::kernels::CpuDepthwiseConv2dNativeKernel>();
    depthwise_conv_kernel->configure(input_to_use->info(), weights_to_use->info(),
                                     biases == nullptr ? nullptr : biases->info(), output_to_use->info(), info);

    if (_impl->is_nchw)
    {
        auto permute_output = std::make_unique<cpu::CpuPermute>();
        permute_output->configure(_impl->permuted_output.info(), output->info(), PermutationVector(1U, 2U, 0U));
        _impl->permuted_output.info()->set_data_layout(DataLayout::NHWC);

        _impl->permuted_input.allocator()->allocate();
        _impl->permuted_weights.allocator()->allocate();
        _impl->permuted_output.allocator()->allocate();
    }
}

Status NEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerGeneric::validate(const ITensorInfo   *input,
                                                                                 const ITensorInfo   *weights,
                                                                                 const ITensorInfo   *biases,
                                                                                 const ITensorInfo   *output,
                                                                                 const PadStrideInfo &conv_info,
                                                                                 unsigned int         depth_multiplier,
                                                                                 const ActivationLayerInfo &act_info,
                                                                                 const Size2D              &dilation)
{
    ConvolutionInfo info{conv_info, depth_multiplier, act_info, dilation};
    return cpu::CpuDepthwiseConv2d::validate(input, weights, biases, output, info);
}

void NEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayerGeneric::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->weights);
    pack.add_tensor(TensorType::ACL_SRC_2, _impl->biases);
    pack.add_tensor(TensorType::ACL_INT_0, &_impl->permuted_input);
    pack.add_tensor(TensorType::ACL_INT_1, &_impl->permuted_weights);
    pack.add_tensor(TensorType::ACL_INT_2, &_impl->permuted_output);
    pack.add_tensor(TensorType::ACL_DST_0, _impl->dst);

    _impl->op->run(pack);
}

NEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _impl(std::make_unique<Impl>())
{
}

#ifndef DOXYGEN_SKIP_THIS
struct NEDepthwiseConvolutionLayer::NEDepthwiseConvolutionLayer::Impl
{
    DepthwiseConvolutionFunction                 depth_conv_func{DepthwiseConvolutionFunction::OPTIMIZED};
    NEDepthwiseConvolutionLayerOptimizedInternal func_optimized{nullptr};
    NEDepthwiseConvolutionLayerGeneric           func_generic{};
    std::shared_ptr<cpu::CpuDepthwiseConv2d>     op{nullptr};
};
#endif // DOXYGEN_SKIP_THIS

void NEDepthwiseConvolutionLayer::configure(ITensor                   *input,
                                            const ITensor             *weights,
                                            const ITensor             *biases,
                                            ITensor                   *output,
                                            const PadStrideInfo       &conv_info,
                                            unsigned int               depth_multiplier,
                                            const ActivationLayerInfo &act_info,
                                            const Size2D              &dilation)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);

    ARM_COMPUTE_LOG_PARAMS(input, weights, output, conv_info, depth_multiplier, biases, act_info, dilation);
    ARM_COMPUTE_ERROR_THROW_ON(NEDepthwiseConvolutionLayer::validate(
        input->info(), weights->info(), (biases == nullptr) ? nullptr : biases->info(), output->info(), conv_info,
        depth_multiplier, act_info, dilation));

    const ConvolutionInfo info{conv_info, depth_multiplier, act_info, dilation};
    _impl->op              = std::make_shared<cpu::CpuDepthwiseConv2d>();
    _impl->depth_conv_func = _impl->op->get_depthwiseconvolution_function(
        input->info(), weights->info(), (biases != nullptr) ? biases->info() : nullptr, output->info(), info);
    switch (_impl->depth_conv_func)
    {
        case DepthwiseConvolutionFunction::OPTIMIZED:
            _impl->func_optimized.configure(input, weights, biases, output, conv_info, depth_multiplier, act_info,
                                            dilation);
            break;
        case DepthwiseConvolutionFunction::GENERIC:
            _impl->func_generic.configure(input, weights, biases, output, conv_info, depth_multiplier, act_info,
                                          dilation);
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported DepthwiseConvolutionFunction");
    }
}

Status NEDepthwiseConvolutionLayer::validate(const ITensorInfo         *input,
                                             const ITensorInfo         *weights,
                                             const ITensorInfo         *biases,
                                             const ITensorInfo         *output,
                                             const PadStrideInfo       &conv_info,
                                             unsigned int               depth_multiplier,
                                             const ActivationLayerInfo &act_info,
                                             const Size2D              &dilation)
{
    ConvolutionInfo info{conv_info, depth_multiplier, act_info, dilation};
    return cpu::CpuDepthwiseConv2d::validate(input, weights, biases, output, info);
}

void NEDepthwiseConvolutionLayer::run()
{
    switch (_impl->depth_conv_func)
    {
        case DepthwiseConvolutionFunction::OPTIMIZED:
            _impl->func_optimized.run();
            break;
        case DepthwiseConvolutionFunction::GENERIC:
            _impl->func_generic.run();
            break;
        default:
            ARM_COMPUTE_ERROR("DepthwiseConvolutionFunction not properly configured");
    }
}

void NEDepthwiseConvolutionLayer::prepare()
{
    switch (_impl->depth_conv_func)
    {
        case DepthwiseConvolutionFunction::OPTIMIZED:
            _impl->func_optimized.prepare();
            break;
        case DepthwiseConvolutionFunction::GENERIC:
            _impl->func_generic.prepare();
            break;
        default:
            ARM_COMPUTE_ERROR("DepthwiseConvolutionFunction not properly configured");
    }
}
} // namespace arm_compute
