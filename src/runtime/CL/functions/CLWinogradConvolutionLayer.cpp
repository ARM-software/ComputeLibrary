/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLWinogradConvolutionLayer.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "src/core/CL/ICLKernel.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/runtime/gpu/cl/operators/ClWinogradConv2d.h"
#include "support/Cast.h"

namespace arm_compute
{
struct CLWinogradConvolutionLayer::Impl
{
    const ICLTensor                          *src{ nullptr };
    const ICLTensor                          *weights{ nullptr };
    const ICLTensor                          *biases{ nullptr };
    ICLTensor                                *dst{ nullptr };
    std::unique_ptr<opencl::ClWinogradConv2d> op{ nullptr };
    ITensorPack                               run_pack{};
    MemoryGroup                               memory_group{};
    WorkspaceData<CLTensor>                   workspace_tensors{};
    bool                                      is_prepared{ false };
};

CLWinogradConvolutionLayer::CLWinogradConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _impl(std::make_unique<Impl>())
{
    _impl->memory_group = MemoryGroup(memory_manager);
}

CLWinogradConvolutionLayer::~CLWinogradConvolutionLayer() = default;

void CLWinogradConvolutionLayer::configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info,
                                           bool enable_fast_math)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, weights, biases, output, conv_info, act_info, enable_fast_math);
}

void CLWinogradConvolutionLayer::configure(const CLCompileContext &compile_context, ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output,
                                           const PadStrideInfo       &conv_info,
                                           const ActivationLayerInfo &act_info, bool enable_fast_math)
{
    _impl->src     = input;
    _impl->weights = weights;
    _impl->biases  = biases;
    _impl->dst     = output;

    _impl->op = std::make_unique<opencl::ClWinogradConv2d>();
    _impl->op->configure(compile_context, input->info(), weights->info(), (biases != nullptr ? biases->info() : nullptr), output->info(), conv_info, act_info, enable_fast_math);

    _impl->run_pack =
    {
        { TensorType::ACL_SRC_0, _impl->src },
        { TensorType::ACL_SRC_1, _impl->weights },
        { TensorType::ACL_SRC_2, _impl->biases },
        { TensorType::ACL_DST, _impl->dst }
    };
    _impl->workspace_tensors = manage_workspace<CLTensor>(_impl->op->workspace(), _impl->memory_group, _impl->run_pack, _impl->run_pack);
}

Status CLWinogradConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                            const ActivationLayerInfo &act_info, bool enable_fast_math)
{
    return opencl::ClWinogradConv2d::validate(input, weights, biases, output, conv_info, act_info, enable_fast_math);
}

void CLWinogradConvolutionLayer::run()
{
    MemoryGroupResourceScope scope_mg(_impl->memory_group);
    prepare();
    _impl->op->run(_impl->run_pack);
}

void CLWinogradConvolutionLayer::prepare()
{
    if(!_impl->is_prepared)
    {
        _impl->op->prepare(_impl->run_pack);

        // Release Preparation tensors
        release_prepare_tensors(_impl->workspace_tensors, _impl->run_pack);
        _impl->run_pack.remove_tensor(TensorType::ACL_SRC_1);
        _impl->is_prepared = true;
    }
}
} // namespace arm_compute