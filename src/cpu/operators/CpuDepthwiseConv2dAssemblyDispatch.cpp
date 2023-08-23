/*
 * Copyright (c) 2019-2023 Arm Limited.
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

#include "src/cpu/operators/CpuDepthwiseConv2dAssemblyDispatch.h"

#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/common/utils/Log.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/utils/AssemblyUtils.h"
#include "src/cpu/kernels/internal/CpuDepthwiseConv2dAssemblyWrapperKernel.h"

namespace arm_compute
{
namespace cpu
{
struct CpuDepthwiseConv2dAssemblyDispatch::LocalImpl
{
    std::unique_ptr<kernels::CpuDepthwiseConv2dAssemblyWrapperKernel> asm_kernel{ nullptr };
    bool                                                              is_prepared{ false };
    bool                                                              are_weights_const{ true };
    experimental::MemoryRequirements                                  mem_req{};
};

#ifndef DOXYGEN_SKIP_THIS
CpuDepthwiseConv2dAssemblyDispatch::CpuDepthwiseConv2dAssemblyDispatch()
    : _pImpl(std::make_unique<LocalImpl>())
{
}
#endif /* DOXYGEN_SKIP_THIS */

CpuDepthwiseConv2dAssemblyDispatch::~CpuDepthwiseConv2dAssemblyDispatch() = default;

void CpuDepthwiseConv2dAssemblyDispatch::configure(const ITensorInfo     *src,
                                                   const ITensorInfo     *weights,
                                                   const ITensorInfo     *bias,
                                                   ITensorInfo           *dst,
                                                   const ConvolutionInfo &info)
{
    ARM_COMPUTE_LOG_PARAMS(src, weights, bias, dst, info);
    const CPUInfo     &ci          = NEScheduler::get().cpu_info();
    const unsigned int num_threads = NEScheduler::get().num_threads();
    _pImpl->is_prepared            = false;
    _pImpl->are_weights_const      = weights->are_values_constant();

    // If we don't support a combination of data types, silently return: it is the caller's responsibility to check if configure() was successful via is_configured()
    if(!CpuDepthwiseConv2dAssemblyDispatch::validate(src, weights, bias, dst, info))
    {
        return;
    }

    auto dwc_wrapper = std::make_unique<kernels::CpuDepthwiseConv2dAssemblyWrapperKernel>();
    ARM_COMPUTE_ERROR_ON(dwc_wrapper == nullptr);
    dwc_wrapper->configure(src, weights, bias, dst, info, ci);

    // Compute memory requirements for assembly kernels
    constexpr size_t alignment = 4096;
    _pImpl->mem_req.push_back({ TensorType::ACL_INT_0, dwc_wrapper->get_working_size(num_threads), alignment });
    _pImpl->mem_req.push_back({ TensorType::ACL_INT_1, dwc_wrapper->get_storage_size(), alignment });
    _pImpl->asm_kernel = std::move(dwc_wrapper);
}

Status CpuDepthwiseConv2dAssemblyDispatch::validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *bias, const ITensorInfo *dst, const ConvolutionInfo &info)
{
    return kernels::CpuDepthwiseConv2dAssemblyWrapperKernel::validate(src, weights, bias, dst, info);
}

experimental::MemoryRequirements CpuDepthwiseConv2dAssemblyDispatch::workspace() const
{
    return _pImpl->mem_req;
}

bool CpuDepthwiseConv2dAssemblyDispatch::is_activation_supported(const ActivationLayerInfo &activation)
{
    arm_gemm::Activation act = assembly_utils::map_to_arm_gemm_activation(activation);
    return act.type != arm_gemm::Activation::Type::None;
}

void CpuDepthwiseConv2dAssemblyDispatch::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");

    prepare(tensors);

    NEScheduler::get().schedule_op(_pImpl->asm_kernel.get(), Window::DimY, _pImpl->asm_kernel->window(), tensors);
}

void CpuDepthwiseConv2dAssemblyDispatch::prepare(ITensorPack &tensors)
{
    const ITensor *weights = tensors.get_const_tensor(TensorType::ACL_SRC_1);

    if((!_pImpl->are_weights_const && weights != nullptr) || !_pImpl->is_prepared)
    {
        // Pack weights and bias
        const ITensor *bias    = tensors.get_const_tensor(TensorType::ACL_SRC_2);
        ITensor       *storage = tensors.get_tensor(TensorType::ACL_INT_1);

        const auto weights_ptr    = weights->buffer() + weights->info()->offset_first_element_in_bytes();
        const auto bias_ptr       = (bias) ? bias->buffer() + bias->info()->offset_first_element_in_bytes() : nullptr;
        auto       parameters_ptr = storage->buffer() + storage->info()->offset_first_element_in_bytes();

        const auto weights_shape   = weights->info()->tensor_shape();
        const auto weights_padding = weights->info()->padding();

        const size_t ld_weights_col = weights_shape[0] + weights_padding.left + weights_padding.right;
        const size_t ld_weights_row = ld_weights_col * (weights_shape[1] + weights_padding.top + weights_padding.bottom);
        _pImpl->asm_kernel->pack_parameters(parameters_ptr, bias_ptr, weights_ptr, ld_weights_col, ld_weights_row);

        weights->mark_as_unused();
        if(bias != nullptr)
        {
            bias->mark_as_unused();
        }
        _pImpl->is_prepared = true;
    }
}
} // namespace cpu
} // namespace arm_compute
