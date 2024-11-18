/*
 * Copyright (c) 2024 Arm Limited.
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
#include "src/cpu/operators/CpuScatter.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/common/utils/Log.h"
#include "src/cpu/kernels/CpuScatterKernel.h"

namespace arm_compute
{
namespace cpu
{

void CpuScatter::configure(const ITensorInfo *src,
                           const ITensorInfo *updates,
                           const ITensorInfo *indices,
                           ITensorInfo       *dst,
                           const ScatterInfo &scatter_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(updates, indices, dst);
    ARM_COMPUTE_LOG_PARAMS(src, updates, indices, dst, scatter_info);
    _fill_zero = scatter_info.zero_initialization;

    if (_fill_zero)
    {
        auto f = std::make_unique<cpu::CpuFill>();
        f->configure(dst, PixelValue(0.0f));
        _fill_operator = std::move(f);
    }
    else if (src != dst)
    {
        auto j = std::make_unique<cpu::CpuCopy>();
        j->configure(src, dst);
        _copy_operator = std::move(j);
        _run_copy      = true;
    }
    auto k = std::make_unique<kernels::CpuScatterKernel>();

    k->configure(updates, indices, dst, scatter_info);

    _kernel = std::move(k);
}

Status CpuScatter::validate(const ITensorInfo *src,
                            const ITensorInfo *updates,
                            const ITensorInfo *indices,
                            const ITensorInfo *dst,
                            const ScatterInfo &scatter_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(updates, indices, dst);

    bool fill_zero = scatter_info.zero_initialization;
    if (fill_zero)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src);
    }
    else if (src != dst)
    {
        ARM_COMPUTE_RETURN_ON_ERROR(cpu::CpuCopy::validate(src, dst));
    }

    return kernels::CpuScatterKernel::validate(updates, indices, dst, scatter_info);
}

void CpuScatter::run(ITensorPack &tensors)
{
    auto src = tensors.get_const_tensor(ACL_SRC_0);
    auto dst = tensors.get_tensor(ACL_DST);

    if (_fill_zero)
    {
        // Fill destination tensor with 0 values if zero init.
        ITensorPack fill_pack{{ACL_SRC, dst}};
        _fill_operator->run(fill_pack);
    }

    if (_run_copy)
    {
        // copy src to dst before scatter op.
        ITensorPack copy_pack{{ACL_SRC, src}, {ACL_DST, dst}};
        _copy_operator->run(copy_pack);
    }
    NEScheduler::get().schedule_op(_kernel.get(), Window::DimY, _kernel->window(), tensors);
}
} // namespace cpu
} // namespace arm_compute
