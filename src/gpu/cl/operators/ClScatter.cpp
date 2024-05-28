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
#include "src/gpu/cl/operators/ClScatter.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include "src/common/utils/Log.h"
#include "src/gpu/cl/kernels/ClCopyKernel.h"
#include "src/gpu/cl/kernels/ClFillKernel.h"
#include "src/gpu/cl/kernels/ClScatterKernel.h"

namespace arm_compute
{
namespace opencl
{
using namespace arm_compute::opencl::kernels;

ClScatter::ClScatter()
{
}

Status ClScatter::validate(const ITensorInfo *src,
                           const ITensorInfo *updates,
                           const ITensorInfo *indices,
                           const ITensorInfo *dst,
                           const ScatterInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(updates, indices, dst);
    if (src != nullptr)
    {
        // Check dst/src are same shape and datatype.
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(src->tensor_shape(), dst->tensor_shape());
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, updates, dst);
        ARM_COMPUTE_RETURN_ON_ERROR(kernels::ClCopyKernel::validate(src, dst)); // Validate Copy kernel
    }
    if (src != dst)
    {
        ARM_COMPUTE_RETURN_ON_ERROR(kernels::ClFillKernel::validate(dst, PixelValue(0.0f))); // Validate Fill kernel.
    }

    return kernels::ClScatterKernel::validate(updates, indices, dst, info);
}

void ClScatter::configure(const CLCompileContext &compile_context,
                          const ITensorInfo      *src,
                          const ITensorInfo      *updates,
                          const ITensorInfo      *indices,
                          ITensorInfo            *dst,
                          const ScatterInfo      &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(updates, indices, dst);
    ARM_COMPUTE_LOG_PARAMS(src, indices, dst, info);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate(src, updates, indices, dst, info));
    _fill_zero = info.zero_initialization;

    // If necessary, create fill kernel to fill dst tensor.
    if (_fill_zero)
    {
        auto f = std::make_unique<kernels::ClFillKernel>();
        f->configure(compile_context, dst, PixelValue(0.0f));
        _fill_kernel = std::move(f);
    }
    else if (src != dst) // Check whether copying is necessary
    {
        // Fill dst with src copy here.
        auto j = std::make_unique<kernels::ClCopyKernel>();
        j->configure(compile_context, src, dst);
        _copy_kernel = std::move(j);
        _run_copy    = true;
    }

    // Configure ClScatterKernel
    auto k = std::make_unique<kernels::ClScatterKernel>();
    k->set_target(CLScheduler::get().target());
    k->configure(compile_context, updates, indices, dst, info);
    _scatter_kernel = std::move(k);
}

void ClScatter::run(ITensorPack &tensors)
{
    // Get tensors.
    auto src     = tensors.get_const_tensor(ACL_SRC_0);
    auto updates = tensors.get_const_tensor(ACL_SRC_1);
    auto indices = tensors.get_const_tensor(ACL_SRC_2);
    auto dst     = tensors.get_tensor(ACL_DST);

    if (_fill_zero)
    {
        // Fill destination tensor with 0 values if zero init.
        ITensorPack fill_pack{{ACL_SRC, dst}};
        CLScheduler::get().enqueue_op(*_fill_kernel, fill_pack, false);
    }

    if (_run_copy)
    {
        // copy src to dst before scatter op.
        ITensorPack copy_pack{{ACL_SRC, src}, {ACL_DST, dst}};
        CLScheduler::get().enqueue_op(*_copy_kernel, copy_pack, false);
    }

    ITensorPack scatter_pack{{ACL_SRC_0, updates}, {ACL_SRC_1, indices}, {ACL_DST, dst}};
    CLScheduler::get().enqueue_op(*_scatter_kernel, scatter_pack, false);
}

} // namespace opencl
} // namespace arm_compute
