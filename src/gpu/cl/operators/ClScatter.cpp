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
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(indices, 1, DataType::U32);

    return kernels::ClScatterKernel::validate(src, updates, indices, dst, info);
}

void ClScatter::configure(const CLCompileContext &compile_context,
                          const ITensorInfo      *src,
                          const ITensorInfo      *updates,
                          const ITensorInfo      *indices,
                          ITensorInfo            *dst,
                          const ScatterInfo      &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, indices, dst);
    ARM_COMPUTE_LOG_PARAMS(src, indices, dst, info);
    ARM_COMPUTE_UNUSED(src);
    ARM_COMPUTE_UNUSED(updates);
    ARM_COMPUTE_UNUSED(indices);
    ARM_COMPUTE_UNUSED(dst);
    ARM_COMPUTE_UNUSED(info);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate(src, updates, indices, dst, info));
    _fill_zero = info.zero_initialization;

    // If necessary, create fill kernel to fill dst tensor.
    if (_fill_zero)
    {
        _fill_kernel = std::make_unique<kernels::ClFillKernel>();
    }

    // Configure ClScatterKernel
    auto k = std::make_unique<kernels::ClScatterKernel>();
    k->set_target(CLScheduler::get().target());
    k->configure(compile_context, src, updates, indices, dst, info);
    _scatter_kernel = std::move(k);
}

void ClScatter::run(ITensorPack &tensors)
{
    ARM_COMPUTE_UNUSED(tensors);
}

} // namespace opencl
} // namespace arm_compute
