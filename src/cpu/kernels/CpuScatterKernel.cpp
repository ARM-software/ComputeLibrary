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
#include "src/cpu/kernels/CpuScatterKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"

#include <vector>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{

/* Scatter */
static const std::vector<typename CpuScatterKernel::ScatterKernel> available_kernels = {

};

} // namespace

const std::vector<typename CpuScatterKernel::ScatterKernel> &CpuScatterKernel::get_available_kernels()
{
    return available_kernels;
}

void CpuScatterKernel::configure(const ITensorInfo *src,
                                 const ITensorInfo *updates,
                                 const ITensorInfo *indices,
                                 ITensorInfo       *dst,
                                 const ScatterInfo &info)
{
    ARM_COMPUTE_UNUSED(src);
    ARM_COMPUTE_UNUSED(updates);
    ARM_COMPUTE_UNUSED(indices);
    ARM_COMPUTE_UNUSED(dst);
    ARM_COMPUTE_UNUSED(info);

    ARM_COMPUTE_UNUSED(_run_method);
}

Status CpuScatterKernel::validate(const ITensorInfo *src,
                                  const ITensorInfo *updates,
                                  const ITensorInfo *indices,
                                  const ITensorInfo *dst,
                                  const ScatterInfo &info)
{
    ARM_COMPUTE_UNUSED(src);
    ARM_COMPUTE_UNUSED(updates);
    ARM_COMPUTE_UNUSED(indices);
    ARM_COMPUTE_UNUSED(dst);
    ARM_COMPUTE_UNUSED(info);

    return Status{ErrorCode::RUNTIME_ERROR, "No configuration implemented yet."};
}

void CpuScatterKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(tensors);
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_UNUSED(info);

    ARM_COMPUTE_UNUSED(_run_method);
}

const char *CpuScatterKernel::name() const
{
    return _name.c_str();
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
