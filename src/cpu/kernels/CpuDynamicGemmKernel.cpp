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
#include "src/cpu/kernels/CpuDynamicGemmKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{

void CpuDynamicGemmKernel::configure(const ITensorInfo *a,
                                     const ITensorInfo *b,
                                     const ITensorInfo *c,
                                     ITensorInfo       *d,
                                     float              alpha,
                                     float              beta,
                                     const GEMMInfo    &gemm_info)
{
    ARM_COMPUTE_UNUSED(a);
    ARM_COMPUTE_UNUSED(b);
    ARM_COMPUTE_UNUSED(c);
    ARM_COMPUTE_UNUSED(d);
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_UNUSED(beta);
    ARM_COMPUTE_UNUSED(gemm_info);
    ARM_COMPUTE_UNUSED(_func);
}

Status CpuDynamicGemmKernel::validate(const ITensorInfo *a,
                                      const ITensorInfo *b,
                                      const ITensorInfo *c,
                                      ITensorInfo       *d,
                                      float              alpha,
                                      float              beta,
                                      const GEMMInfo    &gemm_info)
{
    ARM_COMPUTE_UNUSED(a);
    ARM_COMPUTE_UNUSED(b);
    ARM_COMPUTE_UNUSED(c);
    ARM_COMPUTE_UNUSED(d);
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_UNUSED(beta);
    ARM_COMPUTE_UNUSED(gemm_info);

    return Status{ErrorCode::RUNTIME_ERROR, "Kernel not implemented yet."};
}

void CpuDynamicGemmKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(tensors);
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_UNUSED(info);
}

const char *CpuDynamicGemmKernel::name() const
{
    return "";
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
