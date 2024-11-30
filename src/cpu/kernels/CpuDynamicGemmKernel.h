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
#ifndef ACL_SRC_CPU_KERNELS_CPUDYNAMICGEMMKERNEL_H
#define ACL_SRC_CPU_KERNELS_CPUDYNAMICGEMMKERNEL_H

#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Arm(R) Neon (TM) kernel to perform dynamic GEMM */
class CpuDynamicGemmKernel : public ICpuKernel<CpuDynamicGemmKernel>
{
private:
    using DynamicGemmKernelPtr = std::add_pointer<void(
        const ITensor *, const ITensor *, const ITensor *, ITensor *, const Window &, float, float)>::type;

public:
    struct DynamicGemmKernel
    {
        const char          *name;
        DynamicGemmKernelPtr ukernel;
    };
    CpuDynamicGemmKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuDynamicGemmKernel);
    /** Initialise the kernel's input and output.
     *
     * @param[in]  a         First input tensor info (Matrix A or Vector A). Data type supported: F32
     * @param[in]  b         Second input tensor info (Matrix B). Data type supported: same as @p a
     * @param[in]  c         Third input tensor info (Matrix C). It can be a nullptr if just the multiplication between @p a and @p b is needed. Data type supported: same as @p a
     * @param[out] d         Output tensor info. Data type supported: same as @p a
     * @param[in]  alpha     Weight of the matrix product
     * @param[in]  beta      Weight of matrix C
     * @param[in]  gemm_info (Optional) Specifies if the matrix A and/or matrix B have been reshaped and
     *                       if the reshape of matrix B should happen only for the first run
     */
    void configure(const ITensorInfo *a,
                   const ITensorInfo *b,
                   const ITensorInfo *c,
                   ITensorInfo       *d,
                   float              alpha,
                   float              beta,
                   const GEMMInfo    &gemm_info = GEMMInfo());

    /** Static function to check if given info will lead to a valid configuration of @ref CpuDynamicGemmMatKernel.
     *
     * @note The input and output tensor must have the same dimensions
     *
     * Similar to @ref CpuDynamicGemmKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *a,
                           const ITensorInfo *b,
                           const ITensorInfo *c,
                           ITensorInfo       *d,
                           float              alpha,
                           float              beta,
                           const GEMMInfo    &gemm_info = GEMMInfo());

    // Inherited methods overridden:
    void        run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

private:
    DynamicGemmKernelPtr _func{nullptr};
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif // ACL_SRC_CPU_KERNELS_CPUDYNAMICGEMMKERNEL_H
