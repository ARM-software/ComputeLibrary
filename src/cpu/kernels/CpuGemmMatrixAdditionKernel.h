/*
 * Copyright (c) 2016-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_GEMM_MATRIX_ADDITION_KERNEL_H
#define ARM_COMPUTE_CPU_GEMM_MATRIX_ADDITION_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Kernel to perform the in-place matrix addition between 2 matrices taking into account that the second matrix might be weighted by a scalar value beta:
 *
 * @note [ MTX_OUT = MTX_0 + beta * MTX_1 ] with MTX_0 and MTX_1 of the same size
 *
 * @note This stage is used to finalize the GEMM result and it is computed if and only if beta != 0.0. In case this kernel is used for finalizing GEMM result, we have:
 *        - MTX_0 = A * B * alpha, where MTX_0 is the output of @ref CpuGemmMatrixMultiplyKernel
 *        - MTX_1 = C
 */
class CpuGemmMatrixAdditionKernel : public ICpuKernel<CpuGemmMatrixAdditionKernel>
{
private:
    using GemmMatrixAddKernelPtr = std::add_pointer<void(const ITensor *, ITensor *, const Window &, float)>::type;

public:
    struct GemmMatrixAddKernel
    {
        const char                  *name;
        const DataTypeISASelectorPtr is_selected;
        GemmMatrixAddKernelPtr       ukernel;
    };
    CpuGemmMatrixAdditionKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuGemmMatrixAdditionKernel);
    /** Initialise the kernel's input and output.
     *
     * @note The input and output tensor must have the same dimensions
     *
     * @param[in]      src  Input tensor info (Matrix C). Data types supported: F16/F32
     * @param[in, out] dst  Output tensor info. If this kernel is used to finalize the GEMM result, output contains the result obtained by the kernel @ref CpuGemmMatrixMultiplyKernel. Data type supported: the same as @p src.
     * @param[in]      beta Weight of matrix C
     */
    void configure(const ITensorInfo *src, ITensorInfo *dst, float beta);
    /** Static function to check if given info will lead to a valid configuration of @ref CpuGemmMatrixAdditionKernel.
     *
     * @note The input and output tensor must have the same dimensions
     *
     * Similar to @ref CpuGemmMatrixAdditionKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst, float beta);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

    static const std::vector<GemmMatrixAddKernel> &get_available_kernels();

private:
    /** Common signature for all the matrix addition functions
     *
     * @param[in]  src    An input tensor. Data types supported: F16/F32
     * @param[out] dst    The output tensor. Data type supported: same as @p src
     * @param[in]  window Region on which to execute the kernel.
     * @param[in]  beta   Weight of matrix C
     */
    /** Matrix addition function to use for the particular tensor types passed to configure() */
    GemmMatrixAddKernelPtr _func{ nullptr };
    float                  _beta{ 0.f };
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_GEMM_MATRIX_ADDITION_KERNEL_H */
