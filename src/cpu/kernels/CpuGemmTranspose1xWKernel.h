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
#ifndef ARM_COMPUTE_CPU_GEMM_TRANSPOSE1xW_KERNEL_H
#define ARM_COMPUTE_CPU_GEMM_TRANSPOSE1xW_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Kernel which transposes the elements of a matrix in chunks of 1xW, where W is equal to (16 / element size of the tensor)
 *
 * Following an example of how the transposition1xW works when the input data is F32
 *
 * @f[
 * \left( \begin{array}{cccc}
 * a00 & a01 & a02 & a03 \\
 * a10 & a11 & a12 & a13 \\
 * a20 & a21 & a22 & a23 \\
 * a30 & a31 & a32 & a33 \\
 * \end{array} \right)
 * \rightarrow
 * \left( \begin{array}{ccccccccccccccccc}
 * a00 & a01 & a02 & a03 & a10 & a11 & a12 & a13 & a20 & a21 & a22 & a23 & a30 & a31 & a32 & a33 \\
 * \end{array} \right)
 * @f]
 *
 * Following an example of how the transposition1xW works when the input data type is F16
 *
 * @f[
 * \left( \begin{array}{cccccccc}
 * a00 & a01 & a02 & a03 & a04 & a05 & a06 & a07 \\
 * a10 & a11 & a12 & a13 & a14 & a15 & a16 & a17 \\
 * a20 & a21 & a22 & a23 & a24 & a25 & a26 & a27 \\
 * a30 & a31 & a32 & a33 & a34 & a35 & a36 & a37 \\
 * \end{array} \right)
 * \rightarrow
 * \left( \begin{array}{cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc}
 * a00 & a01 & a02 & a03 & a04 & a05 & a06 & a07 & a10 & a11 & a12 & a13 & a14 & a15 & a16 & a17 & a20 & a21 & a22 & a23 & a24 & a25 & a26 & a27 & a30 & a31 & a32 & a33 & a34 & a35 & a36 & a37\\
 * \end{array} \right)
 * @f]
 *
 * @note The output matrix will have the following shape: [ height * W, ceil(width / W) ], where W = (16 / element size of the tensor)
 *
 */
class CpuGemmTranspose1xWKernel : public ICpuKernel<CpuGemmTranspose1xWKernel>
{
public:
    CpuGemmTranspose1xWKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuGemmTranspose1xWKernel);
    /** Configure kernel for a given list of arguments
     *
     * @param[in]  src Input tensor info. Data types supported: All
     * @param[out] dst Output tensor info. Data type supported: same as @p src.
     */
    void configure(const ITensorInfo *src, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration of @ref CpuGemmTranspose1xWKernel
     *
     * Similar to @ref CpuGemmTranspose1xWKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_GEMM_TRANSPOSE1xW_KERNEL_H */
