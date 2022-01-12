/*
 * Copyright (c) 2017-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_GEMMLOWP_MATRIXMULTIPLY_KERNEL_H
#define ARM_COMPUTE_CPU_GEMMLOWP_MATRIXMULTIPLY_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Kernel to multiply matrices
 *
 * @note @ref CpuGemmLowpMatrixMultiplyKernel low precision matrix product kernel
 *  This kernel performs the following computation:
 *
 *  -# Convert a values from int8 to int32
 *  -# Convert b values from int8 to int32
 *  -# Compute the int32 matrix product of the resulting a * b and store the result as int32
 *
 */
class CpuGemmLowpMatrixMultiplyKernel : public ICpuKernel<CpuGemmLowpMatrixMultiplyKernel>
{
public:
    /** Default constructor */
    CpuGemmLowpMatrixMultiplyKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuGemmLowpMatrixMultiplyKernel);
    /** Initialise the kernel's input and output.
     *
     * The input matrices @p src0 and @p src1 must be the output of the kernels: @ref CpuGemmInterleave4x4Kernel and @ref CpuGemmTranspose1xWKernel. These two
     * kernels change the layout of the original matrices to be more cache-friendly.
     *
     * @param[in]  src0 Input tensor info containing the interleaved Matrix A. Data type supported: U8/QASYMM8/S8/QASYMM8_SIGNED
     * @param[in]  src1 Input tensor info containing the transposed1xW Matrix B. Data type supported: U8/QASYMM8/S8/QASYMM8_SIGNED/QSYMM8/QSYMM8_PER_CHANNEL
     * @param[out] dst  Output tensor info to store the result of matrix multiplication. Data type supported: S32
     */
    void configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuGemmLowpMatrixMultiplyKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

private:
    bool _slide_matrix_b{ true };
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /*ARM_COMPUTE_CPU_GEMMLOWP_MATRIXMULTIPLY_KERNEL_H*/
