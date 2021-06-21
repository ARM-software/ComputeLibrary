/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_GEMM_INTERLEAVE4x4_KERNEL_H
#define ARM_COMPUTE_CPU_GEMM_INTERLEAVE4x4_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/core/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Kernel to interleave the elements of a matrix
 *
 * This function puts the values in a 4x4 block of Matrix A on the same row (Interleaved values)
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
 * a00 & a10 & a20 & a30 & a01 & a11 & a21 & a31 & a02 & a12 & a22 & a32 & a03 & a13 & a23 & a33 \\
 * \end{array} \right)
 * @f]
 *
 * After this operation, the dst matrix will have the following shape: [ height * 4, ceil(width / 4.0f) ]
 */
class CpuGemmInterleave4x4Kernel : public ICpuKernel
{
public:
    /** Default Constructor */
    CpuGemmInterleave4x4Kernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuGemmInterleave4x4Kernel);
    /** Initialise the kernel's src and dst.
     *
     * @param[in]  src Input tensor info. Data types supported: All
     * @param[out] dst Output tensor info which stores the interleaved matrix. Data type supported: same as @p src.
     */
    void configure(const ITensorInfo *src, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration of @ref CpuGemmInterleave4x4Kernel
     *
     * Similar to @ref CpuGemmInterleave4x4Kernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

private:
    /** Common signature for all the specialised gemm interleave 4x4 functions
     *
     * @param[in]  src    Input tensor. Data types supported: uint32_t, uint16_t and uint8_t
     * @param[out] dst    Output tensor. Data types supported: uint32_t, uint16_t and uint8_t
     * @param[in]  window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    using GEMMInterleaveFunctionPtr = void (*)(const ITensor *src, ITensor *dst, const Window &window);

    GEMMInterleaveFunctionPtr _func{ nullptr };
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /*ARM_COMPUTE_CPU_GEMM_INTERLEAVE4x4_KERNEL_H*/
