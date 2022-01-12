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
#ifndef ARM_COMPUTE_CPU_COL2IM_KERNEL_H
#define ARM_COMPUTE_CPU_COL2IM_KERNEL_H

#include "arm_compute/core/Size2D.h"
#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Kernel to perform col2im reshaping.
 *
 * Rearranges each matrix column into image blocks. It's the inverse operation of @ref CpuIm2ColKernel.
 *
 * For example, a vector of 9 elements can be reshaped to a block(image) of 3x3:
 *
 * @f[
 * \left( \begin{array}{ccccccccc}
 * a0 & a1 & a2 & a3 & a4 & a5 & a6 & a7 & a8 \\
 * \end{array} \right)
 * \rightarrow
 * \left( \begin{array}{ccc}
 * a0 & a1 & a2 \\
 * a3 & a4 & a5 \\
 * a6 & a7 & a8 \\
 * \end{array} \right)
 * @f]
 */
class CpuCol2ImKernel : public ICpuKernel<CpuCol2ImKernel>
{
public:
    /** Default constructor */
    CpuCol2ImKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuCol2ImKernel);
    /** Set the input and output of the kernel.
     *
     * @param[in]  src            The input tensor info to convert. Data types supported: All
     * @param[out] dst            The output tensor info. 3 lower dimensions represent a single output [width, height, OFM],
     *                            while the rest represent batch of outputs. Data types supported: Same as @p input
     * @param[in]  convolved_dims Output convolved dimensions.
     */
    void configure(const ITensorInfo *src, ITensorInfo *dst, const Size2D &convolved_dims);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuCol2ImKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst, const Size2D &convolved_dims);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

private:
    Size2D _convolved_dims{};
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /*ARM_COMPUTE_CPU_COL2IM_KERNEL_H */
