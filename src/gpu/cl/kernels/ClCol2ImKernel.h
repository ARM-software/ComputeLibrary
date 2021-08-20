/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CL_COL2IM_KERNEL_H
#define ARM_COMPUTE_CL_COL2IM_KERNEL_H

#include "arm_compute/core/Size2D.h"
#include "src/core/common/Macros.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
/** Interface for the col2im reshaping kernel.
 *
 * Rearranges each matrix column into image blocks. It's the inverse operation of @ref opencl::kernels::ClIm2ColKernel.
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
class ClCol2ImKernel : public IClKernel
{
public:
    /** Default constructor */
    ClCol2ImKernel();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClCol2ImKernel);
    /** Set the input and output of the kernel.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src             The input tensor info to convert. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32
     * @param[out] dst             The output tensor info. 3 lower dimensions represent a single output [width, height, OFM],
     *                             while the rest represent batch of outputs. Data types supported: Same as @p input. Data layout: NCHW
     * @param[in]  convolved_dims  Output convolved dimensions.
     * @param[in]  num_groups      (Optional) Number of groups when performing a grouped convolution
     */
    void configure(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *dst, const Size2D &convolved_dims, unsigned int num_groups = 1);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to ClCol2ImKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst, const Size2D &convolved_dims, unsigned int num_groups = 1);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override;

public:
    Size2D _convolved_dims;
};
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
#endif /*ARM_COMPUTE_CL_COL2IM_KERNEL_H */
