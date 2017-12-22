/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLCOL2IMKERNEL_H__
#define __ARM_COMPUTE_CLCOL2IMKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the col2im reshaping kernel.
 *
 * Rearranges each matrix column into image blocks. It's the inverse operation of @ref CLIm2ColKernel.
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
class CLCol2ImKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLCol2ImKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLCol2ImKernel(const CLCol2ImKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLCol2ImKernel &operator=(const CLCol2ImKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLCol2ImKernel(CLCol2ImKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLCol2ImKernel &operator=(CLCol2ImKernel &&) = default;
    /** Default destructor */
    ~CLCol2ImKernel() = default;

    /** Set the input and output of the kernel.
     *
     * @param[in]  input          The input tensor to convert. Data types supported: QS8/QS16/QASYMM8/F16/F32
     * @param[out] output         The output tensor. 3 lower dimensions represent a single output [width, height, OFM],
     *                            while the rest represent batch of outputs. Data types supported: Same as @p input
     * @param[in]  convolved_dims Output convolved dimensions.
     */
    void configure(const ICLTensor *input, ICLTensor *output, std::pair<unsigned int, unsigned int> convolved_dims);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    ICLTensor       *_output;
    std::pair<unsigned int, unsigned int> _convolved_dims;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLCOL2IMKERNEL_H__ */
