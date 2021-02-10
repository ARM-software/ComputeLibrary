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
#ifndef ARM_COMPUTE_NECOL2IMKERNEL_H
#define ARM_COMPUTE_NECOL2IMKERNEL_H

#include "src/core/NEON/INEKernel.h"

#include "arm_compute/core/Size2D.h"

namespace arm_compute
{
class ITensor;

/** Neon kernel to perform col2im reshaping.
 *
 * Rearranges each matrix column into image blocks. It's the inverse operation of @ref NEIm2ColKernel.
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
class NECol2ImKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NECol2ImKernel";
    }
    /** Default constructor */
    NECol2ImKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NECol2ImKernel(const NECol2ImKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NECol2ImKernel &operator=(const NECol2ImKernel &) = delete;
    /** Allow instances of this class to be moved */
    NECol2ImKernel(NECol2ImKernel &&) = default;
    /** Allow instances of this class to be moved */
    NECol2ImKernel &operator=(NECol2ImKernel &&) = default;
    /** Default destructor */
    ~NECol2ImKernel() = default;

    /** Set the input and output of the kernel.
     *
     * @param[in]  input          The input tensor to convert. Data types supported: All
     * @param[out] output         The output tensor. 3 lower dimensions represent a single output [width, height, OFM],
     *                            while the rest represent batch of outputs. Data types supported: Same as @p input
     * @param[in]  convolved_dims Output convolved dimensions.
     */
    void configure(const ITensor *input, ITensor *output, const Size2D &convolved_dims);
    /** Static function to check if given info will lead to a valid configuration of @ref NECol2ImKernel
     *
     * @param[in] input          The input tensor to convert. Data types supported: All
     * @param[in] output         The output tensor. 3 lower dimensions represent a single output [width, height, OFM],
     *                           while the rest represent batch of outputs. Data types supported: Same as @p input
     * @param[in] convolved_dims Output convolved dimensions.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const Size2D &convolved_dims);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Template function to run the col2im
     *
     * @param[in] window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    template <typename T>
    void run_col2im(const Window &window);

    /** Common signature for all the specialised col2im functions
     *
     * @param[in] window Region on which to execute the kernel.
     */
    using Col2ImFunctionPtr = void (NECol2ImKernel::*)(const Window &window);

    Col2ImFunctionPtr _func;
    const ITensor    *_input;
    ITensor          *_output;
    Size2D            _convolved_dims;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NECOL2IMKERNEL_H */
