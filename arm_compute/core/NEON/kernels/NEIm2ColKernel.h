/*
 * Copyright (c) 2017-2020 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEIM2COLKERNEL_H__
#define __ARM_COMPUTE_NEIM2COLKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;
class Size2D;

/** Interface for the im2col reshape kernel.
 *
 * Rearranges image blocks into columns. It is used to strip out each convolution block to a single column.
 * It is used to transform a convolution to a plain matrix multiplication.
 *
 * For example taking into account the image below and assuming 3x3 image blocks with stride of 1 we have:
 *
 * @f[
 * \left( \begin{array}{cccc}
 * a00 & a01 & a02 & a03 \\
 * a10 & a11 & a12 & a13 \\
 * a20 & a21 & a22 & a23 \\
 * a30 & a31 & a32 & a33 \\
 * \end{array} \right)
 * \rightarrow
 * \left( \begin{array}{ccccccccc}
 * a00 & a01 & a02 & a10 & a11 & a12 & a20 & a21 & a22 \\
 * a01 & a02 & a03 & a11 & a12 & a13 & a21 & a22 & a23 \\
 * a10 & a11 & a12 & a20 & a21 & a22 & a30 & a31 & a32 \\
 * a11 & a12 & a13 & a21 & a22 & a23 & a31 & a32 & a33 \\
 * \end{array} \right)
 * @f]
 */
class NEIm2ColKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEIm2ColKernel";
    }
    /** Default constructor */
    NEIm2ColKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEIm2ColKernel(const NEIm2ColKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEIm2ColKernel &operator=(const NEIm2ColKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEIm2ColKernel(NEIm2ColKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEIm2ColKernel &operator=(NEIm2ColKernel &&) = default;
    /** Default destructor */
    ~NEIm2ColKernel() = default;

    /** Set the input and output of the kernel.
     *
     * @param[in]  input       The input tensor to convert. 3 lower dimensions represent a single input [width, height, IFM],
     *                         while every optional dimension from 4 and above represent a batch of inputs. Data types supported: QASYMM8/F16/F32
     *                         Note: QASYMM8 works only for has_bias = false
     * @param[out] output      The output tensor. Data types supported: Same as @p input
     * @param[in]  kernel_dims The kernel dimensions (width and height).
     * @param[in]  conv_info   Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  has_bias    In case biases are provided expands the matrix with 1.
     * @param[in]  dilation    (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  num_groups  (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is not supported
     */
    void configure(const ITensor *input, ITensor *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info,
                   bool has_bias, const Size2D &dilation = Size2D(1U, 1U), unsigned int num_groups = 1);
    /** Static function to check if given info will lead to a valid configuration of @ref NEIm2ColKernel
     *
     * @param[in] input       The input tensor to convert. 3 lower dimensions represent a single input [width, height, IFM],
     *                        while every optional dimension from 4 and above represent a batch of inputs. Data types supported: QASYMM8/F16/F32
     *                        Note: QASYMM8 works only for has_bias = false
     * @param[in] output      The output tensor. Data types supported: Same as @p input
     * @param[in] kernel_dims The kernel dimensions (width and height).
     * @param[in] conv_info   Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in] has_bias    In case biases are provided expands the matrix with 1.
     * @param[in] dilation    (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in] num_groups  (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is not supported
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info,
                           bool has_bias, const Size2D &dilation = Size2D(1U, 1U), unsigned int num_groups = 1);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Template function to run im2col
     *
     * @param[in] window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    template <typename T, bool has_pads, bool is_nchw>
    void run_im2col(const Window &window);

    /** Common signature for all the specialised im2col functions
     *
     * @param[in] window Region on which to execute the kernel.
     */
    using Im2ColFunctionPtr = void (NEIm2ColKernel::*)(const Window &window);

    Im2ColFunctionPtr _func;
    const ITensor    *_input;
    ITensor          *_output;
    std::pair<unsigned int, unsigned int> _convolved_dims;
    PadStrideInfo _conv_info;
    unsigned int  _kernel_width;
    unsigned int  _kernel_height;
    bool          _has_bias;
    Size2D        _dilation;
    DataLayout    _data_layout;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEIM2COLKERNEL_H__ */
