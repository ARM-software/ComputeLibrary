/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLIM2COLKERNEL_H
#define ARM_COMPUTE_CLIM2COLKERNEL_H

#include "arm_compute/core/Size2D.h"
#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the im2col reshape kernel.
 *
 * Rearranges image blocks into columns. It is used to strip out each convolution block to a single column.
 * It is used to transform a convolution to a plain matrix multiplication.
 *
 * For example taking into account the image below and assuming 3x3 image blocks with stride of 1 we have:
 * @f[
 * \left( \begin{array}{cccc}
 * a00 & a01 & a02 & a03 \\
 * a10 & a11 & a12 & a13 \\
 * a20 & a21 & a22 & a23 \\
 * a30 & a31 & a32 & a33 \\
 * \end{array} \right)
 * =
 * \left( \begin{array}{ccccccccc}
 * a00 & a01 & a02 & a10 & a11 & a12 & a20 & a21 & a22 \\
 * a01 & a02 & a03 & a11 & a12 & a13 & a21 & a22 & a23 \\
 * a10 & a11 & a12 & a20 & a21 & a22 & a30 & a31 & a32 \\
 * a11 & a12 & a13 & a21 & a22 & a23 & a31 & a32 & a33 \\
 * \end{array} \right)
 * @f]
 */
class CLIm2ColKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLIm2ColKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLIm2ColKernel(const CLIm2ColKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLIm2ColKernel &operator=(const CLIm2ColKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLIm2ColKernel(CLIm2ColKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLIm2ColKernel &operator=(CLIm2ColKernel &&) = default;
    /** Set the input and output of the kernel.
     *
     * @param[in]  input       The input tensor to convert. 3 lower dimensions represent a single input [width, height, IFM],
     *                         while every optional dimension from 4 and above represent a batch of inputs. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32
     * @param[out] output      The output tensor. First 2 lower dimensions represent a transform of each 3D input,
     *                         while every dimension above represents a batch. Data types supported: Same as @p input
     * @param[in]  kernel_dims The kernel dimensions (width and height).
     * @param[in]  conv_info   Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  has_bias    In case biases are provided expands the matrix with 1.
     *                         This is valid only for non-quantized inputs.
     * @param[in]  dilation    (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  num_groups  (Optional) Number of groups when performing a grouped convolution.
     *                         Number of groups other than 1 is only supported for NCHW data layout.
     *                         Number of groups should be multiple to the number of channels.
     */
    void configure(const ICLTensor *input, ICLTensor *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias, const Size2D &dilation = Size2D(1U, 1U),
                   unsigned int num_groups = 1);
    /** Set the input and output of the kernel.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           The input tensor to convert. 3 lower dimensions represent a single input [width, height, IFM],
     *                             while every optional dimension from 4 and above represent a batch of inputs. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32
     * @param[out] output          The output tensor. First 2 lower dimensions represent a transform of each 3D input,
     *                             while every dimension above represents a batch. Data types supported: Same as @p input
     * @param[in]  kernel_dims     The kernel dimensions (width and height).
     * @param[in]  conv_info       Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  has_bias        In case biases are provided expands the matrix with 1.
     * @param[in]  dilation        (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  num_groups      (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is only supported for NCHW data layout
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias,
                   const Size2D &dilation   = Size2D(1U, 1U),
                   unsigned int  num_groups = 1);
    /** Static function to check if given info will lead to a valid configuration of @ref CLIm2ColKernel
     *
     * @param[in] input       The input tensor to convert. 3 lower dimensions represent a single input [width, height, IFM],
     *                        while every optional dimension from 4 and above represent a batch of inputs. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32
     * @param[in] output      The output tensor. First 2 lower dimensions represent a transform of each 3D input,
     *                        while every dimension above represents a batch. Data types supported: Same as @p input
     * @param[in] kernel_dims The kernel dimensions (width and height).
     * @param[in] conv_info   Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in] has_bias    In case biases are provided expands the matrix with 1.
     *                        This is valid only for non-quantized inputs.
     * @param[in] dilation    (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in] num_groups  (Optional) Number of groups when performing a grouped convolution.
     *                        Number of groups other than 1 is only supported for NCHW data layout.
     *                        Number of groups should be multiple to the number of channels.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias, const Size2D &dilation = Size2D(1U, 1U),
                           unsigned int num_groups = 1);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

public:
    const ICLTensor *_input;
    ICLTensor       *_output;
    DataLayout       _data_layout;
    std::pair<unsigned int, unsigned int> _convolved_dims;
    unsigned int  _num_elems_processed_per_iteration;
    Size2D        _kernel_dims;
    PadStrideInfo _conv_info;
    unsigned int  _num_groups;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLIM2COLKERNEL_H */
