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
#ifndef __ARM_COMPUTE_CLIM2COLKERNEL_H__
#define __ARM_COMPUTE_CLIM2COLKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

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
     * @param[in]  input          The input tensor to convert. 3 lower dimensions represent a single input [width, height, IFM],
     *                            while every optional dimension from 4 and above represent a batch of inputs. Data types supported: F16, F32
     * @param[out] output         The output tensor. First 2 lower dimensions represent a transform of each 3D input,
     *                            while every dimension above represents a batch. Data types supported: Same as @p input
     * @param[in]  convolved_dims The convolved output dimensions.
     * @param[in]  conv_info      Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  has_bias       In case biases are provided expands the matrix with 1.
     */
    void configure(const ICLTensor *input, ICLTensor *output, std::pair<unsigned int, unsigned int> convolved_dims, const PadStrideInfo &conv_info, bool has_bias);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    /** Run the reshape kernel optimised for the special case (stride is 1, padding is 0 and kernel's low 3 dimensions are same as input)
     *
     * @param[in]     window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     * @param[in,out] queue  Command queue on which to enqueue the kernel.
     */
    void run_reduced(const Window &window, cl::CommandQueue &queue);
    /** run the generic convolution layer input reshape kernel
     *
     * @param[in]     window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     * @param[in,out] queue  Command queue on which to enqueue the kernel.
     */
    void run_generic(const Window &window, cl::CommandQueue &queue);

    /** Common signature for the kernel to run */
    using Im2ColFunction = void (CLIm2ColKernel::*)(const Window &, cl::CommandQueue &);

private:
    const ICLTensor *_input;
    ICLTensor       *_output;
    std::pair<unsigned int, unsigned int> _convolved_dims;
    PadStrideInfo  _conv_info;
    int            _kernel_size;
    unsigned int   _num_elems_processed_per_iteration;
    Im2ColFunction _run_func;
};
}

#endif /*__ARM_COMPUTE_CLIM2COLKERNEL_H__ */
