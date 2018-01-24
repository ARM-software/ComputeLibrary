/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_GCGEMMINTERLEAVE4X4KERNEL_H__
#define __ARM_COMPUTE_GCGEMMINTERLEAVE4X4KERNEL_H__

#include "arm_compute/core/GLES_COMPUTE/IGCKernel.h"

namespace arm_compute
{
class IGCTensor;

/** OpenGL ES kernel which interleaves the elements of a matrix A in chunk of 4x4
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
 * After this operation, the output matrix will have the following shape: [ height * 4, ceil(width / 4.0f) ]
 */
class GCGEMMInterleave4x4Kernel : public IGCKernel
{
public:
    /** Default constructor */
    GCGEMMInterleave4x4Kernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCGEMMInterleave4x4Kernel(const GCGEMMInterleave4x4Kernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCGEMMInterleave4x4Kernel &operator=(const GCGEMMInterleave4x4Kernel &) = delete;
    /** Allow instances of this class to be moved */
    GCGEMMInterleave4x4Kernel(GCGEMMInterleave4x4Kernel &&) = default;
    /** Allow instances of this class to be moved */
    GCGEMMInterleave4x4Kernel &operator=(GCGEMMInterleave4x4Kernel &&) = default;
    /** Initialise the kernel's input and output.
     *
     * @param[in]  input  Input tensor. Data types supported: F16, F32
     * @param[out] output Output tensor. Data type supported: same as @p input
     */
    void configure(const IGCTensor *input, IGCTensor *output);

    // Inherited methods overridden
    void run(const Window &window) override;

private:
    const IGCTensor *_input;
    IGCTensor       *_output;
};
}
#endif /* __ARM_COMPUTE_GCGEMMINTERLEAVE4X4KERNEL_H__ */
