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
#ifndef __ARM_COMPUTE_GCTENSORSHIFTKERNEL_H__
#define __ARM_COMPUTE_GCTENSORSHIFTKERNEL_H__

#include "arm_compute/core/GLES_COMPUTE/IGCKernel.h"
#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"

namespace arm_compute
{
class IGCTensor;
/** Interface for the kernel to shift valid data on a tensor.
 *
 * For example shifting 3x3 valid data with padding of 1 to right:
 * @f[
 * \left( \begin{array}{ccccc}
 * 0   & 0   & 0   & 0 & 0 \\
 * a00 & a01 & a02 & 0 & 0 \\
 * a10 & a11 & a12 & 0 & 0 \\
 * a20 & a21 & a22 & 0 & 0 \\
 * 0   & 0   & 0   & 0 & 0 \\
 * \end{array} \right)
 * =
 * \left( \begin{array}{ccccc}
 * 0  & 0   & 0   & 0   & 0 \\
 * 0  & a00 & a01 & a02 & 0 \\
 * 0  & a10 & a11 & a12 & 0 \\
 * 0  & a20 & a21 & a22 & 0 \\
 * 0  & 0   & 0   & 0   & 0 \\
 * \end{array} \right)
 * @f]
 */
class GCTensorShiftKernel : public IGCKernel
{
public:
    /** Default constructor */
    GCTensorShiftKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCTensorShiftKernel(const GCTensorShiftKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCTensorShiftKernel &operator=(const GCTensorShiftKernel &) = delete;
    /** Allow instances of this class to be moved */
    GCTensorShiftKernel(GCTensorShiftKernel &&) = default;
    /** Allow instances of this class to be moved */
    GCTensorShiftKernel &operator=(GCTensorShiftKernel &&) = default;
    /** Default destructor */
    ~GCTensorShiftKernel() = default;
    /** Set the input of the kernel.
     *
     * @param[in,out] input Source tensor. Data types supported: F16/F32
     */
    void configure(IGCTensor *input);

    // Inherited methods overridden:
    void run(const Window &window) override;

private:
    IGCTensor    *_input;
    gles::NDRange _lws;
};
}
#endif /*__ARM_COMPUTE_GCTENSORSHIFTKERNEL_H__ */
