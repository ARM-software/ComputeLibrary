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
#ifndef ARM_COMPUTE_GCGEMMINTERLEAVE4X4_H
#define ARM_COMPUTE_GCGEMMINTERLEAVE4X4_H

#include "arm_compute/runtime/GLES_COMPUTE/IGCSimpleFunction.h"

namespace arm_compute
{
class ITensor;

/** Basic function to execute GCGEMMInterleave4x4Kernel. This function calls the following OpenGL ES kernel:
 *
 *  -# @ref GCGEMMInterleave4x4Kernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class GCGEMMInterleave4x4 : public IGCSimpleFunction
{
public:
    /** Initialise the kernel's inputs, output
     *
     * @param[in]  input  First input tensor. Data types supported: F32, F16
     * @param[out] output Output tensor. Data type supported: same as @p input
     */
    void configure(const IGCTensor *input, IGCTensor *output);
};
}

#endif /* ARM_COMPUTE_GCGEMMINTERLEAVE4X4_H */
