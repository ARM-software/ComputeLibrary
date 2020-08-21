/*
 * Copyright (c) 2017-2019 Arm Limited.
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
#ifndef ARM_COMPUTE_NEGEMMINTERLEAVE4X4_H
#define ARM_COMPUTE_NEGEMMINTERLEAVE4X4_H

#include "arm_compute/runtime/NEON/INESimpleFunctionNoBorder.h"

namespace arm_compute
{
class ITensor;

/** Basic function to execute NEGEMMInterleave4x4Kernel. This function calls the following NEON kernel:
 *
 *  -# @ref NEGEMMInterleave4x4Kernel
 *
 */
class NEGEMMInterleave4x4 : public INESimpleFunctionNoBorder
{
public:
    /** Initialise the kernel's inputs, output
     *
     * @param[in]  input  First input tensor. Data types supported: All
     * @param[out] output Output tensor. Data type supported: same as @p input
     */
    void configure(const ITensor *input, ITensor *output);
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEGEMMINTERLEAVE4X4_H */
