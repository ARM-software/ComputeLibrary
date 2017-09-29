/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLTABLELOOKUPKERNEL_H__
#define __ARM_COMPUTE_CLTABLELOOKUPKERNEL_H__

#include "arm_compute/core/CL/ICLSimple2DKernel.h"

namespace arm_compute
{
class ICLTensor;
class ICLLut;

/** Interface for the kernel to perform table lookup calculations. */
class CLTableLookupKernel : public ICLSimple2DKernel
{
public:
    /** Initialise the kernel's input, lut and output.
     *
     * @param[in]  input  An input tensor. Data types supported: U8, S16.
     * @param[in]  lut    The input LUT. Data types supported: U8, S16.
     * @param[out] output The output tensor. Data types supported: U8, S16.
     */
    void configure(const ICLTensor *input, const ICLLut *lut, ICLTensor *output);
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLTABLELOOKUPKERNEL_H__ */
