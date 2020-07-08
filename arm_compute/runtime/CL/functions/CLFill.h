/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLFILL_H
#define ARM_COMPUTE_CLFILL_H

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
class ICLTensor;

/** Function to run @ref CLMemsetKernel to fill a tensor with a scalar value */
class CLFill : public ICLSimpleFunction
{
public:
    /** Initialize the function
     *
     * @param[in,out] tensor         Source tensor. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
     * @param[in]     constant_value Constant value to use to fill tensor.
     */
    void configure(ICLTensor *tensor, PixelValue constant_value);
    /** Initialize the function
     *
     * @param[in]     compile_context The compile context to be used.
     * @param[in,out] tensor          Source tensor. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
     * @param[in]     constant_value  Constant value to use to fill tensor.
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *tensor, PixelValue constant_value);
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLFILL_H */
