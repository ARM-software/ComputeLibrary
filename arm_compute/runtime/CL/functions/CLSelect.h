/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLSELECT_H__
#define __ARM_COMPUTE_CLSELECT_H__

#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

#include "arm_compute/core/Types.h"

namespace arm_compute
{
// Forward declarations
class ICLTensor;

/** Basic function to run @ref CLSelect */
class CLSelect : public ICLSimpleFunction
{
public:
    /** Initialise the kernel's inputs and output.
     *
     * @param[in]  c      Condition input tensor. Data types supported: U8.
     * @param[in]  x      First input tensor. Data types supported: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32.
     * @param[in]  y      Second input tensor. Data types supported: Same as @p x
     * @param[out] output Output tensor. Data types supported: Same as @p x.
     */
    void configure(const ICLTensor *c, const ICLTensor *x, const ICLTensor *y, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLSelect
     *
     * @param[in] c      Condition input tensor. Data types supported: U8.
     * @param[in] x      First input tensor. Data types supported: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32.
     * @param[in] y      Second input tensor. Data types supported: Same as @p x
     * @param[in] output Output tensor. Data types supported: Same as @p x.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *c, const ITensorInfo *x, const ITensorInfo *y, const ITensorInfo *output);
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLSELECT_H__ */
