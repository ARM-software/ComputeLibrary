/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLCOPY_H
#define ARM_COMPUTE_CLCOPY_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

#include <cstdint>

namespace arm_compute
{
class ICLTensor;

class CLCopy : public ICLSimpleFunction
{
public:
    /** Initialise the function's source and destination.
     *
     * @param[in]  input  Source tensor. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[out] output Output tensor. Data types supported: Same as @p input.
     *
     */
    void configure(ICLTensor *input, ICLTensor *output);
    /** Initialise the function's source and destination.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[out] output          Output tensor. Data types supported: Same as @p input.
     *
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLCopy
     *
     * @param[in] input  Source tensor. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[in] output Output tensor. Data types supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLCOPY_H */
