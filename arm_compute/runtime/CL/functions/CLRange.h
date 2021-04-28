/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLRANGE_H
#define ARM_COMPUTE_CLRANGE_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

/** Basic function to run @ref CLRangeKernel
 *
 * @note The tensor data type for the output must be U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32.
 * @note The function performs generates a sequence with the given start, end and step.
 */
class CLRange : public ICLSimpleFunction
{
public:
    /** Initialize the kernel's start, end, step and output tensor.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |dst       |
     * |:---------|
     * |U8        |
     * |S8        |
     * |QASYMM8   |
     * |U16       |
     * |S16       |
     * |U32       |
     * |S32       |
     * |F16       |
     * |F32       |
     *
     * @param[out] output Output tensor. Data types supported: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32.
     * @param[in]  start  The starting value of the sequence.
     * @param[in]  end    The ending (not including) value of the sequence.
     * @param[in]  step   The gap between each pair of values in the sequence. Default is 1.
     */
    void configure(ICLTensor *output, float start, float end, float step = 1.f);
    /** Initialize the kernel's start, end, step and output tensor.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[out] output          Output tensor. Data types supported: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32.
     * @param[in]  start           The starting value of the sequence.
     * @param[in]  end             The ending (not including) value of the sequence.
     * @param[in]  step            The gap between each pair of values in the sequence. Default is 1.
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *output, float start, float end, float step = 1.f);
    /** Static function to check if given info will lead to a valid configuration of @ref CLRange
     *
     * @param[in] output Output tensor info. Data types supported: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32.
     * @param[in] start  The starting value of the sequence.
     * @param[in] end    The ending (not including) value of the sequence.
     * @param[in] step   The gap between each pair of values in the sequence. Default is 1.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *output, float start, float end, float step = 1.f);
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLRANGE_H */
