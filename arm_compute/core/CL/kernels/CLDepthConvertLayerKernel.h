/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLDEPTHCONVERTKERNEL_H__
#define __ARM_COMPUTE_CLDEPTHCONVERTKERNEL_H__

#include "arm_compute/core/CL/ICLSimple3DKernel.h"
#include "arm_compute/core/Types.h"

#include <cstdint>

namespace arm_compute
{
class ICLTensor;

/** Interface for the depth conversion kernel. */
class CLDepthConvertLayerKernel : public ICLSimple3DKernel
{
public:
    /** Set the input and output of the kernel.
     *
     * Valid conversions Input -> Output :
     *
     *   - U8  -> S8, U16, S16, U32, S32, F16, F32
     *   - U16 -> U8, S8, S16, U32, S32, F16, F32
     *   - S16 -> U8, S8, U16, U32, S32, F16, F32
     *   - U32 -> U8, S8, U16, S16, S32, F16, F32
     *   - S32 -> U8, S8, U16, S16, U32, F16, F32
     *   - F16 -> U8, S8, U16, S16, U32, F32
     *   - F32 -> U8, S8, U16, S16, U32, F16
     *
     * @param[in]  input  The input tensor to convert. Data types supported: U8/S8/U16/S16/U32/S32/F16/F32.
     * @param[out] output The output tensor. Data types supported: U8/S8/U16/S16/U32/S32/F16/F32.
     * @param[in]  policy Conversion policy
     * @param[in]  shift  Value for down/up conversions. Must be 0 <= shift < 8.
     */
    void configure(const ICLTensor *input, ICLTensor *output, ConvertPolicy policy, uint32_t shift);
    /** Static function to check if given info will lead to a valid configuration of @ref CLDepthConvertLayerKernel
     *
     * @param[in] input  Source tensor info. Data types supported: U8/S8/U16/S16/U32/S32/F16/F32.
     * @param[in] output Destination tensor info. Data type supported: U8/S8/U16/S16/U32/S32/F16/F32.
     * @param[in] policy Conversion policy
     * @param[in] shift  Value for down/up conversions. Must be 0 <= shift < 8.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, ConvertPolicy policy, uint32_t shift);
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLDEPTHCONVERTKERNEL_H__ */
