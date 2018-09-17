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
#ifndef __ARM_COMPUTE_CLDEPTHCONVERT_H__
#define __ARM_COMPUTE_CLDEPTHCONVERT_H__

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

#include <cstdint>

namespace arm_compute
{
class ICLTensor;

/** Basic function to run @ref CLDepthConvertKernel. */
class CLDepthConvert : public ICLSimpleFunction
{
public:
    /** Initialize the function's source, destination
     *
     * Input data type must be different than output data type.
     *
     * Valid conversions Input -> Output :
     *
     *   - U8 -> U16, S16, U32, S32
     *   - U16 -> U8, U32, S32
     *   - S16 -> U8, U32, S32
     *   - U32 -> U8, U16, S16
     *   - S32 -> U8, U16, S16
     *
     * @param[in]  input  The input tensor to convert. Data types supported: U8, U16, S16, U32 or S32.
     * @param[out] output The output tensor. Data types supported: U8, U16, S16, U32 or S32.
     * @param[in]  policy Conversion policy.
     * @param[in]  shift  Value for down/up conversions. Must be 0 <= shift < 8.
     */
    void configure(const ICLTensor *input, ICLTensor *output, ConvertPolicy policy, uint32_t shift);
};
}
#endif /*__ARM_COMPUTE_CLDEPTHCONVERT_H__*/
