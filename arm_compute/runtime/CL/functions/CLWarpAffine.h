/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLWARPAFFINE_H
#define ARM_COMPUTE_CLWARPAFFINE_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

#include <cstdint>

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;

/** Basic function to run @ref CLWarpAffineKernel for AFFINE transformation
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
*/
class CLWarpAffine : public ICLSimpleFunction
{
public:
    /** Initialize the function's source, destination, interpolation policy and border_mode.
     *
     * @param[in,out] input                 Source temspr. Data types supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]    output                Destination tensor, Data types supported: U8.
     * @param[in]     matrix                The affine matrix. Must be 2x3 of type float.
     *                                      The matrix argument requires 9 values, the last 3 values are ignored.
     * @param[in]     policy                The interpolation type.
     * @param[in]     border_mode           Strategy to use for borders.
     * @param[in]     constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(ICLTensor *input, ICLTensor *output, const std::array<float, 9> &matrix, InterpolationPolicy policy, BorderMode border_mode, uint8_t constant_border_value = 0);
    /** Initialize the function's source, destination, interpolation policy and border_mode.
     *
     * @param[in]     compile_context       The compile context to be used.
     * @param[in,out] input                 Source temspr. Data types supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]    output                Destination tensor, Data types supported: U8.
     * @param[in]     matrix                The affine matrix. Must be 2x3 of type float.
     *                                      The matrix argument requires 9 values, the last 3 values are ignored.
     * @param[in]     policy                The interpolation type.
     * @param[in]     border_mode           Strategy to use for borders.
     * @param[in]     constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, const std::array<float, 9> &matrix, InterpolationPolicy policy, BorderMode border_mode,
                   uint8_t constant_border_value = 0);
};
}
#endif /*ARM_COMPUTE_CLWARPAFFINE_H */
