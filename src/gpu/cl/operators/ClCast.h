/*
 * Copyright (c) 2021, 2024 Arm Limited.
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
#ifndef ACL_SRC_GPU_CL_OPERATORS_CLCAST_H
#define ACL_SRC_GPU_CL_OPERATORS_CLCAST_H

#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClOperator.h"

namespace arm_compute
{
namespace opencl
{
/** Basic function to run @ref kernels::ClCastKernel */
class ClCast : public IClOperator
{
public:
    /** Configure operator for a given list of arguments
     *
     * @note Input data type must be different than output data type.
     *
     * Valid data layouts:
     * - All
     *
     * For data type configurations supported, please have a look at @ref CLCast
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src             The source tensor to convert.
     * @param[out] dst             The destinatio tensor.
     * @param[in]  policy          Conversion policy.
     */
    void
    configure(const ClCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst, ConvertPolicy policy);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClCast::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst, ConvertPolicy policy);
};
} // namespace opencl
} // namespace arm_compute
#endif // ACL_SRC_GPU_CL_OPERATORS_CLCAST_H
