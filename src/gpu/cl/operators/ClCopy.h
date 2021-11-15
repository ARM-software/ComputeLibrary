/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CL_COPY_H
#define ARM_COMPUTE_CL_COPY_H

#include "arm_compute/core/Window.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClOperator.h"

namespace arm_compute
{
namespace opencl
{
/** Basic function to run @ref kernels::ClCopyKernel */
class ClCopy : public IClOperator
{
public:
    /** Initialise the function's source and destination.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src             Source tensor info. Data types supported: All.
     * @param[out] dst             Output tensor info. Data types supported: Same as @p src.
     * @param[in]  dst_window      (Optional) Window to be used in case only copying into part of a tensor. Default is nullptr.
     *
     */
    void configure(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *dst, Window *dst_window = nullptr);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClCopy::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst, Window *dst_window = nullptr);
};
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_COPY_H */
