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
#ifndef ARM_COMPUTE_CL_FLOOR_H
#define ARM_COMPUTE_CL_FLOOR_H

#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClOperator.h"

namespace arm_compute
{
namespace opencl
{
/** Basic function to run @ref kernels::ClFloorKernel */
class ClFloor : public IClOperator
{
public:
    /** Configure operator for a given list of arguments
     *
     * @param[in] compile_context The compile context to be used.
     * @param[in] src             Source tensor info. Data types supported: F16/F32.
     * @param[in] dst             Destination tensor info. Data type supported: same as @p src
     */
    void configure(const ClCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to ClFloor::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst);
};
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_FLOOR_H */
