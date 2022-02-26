/*
 * Copyright (c) 2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CL_POOL3D_H
#define ARM_COMPUTE_CL_POOL3D_H

#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClOperator.h"

#include <memory>

namespace arm_compute
{
namespace opencl
{
/** Basic function to simulate a pooling layer with the specified pooling operation. This function calls the following OpenCL kernels:
 *
 * -# @ref opencl::ClPool3d
 */
class ClPool3d : public IClOperator
{
public:
    /** Constructor */
    ClPool3d() = default;
    /** Configure operator for a given list of arguments
     *
     * @note Asymmetric padding is not supported when dimension rounding type == CEIL.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src             Source tensor info.
     * @param[out] dst             Destination tensor info.
     * @param[in]  info            3d Pooling layer parameters.
     */
    void configure(const ClCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst, const Pooling3dLayerInfo &info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to ClPool3d::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst, const Pooling3dLayerInfo &info);
};
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_POOL3D_H */
