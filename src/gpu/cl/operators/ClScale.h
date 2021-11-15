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
#ifndef ARM_COMPUTE_CL_SCALE_H
#define ARM_COMPUTE_CL_SCALE_H

#include "arm_compute/core/KernelDescriptors.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClOperator.h"

namespace arm_compute
{
namespace opencl
{
/** Basic function to simulate a scale layer. This function calls the following OpenCL kernels:
 *
 * -# @ref kernels::ClScaleKernel
 */
class ClScale : public IClOperator
{
public:
    /** Constructor */
    ClScale() = default;
    /** Initialize the function's source, destination, interpolation type and border_mode.
     *
     * @param[in]     compile_context The compile context to be used.
     * @param[in,out] src             Source tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/F16/F32. (Written to only for @p border_mode != UNDEFINED)
     * @param[out]    dst             Destination tensor info. Data types supported: Same as @p src
     *                                All but the lowest two dimensions must be the same size as in the input tensor, i.e. scaling is only performed within the XY-plane.
     * @param[in]     info            @ref ScaleKernelInfo descriptor to be used to configure
     */
    void configure(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *dst, const ScaleKernelInfo &info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to ClScale::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst, const ScaleKernelInfo &info);

    // Inherited method overridden
    void run(ITensorPack &tensors) override;
};
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLSCALE_H */
