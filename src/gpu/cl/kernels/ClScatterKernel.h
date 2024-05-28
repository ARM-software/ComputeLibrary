/*
 * Copyright (c) 2024 Arm Limited.
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

#ifndef ACL_SRC_GPU_CL_KERNELS_CLSCATTERKERNEL_H
#define ACL_SRC_GPU_CL_KERNELS_CLSCATTERKERNEL_H

#include "arm_compute/function_info/ScatterInfo.h"

#include "src/core/common/Macros.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{

class ClScatterKernel : public IClKernel
{
public:
    ClScatterKernel();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClScatterKernel);
    /** Initialise the kernel's input and output.
     *
     * @note Negative indices are treated as out of bounds.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  updates         Input tensor info for the Update matrix. Data type supported: same as @p src
     * @param[in]  indices         Input tensor info for the Indices matrix. Data type supported: S32.
     * @param[out] dst             Output tensor info. Data type supported: same as @p src
     * @param[in]  info            Attributes for Scatter Kernel
     */
    void configure(const ClCompileContext &compile_context,
                   const ITensorInfo      *updates,
                   const ITensorInfo      *indices,
                   ITensorInfo            *dst,
                   const ScatterInfo      &info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClScatterKernel::configure()
     *
     * @return a status
     */
    static Status
    validate(const ITensorInfo *updates, const ITensorInfo *indices, const ITensorInfo *dst, const ScatterInfo &info);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override;
};
} // namespace kernels
} // namespace opencl
} // namespace arm_compute

#endif // ACL_SRC_GPU_CL_KERNELS_CLSCATTERKERNEL_H
