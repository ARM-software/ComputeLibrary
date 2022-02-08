/*
 * Copyright (c) 2021-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CL_WIDTHCONCATENATE_2TENSORS_KERNEL_H
#define ARM_COMPUTE_CL_WIDTHCONCATENATE_2TENSORS_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
/** Interface for the width concatenate kernel of 2 tensors.
 *  The src1 and src2 tensors will be concatenated into the dst tensor.
 */
class ClWidthConcatenate2TensorsKernel : public IClKernel
{
public:
    ClWidthConcatenate2TensorsKernel();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClWidthConcatenate2TensorsKernel);
    /** Initialise the kernel's sources and destination
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src1            First source tensor info. Data types supported: All.
     * @param[in]  src2            Second source tensor info. Data types supported: same as @p src1
     * @param[out] dst             Destination tensor info. Data types supported: Same as @p src1.
     */
    void configure(const CLCompileContext &compile_context, ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClWidthConcatenate2TensorsKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, ::cl::CommandQueue &queue) override;

private:
    int32_t _depth{ 0 };
    int32_t _input1_width{ 0 };
};
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_WIDTH_CONCATENATE_2TENSORS_KERNEL_H */
