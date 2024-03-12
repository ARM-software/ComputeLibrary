/*
 * Copyright (c) 2017-2021, 2023 Arm Limited.
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
#ifndef ACL_SRC_GPU_CL_KERNELS_CLSOFTMAXKERNEL_H
#define ACL_SRC_GPU_CL_KERNELS_CLSOFTMAXKERNEL_H

#include "arm_compute/core/Error.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Window.h"

#include "src/core/common/Macros.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{

/** The CL kernel that performs softmax function. */
class ClSoftmaxKernel : public IClKernel
{
public:
    ClSoftmaxKernel();

    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClSoftmaxKernel);

    /** Check if the kernel arguments are valid.
     *
     * See @ref ClSoftmaxKernel::configure().
     *
     * @return The status.
     */
    static Status validate(const ITensorInfo &src, const ITensorInfo &dst, const SoftmaxKernelInfo &info);

    /** Configure the kernel.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src             Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32 for Softmax and F16/F32 for Log Softmax
     * @param[out] dst             Destination tensor info. Data types supported: same as @p src
     * @param[in]  info            Contains information consumed by kernels for softmax described in @ref SoftmaxKernelInfo.
     */
    void configure(const CLCompileContext  &compile_context,
                   const ITensorInfo       &src,
                   ITensorInfo             &dst,
                   const SoftmaxKernelInfo &info);

    void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override;

    /** Get the tensor info of the temporary tensor. */
    const TensorInfo &tmp_tensor_info() const;

private:
    bool       _prepared{false};
    int32_t    _axis{0};
    TensorInfo _tmp_info{};
};

} // namespace kernels
} // namespace opencl
} // namespace arm_compute
#endif // ACL_SRC_GPU_CL_KERNELS_CLSOFTMAXKERNEL_H
