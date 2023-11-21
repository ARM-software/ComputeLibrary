/*
 * Copyright (c) 2021, 2023 Arm Limited.
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
#ifndef ACL_SRC_GPU_CL_OPERATORS_CLSOFTMAX_H
#define ACL_SRC_GPU_CL_OPERATORS_CLSOFTMAX_H

#include "arm_compute/core/experimental/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"

#include "src/gpu/cl/IClOperator.h"

namespace arm_compute
{
class CLCompileContext;
class ITensorInfo;
class ITensorPack;
struct SoftmaxKernelInfo;

namespace opencl
{
namespace kernels
{
class ClSoftmaxKernel;
} // namespace kernels
class ClSoftmax : public IClOperator
{
public:
    /** Constructor */
    ClSoftmax();
    /** Configure the operator
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
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to ClSoftmax::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo &src, const ITensorInfo &dst, const SoftmaxKernelInfo &info);

    void run(ITensorPack &tensors) override;

    experimental::MemoryRequirements workspace() const override;

private:
    enum InternalTensorIdx
    {
        TMP = 0,
        COUNT,
    };

    TensorInfo                       _tmp_info{};
    experimental::MemoryRequirements _aux_mem;
};

} // namespace opencl
} // namespace arm_compute
#endif // ACL_SRC_GPU_CL_OPERATORS_CLSOFTMAX_H
