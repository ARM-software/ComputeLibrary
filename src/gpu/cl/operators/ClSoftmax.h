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
#ifndef ARM_COMPUTE_CL_SOFTMAX_H
#define ARM_COMPUTE_CL_SOFTMAX_H

#include "arm_compute/runtime/CL/CLTensor.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClOperator.h"

namespace arm_compute
{
struct SoftmaxKernelInfo;

namespace opencl
{
class ClPermute;
namespace kernels
{
class ClLogits1DMaxShiftExpSumKernel;
class ClLogits1DNormKernel;
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
    void configure(const CLCompileContext &compile_context, const ITensorInfo &src, ITensorInfo &dst, const SoftmaxKernelInfo &info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to ClSoftmax::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo &src, const ITensorInfo &dst, const SoftmaxKernelInfo &info);
    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
    experimental::MemoryRequirements workspace() const override;

private:
    enum InternalTensorIdx
    {
        MAX = 0,
        SUM,
        TMP,
        PERMUTED_SRC,
        PERMUTED_DST,
        COUNT
    };

    std::unique_ptr<ClPermute>                               _permute_input;
    std::unique_ptr<ClPermute>                               _permute_output;
    std::unique_ptr<kernels::ClLogits1DMaxShiftExpSumKernel> _max_shift_exp_sum_kernel;
    std::unique_ptr<kernels::ClLogits1DNormKernel>           _norm_kernel;
    bool                                                     _needs_permute{ false };

    TensorInfo _max_info;
    TensorInfo _sum_info;
    TensorInfo _tmp_info;
    TensorInfo _permuted_src_info;
    TensorInfo _permuted_dst_info;

    experimental::MemoryRequirements _aux_mem{};
};

} // opencl
} // arm_compute
#endif /* ARM_COMPUTE_CL_SOFTMAX_H */