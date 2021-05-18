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
#include "src/core/gpu/cl/ClCompileContext.h"
#include "src/runtime/gpu/cl/IClOperator.h"

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
     *
     */
    void configure(const CLCompileContext &compile_context, const ITensorInfo &src, ITensorInfo &dst, const SoftmaxKernelInfo &info);
    /** Static function to check if the given info will lead to a valid configuration
     *
     * @param[in]  src  Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32 for Softmax and F16/F32 for Log Softmax
     * @param[out] dst  Destination tensor info. Data types supported: same as @p src
     * @param[in]  info Contains information consumed by kernels for softmax described in @ref SoftmaxKernelInfo.
     *
     */
    static Status validate(const ITensorInfo &src, const ITensorInfo &dst, const SoftmaxKernelInfo &info);
    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
    experimental::MemoryRequirements workspace() const override;

private:
    enum class InternalTensorIdx
    {
        MAX = 0,
        SUM,
        TMP,
        PERMUTED_SRC,
        PERMUTED_DST,
        COUNT
    };

    /** Create a single internal tensor
     *
     * @param[in] info The information used to create a tensor
     * @param[in] idx  The index within the internal array the created tensor will be held
     */
    void create_internal_tensor(TensorInfo &info, InternalTensorIdx idx);
    /** Create all required internal tensors */
    void create_internal_tensor();
    /** Function to convert from internal tensor index to @ref TensorType used externally */
    TensorType convert_internal_idx_to_tensor_type(InternalTensorIdx idx) const;
    /** Function to import workspace memory allocated by the caller into internal tensor instances */
    void import_workspace_memory(ITensorPack &tensors);
    /** Function to permute the given source tensor when permutation is required */
    void run_source_permute(const ITensor *src);
    /** Function to permute the intemediate tensor to the final destination tensor when permutation is required */
    void run_destination_permute(ITensor *dst);
    /** Function to run @ref arm_compute::opencl::kernels::ClLogits1DMaxShiftExpSumKernel */
    void run_max_sum(const ITensor *src);
    /** Function to run @ref kernels::ClLogits1DNormKernel */
    void run_norm(ITensor *dst);

    std::unique_ptr<ClPermute>                               _permute_input;
    std::unique_ptr<ClPermute>                               _permute_output;
    std::unique_ptr<kernels::ClLogits1DMaxShiftExpSumKernel> _max_shift_exp_sum_kernel;
    std::unique_ptr<kernels::ClLogits1DNormKernel>           _norm_kernel;
    bool                                                     _needs_permute{ false };

    std::array<TensorInfo, static_cast<uint32_t>(InternalTensorIdx::COUNT)>                _internal_info{};
    std::array<std::unique_ptr<CLTensor>, static_cast<uint32_t>(InternalTensorIdx::COUNT)> _internal_tensor{};

    TensorInfo &_max_info;
    TensorInfo &_sum_info;
    TensorInfo &_tmp_info;
    TensorInfo &_permuted_src_info;
    TensorInfo &_permuted_dst_info;
};

} // opencl
} // arm_compute
#endif /* ARM_COMPUTE_CL_SOFTMAX_H */