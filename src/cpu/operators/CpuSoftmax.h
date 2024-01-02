/*
 * Copyright (c) 2021-2024 Arm Limited.
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
#ifndef ACL_SRC_CPU_OPERATORS_CPUSOFTMAX_H
#define ACL_SRC_CPU_OPERATORS_CPUSOFTMAX_H

#include "arm_compute/core/experimental/Types.h"
#include "arm_compute/core/TensorInfo.h"

#include "src/cpu/ICpuKernel.h"
#include "src/cpu/ICpuOperator.h"
#include "src/cpu/operators/CpuPermute.h"

#include <memory>

namespace arm_compute
{
namespace cpu
{
class CpuSoftmaxKernel;

/** Basic function to compute a SoftmaxLayer and a Log SoftmaxLayer.
 *
 * Softmax is calculated by :
 * @f[ out = exp((x - max(x)) * beta) / sum(exp((x - max(x)) * beta)) @f]
 *
 * Log Softmax is calculated by :
 * @f[ out = (x - max(x) * beta) - log(\sum{e^{x - max(x) * beta}}) @f]
 *
 * This function runs the following function/kernels:
 * -# If axis is not 0:
 * -# @ref CpuPermute
 * -# @ref kernels::CpuSoftmaxKernel
 */
class CpuSoftmaxGeneric : public ICpuOperator
{
public:
    CpuSoftmaxGeneric();
    /** Set the input and output tensors.
     *
     * @param[in,out] src    Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     *                       last value of each row to the nearest multiple.
     * @param[out]    dst    Destination tensor ifo. Data types supported: same as @p input.
     * @param[in]     beta   (Optional) A scaling factor for the exponent.
     * @param[in]     axis   (Optional) The dimension in which to apply the function. E.g. for input of shape 4x5x6 and
     *                       axis=1, softmax will be applied to 4x6=24 vectors of size 5. Defaults to 0
     * @param[in]     is_log True if the operation is log-softmax
     */
    void configure(const ITensorInfo *src, ITensorInfo *dst, float beta = 1.0f, int32_t axis = 0, bool is_log = false);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuSoftmaxGeneric::configure()
     *
     * @return a status
     */
    static Status
    validate(const ITensorInfo *src, const ITensorInfo *dst, float beta = 1.0f, int32_t axis = 0, bool is_log = false);

    // Inherited methods overridden:
    void                             run(ITensorPack &tensors) override;
    experimental::MemoryRequirements workspace() const override;

private:
    enum InternalTensorIdx
    {
        TMP = 0,
        PERMUTED_SRC,
        PERMUTED_DST,
        COUNT
    };

    std::unique_ptr<ICPPKernel> _softmax_kernel;

    TensorInfo _tmp;

    experimental::MemoryRequirements _aux_mem{};

    unsigned int _axis = 0;
};

} // namespace cpu
} // namespace arm_compute
#endif // ACL_SRC_CPU_OPERATORS_CPUSOFTMAX_H
