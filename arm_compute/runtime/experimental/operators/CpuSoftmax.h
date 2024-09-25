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

#ifndef ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUSOFTMAX_H
#define ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUSOFTMAX_H

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/IOperator.h"

#include <memory>

namespace arm_compute
{
namespace experimental
{
namespace op
{
class CpuSoftmaxKernel;

/*
 * A shallow wrapper for arm_compute::cpu::CpuSoftmaxGeneric.
 * Any new features should be added to arm_compute::cpu::CpuSoftmaxGeneric
 * and arm_compute::experimental::op::CpuSoftmax should remain a shallow wrapper.
 */
class CpuSoftmax : public IOperator
{
public:
    /** Constructor **/
    CpuSoftmax();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuSoftmax(const CpuSoftmax &) = delete;
    /** Prevent copy assignment */
    CpuSoftmax &operator=(const CpuSoftmax &) = delete;
    /** Default move constructor */
    CpuSoftmax(CpuSoftmax &&) = default;
    /** Default move assignment */
    CpuSoftmax &operator=(CpuSoftmax &&) = default;
    /** Default destructor */
    ~CpuSoftmax() override;
    /** Set the input and output tensors.
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |QASYMM8        |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |
     * |F16            |F16            |
     * |F32            |F32            |
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
     * Similar to @ref CpuSoftmax::configure()
     *
     * @return a status
     */
    static Status
    validate(const ITensorInfo *src, const ITensorInfo *dst, float beta = 1.0f, int32_t axis = 0, bool is_log = false);

    // Inherited methods overridden:
    void                             run(ITensorPack &tensors) override;
    experimental::MemoryRequirements workspace() const override;

    // Unused
    void prepare(ITensorPack &constants) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace op
} // namespace experimental
} // namespace arm_compute
#endif // ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUSOFTMAX_H
