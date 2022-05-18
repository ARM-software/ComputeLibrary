/*
 * Copyright (c) 2017-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_SOFTMAX_KERNEL_H
#define ARM_COMPUTE_CPU_SOFTMAX_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Interface for the identifying the max value of 1D Logits */
class CpuLogits1DMaxKernel : public ICpuKernel<CpuLogits1DMaxKernel>
{
private:
    using SoftmaxLogits1DMaxKernelPtr = std::add_pointer<void(const ITensor *, ITensor *, const Window &)>::type;

public:
    CpuLogits1DMaxKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuLogits1DMaxKernel);
    /** Set the input and output tensors.
     *
     * @param[in]  src Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[out] dst Destination tensor info. Data types supported: same as @p input
     */
    void configure(const ITensorInfo *src, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuLogits1DMaxKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

    struct SoftmaxLogits1DMaxKernel
    {
        const char                  *name;
        const DataTypeISASelectorPtr is_selected;
        SoftmaxLogits1DMaxKernelPtr  ukernel;
    };

    static const std::vector<SoftmaxLogits1DMaxKernel> &get_available_kernels();

private:
    SoftmaxLogits1DMaxKernelPtr _run_method{ nullptr };
    std::string                 _name{};
};

/** Interface for softmax computation for QASYMM8 with pre-computed max. */
template <bool IS_LOG = false>
class CpuLogits1DSoftmaxKernel : public ICpuKernel<CpuLogits1DSoftmaxKernel<IS_LOG>>
{
private:
    using SoftmaxLogits1DKernelPtr = std::add_pointer<void(const ITensor *, const ITensor *, void *const, ITensor *, float, bool, const Window &)>::type;

public:
    CpuLogits1DSoftmaxKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuLogits1DSoftmaxKernel);

    /** Set the input and output tensors.
     *
     * @param[in]  src  Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  max  Max values tensor info. Same shape as input with dimension 0 set to 1.
     *                  Data types supported: same as @p input.
     * @param[out] dst  Destination tensor info. Data types supported: same as @p input.
     * @param[in]  beta A scaling factor for the exponent.
     *
     * @param      tmp    Auxiliary tensor info. Must be type F32 and same shape as the input.
     */
    void configure(const ITensorInfo *src, const ITensorInfo *max, ITensorInfo *dst, const float beta, ITensorInfo *tmp);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuLogits1DSoftmaxKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *max,
                           const ITensorInfo *dst, const float beta, const ITensorInfo *tmp);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

    struct SoftmaxLogits1DKernel
    {
        const char                  *name;
        const DataTypeISASelectorPtr is_selected;
        SoftmaxLogits1DKernelPtr     ukernel;
    };

    static const std::vector<SoftmaxLogits1DKernel> &get_available_kernels();

private:
    float                    _beta{ 1.0f };
    SoftmaxLogits1DKernelPtr _run_method{ nullptr };
    std::string              _name{};
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_SOFTMAX_KERNEL_H */
