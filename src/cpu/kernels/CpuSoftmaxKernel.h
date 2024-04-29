/*
 * Copyright (c) 2017-2024 Arm Limited.
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
#ifndef ACL_SRC_CPU_KERNELS_CPUSOFTMAXKERNEL_H
#define ACL_SRC_CPU_KERNELS_CPUSOFTMAXKERNEL_H

#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Interface for softmax computation */
class CpuSoftmaxKernel : public ICpuKernel<CpuSoftmaxKernel>
{
private:
    using SoftmaxKernelPtr =
        std::add_pointer<void(const ITensor *, void *const, ITensor *, float, int, const Window &)>::type;

public:
    CpuSoftmaxKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuSoftmaxKernel);

    /** Set the input and output tensors.
     *
     * @param[in]  src    Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[out] dst    Destination tensor info. Data types supported: same as @p input.
     * @param[in]  beta   A scaling factor for the exponent.
     * @param[in]  is_log True if the operation is log-softmax.
     * @param[in]  axis   The axis along which to perform the softmax operation.
     *
     * @param      tmp    Auxiliary tensor info. Must be type F32 and same shape as the input.
     */
    void configure(const ITensorInfo *src, ITensorInfo *dst, float beta, bool is_log, int axis, ITensorInfo *tmp);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuSoftmaxKernel::configure()
     *
     * @return a status
     */
    static Status
    validate(const ITensorInfo *src, const ITensorInfo *dst, float beta, int axis, bool is_log, const ITensorInfo *tmp);

    // Inherited methods overridden:
    void        run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

    struct SoftmaxKernel
    {
        const char                                   *name;
        const SoftmaxKernelDataTypeISASelectorDataPtr is_selected;
        SoftmaxKernelPtr                              ukernel;
    };

    static const std::vector<SoftmaxKernel> &get_available_kernels();

private:
    float            _beta{1.0f};
    SoftmaxKernelPtr _run_method{nullptr};
    std::string      _name{};
    int              _axis{};
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif // ACL_SRC_CPU_KERNELS_CPUSOFTMAXKERNEL_H
