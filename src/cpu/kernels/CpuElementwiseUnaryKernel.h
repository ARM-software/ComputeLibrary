/*
 * Copyright (c) 2018-2023 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_ELEMENTWISE_UNARY_KERNEL_H
#define ARM_COMPUTE_CPU_ELEMENTWISE_UNARY_KERNEL_H

#include "arm_compute/core/Types.h"
#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Interface for an element-wise unary operation kernel
 *
 * Element-wise operation is computed by:
 * @f[ dst(x) = OP(src(x))@f]
 */
class CpuElementwiseUnaryKernel : public ICpuKernel<CpuElementwiseUnaryKernel>
{
private:
    using ElementwiseUnaryUkernelPtr = std::add_pointer<void(const ITensor *, ITensor *, const Window &, ElementWiseUnary, const uint8_t *)>::type;
    using ElementwiseUnaryPreparePtr = std::add_pointer<std::unique_ptr<uint8_t[]>(ElementWiseUnary op, const ITensorInfo *, const ITensorInfo *)>::type;

public:
    CpuElementwiseUnaryKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuElementwiseUnaryKernel);

    /** Function to configure the @ref CpuElementwiseUnaryKernel
     *
     * @param[in]  op  Arithmetic operation to be executed.
     * @param[in]  src First tensor input. Data types supported: F16/F32, F16/F32/S32 for NEG/ABS operations.
     * @param[out] dst Output tensor. Data types supported: Same as @p src.
     */
    void configure(ElementWiseUnary op, const ITensorInfo &src, ITensorInfo &dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuElementwiseUnaryKernel::configure()
     *
     * @return a status
     */
    static Status validate(ElementWiseUnary op, const ITensorInfo &src, const ITensorInfo &dst);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

    struct ElementwiseUnaryKernel
    {
        const char                  *name;
        const DataTypeISASelectorPtr is_selected;
        ElementwiseUnaryUkernelPtr   ukernel;
        ElementwiseUnaryPreparePtr   prepare_func;
    };

    static const std::vector<ElementwiseUnaryKernel> &get_available_kernels();

private:
    ElementWiseUnary           _op{};
    ElementwiseUnaryUkernelPtr _run_method{ nullptr };
    std::string                _name{};
    std::unique_ptr<uint8_t[]> _lut{};
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_ELEMENTWISE_UNARY_KERNEL_H */
