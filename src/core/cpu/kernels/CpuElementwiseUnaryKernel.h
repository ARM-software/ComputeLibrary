/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "src/core/cpu/ICpuKernel.h"

namespace arm_compute
{
class ITensor;
namespace cpu
{
namespace kernels
{
/** Interface for an element-wise unary operation kernel
 *
 * Element-wise operation is computed by:
 * @f[ output(x) = OP(input(x))@f]
 *
 */
class CpuElementwiseUnaryKernel : public ICpuKernel
{
public:
    const char *name() const override
    {
        return "CpuElementwiseUnaryKernel";
    }
    /** Default constructor */
    CpuElementwiseUnaryKernel();
    /** Default destructor */
    ~CpuElementwiseUnaryKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuElementwiseUnaryKernel);

    /** Function to configure the @ref CpuElementwiseUnaryKernel
     *
     * @param[in]  op     Arithmetic operation to be executed.
     * @param[in]  input  First tensor input. Data types supported: F16/F32, F16/F32/S32 for NEG/ABS operations.
     * @param[out] output Output tensor. Data types supported: Same as @p input.
     */
    void configure(ElementWiseUnary op, const ITensorInfo &input, ITensorInfo &output);

    /** Static function to check if given info will lead to a valid configuration of @ref CpuElementwiseUnaryKernel
     *
     * @param[in] op     Arithmetic operation to be executed.
     * @param[in] input  First tensor input info. Data types supported: F16/F32, F16/F32/S32 for NEG/ABS operations.
     * @param[in] output Output tensor info. Data types supported: Same as @p input.
     *
     * @return a Status
     */
    static Status validate(ElementWiseUnary op, const ITensorInfo &input, const ITensorInfo &output);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;

    /** Common signature for all the specialised elementwise unary micro-kernels
     *
     * @param[in] window Region on which to execute the kernel.
     */
    using ElementwiseUnaryUkernelPtr = std::add_pointer<void(const ITensor *, ITensor *, const Window &, ElementWiseUnary)>::type;

private:
    ElementWiseUnary _op;
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_ELEMENTWISE_UNARY_KERNEL_H */
