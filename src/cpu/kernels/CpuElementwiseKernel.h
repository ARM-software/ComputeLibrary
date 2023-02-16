/*
 * Copyright (c) 2021-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_ELEMENTWISE_KERNEL_H
#define ARM_COMPUTE_CPU_ELEMENTWISE_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Interface for an element-wise operation kernel
 *
 * Element-wise operation is computed by:
 * @f[ dst(x,y) = OP(src0(x,y), src1(x,y))@f]
 *
 */
template <class Derived>
class CpuElementwiseKernel : public ICpuKernel<Derived>
{
private:
    using ElementwiseKernelPtr = std::add_pointer<void(const ITensor *, const ITensor *, ITensor *, const Window &)>::type;

public:
    CpuElementwiseKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuElementwiseKernel);

    using ElementwiseFunction = void(const ITensor *, const ITensor *, ITensor *, const Window &);
    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;

    const char *name() const override;

    struct ElementwiseKernel
    {
        const char                             *name;
        const ElementwiseDataTypeISASelectorPtr is_selected;
        ElementwiseKernelPtr                    ukernel;
    };

protected:
    /** Validate the argument passed to the kernel
     *
     * @param[in] src0 First tensor input. Data types supported: QASYMM8/S16/F16/S32/F32.
     * @param[in] src1 Second tensor input. Data types supported: Same as @p src0.
     * @param[in] dst  Output tensor. Data types supported: Dependent on subclass.
     */
    static Status validate_arguments_common(const ITensorInfo &src0, const ITensorInfo &src1, const ITensorInfo &dst);

protected:
    ElementwiseKernelPtr _run_method{ nullptr };
    std::string          _name{};
};

class CpuArithmeticKernel : public CpuElementwiseKernel<CpuArithmeticKernel>
{
public:
    CpuArithmeticKernel() = default;

    /** Configure kernel
     *
     * @param[in]  op   Arithmetic operation to be executed.
     * @param[in]  src0 First tensor input info. Data types supported: QASYMM8/S16/F16/S32/F32.
     * @param[in]  src1 Second tensor input info. Data types supported: Same as @p src0.
     * @param[out] dst  Output tensor info. Data types supported: Same as @p src0.
     */
    void configure(ArithmeticOperation op, const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst);

    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuArithmeticKernel::configure()
     *
     * @return a status
     */
    static Status validate(ArithmeticOperation op, const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst);

    static const std::vector<CpuElementwiseKernel<CpuArithmeticKernel>::ElementwiseKernel> &get_available_kernels();

    /** Return minimum workload size of the relevant kernel
     *
     * @param[in] platform     The CPU platform used to create the context.
     * @param[in] thread_count Number of threads in the execution.
     *
     * @return[out] mws Minimum workload size for requested configuration.
     */
    size_t get_mws(const CPUInfo &platform, size_t thread_count) const override;

protected:
    /** Commmon configure function for element-wise operators with no additional options (e.g. Min, Max, SquaredDiff)
     */
    void configure_common(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst);
    // Inherited methods overridden:
    static Status validate_arguments(const ITensorInfo &src0, const ITensorInfo &src1, const ITensorInfo &dst);

    ArithmeticOperation _op{};
};

class CpuDivisionKernel : public CpuArithmeticKernel
{
public:
    CpuDivisionKernel() = default;

    /** Configure kernel
     *
     * @param[in]  src0 First tensor input info. Data types supported: S32/F16/F32.
     * @param[in]  src1 Second tensor input info. Data types supported: Same as @p src0.
     * @param[out] dst  Output tensor info. Data types supported: Same as @p src0.
     */
    void configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst);

    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuDivisionKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst);

    /** Return minimum workload size of the relevant kernel
     *
     * @param[in] platform     The CPU platform used to create the context.
     * @param[in] thread_count Number of threads in the execution.
     *
     * @return[out] mws Minimum workload size for requested configuration.
     */
    size_t get_mws(const CPUInfo &platform, size_t thread_count) const override;

protected:
    // Inherited methods overridden:
    static Status validate_arguments(const ITensorInfo &src0, const ITensorInfo &src1, const ITensorInfo &dst);
};

class CpuPowerKernel : public CpuArithmeticKernel
{
public:
    CpuPowerKernel() = default;

    /** Configure kernel
     *
     * @param[in]  src0 First tensor input info. Data types supported: F16/F32.
     * @param[in]  src1 Second tensor input info. Data types supported: Same as @p src0.
     * @param[out] dst  Output tensor info. Data types supported: Same as @p src0.
     */
    void configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst);

    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuPowerKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst);

protected:
    // Inherited methods overridden:
    static Status validate_arguments(const ITensorInfo &src0, const ITensorInfo &src1, const ITensorInfo &dst);
};

class CpuComparisonKernel : public CpuElementwiseKernel<CpuComparisonKernel>
{
public:
    CpuComparisonKernel() = default;

    /** Configure kernel
     *
     * @param[in]  op   Comparison operation to be executed.
     * @param[in]  src0 First tensor input info. Data types supported: QASYMM8/QASYMM8_SIGNED/S16/F16/S32/F32.
     * @param[in]  src1 Second tensor input info. Data types supported: Same as @p src0.
     * @param[out] dst  Output tensor info. Data types supported: U8.
     */
    void configure(ComparisonOperation op, const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst);

    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuComparisonKernel::configure()
     *
     * @return a status
     */
    static Status validate(ComparisonOperation op, const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst);

    static const std::vector<CpuElementwiseKernel<CpuComparisonKernel>::ElementwiseKernel> &get_available_kernels();

protected:
    /** Commmon configure function for element-wise operators with no additional options (e.g. Min, Max, SquaredDiff)
     */
    void configure_common(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst);
    // Inherited methods overridden:
    static Status validate_arguments(const ITensorInfo &src0, const ITensorInfo &src1, const ITensorInfo &dst);

private:
    /** Function to get the micro kernel implementation
     *
     * @param[in] src0 First input tensor information
     * @param[in] src1 Second input tensor information
     * @param[in] dst  Output tensor information
     *
     * @return the function instance for the micro kernel
     */

    ComparisonOperation _op{};
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_ELEMENTWISE_KERNEL_H */