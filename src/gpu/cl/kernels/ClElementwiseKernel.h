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
#ifndef ARM_COMPUTE_CL_ELEMENTWISE_KERNEL_H
#define ARM_COMPUTE_CL_ELEMENTWISE_KERNEL_H

#include "src/core/KernelTypes.h"
#include "src/core/common/Macros.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
/** Interface for an element-wise operation kernel
 *
 * Element-wise operation is computed by:
 * @f[ dst(x,y) = OP(src1(x,y), src2(x,y))@f]
 *
 * For binary elementwise ops in-place cannot be enabled by passing nullptr to dst, it can only be enabled by passing either src1 or src2 to dst instead.
 *
 */
class ClElementwiseKernel : public IClKernel
{
public:
    ClElementwiseKernel();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClElementwiseKernel);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, ::cl::CommandQueue &queue) override;

protected:
    /** The name of the operation */
    virtual std::string name() = 0;

    /** Configure kernel for a given list of arguments
     *
     * @param[in] src1 First source tensor info. Data types supported: U8/S8/QASYMM8/QASYMM8_SIGNED/U16/S16/F16/U32/S32/F32.
     * @param[in] src2 Second source tensor info. Data types supported: same as @p src1.
     * @param[in] dst  Destination tensor info. Data types supported: same as @p src1.
     *
     * @return a pair of Status and Window
     */
    virtual std::pair<Status, Window> validate_and_configure_window(ITensorInfo &src1, ITensorInfo &src2, ITensorInfo &dst) = 0;

    /** Generate the build options for the specific kernel
     *
     * @reutrn a CLBuildOptions struct
     */
    virtual CLBuildOptions generate_build_options(const ITensorInfo &src1, const ITensorInfo &src2, const ITensorInfo &dst) = 0;

    /** Generate the identifier for tuning
     *
     * @reutrn a string
     */
    virtual std::string generate_id_for_tuning(const std::string &kernel_name, const ITensorInfo &src1, const ITensorInfo &dst) = 0;

    /** Commmon configure function for element-wise operators with no additional options (e.g., Div, Min, Max, SquaredDiff)
     *
     */
    void configure_common(const ClCompileContext &compile_context, ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst);

    ActivationLayerInfo _act_info{};
};

class ClLogicalBinaryKernel : public ClElementwiseKernel
{
public:
    ClLogicalBinaryKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClLogicalBinaryKernel);
    /** Function to configure kernel
     *
     * @param[in] compile_context The compile context to be used.
     * @param[in] op              Logical binary operation to be executed.
     * @param[in] src1            First source tensor info. Data types supported: U8.
     * @param[in] src2            Second source tensor info. Data types supported: same as @p src1.
     * @param[in] dst             Destination tensor info. Data types supported: same as @p src1.
     */
    void configure(const ClCompileContext &compile_context, LogicalOperation op, ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClLogicalBinaryKernel::configure()
     *
     * @return a status
     */
    static Status validate(LogicalOperation op, const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst);

private:
    // Inherited methods overridden:
    std::string name() override;
    std::pair<Status, Window> validate_and_configure_window(ITensorInfo &src1, ITensorInfo &src2, ITensorInfo &dst) override;
    CLBuildOptions generate_build_options(const ITensorInfo &src1, const ITensorInfo &src2, const ITensorInfo &dst) override;
    std::string generate_id_for_tuning(const std::string &kernel_name, const ITensorInfo &src1, const ITensorInfo &dst) override;

    LogicalOperation _op{ LogicalOperation::Unknown };
};

/** Addition operation */
class ClSaturatedArithmeticKernel : public ClElementwiseKernel
{
public:
    ClSaturatedArithmeticKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClSaturatedArithmeticKernel);
    /** Static function to check if given info will lead to a valid configuration of @ref ClSaturatedArithmeticKernel
     *
     * @param[in] compile_context The compile context to be used.
     * @param[in] op              Arithmetic operation to be executed.
     * @param[in] input1          First tensor input info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32.
     * @param[in] input2          Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output          Output tensor info. Data types supported: Same as @p input1.
     * @param[in] policy          Policy to use to handle overflow.
     * @param[in] act_info        (Optional) Activation layer information in case of a fused activation.
     */
    void configure(const ClCompileContext &compile_context, ArithmeticOperation op, ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output, const ConvertPolicy &policy,
                   const ActivationLayerInfo &act_info = ActivationLayerInfo());

    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClSaturatedArithmeticKernel::configure()
     *
     * @return a status
     */
    static Status validate(ArithmeticOperation op, const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ConvertPolicy &policy,
                           const ActivationLayerInfo &act_info = ActivationLayerInfo());

protected:
    // Inherited methods overridden:
    std::string name() override;
    std::pair<Status, Window> validate_and_configure_window(ITensorInfo &input1, ITensorInfo &input2, ITensorInfo &output) override;
    CLBuildOptions generate_build_options(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output) override;
    std::string generate_id_for_tuning(const std::string &kernel_name, const ITensorInfo &input1, const ITensorInfo &output) override;

private:
    ConvertPolicy       _policy{};
    ArithmeticOperation _op{};
};

class ClArithmeticKernel : public ClElementwiseKernel
{
public:
    ClArithmeticKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClArithmeticKernel);

    /** Static function to check if given info will lead to a valid configuration of @ref ClArithmeticKernel
     *
     * @param[in] compile_context The compile context to be used.
     * @param[in] op              Arithmetic operation to be executed.
     * @param[in] src1            First source tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32.
     * @param[in] src2            Second source tensor info. Data types supported: same as @p src1.
     * @param[in] dst             Destination tensor info. Data types supported: same as @p src1.
     * @param[in] act_info        (Optional) Activation layer information in case of a fused activation.
     */
    void configure(const ClCompileContext &compile_context, ArithmeticOperation op, ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst,
                   const ActivationLayerInfo &act_info = ActivationLayerInfo());

    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClArithmeticKernel::configure()
     *
     * @return a status
     */
    static Status validate(ArithmeticOperation op, const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, const ActivationLayerInfo &act_info = ActivationLayerInfo());

protected:
    // Inherited methods overridden:
    std::string name() override;
    std::pair<Status, Window> validate_and_configure_window(ITensorInfo &src1, ITensorInfo &src2, ITensorInfo &dst) override;
    CLBuildOptions generate_build_options(const ITensorInfo &src1, const ITensorInfo &src2, const ITensorInfo &dst) override;
    std::string generate_id_for_tuning(const std::string &kernel_name, const ITensorInfo &src1, const ITensorInfo &dst) override;

private:
    ArithmeticOperation _op{};
};
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_ELEMENTWISE_KERNEL_H */
