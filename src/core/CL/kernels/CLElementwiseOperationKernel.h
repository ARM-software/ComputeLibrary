/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLELEMENTWISEOPERATIONKERNEL_H
#define ARM_COMPUTE_CLELEMENTWISEOPERATIONKERNEL_H

#include "arm_compute/core/Types.h"
#include "src/core/CL/ICLKernel.h"
#include "src/core/KernelTypes.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for an element-wise operation kernel
 *
 * Element-wise operation is computed by:
 * @f[ output(x,y) = OP(input1(x,y), input2(x,y))@f]
 *
 */
class CLElementwiseOperationKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLElementwiseOperationKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLElementwiseOperationKernel(const CLElementwiseOperationKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLElementwiseOperationKernel &operator=(const CLElementwiseOperationKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLElementwiseOperationKernel(CLElementwiseOperationKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLElementwiseOperationKernel &operator=(CLElementwiseOperationKernel &&) = default;
    /** Default destructor */
    ~CLElementwiseOperationKernel() = default;

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override;

protected:
    /** The name of the operation */
    virtual std::string name() = 0;

    /** Initialise the kernel's output.
     *
     * @param[in] input1 First tensor input info. Data types supported: U8/S8/QASYMM8/QASYMM8_SIGNED/U16/S16/F16/U32/S32/F32.
     * @param[in] input2 Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output Output tensor info. Data types supported: Same as @p input1.
     *
     * @return a pair of Status and Window
     */
    virtual std::pair<Status, Window> validate_and_configure_window(ITensorInfo &input1, ITensorInfo &input2, ITensorInfo &output) = 0;

    /** Generate the build options for the specific kernel
     *
     * @reutrn a CLBuildOptions struct
     */
    virtual CLBuildOptions generate_build_options(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output) = 0;

    /** Generate the identifier for tuning
     *
     * @reutrn a string
     */
    virtual std::string generate_id_for_tuning(const std::string &kernel_name, const ITensorInfo &input1, const ITensorInfo &output) = 0;

    /** Commmon configure function for element-wise operators with no additional options (e.g., Div, Min, Max, SquaredDiff)
     *
     */
    void configure_common(ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output);
    /** Commmon configure function for element-wise operators with no additional options (e.g., Div, Min, Max, SquaredDiff)
     *
     */
    void configure_common(const CLCompileContext &compile_context, ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output);

    ActivationLayerInfo _act_info;

private:
    const ITensorInfo *_input1; /**< Source tensor info 1 */
    const ITensorInfo *_input2; /**< Source tensor info 2 */
    ITensorInfo       *_output; /**< Destination tensor info */
};

class CLLogicalBinaryKernel : public CLElementwiseOperationKernel
{
public:
    /** Default constructor */
    CLLogicalBinaryKernel() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLLogicalBinaryKernel(const CLLogicalBinaryKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLLogicalBinaryKernel &operator=(const CLLogicalBinaryKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLLogicalBinaryKernel(CLLogicalBinaryKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLLogicalBinaryKernel &operator=(CLLogicalBinaryKernel &&) = default;
    /** Default destructor */
    ~CLLogicalBinaryKernel() = default;
    /** Function to configure kernel
     *
     * @param[in] compile_context The compile context to be used.
     * @param[in] op              Logical binary operation to be executed.
     * @param[in] input1          First tensor input info. Data types supported: U8.
     * @param[in] input2          Second tensor input info. Data types supported: U8.
     * @param[in] output          Output tensor info. Data types supported: U8.
     */
    void configure(const CLCompileContext &compile_context, kernels::LogicalOperation op, ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output);
    /** Static function to check if the given configuration is valid for this kernel
     *
     * @param[in] op     Logical binary operation to be executed.
     * @param[in] input1 First tensor input info. Data types supported: U8.
     * @param[in] input2 Second tensor input info. Data types supported: U8.
     * @param[in] output Output tensor info. Data types supported: U8.
     */
    static Status validate(kernels::LogicalOperation op, const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);

private:
    // Inherited methods overridden:
    std::string name() override;
    std::pair<Status, Window> validate_and_configure_window(ITensorInfo &input1, ITensorInfo &input2, ITensorInfo &output) override;
    CLBuildOptions generate_build_options(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output) override;
    std::string generate_id_for_tuning(const std::string &kernel_name, const ITensorInfo &input1, const ITensorInfo &output) override;

    kernels::LogicalOperation _op{ kernels::LogicalOperation::Unknown };
};

/** Addition operation */
class CLSaturatedArithmeticOperationKernel : public CLElementwiseOperationKernel
{
public:
    CLSaturatedArithmeticOperationKernel()
        : CLElementwiseOperationKernel(), _policy(), _op()
    {
    }

    /** Static function to check if given info will lead to a valid configuration of @ref CLSaturatedArithmeticOperationKernel
     *
     * @param[in] op       Arithmetic operation to be executed.
     * @param[in] input1   First tensor input info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32.
     * @param[in] input2   Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output   Output tensor info. Data types supported: Same as @p input1.
     * @param[in] policy   Policy to use to handle overflow.
     * @param[in] act_info (Optional) Activation layer information in case of a fused activation.
     */
    void configure(ArithmeticOperation op, ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output, const ConvertPolicy &policy, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CLSaturatedArithmeticOperationKernel
     *
     * @param[in] compile_context The compile context to be used.
     * @param[in] op              Arithmetic operation to be executed.
     * @param[in] input1          First tensor input info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32.
     * @param[in] input2          Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output          Output tensor info. Data types supported: Same as @p input1.
     * @param[in] policy          Policy to use to handle overflow.
     * @param[in] act_info        (Optional) Activation layer information in case of a fused activation.
     */
    void configure(const CLCompileContext &compile_context, ArithmeticOperation op, ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output, const ConvertPolicy &policy,
                   const ActivationLayerInfo &act_info = ActivationLayerInfo());

    /** Static function to check if given info will lead to a valid configuration of @ref CLSaturatedArithmeticOperationKernel
     *
     * @param[in] op       Arithmetic operation to be executed.
     * @param[in] input1   First tensor input info info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32.
     * @param[in] input2   Second tensor input info info. Data types supported: Same as @p input1.
     * @param[in] output   Output tensor info info. Data types supported: Same as @p input1.
     * @param[in] policy   Policy to use to handle overflow.
     * @param[in] act_info (Optional) Activation layer information in case of a fused activation.
     *
     * @return a Status
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
    ConvertPolicy       _policy;
    ArithmeticOperation _op;
};

class CLArithmeticOperationKernel : public CLElementwiseOperationKernel
{
public:
    CLArithmeticOperationKernel()
        : CLElementwiseOperationKernel(), _op()
    {
    }

    /** Static function to check if given info will lead to a valid configuration of @ref CLArithmeticOperationKernel
     *
     * @param[in] op       Arithmetic operation to be executed.
     * @param[in] input1   First tensor input info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32.
     * @param[in] input2   Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output   Output tensor info. Data types supported: Same as @p input1.
     * @param[in] act_info (Optional) Activation layer information in case of a fused activation.
     */
    void configure(ArithmeticOperation op, ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CLArithmeticOperationKernel
     *
     * @param[in] compile_context The compile context to be used.
     * @param[in] op              Arithmetic operation to be executed.
     * @param[in] input1          First tensor input info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32.
     * @param[in] input2          Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output          Output tensor info. Data types supported: Same as @p input1.
     * @param[in] act_info        (Optional) Activation layer information in case of a fused activation.
     */
    void configure(const CLCompileContext &compile_context, ArithmeticOperation op, ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output,
                   const ActivationLayerInfo &act_info = ActivationLayerInfo());

    /** Static function to check if given info will lead to a valid configuration of @ref CLArithmeticOperationKernel
     *
     * @param[in] op       Arithmetic operation to be executed.
     * @param[in] input1   First tensor input info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32.
     * @param[in] input2   Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output   Output tensor info. Data types supported: Same as @p input1.
     * @param[in] act_info (Optional) Activation layer information in case of a fused activation.
     *
     * @return a Status
     */
    static Status validate(ArithmeticOperation op, const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());

protected:
    // Inherited methods overridden:
    std::string name() override;
    std::pair<Status, Window> validate_and_configure_window(ITensorInfo &input1, ITensorInfo &input2, ITensorInfo &output) override;
    CLBuildOptions generate_build_options(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output) override;
    std::string generate_id_for_tuning(const std::string &kernel_name, const ITensorInfo &input1, const ITensorInfo &output) override;

private:
    ArithmeticOperation _op;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLELEMENTWISEOPERATIONKERNEL_H */
