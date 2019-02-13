/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLELEMENTWISEOPERATIONKERNEL_H__
#define __ARM_COMPUTE_CLELEMENTWISEOPERATIONKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/Types.h"

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
    void run(const Window &window, cl::CommandQueue &queue) override;

    BorderSize border_size() const override;

protected:
    /** The name of the operation */
    virtual std::string name() = 0;

    /** Initialise the kernel's output.
     *
     * @param[in] input1 First tensor input. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[in] input2 Second tensor input. Data types supported: Same as @p input1.
     * @param[in] output Output tensor. Data types supported: Same as @p input1.
     *
     * @return a pair of Status and Window
     */
    virtual std::pair<Status, Window> validate_and_configure_window(ITensorInfo &input1, ITensorInfo &input2, ITensorInfo &output) = 0;

    /** Validate the argument passed to the kernel
     *
     * @param[in] input1 First tensor input. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[in] input2 Second tensor input. Data types supported: Same as @p input1.
     * @param[in] output Output tensor. Data types supported: Same as @p input1.
     */
    virtual Status validate_arguments(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output) = 0;

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
    void configure_common(const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output);

private:
    const ICLTensor *_input1; /**< Source tensor 1 */
    const ICLTensor *_input2; /**< Source tensor 2 */
    ICLTensor       *_output; /**< Destination tensor */
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
     * @param[in] op     Arithmetic operation to be executed.
     * @param[in] input1 First tensor input. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[in] input2 Second tensor input. Data types supported: Same as @p input1.
     * @param[in] output Output tensor. Data types supported: Same as @p input1.
     * @param[in] policy Policy to use to handle overflow.
     */
    void configure(ArithmeticOperation op, const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, const ConvertPolicy &policy);

    /** Static function to check if given info will lead to a valid configuration of @ref CLSaturatedArithmeticOperationKernel
     *
     * @param[in] op     Arithmetic operation to be executed.
     * @param[in] input1 First tensor input info. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[in] input2 Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output Output tensor info. Data types supported: Same as @p input1.
     * @param[in] policy Policy to use to handle overflow.
     *
     * @return a Status
     */
    static Status validate(ArithmeticOperation op, const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ConvertPolicy &policy);

protected:
    // Inherited methods overridden:
    std::string name() override;
    std::pair<Status, Window> validate_and_configure_window(ITensorInfo &input1, ITensorInfo &input2, ITensorInfo &output) override;
    Status validate_arguments(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output) override;
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
     * @param[in] op     Arithmetic operation to be executed.
     * @param[in] input1 First tensor input. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[in] input2 Second tensor input. Data types supported: Same as @p input1.
     * @param[in] output Output tensor. Data types supported: Same as @p input1.
     */
    void configure(ArithmeticOperation op, const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output);

    /** Static function to check if given info will lead to a valid configuration of @ref CLArithmeticOperationKernel
     *
     * @param[in] op     Arithmetic operation to be executed.
     * @param[in] input1 First tensor input info. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[in] input2 Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output Output tensor info. Data types supported: Same as @p input1.
     *
     * @return a Status
     */
    static Status validate(ArithmeticOperation op, const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);

protected:
    // Inherited methods overridden:
    std::string name() override;
    std::pair<Status, Window> validate_and_configure_window(ITensorInfo &input1, ITensorInfo &input2, ITensorInfo &output) override;
    Status validate_arguments(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output) override;
    CLBuildOptions generate_build_options(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output) override;
    std::string generate_id_for_tuning(const std::string &kernel_name, const ITensorInfo &input1, const ITensorInfo &output) override;

private:
    ArithmeticOperation _op;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLELEMENTWISEOPERATIONKERNEL_H__ */
