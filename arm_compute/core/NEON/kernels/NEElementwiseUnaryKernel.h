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
#ifndef __ARM_COMPUTE_NEELEMENTWISEUNARYKERNEL_H__
#define __ARM_COMPUTE_NEELEMENTWISEUNARYKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ITensor;

/** Interface for an element-wise unary operation kernel
 *
 * Element-wise operation is computed by:
 * @f[ output(x) = OP(input(x))@f]
 *
 */
class NEElementwiseUnaryKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEElementwiseUnaryKernel";
    }
    /** Default constructor */
    NEElementwiseUnaryKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEElementwiseUnaryKernel(const NEElementwiseUnaryKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEElementwiseUnaryKernel &operator=(const NEElementwiseUnaryKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEElementwiseUnaryKernel(NEElementwiseUnaryKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEElementwiseUnaryKernel &operator=(NEElementwiseUnaryKernel &&) = default;
    /** Default destructor */
    ~NEElementwiseUnaryKernel() = default;

    /** Static function to check if given info will lead to a valid configuration of @ref NEElementwiseUnaryKernel
     *
     * @param[in] op     Arithmetic operation to be executed.
     * @param[in] input  First tensor input. Data types supported: F16/F32.
     * @param[in] output Output tensor. Data types supported: Same as @p input.
     */
    void configure(ElementWiseUnary op, const ITensor *input, ITensor *output);

    /** Static function to check if given info will lead to a valid configuration of @ref NEElementwiseUnaryKernel
     *
     * @param[in] op     Arithmetic operation to be executed.
     * @param[in] input  First tensor input info. Data types supported: F16/F32.
     * @param[in] output Output tensor info. Data types supported: Same as @p input.
     *
     * @return a Status
     */
    static Status validate(ElementWiseUnary op, const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

    /** Common signature for all the specialised arithmetic functions
     *
     * @param[in]  input  An input tensor. Data types supported: F16/F32.
     * @param[out] output The output tensor. Data types supported: F16/F32.
     * @param[in]  window Region on which to execute the kernel.
     */
    using ElementwiseUnaryFunction = void(const ITensor *input, ITensor *output, const Window &window);

protected:
    // Inherited methods overridden:
    static Status validate_arguments(ElementWiseUnary op, const ITensorInfo &input, const ITensorInfo &output);

    /** Function to use for the particular tensor types passed to configure() */
    std::function<void(const ITensor *input, ITensor *output, const Window &window)> _function;

    const ITensor *_input;
    ITensor       *_output;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEELEMENTWISEUNARYKERNEL_H__ */
