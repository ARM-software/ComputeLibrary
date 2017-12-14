/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLARITHMETICADDITIONKERNEL_H__
#define __ARM_COMPUTE_CLARITHMETICADDITIONKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the arithmetic addition kernel
 *
 * Arithmetic addition is computed by:
 * @f[ output(x,y) = input1(x,y) + input2(x,y) @f]
 */
class CLArithmeticAdditionKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLArithmeticAdditionKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLArithmeticAdditionKernel(const CLArithmeticAdditionKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLArithmeticAdditionKernel &operator=(const CLArithmeticAdditionKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLArithmeticAdditionKernel(CLArithmeticAdditionKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLArithmeticAdditionKernel &operator=(CLArithmeticAdditionKernel &&) = default;
    /** Default destructor */
    ~CLArithmeticAdditionKernel() = default;
    /** Initialise the kernel's inputs, output and convertion policy.
     *
     * @param[in]  input1 First tensor input. Data types supported: U8/QS8/QS16/S16/F16/F32.
     * @param[in]  input2 Second tensor input. Data types supported: U8/QS8 (only if @p input1 is QS8), QS16 (only if @p input1 is QS16), S16/F16/F32.
     * @param[out] output Output tensor. Data types supported: U8 (Only if both inputs are U8), QS8 (only if both inputs are QS8), QS16 (only if both inputs are QS16), S16/F16/F32.
     * @param[in]  policy Policy to use to handle overflow.
     */
    void configure(const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, ConvertPolicy policy);
    /** Static function to check if given info will lead to a valid configuration of @ref CLArithmeticAdditionKernel
     *
     * @param[in] input1 First tensor input info. Data types supported: U8/QS8/QS16/S16/F16/F32.
     * @param[in] input2 Second tensor input info. Data types supported: U8/QS8 (only if @p input1 is QS8), QS16 (only if @p input1 is QS16), S16/F16/F32.
     * @param[in] output Output tensor info. Data types supported: U8 (Only if both inputs are U8), QS8 (only if both inputs are QS8), QS16 (only if both inputs are QS16), S16/F16/F32.
     * @param[in] policy Policy to use to handle overflow.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input1; /**< Source tensor 1 */
    const ICLTensor *_input2; /**< Source tensor 2 */
    ICLTensor       *_output; /**< Destination tensor */
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLARITHMETICADDITIONKERNEL_H__ */
