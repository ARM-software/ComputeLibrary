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
#ifndef __ARM_COMPUTE_GCARITHMETICADDITIONKERNEL_H__
#define __ARM_COMPUTE_GCARITHMETICADDITIONKERNEL_H__

#include "arm_compute/core/GLES_COMPUTE/IGCKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class IGCTensor;

/** Interface for the arithmetic addition kernel
 *
 * Arithmetic addition is computed by:
 * @f[ output(x,y) = input1(x,y) + input2(x,y) @f]
 */
class GCArithmeticAdditionKernel : public IGCKernel
{
public:
    /** Default constructor */
    GCArithmeticAdditionKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCArithmeticAdditionKernel(const GCArithmeticAdditionKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCArithmeticAdditionKernel &operator=(const GCArithmeticAdditionKernel &) = delete;
    /** Allow instances of this class to be moved */
    GCArithmeticAdditionKernel(GCArithmeticAdditionKernel &&) = default;
    /** Allow instances of this class to be moved */
    GCArithmeticAdditionKernel &operator=(GCArithmeticAdditionKernel &&) = default;
    /** Default destructor */
    ~GCArithmeticAdditionKernel() = default;
    /** Initialise the kernel's inputs, output and convertion policy.
     *
     * @param[in]  input1 First tensor input. Data types supported: F16.
     * @param[in]  input2 Second tensor input. Data types supported: F16.
     * @param[out] output Output tensor. Data types supported: F16.
     * @param[in]  policy Policy to use to handle overflow.
     */
    void configure(const IGCTensor *input1, const IGCTensor *input2, IGCTensor *output, ConvertPolicy policy);
    /** Static function to check if given info will lead to a valid configuration of @ref GCArithmeticAdditionKernel
     *
     * @param[in]  input1 First tensor input info. Data types supported: F16.
     * @param[in]  input2 Second tensor input info. Data types supported: F16.
     * @param[out] output Output tensor info. Data types supported: F16.
     * @param[in]  policy Policy to use to handle overflow.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy);

    // Inherited methods overridden:
    void run(const Window &window) override;

private:
    const IGCTensor *_input1; /**< Source tensor 1 */
    const IGCTensor *_input2; /**< Source tensor 2 */
    IGCTensor       *_output; /**< Destination tensor */
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GCARITHMETICADDITIONKERNEL_H__ */
