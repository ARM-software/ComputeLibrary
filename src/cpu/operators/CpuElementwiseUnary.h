/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_ELEMENTWISE_UNARY_H
#define ARM_COMPUTE_CPU_ELEMENTWISE_UNARY_H

#include "arm_compute/core/Types.h"

#include "src/cpu/ICpuOperator.h"

namespace arm_compute
{
namespace cpu
{
class CpuElementwiseUnary : public ICpuOperator
{
public:
    /** Initialize the function
     *
     * @param[in]  op  Unary operation to execute
     * @param[in]  src Input tensor information. Data types supported: F16/F32, F16/F32/S32 for NEG/ABS operations.
     * @param[out] dst Output tensor information. Data types supported: Same as @p src.
     */
    void configure(ElementWiseUnary op, const ITensorInfo &src, ITensorInfo &dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuElementwiseUnary::configure()
     *
     * @return a status
     */
    static Status validate(ElementWiseUnary op, const ITensorInfo &src, const ITensorInfo &dst);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
};

} // namespace cpu
} // namespace arm_compute

#endif /* ARM_COMPUTE_CPU_ELEMENTWISE_UNARY_H */
