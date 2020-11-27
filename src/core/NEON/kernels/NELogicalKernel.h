/*
 * Copyright (c) 2020 Arm Limited.
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
#ifndef ARM_COMPUTE_NELOGICALKERNEL_H
#define ARM_COMPUTE_NELOGICALKERNEL_H

#include "src/core/KernelTypes.h"
#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
namespace kernels
{
/** Interface for the kernel to perform logical operations between two tensors
 *
 * Supported logical operations:
 *  - AND
 *  - OR
 *  - NOT
 */
class NELogicalKernel : public INEKernel
{
public:
    /** Initialise the kernel's inputs and output
     *
     * @param[in]  input1 An input tensor. Data type supported: U8.
     * @param[in]  input2 An input tensor. Data type supported: U8
     * @param[out] output Output tensor. Data type supported: U8.
     * @param[out] op     Logical operation to perform
     */
    void configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output, LogicalOperation op);
    /** Static function to check if given info will lead to a valid configuration of @ref NELogicalKernel
     *
     * @param[in] input1 An input tensor. Data type supported: U8.
     * @param[in] input2 An input tensor. Data type supported: U8
     * @param[in] output Output tensor. Data type supported: U8.
     * @param[in] op     Logical operation to perform
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, LogicalOperation op);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

private:
    LogicalOperation _op{};
};
} // namespace kernels
} // namespace arm_compute
#endif /* ARM_COMPUTE_NELOGICALKERNEL_H */
