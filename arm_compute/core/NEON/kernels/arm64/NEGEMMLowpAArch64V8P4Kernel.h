/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEGEMMLOWPAARCH64V8P4KERNEL_H__
#define __ARM_COMPUTE_NEGEMMLOWPAARCH64V8P4KERNEL_H__

#include "arm_compute/core/NEON/kernels/NEGEMMAssemblyBaseKernel.h"

// Enable only if compiled for AArch64-V8.2-A targets
#ifdef ARM_COMPUTE_AARCH64_V8_2

namespace arm_compute
{
class ITensor;

/** AArch64 NEON kernel to multiply two input matrices "A" and "B". */
class NEGEMMLowpAArch64V8P4Kernel : public NEGEMMAssemblyBaseKernel
{
public:
    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMAssemblyBaseKernel
     *
     * The computed function is C = a * AxB + b * C.
     *
     * @param[in] input0 Input tensor info containing the Matrix A. Data types supported: QASYMM8
     * @param[in] input1 Input tensor info containing the Matrix B. Data types supported: same as @p input0
     * @param[in] output Output tensor info to store the result of matrix multiplication.
     *                        If @p beta is not zero the values are multiplied by @p beta before the result is accumulated. Otherwise the values are overwritten by the result. Data types supported: S32
     */
    static Status validate(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *output);

protected:
    void internal_configure(const ITensor *input0, const ITensor *input1, ITensor *output, ITensor *workspace, float alpha, float beta, bool is_transposed_0, bool is_transposed_1) override;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_AARCH64_V8_2 */
#endif /*__ARM_COMPUTE_NEGEMMLOWPAARCH64V8P4KERNEL_H__*/
