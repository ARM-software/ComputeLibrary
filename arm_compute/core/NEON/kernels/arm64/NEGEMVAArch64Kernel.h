/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEGEMVAARCH64KERNEL_H__
#define __ARM_COMPUTE_NEGEMVAARCH64KERNEL_H__

#include "arm_compute/core/NEON/kernels/NEGEMMAssemblyBaseKernel.h"

namespace arm_compute
{
class ITensor;

/** AArch64 NEON kernel to multiply an input vector "A" and a matrix "B". */
class NEGEMVAArch64Kernel : public NEGEMMAssemblyBaseKernel
{
public:
    const char *name() const override
    {
        return "NEGEMVAArch64Kernel";
    }
    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

protected:
    void internal_configure(const ITensor *input0, const ITensor *input1, ITensor *output, ITensor *workspace, float alpha, float beta, bool is_transposed_0, bool is_transposed_1) override;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEGEMVAARCH64KERNEL_H__*/
