/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_CPPUPSAMPLEKERNEL_H__
#define __ARM_COMPUTE_CPPUPSAMPLEKERNEL_H__

#include "arm_compute/core/CPP/ICPPKernel.h"

namespace arm_compute
{
class ITensor;

/** CPP kernel to perform tensor upsample.
 *
 */
class CPPUpsampleKernel : public ICPPKernel
{
public:
    const char *name() const override
    {
        return "CPPUpsampleKernel";
    }
    /** Default constructor */
    CPPUpsampleKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CPPUpsampleKernel(const CPPUpsampleKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CPPUpsampleKernel &operator=(const CPPUpsampleKernel &) = delete;
    /** Allow instances of this class to be moved */
    CPPUpsampleKernel(CPPUpsampleKernel &&) = default;
    /** Allow instances of this class to be moved */
    CPPUpsampleKernel &operator=(CPPUpsampleKernel &&) = default;
    /** Default destructor */
    ~CPPUpsampleKernel() = default;

    /** Set the input and output of the kernel.
     *
     * @param[in]  input  The input tensor to upsample. Data types supported: F32/F16/QASYMM8
     * @param[out] output The output tensor. Data types supported: Same as @p input
     * @param[in]  info   Padding info.
     */
    void configure(const ITensor *input, ITensor *output, const PadStrideInfo &info);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    bool is_parallelisable() const override;

private:
    const ITensor *_input;
    ITensor       *_output;
    PadStrideInfo  _info;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CPPUPSAMPLEKERNEL_H__ */
