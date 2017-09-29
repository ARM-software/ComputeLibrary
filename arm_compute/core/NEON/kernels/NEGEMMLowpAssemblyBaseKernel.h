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
#ifndef __ARM_COMPUTE_NEGEMMLOWPASSEMBLYBASE_H__
#define __ARM_COMPUTE_NEGEMMLOWPASSEMBLYBASE_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** GEMMLOWP AssemblyBase NEON kernel to multiply two input matrices "A" and "B". */
class NEGEMMLowpAssemblyBaseKernel : public INEKernel
{
public:
    /** Constructor */
    NEGEMMLowpAssemblyBaseKernel()
        : _input0(nullptr), _input1(nullptr), _output(nullptr), _workspace(nullptr), _transform_0(true), _transform_1(true)
    {
    }

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMLowpAssemblyBaseKernel(const NEGEMMLowpAssemblyBaseKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMLowpAssemblyBaseKernel &operator=(const NEGEMMLowpAssemblyBaseKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEGEMMLowpAssemblyBaseKernel(NEGEMMLowpAssemblyBaseKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEGEMMLowpAssemblyBaseKernel &operator=(NEGEMMLowpAssemblyBaseKernel &&) = default;

    virtual ~NEGEMMLowpAssemblyBaseKernel() = default;

    /** Initialise the kernel's input and output.
     *
     * The computed function is C = a * AxB + b * C.
     *
     * @param[in]     input0 Input tensor containing the Matrix A. Data types supported: F32
     * @param[in]     input1 Input tensor containing the Matrix B. Data types supported: same as @p input0
     * @param[in,out] output Output tensor to store the result of matrix multiplication. If @p beta is not zero the values are multiplied by @p beta before the result is accumulated. Otherwise the values are overwritten by the result. Data types supported: same as @p input0.
     */
    void configure(const ITensor *input0, const ITensor *input1, ITensor *output)
    {
        internal_configure(input0, input1, output);
    }

protected:
    virtual void internal_configure(const ITensor *input0, const ITensor *input1, ITensor *output) = 0;

    const ITensor *_input0;
    const ITensor *_input1;
    ITensor       *_output;
    ITensor       *_workspace;
    bool           _transform_0;
    bool           _transform_1;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEGEMMLOWPASSEMBLYBASE_H__*/
