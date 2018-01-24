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
#ifndef __ARM_COMPUTE_NEGEMMASSEMBLYBASE_H__
#define __ARM_COMPUTE_NEGEMMASSEMBLYBASE_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Base class for GEMM NEON kernels implemented in Assembly. */
class NEGEMMAssemblyBaseKernel : public INEKernel
{
public:
    /** Constructor */
    NEGEMMAssemblyBaseKernel()
        : _input0(nullptr), _input1(nullptr), _output(nullptr), _workspace(nullptr), _alpha(1.f), _beta(0.f), _is_transposed_0(false), _is_transposed_1(false)
    {
    }

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMAssemblyBaseKernel(const NEGEMMAssemblyBaseKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMAssemblyBaseKernel &operator=(const NEGEMMAssemblyBaseKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEGEMMAssemblyBaseKernel(NEGEMMAssemblyBaseKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEGEMMAssemblyBaseKernel &operator=(NEGEMMAssemblyBaseKernel &&) = default;

    virtual ~NEGEMMAssemblyBaseKernel() = default;

    /** Initialise the kernel's input and output.
     *
     * The computed function is C = a * AxB + b * C.
     *
     * @param[in]     input0          Input tensor containing the Matrix A. Data types supported: F32
     * @param[in]     input1          Input tensor containing the Matrix B. Data types supported: same as @p input0
     * @param[in,out] output          Output tensor to store the result of matrix multiplication. If @p beta is not zero the values are multiplied by @p beta before the result is accumulated. Otherwise the values are overwritten by the result. Data types supported: same as @p input0.
     * @param[out]    workspace       Space for intermediate results.
     * @param[in]     alpha           Weight of the matrix product
     * @param[in]     beta            Weight of the accumulation.
     * @param[in]     is_transposed_0 (Optional)True if @p input0 is transposed else false. (Defaults to false)
     * @param[in]     is_transposed_1 (Optional)True if @p input1 is transposed else false. (Defaults to false)
     */
    void configure(const ITensor *input0, const ITensor *input1, ITensor *output, ITensor *workspace, float alpha = 1.f, float beta = 0.f, bool is_transposed_0 = false, bool is_transposed_1 = false)
    {
        internal_configure(input0, input1, output, workspace, alpha, beta, is_transposed_0, is_transposed_1);
    }

protected:
    virtual void internal_configure(const ITensor *input0, const ITensor *input1, ITensor *output, ITensor *workspace, float alpha, float beta, bool _is_transposed_0, bool _is_transposed_1) = 0;

    const ITensor *_input0;
    const ITensor *_input1;
    ITensor       *_output;
    ITensor       *_workspace;
    float          _alpha;
    float          _beta;
    bool           _is_transposed_0;
    bool           _is_transposed_1;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEGEMMASSEMBLYBASE_H__*/
