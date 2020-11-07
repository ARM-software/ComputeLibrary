/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLGEMMMATRIXVECTORMULTIPLYKERNEL_H
#define ARM_COMPUTE_CLGEMMMATRIXVECTORMULTIPLYKERNEL_H

#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the GEMM matrix vector multiply kernel. **/
class CLGEMMMatrixVectorMultiplyKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLGEMMMatrixVectorMultiplyKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMMatrixVectorMultiplyKernel(const CLGEMMMatrixVectorMultiplyKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMMatrixVectorMultiplyKernel &operator=(const CLGEMMMatrixVectorMultiplyKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLGEMMMatrixVectorMultiplyKernel(CLGEMMMatrixVectorMultiplyKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLGEMMMatrixVectorMultiplyKernel &operator=(CLGEMMMatrixVectorMultiplyKernel &&) = default;
    /** Set the input and output of the kernel.
     *
     * @param[in]  input0 The reshaped input tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32
     * @param[in]  input1 The 2D reshaped weights tensor. Data type supported: Same as @p input.
     * @param[out] output The output 2D tensor. Data types supported: Same as @p input, S32 for QASYMM8/QASYMM8_SIGNED.
     */
    void configure(const ICLTensor *input0, const ICLTensor *input1, ICLTensor *output);
    /** Set the input and output of the kernel.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input0          The reshaped input tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32
     * @param[in]  input1          The 2D reshaped weights tensor. Data type supported: Same as @p input.
     * @param[out] output          The output 2D tensor. Data types supported: Same as @p input, S32 for QASYMM8/QASYMM8_SIGNED.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input0, const ICLTensor *input1, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMMatrixVectorMultiplyKernel
     *
     * @param[in] input0 The reshaped input tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32
     * @param[in] input1 The 2D reshaped weights tensor info. Data type supported: Same as @p input.
     * @param[in] output The output 2D tensor info. Data types supported: Same as @p input, S32 for QASYMM8/QASYMM8_SIGNED.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;
    BorderSize border_size() const override;

private:
    const ICLTensor *_input0;
    const ICLTensor *_input1;
    ICLTensor       *_output;
    int              _num_rows_read_per_iteration;
    BorderSize       _border_size;
};
} // arm_compute
#endif /*ARM_COMPUTE_CLGEMMMATRIXVECTORMULTIPLYKERNEL_H */
