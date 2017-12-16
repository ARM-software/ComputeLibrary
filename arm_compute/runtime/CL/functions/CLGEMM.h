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
#ifndef __ARM_COMPUTE_CLGEMM_H__
#define __ARM_COMPUTE_CLGEMM_H__

#include "arm_compute/core/CL/kernels/CLFillBorderKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMInterleave4x4Kernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMMatrixAdditionKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMMatrixMultiplyKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMTranspose1xWKernel.h"
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"

#include <memory>

namespace arm_compute
{
class ICLTensor;

/** Basic function to execute GEMM on OpenCL. This function calls the following OpenCL kernels:
 *
 *  -# @ref CLGEMMInterleave4x4Kernel (if the output tensor is a matrix)
 *  -# @ref CLGEMMTranspose1xWKernel (if the output tensor is a matrix)
 *  -# @ref CLGEMMMatrixMultiplyKernel
 *  -# @ref CLGEMMMatrixAdditionKernel (if c != nullptr and beta != 0.0)
 *
 */
class CLGEMM : public IFunction
{
public:
    /** Default constructor. */
    CLGEMM(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Initialise the kernel's inputs and output
     *
     * @note GEMM: General Matrix Multiply - [alpha * A * B + beta * C].
     *
     * @note All tensors must have the same data type.
     *
     * @note Whilst the first input tensor can be a vector, the second input tensor must be at least a matrix
     *
     * @param[in]  a         First input tensor  (Matrix or Vector A). Data types supported: QS8/QS16/F16/F32
     * @param[in]  b         Second input tensor (Matrix B). Data type supported: same as @p a.
     * @param[in]  c         Third input tensor  (Matrix C). It can be a nullptr if just the multiplication between @p a and @p b is needed. Data type supported: same as @p a.
     * @param[out] output    Output tensor. Data type supported: same as @p a
     * @param[in]  alpha     Weight of the matrix product
     * @param[in]  beta      Weight of matrix C
     * @param[in]  gemm_info (Optional) Specifies if the matrix A and/or matrix B have been reshaped and
     *                       if the reshape of matrix B should happen only for the first run
     */
    void configure(const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output, float alpha, float beta, const GEMMInfo &gemm_info = GEMMInfo());

    // Inherited methods overridden:
    void run() override;

private:
    CLMemoryGroup              _memory_group;
    CLGEMMInterleave4x4Kernel  _interleave_kernel;
    CLGEMMTranspose1xWKernel   _transpose_kernel;
    CLGEMMMatrixMultiplyKernel _mm_kernel;
    CLGEMMMatrixAdditionKernel _ma_kernel;
    CLTensor                   _tmp_a;
    CLTensor                   _tmp_b;
    bool                       _is_interleaved_transposed;
    bool                       _run_addition;
    bool                       _is_first_run;
    bool                       _reshape_b_only_on_first_run;
};
}

#endif /* __ARM_COMPUTE_CLGEMM_H__ */
