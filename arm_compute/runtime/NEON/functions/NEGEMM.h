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
#ifndef __ARM_COMPUTE_NEGEMM_H__
#define __ARM_COMPUTE_NEGEMM_H__

#include "arm_compute/core/NEON/kernels/NEFillBorderKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMAssemblyBaseKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMInterleave4x4Kernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMMatrixAdditionKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMMatrixMultiplyKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMTranspose1xWKernel.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"

#include <memory>

namespace arm_compute
{
/** Basic function to execute GEMM on NEON. This function calls the following NEON kernels:
 *
 *  -# @ref NEGEMMInterleave4x4Kernel (if the output tensor is a matrix)
 *  -# @ref NEGEMMTranspose1xWKernel (if the output tensor is a matrix)
 *  -# @ref NEGEMMMatrixMultiplyKernel
 *  -# @ref NEGEMMMatrixAdditionKernel (if c != nullptr and beta != 0.0)
 *
 */
class NEGEMM : public IFunction
{
public:
    /** Constructor */
    NEGEMM(std::shared_ptr<IMemoryManager> memory_manager = nullptr);

    /** Initialise the kernel's inputs, output
     *
     * @note GEMM: General Matrix Multiply - [alpha * A * B + beta * C].
     * @note GEMM: The tensors a, b, c, d must have the same data type. You should not mix data types when calling this function.
     *
     * @param[in]  a         First input tensor  (Matrix A or Vector A). Data type supported: QS8/QS16/F16/F32
     * @param[in]  b         Second input tensor (Matrix B). Data type supported: same as @p a
     * @param[in]  c         Third input tensor  (Matrix C). It can be a nullptr if just the multiplication between @p a and @p b is needed. Data type supported: same as @p a
     * @param[out] d         Output tensor. Data type supported: same as @p a
     * @param[in]  alpha     Weight of the matrix product
     * @param[in]  beta      Weight of matrix C
     * @param[in]  gemm_info (Optional) Specifies if the matrix A and/or matrix B have been reshaped and
     *                       if the reshape of matrix B should happen only for the first run
     */
    void configure(const ITensor *a, const ITensor *b, const ITensor *c, ITensor *d, float alpha, float beta, const GEMMInfo &gemm_info = GEMMInfo());

    // Inherited methods overridden:
    void run() override;

private:
    MemoryGroup                               _memory_group;
    NEGEMMInterleave4x4Kernel                 _interleave_kernel;
    NEGEMMTranspose1xWKernel                  _transpose_kernel;
    NEGEMMMatrixMultiplyKernel                _mm_kernel;
    std::unique_ptr<NEGEMMAssemblyBaseKernel> _mm_optimised_kernel;
    NEGEMMMatrixAdditionKernel                _ma_kernel;
    Tensor                                    _tmp_a;
    Tensor                                    _tmp_b;
    Tensor                                    _workspace;
    bool                                      _run_vector_matrix_multiplication;
    bool                                      _run_addition;
    bool                                      _is_first_run;
    bool                                      _reshape_b_only_on_first_run;
};
}
#endif /*__ARM_COMPUTE_NEGEMM_H__ */
