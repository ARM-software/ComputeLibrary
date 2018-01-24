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
#ifndef __ARM_COMPUTE_NEGEMMLOWPMATRIXMULTIPLYCORE_H__
#define __ARM_COMPUTE_NEGEMMLOWPMATRIXMULTIPLYCORE_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMLowpOffsetContributionKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMLowpReductionKernel.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"

#include <memory>

namespace arm_compute
{
class ITensor;

/** Basic function to execute GEMMLowpMatrixMultiplyCore on NEON. This function calls the following NEON kernels if the DOT product instruction is not available:
 *
 *  -# @ref NEGEMMInterleave4x4Kernel
 *  -# @ref NEGEMMTranspose1xWKernel
 *  -# @ref NEGEMMLowpMatrixMultiplyKernel
 *  -# @ref NEGEMMLowpOffsetContributionKernel
 *
 * otherwise if the DOT product instruction is available:
 *
 *  -# @ref NEGEMMInterleaveBlockedKernel
 *  -# @ref NEGEMMLowpAArch64V8P4Kernel
 *  -# @ref NEGEMMLowpOffsetContributionKernel
 *
*/
class NEGEMMLowpMatrixMultiplyCore : public IFunction
{
public:
    /** Constructor */
    NEGEMMLowpMatrixMultiplyCore(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Initialise the kernel's inputs, output
     *
     * @note GEMM_LOWP:  low precision GEMM kernel
     *  This kernel performs the following computations:
     *
     *  -# Convert a values from QASYMM8 to int32 and add a_offset to each of them.
     *  -# Convert b values from QASYMM8 to int32 add b_offset to each of them.
     *  -# Compute the matrix product of the resulting a * b in int32.
     *
     * @param[in]  a         First input tensor  (Matrix A). Data type supported: QASYMM8.
     * @param[in]  b         Second input tensor (Matrix B). Data type supported: same as @p a
     * @param[out] output    Output tensor. Data type supported: Data type supported: S32
     * @param[in]  gemm_info (Optional) Specifies if the matrix A and/or matrix B have been reshaped and
     *                       if the reshape of matrix B should be executed only for the first run
     */
    void configure(const ITensor *a, const ITensor *b, ITensor *output, const GEMMInfo &gemm_info = GEMMInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMLowpMatrixMultiplyCore
     *
     * @param[in] a         First input tensor  (Matrix A). Data type supported: QASYMM8.
     * @param[in] b         Second input tensor (Matrix B). Data type supported: same as @p a
     * @param[in] output    Output tensor. Data type supported: Data type supported: S32
     * @param[in] gemm_info (Optional) Specifies if the matrix A and/or matrix B have been reshaped and
     *                      if the reshape of matrix B should be executed only for the first run
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *output, const GEMMInfo &gemm_info = GEMMInfo());

    // Inherited methods overridden:
    void run() override;

private:
    MemoryGroup                        _memory_group;
    std::unique_ptr<INEKernel>         _mm_kernel;
    std::unique_ptr<INEKernel>         _mtx_a_reshape_kernel;
    std::unique_ptr<INEKernel>         _mtx_b_reshape_kernel;
    NEGEMMLowpMatrixAReductionKernel   _mtx_a_reduction_kernel;
    NEGEMMLowpMatrixBReductionKernel   _mtx_b_reduction_kernel;
    NEGEMMLowpOffsetContributionKernel _offset_contribution_kernel;
    Tensor                             _vector_sum_col;
    Tensor                             _vector_sum_row;
    Tensor                             _tmp_a;
    Tensor                             _tmp_b;
    Tensor                             _workspace;
    int32_t                            _a_offset;
    int32_t                            _b_offset;
    bool                               _run_vector_matrix_multiplication;
    bool                               _dot_product_path;
};
}
#endif /*__ARM_COMPUTE_NEGEMMLOWPMATRIXMULTIPLYCORE_H__ */
