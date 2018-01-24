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
#ifndef __ARM_COMPUTE_CLGEMMLOWPMATRIXMULTIPLYCORE_H__
#define __ARM_COMPUTE_CLGEMMLOWPMATRIXMULTIPLYCORE_H__

#include "arm_compute/core/CL/kernels/CLGEMMInterleave4x4Kernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMLowpMatrixMultiplyKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMLowpOffsetContributionKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMLowpReductionKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMTranspose1xWKernel.h"
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
class IMemoryManager;
class ICLTensor;

/** Basic function to execute GEMMLowpMatrixMultiplyCore on OpenCL. This function calls the following OpenCL kernels:
 *
 *  -# @ref CLGEMMInterleave4x4Kernel  (if the output tensor is a matrix)
 *  -# @ref CLGEMMTranspose1xWKernel  (if the output tensor is a matrix)
 *  -# @ref CLGEMMLowpMatrixMultiplyKernel
 *  -# @ref CLGEMMLowpMatrixAReductionKernel (if the offset of matrix B is not 0)
 *  -# @ref CLGEMMLowpMatrixBReductionKernel (if the offset of matrix A is not 0)
 *  -# @ref CLGEMMLowpOffsetContributionKernel
 *
*/
class CLGEMMLowpMatrixMultiplyCore : public IFunction
{
public:
    /** Constructor */
    CLGEMMLowpMatrixMultiplyCore(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
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
    void configure(const ICLTensor *a, const ICLTensor *b, ICLTensor *output, const GEMMInfo &gemm_info = GEMMInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMLowpMatrixMultiplyCore
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
    CLMemoryGroup                      _memory_group;
    CLGEMMLowpMatrixMultiplyKernel     _mm_kernel;
    CLGEMMInterleave4x4Kernel          _mtx_a_reshape_kernel;
    CLGEMMTranspose1xWKernel           _mtx_b_reshape_kernel;
    CLGEMMLowpMatrixAReductionKernel   _mtx_a_reduction_kernel;
    CLGEMMLowpMatrixBReductionKernel   _mtx_b_reduction_kernel;
    CLGEMMLowpOffsetContributionKernel _offset_contribution_kernel;
    CLTensor                           _vector_sum_col;
    CLTensor                           _vector_sum_row;
    CLTensor                           _tmp_a;
    CLTensor                           _tmp_b;
    int32_t                            _a_offset;
    int32_t                            _b_offset;
    bool                               _is_interleaved_transposed;
    bool                               _is_first_run;
    bool                               _reshape_b_only_on_first_run;
};
}
#endif /*__ARM_COMPUTE_CLGEMMLOWPMATRIXMULTIPLYCORE_H__ */
