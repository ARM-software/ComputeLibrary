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
#ifndef __ARM_COMPUTE_CLGEMMLOWPMATRIXMULTIPLYCORE_H__
#define __ARM_COMPUTE_CLGEMMLOWPMATRIXMULTIPLYCORE_H__

#include "arm_compute/core/CL/kernels/CLGEMMLowpMatrixMultiplyKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMLowpMatrixMultiplyNativeKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMLowpOffsetContributionKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMLowpOffsetContributionOutputStageKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMLowpReductionKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMReshapeRHSMatrixKernel.h"
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
class IMemoryManager;
class ICLTensor;

/** Basic function to execute GEMMLowpMatrixMultiplyCore on OpenCL. This function calls the following OpenCL kernels:
 *
 *  -# @ref CLGEMMReshapeRHSMatrixKernel  (if the output tensor is a matrix)
 *  -# @ref CLGEMMLowpMatrixMultiplyKernel (if the parameter "reshape_b_only_on_first_run" of GEMMInfo is FALSE)
 *  -# @ref CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel (if the parameter "reshape_b_only_on_first_run" of GEMMInfo is TRUE)
 *  -# @ref CLGEMMLowpMatrixAReductionKernel (if the offset of matrix B is not 0)
 *  -# @ref CLGEMMLowpMatrixBReductionKernel (if the offset of matrix A is not 0)
 *  -# @ref CLGEMMLowpOffsetContributionKernel (if gemm_info.gemmlowp_output_stage == NONE)
 *  -# @ref CLGEMMLowpOffsetContributionOutputStageKernel (if gemm_info.gemmlowp_output_stage != NONE)
 *
*/
class CLGEMMLowpMatrixMultiplyCore : public IFunction
{
public:
    /** Constructor */
    CLGEMMLowpMatrixMultiplyCore(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMLowpMatrixMultiplyCore(const CLGEMMLowpMatrixMultiplyCore &) = delete;
    /** Default move constructor */
    CLGEMMLowpMatrixMultiplyCore(CLGEMMLowpMatrixMultiplyCore &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMLowpMatrixMultiplyCore &operator=(const CLGEMMLowpMatrixMultiplyCore &) = delete;
    /** Default move assignment operator */
    CLGEMMLowpMatrixMultiplyCore &operator=(CLGEMMLowpMatrixMultiplyCore &&) = default;
    /** Initialise the kernel's inputs, output
     *
     * @note GEMMLowp:  low precision GEMM kernel. [A * B + C]
     *  This kernel performs the following computations:
     *
     *  -# Convert a values from QASYMM8 to int32 and add a_offset to each of them.
     *  -# Convert b values from QASYMM8 to int32 and add b_offset to each of them.
     *  -# Compute the matrix product of the resulting a * b in int32.
     *  -# Quantize to uint8 if gemm_info.gemmlowp_output_stage != NONE
     *
     * @param[in]  a         First input tensor  (Matrix A). Data type supported: QASYMM8.
     * @param[in]  b         Second input tensor (Matrix B). Data type supported: same as @p a
     * @param[in]  c         Third input tensor  (Matrix C). It can be a nullptr. Data type supported: S32
     * @param[out] output    Output tensor. Data type supported: S32 or QASYMM8 if gemm_info.gemmlowp_output_stage != NONE
     * @param[in]  gemm_info (Optional) Specifies if the matrix A and/or matrix B have been reshaped and
     *                       if the reshape of matrix B should be executed only for the first run
     */
    void configure(const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output, const GEMMInfo &gemm_info = GEMMInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMLowpMatrixMultiplyCore
     *
     * @param[in] a         First input tensor  (Matrix A). Data type supported: QASYMM8.
     * @param[in] b         Second input tensor (Matrix B). Data type supported: same as @p a
     * @param[in] c         Third input tensor  (Matrix C). It can be a nullptr. Data type supported: S32
     * @param[in] output    Output tensor. Data type supported: S32 or QASYMM8 if gemm_info.gemmlowp_output_stage != NONE
     * @param[in] gemm_info (Optional) Specifies if the matrix A and/or matrix B have been reshaped and
     *                      if the reshape of matrix B should be executed only for the first run
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, const GEMMInfo &gemm_info = GEMMInfo());

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    CLMemoryGroup                                 _memory_group;
    CLGEMMLowpMatrixMultiplyKernel                _mm_midgard_kernel;
    CLGEMMLowpMatrixMultiplyNativeKernel          _mm_native_kernel;
    CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel _mm_reshaped_only_rhs_kernel;
    CLGEMMReshapeRHSMatrixKernel                  _mtx_b_reshape_kernel;
    CLGEMMLowpMatrixAReductionKernel              _mtx_a_reduction_kernel;
    CLGEMMLowpMatrixBReductionKernel              _mtx_b_reduction_kernel;
    CLGEMMLowpOffsetContributionKernel            _offset_contribution_kernel;
    CLGEMMLowpOffsetContributionOutputStageKernel _offset_contribution_output_stage_kernel;
    CLTensor                                      _vector_sum_col;
    CLTensor                                      _vector_sum_row;
    CLTensor                                      _tmp_b;
    CLTensor                                      _mm_result_s32;
    const ICLTensor                              *_original_b;
    int32_t                                       _a_offset;
    int32_t                                       _b_offset;
    bool                                          _is_gemm_reshaped;
    bool                                          _is_midgard;
    bool                                          _reshape_b_only_on_first_run;
    bool                                          _is_prepared;
    bool                                          _fuse_output_stage;
};
}
#endif /*__ARM_COMPUTE_CLGEMMLOWPMATRIXMULTIPLYCORE_H__ */