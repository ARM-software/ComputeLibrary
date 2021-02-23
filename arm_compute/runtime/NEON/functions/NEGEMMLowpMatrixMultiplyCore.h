/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NEGEMMLOWPMATRIXMULTIPLYCORE_H
#define ARM_COMPUTE_NEGEMMLOWPMATRIXMULTIPLYCORE_H

#include "NEActivationLayer.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/IWeightsManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"

#include <memory>

namespace arm_compute
{
class ITensor;
class NEConvertQuantizedSignednessKernel;
class NEConvertQuantizedSignednessKernel;
class NEGEMMInterleave4x4Kernel;
class NEGEMMLowpMatrixMultiplyKernel;
class NEGEMMLowpOffsetContributionKernel;
class NEGEMMLowpOffsetContributionOutputStageKernel;
class NEGEMMLowpMatrixAReductionKernel;
class NEGEMMLowpMatrixBReductionKernel;
class NEGEMMTranspose1xWKernel;
class NEGEMMAssemblyDispatch;

/** Basic function to execute GEMMLowpMatrixMultiplyCore on Neon. This function calls the following Neon kernels if the DOT product instruction is not available:
 *
 *  -# @ref NEGEMMInterleave4x4Kernel
 *  -# @ref NEGEMMTranspose1xWKernel
 *  -# @ref NEGEMMLowpMatrixMultiplyKernel
 *  -# @ref NEGEMMLowpOffsetContributionKernel
 *  -# @ref NEActivationLayer
 *
 * otherwise if the DOT product instruction is available:
 *
 *  -# @ref NEGEMMLowpOffsetContributionKernel
 *
*/
class NEGEMMLowpMatrixMultiplyCore : public IFunction
{
public:
    /** Constructor */
    NEGEMMLowpMatrixMultiplyCore(std::shared_ptr<IMemoryManager> memory_manager = nullptr, IWeightsManager *weights_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMLowpMatrixMultiplyCore(const NEGEMMLowpMatrixMultiplyCore &) = delete;
    /** Default move constructor */
    NEGEMMLowpMatrixMultiplyCore(NEGEMMLowpMatrixMultiplyCore &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMLowpMatrixMultiplyCore &operator=(const NEGEMMLowpMatrixMultiplyCore &) = delete;
    /** Default move assignment operator */
    NEGEMMLowpMatrixMultiplyCore &operator=(NEGEMMLowpMatrixMultiplyCore &&) = default;
    /** Default destructor */
    ~NEGEMMLowpMatrixMultiplyCore();
    /** Initialise the kernel's inputs, output
     *
     * @note GEMM_LOWP:  low precision GEMM kernel
     *  This kernel performs the following computations:
     *
     *  -# Convert a values from QASYMM8 to int32 and add a_offset to each of them.
     *  -# Convert b values from QASYMM8 to int32 add b_offset to each of them.
     *  -# Compute the matrix product of the resulting a * b in int32.
     *
     * @note The @p output type is S32 if @p gemm_info.type == GEMMLowpOutputStageType::NONE. It is QASYMM8/QASYMM8_SIGNED otherwise
     *
     * @param[in]  a         First input tensor  (Matrix A). Data type supported: QASYMM8/QASYMM8_SIGNED.
     * @param[in]  b         Second input tensor (Matrix B). Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8/QSYMM8_PER_CHANNEL.
     * @param[in]  c         Third input tensor  (Matrix C). It can be a nullptr. Data type supported: S32
     * @param[out] output    Output tensor. Data type supported: Data type supported: S32/QASYMM8/QASYMM8_SIGNED
     * @param[in]  gemm_info (Optional) Specifies if the matrix A and/or matrix B have been reshaped and
     *                       if the reshape of matrix B should be executed only for the first run
     */
    void configure(const ITensor *a, const ITensor *b, const ITensor *c, ITensor *output, const GEMMInfo &gemm_info = GEMMInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMLowpMatrixMultiplyCore
     *
     * @note The @p output type is S32 if @p gemm_info.type == GEMMLowpOutputStageType::NONE. It is QASYMM8/QASYMM8_SIGNED otherwise
     *
     * @param[in] a         First input tensor info  (Matrix A). Data type supported: QASYMM8/QASYMM8_SIGNED.
     * @param[in] b         Second input tensor info (Matrix B). Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8/QSYMM8_PER_CHANNEL.
     * @param[in] c         Third input tensor  info (Matrix C). It can be a nullptr. Data type supported: S32
     * @param[in] output    Output tensor info. Data type supported: Data type supported: S32/QASYMM8/QASYMM8_SIGNED
     * @param[in] gemm_info (Optional) Specifies if the matrix A and/or matrix B have been reshaped and
     *                      if the reshape of matrix B should be executed only for the first run
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, const GEMMInfo &gemm_info = GEMMInfo());

    // Inherited methods overridden
    void run() override;
    void prepare() override;

private:
    MemoryGroup                                                    _memory_group;
    IWeightsManager                                               *_weights_manager;
    std::unique_ptr<NEGEMMAssemblyDispatch>                        _asm_glue;
    std::unique_ptr<NEGEMMLowpMatrixMultiplyKernel>                _mm_kernel;
    std::unique_ptr<NEGEMMInterleave4x4Kernel>                     _mtx_a_reshape_kernel;
    std::unique_ptr<NEGEMMTranspose1xWKernel>                      _mtx_b_reshape_kernel;
    std::unique_ptr<NEGEMMLowpMatrixAReductionKernel>              _mtx_a_reduction_kernel;
    std::unique_ptr<NEGEMMLowpMatrixBReductionKernel>              _mtx_b_reduction_kernel;
    std::unique_ptr<NEGEMMLowpOffsetContributionKernel>            _offset_contribution_kernel;
    std::unique_ptr<NEGEMMLowpOffsetContributionOutputStageKernel> _offset_contribution_output_stage_kernel;
    NEActivationLayer                                              _activation_func;
    std::unique_ptr<NEConvertQuantizedSignednessKernel>            _convert_to_signed_asymm;
    std::unique_ptr<NEConvertQuantizedSignednessKernel>            _convert_from_signed_asymm;

    Tensor         _vector_sum_col;
    Tensor         _vector_sum_row;
    Tensor         _tmp_a;
    Tensor         _tmp_b;
    Tensor         _mm_result_s32;
    Tensor         _signed_a;
    Tensor         _signed_output;
    const ITensor *_original_b;
    int32_t        _a_offset;
    int32_t        _b_offset;

    bool _run_vector_matrix_multiplication;
    bool _assembly_path;
    bool _fused_assembly_path;
    bool _reshape_b_only_on_first_run;
    bool _is_prepared;
    bool _fuse_output_stage;
    bool _run_activation;
    bool _flip_signedness;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEGEMMLOWPMATRIXMULTIPLYCORE_H */
