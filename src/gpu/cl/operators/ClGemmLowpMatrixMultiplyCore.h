/*
 * Copyright (c) 2017-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CL_GEMMLOWP_MATRIXMULTIPLY_CORE_H
#define ARM_COMPUTE_CL_GEMMLOWP_MATRIXMULTIPLY_CORE_H

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/CL/CLTypes.h"

#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClOperator.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
// Forward declarations
class ClCastKernel;
class ClGemmLowpMatrixMultiplyNativeKernel;
class ClGemmLowpMatrixMultiplyReshapedOnlyRhsKernel;
class ClGemmLowpMatrixMultiplyReshapedOnlyRhsMMULKernel;
class ClGemmReshapeRhsMatrixKernel;
class ClGemmLowpMatrixAReductionKernel;
class ClGemmLowpMatrixBReductionKernel;
class ClGemmLowpOffsetContributionKernel;
class ClGemmLowpOffsetContributionOutputStageKernel;
} // namespace kernels

/** Basic function to execute GEMMLowpMatrixMultiplyCore on OpenCL. */
class ClGemmLowpMatrixMultiplyCore : public IClOperator
{
public:
    ClGemmLowpMatrixMultiplyCore();
    ~ClGemmLowpMatrixMultiplyCore();
    /** Initialise the kernel's inputs, output
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src0           |src1               |src2     |dst            |
     * |:--------------|:------------------|:--------|:--------------|
     * |QASYMM8        |QASYMM8            |S32      |QASYMM8        |
     * |QASYMM8        |QSYMM8_PER_CHANNEL |S32      |QASYMM8        |
     * |QASYMM8        |QSYMM8             |S32      |QASYMM8        |
     * |QASYMM8        |QASYMM8            |S32      |S32            |
     * |QASYMM8        |QSYMM8_PER_CHANNEL |S32      |S32            |
     * |QASYMM8        |QSYMM8             |S32      |S32            |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32      |QASYMM8_SIGNED |
     * |QASYMM8_SIGNED |QSYMM8_PER_CHANNEL |S32      |QASYMM8_SIGNED |
     * |QASYMM8_SIGNED |QSYMM8             |S32      |QASYMM8_SIGNED |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32      |S32            |
     * |QASYMM8_SIGNED |QSYMM8_PER_CHANNEL |S32      |S32            |
     * |QASYMM8_SIGNED |QSYMM8             |S32      |S32            |
     *
     * @note GEMMLowp:  low precision GEMM kernel. [A * B + C]
     *  This kernel performs the following computations:
     *
     *  -# Convert a values from 8-bit quantized to int32 and add a_offset to each of them.
     *  -# Convert b values from 8-bit quantized to int32 and add b_offset to each of them.
     *  -# Compute the matrix product of the resulting a * b in int32.
     *  -# Quantize to uint8 if gemm_info.gemmlowp_output_stage != NONE
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  a               First input tensor  (Matrix A). Data type supported: QASYMM8/QASYMM8_SIGNED.
     * @param[in]  b               Second input tensor (Matrix B). Data type supported: same as @p a
     * @param[in]  c               Third input tensor  (Matrix C). It can be a nullptr. Data type supported: S32
     * @param[out] output          Output tensor. Data type supported: S32 or QASYMM8/QASYMM8_SIGNED if gemm_info.gemmlowp_output_stage != NONE
     * @param[in]  gemm_info       (Optional) Specifies if the matrix A and/or matrix B have been reshaped and
     *                       if the reshape of matrix B should be executed only for the first run
     */
    void configure(const CLCompileContext &compile_context, ITensorInfo *a, ITensorInfo *b, ITensorInfo *c, ITensorInfo *output, const GEMMInfo &gemm_info = GEMMInfo());
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to ClGemmLowpMatrixMultiplyCore::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, const GEMMInfo &gemm_info = GEMMInfo());

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
    void prepare(ITensorPack &constants) override;
    experimental::MemoryRequirements workspace() const override;

private:
    enum AuxTensorIdx
    {
        ResultS32 = 0,
        RhsQAsymm8,
        RhsReshape,
        VecSumCol,
        VecSumRow,
        Multipliers,
        Shifts,
        Count
    };

private:
    // Kernels used
    std::unique_ptr<kernels::ClCastKernel>                                      _weights_to_qasymm8;
    std::unique_ptr<kernels::ClGemmLowpMatrixMultiplyNativeKernel>              _mm_native_kernel;
    std::unique_ptr<kernels::ClGemmLowpMatrixMultiplyReshapedOnlyRhsKernel>     _mm_reshaped_only_rhs_kernel;
    std::unique_ptr<kernels::ClGemmLowpMatrixMultiplyReshapedOnlyRhsMMULKernel> _mm_reshaped_only_rhs_mmul_kernel;
    std::unique_ptr<kernels::ClGemmReshapeRhsMatrixKernel>                      _mtx_b_reshape_kernel;
    std::unique_ptr<kernels::ClGemmLowpMatrixAReductionKernel>                  _mtx_a_reduction_kernel;
    std::unique_ptr<kernels::ClGemmLowpMatrixBReductionKernel>                  _mtx_b_reduction_kernel;
    std::unique_ptr<kernels::ClGemmLowpOffsetContributionKernel>                _offset_contribution_kernel;
    std::unique_ptr<kernels::ClGemmLowpOffsetContributionOutputStageKernel>     _offset_contribution_output_stage_kernel;

    // Temporary tensors
    TensorInfo _qasymm8_weights{};
    TensorInfo _vector_sum_col{};
    TensorInfo _vector_sum_row{};
    TensorInfo _tmp_b{};
    TensorInfo _mm_result_s32{};
    TensorInfo _gemm_output_stage_multipliers{};
    TensorInfo _gemm_output_stage_shifts{};

    int32_t          _a_offset{ 0 };
    int32_t          _b_offset{ 0 };
    bool             _reshape_b_only_on_first_run{ false };
    bool             _run_output_stage{ false };
    bool             _convert_to_qasymm8{ false };
    bool             _run_offset_contribution{ false };
    bool             _is_prepared{ false };
    GEMMInfo         _gemm_info{};
    CLGEMMKernelType _gemm_kernel_type{};

    experimental::MemoryRequirements _aux_mem{};
};
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_GEMMLOWP_MATRIXMULTIPLY_CORE_H */