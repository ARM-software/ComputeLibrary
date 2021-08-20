/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_GEMMLOWP_MATRIXMULTIPLY_CORE_H
#define ARM_COMPUTE_CPU_GEMMLOWP_MATRIXMULTIPLY_CORE_H

#include "arm_compute/core/TensorInfo.h"
#include "src/core/common/Macros.h"
#include "src/cpu/ICpuOperator.h"

#include <memory>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
class CpuGemmInterleave4x4Kernel;
class CpuGemmLowpMatrixMultiplyKernel;
class CpuGemmLowpOffsetContributionKernel;
class CpuGemmLowpOffsetContributionOutputStageKernel;
class CpuGemmLowpMatrixAReductionKernel;
class CpuGemmLowpMatrixBReductionKernel;
class CpuGemmTranspose1xWKernel;
class CpuConvertQuantizedSignednessKernel;
} // namespace kernels
class CpuGemmAssemblyDispatch;
class CpuActivation;

/** Basic function to execute GEMMLowpMatrixMultiplyCore. This function calls the following kernels if the DOT product instruction is not available:
 *
 *  -# @ref kernels::CpuGemmInterleave4x4Kernel
 *  -# @ref kernels::CpuGemmTranspose1xWKernel
 *  -# @ref kernels::CpuGemmLowpMatrixMultiplyKernel
 *  -# @ref kernels::CpuGemmLowpOffsetContributionKernel
 *  -# @ref CpuActivation
 *
 * otherwise if the DOT product instruction is available:
 *
 *  -# @ref kernels::CpuGemmLowpOffsetContributionKernel
 *
*/
class CpuGemmLowpMatrixMultiplyCore : public ICpuOperator
{
public:
    /** Constructor */
    CpuGemmLowpMatrixMultiplyCore();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuGemmLowpMatrixMultiplyCore);
    /** Destructor */
    ~CpuGemmLowpMatrixMultiplyCore();
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
     * @note GEMM_LOWP:  low precision GEMM kernel
     *  This kernel performs the following computations:
     *
     *  -# Convert a values from QASYMM8 to int32 and add a_offset to each of them.
     *  -# Convert b values from QASYMM8 to int32 add b_offset to each of them.
     *  -# Compute the matrix product of the resulting a * b in int32.
     *
     * @note The @p output type is S32 if @p gemm_info.type == GEMMLowpOutputStageType::NONE. It is QASYMM8/QASYMM8_SIGNED otherwise
     *
     * @param[in]  a         First input tensor info (Matrix A). Data type supported: QASYMM8/QASYMM8_SIGNED.
     * @param[in]  b         Second input tensor info (Matrix B). Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8/QSYMM8_PER_CHANNEL.
     * @param[in]  c         Third input tensor info (Matrix C). It can be a nullptr. Data type supported: S32
     * @param[out] dst       Output tensor info. Data type supported: Data type supported: S32/QASYMM8/QASYMM8_SIGNED
     * @param[in]  gemm_info (Optional) Specifies if the matrix A and/or matrix B have been reshaped and
     *                       if the reshape of matrix B should be executed only for the first run
     */
    void configure(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, ITensorInfo *dst, const GEMMInfo &gemm_info = GEMMInfo());
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuGemmLowpMatrixMultiplyCore::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *dst, const GEMMInfo &gemm_info = GEMMInfo());

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
    void prepare(ITensorPack &tensors) override;
    experimental::MemoryRequirements workspace() const override;

private:
    enum AuxTensorIdx
    {
        AsmGemmWorkspace = 0,
        Pretranspose,
        VectorSumCol,
        VectorSumRow,
        TmpA,
        TmpB,
        MMResultS32,
        SignedA,
        SignedOutput,
        Count
    };

    std::unique_ptr<CpuGemmAssemblyDispatch>                                 _asm_glue;
    std::unique_ptr<kernels::CpuGemmLowpMatrixMultiplyKernel>                _mm_kernel;
    std::unique_ptr<kernels::CpuGemmInterleave4x4Kernel>                     _mtx_a_reshape_kernel;
    std::unique_ptr<kernels::CpuGemmTranspose1xWKernel>                      _mtx_b_reshape_kernel;
    std::unique_ptr<kernels::CpuGemmLowpMatrixAReductionKernel>              _mtx_a_reduction_kernel;
    std::unique_ptr<kernels::CpuGemmLowpMatrixBReductionKernel>              _mtx_b_reduction_kernel;
    std::unique_ptr<kernels::CpuGemmLowpOffsetContributionKernel>            _offset_contribution_kernel;
    std::unique_ptr<kernels::CpuGemmLowpOffsetContributionOutputStageKernel> _offset_contribution_output_stage_kernel;
    std::unique_ptr<CpuActivation>                                           _activation_func;
    std::unique_ptr<kernels::CpuConvertQuantizedSignednessKernel>            _convert_to_signed_asymm;
    std::unique_ptr<kernels::CpuConvertQuantizedSignednessKernel>            _convert_from_signed_asymm;

    TensorInfo _vector_sum_col;
    TensorInfo _vector_sum_row;
    TensorInfo _tmp_a;
    TensorInfo _tmp_b;
    TensorInfo _mm_result_s32;
    TensorInfo _signed_a;
    TensorInfo _signed_output;
    int32_t    _a_offset;
    int32_t    _b_offset;

    bool                             _run_vector_matrix_multiplication;
    bool                             _assembly_path;
    bool                             _fused_assembly_path;
    bool                             _reshape_b_only_on_first_run;
    bool                             _is_prepared;
    bool                             _fuse_output_stage;
    bool                             _run_activation;
    bool                             _flip_signedness;
    GEMMInfo                         _gemm_info;
    experimental::MemoryRequirements _aux_mem{};
};
} // namespace cpu
} // namespace arm_compute
#endif /*ARM_COMPUTE_CPU_GEMMLOWP_MATRIXMULTIPLY_CORE_H */
