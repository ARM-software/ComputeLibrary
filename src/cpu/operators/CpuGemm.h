/*
 * Copyright (c) 2021-2023 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_GEMM_H
#define ARM_COMPUTE_CPU_GEMM_H

#include "src/cpu/ICpuOperator.h"

#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "src/cpu/kernels/CpuGemmInterleave4x4Kernel.h"
#include "src/cpu/kernels/CpuGemmMatrixAdditionKernel.h"
#include "src/cpu/kernels/CpuGemmMatrixMultiplyKernel.h"
#include "src/cpu/kernels/CpuGemmTranspose1xWKernel.h"
#include "src/cpu/operators/CpuActivation.h"
#include "src/cpu/operators/CpuAdd.h"
#include "src/cpu/operators/internal/CpuGemmAssemblyDispatch.h"

#include <memory>

namespace arm_compute
{
namespace cpu
{
/** Basic function to execute GEMM. This function calls the following kernels:
 *
 * If optimized assembly is available:
 *  -# @ref cpu::CpuGemmAssemblyDispatch
 *  -# @ref cpu::CpuActivation (if alpha != 1.0)
 * Else:
 *  -# @ref cpu::kernels::CpuGemmInterleave4x4Kernel (if the output tensor is a matrix)
 *  -# @ref cpu::kernels::CpuGemmTranspose1xWKernel (if the output tensor is a matrix)
 *  -# @ref cpu::kernels::CpuGemmMatrixMultiplyKernel
 * In both cases:
 *  -# @ref cpu::kernels::CpuGemmMatrixAdditionKernel (if c != nullptr and beta != 0.0 and is not reshaped once)
 * Else:
 *  -# @ref cpu::CpuAdd (if c != nullptr and is reshaped once and not optimized assembly in place)
 *
 *  -# @ref cpu::CpuActivation (if activation is specified in GEMMInfo)
 */
class CpuGemm : public ICpuOperator
{
public:
    /** Default constructor */
    CpuGemm() = default;
    /** Default destructor */
    ~CpuGemm() = default;
    /** Configure operator for a given list of arguments
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0         |src1        |src2      |dst            |
     * |:------------|:-----------|:---------|:--------------|
     * |F32          |F32         |F32       |F32            |
     * |F16          |F16         |F16       |F16            |
     * |BFLOAT16     |BFLOAT16    |BFLOAT16  |FP32           |
     *
     * @note GEMM: General Matrix Multiply - [alpha * A * B + beta * C].
     * @note GEMM: The tensors a, b, c, d must have the same data type. You should not mix data types when calling this function.
     *
     * @note Batched GEMM only supports broadcasting cases where RHS rank < LHS rank but not the other way around
     *
     * @param[in]  a         First input tensor info (Matrix A or Vector A). Data type supported: BFLOAT16/F16/F32
     * @param[in]  b         Second input tensor info (Matrix B). Data type supported: same as @p a
     * @param[in]  c         Third input tensor info (Matrix C). It can be a nullptr if just the multiplication between @p a and @p b is needed. Data type supported: same as @p a
     * @param[out] d         Output tensor info. Data type supported: same as @p a
     * @param[in]  alpha     Weight of the matrix product
     * @param[in]  beta      Weight of matrix C
     * @param[in]  gemm_info (Optional) Specifies if the matrix A and/or matrix B have been reshaped and
     *                       if the reshape of matrix B should happen only for the first run
     */
    void configure(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, ITensorInfo *d,
                   float alpha, float beta, const GEMMInfo &gemm_info = GEMMInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CpuGemm.
     *
     * Similar to @ref CpuGemm::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *d,
                           float alpha, float beta, const GEMMInfo &gemm_info = GEMMInfo());

    /** Indicates whether or not there is an optimal assembly implementation that can be used to process the given parameters.
     *
     * This method has the same use of @ref
     * NEGEMMConvolutionLayer::has_opt_impl, with the only caveat that
     * the value of arm_compute::WeightFormat need to be passed via the
     * parameter gemm_info.
     */
    static Status has_opt_impl(arm_compute::WeightFormat &weight_format, const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *d,
                               const GEMMInfo &gemm_info = GEMMInfo());

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
    void prepare(ITensorPack &constants) override;
    experimental::MemoryRequirements workspace() const override;

    /** Indicates if the convolution executes in variable weights mode.
     *
     * When ACL executes convolution in variable weights mode, it does
     * not perform any processing of the weights tensor. Instead, it
     * utilizes the data as it is given by the user.
     */
    bool isVarWeightsKernel() const;

private:
    enum AuxTensorIdx
    {
        AsmGemmWorkspace = 0,
        Pretraspose,
        InterleavedLHS,
        TransposedRHS,
        TempResult,
        Count
    };

    std::unique_ptr<kernels::CpuGemmInterleave4x4Kernel>  _interleave_kernel{ nullptr };
    std::unique_ptr<kernels::CpuGemmTranspose1xWKernel>   _transpose_kernel{ nullptr };
    std::unique_ptr<kernels::CpuGemmMatrixMultiplyKernel> _mm_kernel{ nullptr };
    std::unique_ptr<CpuGemmAssemblyDispatch>              _asm_glue{ nullptr };
    std::unique_ptr<kernels::CpuGemmMatrixAdditionKernel> _ma_kernel{ nullptr };
    std::unique_ptr<CpuActivation>                        _alpha_scale_func{ nullptr };
    std::unique_ptr<CpuAdd>                               _add_bias{ nullptr };
    std::unique_ptr<CpuActivation>                        _activation_func{ nullptr };

    TensorInfo _tmp_a{};
    TensorInfo _tmp_b{};
    TensorInfo _tmp_d{};

    bool _run_vector_matrix_multiplication{ false };
    bool _run_alpha_scale{ false };
    bool _run_addition{ false };
    bool _run_bias_addition{ false };
    bool _run_activation{ false };
    bool _reshape_b_only_on_first_run{ false };
    bool _is_prepared{ false };

    experimental::MemoryRequirements _aux_mem{ Count };
};
} // namespace cpu
} // namespace arm_compute
#endif /*ARM_COMPUTE_CPU_GEMM_H */
