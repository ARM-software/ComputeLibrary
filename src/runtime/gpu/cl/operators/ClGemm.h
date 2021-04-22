/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CL_GEMM_H
#define ARM_COMPUTE_CL_GEMM_H

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTypes.h"
#include "src/core/gpu/cl/ClCompileContext.h"
#include "src/core/gpu/cl/IClKernel.h"
#include "src/core/gpu/cl/kernels/ClGemmMatrixMultiplyKernel.h"
#include "src/core/gpu/cl/kernels/ClGemmMatrixMultiplyNativeKernel.h"
#include "src/core/gpu/cl/kernels/ClGemmMatrixMultiplyReshapedKernel.h"
#include "src/core/gpu/cl/kernels/ClGemmMatrixMultiplyReshapedOnlyRhsKernel.h"
#include "src/core/gpu/cl/kernels/ClGemmReshapeLhsMatrixKernel.h"
#include "src/core/gpu/cl/kernels/ClGemmReshapeRhsMatrixKernel.h"
#include "src/runtime/gpu/cl/IClOperator.h"

#include <memory>

namespace arm_compute
{
namespace opencl
{
/** Basic function to execute GEMM on OpenCL. This function calls the following OpenCL kernels:
 *
 *  -# @ref kernels::ClGemmReshapeLhsMatrixKernel (only if the RESHAPED_V1 is selected by the heuristic model)
 *  -# @ref kernels::ClGemmReshapeRhsMatrixKernel (only if either the RESHAPED_V1 or RESHAPED_ONLY_RHS is selected by the select_gemm_kernel method())
 *  -# @ref kernels::ClGemmMatrixMultiplyKernel (only if either the NATIVE or RESHAPED_V1 is selected by the select_gemm_kernel method())
 *  -# @ref kernels::ClGemmMatrixMultiplyReshapedKernel (only if RESHAPED_V1 is selected by the select_gemm_kernel method())
 *  -# @ref kernels::ClGemmMatrixMultiplyReshapedOnlyRhsKernel (only if RESHAPED_ONLY_RHS is selected by the select_gemm_kernel method())
 */
class ClGemm : public IClOperator
{
public:
    /** Constructor */
    ClGemm();
    /** Initialise the kernel's inputs and output
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0         |src1        |src2      |dst            |
     * |:------------|:-----------|:---------|:--------------|
     * |F32          |F32         |F32       |F32            |
     * |F16          |F16         |F16       |F16            |
     *
     * @note GEMM: General Matrix Multiply - [alpha * A * B + beta * C].
     *
     * @note All tensors must have the same data type.
     *
     * @note Whilst the first input tensor can be a vector, the second input tensor must be at least a matrix
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  a               First input tensor  (Matrix or Vector A). Data types supported: F16/F32
     * @param[in]  b               Second input tensor (Matrix B). Data type supported: same as @p a.
     * @param[in]  c               Third input tensor  (Matrix C). It can be a nullptr if just the multiplication between @p a and @p b is needed. Data type supported: same as @p a.
     * @param[out] output          Output tensor. Data type supported: same as @p a
     * @param[in]  alpha           Weight of the matrix product
     * @param[in]  beta            Weight of matrix C
     * @param[in]  gemm_info       (Optional) Specifies if the matrix A and/or matrix B have been reshaped and
     *                       if the reshape of matrix B should happen only for the first run. GEMMInfo also contains information about the reshaping
     *                       in case matrix A and matrix B have been already transformed.
     */
    void configure(const CLCompileContext &compile_context, ITensorInfo *a, ITensorInfo *b, ITensorInfo *c, ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to ClGemm::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
    void prepare(ITensorPack &constants) override;
    experimental::MemoryRequirements workspace() const override;

private:
    void configure_native_v1(const CLCompileContext &compile_context, ITensorInfo *a, ITensorInfo *b, ITensorInfo *c, ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info);
    void configure_reshaped_v1(const CLCompileContext &compile_context, ITensorInfo *a, ITensorInfo *b, ITensorInfo *c, ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info);
    void configure_reshaped_v2(const CLCompileContext &compile_context, ITensorInfo *a, ITensorInfo *b, ITensorInfo *c, ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info);
    void configure_reshaped_only_rhs(const CLCompileContext &compile_context, ITensorInfo *a, ITensorInfo *b, ITensorInfo *c, ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info);

    static Status validate_native_v1(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info);
    static Status validate_reshaped_v1(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info);
    static Status validate_reshaped(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info);
    static Status validate_reshaped_only_rhs(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info);

private:
    enum AuxTensorIdx
    {
        LhsReshape = 0,
        RhsReshape,
        Count
    };

private:
    std::unique_ptr<kernels::ClGemmMatrixMultiplyKernel>                _mm_kernel;
    std::unique_ptr<kernels::ClGemmReshapeLhsMatrixKernel>              _reshape_lhs_kernel;
    std::unique_ptr<kernels::ClGemmReshapeRhsMatrixKernel>              _reshape_rhs_kernel;
    std::unique_ptr<kernels::ClGemmMatrixMultiplyReshapedKernel>        _mm_reshaped_kernel;
    std::unique_ptr<kernels::ClGemmMatrixMultiplyReshapedOnlyRhsKernel> _mm_reshaped_only_rhs_kernel;
    std::unique_ptr<kernels::ClGemmMatrixMultiplyReshapedOnlyRhsKernel> _mm_reshaped_only_rhs_fallback_kernel;
    TensorInfo                                                          _tmp_a;
    TensorInfo                                                          _tmp_b;
    bool                                                                _reshape_b_only_on_first_run;
    CLGEMMKernelType                                                    _gemm_kernel_type;

    experimental::MemoryRequirements _aux_mem{};
};
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLGEMM_H */
