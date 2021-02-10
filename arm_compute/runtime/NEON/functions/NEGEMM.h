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
#ifndef ARM_COMPUTE_NEGEMM_H
#define ARM_COMPUTE_NEGEMM_H

#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/IWeightsManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEArithmeticAddition.h"
#include "arm_compute/runtime/Tensor.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class NEGEMMInterleave4x4Kernel;
class NEGEMMMatrixAdditionKernel;
class NEGEMMMatrixMultiplyKernel;
class NEGEMMTranspose1xWKernel;
class NEGEMMAssemblyDispatch;

/** Basic function to execute GEMM on Neon. This function calls the following Neon kernels:
 *
 * If optimized assembly is available:
 *  -# @ref NEGEMMAssemblyDispatch
 *  -# @ref NEActivationLayer (if alpha != 1.0)
 * Else:
 *  -# @ref NEGEMMInterleave4x4Kernel (if the output tensor is a matrix)
 *  -# @ref NEGEMMTranspose1xWKernel (if the output tensor is a matrix)
 *  -# @ref NEGEMMMatrixMultiplyKernel
 * In both cases:
 *  -# @ref NEGEMMMatrixAdditionKernel (if c != nullptr and beta != 0.0 and is not reshaped once)
 * Else:
 *  -# @ref NEArithmeticAddition (if c != nullptr and is reshaped once and not optimized assembly in place)
 *
 *  -# @ref NEActivationLayer (if activation is specified in GEMMInfo)
 */
class NEGEMM : public IFunction
{
public:
    /** Constructor */
    NEGEMM(std::shared_ptr<IMemoryManager> memory_manager = nullptr, IWeightsManager *weights_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMM(const NEGEMM &) = delete;
    /** Default move constructor */
    NEGEMM(NEGEMM &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMM &operator=(const NEGEMM &) = delete;
    /** Default move assignment operator */
    NEGEMM &operator=(NEGEMM &&) = default;
    /** Default destructor */
    ~NEGEMM();
    /** Initialise the kernel's inputs, output
     *
     * @note GEMM: General Matrix Multiply - [alpha * A * B + beta * C].
     * @note GEMM: The tensors a, b, c, d must have the same data type. You should not mix data types when calling this function.
     *
     * @param[in]  a         First input tensor  (Matrix A or Vector A). Data type supported: BFLOAT16/F16/F32
     * @param[in]  b         Second input tensor (Matrix B). Data type supported: same as @p a
     * @param[in]  c         Third input tensor  (Matrix C). It can be a nullptr if just the multiplication between @p a and @p b is needed. Data type supported: same as @p a
     * @param[out] d         Output tensor. Data type supported: same as @p a
     * @param[in]  alpha     Weight of the matrix product
     * @param[in]  beta      Weight of matrix C
     * @param[in]  gemm_info (Optional) Specifies if the matrix A and/or matrix B have been reshaped and
     *                       if the reshape of matrix B should happen only for the first run
     */
    void configure(const ITensor *a, const ITensor *b, const ITensor *c, ITensor *d, float alpha, float beta, const GEMMInfo &gemm_info = GEMMInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMM.
     *
     * @param[in]  a         First input tensor info  (Matrix or Vector A). Data types supported: BFLOAT16/F16/F32
     * @param[in]  b         Second input tensor info (Matrix B). Data type supported: same as @p a.
     * @param[in]  c         Third input tensor info  (Matrix C). It can be a nullptr if just the multiplication between @p a and @p b is needed. Data type supported: same as @p a.
     * @param[out] output    Output tensor info. Data type supported: same as @p a
     * @param[in]  alpha     Weight of the matrix product
     * @param[in]  beta      Weight of matrix C
     * @param[in]  gemm_info (Optional) Specifies if the matrix A and/or matrix B have been reshaped and
     *                       if the reshape of matrix B should happen only for the first run
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info = GEMMInfo());

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    MemoryGroup                                 _memory_group;
    IWeightsManager                            *_weights_manager;
    std::unique_ptr<NEGEMMInterleave4x4Kernel>  _interleave_kernel;
    std::unique_ptr<NEGEMMTranspose1xWKernel>   _transpose_kernel;
    std::unique_ptr<NEGEMMMatrixMultiplyKernel> _mm_kernel;
    std::unique_ptr<NEGEMMAssemblyDispatch>     _asm_glue;
    std::unique_ptr<NEGEMMMatrixAdditionKernel> _ma_kernel;
    NEActivationLayer                           _alpha_scale_func;
    NEArithmeticAddition                        _add_bias;
    NEActivationLayer                           _activation_func;

    Tensor         _tmp_a;
    Tensor         _tmp_b;
    Tensor         _tmp_d;
    const ITensor *_original_b;
    bool           _run_vector_matrix_multiplication;
    bool           _run_alpha_scale;
    bool           _run_addition;
    bool           _run_bias_addition;
    bool           _run_activation;
    bool           _reshape_b_only_on_first_run;
    bool           _is_prepared;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEGEMM_H */
