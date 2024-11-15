/*
 * Copyright (c) 2024 Arm Limited.
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
#ifndef ACL_SRC_CPU_OPERATORS_CPUDYNAMICGEMM_H
#define ACL_SRC_CPU_OPERATORS_CPUDYNAMICGEMM_H

#include "arm_compute/core/TensorInfo.h"

#include "src/cpu/ICpuOperator.h"
#include "src/cpu/kernels/CpuDynamicGemmKernel.h"

namespace arm_compute
{
namespace cpu
{
/** Basic function to execute dynamic GEMM. This function calls the following kernels:
 *
 *  -# @ref cpu::kernels::CpuDynamicGemmKernel
 */
class CpuDynamicGemm : public ICpuOperator
{
public:
    /** Default constructor */
    CpuDynamicGemm() = default;
    /** Default destructor */
    ~CpuDynamicGemm() = default;
    /** Configure operator for a given list of arguments
     *
     * @note GEMM: General Matrix Multiply - [alpha * A * B + beta * C].
     * @note GEMM: The tensors a, b, c, d must have the same data type. You should not mix data types when calling this function.
     *
     * @param[in]  a         First input tensor info (Matrix A or Vector A). Data type supported: F32
     * @param[in]  b         Second input tensor info (Matrix B). Data type supported: same as @p a
     * @param[in]  c         Third input tensor info (Matrix C). It can be a nullptr if just the multiplication between @p a and @p b is needed. Data type supported: same as @p a
     * @param[out] d         Output tensor info. Data type supported: same as @p a
     * @param[in]  alpha     Weight of the matrix product
     * @param[in]  beta      Weight of matrix C
     * @param[in]  gemm_info (Optional) Specifies if the matrix A and/or matrix B have been reshaped and
     *                       if the reshape of matrix B should happen only for the first run
     */
    void configure(const ITensorInfo *a,
                   const ITensorInfo *b,
                   const ITensorInfo *c,
                   ITensorInfo       *d,
                   float              alpha,
                   float              beta,
                   const GEMMInfo    &gemm_info = GEMMInfo());

    /** Static function to check if given info will lead to a valid configuration of @ref CpuDynamicGemm.
     *
     * Similar to @ref CpuDynamicGemm::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *a,
                           const ITensorInfo *b,
                           const ITensorInfo *c,
                           const ITensorInfo *d,
                           float              alpha,
                           float              beta,
                           const GEMMInfo    &gemm_info = GEMMInfo());

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;

private:
    std::unique_ptr<kernels::CpuDynamicGemmKernel> _kernel{nullptr};
};
} // namespace cpu
} // namespace arm_compute
#endif // ACL_SRC_CPU_OPERATORS_CPUDYNAMICGEMM_H
