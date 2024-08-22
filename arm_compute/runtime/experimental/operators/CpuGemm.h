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

#ifndef ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUGEMM_H
#define ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUGEMM_H

#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/function_info/GEMMInfo.h"
#include "arm_compute/runtime/IOperator.h"

/*
 * A shallow wrapper for arm_compute::cpu::CpuGemm.
 * Any new features should be added to arm_compute::cpu::CpuGemm and
 * arm_compute::experimental::op::CpuGemm should remain a shallow wrapper.
*/

namespace arm_compute
{
namespace experimental
{
namespace op
{
/** Wrapper class for CpuGemm. For information on the operators,
 * see "src/cpu/operators/CpuGemm.h"
*/
class CpuGemm : public IOperator
{
public:
    /** Constructor **/
    CpuGemm();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuGemm(const CpuGemm &) = delete;
    /** Prevent copy assignment */
    CpuGemm operator=(const CpuGemm &) = delete;
    /** Default move constructor */
    CpuGemm(CpuGemm &&) = default;
    /** Default move assignment */
    CpuGemm &operator=(CpuGemm &&) = default;
    /** Default destructor */
    ~CpuGemm() override;

    /** Configure operator for a given list of arguments
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |a            |b           |c         |d              |
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
     * @param[in]      a         First input tensor info (Matrix A or Vector A). Data type supported: BFLOAT16/F16/F32
     * @param[in]      b         Second input tensor info (Matrix B). Data type supported: same as @p a
     * @param[in]      c         Third input tensor info (Matrix C). It can be a nullptr if just the multiplication between @p a and @p b is needed. Data type supported: same as @p a
     * @param[out]     d         Output tensor info. Data type supported: same as @p a
     * @param[in]      alpha     Weight of the matrix product
     * @param[in]      beta      Weight of matrix C
     * @param[in, out] gemm_info (Optional) Specifies if the matrix A and/or matrix B have been reshaped and
     *                       if the reshape of matrix B should happen only for the first run
     */
    void configure(const ITensorInfo *a,
                   const ITensorInfo *b,
                   const ITensorInfo *c,
                   ITensorInfo       *d,
                   float              alpha,
                   float              beta,
                   const GEMMInfo    &gemm_info = GEMMInfo());

    /** Static function to check if given info will lead to a valid configuration of @ref CpuGemm.
     *
     * Similar to @ref CpuGemm::configure()
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

    /** Indicates whether or not there is an optimal assembly implementation that can be used to process the given parameters.
     *
     * This method has the same use of @ref
     * NEGEMMConvolutionLayer::has_opt_impl, with the only caveat that
     * the value of arm_compute::WeightFormat need to be passed via the
     * parameter gemm_info.
     */
    static Status has_opt_impl(arm_compute::WeightFormat &weight_format,
                               const ITensorInfo         *a,
                               const ITensorInfo         *b,
                               const ITensorInfo         *c,
                               const ITensorInfo         *d,
                               const GEMMInfo            &gemm_info = GEMMInfo());

    void                             run(ITensorPack &tensors) override;
    void                             prepare(ITensorPack &constants) override;
    experimental::MemoryRequirements workspace() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace op
} // namespace experimental
} // namespace arm_compute

#endif // ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUGEMM_H
