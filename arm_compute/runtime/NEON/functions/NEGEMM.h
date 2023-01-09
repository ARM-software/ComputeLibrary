/*
 * Copyright (c) 2017-2023 Arm Limited.
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

#include <memory>

namespace arm_compute
{
/** Basic function to execute GEMM. This function calls the following kernels:
 *
 *  -# @ref cpu::CpuGemm
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
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0         |src1        |src2      |dst            |
     * |:------------|:-----------|:---------|:--------------|
     * |F32          |F32         |F32       |F32            |
     * |F16          |F16         |F16       |F16            |
     * |BFLOAT16     |BFLOAT16    |BFLOAT16  |BFLOAT16       |
     *
     * @note GEMM: General Matrix Multiply - [alpha * A * B + beta * C].
     * @note GEMM: The tensors a, b, c, d must have the same data type. You should not mix data types when calling this function.
     *
     * @note Batched GEMM only supports broadcasting cases where RHS rank < LHS rank but not the other way around
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
     * Similar to @ref NEGEMM::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info = GEMMInfo());

    /** Static function that queries whether there exists fixed-format kernel and if it exists it will return in the first argument in what format
     * weights are expected to be reshaped as defined by WeightFormat class. Apart from the first argument the rest of the arguments are the same
     * as in @ref NEGEMM::validate() except that all arguments are required.
     *
     * @return a status
     */
    static Status has_opt_impl(arm_compute::WeightFormat &expected_weight_format, const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output,
                               float alpha, float beta, const GEMMInfo &gemm_info = GEMMInfo());

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEGEMM_H */
