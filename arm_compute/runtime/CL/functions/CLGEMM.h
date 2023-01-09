/*
 * Copyright (c) 2016-2021, 2023 Arm Limited.
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
#ifndef ARM_COMPUTE_CLGEMM_H
#define ARM_COMPUTE_CLGEMM_H

#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTypes.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/IWeightsManager.h"
#include "arm_compute/runtime/MemoryGroup.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

/** Basic function to execute GEMM on OpenCL */
class CLGEMM : public IFunction
{
public:
    /** Default constructor.
     *
     * @param[in] memory_manager  (Optional) Memory manager.
     * @param[in] weights_manager (Optional) Weights manager.
     */
    CLGEMM(std::shared_ptr<IMemoryManager> memory_manager = nullptr, IWeightsManager *weights_manager = nullptr);
    /** Default destructor */
    ~CLGEMM();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMM(const CLGEMM &) = delete;
    /** Default move constructor */
    CLGEMM(CLGEMM &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMM &operator=(const CLGEMM &) = delete;
    /** Default move assignment operator */
    CLGEMM &operator=(CLGEMM &&);
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
     * @note Batched GEMM only allows RHS tensor's rank to be <= 3
     * @note Batched GEMM only supports broadcasting cases where RHS rank < LHS rank but not the other way around
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  a               First input tensor  (Matrix or Vector A). Data types supported: F16/F32
     * @param[in]  b               Second input tensor (Matrix B). Data type supported: same as @p a.
     * @param[in]  c               Third input tensor  (Matrix C). It can be a nullptr if just the multiplication between @p a and @p b is needed. Data type supported: same as @p a.
     * @param[out] output          Output tensor. Data type supported: same as @p a
     * @param[in]  alpha           Weight of the matrix product
     * @param[in]  beta            Weight of matrix C
     * @param[in]  gemm_info       (Optional) Specifies if the matrix A and/or matrix B have been reshaped and
     *                             if the reshape of matrix B should happen only for the first run. GEMMInfo also contains information about the reshaping
     *                             in case matrix A and matrix B have been already transformed.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output, float alpha, float beta, const GEMMInfo &gemm_info = GEMMInfo());

    /** Initialise the kernel's inputs and output
     *
     * Similar to @ref CLGEMM::configure()
     */
    void configure(const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output, float alpha, float beta, const GEMMInfo &gemm_info = GEMMInfo());

    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMM.
     *
     * Similar to @ref CLGEMM::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info = GEMMInfo());

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute

#endif /* ARM_COMPUTE_CLGEMM_H */
