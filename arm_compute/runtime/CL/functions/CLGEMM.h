/*
 * Copyright (c) 2016-2020 ARM Limited.
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

#include "arm_compute/core/CL/kernels/CLGEMMMatrixMultiplyKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMMatrixMultiplyReshapedKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMMatrixMultiplyReshapedOnlyRHSKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMReshapeLHSMatrixKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMReshapeRHSMatrixKernel.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTypes.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/IWeightsManager.h"
#include "arm_compute/runtime/MemoryGroup.h"

namespace arm_compute
{
class ICLTensor;

namespace weights_transformations
{
/** Basic function to manage the reshape weights generated from @ref CLGEMMReshapeRHSMatrixKernel */
class CLGEMMReshapeRHSMatrixKernelManaged : public ITransformWeights
{
public:
    //Inherited method override
    void run() override
    {
        _output.allocator()->allocate();
        CLScheduler::get().enqueue(_kernel, false);
        _reshape_run = true;
    }

    //Inherited method override
    void release() override
    {
        _output.allocator()->free();
    }

    //Inherited method override
    ICLTensor *get_weights() override
    {
        return &_output;
    }

    //Inherited method override
    uint32_t uid() override
    {
        return _uid;
    }

    /** Configures the @ref CLGEMMReshapeRHSMatrixKernel kernel
     *
     * @param[in] input Input tensor. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
     * @param[in] info  RHS matrix information to be used for reshaping.
     */
    void configure(const ICLTensor *input, GEMMRHSMatrixInfo info)
    {
        configure(CLKernelLibrary::get().get_compile_context(), input, info);
    }

    /** Configures the @ref CLGEMMReshapeRHSMatrixKernel kernel
     *
     * @param[in] compile_context The compile context to be used.
     * @param[in] input           Input tensor. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
     * @param[in] info            RHS matrix information to be used for reshaping.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, GEMMRHSMatrixInfo info)
    {
        _kernel.configure(compile_context, input, &_output, info);
    }

private:
    static constexpr uint32_t    _uid = 0x15;
    CLTensor                     _output{};
    CLGEMMReshapeRHSMatrixKernel _kernel{};
};
} // namespace weights_transformations

/** Basic function to execute GEMM on OpenCL. This function calls the following OpenCL kernels:
 *
 *  -# @ref CLGEMMReshapeLHSMatrixKernel (only if the RESHAPED_V1 is selected by the heuristic model)
 *  -# @ref CLGEMMReshapeRHSMatrixKernel (only if either the RESHAPED_V1 or RESHAPED_ONLY_RHS is selected by the select_gemm_kernel method())
 *  -# @ref CLGEMMMatrixMultiplyKernel (only if either the NATIVE or RESHAPED_V1 is selected by the select_gemm_kernel method())
 *  -# @ref CLGEMMMatrixMultiplyReshapedKernel (only if RESHAPED_V1 is selected by the select_gemm_kernel method())
 *  -# @ref CLGEMMMatrixMultiplyReshapedOnlyRHSKernel (only if RESHAPED_ONLY_RHS is selected by the select_gemm_kernel method())
 *
 */
class CLGEMM : public IFunction
{
public:
    /** Default constructor.
     *
     * @param[in] memory_manager  (Optional) Memory manager.
     * @param[in] weights_manager (Optional) Weights manager.
     */
    CLGEMM(std::shared_ptr<IMemoryManager> memory_manager = nullptr, IWeightsManager *weights_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMM(const CLGEMM &) = delete;
    /** Default move constructor */
    CLGEMM(CLGEMM &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMM &operator=(const CLGEMM &) = delete;
    /** Default move assignment operator */
    CLGEMM &operator=(CLGEMM &&) = default;
    /** Initialise the kernel's inputs and output
     *
     * @note GEMM: General Matrix Multiply - [alpha * A * B + beta * C].
     *
     * @note All tensors must have the same data type.
     *
     * @note Whilst the first input tensor can be a vector, the second input tensor must be at least a matrix
     *
     * @param[in]  a         First input tensor  (Matrix or Vector A). Data types supported: F16/F32
     * @param[in]  b         Second input tensor (Matrix B). Data type supported: same as @p a.
     * @param[in]  c         Third input tensor  (Matrix C). It can be a nullptr if just the multiplication between @p a and @p b is needed. Data type supported: same as @p a.
     * @param[out] output    Output tensor. Data type supported: same as @p a
     * @param[in]  alpha     Weight of the matrix product
     * @param[in]  beta      Weight of matrix C
     * @param[in]  gemm_info (Optional) Specifies if the matrix A and/or matrix B have been reshaped and
     *                       if the reshape of matrix B should happen only for the first run. GEMMInfo also contains information about the reshaping
     *                       in case matrix A and matrix B have been already transformed.
     */
    void configure(const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output, float alpha, float beta, const GEMMInfo &gemm_info = GEMMInfo());
    /** Initialise the kernel's inputs and output
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
    void configure(const CLCompileContext &compile_context, const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output, float alpha, float beta, const GEMMInfo &gemm_info = GEMMInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMM.
     *
     * @param[in] a         First input tensor info  (Matrix or Vector A). Data types supported: F16/F32
     * @param[in] b         Second input tensor info (Matrix B). Data type supported: same as @p a.
     * @param[in] c         Third input tensor info  (Matrix C). It can be a nullptr if just the multiplication between @p a and @p b is needed. Data type supported: same as @p a.
     * @param[in] output    Output tensor info. Data type supported: same as @p a
     * @param[in] alpha     Weight of the matrix product
     * @param[in] beta      Weight of matrix C
     * @param[in] gemm_info (Optional) Specifies if the matrix A and/or matrix B have been reshaped and
     *                       if the reshape of matrix B should happen only for the first run
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info = GEMMInfo());

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    static CLGEMMKernelType select_gemm_kernel(unsigned int m, unsigned int n, unsigned int k, DataType data_type, bool reshape_b_only_on_first_run);

    void configure_native_v1(const CLCompileContext &compile_context, const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output, float alpha, float beta, const GEMMInfo &gemm_info);
    void configure_reshaped_v1(const CLCompileContext &compile_context, const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output, float alpha, float beta, const GEMMInfo &gemm_info);
    void configure_reshaped_v2(const CLCompileContext &compile_context, const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output, float alpha, float beta, const GEMMInfo &gemm_info);
    void configure_reshaped_only_rhs(const CLCompileContext &compile_context, const ICLTensor *a, const ICLTensor *b, const ICLTensor *c, ICLTensor *output, float alpha, float beta,
                                     const GEMMInfo &gemm_info);

    static Status validate_native_v1(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info);
    static Status validate_reshaped_v1(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info);
    static Status validate_reshaped(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info);
    static Status validate_reshaped_only_rhs(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, float alpha, float beta, const GEMMInfo &gemm_info);

    MemoryGroup                                                  _memory_group;
    IWeightsManager                                             *_weights_manager;
    CLGEMMMatrixMultiplyKernel                                   _mm_kernel;
    CLGEMMReshapeLHSMatrixKernel                                 _reshape_lhs_kernel;
    CLGEMMReshapeRHSMatrixKernel                                 _reshape_rhs_kernel;
    weights_transformations::CLGEMMReshapeRHSMatrixKernelManaged _reshape_rhs_kernel_managed;
    CLGEMMMatrixMultiplyReshapedKernel                           _mm_reshaped_kernel;
    CLGEMMMatrixMultiplyReshapedOnlyRHSKernel                    _mm_reshaped_only_rhs_kernel;
    CLTensor                                                     _tmp_a;
    CLTensor                                                     _tmp_b;
    const ICLTensor                                             *_original_b;
    bool                                                         _reshape_b_only_on_first_run;
    bool                                                         _is_prepared;
    CLGEMMKernelType                                             _gemm_kernel_type;
};
} // namespace arm_compute

#endif /* ARM_COMPUTE_CLGEMM_H */
