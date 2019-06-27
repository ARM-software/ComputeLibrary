/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEGEMMASSEMBLYDISPATCH_H__
#define __ARM_COMPUTE_NEGEMMASSEMBLYDISPATCH_H__

#include "arm_compute/core/NEON/kernels/assembly/NEGEMMAssemblyWrapperKernel.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"

#include "arm_compute/core/NEON/kernels/assembly/arm_gemm.hpp"

namespace arm_compute
{
/** Assembly kernel glue */
class NEGEMMAssemblyDispatch : public IFunction
{
public:
    /** Default constructor */
    NEGEMMAssemblyDispatch(std::shared_ptr<IMemoryManager> memory_manager = nullptr);

    /** Prevent instances of this class from being copy constructed */
    NEGEMMAssemblyDispatch(const NEGEMMAssemblyDispatch &) = delete;
    /** Prevent instances of this class from being copied */
    NEGEMMAssemblyDispatch &operator=(const NEGEMMAssemblyDispatch &) = delete;
    NEGEMMAssemblyDispatch(NEGEMMAssemblyDispatch &&)                 = default;
    NEGEMMAssemblyDispatch &operator=(NEGEMMAssemblyDispatch &&) = default;
    ~NEGEMMAssemblyDispatch()                                    = default;

    class IFallback
    {
    public:
        virtual void run()                 = 0;
        virtual void prepare()             = 0;
        virtual bool is_configured() const = 0;
        virtual ~IFallback()               = default;
    };

private:
    /** ACL Function */
    std::unique_ptr<IFunction> _function;

    /** If supported create the ACL function corresponding to the GemmMethod provided to process the other passed parameters
     *
     * @param[in]  method    GemmMethod to use to perform the matrix multiplication.
     * @param[in]  a         Input tensor (Matrix A).
     * @param[in]  b         Input tensor (Matrix B).
     * @param[in]  c         Input tensor (Matrix C) used to pass the bias for quantized calculations
     * @param[out] d         Output tensor to store the result of matrix multiplication. Data type supported: same as @p input0.
     * @param[in]  alpha     Scalar multiplier to apply to AB matrix product.
     * @param[in]  beta      Scalar multiplier to apply to input D matrix before adding product.
     * @param[in]  gemm_info GEMM meta-data
     *
     * @return True if the method is supported and the function was successfully created, false otherwise.
     */
    bool create_function(arm_gemm::GemmMethod method, const ITensor *a, const ITensor *b, const ITensor *c, ITensor *d, float alpha, float beta, const GEMMInfo &gemm_info);

    /** Interface for the arm_gemm fallback */
    std::unique_ptr<IFallback>      _arm_gemm;
    MemoryGroup                     _memory_group;   /**< Function memory group */
    std::shared_ptr<IMemoryManager> _memory_manager; /**< Copy of the memory manager used to create the memory group to be used when instantiating new functions */
public:
    /** If supported create an ACL function else fallback to the arm_gemm function.
     *
     * @param[in]  a         Input tensor (Matrix A)
     * @param[in]  b         Input tensor (Matrix B)
     * @param[in]  c         Input tensor (Matrix C) used to pass the bias for quantized calculations
     * @param[out] d         Output tensor to store the result of matrix multiplication. Data type supported: same as @p input0.
     * @param[in]  alpha     Scalar multiplier to apply to AB matrix product.
     * @param[in]  beta      Scalar multiplier to apply to input D matrix before adding product.
     * @param[in]  gemm_info GEMM meta-data
     */
    void configure(const ITensor *a, const ITensor *b, const ITensor *c, ITensor *d, float alpha, float beta, const GEMMInfo &gemm_info);

    /** Indicates whether or not this function can be used to process the given parameters.
     *
     * @param[in] a         Input tensor info (Matrix A)
     * @param[in] b         Input tensor info (Matrix B)
     * @param[in] c         Input tensor info (Matrix C) used to pass the bias for quantized calculations
     * @param[in] d         Output tensor to store the result of matrix multiplication. Data type supported: same as @p input0.
     * @param[in] alpha     Scalar multiplier to apply to AB matrix product.
     * @param[in] beta      Scalar multiplier to apply to input D matrix before adding product.
     * @param[in] gemm_info GEMM meta-data
     *
     * @return a status.
     */
    static Status validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *d, float alpha, float beta, const GEMMInfo &gemm_info);
    /** Was the function successfully configured ?
     *
     * @return True if the function is configured and ready to run
     */
    bool is_configured() const;
    // Inherited methods overridden:
    /** Runs a preparation step, usually for pre-transposing matrix b */
    void prepare() override;
    void run() override;
};

} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEGEMMASSEMBLYDISPATCH_H__ */
