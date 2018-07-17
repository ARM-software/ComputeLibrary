/*
 * Copyright (c) 2018 ARM Limited.
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
template <typename TypeInput, typename TypeOutput>
class NEGEMMAssemblyDispatch : public IFunction
{
public:
    /** Default constructor */
    NEGEMMAssemblyDispatch(std::shared_ptr<IMemoryManager> memory_manager = nullptr);

    /** Prevent instances of this class from being copy constructed */
    NEGEMMAssemblyDispatch(const NEGEMMAssemblyDispatch<TypeInput, TypeOutput> &) = delete;
    /** Prevent instances of this class from being copied */
    NEGEMMAssemblyDispatch<TypeInput, TypeOutput> &operator=(const NEGEMMAssemblyDispatch<TypeInput, TypeOutput> &) = delete;
    NEGEMMAssemblyDispatch(NEGEMMAssemblyDispatch<TypeInput, TypeOutput> &&) = default;
    NEGEMMAssemblyDispatch<TypeInput, TypeOutput> &operator=(NEGEMMAssemblyDispatch<TypeInput, TypeOutput> &&) = default;
    ~NEGEMMAssemblyDispatch() = default;

private:
    /** ACL Function */
    std::unique_ptr<IFunction> _function;

    /** If supported create the ACL function corresponding to the GemmMethod provided to process the other passed parameters
     *
     * @param[in]  method             GemmMethod to use to perform the matrix multiplication.
     * @param[in]  a                  Input tensor (Matrix A).
     * @param[in]  b                  Input tensor (Matrix B).
     * @param[out] d                  Output tensor to store the result of matrix multiplication. Data type supported: same as @p input0.
     * @param[in]  alpha              Scalar multiplier to apply to AB matrix product.
     * @param[in]  beta               Scalar multiplier to apply to input D matrix before adding product.
     * @param[in]  pretransposed_hint Can the B tensor can be pretransposed (ie shared across invocations)?
     *
     * @return True if the method is supported and the function was successfully created, false otherwise.
     */
    bool create_function(arm_gemm::GemmMethod method, const ITensor *a, const ITensor *b, ITensor *d, float alpha, float beta, bool pretranspose_hint);

    //Fallback: use arm_gemm's AssemblyGemm:
    class Fallback
    {
#ifndef DOXYGEN_SKIP_THIS
    public:
        /** Configures the arrays pointers and strides in the assembly kernel and executes the assembly kernel.
         *  The call to set_arrays is needed to deal with the input sizes containing batches (dims > 2)
         */
        void run();
        void configure(const ITensor *a, const ITensor *b, ITensor *d, arm_gemm::GemmArgs<TypeOutput> &args, MemoryGroup &memory_group);
        void prepare();
        bool is_configured() const;
#endif /* DOXYGEN_SKIP_THIS */

    private:
        /** Allocate a workspace tensor.
         *
         * @param[in] workspace_size Size to allocate.
         * @param[in] memory_group   Tensor memory group.
         * @param[in] alignment      Workspace memory alignment.
         */
        void allocate_workspace(size_t workspace_size, MemoryGroup *memory_group, size_t alignment);

        /** Assembly Gemm kernel */
        std::unique_ptr<arm_gemm::GemmCommon<TypeInput, TypeOutput>> _gemm_kernel_asm{ nullptr };
        /** Optimised NEON kernel */
        std::unique_ptr<INEKernel> _optimised_kernel{ nullptr };
        /** Input A */
        const ITensor *_a
        {
            nullptr
        };
        /** Input B */
        const ITensor *_b
        {
            nullptr
        };
        /** Output */
        ITensor *_d{ nullptr };
        /** GEMM workspace */
        Tensor _workspace{};
        /** Pre-transpose tensor */
        Tensor _pretranspose{};
        /** Prepared flag */
        bool _is_prepared{ false };
    } _arm_gemm;               /**< Fallback in case ACL doesn't have a function */
    MemoryGroup _memory_group; /**< Function memory group */
public:
    /** If supported create an ACL function else fallback to the arm_gemm function.
     *
     * @param[in]  a                 Input tensor (Matrix A)
     * @param[in]  b                 Input tensor (Matrix B)
     * @param[out] d                 Output tensor to store the result of matrix multiplication. Data type supported: same as @p input0.
     * @param[in]  alpha             Scalar multiplier to apply to AB matrix product.
     * @param[in]  beta              Scalar multiplier to apply to input D matrix before adding product.
     * @param[in]  pretranspose_hint Can the B tensor can be pretransposed (ie shared across invocations)?
     */
    void configure(const ITensor *a, const ITensor *b, ITensor *d, float alpha, float beta, bool pretranspose_hint);
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

/** Float 32 assembly dispatch kernel */
using NEGEMMAssemblyDispatchF32 = NEGEMMAssemblyDispatch<float, float>;
/** Uint 8 to Uint 32 assembly dispatch kernel */
using NEGEMMAssemblyDispatchU8U32 = NEGEMMAssemblyDispatch<uint8_t, uint32_t>;
/** Int 8 to Int 32 assembly dispatch kernel */
using NEGEMMAssemblyDispatchS8S32 = NEGEMMAssemblyDispatch<int8_t, int32_t>;
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEGEMMASSEMBLYDISPATCH_H__ */
