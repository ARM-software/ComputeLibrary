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
#ifndef __ARM_ASSEMBLY_HELPER_H__
#define __ARM_ASSEMBLY_HELPER_H__

#include "arm_compute/core/ITensor.h"
#include "support/ToolchainSupport.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/Log.h"
#include "arm_compute/core/NEON/kernels/assembly/NEGEMMAssemblyWrapper.h"
#include "arm_compute/core/NEON/kernels/assembly/arm_gemm.hpp"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

namespace arm_compute
{
/** Assembly kernel glue */
template <typename TypeInput, typename TypeOutput>
class AssemblyKernelGlue final
{
public:
    /** Operator type */
    using TypeOperator = TypeInput;
    /** Result type */
    using TypeResult = TypeOutput;
    /** Default constructor. */
    AssemblyKernelGlue()
        : _gemm_kernel_asm(nullptr), _optimised_kernel(nullptr), _a(nullptr), _b(nullptr), _d(nullptr)
    {
    }
    /** Assembly Gemm */
    using AssemblyGemm = arm_gemm::GemmCommon<TypeInput, TypeOutput>;

    /** Prevent instances of this class from being copy constructed */
    const AssemblyKernelGlue<TypeInput, TypeOutput> &operator=(const AssemblyKernelGlue<TypeInput, TypeOutput> &) = delete;
    /** Prevent instances of this class from being copied */
    AssemblyKernelGlue(const AssemblyKernelGlue<TypeInput, TypeOutput> &) = delete;

    /** Assembly Gemm kernel */
    std::unique_ptr<AssemblyGemm> _gemm_kernel_asm;
    /** Optimised NEON kernel */
    std::unique_ptr<INEKernel> _optimised_kernel;
    /** Input A */
    const ITensor *_a;
    /** Input B */
    const ITensor *_b;
    /** Output */
    ITensor *_d;

    /** Configures the arrays pointers and strides in the assembly kernel and executes the assembly kernel.
     *  The call to set_arrays is needed to deal with the input sizes containing batches (dims > 2)
     */
    inline void run()
    {
        const int lda = _a->info()->strides_in_bytes().y() / sizeof(TypeInput);
        const int ldb = _b->info()->strides_in_bytes().y() / sizeof(TypeInput);
        const int ldd = _d->info()->strides_in_bytes().y() / sizeof(TypeOutput);

        // Configure kernel window
        Window     window  = calculate_max_window(*_d->info());
        const auto in1_ptr = reinterpret_cast<const TypeInput *>(_b->buffer());

        // Only iterate over batches
        Window win(window);
        win.set(0, Window::Dimension(0, 1, 1));
        win.set(1, Window::Dimension(0, 1, 1));
        Iterator in0(_a, window);
        Iterator out(_d, window);
        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto in0_ptr = reinterpret_cast<const TypeInput *>(in0.ptr());
            auto       out_ptr = reinterpret_cast<TypeOutput *>(out.ptr());
            _gemm_kernel_asm->set_arrays(in0_ptr, lda, in1_ptr, ldb, out_ptr, ldd);
            NEScheduler::get().schedule(_optimised_kernel.get(), Window::DimX);
        },
        in0, out);
    }
};

/** Float 32 assembly kernel glue */
using AssemblyKernelGlueF32 = AssemblyKernelGlue<float, float>;
/** Uint 8 to Uint 32 kernel glue */
using AssemblyKernelGlueU8U32 = AssemblyKernelGlue<uint8_t, uint32_t>;
/** Int 8 to Int 32 kernel glue */
using AssemblyKernelGlueS8S32 = AssemblyKernelGlue<int8_t, int32_t>;

/** Allocate a workspace tensor.
 *
 * @param[in]  workspace_size Size to allocate.
 * @param[out] workspace      Tensor to allocate.
 * @param[in]  memory_group   Tensor memory group.
 * @param[in]  alignment      Workspace memory alignment.
 * @param[in]  num_threads    Number of workspace threads.
 */
inline void allocate_workspace(size_t workspace_size, Tensor &workspace, MemoryGroup &memory_group, size_t alignment, unsigned int num_threads)
{
    ARM_COMPUTE_ERROR_ON_MSG(workspace_size == 0, "size cannot be 0");
    workspace.allocator()->init(TensorInfo(TensorShape{ (workspace_size + alignment - 1) * num_threads }, 1, DataType::S8));
    workspace.allocator()->allocate();
}

/** Create a wrapper kernel.
 *
 * @param[in]  a     Input tensor A.
 * @param[in]  b     Input tensor B.
 * @param[in]  c     (Optional) Input tensor C.
 * @param[out] d     Output tensor.
 * @param[in]  alpha Alpha value.
 * @param[in]  beta  Beta value.
 *
 * @return the wrapper kernel.
 */
template <typename T>
std::unique_ptr<NEGEMMAssemblyWrapper<T>> create_wrapper_kernel(const ITensor *a, const ITensor *b, const ITensor *c, ITensor *d, float alpha, float beta)
{
    // rework this function, why are we checking data type and other things here ? should we create another function can_run_optimised_kernel() ?
#if defined(__arm__)
    if(NEScheduler::get().cpu_info().CPU == CPUTarget::ARMV7 && a->info()->data_type() == DataType::F32 && (c == nullptr || beta == 0.f))
    {
        return support::cpp14::make_unique<NEGEMMAssemblyWrapper<T>>();
    }
#elif defined(__aarch64__)
    if(NEScheduler::get().cpu_info().CPU >= CPUTarget::ARMV8 && a->info()->data_type() == DataType::F32 && (c == nullptr || beta == 0.f))
    {
        return support::cpp14::make_unique<NEGEMMAssemblyWrapper<T>>();
    }
    else if(a->info()->data_type() == DataType::F16 && (c == nullptr || beta == 0.f))
    {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        return support::cpp14::make_unique<NEGEMMAssemblyWrapper<T>>();
#else  /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        ARM_COMPUTE_ERROR("Recompile the library with arch=arm64-v8.2-a to enable support for FP16.");
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    }
#endif /* defined(__arm__) || defined(__aarch64__) */
    return nullptr;
}

/** Setup assembly kernel.
 *
 * @param[in]  a            Input tensor A.
 * @param[in]  b            Input tensor B.
 * @param[in]  c            (Optional) Input tensor C.
 * @param[in]  d            Output tensor.
 * @param[in]  alpha        Alpha value.
 * @param[in]  beta         Beta value.
 * @param[out] workspace    Workspace tensor
 * @param[in]  memory_group Tensor memory group.
 * @param[out] asm_glue     Assembly glue kernel.
 *
 * @return True if the assembly kernel is setup correctly.
 */
template <typename T>
inline bool setup_assembly_kernel(const ITensor *a, const ITensor *b, const ITensor *c, ITensor *d, float alpha, float beta,
                                  Tensor &workspace, MemoryGroup &memory_group, T &asm_glue)
{
    const ::CPUInfo *ci          = get_CPUInfo();
    const int        M           = d->info()->tensor_shape().y();
    const int        N           = d->info()->tensor_shape().x();
    const int        K           = a->info()->tensor_shape().x();
    unsigned int     num_threads = NEScheduler::get().num_threads();
    // unique_ptr to a Gemm object
    std::unique_ptr<typename T::AssemblyGemm> asm_gemm(arm_gemm::gemm<typename T::TypeOperator, typename T::TypeResult>(*ci, M, N, K, false, false, alpha, beta, num_threads,
                                                                                                                        false));

    // arm_compute wrapper for the Gemm object (see above)
    std::unique_ptr<NEGEMMAssemblyWrapper<typename T::AssemblyGemm>> acl_gemm_wrapper = create_wrapper_kernel<typename T::AssemblyGemm>(a, b, c, d, alpha, beta);
    if(acl_gemm_wrapper != nullptr && asm_gemm != nullptr)
    {
        acl_gemm_wrapper->configure(asm_gemm.get());
        const size_t workspace_size = asm_gemm->get_working_size();
        if(workspace_size)
        {
            // Allocate workspace
            allocate_workspace(workspace_size, workspace, memory_group, 4096, num_threads);
            asm_gemm->set_working_space(reinterpret_cast<typename T::TypeResult *>(workspace.buffer()));
        }
        const unsigned int window_size = asm_gemm->get_window_size();
        if(window_size < num_threads)
        {
            num_threads = window_size;
            asm_gemm->set_nthreads(num_threads);
        }
        asm_glue._gemm_kernel_asm  = std::move(asm_gemm);
        asm_glue._optimised_kernel = std::move(acl_gemm_wrapper);
        // We need to setup the ptrs in the run() method
        asm_glue._a = a;
        asm_glue._b = b;
        asm_glue._d = d;
        return true;
    }
    return false;
}
}
#endif /* __ARM_ASSEMBLY_HELPER_H__ */
