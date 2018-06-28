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
        : _gemm_kernel_asm(nullptr), _optimised_kernel(nullptr), _a(nullptr), _b(nullptr), _d(nullptr), _workspace(nullptr), _pretranspose(nullptr), _is_prepared(false)
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
    /** GEMM workspace */
    ITensor *_workspace;
    /** Pre-transpose tensor */
    ITensor *_pretranspose;
    /** Prepared flag */
    bool _is_prepared;

    /** Runs a preparation step, usually for pre-transposing matrix b */
    void prepare()
    {
        // Pretranspose B if required
        if(_gemm_kernel_asm->B_pretranspose_required())
        {
            const int  ldb            = _b->info()->strides_in_bytes().y() / sizeof(TypeInput);
            const auto in1_ptr        = reinterpret_cast<const TypeInput *>(_b->buffer());
            const int  multi_stride_b = _b->info()->strides_in_bytes().z() / sizeof(TypeInput);

            ARM_COMPUTE_ERROR_ON(_pretranspose == nullptr || _pretranspose->buffer() == nullptr);
            _gemm_kernel_asm->pretranspose_B_array(_pretranspose->buffer(), in1_ptr, ldb, multi_stride_b);
            _b->mark_as_unused();
        }

        _is_prepared = true;
    }

    /** Configures the arrays pointers and strides in the assembly kernel and executes the assembly kernel.
     *  The call to set_arrays is needed to deal with the input sizes containing batches (dims > 2)
     */
    inline void run()
    {
        const int lda = _a->info()->strides_in_bytes().y() / sizeof(TypeInput);
        const int ldb = _b->info()->strides_in_bytes().y() / sizeof(TypeInput);
        const int ldd = _d->info()->strides_in_bytes().y() / sizeof(TypeOutput);

        // In the case of NHWC we want to interpret the output shape as 3D. Thus, the batch stride for A is
        // the relevant multiple of the row stride.
        const bool is_nhwc           = _a->info()->data_layout() == DataLayout::NHWC;
        const int  stride_in_bytes_a = is_nhwc ? _a->info()->strides_in_bytes().y() * _d->info()->dimension(1) : _a->info()->strides_in_bytes().z();

        const int batch_stride_a = stride_in_bytes_a / sizeof(TypeInput);
        const int batch_stride_d = _d->info()->strides_in_bytes().z() / sizeof(TypeOutput);

        const int multi_stride_a = _a->info()->strides_in_bytes()[3] / sizeof(TypeInput);
        const int multi_stride_b = _b->info()->strides_in_bytes().z() / sizeof(TypeInput);
        const int multi_stride_d = _d->info()->strides_in_bytes()[3] / sizeof(TypeOutput);

        const auto in0_ptr = reinterpret_cast<const TypeInput *>(_a->buffer());
        const auto in1_ptr = reinterpret_cast<const TypeInput *>(_b->buffer());
        auto       out_ptr = reinterpret_cast<TypeOutput *>(_d->buffer());

        // Set workspace if needed and reset number of threads as buffer manager gets re-created with max_threads
        if(_workspace != nullptr)
        {
            _gemm_kernel_asm->set_working_space(reinterpret_cast<void *>(_workspace->buffer()));
            const unsigned int window_size = _gemm_kernel_asm->get_window_size();
            unsigned int       num_threads = NEScheduler::get().num_threads();
            if(window_size < num_threads)
            {
                num_threads = window_size;
                _gemm_kernel_asm->set_nthreads(num_threads);
            }
        }

        // Prepare assembly kernel
        prepare();

        // Set gemm parameters
        _gemm_kernel_asm->set_arrays(in0_ptr, lda, batch_stride_a, multi_stride_a, in1_ptr, ldb, multi_stride_b, out_ptr, ldd, batch_stride_d, multi_stride_d);

        // Schedule assembly kernel
        NEScheduler::get().schedule(_optimised_kernel.get(), Window::DimX);
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
 */
inline void allocate_workspace(size_t workspace_size, Tensor &workspace, MemoryGroup *memory_group, size_t alignment)
{
    ARM_COMPUTE_ERROR_ON_MSG(workspace_size == 0, "size cannot be 0");
    workspace.allocator()->init(TensorInfo(TensorShape{ (workspace_size + alignment /* FIXME: remove alignment after COMPMID-1088 */) }, 1, DataType::S8), alignment);
    if(memory_group != nullptr)
    {
        memory_group->manage(&workspace);
    }
    workspace.allocator()->allocate();
}

/** Create a wrapper kernel.
 *
 * @param[in]  a                 Input tensor A.
 * @param[in]  b                 Input tensor B.
 * @param[out] d                 Output tensor.
 * @param[in]  alpha             Alpha value.
 * @param[in]  beta              Beta value.
 * @param[in]  pretranspose_hint Pre-transpose hint in case matrix b should be pre-transposed
 * @param[out] workspace         Workspace tensor
 * @param[out] B_pretranspose    Tensor to hold the pre-transposed B
 * @param[in]  memory_group      Tensor memory group.
 * @param[out] asm_glue          Assembly glue kernel.
 *
 * @return the wrapper kernel.
 */
template <typename T>
inline bool setup_assembly_kernel(const ITensor *a, const ITensor *b, ITensor *d, float alpha, float beta, bool pretranspose_hint,
                                  Tensor &workspace, Tensor &B_pretranspose, MemoryGroup &memory_group, T &asm_glue)
{
    const CPUInfo &ci          = NEScheduler::get().cpu_info();
    const int      M           = d->info()->tensor_shape().y();
    const int      N           = d->info()->tensor_shape().x();
    const int      K           = a->info()->tensor_shape().x();
    const int      batches     = d->info()->tensor_shape().total_size_upper(2);
    const int      multis      = b->info()->tensor_shape().z();
    unsigned int   num_threads = NEScheduler::get().num_threads();

    // unique_ptr to a Gemm object
    std::unique_ptr<typename T::AssemblyGemm>
    asm_gemm(arm_gemm::gemm<typename T::TypeOperator, typename T::TypeResult>(ci, M, N, K, batches, multis, false, false, alpha, beta, num_threads, pretranspose_hint));
    // arm_compute wrapper for the Gemm object (see above)
    std::unique_ptr<NEGEMMAssemblyWrapper<typename T::AssemblyGemm>>
                                                                  acl_gemm_wrapper = support::cpp14::make_unique<NEGEMMAssemblyWrapper<typename T::AssemblyGemm>>();
    if(acl_gemm_wrapper != nullptr && asm_gemm != nullptr)
    {
        acl_gemm_wrapper->configure(asm_gemm.get());
        const size_t workspace_size = asm_gemm->get_working_size();
        if(workspace_size)
        {
            // Allocate workspace
            const unsigned int alignment = 4096;
            allocate_workspace(workspace_size, workspace, &memory_group, alignment);
            asm_glue._workspace = &workspace;
        }

        //if we disable this code below in brackets then ConvLayer deadlocks when threads > 1 and
        //the shapes are In=1x1x1024 Weights=1x1x1024x1001 Biases=1001 Out=1x1x1001
        {
            const unsigned int window_size = asm_gemm->get_window_size();
            if(window_size < num_threads)
            {
                num_threads = window_size;
                asm_gemm->set_nthreads(num_threads);
            }
        }

        // Check for pre-transposed support
        if(asm_gemm->B_pretranspose_required())
        {
            // Forcing 128-byte alignment (required by 32-bit kernels)
            const unsigned int alignment           = 128;
            const size_t       B_pretranspose_size = asm_gemm->get_B_pretransposed_array_size();
            allocate_workspace(B_pretranspose_size, B_pretranspose, nullptr, alignment);
            ARM_COMPUTE_ERROR_ON_NULLPTR(B_pretranspose.buffer());
            asm_glue._pretranspose = &B_pretranspose;
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
