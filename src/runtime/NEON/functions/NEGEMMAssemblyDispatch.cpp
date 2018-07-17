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
#include "arm_compute/runtime/NEON/functions/NEGEMMAssemblyDispatch.h"

#include "arm_compute/core/NEON/kernels/assembly/NEGEMMNativeWrapperKernel.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/NEON/functions/NESimpleAssemblyFunction.h"

namespace arm_compute
{
template <typename TypeInput, typename TypeOutput>
NEGEMMAssemblyDispatch<TypeInput, TypeOutput>::NEGEMMAssemblyDispatch(std::shared_ptr<IMemoryManager> memory_manager)
    : _function(nullptr), _arm_gemm(), _memory_group(std::move(memory_manager))
{
}

template <>
bool NEGEMMAssemblyDispatch<float, float>::create_function(arm_gemm::GemmMethod method, const ITensor *a, const ITensor *b, ITensor *d, float alpha, float beta, bool pretranspose_hint)
{
    ARM_COMPUTE_UNUSED(method);
    ARM_COMPUTE_UNUSED(a);
    ARM_COMPUTE_UNUSED(b);
    ARM_COMPUTE_UNUSED(d);
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_UNUSED(beta);
    ARM_COMPUTE_UNUSED(pretranspose_hint);
    switch(method)
    {
#ifdef __aarch64__
        case arm_gemm::GemmMethod::GEMM_NATIVE:
        {
            auto kernel = support::cpp14::make_unique<NEGEMMNativeWrapperKernel<float, float>>();
            kernel->configure(a, b, d, alpha, beta);
            auto function = support::cpp14::make_unique<NESimpleAssemblyFunction>();
            function->configure(std::move(kernel));
            _function = std::move(function);
            return true;
        }
#endif /* __aarch64__ */
        default:
            return false;
    }
}

template <typename TypeInput, typename TypeOutput>
bool NEGEMMAssemblyDispatch<TypeInput, TypeOutput>::create_function(arm_gemm::GemmMethod method, const ITensor *a, const ITensor *b, ITensor *d, float alpha, float beta, bool pretranspose_hint)
{
    ARM_COMPUTE_UNUSED(method);
    ARM_COMPUTE_UNUSED(a);
    ARM_COMPUTE_UNUSED(b);
    ARM_COMPUTE_UNUSED(d);
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_UNUSED(beta);
    ARM_COMPUTE_UNUSED(pretranspose_hint);
    return false;
}

template <typename TypeInput, typename TypeOutput>
void NEGEMMAssemblyDispatch<TypeInput, TypeOutput>::configure(const ITensor *a, const ITensor *b, ITensor *d, float alpha, float beta, bool pretranspose_hint)
{
    INEGEMMWrapperKernel::Params p           = INEGEMMWrapperKernel::extract_parameters(a, b, d);
    const CPUInfo               &ci          = NEScheduler::get().cpu_info();
    unsigned int                 num_threads = NEScheduler::get().num_threads();

    arm_gemm::GemmArgs<TypeOutput> args(&ci, p.M, p.N, p.K, p.batches, p.multis, false, false, alpha, beta, num_threads, pretranspose_hint);

    //Try to create an ACL function:
    if(!create_function(arm_gemm::get_gemm_method<TypeInput, TypeOutput>(args), a, b, d, alpha, beta, pretranspose_hint))
    {
        //Fallback onto arm_gemm function if ACL doesn't support this method.
        _arm_gemm.configure(a, b, d, args, _memory_group);
    }
}

template <typename TypeInput, typename TypeOutput>
void NEGEMMAssemblyDispatch<TypeInput, TypeOutput>::prepare()
{
    if(_function != nullptr)
    {
        _function->prepare();
    }
    else
    {
        _arm_gemm.prepare();
    }
}

template <typename TypeInput, typename TypeOutput>
bool NEGEMMAssemblyDispatch<TypeInput, TypeOutput>::is_configured() const
{
    return _arm_gemm.is_configured() || _function != nullptr;
}

template <typename TypeInput, typename TypeOutput>
void NEGEMMAssemblyDispatch<TypeInput, TypeOutput>::run()
{
    _memory_group.acquire();
    if(_function != nullptr)
    {
        _function->run();
    }
    else
    {
        _arm_gemm.run();
    }
    _memory_group.release();
}

#ifndef __aarch64__
template <>
void NEGEMMAssemblyDispatch<uint8_t, uint32_t>::configure(const ITensor *a, const ITensor *b, ITensor *d, float alpha, float beta, bool pretranspose_hint)
{
    // arm_gemm::gemm for 8bit only exists for aarch64
    ARM_COMPUTE_UNUSED(a);
    ARM_COMPUTE_UNUSED(b);
    ARM_COMPUTE_UNUSED(d);
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_UNUSED(beta);
    ARM_COMPUTE_UNUSED(pretranspose_hint);
    ARM_COMPUTE_ERROR("Not supported for this architecture");
}

template <>
void NEGEMMAssemblyDispatch<int8_t, int32_t>::configure(const ITensor *a, const ITensor *b, ITensor *d, float alpha, float beta, bool pretranspose_hint)
{
    // arm_gemm::gemm for 8bit only exists for aarch64
    ARM_COMPUTE_UNUSED(a);
    ARM_COMPUTE_UNUSED(b);
    ARM_COMPUTE_UNUSED(d);
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_UNUSED(beta);
    ARM_COMPUTE_UNUSED(pretranspose_hint);
    ARM_COMPUTE_ERROR("Not supported for this architecture");
}

template <>
void NEGEMMAssemblyDispatch<uint8_t, uint32_t>::Fallback::configure(const ITensor *a, const ITensor *b, ITensor *d, arm_gemm::GemmArgs<uint32_t> &args, MemoryGroup &memory_group)
{
    // arm_gemm::gemm for 8bit only exists for aarch64
    ARM_COMPUTE_UNUSED(a);
    ARM_COMPUTE_UNUSED(b);
    ARM_COMPUTE_UNUSED(d);
    ARM_COMPUTE_UNUSED(args);
    ARM_COMPUTE_UNUSED(memory_group);
    ARM_COMPUTE_ERROR("Not supported for this architecture");
}

template <>
void NEGEMMAssemblyDispatch<int8_t, int32_t>::Fallback::configure(const ITensor *a, const ITensor *b, ITensor *d, arm_gemm::GemmArgs<int32_t> &args, MemoryGroup &memory_group)
{
    // arm_gemm::gemm for 8bit only exists for aarch64
    ARM_COMPUTE_UNUSED(a);
    ARM_COMPUTE_UNUSED(b);
    ARM_COMPUTE_UNUSED(d);
    ARM_COMPUTE_UNUSED(args);
    ARM_COMPUTE_UNUSED(memory_group);
    ARM_COMPUTE_ERROR("Not supported for this architecture");
}
#endif // aarch64
template <typename TypeInput, typename TypeOutput>
void NEGEMMAssemblyDispatch<TypeInput, TypeOutput>::Fallback::configure(const ITensor *a, const ITensor *b, ITensor *d, arm_gemm::GemmArgs<TypeOutput> &args, MemoryGroup &memory_group)
{
    _gemm_kernel_asm = arm_gemm::gemm<TypeInput, TypeOutput>(args, nullptr);
    if(_gemm_kernel_asm == nullptr)
    {
        //configuration not supported: Leave function unconfigured:
        return;
    }

    // arm_compute wrapper for the Gemm object (see above)
    std::unique_ptr<NEGEMMAssemblyWrapperKernel<TypeInput, TypeOutput>> acl_gemm_wrapper = support::cpp14::make_unique<NEGEMMAssemblyWrapperKernel<TypeInput, TypeOutput>>();
    ARM_COMPUTE_ERROR_ON(acl_gemm_wrapper == nullptr);
    acl_gemm_wrapper->configure(_gemm_kernel_asm.get());
    const size_t workspace_size = _gemm_kernel_asm->get_working_size();
    if(workspace_size > 0)
    {
        // Allocate workspace
        const unsigned int alignment = 4096;
        //FIXME: is memory_group ever null ?
        allocate_workspace(workspace_size, &memory_group, alignment);
    }

    //if we disable this code below in brackets then ConvLayer deadlocks when threads > 1 and
    //the shapes are In=1x1x1024 Weights=1x1x1024x1001 Biases=1001 Out=1x1x1001
    {
        const int window_size = _gemm_kernel_asm->get_window_size();
        if(window_size < args._maxthreads)
        {
            _gemm_kernel_asm->set_nthreads(window_size);
        }
    }

    _optimised_kernel = std::move(acl_gemm_wrapper);
    _a                = a;
    _b                = b;
    _d                = d;
    // Check for pre-transposed support
    if(_gemm_kernel_asm->B_pretranspose_required())
    {
        // Forcing 128-byte alignment (required by 32-bit kernels)
        const unsigned int alignment           = 128;
        const size_t       B_pretranspose_size = _gemm_kernel_asm->get_B_pretransposed_array_size();
        _pretranspose.allocator()->init(TensorInfo(TensorShape{ (B_pretranspose_size + alignment /* FIXME: remove alignment after COMPMID-1088 */) }, 1, DataType::S8), alignment);
        _pretranspose.allocator()->allocate();
        ARM_COMPUTE_ERROR_ON_NULLPTR(_pretranspose.buffer());
    }
}

template <typename TypeInput, typename TypeOutput>
void NEGEMMAssemblyDispatch<TypeInput, TypeOutput>::Fallback::prepare()
{
    if(!_is_prepared)
    {
        // Pretranspose B if required
        if(_gemm_kernel_asm->B_pretranspose_required())
        {
            const int  ldb            = _b->info()->strides_in_bytes().y() / sizeof(TypeInput);
            const auto in1_ptr        = reinterpret_cast<const TypeInput *>(_b->buffer());
            const int  multi_stride_b = _b->info()->strides_in_bytes().z() / sizeof(TypeInput);

            ARM_COMPUTE_ERROR_ON(_pretranspose.buffer() == nullptr);
            _gemm_kernel_asm->pretranspose_B_array(_pretranspose.buffer(), in1_ptr, ldb, multi_stride_b);
            _b->mark_as_unused();
        }

        _is_prepared = true;
    }
}

template <typename TypeInput, typename TypeOutput>
void NEGEMMAssemblyDispatch<TypeInput, TypeOutput>::Fallback::allocate_workspace(size_t workspace_size, MemoryGroup *memory_group, size_t alignment)
{
    ARM_COMPUTE_ERROR_ON_MSG(workspace_size == 0, "size cannot be 0");
    _workspace.allocator()->init(TensorInfo(TensorShape{ (workspace_size + alignment /* FIXME: remove alignment after COMPMID-1088 */) }, 1, DataType::S8), alignment);
    if(memory_group != nullptr)
    {
        memory_group->manage(&_workspace);
    }
    _workspace.allocator()->allocate();
}

template <typename TypeInput, typename TypeOutput>
bool NEGEMMAssemblyDispatch<TypeInput, TypeOutput>::Fallback::is_configured() const
{
    return _optimised_kernel != nullptr;
}

template <typename TypeInput, typename TypeOutput>
void NEGEMMAssemblyDispatch<TypeInput, TypeOutput>::Fallback::run()
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
    if(_workspace.buffer() != nullptr)
    {
        _gemm_kernel_asm->set_working_space(reinterpret_cast<void *>(_workspace.buffer()));
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

template class NEGEMMAssemblyDispatch<float, float>;
template class NEGEMMAssemblyDispatch<uint8_t, uint32_t>;
template class NEGEMMAssemblyDispatch<int8_t, int32_t>;
} //namespace arm_compute
