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
#include "arm_compute/runtime/NEON/functions/NEGEMMAssemblyDispatch.h"

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/NEON/kernels/assembly/NEGEMMNativeWrapperKernel.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/NEON/functions/NESimpleAssemblyFunction.h"
#include "arm_compute/runtime/NEON/functions/assembly/NEGEMMInterleavedWrapper.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace
{
std::unique_ptr<IFunction> create_function_all_types(arm_gemm::KernelDescription gemm_kernel_info,
                                                     const ITensor *a, const ITensor *b, ITensor *d, float alpha, float beta, bool pretranspose_hint,
                                                     std::shared_ptr<IMemoryManager> memory_manager)

{
    //Note: It's safe to not check for FP16 support because this was already checked in NEGEMMAssemblyDispatch::configure()
    switch(gemm_kernel_info.method)
    {
        case arm_gemm::GemmMethod::GEMM_INTERLEAVED:
        {
            if(!pretranspose_hint)
            {
                return nullptr;
            }
            auto function = support::cpp14::make_unique<NEGEMMInterleavedWrapper>(memory_manager);
            function->configure(a, b, d, alpha, beta, pretranspose_hint);
            return std::move(function);
        }
#if defined(__aarch64__)
        case arm_gemm::GemmMethod::GEMM_NATIVE:
        {
            if(gemm_kernel_info.name.find("sgemm_native_16x4") != std::string::npos)
            {
                auto kernel = support::cpp14::make_unique<NEGEMMNativeWrapperKernel<float, float>>();
                kernel->configure(a, b, d, alpha, beta);
                auto function = support::cpp14::make_unique<NESimpleAssemblyFunction>();
                function->configure(std::move(kernel));
                return std::move(function);
            }
            return nullptr;
        }
#endif // defined(__aarch64__)
        default:
            return nullptr;
    }
}

/** Fallback in case ACL doesn't have a function */
template <typename TypeInput, typename TypeOutput>
class Fallback : public NEGEMMAssemblyDispatch::IFallback
{
public:
    /** Initialise the functions's input and output.
     *
     * @param[in]  a            Input tensor containing the Matrix A.
     * @param[in]  b            Input tensor containing the Matrix B.
     * @param[out] d            Output tensor to store the result of matrix multiplication.
     * @param[in]  args         Matrix multiplication information.
     * @param[in]  memory_group Memory group to be used by the function.
     */
    void configure(const ITensor *a, const ITensor *b, ITensor *d, arm_gemm::GemmArgs<TypeOutput> args, MemoryGroup &memory_group);

    // Inherited methods overridden:
    void run() override;
    void prepare() override;
    bool is_configured() const override;

private:
    /** Allocate a workspace tensor.
     *
     * @param[in] workspace_size Size to allocate.
     * @param[in] memory_group   Tensor memory group.
     * @param[in] alignment      Workspace memory alignment.
     */
    void allocate_workspace(size_t workspace_size, MemoryGroup &memory_group, size_t alignment);

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
};

template <typename TypeInput, typename TypeOutput>
void Fallback<TypeInput, TypeOutput>::configure(const ITensor *a, const ITensor *b, ITensor *d, arm_gemm::GemmArgs<TypeOutput> args, MemoryGroup &memory_group)
{
    arm_gemm::GemmConfig              gemm_cfg;
    const arm_gemm::KernelDescription gemm_kernel_info = arm_gemm::get_gemm_method<TypeInput, TypeOutput>(args);
    if(gemm_kernel_info.method != arm_gemm::GemmMethod::GEMV_BATCHED)
    {
        gemm_cfg.filter = gemm_kernel_info.name;
        args._cfg       = &gemm_cfg;
    }
    _gemm_kernel_asm = arm_gemm::gemm<TypeInput, TypeOutput>(args);
    if(_gemm_kernel_asm == nullptr)
    {
        //configuration not supported: Leave function unconfigured:
        return;
    }

    // arm_compute wrapper for the Gemm object (see above)
    std::unique_ptr<NEGEMMAssemblyWrapperKernel<TypeInput, TypeOutput>> acl_gemm_wrapper = support::cpp14::make_unique<NEGEMMAssemblyWrapperKernel<TypeInput, TypeOutput>>();
    ARM_COMPUTE_ERROR_ON(acl_gemm_wrapper == nullptr);
    acl_gemm_wrapper->configure(_gemm_kernel_asm.get(), gemm_cfg.filter);
    const size_t workspace_size = _gemm_kernel_asm->get_working_size();
    if(workspace_size > 0)
    {
        // Allocate workspace
        const unsigned int alignment = 4096;
        allocate_workspace(workspace_size, memory_group, alignment);
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
    }
}

template <typename TypeInput, typename TypeOutput>
void Fallback<TypeInput, TypeOutput>::prepare()
{
    if(!_is_prepared)
    {
        // Pretranspose B if required
        if(_gemm_kernel_asm->B_pretranspose_required())
        {
            _pretranspose.allocator()->allocate();
            ARM_COMPUTE_ERROR_ON(_pretranspose.buffer() == nullptr);
            const int  ldb            = _b->info()->strides_in_bytes().y() / sizeof(TypeInput);
            const auto in1_ptr        = reinterpret_cast<const TypeInput *>(_b->buffer() + _b->info()->offset_first_element_in_bytes());
            const int  multi_stride_b = _b->info()->strides_in_bytes().z() / sizeof(TypeInput);

            _gemm_kernel_asm->pretranspose_B_array(_pretranspose.buffer(), in1_ptr, ldb, multi_stride_b);
            _b->mark_as_unused();
        }

        _is_prepared = true;
    }
}

template <typename TypeInput, typename TypeOutput>
void Fallback<TypeInput, TypeOutput>::allocate_workspace(size_t workspace_size, MemoryGroup &memory_group, size_t alignment)
{
    ARM_COMPUTE_ERROR_ON_MSG(workspace_size == 0, "size cannot be 0");
    _workspace.allocator()->init(TensorInfo(TensorShape{ (workspace_size + alignment /* FIXME: remove alignment after COMPMID-1088 */) }, 1, DataType::S8), alignment);
    memory_group.manage(&_workspace);
    _workspace.allocator()->allocate();
}

template <typename TypeInput, typename TypeOutput>
bool Fallback<TypeInput, TypeOutput>::is_configured() const
{
    return _optimised_kernel != nullptr;
}

template <typename TypeInput, typename TypeOutput>
void Fallback<TypeInput, TypeOutput>::run()
{
    const int lda = _a->info()->strides_in_bytes().y() / sizeof(TypeInput);
    int       ldb = 0;
    const int ldd = _d->info()->strides_in_bytes().y() / sizeof(TypeOutput);

    // In the case of NHWC we want to interpret the output shape as 3D. Thus, the batch stride for A is
    // the relevant multiple of the row stride.
    const bool is_nhwc           = _a->info()->data_layout() == DataLayout::NHWC;
    const int  stride_in_bytes_a = is_nhwc ? _a->info()->strides_in_bytes().y() * _d->info()->dimension(1) : _a->info()->strides_in_bytes().z();

    const int batch_stride_a = stride_in_bytes_a / sizeof(TypeInput);
    const int batch_stride_d = _d->info()->strides_in_bytes().z() / sizeof(TypeOutput);

    const int multi_stride_a = _a->info()->strides_in_bytes()[3] / sizeof(TypeInput);
    int       multi_stride_b = 0;
    const int multi_stride_d = _d->info()->strides_in_bytes()[3] / sizeof(TypeOutput);

    const auto       in0_ptr = reinterpret_cast<const TypeInput *>(_a->buffer() + _a->info()->offset_first_element_in_bytes());
    const TypeInput *in1_ptr = nullptr;
    auto             out_ptr = reinterpret_cast<TypeOutput *>(_d->buffer() + _d->info()->offset_first_element_in_bytes());

    // Check if B is pre-tranposed and de-reference if not
    if(!_gemm_kernel_asm->B_is_pretransposed())
    {
        ldb            = _b->info()->strides_in_bytes().y() / sizeof(TypeInput);
        multi_stride_b = _b->info()->strides_in_bytes().z() / sizeof(TypeInput);
        in1_ptr        = reinterpret_cast<const TypeInput *>(_b->buffer() + _b->info()->offset_first_element_in_bytes());
    }

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

template <typename TypeInput, typename TypeOutput>
void create_function_or_arm_gemm(std::unique_ptr<IFunction> &acl_function, std::unique_ptr<NEGEMMAssemblyDispatch::IFallback> &arm_gemm, MemoryGroup &memory_group, const ITensor *a, const ITensor *b,
                                 ITensor *d, float alpha, float beta, bool pretranspose_hint, std::shared_ptr<IMemoryManager> memory_manager)
{
    INEGEMMWrapperKernel::Params p           = INEGEMMWrapperKernel::extract_parameters(a, b, d);
    const CPUInfo               &ci          = NEScheduler::get().cpu_info();
    unsigned int                 num_threads = NEScheduler::get().num_threads();

    arm_gemm::GemmArgs<TypeOutput> args(&ci, p.M, p.N, p.K, p.batches, p.multis, false, false, alpha, beta, num_threads, pretranspose_hint);

    //Try to create an ACL function:
    acl_function = create_function_all_types(arm_gemm::get_gemm_method<TypeInput, TypeOutput>(args), a, b, d, alpha, beta, pretranspose_hint, std::move(memory_manager));

    //If we still don't have an ACL function:
    if(acl_function == nullptr)
    {
        //Fallback onto arm_gemm function if ACL doesn't support this method.
        auto fallback = support::cpp14::make_unique<Fallback<TypeInput, TypeOutput>>();
        fallback->configure(a, b, d, args, memory_group);
        arm_gemm = std::move(fallback);
    }
}

} //namespace

NEGEMMAssemblyDispatch::NEGEMMAssemblyDispatch(std::shared_ptr<IMemoryManager> memory_manager)
    : _function(nullptr), _arm_gemm(nullptr), _memory_group(memory_manager), _memory_manager(memory_manager)
{
}

Status NEGEMMAssemblyDispatch::validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *d, float alpha, float beta, bool pretranspose_hint)
{
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_UNUSED(beta);
    ARM_COMPUTE_UNUSED(pretranspose_hint);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(a, b, d);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(a);
#ifndef __aarch64__
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(a->data_type() == DataType::U8 || a->data_type() == DataType::S8 || a->data_type() == DataType::QASYMM8, "8bit integer types only supported for aarch64");
#endif /* __aarch64__ */
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(a, 1, DataType::F32, DataType::U8, DataType::QASYMM8, DataType::S8, DataType::F16);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(a, b);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(a->data_type() == DataType::F32 && d->data_type() != DataType::F32, "Only F32 output supported for F32 input");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(a->data_type() == DataType::F16 && d->data_type() != DataType::F16, "Only F16 output supported for F16 input");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(a->data_type() == DataType::U8 && d->data_type() != DataType::U32, "Only U32 output supported for U8 input");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(a->data_type() == DataType::QASYMM8 && d->data_type() != DataType::S32 && d->data_type() != DataType::U32, "Only U32/S32 output supported for QASYMM8 input");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(a->data_type() == DataType::S8 && d->data_type() != DataType::S32, "Only S32 output supported for S8 input");
    return Status{};
}

void NEGEMMAssemblyDispatch::configure(const ITensor *a, const ITensor *b, ITensor *d, float alpha, float beta, bool pretranspose_hint)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(a);
    ARM_COMPUTE_ERROR_ON_NULLPTR(b);
    ARM_COMPUTE_ERROR_ON_NULLPTR(d);

    //If we don't support a combination of data types, silently return: it is the caller's responsibility to check if configure() was successful via is_configured()
    if(!NEGEMMAssemblyDispatch::validate(a->info(), b->info(), d->info(), alpha, beta, pretranspose_hint))
    {
        return;
    }

    switch(a->info()->data_type())
    {
        case DataType::F32:
            create_function_or_arm_gemm<float, float>(_function, _arm_gemm, _memory_group, a, b, d, alpha, beta, pretranspose_hint, _memory_manager);
            break;
#ifdef __aarch64__
        case DataType::U8:
        case DataType::QASYMM8:
            create_function_or_arm_gemm<uint8_t, uint32_t>(_function, _arm_gemm, _memory_group, a, b, d, alpha, beta, pretranspose_hint, _memory_manager);
            break;
        case DataType::S8:
            create_function_or_arm_gemm<int8_t, int32_t>(_function, _arm_gemm, _memory_group, a, b, d, alpha, beta, pretranspose_hint, _memory_manager);
            break;
#endif /* __aarch64__ */
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            create_function_or_arm_gemm<float16_t, float16_t>(_function, _arm_gemm, _memory_group, a, b, d, alpha, beta, pretranspose_hint, _memory_manager);
            break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        default:
            break;
    }
}

void NEGEMMAssemblyDispatch::prepare()
{
    if(_function != nullptr)
    {
        _function->prepare();
    }
    else
    {
        ARM_COMPUTE_ERROR_ON(_arm_gemm == nullptr);
        _arm_gemm->prepare();
    }
}

bool NEGEMMAssemblyDispatch::is_configured() const
{
    return (_arm_gemm != nullptr && _arm_gemm->is_configured()) || _function != nullptr;
}

void NEGEMMAssemblyDispatch::run()
{
    _memory_group.acquire();
    if(_function != nullptr)
    {
        _function->run();
    }
    else
    {
        ARM_COMPUTE_ERROR_ON(_arm_gemm == nullptr);
        _arm_gemm->run();
    }
    _memory_group.release();
}
} //namespace arm_compute
