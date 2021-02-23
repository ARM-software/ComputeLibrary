/*
 * Copyright (c) 2021 Arm Limited.
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
#include "src/cpu/CpuContext.h"

#include "arm_compute/core/CPP/CPPTypes.h"
#include "src/cpu/CpuTensor.h"
#include "src/runtime/CPUUtils.h"

#include <cstdlib>
#include <malloc.h>

namespace arm_compute
{
namespace cpu
{
namespace
{
void *default_allocate(void *user_data, size_t size)
{
    ARM_COMPUTE_UNUSED(user_data);
    return ::operator new(size);
}
void default_free(void *user_data, void *ptr)
{
    ARM_COMPUTE_UNUSED(user_data);
    ::operator delete(ptr);
}
void *default_aligned_allocate(void *user_data, size_t size, size_t alignment)
{
    ARM_COMPUTE_UNUSED(user_data);
    void *ptr = nullptr;
#if defined(BARE_METAL) || defined(__APPLE__)
    size_t rem       = size % alignment;
    size_t real_size = (rem) ? (size + alignment - rem) : size;
    ptr              = memalign(alignment, real_size);
#else  /* defined(BARE_METAL) || defined(__APPLE__) */
    if(posix_memalign(&ptr, alignment, size) != 0)
    {
        // posix_memalign returns non-zero on failures, the return values will be
        // - EINVAL: wrong alignment
        // - ENOMEM: insufficient memory
        ARM_COMPUTE_LOG_ERROR_ACL("posix_memalign failed, the returned pointer will be invalid");
    }
#endif /* defined(BARE_METAL) || defined(__APPLE__) */
    return ptr;
}
void default_aligned_free(void *user_data, void *ptr)
{
    ARM_COMPUTE_UNUSED(user_data);
    free(ptr);
}
static AclAllocator default_allocator = { &default_allocate,
                                          &default_free,
                                          &default_aligned_allocate,
                                          &default_aligned_free,
                                          nullptr
                                        };

AllocatorWrapper populate_allocator(AclAllocator *external_allocator)
{
    bool is_valid = (external_allocator != nullptr);
    if(is_valid)
    {
        is_valid = is_valid && (external_allocator->alloc != nullptr);
        is_valid = is_valid && (external_allocator->free != nullptr);
        is_valid = is_valid && (external_allocator->aligned_alloc != nullptr);
        is_valid = is_valid && (external_allocator->aligned_free != nullptr);
    }
    return is_valid ? AllocatorWrapper(*external_allocator) : AllocatorWrapper(default_allocator);
}

CpuCapabilities populate_capabilities_legacy(const CPUInfo &cpu_info)
{
    CpuCapabilities caps;

    // Extract SIMD extension
    caps.neon = true;
#ifdef SVE2
    caps.sve2 = true;
#endif /* SVE2 */
    // Extract data-type support
    caps.fp16 = cpu_info.has_fp16();
#ifdef V8P6_BF
    caps.bf16 = true;
#endif /* V8P6_BF */

    // Extract ISA extensions
    caps.dot = cpu_info.has_dotprod();
#ifdef MMLA_FP32
    caps.mmla_fp = true;
#endif /* MMLA_FP32 */
#ifdef MMLA_INT8
    caps.mmla_int8 = true;
#endif /* MMLA_INT8 */

    return caps;
}

CpuCapabilities populate_capabilities_flags(AclTargetCapabilities external_caps)
{
    CpuCapabilities caps;

    // Extract SIMD extension
    caps.neon = external_caps & AclCpuCapabilitiesNeon;
    caps.sve  = external_caps & AclCpuCapabilitiesSve;
    caps.sve2 = external_caps & AclCpuCapabilitiesSve2;
    // Extract data-type support
    caps.fp16 = external_caps & AclCpuCapabilitiesFp16;
    caps.bf16 = external_caps & AclCpuCapabilitiesBf16;
    // Extract ISA extensions
    caps.dot       = external_caps & AclCpuCapabilitiesDot;
    caps.mmla_fp   = external_caps & AclCpuCapabilitiesMmlaFp;
    caps.mmla_int8 = external_caps & AclCpuCapabilitiesMmlaInt8;

    return caps;
}

CpuCapabilities populate_capabilities(AclTargetCapabilities external_caps,
                                      int32_t               max_threads)
{
    // Extract legacy structure
    CPUInfo cpu_info;
    arm_compute::utils::cpu::get_cpu_configuration(cpu_info);

    CpuCapabilities caps;
    if(external_caps != AclCpuCapabilitiesAuto)
    {
        caps = populate_capabilities_flags(external_caps);
    }
    else
    {
        caps = populate_capabilities_legacy(cpu_info);
    }

    // Set max number of threads
#if defined(BARE_METAL)
    ARM_COMPUTE_UNUSED(max_threads);
    caps.max_threads = 1;
#else  /* defined(BARE_METAL) */
    caps.max_threads = (max_threads > 0) ? max_threads : std::thread::hardware_concurrency();
#endif /* defined(BARE_METAL) */

    return caps;
}
} // namespace

CpuContext::CpuContext(const AclContextOptions *options)
    : IContext(Target::Cpu),
      _allocator(default_allocator),
      _caps(populate_capabilities(AclCpuCapabilitiesAuto, -1))
{
    if(options != nullptr)
    {
        _allocator = populate_allocator(options->allocator);
        _caps      = populate_capabilities(options->capabilities, options->max_compute_units);
    }
}

const CpuCapabilities &CpuContext::capabilities() const
{
    return _caps;
}

AllocatorWrapper &CpuContext::allocator()
{
    return _allocator;
}

ITensorV2 *CpuContext::create_tensor(const AclTensorDescriptor &desc, bool allocate)
{
    CpuTensor *tensor = new CpuTensor(this, desc);
    if(tensor != nullptr && allocate)
    {
        tensor->allocate();
    }
    return tensor;
}
} // namespace cpu
} // namespace arm_compute
