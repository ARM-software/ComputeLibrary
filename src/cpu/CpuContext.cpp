/*
 * Copyright (c) 2021-2022 Arm Limited.
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
#include "src/cpu/CpuQueue.h"
#include "src/cpu/CpuTensor.h"

#include <cstdlib>
#if !defined(__APPLE__) && !defined(__OpenBSD__)
#include <malloc.h>

#if defined(_WIN64)
#define posix_memalign _aligned_realloc
#define posix_memalign_free _aligned_free
#endif // defined(_WIN64)
#endif // !defined(__APPLE__) && !defined(__OpenBSD__)

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
#if defined(BARE_METAL)
    size_t rem       = size % alignment;
    size_t real_size = (rem) ? (size + alignment - rem) : size;
    ptr              = memalign(alignment, real_size);
#else  /* defined(BARE_METAL) */
    if(posix_memalign(&ptr, alignment, size) != 0)
    {
        // posix_memalign returns non-zero on failures, the return values will be
        // - EINVAL: wrong alignment
        // - ENOMEM: insufficient memory
        ARM_COMPUTE_LOG_ERROR_ACL("posix_memalign failed, the returned pointer will be invalid");
    }
#endif /* defined(BARE_METAL) */
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

cpuinfo::CpuIsaInfo populate_capabilities_flags(AclTargetCapabilities external_caps)
{
    cpuinfo::CpuIsaInfo isa_caps;

    // Extract SIMD extension
    isa_caps.neon = external_caps & AclCpuCapabilitiesNeon;
    isa_caps.sve  = external_caps & AclCpuCapabilitiesSve;
    isa_caps.sve2 = external_caps & AclCpuCapabilitiesSve2;

    // Extract data-type support
    isa_caps.fp16    = external_caps & AclCpuCapabilitiesFp16;
    isa_caps.bf16    = external_caps & AclCpuCapabilitiesBf16;
    isa_caps.svebf16 = isa_caps.bf16;

    // Extract ISA extensions
    isa_caps.dot      = external_caps & AclCpuCapabilitiesDot;
    isa_caps.i8mm     = external_caps & AclCpuCapabilitiesMmlaInt8;
    isa_caps.svef32mm = external_caps & AclCpuCapabilitiesMmlaFp;

    return isa_caps;
}

CpuCapabilities populate_capabilities(AclTargetCapabilities external_caps,
                                      int32_t               max_threads)
{
    CpuCapabilities caps;

    // Populate capabilities with system information
    caps.cpu_info = cpuinfo::CpuInfo::build();
    if(external_caps != AclCpuCapabilitiesAuto)
    {
        cpuinfo::CpuIsaInfo isa  = populate_capabilities_flags(external_caps);
        auto                cpus = caps.cpu_info.cpus();

        caps.cpu_info = cpuinfo::CpuInfo(isa, cpus);
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

IQueue *CpuContext::create_queue(const AclQueueOptions *options)
{
    return new CpuQueue(this, options);
}
} // namespace cpu
} // namespace arm_compute
