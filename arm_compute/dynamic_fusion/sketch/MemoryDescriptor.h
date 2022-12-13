/*
 * Copyright (c) 2022 Arm Limited.
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
#ifndef ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_MEMORYDESCRIPTOR
#define ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_MEMORYDESCRIPTOR

#include "arm_compute/core/ITensorInfo.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Type of memory used by a workload tensor */
enum class MemoryType
{
    User      = 0, /**< Memory coming directly from users, e.g. for argument tensors */
    Auxiliary = 1, /**< Additional memory required by the workload tensor, e.g. for temporary tensors */
    NoAlloc   = 2, /**< Temporary tile which is not allocated as a whole tensor in the memory */
};

/** Memory information for tensors with @ref MemoryType::Auxiliary.
 * This informs how much additional memory is required for auxiliary tensors
 */
struct AuxMemoryInfo
{
    AuxMemoryInfo() = default;

    AuxMemoryInfo(size_t size, size_t alignment = 0) noexcept
        : size(size),
          alignment(alignment)
    {
    }

    friend bool operator==(const AuxMemoryInfo &info0, const AuxMemoryInfo &info1)
    {
        return info0.size == info1.size && info0.alignment == info1.alignment;
    }
    size_t size{ 0 };      /**< Total memory size in bytes */
    size_t alignment{ 0 }; /**< Memory alignment in bytes */
};

/** Descriptor of a workload tensor memory */
struct MemoryDescriptor
{
    MemoryType    memory_type{};     /**< Memory Type*/
    AuxMemoryInfo aux_memory_info{}; /**< Auxiliary Tensor Memory Information */
};

/** A map from @ref ITensorInfo to their corresponding @ref MemoryDescriptor */
using MemoryDescriptorMap = std::map<ITensorInfo::Id, MemoryDescriptor>;

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_MEMORYDESCRIPTOR */
