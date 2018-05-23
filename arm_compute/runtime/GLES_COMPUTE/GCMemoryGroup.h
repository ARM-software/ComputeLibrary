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
#ifndef __ARM_COMPUTE_GCMEMORYGROUP_H__
#define __ARM_COMPUTE_GCMEMORYGROUP_H__

#include "arm_compute/runtime/MemoryGroupBase.h"

#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"
#include "arm_compute/core/utils/misc/Cast.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensor.h"

namespace arm_compute
{
using GCMemoryGroup = MemoryGroupBase<GCTensor>;

template <>
inline void MemoryGroupBase<GCTensor>::associate_memory_group(GCTensor *obj)
{
    ARM_COMPUTE_ERROR_ON(obj == nullptr);
    ARM_COMPUTE_ERROR_ON(dynamic_cast<GCTensorAllocator *>(obj->allocator()) == nullptr);

    auto allocator = arm_compute::utils::cast::polymorphic_downcast<GCTensorAllocator *>(obj->allocator());
    ARM_COMPUTE_ERROR_ON(allocator == nullptr);
    allocator->set_associated_memory_group(this);
}
} // namespace arm_compute
#endif /*__ARM_COMPUTE_GCMEMORYGROUP_H__ */
