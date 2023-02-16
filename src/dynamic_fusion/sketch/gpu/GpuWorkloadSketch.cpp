/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#include "arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h"
#include "src/dynamic_fusion/sketch/gpu/GpuWorkloadSketchImpl.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
GpuWorkloadSketch::GpuWorkloadSketch(Context *context)
    : _impl{ std::make_unique<Implementation>(context) }
{
}
GpuWorkloadSketch::~GpuWorkloadSketch()
{
}

const GpuWorkloadSketch::Context *GpuWorkloadSketch::gpu_context() const
{
    return _impl->context();
}

void GpuWorkloadSketch::register_new_tensor(ITensorInfo &tensor_info)
{
    tensor_info.set_id(_impl->allocate_new_tensor_id());
    // All input output tensors are User tensors that need real backing memory
    _impl->register_memory_descriptor(tensor_info, MemoryDescriptor{ MemoryType::User });
}

TensorInfo GpuWorkloadSketch::create_tensor_info()
{
    TensorInfo tensor_info{};
    register_new_tensor(tensor_info);
    return tensor_info;
}

GpuWorkloadSketch::Implementation &GpuWorkloadSketch::implementation()
{
    return *_impl;
}
const GpuWorkloadSketch::Implementation &GpuWorkloadSketch::implementation() const
{
    return *_impl;
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
