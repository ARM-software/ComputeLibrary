/*
 * Copyright (c) 2022-2024 Arm Limited.
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

#include "arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadContext.h"

#include "arm_compute/core/CL/CLCompileContext.h"

#include "src/dynamic_fusion/sketch/gpu/GpuWorkloadContextImpl.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
GpuWorkloadContext::GpuWorkloadContext(CLCompileContext *cl_compile_ctx)
    : _impl{std::make_unique<Impl>(GpuLanguage::OpenCL, cl_compile_ctx)}
{
}

GpuWorkloadContext::~GpuWorkloadContext() = default;

GpuWorkloadContext::GpuWorkloadContext(GpuWorkloadContext &&other) = default;

GpuWorkloadContext &GpuWorkloadContext::operator=(GpuWorkloadContext &&other) = default;

GpuTarget GpuWorkloadContext::gpu_target() const
{
    return _impl->cl_compile_context()->get_gpu_target();
}

GpuLanguage GpuWorkloadContext::gpu_language() const
{
    return _impl->gpu_language();
}

const CLCompileContext *GpuWorkloadContext::cl_compile_context() const
{
    return _impl->cl_compile_context();
}

void GpuWorkloadContext::register_user_tensor(std::unique_ptr<TensorInfo> &&tensor_info)
{
    _impl->register_user_tensor(std::move(tensor_info));
}

GpuWorkloadContext::Impl &GpuWorkloadContext::implementation()
{
    return *_impl;
}

const GpuWorkloadContext::Impl &GpuWorkloadContext::implementation() const
{
    return *_impl;
}

GpuWorkloadContext::Impl::Impl(GpuLanguage gpu_language, CLCompileContext *cl_compile_ctx)
    : _gpu_language(gpu_language),
      _cl_compile_ctx(cl_compile_ctx),
      _next_tensor_id(1),
      _mem_map(),
      _managed_tensor_info()
{
}

GpuLanguage GpuWorkloadContext::Impl::gpu_language() const
{
    return _gpu_language;
}

const CLCompileContext *GpuWorkloadContext::Impl::cl_compile_context() const
{
    return _cl_compile_ctx;
}

const MemoryDescriptorMap &GpuWorkloadContext::Impl::mem_map() const
{
    return _mem_map;
}

void GpuWorkloadContext::Impl::register_user_tensor(std::unique_ptr<TensorInfo> &&tensor_info)
{
    ARM_COMPUTE_ERROR_ON(tensor_info->has_valid_id());

    const auto tensor_id = next_tensor_id();

    tensor_info->set_id(tensor_id);
    _mem_map[tensor_id] = MemoryDescriptor{MemoryType::User};
    // Save a *copy* of the user tensor info in workload context for future reference
    // Note that this means if the user modifies the @p tensor_info, the change will not be reflected in the context
    _managed_tensor_info.emplace(tensor_info->id(), std::move(tensor_info));
}

ITensorInfo *GpuWorkloadContext::Impl::create_virtual_tensor()
{
    auto       tensor_info = std::make_unique<TensorInfo>();
    const auto tensor_id   = -next_tensor_id();
    tensor_info->set_id(tensor_id);
    _mem_map[tensor_id] = MemoryDescriptor{MemoryType::Virtual};
    auto inserted       = _managed_tensor_info.emplace(tensor_info->id(), std::move(tensor_info));
    return inserted.first->second.get();
}

ITensorInfo *GpuWorkloadContext::Impl::create_auxiliary_tensor(const ITensorInfo &itensor_info)
{
    auto       tensor_info = std::make_unique<TensorInfo>(itensor_info);
    const auto tensor_id   = next_tensor_id();
    tensor_info->set_id(tensor_id);
    _mem_map[tensor_id] = MemoryDescriptor{MemoryType::Auxiliary, AuxMemoryInfo{tensor_info->total_size()}};
    auto inserted       = _managed_tensor_info.emplace(tensor_info->id(), std::move(tensor_info));
    return inserted.first->second.get();
}

ITensorInfo *GpuWorkloadContext::Impl::get_tensor_info(ITensorInfo::Id id)
{
    return _managed_tensor_info.at(id).get();
}

const ITensorInfo *GpuWorkloadContext::Impl::get_tensor_info(ITensorInfo::Id id) const
{
    return _managed_tensor_info.at(id).get();
}

ITensorInfo::Id GpuWorkloadContext::Impl::next_tensor_id()
{
    return _next_tensor_id++;
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
