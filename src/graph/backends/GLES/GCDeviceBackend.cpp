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
#include "arm_compute/graph/backends/GLES/GCDeviceBackend.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/INode.h"
#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/graph/backends/BackendRegistrar.h"
#include "arm_compute/graph/backends/GLES/GCFunctionFactory.h"
#include "arm_compute/graph/backends/GLES/GCNodeValidator.h"
#include "arm_compute/graph/backends/GLES/GCTensorHandle.h"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCBufferAllocator.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCMemoryGroup.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCScheduler.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"
#include "arm_compute/runtime/PoolManager.h"

#include "support/ToolchainSupport.h"

namespace arm_compute
{
namespace graph
{
namespace backends
{
/** Register GLES backend */
static detail::BackendRegistrar<GCDeviceBackend> GCDeviceBackend_registrar(Target::GC);

GCDeviceBackend::GCDeviceBackend()
    : _initialized(false), _allocator()
{
}

void GCDeviceBackend::initialize_backend()
{
    // Setup Scheduler
    GCScheduler::get().default_init();
}

void GCDeviceBackend::release_backend_context(GraphContext &ctx)
{
    //Nothing to do
    ARM_COMPUTE_UNUSED(ctx);
}

void GCDeviceBackend::setup_backend_context(GraphContext &ctx)
{
    // Force backend initialization
    if(!_initialized)
    {
        initialize_backend();
        _initialized = true;
    }

    // Setup a management backend
    if(ctx.memory_management_ctx(Target::GC) == nullptr)
    {
        MemoryManagerContext mm_ctx;
        mm_ctx.target      = Target::GC;
        mm_ctx.intra_mm    = create_memory_manager(MemoryManagerAffinity::Buffer);
        mm_ctx.cross_mm    = create_memory_manager(MemoryManagerAffinity::Buffer);
        mm_ctx.cross_group = std::make_shared<GCMemoryGroup>(mm_ctx.cross_mm);

        ctx.insert_memory_management_ctx(std::move(mm_ctx));
    }
}

bool GCDeviceBackend::is_backend_supported()
{
    return arm_compute::opengles31_is_available();
}

IAllocator *GCDeviceBackend::backend_allocator()
{
    return &_allocator;
}

std::unique_ptr<ITensorHandle> GCDeviceBackend::create_tensor(const Tensor &tensor)
{
    // Get tensor descriptor
    const TensorDescriptor &tensor_desc = tensor.desc();
    ARM_COMPUTE_ERROR_ON(tensor_desc.target != Target::GC);

    // Create backend tensor handle
    TensorInfo info(tensor_desc.shape, 1, tensor_desc.data_type, tensor_desc.quant_info);
    info.set_data_layout(tensor_desc.layout);
    auto backend_tensor_handle = support::cpp14::make_unique<GCTensorHandle>(info);

    return std::move(backend_tensor_handle);
}

std::unique_ptr<ITensorHandle> GCDeviceBackend::create_subtensor(ITensorHandle *parent, TensorShape shape, Coordinates coords, bool extend_parent)
{
    ARM_COMPUTE_UNUSED(parent, shape, coords, extend_parent);
    ARM_COMPUTE_ERROR("GLES backend has no sub-tensor support!");
    return nullptr;
}

std::unique_ptr<arm_compute::IFunction> GCDeviceBackend::configure_node(INode &node, GraphContext &ctx)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Configuring GC node with ID : " << node.id() << std::endl);
    ARM_COMPUTE_ERROR_ON(node.assigned_target() != Target::GC);

    // Configure node
    return GCFunctionFactory::create(&node, ctx);
}

arm_compute::Status GCDeviceBackend::validate_node(INode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Validating GC node with ID : " << node.id() << std::endl);
    ARM_COMPUTE_ERROR_ON(node.assigned_target() != Target::GC);

    return GCNodeValidator::validate(&node);
}

std::shared_ptr<arm_compute::IMemoryManager> GCDeviceBackend::create_memory_manager(MemoryManagerAffinity affinity)
{
    if(affinity == MemoryManagerAffinity::Offset)
    {
        ARM_COMPUTE_LOG_GRAPH_WARNING("GC Backend does not support offset affinity memory management!");
        return nullptr;
    }

    auto lifetime_mgr = std::make_shared<BlobLifetimeManager>();
    auto pool_mgr     = std::make_shared<PoolManager>();
    auto mm           = std::make_shared<MemoryManagerOnDemand>(lifetime_mgr, pool_mgr);

    mm->set_allocator(&_allocator);

    return mm;
}
} // namespace backends
} // namespace graph
} // namespace arm_compute
