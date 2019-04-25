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
#include "arm_compute/graph/backends/CL/CLDeviceBackend.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/INode.h"
#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/graph/backends/BackendRegistrar.h"
#include "arm_compute/graph/backends/CL/CLFunctionFactory.h"
#include "arm_compute/graph/backends/CL/CLNodeValidator.h"
#include "arm_compute/graph/backends/CL/CLSubTensorHandle.h"
#include "arm_compute/graph/backends/CL/CLTensorHandle.h"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/CL/CLBufferAllocator.h"
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"
#include "arm_compute/runtime/PoolManager.h"

#include "support/ToolchainSupport.h"

namespace arm_compute
{
namespace graph
{
namespace backends
{
namespace
{
bool file_exists(const std::string &filename)
{
    std::ifstream file(filename);
    return file.good();
}
} // namespace

/** Register CL backend */
static detail::BackendRegistrar<CLDeviceBackend> CLDeviceBackend_registrar(Target::CL);

CLDeviceBackend::CLDeviceBackend()
    : _context_count(0), _tuner(), _allocator(nullptr), _tuner_file()
{
}

CLDeviceBackend::~CLDeviceBackend()
{
    // TODO (geopin01) : Shouldn't call non exception safe stuff here
    if(_tuner.tune_new_kernels() && !_tuner.lws_table().empty() && !_tuner_file.empty())
    {
        _tuner.save_to_file(_tuner_file);
    }
}

void CLDeviceBackend::set_kernel_tuning(bool enable_tuning)
{
    _tuner.set_tune_new_kernels(enable_tuning);
}

void CLDeviceBackend::set_kernel_tuning_mode(CLTunerMode tuning_mode)
{
    _tuner.set_tuner_mode(tuning_mode);
}

void CLDeviceBackend::initialize_backend()
{
    // Setup Scheduler
    CLScheduler::get().default_init(&_tuner);

    // Create allocator with new context
    _allocator = support::cpp14::make_unique<CLBufferAllocator>();
}

void CLDeviceBackend::release_backend_context(GraphContext &ctx)
{
    ARM_COMPUTE_UNUSED(ctx);
    _context_count--;
    if(_context_count == 0) // No more context using the backend: free resources
    {
        _allocator = nullptr;
    }
}

void CLDeviceBackend::setup_backend_context(GraphContext &ctx)
{
    // Force backend initialization
    _context_count++;
    if(_context_count == 1)
    {
        initialize_backend();
    }

    // Setup tuner
    _tuner_file = ctx.config().tuner_file;
    // Load tuner data if available
    if(file_exists(_tuner_file))
    {
        _tuner.load_from_file(_tuner_file);
    }

    set_kernel_tuning(ctx.config().use_tuner);
    set_kernel_tuning_mode(ctx.config().tuner_mode);

    // Setup a management backend
    if(ctx.memory_management_ctx(Target::CL) == nullptr)
    {
        MemoryManagerContext mm_ctx;
        mm_ctx.target      = Target::CL;
        mm_ctx.intra_mm    = create_memory_manager(MemoryManagerAffinity::Buffer);
        mm_ctx.cross_mm    = create_memory_manager(MemoryManagerAffinity::Buffer);
        mm_ctx.cross_group = std::make_shared<CLMemoryGroup>(mm_ctx.cross_mm);
        mm_ctx.allocator   = _allocator.get();

        ctx.insert_memory_management_ctx(std::move(mm_ctx));
    }
}

bool CLDeviceBackend::is_backend_supported()
{
    return arm_compute::opencl_is_available();
}

IAllocator *CLDeviceBackend::backend_allocator()
{
    return _allocator.get();
}

std::unique_ptr<ITensorHandle> CLDeviceBackend::create_tensor(const Tensor &tensor)
{
    // Get tensor descriptor
    const TensorDescriptor &tensor_desc = tensor.desc();
    ARM_COMPUTE_ERROR_ON(tensor_desc.target != Target::CL);

    // Create backend tensor handle
    TensorInfo info(tensor_desc.shape, 1, tensor_desc.data_type, tensor_desc.quant_info);
    info.set_data_layout(tensor_desc.layout);
    auto backend_tensor_handle = support::cpp14::make_unique<CLTensorHandle>(info);

    return std::move(backend_tensor_handle);
}

std::unique_ptr<ITensorHandle> CLDeviceBackend::create_subtensor(ITensorHandle *parent, TensorShape shape, Coordinates coords, bool extend_parent)
{
    if(parent == nullptr)
    {
        return nullptr;
    }

    return support::cpp14::make_unique<CLSubTensorHandle>(parent, shape, coords, extend_parent);
}

std::unique_ptr<arm_compute::IFunction> CLDeviceBackend::configure_node(INode &node, GraphContext &ctx)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Configuring CL node with ID : " << node.id() << std::endl);
    ARM_COMPUTE_ERROR_ON(node.assigned_target() != Target::CL);

    // Configure node
    return CLFunctionFactory::create(&node, ctx);
}

arm_compute::Status CLDeviceBackend::validate_node(INode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Validating CL node with ID : " << node.id() << std::endl);
    ARM_COMPUTE_ERROR_ON(node.assigned_target() != Target::CL);

    return CLNodeValidator::validate(&node);
}

std::shared_ptr<arm_compute::IMemoryManager> CLDeviceBackend::create_memory_manager(MemoryManagerAffinity affinity)
{
    if(affinity == MemoryManagerAffinity::Offset)
    {
        ARM_COMPUTE_LOG_GRAPH_WARNING("CL Backend does not support offset affinity memory management!");
        return nullptr;
    }

    auto lifetime_mgr = std::make_shared<BlobLifetimeManager>();
    auto pool_mgr     = std::make_shared<PoolManager>();
    auto mm           = std::make_shared<MemoryManagerOnDemand>(lifetime_mgr, pool_mgr);

    return mm;
}
} // namespace backends
} // namespace graph
} // namespace arm_compute
