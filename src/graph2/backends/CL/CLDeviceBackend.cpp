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
#include "arm_compute/graph2/backends/CL/CLDeviceBackend.h"

#include "arm_compute/graph2/Graph.h"
#include "arm_compute/graph2/GraphContext.h"
#include "arm_compute/graph2/INode.h"
#include "arm_compute/graph2/Logger.h"
#include "arm_compute/graph2/Tensor.h"
#include "arm_compute/graph2/backends/BackendRegistrar.h"
#include "arm_compute/graph2/backends/CL/CLFunctionFactory.h"
#include "arm_compute/graph2/backends/CL/CLSubTensorHandle.h"
#include "arm_compute/graph2/backends/CL/CLTensorHandle.h"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/CL/CLBufferAllocator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"
#include "arm_compute/runtime/PoolManager.h"

#include "support/ToolchainSupport.h"

namespace arm_compute
{
namespace graph2
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

/** Tuner export file */
static const std::string tuner_data_filename = "acl_tuner.csv";

CLDeviceBackend::CLDeviceBackend()
    : _tuner(), _allocator(cl::Context::getDefault())
{
}

CLDeviceBackend::~CLDeviceBackend()
{
    // TODO (geopin01) : Shouldn't call non exception safe stuff here
    if(_tuner.tune_new_kernels() && !_tuner.lws_table().empty())
    {
        _tuner.save_to_file(tuner_data_filename);
    }
}

void CLDeviceBackend::set_kernel_tuning(bool enable_tuning)
{
    _tuner.set_tune_new_kernels(enable_tuning);
}

void CLDeviceBackend::initialize_backend()
{
    // Load tuner data if available
    if(_tuner.lws_table().empty() && file_exists(tuner_data_filename))
    {
        _tuner.load_from_file(tuner_data_filename);
    }

    // Setup Scheduler
    CLScheduler::get().default_init(&_tuner);

    // Create allocator with new context
    _allocator = CLBufferAllocator();
}

void CLDeviceBackend::setup_backend_context(GraphContext &ctx)
{
    // Setup tuner
    set_kernel_tuning(ctx.is_tuning_enabled());

    // Setup a management backend
    if(ctx.memory_management_ctx(Target::CL) == nullptr)
    {
        MemoryManagerContext mm_ctx;
        mm_ctx.target = Target::CL;
        mm_ctx.mm     = create_memory_manager(MemoryManagerAffinity::Buffer);

        ctx.insert_memory_management_ctx(std::move(mm_ctx));
    }
}

std::unique_ptr<ITensorHandle> CLDeviceBackend::create_tensor(const Tensor &tensor)
{
    // Get tensor descriptor
    const TensorDescriptor &tensor_desc = tensor.desc();
    ARM_COMPUTE_ERROR_ON(tensor_desc.target != Target::CL);

    // Create backend tensor handle
    TensorInfo info(tensor_desc.shape, 1, tensor_desc.data_type);
    auto       backend_tensor_handle = support::cpp14::make_unique<CLTensorHandle>(info);

    return std::move(backend_tensor_handle);
}

std::unique_ptr<ITensorHandle> CLDeviceBackend::create_subtensor(ITensorHandle *parent, TensorShape shape, Coordinates coords)
{
    if(parent == nullptr)
    {
        return nullptr;
    }

    return support::cpp14::make_unique<CLSubTensorHandle>(parent, shape, coords);
}

std::unique_ptr<arm_compute::IFunction> CLDeviceBackend::configure_node(INode &node, GraphContext &ctx)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Configuring CL node with ID : " << node.id() << std::endl);
    ARM_COMPUTE_ERROR_ON(node.assigned_target() != Target::CL);

    // Configure node
    return CLFunctionFactory::create(&node, ctx);
}

arm_compute::Status CLDeviceBackend::validate_node(const INode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Validating CL node with ID : " << node.id() << std::endl);

    ARM_COMPUTE_UNUSED(node);

    return Status{};
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

    mm->set_allocator(&_allocator);

    return mm;
}
} // namespace backends
} // namespace graph2
} // namespace arm_compute