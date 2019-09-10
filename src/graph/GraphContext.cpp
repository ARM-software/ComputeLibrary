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
#include "arm_compute/graph/GraphContext.h"

#include "arm_compute/graph.h"
#include "arm_compute/graph/Utils.h"
#include "arm_compute/graph/backends/BackendRegistry.h"

namespace arm_compute
{
namespace graph
{
GraphContext::GraphContext()
    : _config(), _memory_managers(), _weights_managers()
{
}

GraphContext::~GraphContext()
{
    _memory_managers.clear();
    _weights_managers.clear();
    release_default_graph_context(*this);
}

const GraphConfig &GraphContext::config() const
{
    return _config;
}

void GraphContext::set_config(const GraphConfig &config)
{
    _config = config;
}

bool GraphContext::insert_memory_management_ctx(MemoryManagerContext &&memory_ctx)
{
    Target target = memory_ctx.target;
    if(target == Target::UNSPECIFIED || _memory_managers.find(target) != std::end(_memory_managers))
    {
        return false;
    }

    _memory_managers[target] = std::move(memory_ctx);
    return true;
}

MemoryManagerContext *GraphContext::memory_management_ctx(Target target)
{
    return (_memory_managers.find(target) != std::end(_memory_managers)) ? &_memory_managers[target] : nullptr;
}

std::map<Target, MemoryManagerContext> &GraphContext::memory_managers()
{
    return _memory_managers;
}

bool GraphContext::insert_weights_management_ctx(WeightsManagerContext &&weights_managers)
{
    Target target = weights_managers.target;

    if(target != Target::NEON || _weights_managers.find(target) != std::end(_weights_managers))
    {
        return false;
    }

    _weights_managers[target] = std::move(weights_managers);

    return true;
}

WeightsManagerContext *GraphContext::weights_management_ctx(Target target)
{
    return (_weights_managers.find(target) != std::end(_weights_managers)) ? &_weights_managers[target] : nullptr;
}

std::map<Target, WeightsManagerContext> &GraphContext::weights_managers()
{
    return _weights_managers;
}

void GraphContext::finalize()
{
    const size_t num_pools = 1;
    for(auto &mm_obj : _memory_managers)
    {
        ARM_COMPUTE_ERROR_ON(!mm_obj.second.allocator);

        // Finalize intra layer memory manager
        if(mm_obj.second.intra_mm != nullptr)
        {
            mm_obj.second.intra_mm->populate(*mm_obj.second.allocator, num_pools);
        }
        // Finalize cross layer memory manager
        if(mm_obj.second.cross_mm != nullptr)
        {
            mm_obj.second.cross_mm->populate(*mm_obj.second.allocator, num_pools);
        }
    }
}
} // namespace graph
} // namespace arm_compute
