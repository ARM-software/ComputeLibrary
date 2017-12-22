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
#include "arm_compute/graph2/GraphManager.h"

#include "arm_compute/graph2/Graph.h"
#include "arm_compute/graph2/GraphContext.h"
#include "arm_compute/graph2/Logger.h"
#include "arm_compute/graph2/PassManager.h"
#include "arm_compute/graph2/Utils.h"
#include "arm_compute/graph2/detail/ExecutionHelpers.h"

namespace arm_compute
{
namespace graph2
{
GraphManager::GraphManager()
    : _workloads()
{
    detail::default_initialize_backends();
}

void GraphManager::finalize_graph(Graph &graph, GraphContext &ctx, PassManager &pm, Target target)
{
    // Setup graph context if not done manually
    setup_default_graph_context(ctx);

    // Check if graph has been registered
    ARM_COMPUTE_ERROR_ON_MSG(_workloads.find(graph.id()) != std::end(_workloads), "Graph is already registered!");

    // Force target to all graph construct
    // TODO (geopin01) : Support heterogeneous execution
    Target forced_target = is_target_supported(target) ? target : get_default_target();
    force_target_to_graph(graph, forced_target);

    // Configure all tensors
    detail::configure_all_tensors(graph);

    // Apply all mutating passes
    pm.run_all(graph);

    // TODO (geopin01): Perform a graph validation

    // Perform topological sort
    // FIXME : Sort nodes and pass sorted indices in configure all nodes

    // Configure all nodes
    auto workload = detail::configure_all_nodes(graph, ctx);
    ARM_COMPUTE_ERROR_ON_MSG(workload.tasks.empty(), "Could not configure all nodes!");

    // Allocate all tensors
    detail::allocate_all_tensors(graph);

    // Call accessors on all Const nodes
    detail::call_all_const_node_accessors(graph);

    _workloads.insert(std::make_pair(graph.id(), std::move(workload)));
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Created workload for graph with ID : " << graph.id().get() << std::endl);

    // Finalize Graph context
    ctx.finalize();
}

void GraphManager::execute_graph(Graph &graph)
{
    // Check if graph is finalized
    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");

    // Call input accessors
    detail::call_all_input_node_accessors(it->second);

    // Run graph
    detail::call_all_tasks(it->second);

    // Call output accessors
    detail::call_all_output_node_accessors(it->second);
}

void GraphManager::invalidate_graph(Graph &graph)
{
    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");

    _workloads.erase(it);
}
} // namespace graph2
} // namespace arm_compute