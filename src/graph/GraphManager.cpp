/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "arm_compute/graph/GraphManager.h"

#include "arm_compute/graph/algorithms/TopologicalSort.h"
#include "arm_compute/graph/detail/CrossLayerMemoryManagerHelpers.h"
#include "arm_compute/graph/detail/ExecutionHelpers.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/PassManager.h"
#include "arm_compute/graph/TypePrinter.h"
#include "arm_compute/graph/Utils.h"

#include "src/common/utils/Log.h"

namespace arm_compute
{
namespace graph
{
GraphManager::GraphManager() : _workloads()
{
}

void GraphManager::finalize_graph(Graph &graph, GraphContext &ctx, PassManager &pm, Target target)
{
    ARM_COMPUTE_LOG_INFO_WITH_FUNCNAME_ACL("Initiate graph configuration!");

    // Check if graph has been registered
    if (_workloads.find(graph.id()) != std::end(_workloads))
    {
        ARM_COMPUTE_ERROR("Graph is already registered!");
    }

    // Apply IR mutating passes
    pm.run_type(graph, IGraphMutator::MutationType::IR);

    // Force target to all graph construct
    Target forced_target = target;

    // In case CLVK is selected, use the CL backend and
    // update config
    if (target == Target::CLVK)
    {
        forced_target       = Target::CL;
        GraphConfig config  = ctx.config();
        config.backend_type = CLBackendType::Clvk;

        ctx.set_config(config);
    }

    if (!is_target_supported(target))
    {
        forced_target = get_default_target();
        ARM_COMPUTE_LOG_GRAPH_INFO("Switching target from " << target << " to " << forced_target << std::endl);
    }
    force_target_to_graph(graph, forced_target);

    // Setup backend context
    setup_requested_backend_context(ctx, forced_target);

    // Configure all tensors
    detail::configure_all_tensors(graph);

    // Apply backend mutating passes
    pm.run_type(graph, IGraphMutator::MutationType::Backend);

    // Perform topological sort
    std::vector<NodeID> topological_sorted_nodes = dfs(graph);

    // Validate all nodes
    detail::validate_all_nodes(graph);

    // Configure all nodes
    auto workload = detail::configure_all_nodes(graph, ctx, topological_sorted_nodes);
    ARM_COMPUTE_ERROR_ON_MSG(workload.tasks.empty(), "Could not configure all nodes!");

    // Allocate const tensors and call accessors
    detail::allocate_const_tensors(graph);
    detail::call_all_const_node_accessors(graph);

    // Prepare graph
    detail::prepare_all_tasks(workload);

    // Setup tensor memory (Allocate all tensors or setup transition manager)
    if (ctx.config().use_transition_memory_manager)
    {
        detail::configure_transition_manager(graph, ctx, workload);
    }
    else
    {
        detail::allocate_all_tensors(graph);
    }

    // Finalize Graph context
    ctx.finalize();

    // Register graph
    _workloads.insert(std::make_pair(graph.id(), std::move(workload)));
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Created workload for graph with ID : " << graph.id() << std::endl);
}

void GraphManager::execute_graph(Graph &graph)
{
    ARM_COMPUTE_LOG_INFO_WITH_FUNCNAME_ACL("Initiate graph execution!");

    // Check if graph is finalized
    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");
    
    std::cout << "total nodes: ";
    std::cout << it->second.graph->nodes().size() <<std::endl;

     std::cout <<it->second.graph->nodes()[0].get()->id() <<std::endl;
     std::cout <<it->second.graph->nodes()[0].get()->type() <<std::endl;
     std::cout << it->second.graph->nodes()[1].get()->id() <<std::endl;
     std::cout <<it->second.graph->nodes()[1].get()->type() <<std::endl;

    std::cout << "total edges: ";
    std::cout << it->second.graph->edges().size() <<std::endl;


    std::cout << "input number: ";
    std::cout << it->second.inputs.size() <<std::endl;

    std::cout << "output number: ";
    std::cout << it->second.outputs.size() <<std::endl;

    while (true)
    {
        // Call input accessors
        if (!detail::call_all_input_node_accessors(it->second))
        {
            return;
        }
        
        // Run graph
        detail::call_all_tasks(it->second);

        // Call output accessors
        if (!detail::call_all_output_node_accessors(it->second))
        {
            return;
        }

    }
}

void GraphManager::invalidate_graph(Graph &graph)
{
    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");

    _workloads.erase(it);
}
} // namespace graph
} // namespace arm_compute
