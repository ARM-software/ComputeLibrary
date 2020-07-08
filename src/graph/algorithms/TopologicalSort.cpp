/*
 * Copyright (c) 2018 Arm Limited.
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
#include "arm_compute/graph/algorithms/TopologicalSort.h"

#include "arm_compute/graph/Graph.h"

#include "arm_compute/core/utils/misc/Iterable.h"

#include <list>
#include <stack>

namespace arm_compute
{
namespace graph
{
namespace detail
{
/** Checks if all the input dependencies of a node have been visited
 *
 * @param[in] node    Node to check
 * @param[in] visited Vector that contains the visited information
 *
 * @return True if all inputs dependencies have been visited else false
 */
inline bool all_inputs_are_visited(const INode *node, const std::vector<bool> &visited)
{
    ARM_COMPUTE_ERROR_ON(node == nullptr);
    const Graph *graph = node->graph();
    ARM_COMPUTE_ERROR_ON(graph == nullptr);

    bool are_all_visited = true;
    for(const auto &input_edge_id : node->input_edges())
    {
        if(input_edge_id != EmptyNodeID)
        {
            const Edge *input_edge = graph->edge(input_edge_id);
            ARM_COMPUTE_ERROR_ON(input_edge == nullptr);
            ARM_COMPUTE_ERROR_ON(input_edge->producer() == nullptr);
            if(!visited[input_edge->producer_id()])
            {
                are_all_visited = false;
                break;
            }
        }
    }

    return are_all_visited;
}
} // namespace detail

std::vector<NodeID> bfs(Graph &g)
{
    std::vector<NodeID> bfs_order_vector;

    // Created visited vector
    std::vector<bool> visited(g.nodes().size(), false);

    // Create BFS queue
    std::list<NodeID> queue;

    // Push inputs and mark as visited
    for(auto &input : g.nodes(NodeType::Input))
    {
        if(input != EmptyNodeID)
        {
            visited[input] = true;
            queue.push_back(input);
        }
    }

    // Push const nodes and mark as visited
    for(auto &const_node : g.nodes(NodeType::Const))
    {
        if(const_node != EmptyNodeID)
        {
            visited[const_node] = true;
            queue.push_back(const_node);
        }
    }

    // Iterate over vector and edges
    while(!queue.empty())
    {
        // Dequeue a node from queue and process
        NodeID n = queue.front();
        bfs_order_vector.push_back(n);
        queue.pop_front();

        const INode *node = g.node(n);
        ARM_COMPUTE_ERROR_ON(node == nullptr);
        for(const auto &eid : node->output_edges())
        {
            const Edge *e = g.edge(eid);
            ARM_COMPUTE_ERROR_ON(e == nullptr);
            if(!visited[e->consumer_id()] && detail::all_inputs_are_visited(e->consumer(), visited))
            {
                visited[e->consumer_id()] = true;
                queue.push_back(e->consumer_id());
            }
        }
    }

    return bfs_order_vector;
}

std::vector<NodeID> dfs(Graph &g)
{
    std::vector<NodeID> dfs_order_vector;

    // Created visited vector
    std::vector<bool> visited(g.nodes().size(), false);

    // Create DFS stack
    std::stack<NodeID> stack;

    // Push inputs and mark as visited
    for(auto &input : g.nodes(NodeType::Input))
    {
        if(input != EmptyNodeID)
        {
            visited[input] = true;
            stack.push(input);
        }
    }

    // Push const nodes and mark as visited
    for(auto &const_node : g.nodes(NodeType::Const))
    {
        if(const_node != EmptyNodeID)
        {
            visited[const_node] = true;
            stack.push(const_node);
        }
    }

    // Iterate over vector and edges
    while(!stack.empty())
    {
        // Pop a node from stack and process
        NodeID n = stack.top();
        dfs_order_vector.push_back(n);
        stack.pop();

        // Mark node as visited
        if(!visited[n])
        {
            visited[n] = true;
        }

        const INode *node = g.node(n);
        ARM_COMPUTE_ERROR_ON(node == nullptr);
        // Reverse iterate to push branches from right to left and pop on the opposite order
        for(const auto &eid : arm_compute::utils::iterable::reverse_iterate(node->output_edges()))
        {
            const Edge *e = g.edge(eid);
            ARM_COMPUTE_ERROR_ON(e == nullptr);
            if(!visited[e->consumer_id()] && detail::all_inputs_are_visited(e->consumer(), visited))
            {
                stack.push(e->consumer_id());
            }
        }
    }

    return dfs_order_vector;
}
} // namespace graph
} // namespace arm_compute