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
#ifndef __ARM_COMPUTE_GRAPH2_ALGORITHM_BFS_H__
#define __ARM_COMPUTE_GRAPH2_ALGORITHM_BFS_H__

#include "arm_compute/graph2/Graph.h"

#include <list>
#include <vector>

namespace arm_compute
{
namespace graph2
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

/** Breadth first search traversal
 *
 * @param g Graph to traverse
 *
 * @return A vector with the node id traversal order
 */
inline std::vector<NodeID> bfs(Graph &g)
{
    std::vector<NodeID> bfs_order_vector;

    // Created visited vector
    std::vector<bool> visited(g.nodes().size(), false);

    // Create BFS queue
    std::list<NodeID> queue;

    // Push inputs and mark as visited
    for(auto &input : g.inputs())
    {
        if(input != EmptyNodeID)
        {
            visited[input] = true;
            queue.push_back(input);
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
} // namespace graph2
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH2_ALGORITHM_BFS_H__ */
