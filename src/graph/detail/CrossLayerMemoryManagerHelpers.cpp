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
#include "arm_compute/graph/detail/CrossLayerMemoryManagerHelpers.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/GraphManager.h"
#include "arm_compute/graph/INode.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/graph/Types.h"
#include "arm_compute/graph/backends/BackendRegistry.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/utils/misc/Cast.h"

#include <algorithm>
#include <map>

namespace arm_compute
{
namespace graph
{
namespace detail
{
namespace
{
using HandleCountPair     = std::pair<ITensorHandle *, unsigned int>;
using HandleCounter       = std::map<HandleCountPair::first_type, HandleCountPair::second_type>;
using TargetHandleCounter = std::map<Target, HandleCounter>;

/** Holds managed IO tensor handles if a task */
struct TaskHandles
{
    std::vector<std::pair<ITensorHandle *, IMemoryGroup *>> input_handles  = {}; /**< Input handles to a task */
    std::vector<std::pair<ITensorHandle *, IMemoryGroup *>> output_handles = {}; /**< Output handles of a task */
};

/** Returns memory group depending on handle backend type
 *
 * @param[in] ctx    Graph context
 * @param[in] handle Tensor handle
 *
 * @return Memory groupb
 */
IMemoryGroup *get_memory_group_from_handle(GraphContext &ctx, ITensorHandle *handle)
{
    ARM_COMPUTE_ERROR_ON(handle == nullptr);
    return ctx.memory_management_ctx(handle->target())->cross_group.get();
}

/** Get handles of const tensors of graph
 *
 * @param[in] g Graph
 *
 * @return Handles of const tensors of graph
 */
std::set<ITensorHandle *> get_const_handles(const Graph &g)
{
    std::set<NodeType> const_node_types = { NodeType::Input, NodeType::Output, NodeType::Const };

    std::set<ITensorHandle *> const_tensors;

    auto &nodes = g.nodes();
    for(auto &node : nodes)
    {
        // If its a const node:
        if(node != nullptr && const_node_types.find(node->type()) != std::end(const_node_types))
        {
            // TODO (geopin01) : Create IO iterator wrappers
            // Add all its inputs / outputs to the list of constant handles
            for(unsigned int i = 0; i < node->num_inputs(); ++i)
            {
                if(node->input(i) != nullptr)
                {
                    const_tensors.insert(node->input(i)->handle()->parent_handle());
                }
            }
            for(unsigned int i = 0; i < node->num_outputs(); ++i)
            {
                if(node->output(i) != nullptr)
                {
                    const_tensors.insert(node->output(i)->handle()->parent_handle());
                }
            }
        }
    }

    return const_tensors;
}

/** Builds a list of all the transition handles (Handles that are used to link two nodes)
 *
 * @param[in] ctx           Graph context
 * @param[in] task          Workload task
 * @param[in] const_tensors Constant tensors
 *
 * @return List of transition handles
 */
TaskHandles get_transition_handles(GraphContext                    &ctx,
                                   ExecutionTask                   &task,
                                   const std::set<ITensorHandle *> &const_tensors)
{
    ARM_COMPUTE_ERROR_ON(task.node == nullptr || task.task == nullptr);
    INode &node = *task.node;

    TaskHandles transition_handles;

    // Add input handles
    for(unsigned int i = 0; i < node.input_edges().size(); ++i)
    {
        Edge *input_edge = node.input_edge(i);
        // If this input is the output of another node
        if(input_edge != nullptr && input_edge->tensor() != nullptr && const_tensors.find(input_edge->tensor()->handle()->parent_handle()) == std::end(const_tensors))
        {
            // Then add it to the list of transition buffers
            ITensorHandle *tensor_handle = input_edge->tensor()->handle()->parent_handle();
            IMemoryGroup *mm_group      = get_memory_group_from_handle(ctx, tensor_handle);
            transition_handles.input_handles.emplace_back(std::make_pair(tensor_handle, mm_group));
        }
    }

    // Add output handles
    for(unsigned int i = 0; i < node.num_outputs(); ++i)
    {
        Tensor *output_tensor = node.output(i);
        // If this output is used as an input for another node
        if(output_tensor != nullptr && const_tensors.find(output_tensor->handle()->parent_handle()) == std::end(const_tensors))
        {
            ITensorHandle *tensor_handle = output_tensor->handle()->parent_handle();
            IMemoryGroup *mm_group      = get_memory_group_from_handle(ctx, tensor_handle);
            transition_handles.output_handles.emplace_back(std::make_pair(tensor_handle, mm_group));
        }
    }

    return transition_handles;
}

/** Counts handles refcount for each input handle of each target
 *
 * @param[in]     task           Execution task containing the managed handles
 * @param[in,out] handle_counter Data structure that keeps the handles reference count
 */
void count_input_handles_per_target(const TaskHandles &task_handles, TargetHandleCounter &handle_counter)
{
    for(const auto &handle : task_handles.input_handles)
    {
        ITensorHandle *key            = handle.first;
        HandleCounter &target_counter = handle_counter[key->target()];
        if(target_counter.find(key) == std::end(target_counter))
        {
            target_counter.emplace(std::make_pair(key, 1));
        }
        else
        {
            ++target_counter[key];
        }
    }
}

/** Calculates the lifetime of each tensor handle
 *
 * @param[in, out] tasks_handles Tensor handles for each task
 * @param[in]      hc            Data structure that keeps the handles reference count
 */
void configure_handle_lifetime(std::vector<TaskHandles> &tasks_handles, const HandleCounter &hc)
{
    // Identify max number of tensors in flight
    HandleCounter tensors_in_flight;

    // Acquires the given handles and sets them as in flight if they aren't already
    auto acquire = [&](std::vector<std::pair<ITensorHandle *, IMemoryGroup *>> &handles)
    {
        for(auto &handle : handles)
        {
            ITensorHandle *parent_handle = handle.first;
            ARM_COMPUTE_ERROR_ON(parent_handle == nullptr);
            // If the tensor is not already in flight:
            if(tensors_in_flight.find(parent_handle) == std::end(tensors_in_flight))
            {
                ARM_COMPUTE_ERROR_ON(hc.find(parent_handle) == std::end(hc));
                // Then add it to the list of in flight tensors
                tensors_in_flight.insert(std::make_pair(parent_handle, hc.at(parent_handle)));
                // Start of allocation's lifetime
                parent_handle->manage(handle.second);
            }
        }
    };

    for(auto &task_handle : tasks_handles)
    {
        // Marking all the input and output tensors of the task as in flight
        acquire(task_handle.input_handles);
        acquire(task_handle.output_handles);

        // Releasing the input tensors
        for(auto &input_handle : task_handle.input_handles)
        {
            ITensorHandle *ihandle = input_handle.first;
            ARM_COMPUTE_ERROR_ON(ihandle == nullptr);
            ARM_COMPUTE_ERROR_ON(tensors_in_flight.find(ihandle) == std::end(tensors_in_flight));
            --tensors_in_flight[ihandle];
            if(tensors_in_flight[ihandle] <= 0)
            {
                // Remove tensor for tensors in flight
                tensors_in_flight.erase(ihandle);
                // End of allocation's lifetime
                ihandle->allocate();
            }
        }
    }
}
} // namespace

void configure_transition_manager(Graph &g, GraphContext &ctx, ExecutionWorkload &workload)
{
    // Get const tensors (un-managed)
    std::set<ITensorHandle *> const_tensors = get_const_handles(g);

    std::vector<TaskHandles> tasks_handles;
    TargetHandleCounter      target_handle_count;

    // Count handles
    for(auto &task : workload.tasks)
    {
        // Populates IO handles
        tasks_handles.push_back(get_transition_handles(ctx, task, const_tensors));

        // Count handles
        count_input_handles_per_target(tasks_handles.back(), target_handle_count);
    }

    // Setup memory managers
    for(auto &hc : target_handle_count)
    {
        MemoryManagerContext *mm_ctx = ctx.memory_management_ctx(hc.first);
        if(mm_ctx != nullptr)
        {
            if(mm_ctx->cross_mm != nullptr && mm_ctx->cross_group != nullptr)
            {
                // Manage and allocate tensors
                configure_handle_lifetime(tasks_handles, hc.second);
            }
        }
    }
}
} // namespace detail
} // namespace graph
} // namespace arm_compute
