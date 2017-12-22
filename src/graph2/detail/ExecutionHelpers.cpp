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
#include "arm_compute/graph2/detail/ExecutionHelpers.h"

#include "arm_compute/graph2/Graph.h"
#include "arm_compute/graph2/GraphContext.h"
#include "arm_compute/graph2/GraphManager.h"
#include "arm_compute/graph2/Tensor.h"
#include "arm_compute/graph2/backends/BackendRegistry.h"

namespace arm_compute
{
namespace graph2
{
namespace detail
{
void default_initialize_backends()
{
    for(const auto &backend : backends::BackendRegistry::get().backends())
    {
        backend.second->initialize_backend();
    }
}

void configure_all_tensors(Graph &g)
{
    auto &tensors = g.tensors();

    for(auto &tensor : tensors)
    {
        if(tensor)
        {
            Target target  = tensor->desc().target;
            auto   backend = backends::BackendRegistry::get().find_backend(target);
            ARM_COMPUTE_ERROR_ON_MSG(!backend, "Requested backend doesn't exist!");
            auto handle = backend->create_tensor(*tensor);
            ARM_COMPUTE_ERROR_ON_MSG(!backend, "Couldn't create backend handle!");
            tensor->set_handle(std::move(handle));
        }
    }
}

void allocate_all_tensors(Graph &g)
{
    auto &tensors = g.tensors();

    for(auto &tensor : tensors)
    {
        if(tensor && !tensor->bound_edges().empty())
        {
            ARM_COMPUTE_ERROR_ON_MSG(!tensor->handle(), "Tensor handle is not configured!");
            tensor->handle()->allocate();
        }
    }
}

ExecutionWorkload configure_all_nodes(Graph &g, GraphContext &ctx)
{
    ExecutionWorkload workload;
    auto             &nodes = g.nodes();

    // Create tasks
    for(auto &node : nodes)
    {
        if(node != nullptr)
        {
            Target assigned_target = node->assigned_target();
            auto   backend         = backends::BackendRegistry::get().find_backend(assigned_target);
            ARM_COMPUTE_ERROR_ON_MSG(!backend, "Requested backend doesn't exist!");
            auto func = backend->configure_node(*node, ctx);
            if(func != nullptr)
            {
                ExecutionTask task;
                task.task = std::move(func);
                task.node = node.get();
                workload.tasks.push_back(std::move(task));
            }
        }
    }

    // Add inputs and outputs
    for(auto &node : nodes)
    {
        if(node != nullptr && node->type() == NodeType::Input)
        {
            workload.inputs.push_back(node->output(0));
        }

        if(node != nullptr && node->type() == NodeType::Output)
        {
            workload.outputs.push_back(node->input(0));
            continue;
        }
    }

    return workload;
}

void call_tensor_accessor(Tensor *tensor)
{
    ARM_COMPUTE_ERROR_ON(!tensor);
    tensor->call_accessor();
}

void call_all_const_node_accessors(Graph &g)
{
    auto &nodes = g.nodes();

    for(auto &node : nodes)
    {
        if(node != nullptr && node->type() == NodeType::Const)
        {
            call_tensor_accessor(node->output(0));
        }
    }
}

void call_all_input_node_accessors(ExecutionWorkload &workload)
{
    for(auto &input : workload.inputs)
    {
        if(input != nullptr)
        {
            input->call_accessor();
        }
    }
}

void call_all_tasks(ExecutionWorkload &workload)
{
    for(auto &task : workload.tasks)
    {
        task();
    }
}

void call_all_output_node_accessors(ExecutionWorkload &workload)
{
    for(auto &output : workload.outputs)
    {
        if(output != nullptr)
        {
            output->call_accessor();
        }
    }
}
} // namespace detail
} // namespace graph2
} // namespace arm_compute