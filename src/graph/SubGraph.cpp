/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/graph/SubGraph.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INode.h"
#include "arm_compute/graph/Tensor.h"

using namespace arm_compute::graph;

SubGraph::SubGraph()
    : _nodes(), _input(nullptr), _output(nullptr)
{
}

void SubGraph::add_node(std::unique_ptr<INode> node)
{
    _nodes.push_back(std::move(node));
}

void SubGraph::add_tensor_object(std::unique_ptr<ITensorObject> tensor)
{
    // If it's the first Tensor added then it will be the input of the Graph.
    if(_input == nullptr)
    {
        _input = std::move(tensor);
    }
    else
    {
        _output = std::move(tensor);
    }
}

std::unique_ptr<Graph> SubGraph::construct(const GraphContext &ctx, std::unique_ptr<ITensorObject> input, std::unique_ptr<ITensorObject> output)
{
    auto graph = arm_compute::support::cpp14::make_unique<Graph>();

    // Set hint
    graph->hints() = ctx.hints();

    // Configure input
    if(_input == nullptr)
    {
        _input = std::move(input);
    }
    graph->add_tensor_object(std::move(_input));

    // Construct nodes
    for(auto &node : _nodes)
    {
        graph->add_node(std::move(node));
    }

    // Configure output
    if(_output == nullptr)
    {
        _output = std::move(output);
    }
    graph->add_tensor_object(std::move(_output));

    return graph;
}

bool SubGraph::has_input() const
{
    return _input != nullptr;
}

bool SubGraph::has_output() const
{
    return _output != nullptr;
}

SubGraph &arm_compute::graph::operator<<(SubGraph &graph, Tensor &&tensor)
{
    graph.add_tensor_object(arm_compute::support::cpp14::make_unique<Tensor>(std::move(tensor)));
    return graph;
}

SubGraph &arm_compute::graph::operator<<(SubGraph &graph, SubTensor &&sub_tensor)
{
    graph.add_tensor_object(arm_compute::support::cpp14::make_unique<SubTensor>(std::move(sub_tensor)));
    return graph;
}
