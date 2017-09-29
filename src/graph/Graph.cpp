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
#include "arm_compute/graph/Graph.h"

#include "arm_compute/graph/CL/CLMap.h"
#include "arm_compute/graph/CL/CLUnmap.h"
#include "arm_compute/graph/INode.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/Tensor.h"

using namespace arm_compute::graph;

struct Stage
{
    Tensor                                 *_input;
    Tensor                                 *_output;
    std::unique_ptr<arm_compute::IFunction> _function;
};

struct Graph::Private
{
public:
    /** Finalizes the current node's configuration
     *
     * @param _next_hint Device execution hint
     */
    void configure(Hint _next_hint);

    /** Sets whether to enable information print out
     *
     * @param[in] is_enabled Set to true if need info printed out
     */
    void set_info_enablement(bool is_enabled);

    std::vector<Stage>                   _pipeline{};
    std::vector<std::unique_ptr<Tensor>> _tensors{};
    std::vector<std::unique_ptr<INode>>  _nodes{};
    Hint                                 _current_hint{ Hint::DONT_CARE };
    Hint                                 _next_hint{ Hint::DONT_CARE };
    std::unique_ptr<Tensor>              _graph_input{ nullptr };
    std::unique_ptr<Tensor>              _graph_output{ nullptr };
    std::unique_ptr<INode>               _current_node{ nullptr };
    Tensor                              *_current_output{ nullptr };
    bool                                 _info_enabled{ false };

private:
    Tensor *_current_input{ nullptr };
    Hint    _previous_hint{ Hint::DONT_CARE };
};

Graph::~Graph() //NOLINT
{
    //Can't use =default because the destructor must be defined after Graph::Private's definition
}

Graph::Graph()
    : _pimpl{ new Private() }
{
}

void Graph::run()
{
    while(true)
    {
        if(!_pimpl->_graph_input->call_accessor())
        {
            return;
        }

        for(auto &stage : _pimpl->_pipeline)
        {
            stage._function->run();
        }

        if(!_pimpl->_graph_output->call_accessor())
        {
            return;
        }
    }
}

//Finalize current node's configuration
void Graph::Private::configure(Hint _next_hint)
{
    ARM_COMPUTE_ERROR_ON(_current_node == nullptr);
    ARM_COMPUTE_ERROR_ON(_graph_input == nullptr);

    // Is it the first node of the graph ?
    if(_current_input == nullptr)
    {
        _graph_input->set_target(_current_hint);
        _current_input = _graph_input.get();
        _previous_hint = _current_hint; // For the first node just assume the previous node was of the same type as this one
    }

    //Automatic output configuration ?
    if(_current_output == nullptr)
    {
        _tensors.push_back(arm_compute::support::cpp14::make_unique<Tensor>(TensorInfo()));
        _current_output = _tensors.back().get();
    }

    // If either the writer or reader node needs OpenCL then use OpenCL memory:
    if((_next_hint == Hint::OPENCL || _current_hint == Hint::OPENCL))
    {
        _current_output->set_target(Hint::OPENCL);
    }
    else
    {
        _current_output->set_target(Hint::NEON);
    }

    // Map input if needed
    std::unique_ptr<arm_compute::IFunction> func = _current_node->instantiate_node(_current_hint, _current_input->tensor(), _current_output->tensor());
    _current_input->allocate();

    if(_current_input->target() == Hint::OPENCL)
    {
        if(_previous_hint == Hint::NEON)
        {
            ARM_COMPUTE_ERROR_ON(_current_hint == Hint::NEON);
            _pipeline.push_back({ _current_input, _current_input, arm_compute::support::cpp14::make_unique<CLUnmap>(_current_input) });
        }
        if(_current_hint == Hint::NEON)
        {
            ARM_COMPUTE_ERROR_ON(_previous_hint == Hint::NEON);
            _pipeline.push_back({ _current_input, _current_input, arm_compute::support::cpp14::make_unique<CLMap>(_current_input, true) });
        }
    }

    _pipeline.push_back({ _current_input, _current_output, std::move(func) });

    _current_input  = _current_output;
    _current_output = nullptr;
    _previous_hint  = _current_hint;
    _current_hint   = _next_hint;
}

void Graph::Private::set_info_enablement(bool is_enabled)
{
    _info_enabled = is_enabled;
}

void Graph::add_node(std::unique_ptr<INode> node)
{
    ARM_COMPUTE_ERROR_ON_MSG(_pimpl->_graph_input == nullptr, "The graph's input must be set before the first node is added");
    ARM_COMPUTE_ERROR_ON_MSG(_pimpl->_graph_output != nullptr, "Nothing can be added after the output tensor");
    //Trigger the creation of the current Node:

    Hint _next_hint = node->override_hint(_pimpl->_next_hint);
    ARM_COMPUTE_ERROR_ON(_next_hint == Hint::DONT_CARE);
    if(_pimpl->_current_node)
    {
        //Finalize the previous Node:
        _pimpl->configure(_pimpl->_next_hint);

        if(_pimpl->_info_enabled)
        {
            _pimpl->_current_node->print_info();
        }
    }
    else
    {
        // If that's the first node then use the same Hint before and after the node.
        _pimpl->_current_hint = _next_hint;
    }
    if(_pimpl->_current_node)
    {
        _pimpl->_nodes.push_back(std::move(_pimpl->_current_node));
    }
    _pimpl->_current_node = std::move(node);
}
void Graph::set_hint(Hint hint)
{
    _pimpl->_next_hint = hint;
}

void Graph::set_info_enablement(bool is_enabled)
{
    _pimpl->set_info_enablement(is_enabled);
}

//Add a tensor with an Accessor (i.e either the input or output of the graph)
void Graph::add_tensor(std::unique_ptr<Tensor> tensor)
{
    // If it's the first Tensor added then it will be the input of the Graph.
    if(_pimpl->_graph_input == nullptr)
    {
        ARM_COMPUTE_ERROR_ON(_pimpl->_graph_output != nullptr);
        ARM_COMPUTE_ERROR_ON(_pimpl->_current_node != nullptr);
        _pimpl->_graph_input = std::move(tensor);
    }
    else
    {
        // Else it will be the output of the Graph
        ARM_COMPUTE_ERROR_ON(_pimpl->_graph_output != nullptr);
        ARM_COMPUTE_ERROR_ON(_pimpl->_current_node == nullptr);
        _pimpl->_graph_output   = std::move(tensor);
        _pimpl->_current_output = _pimpl->_graph_output.get();

        // Finalize the graph by configuring the last Node of the graph:
        _pimpl->configure(_pimpl->_current_hint); // Ignore _next_hint as this is the last node, and just use the same hint as before this node.
        _pimpl->_graph_output->allocate();
    }
}

void Graph::set_temp(TensorInfo &&tmp)
{
    ARM_COMPUTE_ERROR_ON(_pimpl->_graph_input == nullptr);
    ARM_COMPUTE_ERROR_ON(_pimpl->_graph_output != nullptr);
    ARM_COMPUTE_ERROR_ON_MSG(_pimpl->_current_output != nullptr, "TensorInfo for temporary tensor already set");

    _pimpl->_tensors.push_back(arm_compute::support::cpp14::make_unique<Tensor>(std::move(tmp)));
    _pimpl->_current_output = _pimpl->_tensors.back().get();
}

Graph &arm_compute::graph::operator<<(Graph &graph, TensorInfo &&info)
{
    graph.set_temp(std::move(info));
    return graph;
}

Graph &arm_compute::graph::operator<<(Graph &graph, Tensor &&tensor)
{
    graph.add_tensor(arm_compute::support::cpp14::make_unique<Tensor>(std::move(tensor)));
    return graph;
}

Graph &arm_compute::graph::operator<<(Graph &graph, Hint hint)
{
    graph.set_hint(hint);
    return graph;
}
