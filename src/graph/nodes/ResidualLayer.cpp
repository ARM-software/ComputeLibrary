/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/graph/nodes/ResidualLayer.h"

#include "arm_compute/graph/Error.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/NodeContext.h"
#include "arm_compute/graph/OperationRegistry.h"
#include "arm_compute/graph/SubGraph.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/runtime/IFunction.h"
#include "support/ToolchainSupport.h"
#include "utils/Utils.h"

#include <memory>
#include <tuple>
#include <vector>

using namespace arm_compute::graph;

/** Residual function */
class ResidualFunction final : public arm_compute::IFunction
{
public:
    /** Default Constructor */
    ResidualFunction(GraphContext &ctx, ITensorObject *output)
        : _ctx(ctx), _input(nullptr), _output(output), _func(nullptr), _graphs(), _graph_outputs()
    {
    }

    /** Prevent instances from being copy constructed */
    ResidualFunction(const ResidualFunction &) = delete;
    /** Prevent instances from being copy assigned */
    const ResidualFunction &operator=(const ResidualFunction &) = delete;
    /** Prevent instances from being move constructed */
    ResidualFunction(ResidualFunction &&) = delete;
    /** Prevent instances from being move assigned */
    ResidualFunction &operator=(ResidualFunction &&) = delete;
    /** Default destructor */
    ~ResidualFunction() override = default;

    /** Set the input (when using only one sub graph)
     *
     * @param[in] input Input to set
     */
    void set_input(std::unique_ptr<ITensorObject> input)
    {
        _input = std::move(input);
    }

    /** Registers graph to be executed by the residual function
     *
     * @param[in] graph  Graph to register
     * @param[in] output Output to register
     */
    void register_graph(std::unique_ptr<Graph> graph, std::unique_ptr<ITensorObject> output)
    {
        _graphs.push_back(std::move(graph));
        _graph_outputs.push_back(std::move(output));
    }

    /** Configure the function */
    void configure()
    {
        ARM_COMPUTE_ERROR_ON(_graphs.size() < 1 || _graphs.size() > 2);
        TargetHint target_hint = _ctx.hints().target_hint();

        // Create node context
        NodeContext node_ctx(OperationType::ArithmeticAddition);
        node_ctx.set_target(target_hint);

        if(_graphs.size() == 1)
        {
            arm_compute::ITensor *in = _input->tensor();
            node_ctx.add_input(in);
        }

        for(auto &o : _graph_outputs)
        {
            arm_compute::ITensor *in = o->tensor();
            node_ctx.add_input(in);
        }

        arm_compute::ITensor *out = _output->tensor();
        auto_init_if_empty(*out->info(), *_graph_outputs[0]->tensor()->info());
        node_ctx.add_output(out);

        _func = OperationRegistry::get().find_operation(OperationType::ArithmeticAddition, target_hint)->configure(node_ctx);

        for(auto &o : _graph_outputs)
        {
            o->allocate();
        }
    }

    // Inherited methods overriden:
    void run() override
    {
        ARM_COMPUTE_ERROR_ON(_graphs.size() < 1 || _graphs.size() > 2);

        for(auto &g : _graphs)
        {
            ARM_COMPUTE_ERROR_ON(g.get() == nullptr);
            g->run();
        }

        _func->run();
    }

private:
    GraphContext                                _ctx;
    std::unique_ptr<ITensorObject>              _input;
    ITensorObject                              *_output;
    std::unique_ptr<arm_compute::IFunction>     _func;
    std::vector<std::unique_ptr<Graph>>         _graphs;
    std::vector<std::unique_ptr<ITensorObject>> _graph_outputs;
};

std::unique_ptr<arm_compute::IFunction> ResidualLayer::instantiate_node(GraphContext &ctx, ITensorObject *input, ITensorObject *output)
{
    ARM_COMPUTE_ERROR_ON_UNALLOCATED_TENSOR_OBJECT(input, output);
    ARM_COMPUTE_ERROR_ON(dynamic_cast<Tensor *>(input) == nullptr);
    ARM_COMPUTE_ERROR_ON(dynamic_cast<Tensor *>(output) == nullptr);

    // Create residual function
    auto func = arm_compute::support::cpp14::make_unique<ResidualFunction>(ctx, output);

    if(_sub_graphs.size() == 1)
    {
        std::unique_ptr<ITensorObject> original_in;
        original_in = arm_compute::support::cpp14::make_unique<SubTensor>(*dynamic_cast<Tensor *>(input),
                                                                          input->tensor()->info()->tensor_shape(),
                                                                          Coordinates());
        func->set_input(std::move(original_in));
    }

    // Constuct all sub-graphs given the input/output
    for(auto &sg : _sub_graphs)
    {
        ARM_COMPUTE_ERROR_ON(sg.get() == nullptr);

        // IO buffers
        std::unique_ptr<ITensorObject> in;
        std::unique_ptr<ITensorObject> out;
        std::unique_ptr<ITensorObject> func_in;

        // Create input sub-tensor
        if(!sg->has_input())
        {
            in = arm_compute::support::cpp14::make_unique<SubTensor>(*dynamic_cast<Tensor *>(input),
                                                                     input->tensor()->info()->tensor_shape(),
                                                                     Coordinates());
        }

        // Create output sub-tensor
        if(!sg->has_output())
        {
            ITensorInfo *info = input->tensor()->info();
            func_in           = arm_compute::support::cpp14::make_unique<Tensor>(TensorInfo(info->num_channels(), info->data_type(), info->fixed_point_position()));
            func_in->set_target(ctx.hints().target_hint());
            out = arm_compute::support::cpp14::make_unique<SubTensor>(func_in->tensor(),
                                                                      TensorShape(),
                                                                      Coordinates(0, 0, 0),
                                                                      func_in->target(),
                                                                      true);
        }

        // Construct sub_graph
        auto g = sg->construct(ctx, std::move(in), std::move(out));

        // Register graph to function
        func->register_graph(std::move(g), std::move(func_in));
    }

    func->configure();

    return std::move(func);
}
