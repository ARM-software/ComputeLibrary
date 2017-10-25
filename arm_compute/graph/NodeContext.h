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
#ifndef __ARM_COMPUTE_GRAPH_NODE_CONTEXT_H__
#define __ARM_COMPUTE_GRAPH_NODE_CONTEXT_H__

#include "arm_compute/core/Error.h"
#include "arm_compute/graph/NodeParameter.h"
#include "arm_compute/graph/Types.h"
#include "support/ToolchainSupport.h"

#include <map>
#include <memory>
#include <string>

namespace arm_compute
{
namespace graph
{
/** Node Context class
 *
 * Node context class is used to hold all the parameters required by a node to execute
 */
class NodeContext
{
public:
    /** Default Constructor
     *
     * @param[in] operation Name of the operation
     */
    NodeContext(OperationType operation)
        : _operation(operation), _target(TargetHint::DONT_CARE), _inputs(), _outputs(), _parameters() {};
    /** Sets the execution target of the node
     *
     * @param[in] target Execution target of the node
     */
    void set_target(TargetHint target);
    /** Adds an input tensor to the context
     *
     * @param[in] input Input to add
     */
    void add_input(arm_compute::ITensor *input);
    /** Adds and output to the context
     *
     * @param[in] output Output to add
     */
    void add_output(arm_compute::ITensor *output);
    /** Adds a parameter to the context
     *
     * @param[in] name      Parameter name
     * @param[in] parameter Parameter to add
     */
    template <typename T>
    void add_parameter(std::string name, T parameter);
    /** Returns the operation of this node.
     *
     * @return The operation type
     */
    OperationType operation() const;
    /** Returns the execution target of this node
     *
     * @return The execution target
     */
    TargetHint target() const;
    /** Returns input tensor of a given index
     *
     * @param[in] idx Index of the input tensor
     *
     * @return A pointer the requested input tensor else nullptr
     */
    arm_compute::ITensor *input(size_t idx) const;
    /** Returns output tensor of a given index
     *
     * @param[in] idx Index of the output tensor
     *
     * @return A pointer the requested output tensor else nullptr
     */
    arm_compute::ITensor *output(size_t idx) const;
    /** Returns the parameter with the given name
     *
     * @param[in] name Parameter name
     *
     * @return The requested parameter else an empty object
     */
    template <typename T>
    T parameter(std::string name) const;
    /** Returns number of inputs
     *
     * @return Number of inputs
     */
    size_t num_inputs() const;
    /** Returns number of output
     *
     * @return Number of outputs
     */
    size_t num_outputs() const;

private:
    OperationType                       _operation;
    TargetHint                          _target;
    std::vector<arm_compute::ITensor *> _inputs;
    std::vector<arm_compute::ITensor *> _outputs;
    std::map<std::string, std::unique_ptr<NodeParameterBase>> _parameters;
};

template <typename T>
inline void NodeContext::add_parameter(std::string name, T parameter)
{
    ARM_COMPUTE_ERROR_ON_MSG(_parameters.find(name) != _parameters.end(), "Parameter already exists!");
    _parameters[name] = support::cpp14::make_unique<NodeParameter<T>>(name, parameter);
}

template <typename T>
inline T NodeContext::parameter(std::string name) const
{
    auto it = _parameters.find(name);
    ARM_COMPUTE_ERROR_ON(it == _parameters.end());
    return static_cast<NodeParameter<T> *>(it->second.get())->value();
}
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_NODE_CONTEXT_H__ */
