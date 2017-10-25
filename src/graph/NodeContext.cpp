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
#include "arm_compute/graph/NodeContext.h"

using namespace arm_compute::graph;

void NodeContext::set_target(TargetHint target)
{
    _target = target;
}

void NodeContext::add_input(arm_compute::ITensor *input)
{
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    _inputs.emplace_back(input);
}

void NodeContext::add_output(arm_compute::ITensor *output)
{
    ARM_COMPUTE_ERROR_ON(output == nullptr);
    _outputs.emplace_back(output);
}

OperationType NodeContext::operation() const
{
    return _operation;
}

TargetHint NodeContext::target() const
{
    return _target;
}

arm_compute::ITensor *NodeContext::input(size_t idx) const
{
    ARM_COMPUTE_ERROR_ON(idx >= _inputs.size());
    return _inputs[idx];
}

arm_compute::ITensor *NodeContext::output(size_t idx) const
{
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());
    return _outputs[idx];
}

size_t NodeContext::num_inputs() const
{
    return _inputs.size();
}

size_t NodeContext::num_outputs() const
{
    return _outputs.size();
}