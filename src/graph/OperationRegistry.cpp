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
#include "arm_compute/graph/OperationRegistry.h"

using namespace arm_compute::graph;

OperationRegistry::OperationRegistry()
    : _registered_ops()
{
}

OperationRegistry &OperationRegistry::get()
{
    static OperationRegistry instance;
    return instance;
}

IOperation *OperationRegistry::find_operation(OperationType operation, TargetHint target)
{
    ARM_COMPUTE_ERROR_ON(!contains(operation, target));
    auto it = std::find_if(_registered_ops[operation].begin(), _registered_ops[operation].end(), [&](const std::unique_ptr<IOperation> &op)
    {
        return (op->target() == target);
    });
    ARM_COMPUTE_ERROR_ON(it == _registered_ops[operation].end());
    return (*it).get();
}

bool OperationRegistry::contains(OperationType operation, TargetHint target) const
{
    auto it = _registered_ops.find(operation);
    if(it != _registered_ops.end())
    {
        return std::any_of(it->second.begin(), it->second.end(), [&](const std::unique_ptr<IOperation> &op)
        {
            return (op->target() == target);
        });
    }
    return false;
}
