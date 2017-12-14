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
#ifndef __ARM_COMPUTE_GRAPH_OPERATION_REGISTRY_H__
#define __ARM_COMPUTE_GRAPH_OPERATION_REGISTRY_H__

#include "arm_compute/graph/IOperation.h"
#include "arm_compute/graph/Types.h"
#include "support/ToolchainSupport.h"

#include <map>
#include <memory>
#include <string>

namespace arm_compute
{
namespace graph
{
/** Registry holding all the supported operations */
class OperationRegistry
{
public:
    /** Gets operation registry instance
     *
     * @return Operation registry instance
     */
    static OperationRegistry &get();
    /** Finds an operation in the registry
     *
     * @param[in] operation Type of the operation to find
     * @param[in] target    Target of the operation
     *
     * @return Pointer to the operation functor if found, else nullptr
     */
    IOperation *find_operation(OperationType operation, TargetHint target);
    /** Checks if an operation for a given target exists
     *
     * @param[in] operation Operation type
     * @param[in] target    Execution target
     *
     * @return True if exists else false
     */
    bool contains(OperationType operation, TargetHint target) const;
    /** Registers an operation to the registry
     *
     * @param operation Operation to register
     */
    template <typename T>
    void add_operation(OperationType operation);

private:
    /** Default Constructor */
    OperationRegistry();

private:
    std::map<OperationType, std::vector<std::unique_ptr<IOperation>>> _registered_ops;
};

template <typename T>
inline void OperationRegistry::add_operation(OperationType operation)
{
    _registered_ops[operation].emplace_back(support::cpp14::make_unique<T>());
}
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_OPERATION_REGISTRY_H__ */
