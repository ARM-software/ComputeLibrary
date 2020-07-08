/*
 * Copyright (c) 2018-2019 Arm Limited.
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
#ifndef ARM_COMPUTE_GRAPH_NODE_EXECUTION_METHOD_MUTATOR_H
#define ARM_COMPUTE_GRAPH_NODE_EXECUTION_METHOD_MUTATOR_H

#include "arm_compute/graph/IGraphMutator.h"

namespace arm_compute
{
namespace graph
{
/** Mutation pass to fall-back to default execution method
 *
 * @note This operates on nodes that support multiple execution methods (e.g. ConvolutionLayerNode)
 *       and in case the requested execution method is not supported for a given configuration.
 *       Thus this is a fall-back mechanism to ensure graph execution.
 */
class NodeExecutionMethodMutator final : public IGraphMutator
{
public:
    // Inherited methods overridden
    virtual void mutate(Graph &g) override;
    MutationType type() const override;
    const char *name() override;
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_NODE_EXECUTION_METHOD_MUTATOR_H */
