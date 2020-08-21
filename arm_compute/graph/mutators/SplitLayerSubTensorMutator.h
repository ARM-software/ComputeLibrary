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
#ifndef ARM_COMPUTE_GRAPH_SPLIT_LAYER_SUBTENSOR_MUTATOR_H
#define ARM_COMPUTE_GRAPH_SPLIT_LAYER_SUBTENSOR_MUTATOR_H

#include "arm_compute/graph/IGraphMutator.h"

namespace arm_compute
{
namespace graph
{
/** Mutation pass to optimize split operations by using sub-tensors
 *
 * @warning This is compulsory to run in case Split layers are present in the model
 **/
class SplitLayerSubTensorMutator final : public IGraphMutator
{
public:
    // Inherited methods overridden
    virtual void mutate(Graph &g) override;
    MutationType type() const override;
    const char *name() override;
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_SPLIT_LAYER_SUBTENSOR_MUTATOR_H */
