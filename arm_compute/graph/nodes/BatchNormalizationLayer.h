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
#ifndef __ARM_COMPUTE_GRAPH_BATCHNORMALIZATION_LAYER_H__
#define __ARM_COMPUTE_GRAPH_BATCHNORMALIZATION_LAYER_H__

#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/INode.h"
#include "arm_compute/graph/ITensorObject.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/graph/Types.h"

namespace arm_compute
{
namespace graph
{
/** BatchNormalization layer node */
class BatchNormalizationLayer final : public INode
{
public:
    /** Default constructor
     *
     * @param[in] mean    Mean values tensor
     * @param[in] var     Var values tensor
     * @param[in] gamma   Gamma values tensor
     * @param[in] beta    Beta values tensor
     * @param[in] epsilon Epsilon value
     */
    template <typename AccessorType>
    BatchNormalizationLayer(AccessorType &&mean, AccessorType &&var, AccessorType &&gamma, AccessorType &&beta, float epsilon)
        : _mean(std::move(mean)), _var(std::move(var)), _gamma(std::move(gamma)), _beta(std::move(beta)), _epsilon(epsilon)
    {
    }

    // Inherited methods overriden:
    std::unique_ptr<arm_compute::IFunction> instantiate_node(GraphContext &ctx, ITensorObject *input, ITensorObject *output) override;

private:
    Tensor _mean;
    Tensor _var;
    Tensor _gamma;
    Tensor _beta;
    float  _epsilon;
};
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_BATCHNORMALIZATION_LAYER_H__ */
