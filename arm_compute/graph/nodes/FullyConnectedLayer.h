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
#ifndef __ARM_COMPUTE_GRAPH_FULLY_CONNECTED_LAYER_H__
#define __ARM_COMPUTE_GRAPH_FULLY_CONNECTED_LAYER_H__

#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/INode.h"
#include "arm_compute/graph/ITensorObject.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/graph/Types.h"

namespace arm_compute
{
namespace graph
{
/** Fully connected layer node */
class FullyConnectedLayer final : public INode
{
public:
    /** Default constructor
     *
     * @param[in] num_neurons Number of neurons
     * @param[in] weights     Weights of the fully connected layer
     * @param[in] biases      Biases of the fully connected layer
     */
    template <typename AccessorTypeWeights, typename AccessorTypeBiases>
    FullyConnectedLayer(unsigned int num_neurons, AccessorTypeWeights &&weights, AccessorTypeBiases &&biases)
        : _num_neurons(num_neurons), _weights(std::move(weights)), _biases(std::move(biases))
    {
    }

    // Inherited methods overriden:
    std::unique_ptr<arm_compute::IFunction> instantiate_node(GraphContext &ctx, ITensorObject *input, ITensorObject *output) override;

    // Inherited methods overriden:
private:
    unsigned int _num_neurons; /**< Number of neurons */
    Tensor       _weights;     /**< Weights tensor */
    Tensor       _biases;      /**< Biases tensor */
};
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_FULLY_CONNECTED_LAYER_H__ */
