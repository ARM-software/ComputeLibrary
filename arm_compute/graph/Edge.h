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
#ifndef ARM_COMPUTE_GRAPH_EDGE_H
#define ARM_COMPUTE_GRAPH_EDGE_H

#include "arm_compute/graph/INode.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/graph/Types.h"

namespace arm_compute
{
namespace graph
{
// Forward declarations
class Graph;

/** Graph Edge */
class Edge final
{
public:
    /** Default Constructor
     *
     * @param[in] id           Edge id
     * @param[in] producer     Producer node id
     * @param[in] producer_idx Producer node output index
     * @param[in] consumer     Consumer node id
     * @param[in] consumer_idx Consumer node input index
     * @param[in] tensor       Tensor associated with the edge
     */
    Edge(EdgeID id, INode *producer, unsigned int producer_idx, INode *consumer, unsigned int consumer_idx, Tensor *tensor)
        : _id(id), _producer(producer), _consumer(consumer), _producer_idx(producer_idx), _consumer_idx(consumer_idx), _tensor(tensor)

    {
    }
    /** Returns edge id
     *
     * @return Edge id
     */
    EdgeID id() const
    {
        return _id;
    }
    /** Returns producer node id
     *
     * @return Producer node id
     */
    NodeID producer_id() const
    {
        return (_producer == nullptr) ? EmptyNodeID : _producer->id();
    }
    /** Returns sink node id
     *
     * @return Sink node id
     */
    NodeID consumer_id() const
    {
        return (_consumer == nullptr) ? EmptyNodeID : _consumer->id();
    }
    /** Returns producer node
     *
     * @return Producer node
     */
    INode *producer() const
    {
        return _producer;
    }
    /** Returns consumer node
     *
     * @return Consumer node
     */
    INode *consumer() const
    {
        return _consumer;
    }
    /** Returns the index of the output that produces the result in the producer node
     *
     * @return Producer node output index
     */
    unsigned int producer_idx() const
    {
        return _producer_idx;
    }
    /** Returns the index of the input that consumes the result in the consumer node
     *
     * @return Consumer node input index
     */
    unsigned int consumer_idx() const
    {
        return _consumer_idx;
    }
    /** Returns the tensor associated with this edge
     *
     * @return Tensor id
     */
    Tensor *tensor() const
    {
        return _tensor;
    }
    /** Returns the tensor id associated with this edge
     *
     * @return Tensor id
     */
    TensorID tensor_id() const
    {
        return (_tensor == nullptr) ? NullTensorID : _tensor->id();
    }
    /** Bind the edge to another tensor
     *
     * @note If tensor is nullptr then nothing happens
     *
     * @param[in] tensor Tensor to bind the edge to
     */
    void update_bound_tensor(Tensor *tensor)
    {
        _tensor = (tensor != nullptr) ? tensor : _tensor;
    }

private:
    friend class Graph;

private:
    EdgeID       _id;
    INode       *_producer;
    INode       *_consumer;
    unsigned int _producer_idx;
    unsigned int _consumer_idx;
    Tensor      *_tensor;
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_EDGE_H */
