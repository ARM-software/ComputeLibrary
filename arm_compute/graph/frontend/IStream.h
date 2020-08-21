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
#ifndef ARM_COMPUTE_GRAPH_ISTREAM_H
#define ARM_COMPUTE_GRAPH_ISTREAM_H

#include "arm_compute/graph/frontend/Types.h"

namespace arm_compute
{
namespace graph
{
// Forward declarations
class Graph;

namespace frontend
{
// Forward declarations
class ILayer;

/** Stream interface **/
class IStream
{
public:
    virtual ~IStream() = default;
    /** Adds a layer to the stream
     *
     * @param[in] layer Layer to add
     */
    virtual void add_layer(ILayer &layer) = 0;
    /** Returns the underlying graph
     *
     * @return Underlying graph
     */
    virtual Graph &graph() = 0;
    /** Returns the underlying graph
     *
     * @return Underlying graph
     */
    virtual const Graph &graph() const = 0;
    /** Returns the tail node of the Stream
     *
     * @return Tail Node ID
     */
    NodeID tail_node()
    {
        return _tail_node;
    }
    /** Returns the stream hints that are currently used
     *
     * @return Stream hints
     */
    StreamHints &hints()
    {
        return _hints;
    }
    /** Forwards tail of stream to a given nid
     *
     * @param[in] nid NodeID of the updated tail node
     */
    void forward_tail(NodeID nid)
    {
        _tail_node = (nid != NullTensorID) ? nid : _tail_node;
    }

protected:
    StreamHints _hints     = {};              /**< Execution and algorithmic hints */
    NodeID      _tail_node = { EmptyNodeID }; /**< NodeID pointing to the last(tail) node of the graph */
};
} // namespace frontend
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_ISTREAM_H */
