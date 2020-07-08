/*
 * Copyright (c) 2018 Arm Limited.
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
#include "arm_compute/graph/frontend/SubStream.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/frontend/ILayer.h"

namespace arm_compute
{
namespace graph
{
namespace frontend
{
SubStream::SubStream(IStream &s)
    : _s(s)
{
    _hints     = s.hints();
    _tail_node = s.tail_node();
}

void SubStream::add_layer(ILayer &layer)
{
    auto nid   = layer.create_layer(*this);
    _tail_node = nid;
}

const Graph &SubStream::graph() const
{
    return _s.graph();
}

Graph &SubStream::graph()
{
    return _s.graph();
}
} // namespace frontend
} // namespace graph
} // namespace arm_compute
