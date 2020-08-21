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
#ifndef ARM_COMPUTE_GRAPH_SUB_STREAM_H
#define ARM_COMPUTE_GRAPH_SUB_STREAM_H

#include "arm_compute/graph/frontend/IStream.h"
#include "arm_compute/graph/frontend/IStreamOperators.h"
#include "arm_compute/graph/frontend/Types.h"

#include <memory>
#include <vector>

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

/** Sub stream class*/
class SubStream final : public IStream
{
public:
    /** Default Constructor
     *
     * @param[in] s Parent stream
     */
    SubStream(IStream &s);

    // Inherited overridden methods
    void add_layer(ILayer &layer) override;
    Graph       &graph() override;
    const Graph &graph() const override;

private:
    IStream &_s; /**< Parent stream (assume that the lifetime of the parent is longer) */
};
} // namespace frontend
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_SUB_STREAM_H */
