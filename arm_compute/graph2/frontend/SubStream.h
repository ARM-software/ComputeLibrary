/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_GRAPH2_SUB_STREAM_H__
#define __ARM_COMPUTE_GRAPH2_SUB_STREAM_H__

#include "arm_compute/graph2/frontend/IStream.h"
#include "arm_compute/graph2/frontend/IStreamOperators.h"
#include "arm_compute/graph2/frontend/Types.h"

#include <memory>
#include <vector>

namespace arm_compute
{
namespace graph2
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
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    SubStream(const SubStream &) = delete;
    /** Default move constructor */
    SubStream(SubStream &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    SubStream &operator=(const SubStream &) = delete;
    /** Default move assignment operator */
    SubStream &operator=(SubStream &&) = default;

    // Inherited overridden methods
    void add_layer(ILayer &layer) override;
    Graph       &graph() override;
    const Graph &graph() const override;

private:
    IStream &_s; /**< Parent stream (assume that the lifetime of the parent is longer) */
};
} // namespace frontend
} // namespace graph2
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH2_SUB_STREAM_H__ */
