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
#ifndef __ARM_COMPUTE_GRAPH2_STREAM_H__
#define __ARM_COMPUTE_GRAPH2_STREAM_H__

#include "arm_compute/graph2/frontend/IStream.h"
#include "arm_compute/graph2/frontend/IStreamOperators.h"
#include "arm_compute/graph2/frontend/Types.h"

#include "arm_compute/graph2/Graph.h"
#include "arm_compute/graph2/GraphContext.h"
#include "arm_compute/graph2/GraphManager.h"

namespace arm_compute
{
namespace graph2
{
namespace frontend
{
// Forward Declarations
class ILayer;

/** Stream frontend class to construct simple graphs in a stream fashion */
class Stream final : public IStream
{
public:
    /** Constructor
     *
     * @param[in] id   Stream id
     * @param[in] name Stream name
     */
    Stream(size_t id, std::string name);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    Stream(const Stream &) = delete;
    /** Default move constructor */
    Stream(Stream &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    Stream &operator=(const Stream &) = delete;
    /** Default move assignment operator */
    Stream &operator=(Stream &&) = default;
    /** Finalizes the stream for an execution target
     *
     * @note enable_tuning only works if the target is OpenCL.
     * @note tuning increases the execution time of first run of the graph
     *
     * @param[in] target                   Execution target
     * @param[in] enable_tuning            (Optional) Enables the tuning interface. Defaults to false
     * @param[in] enable_memory_management (Optional) Enables the memory management interface. Defaults to false
     */
    void finalize(Target target, bool enable_tuning = false, bool enable_memory_management = false);
    /** Executes the stream **/
    void run();

    // Inherited overridden methods
    void add_layer(ILayer &layer) override;
    Graph       &graph() override;
    const Graph &graph() const override;

private:
    GraphManager _manager; /**< Graph manager */
    GraphContext _ctx;     /**< Graph context to use */
    Graph        _g;       /**< Internal graph representation of the stream */
};
} // namespace frontend
} // namespace graph2
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH2_STREAM_H__ */