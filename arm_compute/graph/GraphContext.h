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
#ifndef ARM_COMPUTE_GRAPH_GRAPH_CONTEXT_H
#define ARM_COMPUTE_GRAPH_GRAPH_CONTEXT_H

#include "arm_compute/graph/Types.h"

#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/IWeightsManager.h"

#include <map>
#include <memory>

namespace arm_compute
{
namespace graph
{
/** Contains structs required for memory management */
struct MemoryManagerContext
{
    Target                                       target      = { Target::UNSPECIFIED }; /**< Target */
    std::shared_ptr<arm_compute::IMemoryManager> intra_mm    = { nullptr };             /**< Intra-function memory manager */
    std::shared_ptr<arm_compute::IMemoryManager> cross_mm    = { nullptr };             /**< Cross-function memory manager */
    std::shared_ptr<arm_compute::IMemoryGroup>   cross_group = { nullptr };             /**< Cross-function memory group */
    IAllocator                                  *allocator   = { nullptr };             /**< Backend allocator to use */
};

/** Contains structs required for weights management */
struct WeightsManagerContext
{
    Target                                        target = { Target::UNSPECIFIED }; /**< Target */
    std::shared_ptr<arm_compute::IWeightsManager> wm     = { nullptr };             /**< Weights manager */
};

/** Graph context **/
class GraphContext final
{
public:
    /** Constructor */
    GraphContext();
    /** Destructor */
    ~GraphContext();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GraphContext(const GraphContext &) = delete;
    /** Default move constructor */
    GraphContext(GraphContext &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GraphContext &operator=(const GraphContext &) = delete;
    /** Default move assignment operator */
    GraphContext &operator=(GraphContext &&) = default;
    /** Graph configuration accessor
     *
     * @note Every alteration has to be done before graph finalization
     *
     * @return The graph configuration
     */
    const GraphConfig &config() const;
    /** Sets graph configuration
     *
     * @param[in] config Configuration to use
     */
    void set_config(const GraphConfig &config);
    /** Inserts a memory manager context
     *
     * @param[in] memory_ctx Memory manage context
     *
     * @return True if the insertion succeeded else false
     */
    bool insert_memory_management_ctx(MemoryManagerContext &&memory_ctx);
    /** Gets a memory manager context for a given target
     *
     * @param[in] target To retrieve the management context
     *
     * @return Management context for the target if exists else nullptr
     */
    MemoryManagerContext *memory_management_ctx(Target target);
    /** Gets the memory managers map
     *
     * @return Memory manager contexts
     */
    std::map<Target, MemoryManagerContext> &memory_managers();
    /** Inserts a weights manager context
     *
     * @param[in] weights_ctx Weights manager context
     *
     * @return True if the insertion succeeded else false
     */
    bool insert_weights_management_ctx(WeightsManagerContext &&weights_ctx);

    /** Gets a weights manager context for a given target
     *
     * @param[in] target To retrieve the weights management context
     *
     * @return Management context for the target if exists else nullptr
     */
    WeightsManagerContext *weights_management_ctx(Target target);

    /** Gets the weights managers map
     *
     * @return Weights manager contexts
     */
    std::map<Target, WeightsManagerContext> &weights_managers();
    /** Finalizes memory managers in graph context */
    void finalize();

private:
    GraphConfig _config;                                       /**< Graph configuration */
    std::map<Target, MemoryManagerContext>  _memory_managers;  /**< Memory managers for each target */
    std::map<Target, WeightsManagerContext> _weights_managers; /**< Weights managers for each target */
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_GRAPH_CONTEXT_H */
