/*
 * Copyright (c) 2018-2019,2021 Arm Limited.
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
#ifndef ARM_COMPUTE_GRAPH_IDEVICEBACKEND_H
#define ARM_COMPUTE_GRAPH_IDEVICEBACKEND_H

#include "arm_compute/graph/ITensorHandle.h"
#include "arm_compute/graph/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/IWeightsManager.h"

#include <memory>

namespace arm_compute
{
namespace graph
{
// Forward declarations
class Graph;
class GraphContext;
class Tensor;
class INode;

namespace backends
{
/** Device backend interface */
class IDeviceBackend
{
public:
    /** Virtual Destructor */
    virtual ~IDeviceBackend() = default;
    /** Initializes the backend */
    virtual void initialize_backend() = 0;
    /** Setups the given graph context
     *
     * @param[in,out] ctx Graph context
     */
    virtual void setup_backend_context(GraphContext &ctx) = 0;
    /** Release the backend specific resources associated to a given graph context
     *
     * @param[in,out] ctx Graph context
     */
    virtual void release_backend_context(GraphContext &ctx) = 0;
    /** Checks if an instantiated backend is actually supported
     *
     * @return True if the backend is supported else false
     */
    virtual bool is_backend_supported() = 0;
    /** Gets a backend memory allocator
     *
     * @return Backend memory allocator
     */
    virtual IAllocator *backend_allocator() = 0;
    /** Create a backend Tensor
     *
     * @param[in] tensor The tensor we want to create a backend tensor for
     *
     * @return Backend tensor handle
     */
    virtual std::unique_ptr<ITensorHandle> create_tensor(const Tensor &tensor) = 0;
    /** Create a backend Sub-Tensor
     *
     * @param[in] parent        Parent sub-tensor handle
     * @param[in] shape         Shape of the sub-tensor
     * @param[in] coords        Starting coordinates of the sub-tensor
     * @param[in] extend_parent Extends parent shape if true
     *
     * @return Backend sub-tensor handle
     */
    virtual std::unique_ptr<ITensorHandle> create_subtensor(ITensorHandle *parent, TensorShape shape, Coordinates coords, bool extend_parent) = 0;
    /** Configure a backend Node
     *
     * @note This creates an appropriate configured backend function for the given node
     *
     * @param[in] node The node we want to configure
     * @param[in] ctx  Context to use
     *
     * @return Backend execution function
     */
    virtual std::unique_ptr<arm_compute::IFunction> configure_node(INode &node, GraphContext &ctx) = 0;
    /** Validate a node
     *
     * @param[in] node The node we want to validate
     *
     * @return An error status
     */
    virtual Status validate_node(INode &node) = 0;
    /** Create a backend memory manager given its affinity
     *
     * @param[in] affinity Memory Manager affinity
     *
     * @return Memory manager
     */
    virtual std::shared_ptr<arm_compute::IMemoryManager> create_memory_manager(MemoryManagerAffinity affinity) = 0;
    /** Create a backend weights manager
     *
     * @return Weights manager
     */
    virtual std::shared_ptr<arm_compute::IWeightsManager> create_weights_manager() = 0;
    /** Synchronize kernels execution on the backend. On GPU, this results in a blocking call waiting for all kernels to be completed. */
    virtual void sync() = 0;
};
} // namespace backends
} // namespace graph
} // namespace arm_compute
#endif //ARM_COMPUTE_GRAPH_IDEVICEBACKEND_H
