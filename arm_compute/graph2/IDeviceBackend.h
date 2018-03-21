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
#ifndef __ARM_COMPUTE_GRAPH2_IDEVICEBACKEND_H__
#define __ARM_COMPUTE_GRAPH2_IDEVICEBACKEND_H__

#include "arm_compute/graph2/ITensorHandle.h"
#include "arm_compute/graph2/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"

#include <memory>

namespace arm_compute
{
namespace graph2
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
     * @param[in] ctx Graph context
     */
    virtual void setup_backend_context(GraphContext &ctx) = 0;
    /** Create a backend Tensor
     *
     * @param[in] tensor The tensor we want to create a backend tensor for
     *
     * @return Backend tensor handle
     */
    virtual std::unique_ptr<ITensorHandle> create_tensor(const Tensor &tensor) = 0;
    /** Create a backend Sub-Tensor
     *
     * @param[in] parent Parent sub-tensor handle
     * @param[in] shape  Shape of the sub-tensor
     * @param[in] coords Starting coordinates of the sub-tensor
     *
     * @return Backend sub-tensor handle
     */
    virtual std::unique_ptr<ITensorHandle> create_subtensor(ITensorHandle *parent, TensorShape shape, Coordinates coords) = 0;
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
};
} // namespace backends
} // namespace graph2
} // namespace arm_compute
#endif //__ARM_COMPUTE_GRAPH2_IDEVICEBACKEND_H__
