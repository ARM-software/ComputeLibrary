/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_GRAPH_NEDEVICEBACKEND_H
#define ARM_COMPUTE_GRAPH_NEDEVICEBACKEND_H

#include "arm_compute/graph/IDeviceBackend.h"

#include "arm_compute/runtime/Allocator.h"

namespace arm_compute
{
namespace graph
{
namespace backends
{
/** Neon device backend */
class NEDeviceBackend final : public IDeviceBackend
{
public:
    NEDeviceBackend();

    // Inherited overridden methods
    void initialize_backend() override;
    void setup_backend_context(GraphContext &ctx) override;
    void release_backend_context(GraphContext &ctx) override;
    bool                           is_backend_supported() override;
    IAllocator                    *backend_allocator() override;
    std::unique_ptr<ITensorHandle> create_tensor(const Tensor &tensor) override;
    std::unique_ptr<ITensorHandle> create_subtensor(ITensorHandle *parent, TensorShape shape, Coordinates coords, bool extend_parent) override;
    std::unique_ptr<arm_compute::IFunction> configure_node(INode &node, GraphContext &ctx) override;
    Status validate_node(INode &node) override;
    std::shared_ptr<arm_compute::IMemoryManager> create_memory_manager(MemoryManagerAffinity affinity) override;
    std::shared_ptr<arm_compute::IWeightsManager> create_weights_manager() override;

private:
    Allocator _allocator; /**< Neon backend allocator */
};
} // namespace backends
} // namespace graph
} // namespace arm_compute
#endif //ARM_COMPUTE_GRAPH_NEDEVICEBACKEND_H
