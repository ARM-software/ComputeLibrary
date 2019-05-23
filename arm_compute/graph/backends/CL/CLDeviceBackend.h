/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_GRAPH_CLDEVICEBACKEND_H__
#define __ARM_COMPUTE_GRAPH_CLDEVICEBACKEND_H__

#include "arm_compute/graph/IDeviceBackend.h"

#include "arm_compute/runtime/CL/CLBufferAllocator.h"
#include "arm_compute/runtime/CL/CLTuner.h"

namespace arm_compute
{
namespace graph
{
namespace backends
{
/** OpenCL device backend */
class CLDeviceBackend final : public IDeviceBackend
{
public:
    /** Default Constructor */
    CLDeviceBackend();
    /** Destructor */
    ~CLDeviceBackend();
    /** Switchs on or off the kernel tuning
     *
     * @note When true the tuner set is used, if no tuner is set a new default one is created
     *
     * @param[in] enable_tuning Enables tuning if false else true
     */
    void set_kernel_tuning(bool enable_tuning);
    /** Set kernel tuning mode
     *
     * @param[in] tuning_mode Indicates how exhaustive the search for the optimal LWS should be while tuning
     */
    void set_kernel_tuning_mode(CLTunerMode tuning_mode);

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

private:
    int                                _context_count; /**< Counts how many contexts are currently using the backend */
    CLTuner                            _tuner;         /**< CL kernel tuner */
    std::unique_ptr<CLBufferAllocator> _allocator;     /**< CL buffer affinity allocator */
    std::string                        _tuner_file;    /**< Filename to load/store the tuner's values from */
};
} // namespace backends
} // namespace graph
} // namespace arm_compute
#endif //__ARM_COMPUTE_GRAPH_CLDEVICEBACKEND_H__
