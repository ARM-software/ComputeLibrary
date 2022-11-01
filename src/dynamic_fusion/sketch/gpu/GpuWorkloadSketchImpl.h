/*
 * Copyright (c) 2022 Arm Limited.
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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUWORKLOADSKETCHIMPL
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUWORKLOADSKETCHIMPL

#include "arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h"
#include "src/dynamic_fusion/sketch/gpu/GpuComponentServices.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGraph.h"
#include "src/dynamic_fusion/sketch/gpu/GpuOperatorGroup.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Internal implementation of @ref GpuWorkloadSketch */
class GpuWorkloadSketch::Implementation
{
public:
    /** Constructor
     *
     * @param[in] context global workload creation context
     */
    explicit Implementation(
        Context *context)
        : _context{ context },
          _comp_services{},
          _component_graph{ &_comp_services },
          _operator_group{}
    {
    }
    /** Prevent instances of this class from being copy constructed */
    Implementation(const Implementation &impl) = delete;
    /** Prevent instances of this class from being copied */
    Implementation &operator=(const Implementation &impl) = delete;
    /** Allow instances of this class to be move constructed */
    Implementation(Implementation &&impl) = default;
    /** Allow instances of this class to be moved */
    Implementation &operator=(Implementation &&impl) = default;
    /** Get workload context */
    const Context *context() const
    {
        return _context;
    }
    /** Get component graph */
    const GpuKernelComponentGraph &component_graph() const
    {
        return _component_graph;
    }
    /** Get component graph */
    GpuKernelComponentGraph &component_graph()
    {
        return _component_graph;
    }
    /** Get operator group */
    const GpuOperatorGroup &operator_group() const
    {
        return _operator_group;
    }
    /** Get operator group */
    GpuOperatorGroup &operator_group()
    {
        return _operator_group;
    }
    ITensorInfo::Id allocate_new_tensor_id()
    {
        return ++_next_id;
    }
    /** Generate @ref GpuWorkloadSourceCode from the workload sketch
     * @note The sketch must be valid. Any error encountered during the building of the code will be thrown.
     *
     * @return GpuWorkloadSourceCode  The generated workload code
     */
    GpuWorkloadSourceCode generate_source_code() const
    {
        return component_graph().fuse().write_workload_code();
    }

private:
    Context                *_context;
    GpuComponentServices    _comp_services;
    GpuKernelComponentGraph _component_graph;
    GpuOperatorGroup        _operator_group;
    ITensorInfo::Id         _next_id{ ITensorInfo::invalid_tensor_id };
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUWORKLOADSKETCHIMPL */
