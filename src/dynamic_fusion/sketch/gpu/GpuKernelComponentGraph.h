/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUKERNELCOMPONENTGRAPH
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUKERNELCOMPONENTGRAPH

#include "src/dynamic_fusion/sketch/ArgumentPack.h"
#include "src/dynamic_fusion/sketch/gpu/GpuComponentServices.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentStream.h"
#include "src/dynamic_fusion/sketch/utils/DependencyGraph.h"

#include <vector>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
class IGpuKernelComponent;

/** A multi-input (tensors), multi-output (tensors) acyclic directed graph of gpu kernel components
 * Its main purposes are:
 *  - Perform "graph-level" optimizations like fusion of kernel components (not the fusion of operators)
 *  - Automatically assign memory descriptions @ref MemoryDescriptor of all tensors based on graph topology
 */
class GpuKernelComponentGraph
{
public:
    /** Constructor
     *
     * @param[in] services @ref GpuComponentServices to be used by the graph
     */
    GpuKernelComponentGraph(GpuComponentServices *services);
    /** Prevent instances of this class from being copy constructed */
    GpuKernelComponentGraph(const GpuKernelComponentGraph &graph) = delete;
    /** Prevent instances of this class from being copied */
    GpuKernelComponentGraph &operator=(const GpuKernelComponentGraph &graph) = delete;
    /** Allow instances of this class to be move constructed */
    GpuKernelComponentGraph(GpuKernelComponentGraph &&graph) = default;
    /** Allow instances of this class to be moved */
    GpuKernelComponentGraph &operator=(GpuKernelComponentGraph &&graph) = default;
    /** Create a new component and add it to the component graph
     * Component id is automatically allocated
     *
     * @tparam T    Component type
     * @tparam Args Component argument types
     *
     * @param[in] args Component arguments except for component id, which is auto-allocated
     */
    template <typename T, typename... Args>
    void add_new_component(Args &&... args)
    {
        auto                      comp           = _services->component_factory().create<T>(std::forward<Args>(args)...);
        ArgumentPack<ITensorInfo> tensors        = comp->tensors();
        const auto                src_tensor_ids = get_tensor_ids(tensors.get_const_src_tensors());
        const auto                dst_tensor_ids = get_tensor_ids(tensors.get_const_dst_tensors());
        bool                      success        = _dependency_graph.add_operator(comp->id(), src_tensor_ids, dst_tensor_ids);
        ARM_COMPUTE_UNUSED(success);
        ARM_COMPUTE_ERROR_ON(!success);
        _components[comp->id()] = std::move(comp);
        for(auto t : tensors.get_const_src_tensors())
        {
            _tensors[t->id()] = t;
        }
        for(auto t : tensors.get_const_dst_tensors())
        {
            _tensors[t->id()] = t;
        }
    }
    /** Perform component fusion and serialize the graph into a stream of component groups
     *
     * @param[in] mem_map MemoryDescriptorMap for all the tensors in the component graph
     *
     * @return GpuKernelComponentStream
     */
    GpuKernelComponentStream fuse(const MemoryDescriptorMap &mem_map) const;

private:
    static std::vector<DependencyGraph::TensorId> get_tensor_ids(const std::vector<const ITensorInfo *> tensors);
    GpuComponentServices *_services;
    std::map<ComponentId, std::unique_ptr<IGpuKernelComponent>> _components;
    std::map<ITensorInfo::Id, const ITensorInfo *>              _tensors;
    DependencyGraph _dependency_graph{};
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUKERNELCOMPONENTGRAPH */
