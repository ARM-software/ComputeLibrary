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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUWORKLOADSKETCHIMPL
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUWORKLOADSKETCHIMPL

#include "arm_compute/dynamic_fusion/sketch/MemoryDescriptor.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h"
#include "src/dynamic_fusion/sketch/gpu/GpuComponentServices.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGraph.h"
#include "src/dynamic_fusion/sketch/gpu/GpuOperatorGroup.h"

#include <memory>
#include <vector>

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
          _operator_group{},
          _managed_tensor_info_list{ std::vector<std::unique_ptr<TensorInfo>>() },
          _mem_map{}
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
        return component_graph().fuse(_mem_map).write_workload_code();
    }
    /** Create a virtual (see @ref MemoryType) tensor info and save it
     *
     * @return ITensorInfo*  The created virtual tensor info object pointer
     */
    ITensorInfo *create_virtual_tensor()
    {
        auto uptr = std::make_unique<TensorInfo>();
        uptr->set_id(-allocate_new_tensor_id()); // virtual tensors must have negative id
        register_memory_descriptor(*uptr, MemoryDescriptor{ MemoryType::Virtual });
        _managed_tensor_info_list.emplace_back(std::move(uptr));
        return _managed_tensor_info_list.back().get();
    }
    /** Create an auxiliary (see @ref MemoryType) tensor info and save it
     *
     * @return ITensorInfo*  The created auxiliary tensor info object pointer
     */

    /** Create an auxiliary (see @ref MemoryType) tensor info and save it
     *
     * @param[in] tensor_info @ref ITensorInfo to copy from
     *
     * @return ITensorInfo*  The created auxiliary tensor info object pointer
     */
    ITensorInfo *create_auxiliary_tensor(const ITensorInfo &tensor_info)
    {
        auto uptr = std::make_unique<TensorInfo>(tensor_info);
        uptr->set_id(allocate_new_tensor_id());
        register_memory_descriptor(*uptr, MemoryDescriptor{ MemoryType::Auxiliary, AuxMemoryInfo{ uptr->total_size() } });
        _managed_tensor_info_list.emplace_back(std::move(uptr));
        return _managed_tensor_info_list.back().get();
    }
    /** Register memory descriptor of a tensor info
     *
     * @param[in] info     @ref ITensorInfo to be registered
     * @param[in] mem_desc @ref MemoryDescriptor to be registered with @p info
     */
    void register_memory_descriptor(const ITensorInfo &info, const MemoryDescriptor &mem_desc)
    {
        _mem_map[info.id()] = mem_desc;
    }

private:
    Context                                 *_context;
    GpuComponentServices                     _comp_services;
    GpuKernelComponentGraph                  _component_graph;
    GpuOperatorGroup                         _operator_group;
    ITensorInfo::Id                          _next_id{ ITensorInfo::invalid_tensor_id };
    std::vector<std::unique_ptr<TensorInfo>> _managed_tensor_info_list;
    MemoryDescriptorMap                      _mem_map;
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUWORKLOADSKETCHIMPL */
