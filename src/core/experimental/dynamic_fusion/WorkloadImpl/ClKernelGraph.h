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
#ifdef ENABLE_EXPERIMENTAL_DYNAMIC_FUSION
#ifndef ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_CLKERNELGRAPH_H
#define ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_CLKERNELGRAPH_H

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/experimental/ClWorkload.h"
#include "arm_compute/core/experimental/DependencyGraph.h"
#include "src/core/experimental/dynamic_fusion/WorkloadImpl/ClKernelDescriptors.h"
#include "src/core/experimental/dynamic_fusion/WorkloadImpl/ITensorDescPack.h"
#include "support/DeepCopy.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
struct ClKernelGraph;
class ClKernelBlueprint;

enum class Complexity
{
    Simple,
    Complex
};

/** Configurations for ClKernel
 *
 */
struct ClKernelConfig
{
    UnitWorkloadStage stage{};
    TileDescriptor    tile_desc{};
    StoreType         store_type{};
    friend bool operator==(const ClKernelConfig &config0, const ClKernelConfig &config1)
    {
        return config0.stage == config1.stage && config0.tile_desc == config1.tile_desc && config0.store_type == config1.store_type;
    }
};

struct ClKernelTensor
{
public:
    using Id         = DependencyGraph::Id;
    ClKernelTensor() = default;
    ClKernelTensor(Id id, ITensorInfo *desc, MemoryType memory_type, const AuxMemoryInfo &memory_info)
        : id{ id }, desc{ desc }, memory_type{ memory_type }, memory_info{ memory_info }
    {
    }
    bool operator==(const ClKernelTensor &other) const
    {
        return desc == other.desc;
    }

    Id            id{};
    ITensorInfo *desc{};
    MemoryType    memory_type{};
    AuxMemoryInfo memory_info{};
};

struct ClKernel
{
public:
    using Id                         = DependencyGraph::Id;
    ClKernel()                       = default;
    virtual ~ClKernel()              = default;
    ClKernel(const ClKernel &kernel) = default;
    ClKernel &operator=(const ClKernel &kernel) = default;
    ClKernel(ClKernel &&kernel)                 = default;
    ClKernel &operator=(ClKernel &&kernel) = default;
    ClKernel(const ClKernelGraph *graph, Id id, const ClKernelConfig &config, const ITensorDescPack<ClKernelTensor> &tensors)
        : _graph{ graph }, _id{ id }, _config{ config }, _tensors{ tensors }
    {
    }
    virtual bool operator==(const ClKernel &other) const = 0;
    virtual Complexity complexity() const                = 0;
    virtual Status generate(ClKernelBlueprint &bp) const = 0;
    Id id() const
    {
        return _id;
    }
    ITensorDescPack<ClKernelTensor> tensors() const
    {
        return _tensors;
    }
    ClKernelConfig config() const
    {
        return _config;
    }

protected:
    const ClKernelGraph            *_graph {};
    Id                              _id{};
    ClKernelConfig                  _config{};
    ITensorDescPack<ClKernelTensor> _tensors{};
};

struct ClDirectConv2dKernel : public ClKernel
{
public:
    Complexity complexity() const override
    {
        return Complexity::Complex;
    }
    ClDirectConv2dKernel()           = default;
    ~ClDirectConv2dKernel() override = default;
    ClDirectConv2dKernel(const ClKernelGraph *graph, Id id, const ClKernelConfig config, const ClDirectConv2dKernelDescriptor &desc, const ITensorDescPack<ClKernelTensor> tensors)
        : ClKernel{ graph, id, config, tensors }, desc{ desc }
    {
    }
    static Status validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const ClDirectConv2dKernelDescriptor &conv2d_desc);
    bool operator==(const ClKernel &other) const override;
    Status generate(ClKernelBlueprint &bp) const override;

    ClDirectConv2dKernelDescriptor desc{};
};

struct ClAddKernel : public ClKernel
{
public:
    Complexity complexity() const override
    {
        return Complexity::Simple;
    }
    ClAddKernel()           = default;
    ~ClAddKernel() override = default;
    ClAddKernel(const ClKernelGraph *graph, Id id, const ClKernelConfig &config, const ClEltwiseAddKernelDescriptor &desc, const ITensorDescPack<ClKernelTensor> tensors)
        : ClKernel{ graph, id, config, tensors }, desc{ desc }
    {
    }
    static Status validate(const ITensorInfo *lhs, const ITensorInfo *rhs, const ITensorInfo *dst);
    bool operator==(const ClKernel &other) const override;
    Status generate(ClKernelBlueprint &bp) const override;

    ClEltwiseAddKernelDescriptor desc{};
};

struct ClKernelGraph
{
public:
    using Id              = DependencyGraph::Id;
    using KernelMap       = std::map<Id, utils::memory::deep_unique_ptr<ClKernel>>;
    using KernelTensorMap = std::map<Id, utils::memory::deep_unique_ptr<ClKernelTensor>>;

    ClKernelGraph()  = default;
    ~ClKernelGraph() = default;

    friend bool operator==(const ClKernelGraph &graph0, const ClKernelGraph &graph1)
    {
        return graph0.graph == graph1.graph && graph0.kernels == graph1.kernels && graph0.tensors == graph1.tensors;
    }

    Status add_kernel_tensor(ITensorInfo *desc, MemoryType memory_type, const AuxMemoryInfo &memory_info, Id &tensor_id, Id merge_point = DependencyGraph::empty_id())
    {
        tensor_id = graph.add_tensor(merge_point);
        if(tensors.find(tensor_id) == tensors.end())
        {
            tensors[tensor_id] = utils::memory::make_deep_unique<ClKernelTensor, ClKernelTensor>(tensor_id, desc, memory_type, memory_info);
        }
        return Status{};
    }

    template <typename ContentT, typename KernelDescT>
    Status add_kernel(const ClKernelConfig &config, const KernelDescT &desc, const ITensorDescPack<ClKernelTensor> &tensors, Id &kernel_id)
    {
        const auto      src_tensors = tensors.get_const_src_tensors();
        const auto      dst_tensors = tensors.get_const_dst_tensors();
        std::vector<Id> src_tensor_ids{};
        std::vector<Id> dst_tensor_ids{};
        for(const auto &t : src_tensors)
        {
            src_tensor_ids.push_back(t->id);
        }
        for(const auto &t : dst_tensors)
        {
            dst_tensor_ids.push_back(t->id);
        }
        kernel_id          = graph.add_operator(src_tensor_ids, dst_tensor_ids).second;
        auto k             = utils::memory::make_deep_unique<ClKernel, ContentT>(this, kernel_id, config, desc, tensors);
        kernels[kernel_id] = std::move(k);
        return Status{};
    }

    ClKernel *get_kernel(Id id)
    {
        return kernels.at(id).get();
    }
    const ClKernel *get_kernel(Id id) const
    {
        return kernels.at(id).get();
    }

    ClKernelTensor *get_tensor(Id id)
    {
        return tensors.at(id).get();
    }
    const ClKernelTensor *get_tensor(Id id) const
    {
        return tensors.at(id).get();
    }

    DependencyGraph graph{};
    KernelMap       kernels{};
    KernelTensorMap tensors{};
};
using Id = DependencyGraph::Id;

std::vector<const ClKernel *> traverse(const ClKernelGraph &graph);
std::vector<ClKernel *> traverse(ClKernelGraph &graph);

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif //ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_CLKERNELGRAPH_H
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */