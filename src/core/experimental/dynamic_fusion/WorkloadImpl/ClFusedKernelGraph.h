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
#ifndef ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_CLFUSEDKERNELGRAPH_H
#define ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_CLFUSEDKERNELGRAPH_H
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/experimental/DependencyGraph.h"
#include "src/core/experimental/dynamic_fusion/ClKernelBuildingAPI.h"
#include "src/core/experimental/dynamic_fusion/WorkloadImpl/ClKernelGraph.h"
#include "support/DeepCopy.h"

#include <vector>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
struct ClKernelFusionGroup;

/** A const view of a subgraph of the @ref ClKernelGraph to be fused together
 *
 */
struct ClKernelFusionGroup
{
public:
    using Id = DependencyGraph::Id;

    ClKernelFusionGroup() = default;
    ClKernelFusionGroup(Id id)
        : id{ id }, graph{}, fused_kernels{}, tensors{}
    {
    }
    ~ClKernelFusionGroup() = default;

    void set_id(Id i)
    {
        id = i;
    }

    Id add_fused_kernel(const ClKernel *kernel)
    {
        /// PRE: Acyclicity ensured by DependencyGraph
        /// PRE: Connectedness ensured by DependencyGraph
        /// PRE: Single-rootedness ensured by User
        std::vector<Id> src_tensors;
        for(const auto t : kernel->tensors().get_const_src_tensors())
        {
            auto id = graph.add_tensor(t->id);
            if(tensors.find(id) == tensors.end())
            {
                tensors[id] = t;
            }
            src_tensors.push_back(id);
        }
        std::vector<Id> dst_tensors;
        for(const auto t : kernel->tensors().get_const_dst_tensors())
        {
            auto id = graph.add_tensor(t->id);
            if(tensors.find(id) == tensors.end())
            {
                tensors[id] = t;
            }
            dst_tensors.push_back(id);
        }
        auto id                  = graph.add_operator(src_tensors, dst_tensors);
        fused_kernels[id.second] = kernel;
        return id.second;
    }

    const ClKernel *get_root_kernel() const
    {
        auto root_kernels = graph.get_root_ops();
        ARM_COMPUTE_ERROR_ON(root_kernels.size() != 1);
        return fused_kernels.at(root_kernels.at(0));
    }

    std::vector<const ClKernelTensor *> get_src_tensors() const
    {
        std::vector<const ClKernelTensor *> src_tensors;
        for(auto tensor_id : graph.src_tensors())
        {
            src_tensors.push_back(tensors.at(tensor_id));
        }
        return src_tensors;
    }

    std::vector<const ClKernelTensor *> get_dst_tensors() const
    {
        std::vector<const ClKernelTensor *> dst_tensors;
        for(auto tensor_id : graph.dst_tensors())
        {
            dst_tensors.push_back(tensors.at(tensor_id));
        }
        return dst_tensors;
    }

    friend bool operator==(const ClKernelFusionGroup &fg0, const ClKernelFusionGroup &fg1)
    {
        return fg0.id == fg1.id && fg0.graph == fg1.graph && fg0.fused_kernels == fg1.fused_kernels && fg0.tensors == fg1.tensors;
    }

    Id              id{};
    DependencyGraph graph{}; // A subgraph of the original ClKernelGraph
    std::map<Id, const ClKernel *>       fused_kernels{};
    std::map<Id, const ClKernelTensor *> tensors{};
};

std::vector<const ClKernel *> traverse(const ClKernelFusionGroup &group);

struct ClFusedKernelGraph
{
public:
    using Id = DependencyGraph::Id;

    using KernelFusionGroupMap = std::map<Id, utils::memory::deep_unique_ptr<ClKernelFusionGroup>>;

    ClFusedKernelGraph()                                = default;
    ~ClFusedKernelGraph()                               = default;
    ClFusedKernelGraph(const ClFusedKernelGraph &graph) = default;
    ClFusedKernelGraph &operator=(const ClFusedKernelGraph &graph) = default;
    ClFusedKernelGraph(ClFusedKernelGraph &&graph)                 = default;
    ClFusedKernelGraph &operator=(ClFusedKernelGraph &&graph) = default;

    friend bool operator==(const ClFusedKernelGraph &graph0, const ClFusedKernelGraph &graph1)
    {
        /// NOTE: fg_dependency may change based on the order of fusion, and thus is omitted in the comparison.
        ///       The fusion groups can already guarantee the equivalence of fusion
        ///       In the future we may want to enforce a stronger equivalence by implementing topological comparison between @ref DependencyGraph s
        return graph0.original_graph == graph1.original_graph && graph0.fusion_groups == graph1.fusion_groups;
    }

    Id add_fusion_group(const std::vector<const ClKernel *> &fused_kernels)
    {
        auto fg = utils::memory::make_deep_unique<ClKernelFusionGroup, ClKernelFusionGroup>();
        for(const auto k : fused_kernels)
        {
            fg->add_fused_kernel(k);
        }
        const auto      src_tensors = fg->get_src_tensors();
        const auto      dst_tensors = fg->get_dst_tensors();
        std::vector<Id> inputs{};
        std::transform(std::begin(src_tensors), std::end(src_tensors), std::back_inserter(inputs), [this](auto kernel)
        {
            return fg_dependency.add_tensor(kernel->id);
        });
        std::vector<Id> outputs{};
        std::transform(std::begin(dst_tensors), std::end(dst_tensors), std::back_inserter(outputs), [this](auto kernel)
        {
            return fg_dependency.add_tensor(kernel->id);
        });
        const auto id = fg_dependency.add_operator(inputs, outputs);
        fg->set_id(id.second);
        fusion_groups[id.second] = std::move(fg);
        return id.second;
    }

    Status fuse(ClKernelFusionGroup &fg0, ClKernelFusionGroup &fg1)
    {
        /// PRE: Already checked by can_fuse, and thus all the INVs and ASSUMPTIONS still hold
        ClKernelFusionGroup *fg_src{};
        ClKernelFusionGroup *fg_dst{};
        // Find fg_src (parent / root) and fg_dst (child / non-root)
        if(is_in(fg1.id, fg_dependency.dst_ops(fg0.id)))
        {
            fg_src = &fg0;
            fg_dst = &fg1;
        }
        else if(is_in(fg0.id, fg_dependency.dst_ops(fg1.id)))
        {
            fg_src = &fg1;
            fg_dst = &fg0;
        }
        else
        {
            return Status{ ErrorCode::RUNTIME_ERROR, "Invalid fusion: Not directly connected fusion groups cannot be fused together" };
        }

        for(const auto &t : fg_dependency.src_tensors(fg_dst->id))
        {
            if(!is_in(t, fg_dependency.dst_tensors(fg_src->id)))
            {
                // Link any incoming tensors of fg_dst, that ARE NOT in between fg_src and fg_dst, to fg_src

                // Before:
                // fg_src
                // |
                // ..          t1
                // |           |
                // -> fg_dst <-
                //
                // After:
                // fg_src <---t1
                //
                const auto st = link_src_tensors(fg_src->id, { t });
                if(!bool(st))
                {
                    return st;
                }
            }
            else
            {
                const auto dst_fgs = fg_dependency.dst_ops_from_tensor(t);
                if(dst_fgs.size() == 1U && dst_fgs.at(0) == fg_dst->id)
                {
                    // Remove any incoming tensors of fg_dst, that ARE in between fg_src and fg_dst
                    // AND that are not connected to any other outgoing fgs (Note that they cannot connect to any other incoming fgs as all tensors can have at most 1 incoming fg (ASSUMPTION 3))

                    // Before:
                    // fg_src
                    // |
                    // t0
                    // |
                    // -> fg_dst
                    //
                    // After:
                    // fg_src
                    //
                    const auto st = remove_fg_tensor(t);
                    if(!bool(st))
                    {
                        return st;
                    }
                }
                else
                {
                    // If the tensors ARE in between fg_src and fg_dst
                    // BUT have any other outgoing fgs than fg_dst, then we leave it as a dst tensor to the fused fg_src

                    // Before:
                    // fg_src
                    // |
                    // t0
                    // |
                    // |-----------
                    // |          |
                    // -> fg_dst  -> fg_other
                    //
                    // After:
                    // fg_src
                    // |
                    // t0
                    // |
                    // -> fg_other
                    //

                    // Note that this may seem like a case we shouldn't fuse. But actually all it means is that t0 is an
                    // intermediate tensor between the fused fg_src and fg_dst, but only that we also STORE it to memory
                    // so that any unfused fg's (fg_other in this case) can read it.
                    // So all this means that we not only can STORE the tensors at the "end" of a fusion group,
                    // but also any other tensors that are not source tensors. And all tensors that are STORED (exported),
                    // can be termed "dst tensors" to a fusion group
                    void();
                }
            }
        }

        for(const auto &t : fg_dependency.dst_tensors(fg_dst->id))
        {
            // Link any outgoing tensors of fg_dst to fg_src

            // Before:
            // fg_src
            // |
            // ..
            // |
            // -> fg_dst
            //    |
            //    |--------
            //    |       |
            //    |-> t0  |-> t1
            //
            // After:
            // fg_src
            // |
            // |--------
            // |       |
            // |-> t0  |-> t1
            //
            const auto st = link_dst_tensors(fg_src->id, { t });
            if(!bool(st))
            {
                return st;
            }
        }

        // Merge fg_dst's graph into fg_src's graph
        for(const auto kernel : traverse(*fg_dst))
        {
            fg_src->add_fused_kernel(kernel);
        }

        const auto st = remove_fg(fg_dst->id);
        return st;
    }
    Status can_fuse(const ClKernelFusionGroup &fg0, const ClKernelFusionGroup &fg1) const
    {
        /// ASSUMPTION0: All tensors have 0 or 1 incoming kernel
        /// ASSUMPTION1: All kernels have exactly 1 dst tensor (Temporary, can be lifted once we start supporting multi-dst kernels)
        ///              Note that this does not apply to fusion groups
        /// ASSUMPTION2: Simple kernels' tile infos can be overriden (share with) that of the root kernel's
        /// ASSUMPTION3: Extension of ASSUMPTION0: All tensors have 0 or 1 incoming fusion group
        /// INV0: All Fusion groups have a single root
        /// INV1: All Fusion groups have no cycles or loops within themselves <- guaranteed by the underlying ClKernelGraph having no cycles or loops; enforced by DependencyGraph
        /// INV2: The ClKernelFusionGroup itself has no cycles or loops <- enforced by DependencyGraph
        /// INV3: All non-roots are Simple kernels
        /// INV4: All non roots' dst tensors have the same shape as that of the root kernel
        /// INV5: All kernels within a fusion group have the same UnitWorkloadStage
        const ClKernelFusionGroup *fg_src {};
        const ClKernelFusionGroup *fg_dst{};

        // Check 0: Ensure fg0 and fg1 are "directly connected": one of them is a direct parent of the other
        // This guarantess INV0
        // This also finds fg_src (parent / root) and fg_dst (child / non-root)
        if(is_in(fg1.id, fg_dependency.dst_ops(fg0.id)))
        {
            fg_src = &fg0;
            fg_dst = &fg1;
        }
        else if(is_in(fg0.id, fg_dependency.dst_ops(fg1.id)))
        {
            fg_src = &fg1;
            fg_dst = &fg0;
        }
        else
        {
            return Status{ ErrorCode::RUNTIME_ERROR, "Invalid fusion: Not directly connected fusion groups cannot be fused together" };
        }

        // Find unconnected tensors between fg_src and fg_dst
        std::vector<Id> unconnected_tensors{};
        for(const auto &t : fg_dependency.dst_tensors(fg_src->id))
        {
            if(!is_in(t, fg_dependency.src_tensors(fg_dst->id)))
            {
                unconnected_tensors.push_back(t);
            }
        }

        // Check 1: Any unconnected tensor cannot be an ancestor of fg_dst
        // This guarantees INV2: That is, the fused graph does not have any cycles or loops between different fusion groups
        for(const auto &t : unconnected_tensors)
        {
            if(fg_dependency.path_exists_from_tensor_to_op(t, fg_dst->id))
            {
                return Status{ ErrorCode::RUNTIME_ERROR, "Invalid fusion: the fusion would result in cycles or loops" };
            }
        }

        // Check 2: All non-root fgs are simple. Ensure INV3
        if(fg_dst->get_root_kernel()->complexity() != Complexity::Simple)
        {
            return Status{ ErrorCode::RUNTIME_ERROR, "Invalid fusion: only root kernel can be a complex kernel" };
        }

        // Check 3: All non roots' dst tensors have the same shape as that of the root kernel. Ensure INV4
        const auto root_kernel_dst_tensors = fg_dependency.dst_tensors(fg_src->id);
        ARM_COMPUTE_ERROR_ON(root_kernel_dst_tensors.size() != 1); // (ASSUMPTION 1: All kernels have exactly 1 dst tensor)
        const auto root_kernel_dst_tensor_info = original_graph->get_tensor(root_kernel_dst_tensors[0])->desc;

        for(const auto &t : fg_dependency.dst_tensors(fg_dst->id))
        {
            const auto t_info = original_graph->get_tensor(t)->desc;
            if(detail::have_different_dimensions(root_kernel_dst_tensor_info->tensor_shape(), t_info->tensor_shape(), 0))
            {
                return Status{ ErrorCode::RUNTIME_ERROR, "Invalid fusion: all non roots' dst tensors should have the same shape as that of the root kernel" };
            }
        }

        // Check 4: All kernels within a fg have the same UnitWorkloadStage. Ensure INV5
        if(!(fg_src->get_root_kernel()->config().stage == fg_dst->get_root_kernel()->config().stage))
        {
            return Status{ ErrorCode::RUNTIME_ERROR, "Invalid fusion: all kernels within a fusion group should have the same UnitWorkloadStage" };
        }

        return Status{};
    }

    const ClKernelGraph *original_graph{};
    DependencyGraph      fg_dependency{};
    KernelFusionGroupMap fusion_groups{};
    // Note: no need to store tensors pointers in the ClFusedKernelGraph, as they are stored in side the individual fusion groups.

private:
    Status link_src_tensors(Id fg, const std::vector<Id> &src_tensors)
    {
        for(auto t : src_tensors)
        {
            fg_dependency.link_input(fg, t);
        }
        return Status{};
    }
    Status link_dst_tensors(Id fg, const std::vector<Id> &dst_tensors)
    {
        for(auto t : dst_tensors)
        {
            fg_dependency.link_output(fg, t);
        }
        return Status{};
    }
    Status remove_fg(Id fg)
    {
        fg_dependency.remove_operator(fg);
        fusion_groups.erase(fg);
        return Status{};
    }
    Status remove_fg_tensor(Id tensor)
    {
        fg_dependency.remove_tensor(tensor);
        return Status{};
    }
};

std::vector<const ClKernelFusionGroup *> traverse(const ClFusedKernelGraph &graph);
std::vector<ClKernelFusionGroup *> traverse(ClFusedKernelGraph &graph);

std::pair<Status, ClFusedKernelGraph> init_fusion_graph(const ClKernelGraph &kernel_graph);

Status fuse(ClFusedKernelGraph &fused_kernel_graph);

Status generate_store(ClKernelBlueprint &bp, const ClFusedKernelGraph &fused_kernel_graph, const ClKernelFusionGroup &fg);

Status generate(ClWorkload &workload, const ClWorkloadContext &ctx, const ClFusedKernelGraph &fused_kernel_graph);

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif //ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_CLFUSEDKERNELGRAPH_H
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */