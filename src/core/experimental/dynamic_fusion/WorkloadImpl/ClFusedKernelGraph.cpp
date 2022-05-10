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
#include "src/core/experimental/dynamic_fusion/WorkloadImpl/ClFusedKernelGraph.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
namespace
{
std::vector<std::pair<ClKernelFusionGroup *, ClKernelFusionGroup *>> get_combinations(const std::vector<ClKernelFusionGroup *> &sorted_fgs)
{
    ARM_COMPUTE_ERROR_ON(sorted_fgs.size() <= 1);
    std::vector<std::pair<ClKernelFusionGroup *, ClKernelFusionGroup *>> combo;
    for(size_t i = 0; i < sorted_fgs.size() - 1; ++i)
    {
        for(size_t j = i + 1; j < sorted_fgs.size(); ++j)
        {
            combo.push_back(std::make_pair(sorted_fgs.at(i), sorted_fgs.at(j)));
        }
    }
    return combo;
}
} // namespace
std::vector<const ClKernel *> traverse(const ClKernelFusionGroup &group)
{
    std::vector<const ClKernel *> kernels;
    const auto                    sorted = group.graph.topological_sort();
    for(const auto &pack : sorted.second)
    {
        kernels.push_back(group.fused_kernels.at(pack.op));
    }
    return kernels;
}

std::vector<const ClKernelFusionGroup *> traverse(const ClFusedKernelGraph &graph)
{
    std::vector<const ClKernelFusionGroup *> kernels;
    const auto                               sorted = graph.fg_dependency.topological_sort();
    for(const auto &pack : sorted.second)
    {
        kernels.push_back(graph.fusion_groups.at(pack.op).get());
    }
    return kernels;
}

std::vector<ClKernelFusionGroup *> traverse(ClFusedKernelGraph &graph)
{
    std::vector<ClKernelFusionGroup *> kernels;
    const auto                         sorted = graph.fg_dependency.topological_sort();
    for(const auto &pack : sorted.second)
    {
        kernels.push_back(graph.fusion_groups.at(pack.op).get());
    }
    return kernels;
}

std::pair<Status, ClFusedKernelGraph> init_fusion_graph(const ClKernelGraph &kernel_graph)
{
    ClFusedKernelGraph fused_kernel_graph{};
    fused_kernel_graph.original_graph = &kernel_graph; // Create a copy of the original kernel graph
    fused_kernel_graph.fg_dependency  = DependencyGraph();
    // Initialize all fusion groups
    for(const auto &kernel : traverse(kernel_graph))
    {
        fused_kernel_graph.add_fusion_group({ kernel });
    }
    return { Status{}, fused_kernel_graph };
}

Status fuse(ClFusedKernelGraph &fused_kernel_graph)
{
    // A naive fusion algorithm that's guaranteed to find optimal pattern if there are no branches
    // If there are branches, the algorithm cannot guanrantee optimality as it doesn't perform any searches

    bool fusion_found = false;
    do
    {
        fusion_found          = false;
        const auto sorted_fgs = traverse(fused_kernel_graph);
        if(sorted_fgs.size() <= 1)
        {
            // Only one or zero fusion group, thus no need to perform fusion
            return Status{};
        }
        auto fgs_combo = get_combinations(sorted_fgs);
        for(auto fgs : fgs_combo)
        {
            auto       fg0 = fgs.first;
            auto       fg1 = fgs.second;
            const auto st  = fused_kernel_graph.can_fuse(*fg0, *fg1);
            if(bool(st))
            {
                const auto st = fused_kernel_graph.fuse(*fg0, *fg1);
                if(!bool(st))
                {
                    return st;
                }
                fusion_found = true;
                break;
            }
        }
    }
    while(fusion_found);
    return Status{};
}
Status generate_store(ClKernelBlueprint &bp, const ClFusedKernelGraph &fused_kernel_graph, const ClKernelFusionGroup &fg)
{
    Status st{};
    for(const auto &dst_t_id : fused_kernel_graph.fg_dependency.dst_tensors(fg.id))
    {
        const auto dst_t = fused_kernel_graph.original_graph->get_tensor(dst_t_id);

        /// NOTE: dst tensor must have already been added to the blueprint at this point
        ArgumentID dst_id;
        st = add_tensor(bp, dst_t->desc, dst_id, dst_t->id);
        if(!bool(st))
        {
            return st;
        }
        /// NOTE: the extra dst tensor is needed as the store kcomp requires 2 tensors. But this is irrelevant to the fused kernel graph
        /// since both tensors share the exact same info and kernel arg descriptor
        ArgumentID dst_dst_id;
        st = add_tensor(bp, dst_t->desc, dst_dst_id);
        if(!bool(st))
        {
            return st;
        }
        /// NOTE: Update the merge point map to link dst_dst_id with dst_t->id instead.
        /// This is required because the get_arguments() returned by the blueprint returns the dst tensor added by the store component
        st = update_merge_point(bp, dst_dst_id, dst_t->id);
        if(!bool(st))
        {
            return st;
        }
        st = add_kcomp_store(bp, fg.get_root_kernel()->config().store_type, dst_id, dst_dst_id);
        if(!bool(st))
        {
            return st;
        }
    }
    return st;
}

Status generate(ClWorkload &workload, const ClWorkloadContext &ctx, const ClFusedKernelGraph &fused_kernel_graph)
{
    workload.context = ctx;
    for(const auto &fg : traverse(fused_kernel_graph))
    {
        ClKernelBlueprint bp{};
        for(const auto &kernel : traverse(*fg))
        {
            const auto st = kernel->generate(bp);
            if(!bool(st))
            {
                return st;
            }
        }
        auto st = set_tile_info(bp, fg->get_root_kernel()->config().tile_desc);
        if(!bool(st))
        {
            return st;
        }
        st = generate_store(bp, fused_kernel_graph, *fg);
        if(!bool(st))
        {
            return st;
        }

        ClKernelCode code{};
        st = build(code, ClCodeBuilderContext{ ctx.gpu_info }, bp);
        if(!bool(st))
        {
            return st;
        }
        const auto bp_graph = get_dependency_graph(bp);

        // Get tensor info
        std::vector<Id> workload_src_tensors{};
        for(const auto &src_t_id : fused_kernel_graph.fg_dependency.src_tensors(fg->id))
        {
            const auto src_t = fused_kernel_graph.original_graph->get_tensor(src_t_id);
            // Get corresponding kernel arg descriptor
            const auto arg_desc    = code.arguments.at(bp_graph.get_merge_points().at(src_t->id));
            const auto kernel_t_id = workload.add_workload_tensor(src_t->desc, src_t->memory_type, src_t->memory_info, arg_desc, src_t->id);
            workload_src_tensors.push_back(kernel_t_id);
        }
        std::vector<Id> workload_dst_tensors{};
        for(const auto &dst_t_id : fused_kernel_graph.fg_dependency.dst_tensors(fg->id))
        {
            const auto dst_t = fused_kernel_graph.original_graph->get_tensor(dst_t_id);
            // Get corresponding kernel arg descriptor
            const auto arg_desc    = code.arguments.at(bp_graph.get_merge_points().at(dst_t->id));
            const auto kernel_t_id = workload.add_workload_tensor(dst_t->desc, dst_t->memory_type, dst_t->memory_info, arg_desc, dst_t->id);
            workload_dst_tensors.push_back(kernel_t_id);
        }

        workload.add_unit_workload(fg->get_root_kernel()->config().stage, code, workload_src_tensors, workload_dst_tensors);
    }

    return Status{};
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */