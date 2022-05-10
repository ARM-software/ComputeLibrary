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
#include "arm_compute/core/experimental/ClWorkload.h"
#include "src/core/experimental/dynamic_fusion/WorkloadImpl/ClFusedKernelGraph.h"
#include "src/core/experimental/dynamic_fusion/WorkloadImpl/ClKernelGraph.h"
#include "src/core/experimental/dynamic_fusion/WorkloadImpl/OperatorGraphImpl.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
Status build(ClWorkload &workload, const OperatorGraph &op_graph, const ClWorkloadContext &ctx)
{
    workload.context = ctx;
    ClKernelGraph kernel_graph;
    workload.status = validate(op_graph);
    ARM_COMPUTE_RETURN_ON_ERROR(workload.status);
    workload.status = translate(kernel_graph, *op_graph.impl());
    ARM_COMPUTE_RETURN_ON_ERROR(workload.status);
    ClFusedKernelGraph fused_k_graph;
    std::tie(workload.status, fused_k_graph) = init_fusion_graph(kernel_graph);
    ARM_COMPUTE_RETURN_ON_ERROR(workload.status);
    workload.status = fuse(fused_k_graph);
    ARM_COMPUTE_RETURN_ON_ERROR(workload.status);
    workload.status = generate(workload, ctx, fused_k_graph);
    ARM_COMPUTE_RETURN_ON_ERROR(workload.status);

    // Get operator tensor id to workload tensor id map
    const auto op_tensor_to_kernel_tensor       = fused_k_graph.original_graph->graph.get_merge_points();
    const auto kernel_tensor_to_workload_tensor = workload.graph.get_merge_points();
    for(const auto op_t : op_graph.impl()->graph.src_tensors())
    {
        const auto kernel_t                   = op_tensor_to_kernel_tensor.at(op_t);
        const auto workload_t                 = kernel_tensor_to_workload_tensor.at(kernel_t);
        workload.op_tensor_id_lut[workload_t] = op_t;
    }
    for(const auto op_t : op_graph.impl()->graph.dst_tensors())
    {
        const auto kernel_t                   = op_tensor_to_kernel_tensor.at(op_t);
        const auto workload_t                 = kernel_tensor_to_workload_tensor.at(kernel_t);
        workload.op_tensor_id_lut[workload_t] = op_t;
    }
    return workload.status;
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */