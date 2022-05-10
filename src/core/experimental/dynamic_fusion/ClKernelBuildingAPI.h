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

#ifndef ARM_COMPUTE_EXPERIMENTAL_CLKERNELBUILDINGAPI_H
#define ARM_COMPUTE_EXPERIMENTAL_CLKERNELBUILDINGAPI_H

#include "arm_compute/core/CL/CLCompileContext.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/experimental/ClWorkload.h"
#include "arm_compute/core/experimental/DependencyGraph.h"
#include "src/core/experimental/dynamic_fusion/WorkloadImpl/ClKernelDescriptors.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
using ArgumentID = DependencyGraph::Id;

static constexpr ArgumentID g_arg_placeholder = DependencyGraph::empty_id();

/** Intermediate representation of the final, complete kernel source. */
class ClKernelBlueprint
{
public:
    ClKernelBlueprint();
    ~ClKernelBlueprint();

private:
    struct Implementation;
    std::unique_ptr<Implementation> _impl;

public:
    Implementation       &impl();
    const Implementation &impl() const;
};

///// Kernel Components /////
/** Component: Eltwise Add */
Status add_kcomp_eltwise_add(ClKernelBlueprint &, const ClEltwiseAddKernelDescriptor &, ArgumentID src0_id,
                             ArgumentID src1_id, ArgumentID &dst_id);

/** Component: Activation */
Status add_kcomp_activation(ClKernelBlueprint &, const ClActivationKernelDescriptor &, ArgumentID src_id, ArgumentID &dst_id);

/** Component: Direct Convolution **/
Status add_kcomp_direct_conv2d(ClKernelBlueprint &, const ClDirectConv2dKernelDescriptor &,
                               ArgumentID src_id, ArgumentID weight_id, ArgumentID bias_id, ArgumentID &dst_id);

Status add_kcomp_store(ClKernelBlueprint &, const StoreType &store_type, ArgumentID src_id, ArgumentID dst_id);

Status add_tensor(ClKernelBlueprint &, ITensorInfo *, ArgumentID &, ArgumentID merge_point = DependencyGraph::empty_id());

///// Kernel Components /////

///// Building /////

/** Update existing merge tensor @p merge_point to point to @p t_id
 *
 * @param t_id
 * @param merge_point
 * @return Status
 */
Status update_merge_point(ClKernelBlueprint &, ArgumentID t_id, ArgumentID merge_point);

/** Get dependency graph
 *
 * @return DependencyGraph
 */
DependencyGraph get_dependency_graph(const ClKernelBlueprint &blueprint);

/** All information required for building the @ref ClKernelCode */
struct ClCodeBuilderContext
{
    GpuInfo gpu_info{};
};

Status set_tile_info(ClKernelBlueprint &, const TileDescriptor &);

/** Build final kernel source from KernelBlueprint */
Status build(ClKernelCode &code, const ClCodeBuilderContext &, ClKernelBlueprint &);

///// Building /////

///// Tuning /////

Status tune_static(ClExecutionDescriptor &, const ClKernelCode &);

///// Tuning /////

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif //ARM_COMPUTE_EXPERIMENTAL_CLKERNELBUILDINGAPI_H
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */