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

#include "src/core/experimental/dynamic_fusion/ClKernelBuildingAPI.h"
#include "src/core/experimental/dynamic_fusion/ClKernelBuildingImpl/Common.h"
#include "src/core/experimental/dynamic_fusion/ClKernelBuildingImpl/components/ClKernelComponents.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
ClKernelBlueprint::ClKernelBlueprint()
    : _impl{ std::make_unique<ClKernelBlueprint::Implementation>() }
{
}

ClKernelBlueprint::~ClKernelBlueprint() = default;

ClKernelBlueprint::Implementation &ClKernelBlueprint::impl()
{
    return *_impl;
}
const ClKernelBlueprint::Implementation &ClKernelBlueprint::impl() const
{
    return *_impl;
}

Status add_tensor(ClKernelBlueprint &kernel_blueprint, ITensorInfo *tensor_info, ArgumentID &id, ArgumentID merge_point)
{
    id = kernel_blueprint.impl().add_kernel_tensor(tensor_info, merge_point);
    return Status{};
}

Status add_kcomp_eltwise_add(ClKernelBlueprint &kernel_blueprint, const ClEltwiseAddKernelDescriptor &,
                             ArgumentID src0_id, ArgumentID src1_id, ArgumentID &dst_id)
{
    kernel_blueprint.impl().add_component(
        std::make_unique<ClElementwiseAddKernelComponent>(
            &kernel_blueprint,
            SharedVarLink{ src0_id, SharedVarIO::Input },
            SharedVarLink{ src1_id, SharedVarIO::Input },
            SharedVarLink{ dst_id, SharedVarIO::Output }));

    return Status{};
}
Status add_kcomp_activation(ClKernelBlueprint &, const ClActivationKernelDescriptor &, ArgumentID, ArgumentID &)
{
    return Status{};
}

Status add_kcomp_direct_conv2d(ClKernelBlueprint                    &kernel_blueprint,
                               const ClDirectConv2dKernelDescriptor &direct_conv2d_desc,
                               ArgumentID src_id, ArgumentID weight_id, ArgumentID bias_id, ArgumentID &dst_id)
{
    kernel_blueprint.impl().add_component(
        std::make_unique<ClDirectConvolutionKernelComponent>(
            &kernel_blueprint,
            direct_conv2d_desc,
            SharedVarLink{ src_id, SharedVarIO::Input },
            SharedVarLink{ weight_id, SharedVarIO::Input },
            SharedVarLink{ dst_id, SharedVarIO::Output },
            SharedVarLink{ bias_id, SharedVarIO::Input }));

    return Status{};
}

Status add_kcomp_store(ClKernelBlueprint &kernel_blueprint, const StoreType &store_type, ArgumentID src_tile, ArgumentID dst_tile)
{
    switch(store_type)
    {
        case StoreType::StoreBlockBoundaryAware:
            kernel_blueprint.impl().add_component(
                std::make_unique<ClStoreBlockBoundaryAwareKernelComponent>(
                    &kernel_blueprint,
                    SharedVarLink{ src_tile, SharedVarIO::Input },
                    SharedVarLink{ dst_tile, SharedVarIO::Output }));
            break;
        case StoreType::TStoreIndirectWidthSelect:
            kernel_blueprint.impl().add_component(
                std::make_unique<ClStoreIndirectWidthSelectKernelComponent>(
                    &kernel_blueprint,
                    SharedVarLink{ src_tile, SharedVarIO::Input },
                    SharedVarLink{ dst_tile, SharedVarIO::Output }));
            break;
        default:
            ARM_COMPUTE_ERROR("Store mode not yet supported.");
    }

    return Status{};
}

Status update_merge_point(ClKernelBlueprint &bp, ArgumentID t_id, ArgumentID merge_point)
{
    return bp.impl().update_merge_point(t_id, merge_point);
}

Status set_tile_info(ClKernelBlueprint &bp, const TileDescriptor &tile_info)
{
    bp.impl().set_tile_info(tile_info);
    return Status{};
}
Status build(ClKernelCode &code, const ClCodeBuilderContext &, ClKernelBlueprint &kernel_blueprint)
{
    kernel_blueprint.impl().finalize();
    code.name = kernel_blueprint.impl().build_kernel_name();
    code.code = kernel_blueprint.impl().build_code();

    code.config_id     = kernel_blueprint.impl().build_config_id();
    code.build_options = kernel_blueprint.impl().build_options();
    code.window        = kernel_blueprint.impl().get_execution_window();
    code.arguments     = kernel_blueprint.impl().get_arguments();

    return Status{};
}
DependencyGraph get_dependency_graph(const ClKernelBlueprint &blueprint)
{
    return blueprint.impl().get_graph();
}
Status tune_static(ClExecutionDescriptor &, const ClKernelCode &)
{
    return Status{};
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */