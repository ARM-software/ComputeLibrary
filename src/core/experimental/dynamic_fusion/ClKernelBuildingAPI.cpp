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
#if defined(ENABLE_EXPERIMENTAL_DYNAMIC_FUSION)

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

Status add_tensor_argument(ClKernelBlueprint &kernel_blueprint, const ClTensorDescriptor &tensor_desc, ArgumentID &id)
{
    id = kernel_blueprint.impl().add_kernel_argument(tensor_desc);
    return Status{};
}

Status add_tensor_intermed(ClKernelBlueprint &kernel_blueprint, ArgumentID &id)
{
    id = kernel_blueprint.impl().add_intermediate_tensor();
    return Status{};
}

Status add_kcomp_gemm_native(ClKernelBlueprint          &kernel_blueprint, const ClKernelComponentDescriptor &,
                             const GemmNativeDescriptor &gemm_native_desc,
                             ArgumentID lhs_id, ArgumentID rhs_id, ArgumentID bias_id, ArgumentID &dst_id)
{
    kernel_blueprint.impl().validate_arg_ids({ lhs_id, rhs_id, bias_id, dst_id });
    kernel_blueprint.impl().add_component(
        std::make_unique<ClGemmNativeKernelComponent>(
            &kernel_blueprint,
            gemm_native_desc,
            SharedVarLink{ lhs_id, SharedVarIO::Input, kernel_blueprint.impl().group(lhs_id) },
            SharedVarLink{ rhs_id, SharedVarIO::Input, kernel_blueprint.impl().group(rhs_id) },
            SharedVarLink{ dst_id, SharedVarIO::Output, kernel_blueprint.impl().group(dst_id) },
            SharedVarLink{ bias_id, SharedVarIO::Input, kernel_blueprint.impl().group(bias_id) }));

    return Status{};
}

Status add_kcomp_eltwise_add(ClKernelBlueprint &kernel_blueprint, const ClKernelComponentDescriptor &, const EltwiseAddDescriptor &,
                             ArgumentID src0_id, ArgumentID src1_id, ArgumentID &dst_id)
{
    kernel_blueprint.impl().add_component(
        std::make_unique<ClElementwiseAddKernelComponent>(
            &kernel_blueprint,
            SharedVarLink{ src0_id, SharedVarIO::Input, kernel_blueprint.impl().group(src0_id) },
            SharedVarLink{ src1_id, SharedVarIO::Input, kernel_blueprint.impl().group(src1_id) },
            SharedVarLink{ dst_id, SharedVarIO::Output, kernel_blueprint.impl().group(dst_id) }));

    return Status{};
}
Status add_kcomp_activation(ClKernelBlueprint &, const ClKernelComponentDescriptor &, const ActivationDescriptor &, ArgumentID, ArgumentID &)
{
    return Status{};
}

Status add_kcomp_direct_conv(ClKernelBlueprint                 &kernel_blueprint, const ClKernelComponentDescriptor &,
                             const DirectConvolutionDescriptor &direct_conv2d_desc,
                             ArgumentID src_id, ArgumentID weight_id, ArgumentID bias_id, ArgumentID &dst_id)
{
    kernel_blueprint.impl().add_component(
        std::make_unique<ClDirectConvolutionKernelComponent>(
            &kernel_blueprint,
            direct_conv2d_desc,
            SharedVarLink{ src_id, SharedVarIO::Input, kernel_blueprint.impl().group(src_id) },
            SharedVarLink{ weight_id, SharedVarIO::Input, kernel_blueprint.impl().group(weight_id) },
            SharedVarLink{ dst_id, SharedVarIO::Output, kernel_blueprint.impl().group(dst_id) },
            SharedVarLink{ bias_id, SharedVarIO::Input, kernel_blueprint.impl().group(bias_id) }));

    return Status{};
}

Status add_kcomp_store(ClKernelBlueprint &kernel_blueprint, const ClKernelComponentDescriptor &, ArgumentID src_tile, ArgumentID dst_tile, const StoreType &store_type)
{
    switch(store_type)
    {
        case StoreType::StoreBlockBoundaryAware:
            kernel_blueprint.impl().add_component(
                std::make_unique<ClStoreBlockBoundaryAwareKernelComponent>(
                    &kernel_blueprint,
                    SharedVarLink{ src_tile, SharedVarIO::Input, kernel_blueprint.impl().group(src_tile) },
                    SharedVarLink{ dst_tile, SharedVarIO::Output, kernel_blueprint.impl().group(dst_tile) }));
            break;
        case StoreType::TStoreIndirectWidthSelect:
            kernel_blueprint.impl().add_component(
                std::make_unique<ClStoreIndirectWidthSelectKernelComponent>(
                    &kernel_blueprint,
                    SharedVarLink{ src_tile, SharedVarIO::Input, kernel_blueprint.impl().group(src_tile) },
                    SharedVarLink{ dst_tile, SharedVarIO::Output, kernel_blueprint.impl().group(dst_tile) }));
            break;
        default:
            ARM_COMPUTE_ERROR("Store mode not yet supported.");
    }

    return Status{};
}

Status set_tile_info(ClKernelBlueprint &bp, const TileDescriptor &tile_info)
{
    bp.impl().set_tile_info(tile_info);
    return Status{};
}
Status build(ClKernelCode &code, const ClCodeBuilderContext &, ClKernelBlueprint &kernel_blueprint)
{
    code.name = kernel_blueprint.impl().build_kernel_name();
    code.code = kernel_blueprint.impl().build_code();

    code.config_id     = kernel_blueprint.impl().build_config_id();
    code.build_options = kernel_blueprint.impl().build_options();
    code.window        = kernel_blueprint.impl().get_execution_window();
    code.arguments     = kernel_blueprint.impl().get_arguments();

    return Status{};
}
Status tune_static(ClExecutionDescriptor &, const ClKernelCode &)
{
    return Status{};
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif // defined(ENABLE_EXPERIMENTAL_DYNAMIC_FUSION)