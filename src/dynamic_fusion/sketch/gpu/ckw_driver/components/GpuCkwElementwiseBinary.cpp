/*
 * Copyright (c) 2023-2024 Arm Limited.
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
#include "GpuCkwElementwiseBinary.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/utils/StringUtils.h"
#include "arm_compute/core/Validate.h"

#include "src/core/helpers/WindowHelpers.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/CkwHelper.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/type_converter/Common.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/type_converter/ElementwiseBinary.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwScopedKernelWriter.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwVariableTable.h"
#include "src/dynamic_fusion/sketch/gpu/components/utils/type_printer/ElementwiseBinary.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelArgument.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGroup.h"
#include "support/StringSupport.h"

#include "compute_kernel_writer/include/ckw/KernelWriter.h"
#include "compute_kernel_writer/include/ckw/types/ConstantData.h"
#include "compute_kernel_writer/include/ckw/types/TensorSamplerTypes.h"
#include <cstdint>
#include <string>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
GpuCkwElementwiseBinary::GpuCkwElementwiseBinary(ComponentId                      id,
                                                 const ArgumentPack<ITensorInfo> &tensors,
                                                 const Attributes                &attributes)
    : IGpuCkwComponentDriver{id, tensors}, _lhs{}, _rhs{}, _dst{}, _attributes{attributes}
{
    _lhs = this->tensors().get_const_tensor(TensorType::ACL_SRC_0);
    _rhs = this->tensors().get_const_tensor(TensorType::ACL_SRC_1);
    _dst = this->tensors().get_const_tensor(TensorType::ACL_DST_0);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_lhs, _rhs, _dst);
}

void GpuCkwElementwiseBinary::write_component_code(const ComponentGroup    &comp_group,
                                                   GpuCkwVariableTable     &vtable,
                                                   GpuCkwScopedKernelWriter writer) const
{
    /********************************************************************************
     * 1 - Define tensors
     ********************************************************************************/
    GpuCkwComponentArgument *lhs = vtable.declare_variable(comp_group, writer, _lhs, "lhs");
    GpuCkwComponentArgument *rhs = vtable.declare_variable(comp_group, writer, _rhs, "rhs");
    GpuCkwComponentArgument *dst = vtable.declare_variable(comp_group, writer, _dst, "dst");

    /********************************************************************************
     * 2 - Define CKW constants
     ********************************************************************************/
    const auto dst_h = static_cast<int32_t>(_dst->dimension(1));

    // CKW constants
    auto const_dst_h_i32 = writer->declare_constant_tile(ckw::ConstantData({{dst_h}}, ckw::DataType::Int32));
    auto const_pos_1_i32 = writer->declare_constant_tile(ckw::ConstantData({{1}}, ckw::DataType::Int32));
    auto const_0_i32     = writer->declare_constant_tile(ckw::ConstantData({{0}}, ckw::DataType::Int32));

    /********************************************************************************
     * 3 - Define the compute block parameters and destination tile (if not root component)
     *     Bind the tile to the tensor to share it among different components and
     *     initialize the compute block parameters
     ********************************************************************************/
    // The compute block parameters depend on the employed tensor format

    // Destination compute block size
    int32_t dst_n0 = -1;
    int32_t dst_m0 = -1;

    // Destination compute block size left-over
    int32_t dst_n0_partial = -1;
    int32_t dst_m0_partial = -1;

    if (!dst->has_tile())
    {
        // If ROOT component, we use ckw::TensorSamplerFormat::Dim0_Dim1xDim2_1
        // as tensor format
        const auto root_window = comp_group.get_root_component()->ckw_component_driver()->get_window();

        dst_n0         = root_window.x().step();
        dst_m0         = root_window.y().step();
        dst_n0_partial = _dst->dimension(0) % dst_n0;
        dst_m0_partial = (_dst->dimension(1) * _dst->dimension(2)) % dst_m0;

        ckw::TensorSampler sampler_dst;
        sampler_dst.format(ckw::TensorSamplerFormat::Dim0_Dim1xDim2_1);
        if (dst_n0_partial == 0)
        {
            sampler_dst.address_mode_x(ckw::TensorSamplerAddressModeX::None);
        }
        else
        {
            sampler_dst.address_mode_x(ckw::TensorSamplerAddressModeX::OverlappingMin);
        }

        if (dst_m0_partial == 0)
        {
            sampler_dst.address_mode_y(ckw::TensorSamplerAddressModeY::None);
        }
        else
        {
            sampler_dst.address_mode_y(ckw::TensorSamplerAddressModeY::ClampToBorderMaxOnly);
        }
        sampler_dst.address_mode_z(ckw::TensorSamplerAddressModeZ::None);
        sampler_dst.storage(ckw::TensorStorageType::BufferUint8Ptr);

        // Declare destination tile
        ckw::DataType dst_dt   = to_ckw(_dst->data_type());
        auto          tile_dst = writer->declare_tile("dst", ckw::TileInfo(dst_dt, dst_m0, dst_n0));

        // Bind tile to the tensor
        dst->init_virtual_tensor(tile_dst, sampler_dst);
    }
    else
    {
        // Change dst_n0 and dst_m0 if NOT root component!
        dst_n0 = dst->tile().tile_info().width();
        dst_m0 = dst->tile().tile_info().height();

        // Here, it is not required the calculation of dst_n0_partial and dst_m0_partial
        // because if we enter this condition it means that the element-wise op is not the
        // root component and the address modes have been already set.
    }

    const auto &tile_dst = dst->tile();

    /********************************************************************************
     * 4 - Define the compute block parameters CKW constants
     ********************************************************************************/
    // ...

    /********************************************************************************
     * 5 - Define the samplers for the input tensors
     ********************************************************************************/
    // Check whether the lhs tensor is a tile or tensor
    // If it is a tile, create a sampler and load the content in a tile
    if (!lhs->has_tile())
    {
        // Sampler
        ckw::TensorSampler sampler_lhs = dst->tensor_sampler();

        bool broadcast_x = false;
        bool broadcast_y = false;

        int32_t lhs_n0 = dst_n0;
        int32_t lhs_m0 = dst_m0;

        // Check whether we have broadcasting
        // In case of broadcast, lhs can only be a vector or scalar.
        // Broadcasting in other dimensions is not supported
        if (_dst->dimension(0) != _lhs->dimension(0))
        {
            broadcast_x = true;
            lhs_n0      = 1;
        }

        if (sampler_lhs.format() == ckw::TensorSamplerFormat::Dim0_Dim1xDim2_1)
        {
            if (_dst->dimension(1) * _dst->dimension(2) != _lhs->dimension(1) * _lhs->dimension(2))
            {
                broadcast_y = true;
                lhs_m0      = 1;
            }
        }
        else if (sampler_lhs.format() == ckw::TensorSamplerFormat::Dim0_Dim1_Dim2)
        {
            if (_dst->dimension(1) != _lhs->dimension(1))
            {
                broadcast_y = true;
                lhs_m0      = 1;
            }
        }

        const int32_t lhs_partial_n0 = _lhs->dimension(0) % lhs_n0;
        const int32_t lhs_shift_back = (lhs_n0 - lhs_partial_n0) % lhs_n0;

        // Constants
        auto const_lhs_n0_i32 = writer->declare_constant_tile(ckw::ConstantData({{lhs_n0}}, ckw::DataType::Int32));
        auto const_lhs_m0_i32 = writer->declare_constant_tile(ckw::ConstantData({{lhs_m0}}, ckw::DataType::Int32));
        auto const_lhs_shift_back_n0_i32 =
            writer->declare_constant_tile(ckw::ConstantData({{lhs_shift_back}}, ckw::DataType::Int32));

        auto tile_gid_0 = writer->declare_tile("gid_0_lhs", ckw::TileInfo(ckw::DataType::Int32));
        auto tile_gid_1 = writer->declare_tile("gid_1_lhs", ckw::TileInfo(ckw::DataType::Int32));
        auto tile_gid_2 = writer->declare_tile("gid_2_lhs", ckw::TileInfo(ckw::DataType::Int32));

        writer->op_get_global_id(tile_gid_0, 0);
        writer->op_get_global_id(tile_gid_1, 1);
        writer->op_get_global_id(tile_gid_2, 2);

        auto tile_cout0 = writer->declare_tile("cout0_lhs", ckw::TileInfo(ckw::DataType::Int32)); // OFM
        auto tile_mout0 =
            writer->declare_tile("mout0_lhs", ckw::TileInfo(ckw::DataType::Int32)); // WIDTH or WIDTH x HEIGHT
        auto tile_mout1 = writer->declare_tile("mout1_lhs", ckw::TileInfo(ckw::DataType::Int32)); // HEIGHT or 0
        auto tile_bout0 = writer->declare_tile("bout0_lhs", ckw::TileInfo(ckw::DataType::Int32)); // BATCH SIZE IDX

        // Calculate coordinates
        if (!broadcast_x)
        {
            get_coordinate_from_gws_overlapping_min(writer, tile_cout0, tile_gid_0, const_lhs_n0_i32,
                                                    const_lhs_shift_back_n0_i32, const_0_i32);
        }
        else
        {
            writer->op_assign(tile_cout0, const_0_i32);
        }

        if (!broadcast_y)
        {
            get_coordinate_from_gws(writer, tile_mout0, tile_gid_1, const_lhs_m0_i32);
        }
        else
        {
            writer->op_assign(tile_mout0, const_0_i32);
        }

        // Get the boundary aware coordinates at each global dimension index
        if (sampler_lhs.format() == ckw::TensorSamplerFormat::Dim0_Dim1xDim2_1)
        {
            writer->op_assign(tile_mout1, const_0_i32);
            get_coordinate_from_gws(writer, tile_bout0, tile_gid_2, const_pos_1_i32);
        }
        else if (sampler_lhs.format() == ckw::TensorSamplerFormat::Dim0_Dim1_Dim2)
        {
            // For tile_mout1 and tile_bout0 the step can only be 1
            if (!broadcast_y)
            {
                writer->op_binary(tile_mout1, ckw::BinaryOp::Mod, tile_gid_2, const_dst_h_i32);
            }
            else
            {
                // If broadcast_y == true, it means that we have either a scalar or vector
                // because broadcasting in other dimensions is not supported
                writer->op_assign(tile_mout1, const_0_i32);
            }

            writer->op_binary(tile_bout0, ckw::BinaryOp::Div, tile_gid_2, const_dst_h_i32);
        }

        ckw::DataType lhs_dt   = to_ckw(_lhs->data_type());
        auto          tile_lhs = writer->declare_tile("lhs", ckw::TileInfo(lhs_dt, lhs_m0, lhs_n0));

        writer->op_load(tile_lhs, lhs->tensor(), sampler_lhs, tile_cout0, tile_mout0, tile_mout1, tile_bout0);

        // Here, init_virtual_tensor() is used to bring the tile_lhs outside the compound statement
        lhs->init_virtual_tensor(tile_lhs, sampler_lhs);
    }

    // Check whether the rhs tensor is a tile or tensor
    // If it is a tile, create a sampler and load the content in a tile
    if (!rhs->has_tile())
    {
        // Sampler
        ckw::TensorSampler sampler_rhs = dst->tensor_sampler();

        bool broadcast_x = false;
        bool broadcast_y = false;

        int32_t rhs_n0 = dst_n0;
        int32_t rhs_m0 = dst_m0;

        // Check whether we have broadcasting
        // In case of broadcast, rhs can only be a vector or scalar.
        // Broadcasting in other dimensions is not supported
        if (_dst->dimension(0) != _rhs->dimension(0))
        {
            broadcast_x = true;
            rhs_n0      = 1;
        }

        if (sampler_rhs.format() == ckw::TensorSamplerFormat::Dim0_Dim1xDim2_1)
        {
            if (_dst->dimension(1) * _dst->dimension(2) != _rhs->dimension(1) * _rhs->dimension(2))
            {
                broadcast_y = true;
                rhs_m0      = 1;
            }
        }
        else if (sampler_rhs.format() == ckw::TensorSamplerFormat::Dim0_Dim1_Dim2)
        {
            if (_dst->dimension(1) != _rhs->dimension(1))
            {
                broadcast_y = true;
                rhs_m0      = 1;
            }
        }

        const int32_t rhs_partial_n0 = _rhs->dimension(0) % rhs_n0;
        const int32_t rhs_shift_back = (rhs_n0 - rhs_partial_n0) % rhs_n0;

        // Constants
        auto const_rhs_n0_i32 = writer->declare_constant_tile(ckw::ConstantData({{rhs_n0}}, ckw::DataType::Int32));
        auto const_rhs_m0_i32 = writer->declare_constant_tile(ckw::ConstantData({{rhs_m0}}, ckw::DataType::Int32));
        auto const_rhs_shift_back_n0_i32 =
            writer->declare_constant_tile(ckw::ConstantData({{rhs_shift_back}}, ckw::DataType::Int32));

        auto tile_gid_0 = writer->declare_tile("gid_0_rhs", ckw::TileInfo(ckw::DataType::Int32));
        auto tile_gid_1 = writer->declare_tile("gid_1_rhs", ckw::TileInfo(ckw::DataType::Int32));
        auto tile_gid_2 = writer->declare_tile("gid_2_rhs", ckw::TileInfo(ckw::DataType::Int32));

        writer->op_get_global_id(tile_gid_0, 0);
        writer->op_get_global_id(tile_gid_1, 1);
        writer->op_get_global_id(tile_gid_2, 2);

        auto tile_cout0 = writer->declare_tile("cout0_rhs", ckw::TileInfo(ckw::DataType::Int32)); // OFM
        auto tile_mout0 =
            writer->declare_tile("mout0_rhs", ckw::TileInfo(ckw::DataType::Int32)); // WIDTH or WIDTH x HEIGHT
        auto tile_mout1 = writer->declare_tile("mout1_rhs", ckw::TileInfo(ckw::DataType::Int32)); // HEIGHT or 0
        auto tile_bout0 = writer->declare_tile("bout0_rhs", ckw::TileInfo(ckw::DataType::Int32)); // BATCH SIZE IDX

        // Calculate coordinates
        if (!broadcast_x)
        {
            get_coordinate_from_gws_overlapping_min(writer, tile_cout0, tile_gid_0, const_rhs_n0_i32,
                                                    const_rhs_shift_back_n0_i32, const_0_i32);
        }
        else
        {
            writer->op_assign(tile_cout0, const_0_i32);
        }

        if (!broadcast_y)
        {
            get_coordinate_from_gws(writer, tile_mout0, tile_gid_1, const_rhs_m0_i32);
        }
        else
        {
            writer->op_assign(tile_mout0, const_0_i32);
        }

        // Get the boundary aware coordinates at each global dimension index
        if (sampler_rhs.format() == ckw::TensorSamplerFormat::Dim0_Dim1xDim2_1)
        {
            writer->op_assign(tile_mout1, const_0_i32);
            get_coordinate_from_gws(writer, tile_bout0, tile_gid_2, const_pos_1_i32);
        }
        else if (sampler_rhs.format() == ckw::TensorSamplerFormat::Dim0_Dim1_Dim2)
        {
            // For tile_mout1 and tile_bout0 the step can only be 1
            const auto src_w       = static_cast<int32_t>(_rhs->dimension(1));
            auto       const_src_w = writer->declare_constant_tile(ckw::ConstantData({{src_w}}, ckw::DataType::Int32));
            if (!broadcast_y)
            {
                writer->op_binary(tile_mout1, ckw::BinaryOp::Mod, tile_mout1, const_src_w);
            }
            else
            {
                // If broadcast_y == true, it means that we have either a scalar or vector
                // because broadcasting in other dimensions is not supported
                writer->op_assign(tile_mout1, const_0_i32);
            }

            writer->op_binary(tile_bout0, ckw::BinaryOp::Div, tile_mout1, const_src_w);
        }

        ckw::DataType rhs_dt   = to_ckw(_rhs->data_type());
        auto          tile_rhs = writer->declare_tile("rhs", ckw::TileInfo(rhs_dt, rhs_m0, rhs_n0));

        writer->op_load(tile_rhs, rhs->tensor(), sampler_rhs, tile_cout0, tile_mout0, tile_mout1, tile_bout0);

        // Here, init_virtual_tensor() is used to bring the tile_rhs outside the compound statement
        rhs->init_virtual_tensor(tile_rhs, sampler_rhs);
    }

    const auto &tile_lhs = lhs->tile();
    const auto &tile_rhs = rhs->tile();

    /********************************************************************************
     * 7 - Write the rest of the code
     ********************************************************************************/
    // Perform the element-wise operation
    writer->op_binary(tile_dst, to_ckw(_attributes), tile_lhs, tile_rhs);

    ARM_COMPUTE_ERROR_ON_MSG(dst->has_tile() == false, "You must bind a tile before appending another component");
}

Window GpuCkwElementwiseBinary::get_window() const
{
    ARM_COMPUTE_ERROR_ON_MSG(_dst->tensor_shape().total_size() == 0U, "Destination tensor is not initialized");

    TensorShape output_shape = _dst->tensor_shape();
    // Collapse Dim 1 (W) and Dim 2 (H) together, leave Dim 0 (C) unchanged
    // This is in line with the collapsing convention used by operators like Conv2d
    output_shape.collapse(2U, 1U);
    constexpr uint32_t vector_size_byte_opencl = 16;
    const uint32_t     num_elems_processed_per_iteration =
        adjust_vec_size(vector_size_byte_opencl / _dst->element_size(), _dst->dimension(0));
    Window win = calculate_max_window(output_shape, Steps(num_elems_processed_per_iteration));

    return win;
}

std::string GpuCkwElementwiseBinary::get_name(const ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);
    const std::vector<std::string> build_params = {
        "elementwise_binary",
        "op",
        to_string(_attributes.operation()),
        "dt",
        lower_string(string_from_data_type(_dst->data_type())),
    };
    return join(build_params, "_");
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
