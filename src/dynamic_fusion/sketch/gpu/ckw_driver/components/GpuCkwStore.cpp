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
#include "GpuCkwStore.h"

#include "arm_compute/core/Error.h"

#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/CkwHelper.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwScopedKernelWriter.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwVariableTable.h"

#include <cstdint>
#include <string>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
GpuCkwStore::GpuCkwStore(ComponentId id, const ArgumentPack<ITensorInfo> &tensors)
    : IGpuCkwComponentDriver{id, tensors}, _src{}, _dst{}
{
    _src = this->tensors().get_const_tensor(TensorType::ACL_SRC_0);
    _dst = this->tensors().get_const_tensor(TensorType::ACL_DST_0);
}
void GpuCkwStore::write_component_code(const ComponentGroup    &comp_group,
                                       GpuCkwVariableTable     &vtable,
                                       GpuCkwScopedKernelWriter writer) const
{
    /********************************************************************************
     * 1 - Define tensors
     ********************************************************************************/
    GpuCkwComponentArgument *src = vtable.declare_variable(comp_group, writer, _src, "src");
    GpuCkwComponentArgument *dst = vtable.declare_variable(comp_group, writer, _dst, "dst");

    /********************************************************************************
     * 2 - Define CKW constants
     ********************************************************************************/
    const auto dst_h = static_cast<int32_t>(_dst->dimension(2));

    auto const_0_i32     = writer->declare_constant_tile(ckw::ConstantData({{0}}, ckw::DataType::Int32));
    auto const_pos_1_i32 = writer->declare_constant_tile(ckw::ConstantData({{1}}, ckw::DataType::Int32));
    auto const_dst_h_i32 = writer->declare_constant_tile(ckw::ConstantData({{dst_h}}, ckw::DataType::Int32));

    /********************************************************************************
     * 3 - Define the compute block parameters and destination tile (if not root component)
     *     Bind the tile to the tensor to share it among different components and
     *     initialize the compute block parameters
     ********************************************************************************/
    const auto &tile_src    = src->tile();
    auto       &sampler_src = src->tensor_sampler();

    const auto    dst_n0         = static_cast<int32_t>(tile_src.tile_info().width());
    const auto    dst_m0         = static_cast<int32_t>(tile_src.tile_info().height());
    const int32_t dst_n0_partial = _dst->dimension(0) % dst_n0;
    const int32_t dst_shift_back = (dst_n0 - dst_n0_partial) % dst_n0;

    /********************************************************************************
     * 4 - Define the compute block parameters CKW constants
     ********************************************************************************/
    auto const_n0_i32 = writer->declare_constant_tile(ckw::ConstantData({{dst_n0}}, ckw::DataType::Int32));
    auto const_m0_i32 = writer->declare_constant_tile(ckw::ConstantData({{dst_m0}}, ckw::DataType::Int32));
    auto const_shift_back_n0_i32 =
        writer->declare_constant_tile(ckw::ConstantData({{dst_shift_back}}, ckw::DataType::Int32));

    /********************************************************************************
     * 5 - Define the samplers for the input tensor
     ********************************************************************************/
    // Not required

    /********************************************************************************
     * 6 - Extra operations required before writing the main code
     ********************************************************************************/
    // Not required

    /********************************************************************************
     * 7 - Get the coordinates of the destination tile
     ********************************************************************************/
    auto tile_gid_0 = writer->declare_tile("gid_0", ckw::TileInfo(ckw::DataType::Int32));
    auto tile_gid_1 = writer->declare_tile("gid_1", ckw::TileInfo(ckw::DataType::Int32));
    auto tile_gid_2 = writer->declare_tile("gid_2", ckw::TileInfo(ckw::DataType::Int32));

    writer->op_get_global_id(tile_gid_0, 0);
    writer->op_get_global_id(tile_gid_1, 1);
    writer->op_get_global_id(tile_gid_2, 2);

    auto tile_nout0 = writer->declare_tile("cout0", ckw::TileInfo(ckw::DataType::Int32)); // OFM
    auto tile_mout0 = writer->declare_tile("mout0", ckw::TileInfo(ckw::DataType::Int32)); // WIDTH or WIDTH x HEIGHT
    auto tile_mout1 = writer->declare_tile("mout1", ckw::TileInfo(ckw::DataType::Int32)); // HEIGHT or 0
    auto tile_bout0 = writer->declare_tile("bout0", ckw::TileInfo(ckw::DataType::Int32)); // BATCH SIZE IDX

    // Calculate coordinates
    get_coordinate_from_gws_overlapping_min(writer, tile_nout0, tile_gid_0, const_n0_i32, const_shift_back_n0_i32,
                                            const_0_i32);
    get_coordinate_from_gws(writer, tile_mout0, tile_gid_1, const_m0_i32);

    // Get the boundary aware coordinates at each global dimension index
    if (sampler_src.format() == ckw::TensorSamplerFormat::Dim0_Dim1xDim2_1)
    {
        writer->op_assign(tile_mout1, const_0_i32);
        get_coordinate_from_gws(writer, tile_bout0, tile_gid_2, const_pos_1_i32);
    }
    else if (sampler_src.format() == ckw::TensorSamplerFormat::Dim0_Dim1_Dim2)
    {
        // For tile_mout1 and tile_bout0 the step can only be 1
        writer->op_binary(tile_mout1, ckw::BinaryOp::Mod, tile_gid_2, const_dst_h_i32);
        writer->op_binary(tile_bout0, ckw::BinaryOp::Div, tile_gid_2, const_dst_h_i32);
    }

    /********************************************************************************
     * 8 - Write the rest of the code
     ********************************************************************************/
    writer->op_store(dst->tensor(), tile_src, sampler_src, tile_nout0, tile_mout0, tile_mout1, tile_bout0);
}

std::string GpuCkwStore::get_name(const ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);
    return "store";
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
