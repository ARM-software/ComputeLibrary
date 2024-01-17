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

#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/GpuCkwResize.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/Validate.h"

#include "src/core/helpers/WindowHelpers.h"
#include "src/core/utils/ScaleUtils.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/CkwHelper.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/type_converter/Common.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwScopedKernelWriter.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwVariableTable.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelArgument.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGroup.h"
#include "support/StringSupport.h"

#include <cstdint>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
namespace
{
constexpr uint32_t opencl_vector_size_in_bytes = 16;
} // namespace

GpuCkwResize::GpuCkwResize(ComponentId id, const ArgumentPack<ITensorInfo> &tensors, const Attributes &attributes)
    : IGpuCkwComponentDriver{id, tensors}, _src{}, _dst{}, _attributes{attributes}
{
    _src = this->tensors().get_const_tensor(TensorType::ACL_SRC);
    _dst = this->tensors().get_const_tensor(TensorType::ACL_DST);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);
}

void GpuCkwResize::do_nearest_neighbor_resize(const ComponentGroup    &comp_group,
                                              GpuCkwVariableTable     &vtable,
                                              GpuCkwScopedKernelWriter writer) const
{
    const uint32_t width_idx  = get_data_layout_dimension_index(_dst->data_layout(), DataLayoutDimension::WIDTH);
    const uint32_t height_idx = get_data_layout_dimension_index(_dst->data_layout(), DataLayoutDimension::HEIGHT);

    /********************************************************************************
     * 1 - Define tensors
     ********************************************************************************/
    GpuCkwComponentArgument *src = vtable.declare_variable(comp_group, writer, _src, "src");
    GpuCkwComponentArgument *dst = vtable.declare_variable(comp_group, writer, _dst, "dst");

    /********************************************************************************
     * 2 - Define CKW constants
     ********************************************************************************/
    const auto  dst_dt  = to_ckw(_dst->data_type());
    const float scale_x = scale_utils::calculate_resize_ratio(_src->dimension(width_idx), _dst->dimension(width_idx),
                                                              _attributes.align_corners());
    const float scale_y = scale_utils::calculate_resize_ratio(_src->dimension(height_idx), _dst->dimension(height_idx),
                                                              _attributes.align_corners());
    const auto  src_w   = static_cast<int32_t>(_src->dimension(width_idx));
    const auto  src_h   = static_cast<int32_t>(_src->dimension(height_idx));
    const auto  dst_h   = static_cast<int32_t>(_dst->dimension(height_idx));

    // CKW constants
    auto const_src_w_i32  = writer->declare_constant_tile(ckw::ConstantData({{src_w}}, ckw::DataType::Int32));
    auto const_src_h_i32  = writer->declare_constant_tile(ckw::ConstantData({{src_h}}, ckw::DataType::Int32));
    auto const_dst_h_i32  = writer->declare_constant_tile(ckw::ConstantData({{dst_h}}, ckw::DataType::Int32));
    auto const_pos_1_i32  = writer->declare_constant_tile(ckw::ConstantData({{1}}, ckw::DataType::Int32));
    auto const_0_i32      = writer->declare_constant_tile(ckw::ConstantData({{0}}, ckw::DataType::Int32));
    auto const_0_fp       = writer->declare_constant_tile(ckw::ConstantData({{0.0f}}, dst_dt));
    auto const_pos_0_5_fp = writer->declare_constant_tile(ckw::ConstantData({{0.5f}}, ckw::DataType::Fp32));
    auto const_scale_x_fp = writer->declare_constant_tile(ckw::ConstantData({{scale_x}}, ckw::DataType::Fp32));
    auto const_scale_y_fp = writer->declare_constant_tile(ckw::ConstantData({{scale_y}}, ckw::DataType::Fp32));

    /********************************************************************************
     * 3 - Define the compute block parameters and destination tile (if not root component)
     *     Bind the tile to the tensor to share it among different components and
     *     initialize the compute block parameters
     ********************************************************************************/
    // The n0 and m0 parameters from root_window only refers to the output
    const auto root_window = comp_group.get_root_component()->ckw_component_driver()->get_window();

    // Destination compute block size
    const int32_t dst_n0 = root_window.x().step();

    // dst_m0 must be 1
    ARM_COMPUTE_ERROR_ON(root_window.y().step() != 1);

    // Destination compute block size left-over
    const int32_t dst_n0_partial = _dst->dimension(0) % dst_n0;

    // Shift-back for the overlapping-min strategy
    const int32_t dst_shift_back = (dst_n0 - dst_n0_partial) % dst_n0;

    ckw::TensorSampler sampler_dst;
    sampler_dst.format(ckw::TensorSamplerFormat::Dim0_Dim1_Dim2);
    if (dst_n0_partial == 0)
    {
        sampler_dst.address_mode_x(ckw::TensorSamplerAddressModeX::None);
    }
    else
    {
        sampler_dst.address_mode_x(ckw::TensorSamplerAddressModeX::OverlappingMin);
    }
    sampler_dst.address_mode_y(ckw::TensorSamplerAddressModeY::None);
    sampler_dst.address_mode_z(ckw::TensorSamplerAddressModeZ::None);
    sampler_dst.storage(ckw::TensorStorageType::BufferUint8Ptr);

    // Declare destination tile
    auto tile_dst = writer->declare_tile("dst", ckw::TileInfo(dst_dt, 1, dst_n0));

    // Initialize destination tile
    writer->op_assign(tile_dst, const_0_fp);

    // Bind tile to the tensor
    dst->init_virtual_tensor(tile_dst, sampler_dst);

    /********************************************************************************
     * 4 - Define the compute block parameters CKW constants
     ********************************************************************************/
    auto const_n0_i32 = writer->declare_constant_tile(ckw::ConstantData({{dst_n0}}, ckw::DataType::Int32));
    auto const_shift_back_n0_i32 =
        writer->declare_constant_tile(ckw::ConstantData({{dst_shift_back}}, ckw::DataType::Int32));

    /********************************************************************************
     * 5 - Define the samplers for the input tensor
     ********************************************************************************/
    ckw::TensorSampler sampler_src;
    sampler_src.format(ckw::TensorSamplerFormat::Dim0_Dim1_Dim2);
    sampler_src.address_mode_x(ckw::TensorSamplerAddressModeX::None);
    sampler_src.address_mode_y(ckw::TensorSamplerAddressModeY::None);
    sampler_src.address_mode_z(ckw::TensorSamplerAddressModeZ::None);

    /********************************************************************************
     * 6 - Extra operations required before writing the main code
     ********************************************************************************/

    // ....

    /********************************************************************************
     * 7 - Get the coordinates of the destination tile
     ********************************************************************************/
    auto tile_gid_0 = writer->declare_tile("gid_0", ckw::TileInfo(ckw::DataType::Int32));
    auto tile_gid_1 = writer->declare_tile("gid_1", ckw::TileInfo(ckw::DataType::Int32));
    auto tile_gid_2 = writer->declare_tile("gid_2", ckw::TileInfo(ckw::DataType::Int32));

    writer->op_get_global_id(tile_gid_0, 0);
    writer->op_get_global_id(tile_gid_1, 1);
    writer->op_get_global_id(tile_gid_2, 2);

    auto tile_co = writer->declare_tile("co", ckw::TileInfo(ckw::DataType::Int32)); // OFM
    auto tile_xo = writer->declare_tile("xo", ckw::TileInfo(ckw::DataType::Int32)); // WIDTH
    auto tile_yo = writer->declare_tile("yo", ckw::TileInfo(ckw::DataType::Int32)); // HEIGHT
    auto tile_bo = writer->declare_tile("bo", ckw::TileInfo(ckw::DataType::Int32)); // BATCH SIZE IDX

    // Calculate coordinates
    get_coordinate_from_gws_overlapping_min(writer, tile_co, tile_gid_0, const_n0_i32, const_shift_back_n0_i32,
                                            const_0_i32);
    writer->op_assign(tile_xo, tile_gid_1);
    writer->op_binary(tile_yo, ckw::BinaryOp::Mod, tile_gid_2, const_dst_h_i32);
    writer->op_binary(tile_bo, ckw::BinaryOp::Div, tile_gid_2, const_dst_h_i32);

    /********************************************************************************
     * 8 - Write the rest of the code
     ********************************************************************************/
    auto tile_xi_f = writer->declare_tile("xi_f", ckw::DataType::Fp32);
    auto tile_yi_f = writer->declare_tile("yi_f", ckw::DataType::Fp32);

    switch (_attributes.sampling_policy())
    {
        case SamplingPolicy::TOP_LEFT:
            // xi_f = (xo * scale_x)
            // yi_f = (yo * scale_y)
            writer->op_cast(tile_xi_f, tile_xo, ckw::ConvertPolicy::None);
            writer->op_cast(tile_yi_f, tile_yo, ckw::ConvertPolicy::None);
            writer->op_binary(tile_xi_f, ckw::BinaryOp::Mul, tile_xi_f, const_scale_x_fp);
            writer->op_binary(tile_yi_f, ckw::BinaryOp::Mul, tile_yi_f, const_scale_y_fp);
            break;
        case SamplingPolicy::CENTER:
        {
            // xi_f = ((xo + 0.5f) * scale_x)
            // yi_f = ((yo + 0.5f) * scale_y)
            const auto &tile_xo_plus_half = writer->declare_tile("xo_plus_half", ckw::DataType::Fp32);
            const auto &tile_yo_plus_half = writer->declare_tile("yo_plus_half", ckw::DataType::Fp32);

            writer->op_cast(tile_xo_plus_half, tile_xo, ckw::ConvertPolicy::None);
            writer->op_cast(tile_yo_plus_half, tile_yo, ckw::ConvertPolicy::None);
            writer->op_binary(tile_xo_plus_half, ckw::BinaryOp::Add, tile_xo_plus_half, const_pos_0_5_fp);
            writer->op_binary(tile_yo_plus_half, ckw::BinaryOp::Add, tile_yo_plus_half, const_pos_0_5_fp);
            writer->op_binary(tile_xi_f, ckw::BinaryOp::Mul, tile_xo_plus_half, const_scale_x_fp);
            writer->op_binary(tile_yi_f, ckw::BinaryOp::Mul, tile_yo_plus_half, const_scale_y_fp);
        }
        break;
        default:
            ARM_COMPUTE_ERROR("Unsupported sampling policy");
    }

    if (_attributes.align_corners())
    {
        writer->op_unary(tile_xi_f, ckw::UnaryOp::Round, tile_xi_f);
        writer->op_unary(tile_yi_f, ckw::UnaryOp::Round, tile_yi_f);
    }

    // xi0 = clamp((int)xi_f, 0, (int)src_w - 1)
    // yi0 = clamp((int)yi_f, 0, (int)src_h - 1)
    auto tile_xi_f_int = writer->declare_tile("xi_f_int", ckw::DataType::Int32);
    auto tile_yi_f_int = writer->declare_tile("yi_f_int", ckw::DataType::Int32);

    writer->op_cast(tile_xi_f_int, tile_xi_f, ckw::ConvertPolicy::None);
    writer->op_cast(tile_yi_f_int, tile_yi_f, ckw::ConvertPolicy::None);

    auto tile_src_w_minus_1 = writer->declare_tile("src_w_minus_1", ckw::DataType::Int32);
    auto tile_src_h_minus_1 = writer->declare_tile("src_h_minus_1", ckw::DataType::Int32);

    writer->op_binary(tile_src_w_minus_1, ckw::BinaryOp::Sub, const_src_w_i32, const_pos_1_i32);
    writer->op_binary(tile_src_h_minus_1, ckw::BinaryOp::Sub, const_src_h_i32, const_pos_1_i32);

    auto tile_xi0 = writer->declare_tile("xi0", ckw::DataType::Int32);
    auto tile_yi0 = writer->declare_tile("yi0", ckw::DataType::Int32);

    writer->op_ternary(tile_xi0, ckw::TernaryOp::Clamp, tile_xi_f_int, const_0_i32, tile_src_w_minus_1);
    writer->op_ternary(tile_yi0, ckw::TernaryOp::Clamp, tile_yi_f_int, const_0_i32, tile_src_h_minus_1);

    auto tile_src = writer->declare_tile("src_tile", ckw::TileInfo(dst_dt, 1, dst_n0));
    writer->op_load(tile_src, src->tensor(), sampler_src, tile_co, tile_xi0, tile_yi0, tile_bo);

    writer->op_assign(tile_dst, tile_src);
}

void GpuCkwResize::do_bilinear_resize(const ComponentGroup    &comp_group,
                                      GpuCkwVariableTable     &vtable,
                                      GpuCkwScopedKernelWriter writer) const
{
    const size_t width_idx  = get_data_layout_dimension_index(_dst->data_layout(), DataLayoutDimension::WIDTH);
    const size_t height_idx = get_data_layout_dimension_index(_dst->data_layout(), DataLayoutDimension::HEIGHT);

    /********************************************************************************
     * 1 - Define tensors
     ********************************************************************************/
    GpuCkwComponentArgument *src = vtable.declare_variable(comp_group, writer, _src, "src");
    GpuCkwComponentArgument *dst = vtable.declare_variable(comp_group, writer, _dst, "dst");

    /********************************************************************************
     * 2 - Define CKW constants
     ********************************************************************************/
    const auto  dst_dt  = to_ckw(_dst->data_type());
    const float scale_x = scale_utils::calculate_resize_ratio(_src->dimension(width_idx), _dst->dimension(width_idx),
                                                              _attributes.align_corners());
    const float scale_y = scale_utils::calculate_resize_ratio(_src->dimension(height_idx), _dst->dimension(height_idx),
                                                              _attributes.align_corners());
    const auto  src_w   = static_cast<int32_t>(_src->dimension(width_idx));
    const auto  src_h   = static_cast<int32_t>(_src->dimension(height_idx));
    const auto  dst_h   = static_cast<int32_t>(_dst->dimension(height_idx));

    // CKW constants
    auto const_src_w_i32  = writer->declare_constant_tile(ckw::ConstantData({{src_w}}, ckw::DataType::Int32));
    auto const_src_h_i32  = writer->declare_constant_tile(ckw::ConstantData({{src_h}}, ckw::DataType::Int32));
    auto const_dst_h_i32  = writer->declare_constant_tile(ckw::ConstantData({{dst_h}}, ckw::DataType::Int32));
    auto const_pos_1_i32  = writer->declare_constant_tile(ckw::ConstantData({{1}}, ckw::DataType::Int32));
    auto const_0_i32      = writer->declare_constant_tile(ckw::ConstantData({{0}}, ckw::DataType::Int32));
    auto const_0_fp       = writer->declare_constant_tile(ckw::ConstantData({{0.0f}}, dst_dt));
    auto const_pos_1_fp   = writer->declare_constant_tile(ckw::ConstantData({{1.0f}}, ckw::DataType::Fp32));
    auto const_pos_0_5_fp = writer->declare_constant_tile(ckw::ConstantData({{0.5f}}, ckw::DataType::Fp32));
    auto const_scale_x_fp = writer->declare_constant_tile(ckw::ConstantData({{scale_x}}, ckw::DataType::Fp32));
    auto const_scale_y_fp = writer->declare_constant_tile(ckw::ConstantData({{scale_y}}, ckw::DataType::Fp32));

    /********************************************************************************
     * 3 - Define the compute block parameters and destination tile (if not root component)
     *     Bind the tile to the tensor to share it among different components and
     *     initialize the compute block parameters
     ********************************************************************************/
    // The n0 and m0 parameters from root_window only refers to the output
    const auto root_window = comp_group.get_root_component()->ckw_component_driver()->get_window();

    // Destination compute block size
    const int32_t dst_n0 = root_window.x().step();

    // dst_m0 must be 1
    ARM_COMPUTE_ERROR_ON(root_window.y().step() != 1);

    // Destination compute block size left-over
    const int32_t dst_n0_partial = _dst->dimension(0) % dst_n0;

    // Shift-back for the overlapping-min strategy
    const int32_t dst_shift_back = (dst_n0 - dst_n0_partial) % dst_n0;

    ckw::TensorSampler sampler_dst;
    sampler_dst.format(ckw::TensorSamplerFormat::Dim0_Dim1_Dim2);
    if (dst_n0_partial == 0)
    {
        sampler_dst.address_mode_x(ckw::TensorSamplerAddressModeX::None);
    }
    else
    {
        sampler_dst.address_mode_x(ckw::TensorSamplerAddressModeX::OverlappingMin);
    }
    sampler_dst.address_mode_y(ckw::TensorSamplerAddressModeY::None);
    sampler_dst.address_mode_z(ckw::TensorSamplerAddressModeZ::None);
    sampler_dst.storage(ckw::TensorStorageType::BufferUint8Ptr);

    // Declare destination tile
    auto tile_dst = writer->declare_tile("dst", ckw::TileInfo(dst_dt, 1, dst_n0));

    // Initialize destination tile
    writer->op_assign(tile_dst, const_0_fp);

    // Bind tile to the tensor
    dst->init_virtual_tensor(tile_dst, sampler_dst);

    /********************************************************************************
     * 4 - Define the compute block parameters CKW constants
     ********************************************************************************/
    auto const_n0_i32 = writer->declare_constant_tile(ckw::ConstantData({{dst_n0}}, ckw::DataType::Int32));
    auto const_shift_back_n0_i32 =
        writer->declare_constant_tile(ckw::ConstantData({{dst_shift_back}}, ckw::DataType::Int32));

    /********************************************************************************
     * 5 - Define the sampler for the input tensor
     ********************************************************************************/
    ckw::TensorSampler sampler_src;
    sampler_src.format(ckw::TensorSamplerFormat::Dim0_Dim1_Dim2);
    sampler_src.address_mode_x(ckw::TensorSamplerAddressModeX::None);
    sampler_src.address_mode_y(ckw::TensorSamplerAddressModeY::None);
    sampler_src.address_mode_z(ckw::TensorSamplerAddressModeZ::None);

    /********************************************************************************
     * 6 - Extra operations required before writing the main code
     ********************************************************************************/

    // ....

    /********************************************************************************
     * 7 - Get the coordinates of the destination tile
     ********************************************************************************/
    auto tile_gid_0 = writer->declare_tile("gid_0", ckw::TileInfo(ckw::DataType::Int32));
    auto tile_gid_1 = writer->declare_tile("gid_1", ckw::TileInfo(ckw::DataType::Int32));
    auto tile_gid_2 = writer->declare_tile("gid_2", ckw::TileInfo(ckw::DataType::Int32));

    writer->op_get_global_id(tile_gid_0, 0);
    writer->op_get_global_id(tile_gid_1, 1);
    writer->op_get_global_id(tile_gid_2, 2);

    auto tile_co = writer->declare_tile("co", ckw::TileInfo(ckw::DataType::Int32)); // OFM
    auto tile_xo = writer->declare_tile("xo", ckw::TileInfo(ckw::DataType::Int32)); // WIDTH
    auto tile_yo = writer->declare_tile("yo", ckw::TileInfo(ckw::DataType::Int32)); // HEIGHT
    auto tile_bo = writer->declare_tile("bo", ckw::TileInfo(ckw::DataType::Int32)); // BATCH SIZE IDX

    // Calculate coordinates
    get_coordinate_from_gws_overlapping_min(writer, tile_co, tile_gid_0, const_n0_i32, const_shift_back_n0_i32,
                                            const_0_i32);
    writer->op_assign(tile_xo, tile_gid_1);
    writer->op_binary(tile_yo, ckw::BinaryOp::Mod, tile_gid_2, const_dst_h_i32);
    writer->op_binary(tile_bo, ckw::BinaryOp::Div, tile_gid_2, const_dst_h_i32);

    /********************************************************************************
     * 8 - Write the rest of the code
     ********************************************************************************/
    auto tile_xi_f = writer->declare_tile("xi_f", ckw::DataType::Fp32);
    auto tile_yi_f = writer->declare_tile("yi_f", ckw::DataType::Fp32);

    switch (_attributes.sampling_policy())
    {
        case SamplingPolicy::TOP_LEFT:
            // xi_f = (xo * scale_x)
            // yi_f = (yo * scale_y)
            writer->op_cast(tile_xi_f, tile_xo, ckw::ConvertPolicy::None);
            writer->op_cast(tile_yi_f, tile_yo, ckw::ConvertPolicy::None);
            writer->op_binary(tile_xi_f, ckw::BinaryOp::Mul, tile_xi_f, const_scale_x_fp);
            writer->op_binary(tile_yi_f, ckw::BinaryOp::Mul, tile_yi_f, const_scale_y_fp);
            break;
        case SamplingPolicy::CENTER:
        {
            // xi_f = ((xo + 0.5f) * scale_x - 0.5f)
            // yi_f = ((yo + 0.5f) * scale_y - 0.5f)
            const auto &tile_xo_plus_half = writer->declare_tile("xo_plus_half", ckw::DataType::Fp32);
            const auto &tile_yo_plus_half = writer->declare_tile("yo_plus_half", ckw::DataType::Fp32);

            writer->op_cast(tile_xo_plus_half, tile_xo, ckw::ConvertPolicy::None);
            writer->op_cast(tile_yo_plus_half, tile_yo, ckw::ConvertPolicy::None);
            writer->op_binary(tile_xo_plus_half, ckw::BinaryOp::Add, tile_xo_plus_half, const_pos_0_5_fp);
            writer->op_binary(tile_yo_plus_half, ckw::BinaryOp::Add, tile_yo_plus_half, const_pos_0_5_fp);
            writer->op_binary(tile_xi_f, ckw::BinaryOp::Mul, tile_xo_plus_half, const_scale_x_fp);
            writer->op_binary(tile_yi_f, ckw::BinaryOp::Mul, tile_yo_plus_half, const_scale_y_fp);

            writer->op_binary(tile_xi_f, ckw::BinaryOp::Sub, tile_xi_f, const_pos_0_5_fp);
            writer->op_binary(tile_yi_f, ckw::BinaryOp::Sub, tile_yi_f, const_pos_0_5_fp);
        }
        break;
        default:
            ARM_COMPUTE_ERROR("Unsupported sampling policy");
    }

    // xi = (int)floor(xi_f);
    // yi = (int)floor(yi_f);
    auto tile_xi_f_floor = writer->declare_tile("xi_f_floor", ckw::DataType::Fp32);
    auto tile_yi_f_floor = writer->declare_tile("yi_f_floor", ckw::DataType::Fp32);
    writer->op_unary(tile_xi_f_floor, ckw::UnaryOp::Floor, tile_xi_f);
    writer->op_unary(tile_yi_f_floor, ckw::UnaryOp::Floor, tile_yi_f);

    auto tile_xi = writer->declare_tile("xi", ckw::DataType::Int32);
    auto tile_yi = writer->declare_tile("yi", ckw::DataType::Int32);
    writer->op_cast(tile_xi, tile_xi_f_floor, ckw::ConvertPolicy::None);
    writer->op_cast(tile_yi, tile_yi_f_floor, ckw::ConvertPolicy::None);

    // xi0  = clamp(xi, 0, (int)src_w - 1);
    // yi0  = clamp(yi, 0, (int)src_h - 1);
    // xi1  = clamp(xi + 1, 0, (int)src_w - 1);
    // yi1  = clamp(yi + 1, 0, (int)src_h - 1);
    auto tile_src_w_minus_1 = writer->declare_tile("src_w_minus_1", ckw::DataType::Int32);
    auto tile_src_h_minus_1 = writer->declare_tile("src_h_minus_1", ckw::DataType::Int32);
    writer->op_binary(tile_src_w_minus_1, ckw::BinaryOp::Sub, const_src_w_i32, const_pos_1_i32);
    writer->op_binary(tile_src_h_minus_1, ckw::BinaryOp::Sub, const_src_h_i32, const_pos_1_i32);

    auto tile_xi_plus_1 = writer->declare_tile("xi_plus_1", ckw::DataType::Int32);
    auto tile_yi_plus_1 = writer->declare_tile("yi_plus_1", ckw::DataType::Int32);
    writer->op_binary(tile_xi_plus_1, ckw::BinaryOp::Add, tile_xi, const_pos_1_i32);
    writer->op_binary(tile_yi_plus_1, ckw::BinaryOp::Add, tile_yi, const_pos_1_i32);

    auto tile_xi0 = writer->declare_tile("xi0", ckw::DataType::Int32);
    auto tile_yi0 = writer->declare_tile("yi0", ckw::DataType::Int32);
    auto tile_xi1 = writer->declare_tile("xi1", ckw::DataType::Int32);
    auto tile_yi1 = writer->declare_tile("yi1", ckw::DataType::Int32);

    writer->op_ternary(tile_xi0, ckw::TernaryOp::Clamp, tile_xi, const_0_i32, tile_src_w_minus_1);
    writer->op_ternary(tile_yi0, ckw::TernaryOp::Clamp, tile_yi, const_0_i32, tile_src_h_minus_1);
    writer->op_ternary(tile_xi1, ckw::TernaryOp::Clamp, tile_xi_plus_1, const_0_i32, tile_src_w_minus_1);
    writer->op_ternary(tile_yi1, ckw::TernaryOp::Clamp, tile_yi_plus_1, const_0_i32, tile_src_h_minus_1);

    auto tile_in00 = writer->declare_tile("in00", ckw::TileInfo(dst_dt, 1, dst_n0));
    auto tile_in01 = writer->declare_tile("in01", ckw::TileInfo(dst_dt, 1, dst_n0));
    auto tile_in10 = writer->declare_tile("in10", ckw::TileInfo(dst_dt, 1, dst_n0));
    auto tile_in11 = writer->declare_tile("in11", ckw::TileInfo(dst_dt, 1, dst_n0));

    writer->op_load(tile_in00, src->tensor(), sampler_src, tile_co, tile_xi0, tile_yi0, tile_bo);
    writer->op_load(tile_in01, src->tensor(), sampler_src, tile_co, tile_xi1, tile_yi0, tile_bo);
    writer->op_load(tile_in10, src->tensor(), sampler_src, tile_co, tile_xi0, tile_yi1, tile_bo);
    writer->op_load(tile_in11, src->tensor(), sampler_src, tile_co, tile_xi1, tile_yi1, tile_bo);

    // Weights of each nearest pixel
    auto tile_a  = writer->declare_tile("a", ckw::DataType::Fp32);
    auto tile_b  = writer->declare_tile("b", ckw::DataType::Fp32);
    auto tile_a1 = writer->declare_tile("a1", ckw::DataType::Fp32);
    auto tile_b1 = writer->declare_tile("b1", ckw::DataType::Fp32);

    // a = (xi_f - (float)xi)
    // b = (1.f - a)
    // a1 = (yi_f - (float)yi)
    // b1 = (1.f - a1)
    auto tile_xi_float = writer->declare_tile("xi_float", ckw::DataType::Fp32);
    auto tile_yi_float = writer->declare_tile("yi_float", ckw::DataType::Fp32);
    writer->op_cast(tile_xi_float, tile_xi, ckw::ConvertPolicy::None);
    writer->op_cast(tile_yi_float, tile_yi, ckw::ConvertPolicy::None);

    writer->op_binary(tile_a, ckw::BinaryOp::Sub, tile_xi_f, tile_xi_float);
    writer->op_binary(tile_b, ckw::BinaryOp::Sub, const_pos_1_fp, tile_a);
    writer->op_binary(tile_a1, ckw::BinaryOp::Sub, tile_yi_f, tile_yi_float);
    writer->op_binary(tile_b1, ckw::BinaryOp::Sub, const_pos_1_fp, tile_a1);

    // Cast weights to source type
    const auto &tile_a_src_type  = writer->declare_tile("a_src_t", to_ckw(_src->data_type()));
    const auto &tile_b_src_type  = writer->declare_tile("b_src_t", to_ckw(_src->data_type()));
    const auto &tile_a1_src_type = writer->declare_tile("a1_src_t", to_ckw(_src->data_type()));
    const auto &tile_b1_src_type = writer->declare_tile("b1_src_t", to_ckw(_src->data_type()));

    writer->op_cast(tile_a_src_type, tile_a, ckw::ConvertPolicy::None);
    writer->op_cast(tile_b_src_type, tile_b, ckw::ConvertPolicy::None);
    writer->op_cast(tile_a1_src_type, tile_a1, ckw::ConvertPolicy::None);
    writer->op_cast(tile_b1_src_type, tile_b1, ckw::ConvertPolicy::None);

    // in00 * b * b1
    writer->op_binary(tile_in00, ckw::BinaryOp::Mul, tile_in00, tile_b_src_type);
    writer->op_binary(tile_in00, ckw::BinaryOp::Mul, tile_in00, tile_b1_src_type);

    // in01 * a * b1
    writer->op_binary(tile_in01, ckw::BinaryOp::Mul, tile_in01, tile_a_src_type);
    writer->op_binary(tile_in01, ckw::BinaryOp::Mul, tile_in01, tile_b1_src_type);

    // in10 * b * a1
    writer->op_binary(tile_in10, ckw::BinaryOp::Mul, tile_in10, tile_b_src_type);
    writer->op_binary(tile_in10, ckw::BinaryOp::Mul, tile_in10, tile_a1_src_type);

    // in11 * a * a1
    writer->op_binary(tile_in11, ckw::BinaryOp::Mul, tile_in11, tile_a_src_type);
    writer->op_binary(tile_in11, ckw::BinaryOp::Mul, tile_in11, tile_a1_src_type);

    // Summation of above terms
    writer->op_assign(tile_dst, tile_in00);
    writer->op_binary(tile_dst, ckw::BinaryOp::Add, tile_dst, tile_in01);
    writer->op_binary(tile_dst, ckw::BinaryOp::Add, tile_dst, tile_in10);
    writer->op_binary(tile_dst, ckw::BinaryOp::Add, tile_dst, tile_in11);
}

void GpuCkwResize::write_component_code(const ComponentGroup    &comp_group,
                                        GpuCkwVariableTable     &vtable,
                                        GpuCkwScopedKernelWriter writer) const
{
    switch (_attributes.interpolation_policy())
    {
        case InterpolationPolicy::NEAREST_NEIGHBOR:
            do_nearest_neighbor_resize(comp_group, vtable, writer);
            break;
        case InterpolationPolicy::BILINEAR:
            do_bilinear_resize(comp_group, vtable, writer);
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported interpolation policy");
    }
}

Window GpuCkwResize::get_window() const
{
    ARM_COMPUTE_ERROR_ON_MSG(_dst->tensor_shape().total_size() == 0U, "Destination tensor is not initialized");

    const uint32_t n0  = adjust_vec_size(opencl_vector_size_in_bytes / _src->element_size(), _src->dimension(0));
    Window         win = calculate_max_window(*_dst, Steps(n0));
    return win.collapse(win, Window::DimZ);
}

std::string GpuCkwResize::get_tuner_id(const ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);

    std::string tuner_id = "resize_";
    tuner_id += _attributes.interpolation_policy() == InterpolationPolicy::NEAREST_NEIGHBOR ? "nearest_neighbor" : "";
    tuner_id += _attributes.interpolation_policy() == InterpolationPolicy::BILINEAR ? "bilinear" : "";
    tuner_id += "_";
    tuner_id += _attributes.sampling_policy() == SamplingPolicy::CENTER ? "center" : "topleft";
    tuner_id += "_";
    tuner_id += support::cpp11::to_string(_dst->dimension(0));
    tuner_id += "_";
    tuner_id += support::cpp11::to_string(_dst->dimension(1));
    tuner_id += "_";
    tuner_id += support::cpp11::to_string(_dst->dimension(2));
    tuner_id += "_";
    tuner_id += support::cpp11::to_string(_dst->dimension(3));

    return tuner_id;
}

std::string GpuCkwResize::get_name(const ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);

    std::string name = "resize_";
    name += _attributes.interpolation_policy() == InterpolationPolicy::NEAREST_NEIGHBOR ? "nearest_neighbor" : "";
    name += _attributes.interpolation_policy() == InterpolationPolicy::BILINEAR ? "bilinear" : "";

    return name;
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
