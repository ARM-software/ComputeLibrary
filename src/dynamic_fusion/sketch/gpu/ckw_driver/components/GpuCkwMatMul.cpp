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

#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/GpuCkwMatMul.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/Validate.h"

#include "src/core/helpers/WindowHelpers.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/CkwHelper.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/type_converter/Common.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwComponentArgument.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwVariableTable.h"
#include "support/StringSupport.h"

#include "compute_kernel_writer/include/ckw/KernelWriter.h"
#include <cstdint>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{

GpuCkwMatMul::GpuCkwMatMul(ComponentId                      id,
                           const ArgumentPack<ITensorInfo> &tensors,
                           const Attributes                &attributes,
                           const Settings                  &settings)
    : IGpuCkwComponentDriver{id, tensors}, _lhs{}, _rhs{}, _dst{}, _attributes{attributes}, _settings{settings}
{
    _lhs = this->tensors().get_const_tensor(TensorType::ACL_SRC_0);
    _rhs = this->tensors().get_const_tensor(TensorType::ACL_SRC_1);
    _dst = this->tensors().get_const_tensor(TensorType::ACL_DST_0);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_lhs, _rhs, _dst);
}

void GpuCkwMatMul::write_component_code(const ComponentGroup    &comp_group,
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
    const auto k =
        _attributes.adj_lhs() ? static_cast<int32_t>(_lhs->dimension(1)) : static_cast<int32_t>(_lhs->dimension(0));
    const auto k0     = static_cast<int32_t>(adjust_vec_size(_settings.k0(), k));
    const auto dst_dt = to_ckw(_dst->data_type());

    // CKW constants
    auto const_k_i32          = writer->declare_constant_tile(ckw::ConstantData({{k}}, ckw::DataType::Int32));
    auto const_k0_i32         = writer->declare_constant_tile(ckw::ConstantData({{k0}}, ckw::DataType::Int32));
    auto const_0_i32          = writer->declare_constant_tile(ckw::ConstantData({{0}}, ckw::DataType::Int32));
    auto const_pos_1_i32      = writer->declare_constant_tile(ckw::ConstantData({{1}}, ckw::DataType::Int32));
    auto const_0_fp           = writer->declare_constant_tile(ckw::ConstantData({{0.0f}}, dst_dt));
    auto const_k_minus_k0_i32 = writer->declare_constant_tile(ckw::ConstantData({{k - k0}}, ckw::DataType::Int32));

    /********************************************************************************
     * 3 - Define the compute block parameters and destination tile (if not root component)
     *     Bind the tile to the tensor to share it among different components and
     *     initialize the compute block parameters
     ********************************************************************************/
    // The n0 and m0 parameters from root_window only refers to the output
    const auto root_window = comp_group.get_root_component()->ckw_component_driver()->get_window();

    // Destination compute block size
    const int32_t dst_n0 = root_window.x().step();
    const int32_t dst_m0 = root_window.y().step();

    // Destination compute block size left-over
    const int32_t dst_n0_partial = _dst->dimension(0) % dst_n0;
    const int32_t dst_m0_partial = _dst->dimension(1) % dst_m0;

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
    auto tile_dst = writer->declare_tile("dst", ckw::TileInfo(dst_dt, dst_m0, dst_n0));

    // Initialize destination tile
    writer->op_assign(tile_dst, const_0_fp);

    // Bind tile to the tensor
    dst->init_virtual_tensor(tile_dst, sampler_dst);

    /********************************************************************************
     * 4 - Define the compute block parameters CKW constants
     ********************************************************************************/
    // Only now we can declare the N0 and M0 as constant
    auto const_dst_n0_i32 = writer->declare_constant_tile(ckw::ConstantData({{dst_n0}}, ckw::DataType::Int32));
    auto const_dst_m0_i32 = writer->declare_constant_tile(ckw::ConstantData({{dst_m0}}, ckw::DataType::Int32));
    auto const_shift_back_dst_n0_i32 =
        writer->declare_constant_tile(ckw::ConstantData({{dst_shift_back}}, ckw::DataType::Int32));

    /********************************************************************************
     * 5 - Define the samplers for the input tensors
     ********************************************************************************/
    // LHS SAMPLER
    // The assumption here is that M is multiple of M0. This limitation will be removed once
    // we have the support for OverlappingMin as address mode for the Y direction
    ckw::TensorSampler sampler_lhs;
    sampler_lhs.format(ckw::TensorSamplerFormat::Dim0_Dim1_Dim2);
    sampler_lhs.address_mode_x(ckw::TensorSamplerAddressModeX::None);
    sampler_lhs.address_mode_y(ckw::TensorSamplerAddressModeY::None);
    sampler_lhs.address_mode_z(ckw::TensorSamplerAddressModeZ::None);
    sampler_lhs.storage(ckw::TensorStorageType::BufferUint8Ptr);

    // RHS SAMPLER
    ckw::TensorSampler sampler_rhs;
    sampler_rhs.format(ckw::TensorSamplerFormat::Dim0_Dim1_Dim2);
    sampler_rhs.address_mode_x(ckw::TensorSamplerAddressModeX::None);
    sampler_rhs.address_mode_y(ckw::TensorSamplerAddressModeY::None);
    sampler_rhs.address_mode_z(ckw::TensorSamplerAddressModeZ::None);
    sampler_rhs.storage(ckw::TensorStorageType::BufferUint8Ptr);

    /********************************************************************************
     * 6 - Extra operations required before writing the main code (optional)
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

    auto tile_idx_n = writer->declare_tile("idx_n", ckw::TileInfo(ckw::DataType::Int32)); // N index
    auto tile_idx_m = writer->declare_tile("idx_m", ckw::TileInfo(ckw::DataType::Int32)); // M index
    auto tile_idx_b = writer->declare_tile("idx_b", ckw::TileInfo(ckw::DataType::Int32)); // BATCH index

    // Calculate coordinates
    get_coordinate_from_gws_overlapping_min(writer, tile_idx_n, tile_gid_0, const_dst_n0_i32,
                                            const_shift_back_dst_n0_i32, const_0_i32);
    get_coordinate_from_gws(writer, tile_idx_m, tile_gid_1, const_dst_m0_i32);
    get_coordinate_from_gws(writer, tile_idx_b, tile_gid_2, const_pos_1_i32);

    /********************************************************************************
     * 8 - Write the rest of the code
     ********************************************************************************/
    auto tile_idx_k = writer->declare_tile("idx_k", ckw::TileInfo(ckw::DataType::Int32)); // K index

    writer->op_assign(tile_idx_k, const_0_i32);

    // clang-format off
    writer->op_for_loop(tile_idx_k, ckw::BinaryOp::LessEqual, const_k_minus_k0_i32, tile_idx_k, ckw::AssignmentOp::Increment, const_k0_i32,
    [&]()
    {
        auto tile_lhs = writer->declare_tile("lhs", ckw::TileInfo(to_ckw(_lhs->data_type()), dst_m0, k0));
        auto tile_rhs = writer->declare_tile("rhs", ckw::TileInfo(to_ckw(_rhs->data_type()), dst_n0, k0));
        writer->op_assign(tile_lhs, const_0_fp);
        writer->op_assign(tile_rhs, const_0_fp);

        writer->op_load(tile_lhs, lhs->tensor(), sampler_lhs, tile_idx_k, tile_idx_m, tile_idx_b, const_0_i32);
        writer->op_load(tile_rhs, rhs->tensor(), sampler_rhs, tile_idx_k, tile_idx_n, tile_idx_b, const_0_i32);

        writer->op_binary(tile_dst, ckw::BinaryOp::MatMul_Nt_T, tile_lhs, tile_rhs);

    });

    // Left-over accumulations for when K is not a multiple of k0
    if(((k % k0) != 0))
    {
        writer->op_for_loop(tile_idx_k, ckw::BinaryOp::Less, const_k_i32, tile_idx_k, ckw::AssignmentOp::Increment, const_pos_1_i32, [&]()
        {
            auto tile_lhs = writer->declare_tile("lhs", ckw::TileInfo(to_ckw(_lhs->data_type()), dst_m0, 1));
            auto tile_rhs = writer->declare_tile("rhs", ckw::TileInfo(to_ckw(_rhs->data_type()), dst_n0, 1));
            writer->op_assign(tile_lhs, const_0_fp);
            writer->op_assign(tile_rhs, const_0_fp);

            writer->op_load(tile_lhs, lhs->tensor(), sampler_lhs, tile_idx_k, tile_idx_m, tile_idx_b, const_0_i32);
            writer->op_load(tile_rhs, rhs->tensor(), sampler_rhs, tile_idx_k, tile_idx_n, tile_idx_b, const_0_i32);

            writer->op_binary(tile_dst, ckw::BinaryOp::MatMul_Nt_T, tile_lhs, tile_rhs);
        });
    }
    // clang-format on
}

Window GpuCkwMatMul::get_window() const
{
    ARM_COMPUTE_ERROR_ON_MSG(_dst->tensor_shape().total_size() == 0U, "Destination tensor is not initialized");

    const int32_t m       = _dst->dimension(1);
    const int32_t n       = _dst->dimension(0);
    const bool    adj_lhs = _attributes.adj_lhs();

    const int32_t m0 = adj_lhs ? adjust_vec_size(_settings.m0(), m) : std::min(_settings.m0(), m);
    const int32_t n0 = adjust_vec_size(_settings.n0(), n);

    // Configure kernel window
    Window win = calculate_max_window(_dst->tensor_shape(), Steps(n0, m0));
    win        = win.collapse(win, Window::DimZ);

    return win;
}

std::string GpuCkwMatMul::get_name(const ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);

    std::string kernel_name("mat_mul_native");

    const int32_t m = _dst->dimension(1);
    const int32_t n = _dst->dimension(0);
    const int32_t k = _attributes.adj_lhs() ? _lhs->tensor_shape().y() : _lhs->tensor_shape().x();

    kernel_name += _attributes.adj_lhs() ? "_t" : "_nt";
    kernel_name += _attributes.adj_rhs() ? "_t" : "_nt";
    kernel_name += "_";
    kernel_name += support::cpp11::to_string(m);
    kernel_name += "_";
    kernel_name += support::cpp11::to_string(n);
    kernel_name += "_";
    kernel_name += support::cpp11::to_string(k);
    kernel_name += "_";
    kernel_name += support::cpp11::to_string(_dst->dimension(2));
    kernel_name += "_";
    kernel_name += support::cpp11::to_string(_settings.m0());
    kernel_name += "_";
    kernel_name += support::cpp11::to_string(_settings.n0());
    kernel_name += "_";
    kernel_name += support::cpp11::to_string(_settings.k0());

    return kernel_name;
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
