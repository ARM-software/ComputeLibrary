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
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/GpuCkwDepthwiseConv2d.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/Validate.h"

#include "src/core/helpers/WindowHelpers.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/CkwHelper.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/type_converter/Common.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwScopedKernelWriter.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwVariableTable.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelArgument.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGroup.h"

#include "compute_kernel_writer/include/ckw/KernelWriter.h"
#include <cstdint>
#include <string>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
GpuCkwDepthwiseConv2d::GpuCkwDepthwiseConv2d(ComponentId                      id,
                                             const ArgumentPack<ITensorInfo> &tensors,
                                             const Attributes                &attributes,
                                             const Settings                  &settings)
    : IGpuCkwComponentDriver{id, tensors}, _src{}, _wei{}, _bia{}, _dst{}, _attributes{attributes}, _settings{settings}
{
    _src = this->tensors().get_const_tensor(TensorType::ACL_SRC_0);
    _wei = this->tensors().get_const_tensor(TensorType::ACL_SRC_1);
    if (this->tensors().get_const_tensor(TensorType::ACL_SRC_2))
    {
        _bia = this->tensors().get_const_tensor(TensorType::ACL_SRC_2);
    }
    _dst = this->tensors().get_const_tensor(TensorType::ACL_DST_0);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _wei, _bia, _dst);
}

void GpuCkwDepthwiseConv2d::write_component_code(const ComponentGroup    &comp_group,
                                                 GpuCkwVariableTable     &vtable,
                                                 GpuCkwScopedKernelWriter writer) const
{
    // Data Layout is NHWC
    const uint32_t width_idx  = get_data_layout_dimension_index(_wei->data_layout(), DataLayoutDimension::WIDTH);
    const uint32_t height_idx = get_data_layout_dimension_index(_wei->data_layout(), DataLayoutDimension::HEIGHT);

    /********************************************************************************
     * 1 - Define tensors
     ********************************************************************************/
    GpuCkwComponentArgument *src = vtable.declare_variable(comp_group, writer, _src, "src");
    GpuCkwComponentArgument *wei = vtable.declare_variable(comp_group, writer, _wei, "wei");
    GpuCkwComponentArgument *dst = vtable.declare_variable(comp_group, writer, _dst, "dst");
    GpuCkwComponentArgument *bia = nullptr;

    const bool using_bias = _bia != nullptr;

    if (using_bias)
    {
        bia = vtable.declare_variable(comp_group, writer, _bia, "bia");
    }

    /********************************************************************************
     * 2 - Define CKW constants
     ********************************************************************************/
    const auto dst_dt           = to_ckw(_dst->data_type());
    const auto kernel_height    = static_cast<int32_t>(_wei->dimension(height_idx));
    const auto kernel_width     = static_cast<int32_t>(_wei->dimension(width_idx));
    const auto src_w            = static_cast<int32_t>(_src->dimension(width_idx));
    const auto src_h            = static_cast<int32_t>(_src->dimension(height_idx));
    const auto dst_h            = static_cast<int32_t>(_dst->dimension(height_idx));
    const auto stride_x         = static_cast<int32_t>(_attributes.stride().x());
    const auto stride_y         = static_cast<int32_t>(_attributes.stride().y());
    const auto pad_x            = static_cast<int32_t>(_attributes.pad().left);
    const auto pad_y            = static_cast<int32_t>(_attributes.pad().top);
    const auto depth_multiplier = static_cast<int32_t>(_attributes.depth_multiplier());
    const auto dilation_x       = static_cast<int32_t>(_attributes.dilation().x());
    const auto dilation_y       = static_cast<int32_t>(_attributes.dilation().y());
    const auto kernel_size      = kernel_width * kernel_height;

    // CKW constants
    auto const_kernel_w_i32 = writer->declare_constant_tile(ckw::ConstantData({{kernel_width}}, ckw::DataType::Int32));
    auto const_kernel_size_i32 =
        writer->declare_constant_tile(ckw::ConstantData({{kernel_size}}, ckw::DataType::Int32));
    auto const_dst_h_i32    = writer->declare_constant_tile(ckw::ConstantData({{dst_h}}, ckw::DataType::Int32));
    auto const_src_w_i32    = writer->declare_constant_tile(ckw::ConstantData({{src_w}}, ckw::DataType::Int32));
    auto const_src_h_i32    = writer->declare_constant_tile(ckw::ConstantData({{src_h}}, ckw::DataType::Int32));
    auto const_stride_x_i32 = writer->declare_constant_tile(ckw::ConstantData({{stride_x}}, ckw::DataType::Int32));
    auto const_stride_y_i32 = writer->declare_constant_tile(ckw::ConstantData({{stride_y}}, ckw::DataType::Int32));
    auto const_pad_x_i32    = writer->declare_constant_tile(ckw::ConstantData({{pad_x}}, ckw::DataType::Int32));
    auto const_pad_y_i32    = writer->declare_constant_tile(ckw::ConstantData({{pad_y}}, ckw::DataType::Int32));
    auto const_0_i32        = writer->declare_constant_tile(ckw::ConstantData({{0}}, ckw::DataType::Int32));
    auto const_neg_1_i32    = writer->declare_constant_tile(ckw::ConstantData({{-1}}, ckw::DataType::Int32));
    auto const_depth_multiplier_i32 =
        writer->declare_constant_tile(ckw::ConstantData({{depth_multiplier}}, ckw::DataType::Int32));
    auto const_dilation_x_i32 = writer->declare_constant_tile(ckw::ConstantData({{dilation_x}}, ckw::DataType::Int32));
    auto const_dilation_y_i32 = writer->declare_constant_tile(ckw::ConstantData({{dilation_y}}, ckw::DataType::Int32));
    auto const_0_fp           = writer->declare_constant_tile(ckw::ConstantData({{0.0f}}, dst_dt));

    /********************************************************************************
     * 3 - Define the compute block parameters and destination tile (if not root component)
     *     Bind the tile to the tensor to share it among different components and
     *     initialize the compute block parameters
     ********************************************************************************/
    // The compute block parameters depend on the employed tensor format
    const auto root_window = comp_group.get_root_component()->ckw_component_driver()->get_window();

    // Destination compute block size
    const int32_t dst_n0 = root_window.x().step();
    const int32_t dst_m0 = root_window.y().step();

    // Destination compute block size left-over
    const int32_t dst_n0_partial = _dst->dimension(0) % dst_n0;
    const int32_t dst_m0_partial = _dst->dimension(1) % dst_m0;

    // Shift-back for the overlapping-min strategy
    const int32_t dst_shift_back = (dst_n0 - dst_n0_partial) % dst_n0;

    const int32_t src_m0 = kernel_width + (dst_m0 - 1);
    const int32_t src_n0 = depth_multiplier > 1 ? 1 : dst_n0;
    const int32_t wei_m0 = kernel_width;
    const int32_t wei_n0 = dst_n0;

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

    // Initialize the destination tile
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
     * 5 - Define the sampler for the input tensors
     ********************************************************************************/
    // SOURCE SAMPLER
    ckw::TensorSampler sampler_src;
    sampler_src.format(ckw::TensorSamplerFormat::Dim0_Dim1_Dim2);
    sampler_src.address_mode_x(ckw::TensorSamplerAddressModeX::None);
    sampler_src.address_mode_y(ckw::TensorSamplerAddressModeY::SkipLessThanZero);
    sampler_src.address_mode_z(ckw::TensorSamplerAddressModeZ::None);
    sampler_src.storage(ckw::TensorStorageType::BufferUint8Ptr);

    // WEIGHTS SAMPLER
    // We cannot have out-of-bounds accesses for the weights
    ckw::TensorSampler sampler_wei;
    sampler_wei.format(ckw::TensorSamplerFormat::Dim0_Dim1_Dim2);
    sampler_wei.address_mode_x(ckw::TensorSamplerAddressModeX::None);
    sampler_wei.address_mode_y(ckw::TensorSamplerAddressModeY::None);
    sampler_wei.address_mode_z(ckw::TensorSamplerAddressModeZ::None);
    if (_settings.export_weights_to_cl_image())
    {
        sampler_wei.storage(ckw::TensorStorageType::Texture2dReadOnly);
    }
    else
    {
        sampler_wei.storage(ckw::TensorStorageType::BufferUint8Ptr);
    }

    // BIAS SAMPLER
    ckw::TensorSampler sampler_bia;
    sampler_bia.format(ckw::TensorSamplerFormat::Dim0_Dim1_Dim2);
    sampler_bia.address_mode_x(sampler_dst.address_mode_x());
    sampler_bia.address_mode_y(ckw::TensorSamplerAddressModeY::None);
    sampler_bia.address_mode_z(ckw::TensorSamplerAddressModeZ::None);
    sampler_bia.storage(ckw::TensorStorageType::BufferUint8Ptr);

    /********************************************************************************
     * 6 - Extra operations required before writing the main code (Optional)
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

    auto tile_cout0 = writer->declare_tile("cout0", ckw::TileInfo(ckw::DataType::Int32)); // OFM
    auto tile_mout0 = writer->declare_tile("mout0", ckw::TileInfo(ckw::DataType::Int32)); // WIDTH
    auto tile_mout1 = writer->declare_tile("mout1", ckw::TileInfo(ckw::DataType::Int32)); // HEIGHT
    auto tile_bout0 = writer->declare_tile("bout0", ckw::TileInfo(ckw::DataType::Int32)); // BATCH SIZE IDX

    // Calculate coordinates
    get_coordinate_from_gws_overlapping_min(writer, tile_cout0, tile_gid_0, const_dst_n0_i32,
                                            const_shift_back_dst_n0_i32, const_0_i32);
    get_coordinate_from_gws(writer, tile_mout0, tile_gid_1, const_dst_m0_i32);
    writer->op_binary(tile_mout1, ckw::BinaryOp::Mod, tile_gid_2, const_dst_h_i32);
    writer->op_binary(tile_bout0, ckw::BinaryOp::Div, tile_gid_2, const_dst_h_i32);

    auto tile_src_ci = writer->declare_tile("src_ci", ckw::DataType::Int32);
    writer->op_binary(tile_src_ci, ckw::BinaryOp::Div, tile_cout0, const_depth_multiplier_i32);

    auto tile_src_xi = writer->declare_tile("src_xi", ckw::DataType::Int32);
    writer->op_binary(tile_src_xi, ckw::BinaryOp::Mul, tile_mout0, const_stride_x_i32);
    writer->op_binary(tile_src_xi, ckw::BinaryOp::Sub, tile_src_xi, const_pad_x_i32);

    auto tile_src_yi = writer->declare_tile("src_yi", ckw::DataType::Int32);
    writer->op_binary(tile_src_yi, ckw::BinaryOp::Mul, tile_mout1, const_stride_y_i32);
    writer->op_binary(tile_src_yi, ckw::BinaryOp::Sub, tile_src_yi, const_pad_y_i32);

    // Loop variables
    auto tile_yk = writer->declare_tile("yk", ckw::DataType::Int32);

    writer->op_assign(tile_yk, const_0_i32);

    // clang-format off
    writer->op_for_loop(tile_yk, ckw::BinaryOp::Less, const_kernel_size_i32, tile_yk, ckw::AssignmentOp::Increment, const_kernel_w_i32,
    [&]()
    {
        auto tile_src = writer->declare_tile("a", ckw::TileInfo(to_ckw(_src->data_type()), src_m0, src_n0));
        auto tile_wei = writer->declare_tile("b", ckw::TileInfo(to_ckw(_wei->data_type()), wei_m0, wei_n0));

        writer->op_assign(tile_src, const_0_fp);

        auto tile_x_gte_0 = writer->declare_tile("x_gte_0", ckw::TileInfo(ckw::DataType::Int32));
        auto tile_y_gte_0 = writer->declare_tile("y_gte_0", ckw::TileInfo(ckw::DataType::Int32));
        auto tile_x_lt_w  = writer->declare_tile("x_lt_w", ckw::TileInfo(ckw::DataType::Int32));
        auto tile_y_lt_h  = writer->declare_tile("y_lt_h", ckw::TileInfo(ckw::DataType::Int32));

        // Check if yi + yk * DILATION_Y is out-of-bound
        writer->op_binary(tile_y_gte_0, ckw::BinaryOp::GreaterEqual, tile_src_yi, const_0_i32);
        writer->op_binary(tile_y_lt_h, ckw::BinaryOp::Less, tile_src_yi, const_src_h_i32);

        auto tile_src_mi = writer->declare_tile("src_mi", ckw::TileInfo(ckw::DataType::Int32));

        // Load src
        for(int32_t xk = 0; xk < src_m0; ++xk)
        {
            auto const_xk_i32 = writer->declare_constant_tile(ckw::ConstantData({{xk}}, ckw::DataType::Int32));

            // xi + xk * DILATION_X
            writer->op_binary(tile_src_mi, ckw::BinaryOp::Mul, const_xk_i32, const_dilation_x_i32);
            writer->op_binary(tile_src_mi, ckw::BinaryOp::Add, tile_src_mi, tile_src_xi);

            // Check if xi + xk * DILATION_X is out-of-bound
            writer->op_binary(tile_x_gte_0, ckw::BinaryOp::GreaterEqual, tile_src_mi, const_0_i32);
            writer->op_binary(tile_x_lt_w, ckw::BinaryOp::Less, tile_src_mi, const_src_w_i32);

            // Set mi to -1 if we have out-of-bound memory accesses
            writer->op_ternary(tile_src_mi, ckw::TernaryOp::Select, const_neg_1_i32, tile_src_mi, tile_x_gte_0);
            writer->op_ternary(tile_src_mi, ckw::TernaryOp::Select, const_neg_1_i32, tile_src_mi, tile_x_lt_w);
            writer->op_ternary(tile_src_mi, ckw::TernaryOp::Select, const_neg_1_i32, tile_src_mi, tile_y_gte_0);
            writer->op_ternary(tile_src_mi, ckw::TernaryOp::Select, const_neg_1_i32, tile_src_mi, tile_y_lt_h);

            writer->op_load(tile_src.row(xk), src->tensor(), sampler_src, tile_src_ci, tile_src_mi, tile_src_yi, tile_bout0);
        }

        // Load wei
        writer->op_load(tile_wei, wei->tensor(), sampler_wei, tile_cout0, tile_yk, const_0_i32, const_0_i32);

        // Attention: MAC (Multiply-and-Accumulate) ternary operator is currently unsupported in CKW
        // Therefore, this part should be replaced with the MAC ternary operator when availabe
        auto tile_tmp = writer->declare_tile("tmp", ckw::TileInfo(to_ckw(_src->data_type()), 1, dst_n0));
        for(int32_t m0 = 0; m0 < dst_m0; ++m0)
        {
            for(int32_t xk = 0; xk < kernel_width; ++xk)
            {
                auto tile_a = tile_src.row(m0 + xk);
                auto tile_b = tile_wei.row(xk);
                auto tile_c = tile_dst.row(m0);

                writer->op_binary(tile_tmp, ckw::BinaryOp::Mul, tile_a, tile_b);
                writer->op_binary(tile_c, ckw::BinaryOp::Add, tile_c, tile_tmp);
            }
        }
        writer->op_binary(tile_src_yi, ckw::BinaryOp::Add, tile_src_yi, const_dilation_y_i32);
    });
    // clang-format on

    // Bias addition
    // NOTE: This operation will be removed from this kernel as the interface is standardized. The intended way of
    // performing bias addition is to fuse this convolution kernel with a following elementwise addition kernel.
    if (using_bias)
    {
        if (!bia->has_tile())
        {
            auto tile_bia = writer->declare_tile("bia", ckw::TileInfo(to_ckw(_src->data_type()), 1, dst_n0));
            writer->op_load(tile_bia, bia->tensor(), sampler_bia, tile_cout0, const_0_i32, const_0_i32, const_0_i32);
            bia->init_virtual_tensor(tile_bia, sampler_bia);
        }
        auto &tile_bia = bia->tile();

        writer->op_binary(tile_dst, ckw::BinaryOp::Add, tile_dst, tile_bia);
    }

    ARM_COMPUTE_ERROR_ON_MSG(dst->has_tile() == false, "You must bind a tile before appending another component");
}

Window GpuCkwDepthwiseConv2d::get_window() const
{
    ARM_COMPUTE_ERROR_ON_MSG(_dst->tensor_shape().total_size() == 0U, "Destination tensor is not initialized");
    TensorShape output_shape = _dst->tensor_shape();

    Window win = calculate_max_window(output_shape, Steps(_settings.n0(), _settings.m0()));
    return win.collapse(win, Window::DimZ);
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
