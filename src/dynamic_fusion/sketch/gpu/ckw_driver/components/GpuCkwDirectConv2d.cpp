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

#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/GpuCkwDirectConv2d.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/Validate.h"

#include "src/core/helpers/WindowHelpers.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/CkwHelper.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/type_converter/Common.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwComponentArgument.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwVariableTable.h"

#include "compute_kernel_writer/include/ckw/KernelWriter.h"
#include <cstdint>
#include <string>
#include <vector>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{

using TileContainer = std::vector<std::vector<int32_t>>;

GpuCkwDirectConv2d::GpuCkwDirectConv2d(ComponentId                      id,
                                       const ArgumentPack<ITensorInfo> &tensors,
                                       const Attributes                &attributes,
                                       const Settings                  &settings)
    : IGpuCkwComponentDriver{id, tensors}, _src{}, _wei{}, _bia{}, _dst{}, _attributes{attributes}, _settings{settings}
{
    _src = this->tensors().get_const_tensor(TensorType::ACL_SRC_0);
    _wei = this->tensors().get_const_tensor(TensorType::ACL_SRC_1);
    _bia = this->tensors().get_const_tensor(TensorType::ACL_SRC_2);
    _dst = this->tensors().get_const_tensor(TensorType::ACL_DST_0);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _wei, _dst); // Bias can be null
}

void GpuCkwDirectConv2d::write_component_code(const ComponentGroup    &comp_group,
                                              GpuCkwVariableTable     &vtable,
                                              GpuCkwScopedKernelWriter writer) const
{
    const auto desc = _settings.direct_conv_descriptor();
    ARM_COMPUTE_ERROR_ON_MSG(desc.export_input_to_cl_image || desc.export_output_to_cl_image,
                             "Only the weights tensor can be exported to cl_image");

    const uint32_t channel_idx = get_data_layout_dimension_index(_src->data_layout(), DataLayoutDimension::CHANNEL);
    const uint32_t width_idx   = get_data_layout_dimension_index(_wei->data_layout(), DataLayoutDimension::WIDTH);
    const uint32_t height_idx  = get_data_layout_dimension_index(_wei->data_layout(), DataLayoutDimension::HEIGHT);

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
    const auto dst_dt        = to_ckw(_dst->data_type());
    const auto kernel_height = static_cast<int32_t>(_wei->dimension(height_idx));
    const auto kernel_width  = static_cast<int32_t>(_wei->dimension(width_idx));
    const auto src_c         = static_cast<int32_t>(_src->dimension(channel_idx));
    const auto src_w         = static_cast<int32_t>(_src->dimension(width_idx));
    const auto src_h         = static_cast<int32_t>(_src->dimension(height_idx));
    const auto dst_w         = static_cast<int32_t>(_dst->dimension(width_idx));
    const auto stride_x      = static_cast<int32_t>(_attributes.stride().x());
    const auto stride_y      = static_cast<int32_t>(_attributes.stride().y());
    const auto pad_x         = static_cast<int32_t>(_attributes.pad().left);
    const auto pad_y         = static_cast<int32_t>(_attributes.pad().top);
    const auto kernel_size   = kernel_width * kernel_height;
    const auto k0 =
        static_cast<int32_t>(adjust_vec_size(_settings.direct_conv_descriptor().k0, _src->dimension(channel_idx)));

    // CKW constants
    auto const_kernel_w_i32 = writer->declare_constant_tile(ckw::ConstantData({{kernel_width}}, ckw::DataType::Int32));
    auto const_kernel_size_i32 =
        writer->declare_constant_tile(ckw::ConstantData({{kernel_size}}, ckw::DataType::Int32));
    auto const_src_c_i32    = writer->declare_constant_tile(ckw::ConstantData({{src_c}}, ckw::DataType::Int32));
    auto const_src_w_i32    = writer->declare_constant_tile(ckw::ConstantData({{src_w}}, ckw::DataType::Int32));
    auto const_src_h_i32    = writer->declare_constant_tile(ckw::ConstantData({{src_h}}, ckw::DataType::Int32));
    auto const_dst_w_i32    = writer->declare_constant_tile(ckw::ConstantData({{dst_w}}, ckw::DataType::Int32));
    auto const_stride_x_i32 = writer->declare_constant_tile(ckw::ConstantData({{stride_x}}, ckw::DataType::Int32));
    auto const_stride_y_i32 = writer->declare_constant_tile(ckw::ConstantData({{stride_y}}, ckw::DataType::Int32));
    auto const_pad_x_i32    = writer->declare_constant_tile(ckw::ConstantData({{pad_x}}, ckw::DataType::Int32));
    auto const_pad_y_i32    = writer->declare_constant_tile(ckw::ConstantData({{pad_y}}, ckw::DataType::Int32));
    auto const_k0_i32       = writer->declare_constant_tile(ckw::ConstantData({{k0}}, ckw::DataType::Int32));
    auto const_0_i32        = writer->declare_constant_tile(ckw::ConstantData({{0}}, ckw::DataType::Int32));
    auto const_pos_1_i32    = writer->declare_constant_tile(ckw::ConstantData({{1}}, ckw::DataType::Int32));
    auto const_neg_1_i32    = writer->declare_constant_tile(ckw::ConstantData({{-1}}, ckw::DataType::Int32));
    auto const_0_fp         = writer->declare_constant_tile(ckw::ConstantData({{0.0f}}, dst_dt));
    auto const_src_c_i32_minus_k0_i32 =
        writer->declare_constant_tile(ckw::ConstantData({{src_c - k0}}, ckw::DataType::Int32));

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
    const int32_t dst_m0_partial = (_dst->dimension(1) * _dst->dimension(2)) % dst_m0;

    // Shift-back for the overlapping-min strategy
    const int32_t dst_shift_back = (dst_n0 - dst_n0_partial) % dst_n0;

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
    // Exporting the weights tensor to an OpenCL image object is currently only supported when:
    //   a) k0 is equal to 4
    // The current implementation expects to read a vector of 4 float values into the OpenCL image object.
    //   b) K is a multiple of 4
    // This is a limitation in the current interface due to the variable table being responsible for maintaining
    // information about the TensorStorageType rather than the TensorTileSampler. As a result, TensorStorageType cannot
    // be reassigned, and we cannot use a texture object for the weights tensor in cases where we expect to have an
    // extra loop to compute the left-over elements.
    const bool use_cl_image_for_weights = desc.export_weights_to_cl_image && (k0 == 4) && (src_c % 4 == 0);

    // SOURCE SAMPLER
    // - We cannot have out-of-bounds reads in the X dimension (mapped to the IFMs) as we have an extra loop to
    //   compute left-over elements
    // - We cannot have out-of-bounds reads when the kernel height is equal to 1. In all other cases, we need to ensure the
    //   indirection buffer mi does not contain negative values representing out-of-bounds reads.
    auto address_mode_y_src =
        kernel_height == 1 ? ckw::TensorSamplerAddressModeY::None : ckw::TensorSamplerAddressModeY::SkipLessThanZero;
    ckw::TensorSampler sampler_src;
    sampler_src.format(ckw::TensorSamplerFormat::Dim0_Dim1xDim2_1); // 3rd dimension collapsed with 2nd dimension
    sampler_src.address_mode_x(ckw::TensorSamplerAddressModeX::None);
    sampler_src.address_mode_y(address_mode_y_src);
    sampler_src.address_mode_z(ckw::TensorSamplerAddressModeZ::None);
    sampler_src.storage(ckw::TensorStorageType::BufferUint8Ptr);

    // WEIGHTS SAMPLER
    // We cannot have out-of-bounds accesses for the weights
    ckw::TensorSampler sampler_wei;
    sampler_wei.format(ckw::TensorSamplerFormat::Dim0_Dim1xDim2_1); // 3rd dimension collapsed with 2nd dimension
    sampler_wei.address_mode_x(ckw::TensorSamplerAddressModeX::None);
    sampler_wei.address_mode_y(ckw::TensorSamplerAddressModeY::None);
    sampler_wei.address_mode_z(ckw::TensorSamplerAddressModeZ::None);
    if (use_cl_image_for_weights)
    {
        sampler_wei.storage(ckw::TensorStorageType::Texture2dReadOnly);
    }
    else
    {
        sampler_wei.storage(ckw::TensorStorageType::BufferUint8Ptr);
    }

    // BIAS SAMPLER
    ckw::TensorSampler sampler_bia;

    if (using_bias)
    {
        sampler_bia.format(ckw::TensorSamplerFormat::Dim0_Dim1xDim2_1);
        sampler_bia.address_mode_x(sampler_dst.address_mode_x());
        sampler_bia.address_mode_y(ckw::TensorSamplerAddressModeY::None);
        sampler_bia.address_mode_z(ckw::TensorSamplerAddressModeZ::None);
        sampler_bia.storage(ckw::TensorStorageType::BufferUint8Ptr);
    }

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

    auto tile_cout = writer->declare_tile("cout", ckw::TileInfo(ckw::DataType::Int32)); // OFM
    auto tile_mout = writer->declare_tile("mout", ckw::TileInfo(ckw::DataType::Int32)); // WIDTH x HEIGHT
    auto tile_bout = writer->declare_tile("bout", ckw::TileInfo(ckw::DataType::Int32)); // BATCH SIZE IDX

    // Calculate coordinates
    get_coordinate_from_gws_overlapping_min(writer, tile_cout, tile_gid_0, const_dst_n0_i32,
                                            const_shift_back_dst_n0_i32, const_0_i32);
    get_coordinate_from_gws(writer, tile_mout, tile_gid_1, const_dst_m0_i32);
    get_coordinate_from_gws(writer, tile_bout, tile_gid_2, const_pos_1_i32);

    /********************************************************************************
     * 8 - Write the rest of the code
     ********************************************************************************/
    // We create a 2d container of size (dst_m0, 1) to store the indices for iteration
    TileContainer it;
    for (int32_t m = 0; m < dst_m0; ++m)
    {
        std::vector<int32_t> idx{m};
        it.push_back({idx});
    }

    const auto &const_idxs = writer->declare_constant_tile(ckw::ConstantData(it, ckw::DataType::Int32));

    auto tile_xi = writer->declare_tile("xi", ckw::TileInfo(ckw::DataType::Int32, dst_m0, 1));
    auto tile_yi = writer->declare_tile("yi", ckw::TileInfo(ckw::DataType::Int32, dst_m0, 1));

    // Convert the linear index to coordinate
    // xi = ((mout + i) % dst_w) * stride_x - pad_x
    // yi = ((mout + i) / dst_w) * stride_y - pad_y
    writer->op_binary(tile_xi, ckw::BinaryOp::Add, tile_mout, const_idxs);
    writer->op_binary(tile_yi, ckw::BinaryOp::Add, tile_mout, const_idxs);
    writer->op_binary(tile_xi, ckw::BinaryOp::Mod, tile_xi, const_dst_w_i32);
    writer->op_binary(tile_yi, ckw::BinaryOp::Div, tile_yi, const_dst_w_i32);
    writer->op_binary(tile_xi, ckw::BinaryOp::Mul, tile_xi, const_stride_x_i32);
    writer->op_binary(tile_yi, ckw::BinaryOp::Mul, tile_yi, const_stride_y_i32);
    writer->op_binary(tile_xi, ckw::BinaryOp::Sub, tile_xi, const_pad_x_i32);
    writer->op_binary(tile_yi, ckw::BinaryOp::Sub, tile_yi, const_pad_y_i32);

    auto tile_y_b = writer->declare_tile("y_b", ckw::TileInfo(ckw::DataType::Int32));
    writer->op_binary(tile_y_b, ckw::BinaryOp::Mul, tile_cout, const_kernel_size_i32);

    auto tile_i = writer->declare_tile("i", ckw::TileInfo(ckw::DataType::Int32));
    writer->op_assign(tile_i, const_0_i32);

    // clang-format off
    writer->op_for_loop(tile_i, ckw::BinaryOp::Less, const_kernel_size_i32, tile_i, ckw::AssignmentOp::Increment, const_pos_1_i32, [&]()
    {
        auto tile_x_k = writer->declare_tile("x_k", ckw::TileInfo(ckw::DataType::Int32));
        auto tile_y_k = writer->declare_tile("y_k", ckw::TileInfo(ckw::DataType::Int32));

        writer->op_binary(tile_x_k, ckw::BinaryOp::Mod, tile_i, const_kernel_w_i32);
        writer->op_binary(tile_y_k, ckw::BinaryOp::Div, tile_i, const_kernel_w_i32);

        auto tile_ck = writer->declare_tile("ck", ckw::TileInfo(ckw::DataType::Int32));
        writer->op_assign(tile_ck, const_0_i32);

        // Construct an indirection buffer containing the precalculated addresses of elements in the source tensor
        // x_s = xi + x_k
        // y_s = yi + y_k
        // mi = x_s + y_s * width;
        // mi = select(-1, mi, x_s >= 0);
        // mi = select(-1, mi, x_s < width);
        // mi = select(-1, mi, y_s >= 0);
        // mi = select(-1, mi, y_s < height);
        auto tile_xs = writer->declare_tile("xs", ckw::TileInfo(ckw::DataType::Int32, dst_m0, 1));
        auto tile_ys = writer->declare_tile("ys", ckw::TileInfo(ckw::DataType::Int32, dst_m0, 1));
        auto tile_mi = writer->declare_tile("mi", ckw::TileInfo(ckw::DataType::Int32, dst_m0, 1));

        auto tile_xs_gte_0 = writer->declare_tile("xs_gte_0", ckw::TileInfo(ckw::DataType::Int32, dst_m0, 1));
        auto tile_ys_gte_0 = writer->declare_tile("ys_gte_0", ckw::TileInfo(ckw::DataType::Int32, dst_m0, 1));
        auto tile_xs_lt_w  = writer->declare_tile("xs_lt_w", ckw::TileInfo(ckw::DataType::Int32, dst_m0, 1));
        auto tile_ys_lt_h  = writer->declare_tile("ys_lt_h", ckw::TileInfo(ckw::DataType::Int32, dst_m0, 1));

        writer->op_binary(tile_xs, ckw::BinaryOp::Add, tile_xi, tile_x_k);
        writer->op_binary(tile_ys, ckw::BinaryOp::Add, tile_yi, tile_y_k);
        writer->op_binary(tile_mi, ckw::BinaryOp::Mul, tile_ys, const_src_w_i32);
        writer->op_binary(tile_mi, ckw::BinaryOp::Add, tile_mi, tile_xs);
        writer->op_binary(tile_xs_gte_0, ckw::BinaryOp::GreaterEqual, tile_xs, const_0_i32);
        writer->op_binary(tile_ys_gte_0, ckw::BinaryOp::GreaterEqual, tile_ys, const_0_i32);
        writer->op_binary(tile_xs_lt_w, ckw::BinaryOp::Less, tile_xs, const_src_w_i32);
        writer->op_binary(tile_ys_lt_h, ckw::BinaryOp::Less, tile_ys, const_src_h_i32);
        writer->op_ternary(tile_mi, ckw::TernaryOp::Select, const_neg_1_i32, tile_mi, tile_xs_gte_0);
        writer->op_ternary(tile_mi, ckw::TernaryOp::Select, const_neg_1_i32, tile_mi, tile_ys_gte_0);
        writer->op_ternary(tile_mi, ckw::TernaryOp::Select, const_neg_1_i32, tile_mi, tile_xs_lt_w);
        writer->op_ternary(tile_mi, ckw::TernaryOp::Select, const_neg_1_i32, tile_mi, tile_ys_lt_h);

        writer->op_for_loop(tile_ck, ckw::BinaryOp::LessEqual, const_src_c_i32_minus_k0_i32, tile_ck, ckw::AssignmentOp::Increment, const_k0_i32, [&]()
        {
            auto tile_lhs = writer->declare_tile("lhs", ckw::TileInfo(to_ckw(_src->data_type()), dst_m0, k0));
            auto tile_rhs = writer->declare_tile("rhs", ckw::TileInfo(to_ckw(_wei->data_type()), dst_n0, k0));
            writer->op_assign(tile_lhs, const_0_fp);
            writer->op_assign(tile_rhs, const_0_fp);

            writer->op_load_indirect(tile_lhs, src->tensor(), sampler_src, tile_ck, tile_mi, const_0_i32, tile_bout);
            writer->op_load_dilated(tile_rhs, wei->tensor(), sampler_wei, tile_ck, tile_y_b, const_0_i32, const_0_i32, const_pos_1_i32, const_kernel_size_i32);

            writer->op_binary(tile_dst, ckw::BinaryOp::MatMul_Nt_T, tile_lhs, tile_rhs);
        });

        // Left-over accumulations for when K is not a multiple of k0
        if(((src_c % k0) != 0))
        {
            writer->op_for_loop(tile_ck, ckw::BinaryOp::Less, const_src_c_i32, tile_ck, ckw::AssignmentOp::Increment, const_pos_1_i32, [&]()
            {
                auto tile_lhs = writer->declare_tile("lhs_leftover", ckw::TileInfo(to_ckw(_src->data_type()), dst_m0, 1));
                auto tile_rhs = writer->declare_tile("rhs_leftover", ckw::TileInfo(to_ckw(_wei->data_type()), dst_n0, 1));
                writer->op_assign(tile_lhs, const_0_fp);
                writer->op_assign(tile_rhs, const_0_fp);

                writer->op_load_indirect(tile_lhs, src->tensor(), sampler_src, tile_ck, tile_mi, const_0_i32, tile_bout);
                writer->op_load_dilated(tile_rhs, wei->tensor(), sampler_wei, tile_ck, tile_y_b, const_0_i32, const_0_i32, const_pos_1_i32, const_kernel_size_i32);

                writer->op_binary(tile_dst, ckw::BinaryOp::MatMul_Nt_T, tile_lhs, tile_rhs);
            });
        }

        writer->op_binary(tile_y_b, ckw::BinaryOp::Add, tile_y_b, const_pos_1_i32);
    });
    // clang-format on

    // NOTE: The bias addition will be removed from this kernel as the interface is standardized. The intended way of
    // performing bias addition is to fuse this convolution kernel with a following elementwise addition kernel.
    if (using_bias)
    {
        if (!bia->has_tile())
        {
            auto tile_bia = writer->declare_tile("bia", ckw::TileInfo(to_ckw(_src->data_type()), 1, dst_n0));
            writer->op_load(tile_bia, bia->tensor(), sampler_bia, tile_cout, const_0_i32, const_0_i32, const_0_i32);
            bia->init_virtual_tensor(tile_bia, sampler_bia);
        }
        auto &tile_bia = bia->tile();

        writer->op_binary(tile_dst, ckw::BinaryOp::Add, tile_dst, tile_bia);
    }

    ARM_COMPUTE_ERROR_ON_MSG(dst->has_tile() == false, "You must bind a tile before appending another component");
}

Window GpuCkwDirectConv2d::get_window() const
{
    ARM_COMPUTE_ERROR_ON_MSG(_dst->tensor_shape().total_size() == 0U, "Destination tensor is not initialized");

    const auto dst_shape = _dst->tensor_shape();
    const auto desc      = _settings.direct_conv_descriptor();

    const uint32_t dst_n0 = adjust_vec_size(desc.n0, dst_shape[0]);
    const uint32_t dst_m0 = adjust_vec_size(desc.m0, dst_shape[1] * dst_shape[2]);

    Window win = calculate_max_window(dst_shape, Steps(dst_n0, dst_m0));

    const size_t dim_y_collapsed = ceil_to_multiple(dst_shape[1] * dst_shape[2], dst_m0);
    win.set(Window::DimY, Window::Dimension(0, dim_y_collapsed, dst_m0));
    win.set(Window::DimZ, Window::Dimension(0, dst_shape.total_size_upper(3), 1));

    return win;
}

std::string GpuCkwDirectConv2d::get_name(const ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);

    return "direct_conv2d";
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
