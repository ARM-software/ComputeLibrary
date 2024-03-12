/*
 * Copyright (c) 2023 Arm Limited.
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
#include "arm_compute/core/utils/StringUtils.h"
#include "arm_compute/core/Validate.h"
#include "ckw/TensorTileSampler.h"
#include "ckw/TileInfo.h"

#include "src/core/helpers/WindowHelpers.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/type_converter/Common.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/WriterHelper.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwKernelWriter.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwScopedKernelWriter.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwVariableTable.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelArgument.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGroup.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{

using TileContainer = std::vector<std::vector<std::string>>;

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

    const unsigned int channel_idx = get_data_layout_dimension_index(_src->data_layout(), DataLayoutDimension::CHANNEL);
    const unsigned int width_idx   = get_data_layout_dimension_index(_wei->data_layout(), DataLayoutDimension::WIDTH);
    const unsigned int height_idx  = get_data_layout_dimension_index(_wei->data_layout(), DataLayoutDimension::HEIGHT);

    const auto root_window = comp_group.get_root_component()->ckw_component_driver()->get_window();

    // Tunable parameters
    const int32_t m0         = root_window.y().step();
    const int32_t n0         = root_window.x().step();
    const int32_t k0         = adjust_vec_size(_settings.direct_conv_descriptor().k0, _src->dimension(channel_idx));
    const int32_t partial_n0 = _dst->dimension(0) % n0;

    const int32_t K = _src->dimension(channel_idx);

    // Exporting the weights tensor to an OpenCL image object is currently only supported when:
    //   a) k0 is equal to 4
    // The current implementation expects to read a vector of 4 float values into the OpenCL image object.
    //   b) K is a multiple of 4
    // This is a limitation in the current interface due to the variable table being responsible for maintaining
    // information about the TensorStorageType rather than the TensorTileSampler. As a result, TensorStorageType cannot
    // be reassigned, and we cannot use a texture object for the weights tensor in cases where we expect to have an
    // extra loop to compute the left-over elements.
    const bool use_cl_image_for_weights = desc.export_weights_to_cl_image && (k0 == 4) && (K % 4 == 0);

    GpuCkwComponentArgument *src =
        vtable.declare_variable(comp_group, writer, _src, TensorStorageType::ClBufferUint8Ptr, "src");
    GpuCkwComponentArgument *wei = vtable.declare_variable(
        comp_group, writer, _wei,
        use_cl_image_for_weights ? TensorStorageType::ClImage2dReadOnly : TensorStorageType::ClBufferUint8Ptr, "wei");
    GpuCkwComponentArgument *dst =
        vtable.declare_variable(comp_group, writer, _dst, TensorStorageType::ClBufferUint8Ptr, "dst");
    GpuCkwComponentArgument *bia = nullptr;

    const bool using_bias = _bia != nullptr;

    if (using_bias)
    {
        bia = vtable.declare_variable(comp_group, writer, _bia, TensorStorageType::ClBufferUint8Ptr, "bia");
    }

    // Constants
    const auto kernel_height    = static_cast<int32_t>(_wei->dimension(height_idx));
    const auto kernel_width     = static_cast<int32_t>(_wei->dimension(width_idx));
    const auto src_channels     = static_cast<int32_t>(_src->dimension(channel_idx));
    auto      &tile_kernel_w    = writer->declare_tile("kernel_w", kernel_width);
    auto      &tile_kernel_size = writer->declare_tile("kernel_size", kernel_width * kernel_height);
    auto      &tile_src_c       = writer->declare_tile("src_c", static_cast<int32_t>(_src->dimension(channel_idx)));
    auto      &tile_dst_w       = writer->declare_tile("dst_w", static_cast<int32_t>(_dst->dimension(width_idx)));
    auto      &tile_stride_x    = writer->declare_tile("stride_x", static_cast<int32_t>(_attributes.stride().x()));
    auto      &tile_stride_y    = writer->declare_tile("stride_y", static_cast<int32_t>(_attributes.stride().y()));
    auto      &tile_pad_x       = writer->declare_tile("pad_x", static_cast<int32_t>(_attributes.pad().left));
    auto      &tile_pad_y       = writer->declare_tile("pad_y", static_cast<int32_t>(_attributes.pad().top));
    auto      &tile_k0          = writer->declare_tile("k0", k0);
    auto      &tile_0           = writer->declare_tile("0", 0);
    auto      &tile_1           = writer->declare_tile("1", 1);

    auto &tile_gid_0 = writer->declare_tile("gid_0", ckw::DataType::Int32);
    auto &tile_gid_1 = writer->declare_tile("gid_1", ckw::DataType::Int32);
    auto &tile_gid_2 = writer->declare_tile("gid_2", ckw::DataType::Int32);

    writer->op_get_global_id(tile_gid_0, 0);
    writer->op_get_global_id(tile_gid_1, 1);
    writer->op_get_global_id(tile_gid_2, 2);

    auto &tile_cout = writer->declare_tile("cout", ckw::DataType::Int32); // OFM
    auto &tile_mout = writer->declare_tile("mout", ckw::DataType::Int32); // WIDTH x HEIGHT
    auto &tile_bout = writer->declare_tile("bout", ckw::DataType::Int32); // BATCH SIZE IDX

    // Get the boundary aware coordinates at each global dimension index
    get_coord(writer, tile_cout, tile_gid_0, n0, partial_n0, tile_cout.name() + "_dim0_", tile_0);
    get_coord(writer, tile_mout, tile_gid_1, m0, 0, tile_mout.name() + "_dim1_", tile_0);
    get_coord(writer, tile_bout, tile_gid_2, 1, 0, tile_bout.name() + "_dim2_", tile_0);

    TensorTileSampler src_sampler;
    src_sampler.width(k0);
    src_sampler.height(m0);
    src_sampler.format(TensorSamplerFormat::C_WH_1);
    // We cannot have out-of-bounds reads in the X dimension (mapped to the IFMs) as we have an extra loop to
    // compute left-over elements
    src_sampler.address_mode_x(TensorSamplerAddressModeX::None);
    // We cannot have out-of-bounds reads when the kernel height is equal to 1. Otherwise, we need to ensure the
    // indirection buffer mi does not contain negative values representing out-of-bounds reads.
    src_sampler.address_mode_y(kernel_height == 1 ? TensorSamplerAddressModeY::None
                                                  : TensorSamplerAddressModeY::SkipMinEdgeOnly);
    src_sampler.address_mode_z(TensorSamplerAddressModeZ::None);

    TensorTileSampler wei_sampler;
    wei_sampler.width(k0);
    wei_sampler.height(n0);
    wei_sampler.format(TensorSamplerFormat::C_WH_1);
    // We cannot have out-of-bounds accesses for the weights
    wei_sampler.address_mode_x(TensorSamplerAddressModeX::None);
    wei_sampler.address_mode_y(TensorSamplerAddressModeY::None);
    wei_sampler.address_mode_z(TensorSamplerAddressModeZ::None);

    TensorTileSampler dst_sampler;
    dst_sampler.width(n0);
    dst_sampler.height(m0);
    dst_sampler.format(TensorSamplerFormat::C_WH_1);
    dst_sampler.address_mode_x(TensorSamplerAddressModeX::OverlappingMin);
    dst_sampler.address_mode_y(TensorSamplerAddressModeY::ClampToMaxEdgeOnly);
    dst_sampler.address_mode_z(TensorSamplerAddressModeZ::None);
    dst_sampler.x(tile_cout);
    dst_sampler.y(tile_mout);
    dst_sampler.z(tile_0);
    dst_sampler.b(tile_bout);

    if (!dst->has_tile())
    {
        auto &tile = writer->declare_tile("dst", TileInfo(to_ckw(_dst->data_type()), m0, n0));
        dst->init_virtual_tensor(tile, dst_sampler);
    }
    auto &tile_dst = dst->tile();

    writer->op_assign(tile_dst, tile_0);

    // We create a 2d container of size (M0, 1) to store the indices for iteration
    TileContainer it;
    for (int m = 0; m < m0; ++m)
    {
        std::vector<std::string> idx{std::to_string(m)};
        it.push_back({idx});
    }
    const auto &tile_it = writer->declare_tile("it", it, ckw::DataType::Int32);

    auto &tile_xi = writer->declare_tile("xi", TileInfo(ckw::DataType::Int32, m0, 1));
    auto &tile_yi = writer->declare_tile("yi", TileInfo(ckw::DataType::Int32, m0, 1));

    // Convert the linear index to coordinate
    // xi = ((mout + i) % dst_w) * stride_x - pad_x
    // yi = ((mout + i) / dst_w) * stride_y - pad_y
    writer->op_binary_expression(tile_xi, tile_mout, BinaryOp::Add, tile_it);
    writer->op_binary_expression(tile_yi, tile_mout, BinaryOp::Add, tile_it);
    writer->op_binary_expression(tile_xi, tile_xi, BinaryOp::Mod, tile_dst_w);
    writer->op_binary_expression(tile_yi, tile_yi, BinaryOp::Div, tile_dst_w);
    writer->op_binary_expression(tile_xi, tile_xi, BinaryOp::Mul, tile_stride_x);
    writer->op_binary_expression(tile_yi, tile_yi, BinaryOp::Mul, tile_stride_y);
    writer->op_binary_expression(tile_xi, tile_xi, BinaryOp::Sub, tile_pad_x);
    writer->op_binary_expression(tile_yi, tile_yi, BinaryOp::Sub, tile_pad_y);

    auto &tile_y_b = writer->declare_tile("y_b", ckw::DataType::Int32);
    writer->op_binary_expression(tile_y_b, tile_cout, BinaryOp::Mul, tile_kernel_size);

    auto &tile_i = writer->declare_tile("i", ckw::DataType::Int32);
    writer->op_assign(tile_i, tile_0);

    // clang-format off
    writer->op_for_loop(tile_i, BinaryOp::Less, tile_kernel_size, tile_i, AssignmentOp::Increment, tile_1, [&]()
    {
        auto &tile_x_k = writer->declare_tile("x_k", ckw::DataType::Int32);
        auto &tile_y_k = writer->declare_tile("y_k", ckw::DataType::Int32);

        writer->op_binary_expression(tile_x_k, tile_i, BinaryOp::Mod, tile_kernel_w);
        writer->op_binary_expression(tile_y_k, tile_i, BinaryOp::Div, tile_kernel_w);

        auto &tile_ck = writer->declare_tile("ck", ckw::DataType::Int32);
        writer->op_assign(tile_ck, tile_0);

        auto &tile_mi = writer->declare_tile("mi", TileInfo(ckw::DataType::Int32, m0, 1));
        // Construct an indirection buffer containing the precalculated addresses of elements in the source tensor
        // x_s = xi + x_k
        // y_s = yi + y_k
        // mi = x_s + y_s * width;
        // mi = select(-1, mi, x_s >= 0);
        // mi = select(-1, mi, x_s < width);
        // mi = select(-1, mi, y_s >= 0);
        // mi = select(-1, mi, y_s < height);
        writer->util_get_indirect_buffer(tile_mi, src->tensor(), src_sampler, tile_xi, tile_yi, tile_x_k, tile_y_k);

        src_sampler.x(tile_ck);
        src_sampler.y(tile_mi);
        src_sampler.z(tile_0);
        src_sampler.b(tile_bout);

        wei_sampler.x(tile_ck);
        wei_sampler.y(tile_y_b);
        wei_sampler.z(tile_0);
        wei_sampler.b(tile_0);

        auto &tile_src_c_minus_k0 = writer->declare_tile("src_c_minus_k0", src_channels - k0);

        writer->op_for_loop(tile_ck, BinaryOp::LessEqual, tile_src_c_minus_k0, tile_ck, AssignmentOp::Increment, tile_k0, [&]()
        {
            auto &tile_lhs = writer->declare_tile("lhs", TileInfo(to_ckw(_src->data_type()), m0, k0));
            auto &tile_rhs = writer->declare_tile("rhs", TileInfo(to_ckw(_wei->data_type()), n0, k0));
            writer->op_assign(tile_lhs, tile_0);
            writer->op_assign(tile_rhs, tile_0);

            writer->op_load_indirect(tile_lhs, src->tensor(), src_sampler);
            writer->op_load(tile_rhs, wei->tensor(), wei_sampler, tile_kernel_size);

            writer->op_binary_expression(tile_dst, tile_lhs, BinaryOp::MatMul_Nt_T, tile_rhs);
        });

        // Left-over accumulations for when K is not a multiple of k0
        if(!(K % k0 == 0))
        {
            writer->op_for_loop(tile_ck, BinaryOp::Less, tile_src_c, tile_ck, AssignmentOp::Increment, tile_1, [&]()
            {
                auto &tile_lhs = writer->declare_tile("lhs_leftover", TileInfo(to_ckw(_src->data_type()), m0, 1));
                auto &tile_rhs = writer->declare_tile("rhs_leftover", TileInfo(to_ckw(_wei->data_type()), n0, 1));
                writer->op_assign(tile_lhs, tile_0);
                writer->op_assign(tile_rhs, tile_0);

                writer->op_load_indirect(tile_lhs, src->tensor(), src_sampler);
                writer->op_load(tile_rhs, wei->tensor(), wei_sampler, tile_kernel_size);

                writer->op_binary_expression(tile_dst, tile_lhs, BinaryOp::MatMul_Nt_T, tile_rhs);
            });
        }

    writer->op_binary_expression(tile_y_b, tile_y_b, BinaryOp::Add, tile_1);
    });
    // clang-format on

    // Bias addition
    // NOTE: This operation will be removed from this kernel as the interface is standardized. The intended way of
    // performing bias addition is to fuse this convolution kernel with a following elementwise addition kernel.
    if (using_bias)
    {
        if (!bia->has_tile())
        {
            // Reuse the destination sampler for the bias
            writer->op_load_once(bia, dst_sampler);
        }
        auto &tile_bia = bia->tile();

        writer->op_binary_expression(tile_dst, tile_dst, BinaryOp::Add, tile_bia);
    }
}

Window GpuCkwDirectConv2d::get_window() const
{
    ARM_COMPUTE_ERROR_ON_MSG(_dst->tensor_shape().total_size() == 0U, "Destination tensor is not initialized");

    const auto dst_shape = _dst->tensor_shape();
    const auto desc      = _settings.direct_conv_descriptor();

    const unsigned int n0 = adjust_vec_size(desc.n0, dst_shape[0]);
    const unsigned int m0 = adjust_vec_size(desc.m0, dst_shape[1] * dst_shape[2]);

    Window win = calculate_max_window(dst_shape, Steps(n0, m0));

    const size_t dim_y_collapsed = ceil_to_multiple(dst_shape[1] * dst_shape[2], m0);
    win.set(Window::DimY, Window::Dimension(0, dim_y_collapsed, m0));
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
