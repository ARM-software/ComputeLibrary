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
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/GpuCkwPool2d.h"

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

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
GpuCkwPool2d::GpuCkwPool2d(ComponentId                      id,
                           const ArgumentPack<ITensorInfo> &tensors,
                           const Attributes                &attributes,
                           const Settings                  &settings)
    : IGpuCkwComponentDriver{id, tensors}, _src{}, _dst{}, _attributes{attributes}, _settings{settings}

{
    _src = this->tensors().get_const_tensor(TensorType::ACL_SRC_0);
    _dst = this->tensors().get_const_tensor(TensorType::ACL_DST_0);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);
}

void GpuCkwPool2d::write_component_code(const ComponentGroup    &comp_group,
                                        GpuCkwVariableTable     &vtable,
                                        GpuCkwScopedKernelWriter writer) const
{
    const uint32_t width_idx  = get_data_layout_dimension_index(_src->data_layout(), DataLayoutDimension::WIDTH);
    const uint32_t height_idx = get_data_layout_dimension_index(_src->data_layout(), DataLayoutDimension::HEIGHT);

    /********************************************************************************
     * 1 - Define tensors
     ********************************************************************************/
    GpuCkwComponentArgument *src = vtable.declare_variable(comp_group, writer, _src, "src");
    GpuCkwComponentArgument *dst = vtable.declare_variable(comp_group, writer, _dst, "dst");

    /********************************************************************************
     * 2 - Define CKW constants
     ********************************************************************************/
    const auto dst_dt    = to_ckw(_dst->data_type());
    const auto pool_sz_x = static_cast<int32_t>(_attributes.pool_size().x());
    const auto pool_sz_y = static_cast<int32_t>(_attributes.pool_size().y());
    const auto pad_x     = static_cast<int32_t>(_attributes.pad().left);
    const auto pad_y     = static_cast<int32_t>(_attributes.pad().top);
    const auto stride_x  = static_cast<int32_t>(_attributes.stride().x());
    const auto stride_y  = static_cast<int32_t>(_attributes.stride().y());
    const auto src_w     = static_cast<int32_t>(_src->dimension(width_idx));
    const auto src_h     = static_cast<int32_t>(_src->dimension(height_idx));
    const auto dst_h     = static_cast<int32_t>(_dst->dimension(height_idx));

    // CKW constants
    auto const_pool_sz_x_i32 = writer->declare_constant_tile(ckw::ConstantData({{pool_sz_x}}, ckw::DataType::Int32));
    auto const_pool_sz_y_i32 = writer->declare_constant_tile(ckw::ConstantData({{pool_sz_y}}, ckw::DataType::Int32));
    auto const_pad_x_i32     = writer->declare_constant_tile(ckw::ConstantData({{pad_x}}, ckw::DataType::Int32));
    auto const_pad_y_i32     = writer->declare_constant_tile(ckw::ConstantData({{pad_y}}, ckw::DataType::Int32));
    auto const_stride_x_i32  = writer->declare_constant_tile(ckw::ConstantData({{stride_x}}, ckw::DataType::Int32));
    auto const_stride_y_i32  = writer->declare_constant_tile(ckw::ConstantData({{stride_y}}, ckw::DataType::Int32));
    auto const_src_w_i32     = writer->declare_constant_tile(ckw::ConstantData({{src_w}}, ckw::DataType::Int32));
    auto const_src_h_i32     = writer->declare_constant_tile(ckw::ConstantData({{src_h}}, ckw::DataType::Int32));
    auto const_dst_h_i32     = writer->declare_constant_tile(ckw::ConstantData({{dst_h}}, ckw::DataType::Int32));
    auto const_0_i32         = writer->declare_constant_tile(ckw::ConstantData({{0}}, ckw::DataType::Int32));
    auto const_pos_1_i32     = writer->declare_constant_tile(ckw::ConstantData({{1}}, ckw::DataType::Int32));
    auto const_0_fp          = writer->declare_constant_tile(ckw::ConstantData({{0.0f}}, dst_dt));
    auto const_lowest_val_fp =
        writer->declare_constant_tile(ckw::ConstantData({{std::numeric_limits<float>::lowest()}}, ckw::DataType::Fp32));
    auto const_neg_inf_val_fp = writer->declare_constant_tile(ckw::ConstantData({{-1.0f / 0.0f}}, ckw::DataType::Fp32));

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
    // Check if it is global pooling
    const bool is_global_pooling = (pool_sz_x == src_w) && (pool_sz_y == src_h) && (pad_x == 0) && (pad_y == 0);

    // Accumulate always in F32 if the pool type is not MAX
    const bool acc_f32 = (dst_dt == ckw::DataType::Fp32) ||
                         ((dst_dt == ckw::DataType::Fp16) && _attributes.pool_type() != PoolingType::MAX);

    const auto acc_dt = acc_f32 ? ckw::DataType::Fp32 : ckw::DataType::Fp16;

    const bool is_wider_acc = dst_dt != acc_dt;

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

    /********************************************************************************
     * 8 - Write the rest of the code
     ********************************************************************************/
    // A tile used to temporarily store results or as an accumulator in case of AVG and L2 pooling.
    auto tile_res = writer->declare_tile("tile_res", ckw::TileInfo(acc_dt, dst_m0, dst_n0));

    // Initialise result tile with appropriate value
    if (_attributes.pool_type() == PoolingType::MAX)
    {
        if (_settings.use_inf_as_limit())
        {
            writer->op_cast(tile_res, const_neg_inf_val_fp, ckw::ConvertPolicy::None);
        }
        else
        {
            writer->op_cast(tile_res, const_lowest_val_fp, ckw::ConvertPolicy::None);
        }
    }
    else
    {
        writer->op_cast(tile_res, const_0_fp, ckw::ConvertPolicy::None);
    }

    // tile_idx_in_w = tile_mout0 * STRIDE_X - PAD_X
    auto tile_src_coord_x_start = writer->declare_tile("idx_in_w", ckw::DataType::Int32);
    writer->op_binary(tile_src_coord_x_start, ckw::BinaryOp::Mul, tile_mout0, const_stride_x_i32);
    writer->op_binary(tile_src_coord_x_start, ckw::BinaryOp::Sub, tile_src_coord_x_start, const_pad_x_i32);

    // tile_idx_in_h = tile_mout1 * STRIDE_Y - PAD_Y
    auto tile_src_coord_y_start = writer->declare_tile("idx_in_h", ckw::DataType::Int32);
    writer->op_binary(tile_src_coord_y_start, ckw::BinaryOp::Mul, tile_mout1, const_stride_y_i32);
    writer->op_binary(tile_src_coord_y_start, ckw::BinaryOp::Sub, tile_src_coord_y_start, const_pad_y_i32);

    auto tile_neg_src_coord_x_start = writer->declare_tile("neg_src_coord_x_start", ckw::DataType::Int32);
    auto tile_neg_src_coord_y_start = writer->declare_tile("neg_src_coord_y_start", ckw::DataType::Int32);

    writer->op_binary(tile_neg_src_coord_x_start, ckw::BinaryOp::Sub, const_0_i32, tile_src_coord_x_start);
    writer->op_binary(tile_neg_src_coord_y_start, ckw::BinaryOp::Sub, const_0_i32, tile_src_coord_y_start);

    // int pool_x_s = max((int)0, -idx_in_w);
    // int pool_x_e = min((int)POOL_SIZE_X, (int)SRC_WIDTH - idx_in_w);
    // int pool_y_s = max((int)0, -idx_in_h);
    // int pool_y_e = min((int)POOL_SIZE_Y, (int)SRC_HEIGHT - idx_in_h);
    auto tile_pool_x_s = writer->declare_tile("pool_x_s", ckw::DataType::Int32);
    auto tile_pool_y_s = writer->declare_tile("pool_y_s", ckw::DataType::Int32);
    auto tile_pool_x_e = writer->declare_tile("pool_x_e", ckw::DataType::Int32);
    auto tile_pool_y_e = writer->declare_tile("pool_y_e", ckw::DataType::Int32);

    writer->op_binary(tile_pool_x_s, ckw::BinaryOp::Max, const_0_i32, tile_neg_src_coord_x_start);
    writer->op_binary(tile_pool_x_e, ckw::BinaryOp::Add, const_src_w_i32, tile_neg_src_coord_x_start);
    writer->op_binary(tile_pool_x_e, ckw::BinaryOp::Min, const_pool_sz_x_i32, tile_pool_x_e);
    writer->op_binary(tile_pool_y_s, ckw::BinaryOp::Max, const_0_i32, tile_neg_src_coord_y_start);
    writer->op_binary(tile_pool_y_e, ckw::BinaryOp::Add, const_src_h_i32, tile_neg_src_coord_y_start);
    writer->op_binary(tile_pool_y_e, ckw::BinaryOp::Min, const_pool_sz_y_i32, tile_pool_y_e);

    // #if defined(EXCLUDE_PADDING)
    // int filter_size = (pool_y_e - pool_y_s) * (pool_x_e - pool_x_s);
    // #else // defined(EXCLUDE_PADDING)
    // int filter_size = POOL_SIZE_X * POOL_SIZE_Y;
    // #endif // defined(EXCLUDE_PADDING)
    auto tile_filter_size = writer->declare_tile("filter_size", ckw::DataType::Int32);
    if (_attributes.exclude_padding())
    {
        auto tile_x_diff = writer->declare_tile("x_diff", ckw::DataType::Int32);
        auto tile_y_diff = writer->declare_tile("y_diff", ckw::DataType::Int32);

        writer->op_binary(tile_x_diff, ckw::BinaryOp::Sub, tile_pool_x_e, tile_pool_x_s);
        writer->op_binary(tile_y_diff, ckw::BinaryOp::Sub, tile_pool_y_e, tile_pool_y_s);
        writer->op_binary(tile_filter_size, ckw::BinaryOp::Mul, tile_x_diff, tile_y_diff);
    }
    else
    {
        writer->op_binary(tile_filter_size, ckw::BinaryOp::Mul, const_pool_sz_x_i32, const_pool_sz_y_i32);
    }

    auto tile_x = writer->declare_tile("x", ckw::DataType::Int32);
    auto tile_y = writer->declare_tile("y", ckw::DataType::Int32);

    if (is_global_pooling)
    {
        writer->op_assign(tile_y, const_0_i32);
        writer->op_assign(tile_pool_y_e, const_pool_sz_y_i32);
    }
    else
    {
        writer->op_assign(tile_y, tile_pool_y_s);
    }

    // Y dim for-loop
    writer->op_for_loop(
        tile_y, ckw::BinaryOp::Less, tile_pool_y_e, tile_y, ckw::AssignmentOp::Increment, const_pos_1_i32,
        [&]()
        {
            // Reset the iterator for the inner loop
            if (is_global_pooling)
            {
                writer->op_assign(tile_x, const_0_i32);
                writer->op_assign(tile_pool_x_e, const_pool_sz_x_i32);
            }
            else
            {
                writer->op_assign(tile_x, tile_pool_x_s);
            }

            auto tile_src_coord_y = writer->declare_tile("src_coord_y", ckw::DataType::Int32);
            writer->op_binary(tile_src_coord_y, ckw::BinaryOp::Add, tile_src_coord_y_start, tile_y);

            // X dim for-loop
            writer->op_for_loop(
                tile_x, ckw::BinaryOp::Less, tile_pool_x_e, tile_x, ckw::AssignmentOp::Increment, const_pos_1_i32,
                [&]()
                {
                    auto tile_src_coord_x = writer->declare_tile("src_coord_x", ckw::DataType::Int32);
                    writer->op_binary(tile_src_coord_x, ckw::BinaryOp::Add, tile_src_coord_x_start, tile_x);

                    ckw::DataType src_dt   = to_ckw(_src->data_type());
                    auto          tile_src = writer->declare_tile("tile_src", ckw::TileInfo(acc_dt, dst_m0, dst_n0));

                    // Load src tile
                    if (is_wider_acc)
                    {
                        auto tile_src0 = writer->declare_tile("src_tile0", ckw::TileInfo(src_dt, dst_m0, dst_n0));
                        writer->op_load(tile_src0, src->tensor(), sampler_src, tile_cout0, tile_src_coord_x,
                                        tile_src_coord_y, tile_bout0);
                        writer->op_cast(tile_src, tile_src0, ckw::ConvertPolicy::None);
                    }
                    else
                    {
                        writer->op_load(tile_src, src->tensor(), sampler_src, tile_cout0, tile_src_coord_x,
                                        tile_src_coord_y, tile_bout0);
                    }

                    // Take the square of the input, for L2 Pooling
                    if (_attributes.pool_type() == PoolingType::L2)
                    {
                        writer->op_binary(tile_src, ckw::BinaryOp::Mul, tile_src, tile_src);
                    }

                    // Perfom Pooling op
                    if (_attributes.pool_type() == PoolingType::MAX)
                    {
                        writer->op_binary(tile_res, ckw::BinaryOp::Max, tile_res, tile_src);
                    }
                    else
                    {
                        writer->op_binary(tile_res, ckw::BinaryOp::Add, tile_res, tile_src);
                    }
                });
        });

    if ((_attributes.pool_type() == PoolingType::AVG) || (_attributes.pool_type() == PoolingType::L2))
    {
        // Filter_size is automatically broadcasted in the operation
        auto tile_filter_size_fp = writer->declare_tile("filter_size_fp", ckw::TileInfo(acc_dt));
        writer->op_cast(tile_filter_size_fp, tile_filter_size, ckw::ConvertPolicy::None);
        writer->op_binary(tile_res, ckw::BinaryOp::Div, tile_res, tile_filter_size_fp);
    }

    // Take square root of the result in L2 pooling
    if (_attributes.pool_type() == PoolingType::L2)
    {
        writer->op_unary(tile_res, ckw::UnaryOp::Sqrt, tile_res);
    }

    // Store the results and do casting if mixed precision
    if (is_wider_acc)
    {
        writer->op_cast(tile_dst, tile_res, ckw::ConvertPolicy::None);
    }
    else
    {
        writer->op_assign(tile_dst, tile_res);
    }
}

Window GpuCkwPool2d::get_window() const
{
    ARM_COMPUTE_ERROR_ON_MSG(_dst->tensor_shape().total_size() == 0U, "Destination tensor is not initialized");

    TensorShape    output_shape = _dst->tensor_shape();
    const uint32_t vec_size     = adjust_vec_size(((_dst->data_type() == DataType::F32) ? 2 : 4), _dst->dimension(0));
    // Create and configure kernel window
    auto win = calculate_max_window(output_shape, Steps(vec_size));
    win      = win.collapse_if_possible(win, Window::DimZ); // collapse window on batch size.
    return win;
}

std::string GpuCkwPool2d::get_name(const ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);

    return "pool2dMxN";
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
