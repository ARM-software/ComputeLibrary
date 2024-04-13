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
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/GpuCkwPool2d.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/Validate.h"
#include "ckw/TensorTileSampler.h"

#include "src/core/helpers/WindowHelpers.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/type_converter/Common.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/WriterHelper.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwKernelWriter.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwScopedKernelWriter.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwVariableTable.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelArgument.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGroup.h"

using namespace ckw;

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
    const auto         root_window = comp_group.get_root_component()->ckw_component_driver()->get_window();
    const unsigned int n0          = root_window.x().step();
    const unsigned int m0          = root_window.y().step();

    GpuCkwComponentArgument *src =
        vtable.declare_variable(comp_group, writer, _src, TensorStorageType::ClBufferUint8Ptr, "src");
    GpuCkwComponentArgument *dst =
        vtable.declare_variable(comp_group, writer, _dst, TensorStorageType::ClBufferUint8Ptr, "dst");

    TileOperand &gid_0 = writer->declare_tile("gid_0", ckw::DataType::Int32);
    TileOperand &gid_1 = writer->declare_tile("gid_1", ckw::DataType::Int32);
    TileOperand &gid_2 = writer->declare_tile("gid_2", ckw::DataType::Int32);

    writer->op_get_global_id(gid_0, 0);
    writer->op_get_global_id(gid_1, 1);
    writer->op_get_global_id(gid_2, 2);

    // Data Layout is NHWC
    constexpr int width_idx  = 1;
    constexpr int height_idx = 2;

    const int32_t pool_size_x   = static_cast<int32_t>(_attributes.pool_size().x());
    const int32_t pool_size_y   = static_cast<int32_t>(_attributes.pool_size().y());
    const int32_t pad_x         = static_cast<int32_t>(_attributes.pad().left);
    const int32_t pad_y         = static_cast<int32_t>(_attributes.pad().top);
    const int32_t src_width     = static_cast<int32_t>(_src->dimension(width_idx));
    const int32_t src_height    = static_cast<int32_t>(_src->dimension(height_idx));
    const auto    src_data_type = _src->data_type();

    // Check if this is global pooling path
    const bool is_global_pooling =
        (pool_size_x == src_width) && (pool_size_y == src_height) && (pad_x == 0) && (pad_y == 0);
    // Check if this a case of FP_MIXED_PRECISION
    const bool use_fp_mixed_precision =
        (src_data_type == DataType::F16) && _settings.mixed_precision() && _attributes.pool_type() != PoolingType::MAX;
    const auto acc_data_type = (use_fp_mixed_precision) ? (DataType::F32) : (src_data_type);

    TileOperand       &const_0            = writer->declare_tile("0", 0);
    const TileOperand &const_1            = writer->declare_tile("1", 1);
    const TileOperand &const_lowest_value = writer->declare_tile("LOWEST_VALUE", std::numeric_limits<float>::lowest());
    const TileOperand &pool_size_x_tile   = writer->declare_tile("POOL_SIZE_X", pool_size_x);
    const TileOperand &pool_size_y_tile   = writer->declare_tile("POOL_SIZE_Y", pool_size_y);
    const TileOperand &stride_x_tile = writer->declare_tile("STRIDE_X", static_cast<int32_t>(_attributes.stride().x()));
    const TileOperand &stride_y_tile = writer->declare_tile("STRIDE_Y", static_cast<int32_t>(_attributes.stride().y()));
    const TileOperand &pad_x_tile    = writer->declare_tile("PAD_X", pad_x);
    const TileOperand &pad_y_tile    = writer->declare_tile("PAD_Y", pad_y);
    const TileOperand &dst_height_tile =
        writer->declare_tile("DST_HEIGHT", static_cast<int32_t>(_dst->dimension(height_idx)));
    const TileOperand &src_height_tile = writer->declare_tile("SRC_HEIGHT", src_height);
    const TileOperand &src_width_tile  = writer->declare_tile("SRC_WIDTH", src_width);

    TileOperand &idx_out_n = writer->declare_tile("idx_out_n", ckw::DataType::Int32);
    TileOperand &idx_out_h = writer->declare_tile("idx_out_h", ckw::DataType::Int32);
    TileOperand &idx_out_w = writer->declare_tile("idx_out_w", ckw::DataType::Int32);
    TileOperand &idx_out_c = writer->declare_tile("idx_out_c", ckw::DataType::Int32);

    const int32_t dst_partial_n0_v = _dst->tensor_shape()[0] % n0;

    get_coord(writer, idx_out_c, gid_0, n0, dst_partial_n0_v, "dst_x_", const_0);
    get_coord(writer, idx_out_w, gid_1, 1, 0, "dst_y_", const_0);

    writer->op_binary_expression(idx_out_h, gid_2, BinaryOp::Mod, dst_height_tile); // gid_2 % h
    writer->op_binary_expression(idx_out_n, gid_2, BinaryOp::Div, dst_height_tile); // gid_2 / h

    TensorTileSampler src_sampler;
    src_sampler.width(n0);
    src_sampler.height(m0);
    src_sampler.format(TensorSamplerFormat::C_W_H);
    src_sampler.address_mode_x(TensorSamplerAddressModeX::None);
    src_sampler.address_mode_y(TensorSamplerAddressModeY::None);
    src_sampler.address_mode_z(TensorSamplerAddressModeZ::None);
    src_sampler.x(idx_out_c);
    src_sampler.b(idx_out_n);

    TensorTileSampler dst_sampler;
    dst_sampler.width(n0);
    dst_sampler.height(m0);
    dst_sampler.format(TensorSamplerFormat::C_W_H);
    dst_sampler.address_mode_x(TensorSamplerAddressModeX::OverlappingMin);
    dst_sampler.address_mode_y(TensorSamplerAddressModeY::None);
    dst_sampler.address_mode_z(TensorSamplerAddressModeZ::None);
    dst_sampler.x(idx_out_c);
    dst_sampler.y(idx_out_w);
    dst_sampler.z(idx_out_h);
    dst_sampler.b(idx_out_n);

    // Prepare dst tensor and tile
    TileInfo dst_tile_info = TileInfo(to_ckw(src_data_type), m0, n0);
    if (!dst->has_tile())
    {
        TileOperand &dst_tile = writer->declare_tile("dst_tile", dst_tile_info);
        dst->init_virtual_tensor(dst_tile, dst_sampler);
    }
    const TileOperand &dst_tile = dst->tile();

    // A tile used to temporarily store results or as an accumulator in case of AVG and L2 pooling.
    const TileOperand &res_tile = writer->declare_tile("res_tile", TileInfo(to_ckw(acc_data_type), m0, n0));

    // Initialise result tile with appropriate value
    if (_attributes.pool_type() == PoolingType::MAX)
    {
        if (_settings.use_inf_as_limit())
        {
            TileContainer            minus_inf_tile_container;
            std::vector<std::string> value = std::vector<std::string>(n0, "(-INFINITY)");
            minus_inf_tile_container.push_back({value});
            const TileOperand &minus_inf =
                writer->declare_tile("minus_inf_const", minus_inf_tile_container, to_ckw(acc_data_type));
            writer->op_assign(res_tile, minus_inf);
        }
        else
        {
            writer->op_assign(res_tile, const_lowest_value);
        }
    }
    else
    {
        writer->op_assign(res_tile, const_0);
    }

    // idx_in_w = idx_out_w * STRIDE_X - PAD_X
    TileOperand &idx_in_w = writer->declare_tile("idx_in_w", ckw::DataType::Int32);
    writer->op_binary_expression(idx_in_w, idx_out_w, BinaryOp::Mul, stride_x_tile);
    writer->op_binary_expression(idx_in_w, idx_in_w, BinaryOp::Sub, pad_x_tile);

    // idx_in_h = idx_out_h * STRIDE_Y - PAD_Y
    TileOperand &idx_in_h = writer->declare_tile("idx_in_h", ckw::DataType::Int32);
    writer->op_binary_expression(idx_in_h, idx_out_h, BinaryOp::Mul, stride_y_tile);
    writer->op_binary_expression(idx_in_h, idx_in_h, BinaryOp::Sub, pad_y_tile);

    TileOperand &minus_idx_in_w = writer->declare_tile("minus_idx_in_w", ckw::DataType::Int32);
    TileOperand &minus_idx_in_h = writer->declare_tile("minus_idx_in_h", ckw::DataType::Int32);

    writer->op_unary_expression(minus_idx_in_w, UnaryOp::Negate, idx_in_w);
    writer->op_unary_expression(minus_idx_in_h, UnaryOp::Negate, idx_in_h);

    // Pooling starting/ending offsets for X dim
    TileOperand &pool_x_s = writer->declare_tile("pool_x_s", ckw::DataType::Int32);
    TileOperand &pool_x_e = writer->declare_tile("pool_x_e", ckw::DataType::Int32);

    writer->op_binary_elementwise_function(pool_x_s, BinaryFunction::Max, const_0, minus_idx_in_w);
    writer->op_binary_expression(pool_x_e, src_width_tile, BinaryOp::Add, minus_idx_in_w);
    writer->op_binary_elementwise_function(pool_x_e, BinaryFunction::Min, pool_size_x_tile, pool_x_e);

    // Pooling starting/ending offsets for Y dim
    TileOperand &pool_y_s = writer->declare_tile("pool_y_s", ckw::DataType::Int32);
    TileOperand &pool_y_e = writer->declare_tile("pool_y_e", ckw::DataType::Int32);

    writer->op_binary_elementwise_function(pool_y_s, BinaryFunction::Max, const_0, minus_idx_in_h);
    writer->op_binary_expression(pool_y_e, src_height_tile, BinaryOp::Add, minus_idx_in_h);
    writer->op_binary_elementwise_function(pool_y_e, BinaryFunction::Min, pool_size_y_tile, pool_y_e);

    const TileOperand &filter_size = writer->declare_tile("filter_size", ckw::DataType::Int32);
    if (_attributes.exclude_padding())
    {
        const TileOperand &y_diff = writer->declare_tile("y_diff", ckw::DataType::Int32);
        const TileOperand &x_diff = writer->declare_tile("x_diff", ckw::DataType::Int32);

        writer->op_binary_expression(y_diff, pool_y_e, BinaryOp::Sub, pool_y_s);
        writer->op_binary_expression(x_diff, pool_x_e, BinaryOp::Sub, pool_x_s);

        writer->op_binary_expression(filter_size, y_diff, BinaryOp::Mul, x_diff);
    }
    else
    {
        writer->op_binary_expression(filter_size, pool_size_x_tile, BinaryOp::Mul, pool_size_y_tile);
    }

    const TileOperand &x = writer->declare_tile("x", ckw::DataType::Int32);
    const TileOperand &y = writer->declare_tile("y", ckw::DataType::Int32);

    if (is_global_pooling)
    {
        writer->op_assign(x, const_0);
        writer->op_assign(y, const_0);

        writer->op_assign(pool_y_e, pool_size_y_tile);
        writer->op_assign(pool_x_e, pool_size_x_tile);
    }
    else
    {
        writer->op_assign(x, pool_x_s);
        writer->op_assign(y, pool_y_s);
    }

    // Y dim for-loop
    writer->op_for_loop(
        y, BinaryOp::Less, pool_y_e, y, AssignmentOp::Increment, const_1,
        [&]()
        {
            // Reset the iterator for the inner loop
            if (is_global_pooling)
            {
                writer->op_assign(x, const_0);
            }
            else
            {
                writer->op_assign(x, pool_x_s);
            }

            TileOperand &a_y = writer->declare_tile("a_y", ckw::DataType::Int32);
            writer->op_binary_expression(a_y, idx_in_h, BinaryOp::Add, y);

            // X dim for-loop
            writer->op_for_loop(
                x, BinaryOp::Less, pool_x_e, x, AssignmentOp::Increment, const_1,
                [&]()
                {
                    TileOperand &a_x = writer->declare_tile("a_x", ckw::DataType::Int32);
                    writer->op_binary_expression(a_x, idx_in_w, BinaryOp::Add, x);

                    TileOperand &src_tile = writer->declare_tile("src_tile", TileInfo(to_ckw(acc_data_type), m0, n0));

                    src_sampler.y(a_x);
                    src_sampler.z(a_y);

                    // Load src tile
                    if (use_fp_mixed_precision)
                    {
                        TileOperand &src_uncasted_tile = writer->declare_tile("uncasted_src_tile", dst_tile_info);
                        writer->op_load(src_uncasted_tile, src->tensor(), src_sampler);
                        writer->op_cast_expression(src_tile, src_uncasted_tile, ckw::ConvertPolicy::None);
                    }
                    else
                    {
                        writer->op_load(src_tile, src->tensor(), src_sampler);
                    }

                    // Take the square of the input, for L2 Pooling
                    if (_attributes.pool_type() == PoolingType::L2)
                    {
                        writer->op_binary_expression(src_tile, src_tile, BinaryOp::Mul, src_tile);
                    }

                    // Perfom Pooling op
                    if (_attributes.pool_type() == PoolingType::MAX)
                    {
                        writer->op_binary_elementwise_function(res_tile, BinaryFunction::Max, res_tile, src_tile);
                    }
                    else
                    {
                        writer->op_binary_expression(res_tile, res_tile, BinaryOp::Add, src_tile);
                    }
                });
        });

    if ((_attributes.pool_type() == PoolingType::AVG) || (_attributes.pool_type() == PoolingType::L2))
    {
        // filter_size is automatically broadcasted in the operation
        writer->op_binary_expression(res_tile, res_tile, BinaryOp::Div, filter_size);
    }

    // Take square root of the result in L2 pooling
    if (_attributes.pool_type() == PoolingType::L2)
    {
        writer->op_unary_elementwise_function(res_tile, UnaryFunction::Sqrt, res_tile);
    }

    // Store the results and do casting if FP_MIXED_PRECISION
    if (use_fp_mixed_precision)
    {
        writer->op_cast_expression(dst_tile, res_tile, ckw::ConvertPolicy::None);
    }
    else
    {
        writer->op_assign(dst_tile, res_tile);
    }
}

Window GpuCkwPool2d::get_window() const
{
    ARM_COMPUTE_ERROR_ON_MSG(_dst->tensor_shape().total_size() == 0U, "Destination tensor is not initialized");

    TensorShape        output_shape = _dst->tensor_shape();
    const unsigned int vec_size = adjust_vec_size(((_dst->data_type() == DataType::F32) ? 2 : 4), _dst->dimension(0));
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
