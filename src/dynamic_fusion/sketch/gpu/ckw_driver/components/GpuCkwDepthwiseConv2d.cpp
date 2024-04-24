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
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/GpuCkwDepthwiseConv2d.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Validate.h"
#include "ckw/TensorTileSampler.h"

#include "src/core/helpers/WindowHelpers.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/type_converter/Common.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/WriterHelper.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwKernelWriter.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwScopedKernelWriter.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwVariableTable.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGroup.h"

#include <cstdint>
#include <string>

using namespace ckw;
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
    : IGpuCkwComponentDriver{id, tensors},
      _src{},
      _weight{},
      _bias{},
      _dst{},
      _attributes{attributes},
      _settings{settings}
{
    _src    = this->tensors().get_const_tensor(TensorType::ACL_SRC_0);
    _weight = this->tensors().get_const_tensor(TensorType::ACL_SRC_1);
    if (this->tensors().get_const_tensor(TensorType::ACL_SRC_2))
    {
        _bias = this->tensors().get_const_tensor(TensorType::ACL_SRC_2);
    }
    _dst = this->tensors().get_const_tensor(TensorType::ACL_DST_0);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _weight, _bias, _dst);
}

void GpuCkwDepthwiseConv2d::write_component_code(const ComponentGroup    &comp_group,
                                                 GpuCkwVariableTable     &vtable,
                                                 GpuCkwScopedKernelWriter writer) const
{
    // Data Layout is NHWC
    constexpr int32_t width_idx  = 1;
    constexpr int32_t height_idx = 2;

    const auto root_window = comp_group.get_root_component()->ckw_component_driver()->get_window();

    // Tunable parameters
    // Currently only m0 and n0 = 1 are supported.
    const int32_t     m0       = root_window.y().step();
    const int32_t     n0       = root_window.x().step();
    constexpr int32_t m0_a_val = 1;
    constexpr int32_t n0_a_val = 1;
    constexpr int32_t m0_b_val = 1;

    GpuCkwComponentArgument *src =
        vtable.declare_variable(comp_group, writer, _src, TensorStorageType::ClBufferUint8Ptr, "src");
    GpuCkwComponentArgument *wei =
        vtable.declare_variable(comp_group, writer, _weight, TensorStorageType::ClBufferUint8Ptr, "wei");
    GpuCkwComponentArgument *bia = nullptr;

    if (_bias && _bias->has_valid_id())
    {
        bia = vtable.declare_variable(comp_group, writer, _bias, TensorStorageType::ClBufferUint8Ptr, "bia");
    }
    GpuCkwComponentArgument *dst =
        vtable.declare_variable(comp_group, writer, _dst, TensorStorageType::ClBufferUint8Ptr, "dst");

    // Constants
    const auto &const_1    = writer->declare_tile("1", 1);
    const auto &wei_height = writer->declare_tile("WEI_HEIGHT", static_cast<int32_t>(_weight->dimension(height_idx)));
    const auto &wei_width  = writer->declare_tile("WEI_WIDTH", static_cast<int32_t>(_weight->dimension(width_idx)));
    const auto &dst_height = writer->declare_tile("DST_HEIGHT", static_cast<int32_t>(_dst->dimension(height_idx)));
    const auto &stride_x   = writer->declare_tile("STRIDE_X", static_cast<int32_t>(_attributes.stride().x()));
    const auto &stride_y   = writer->declare_tile("STRIDE_Y", static_cast<int32_t>(_attributes.stride().y()));
    const auto &pad_left   = writer->declare_tile("PAD_LEFT", static_cast<int32_t>(_attributes.pad().left));
    const auto &pad_top    = writer->declare_tile("PAD_TOP", static_cast<int32_t>(_attributes.pad().top));
    const auto &depth_multiplier =
        writer->declare_tile("DEPTH_MULTIPLIER", static_cast<int32_t>(_attributes.depth_multiplier()));
    auto &const_0 = writer->declare_tile("0", 0);
    auto &yo      = writer->declare_tile("yo", ckw::DataType::Int32);

    auto &gid_0 = writer->declare_tile("gid_0", ckw::DataType::Int32);
    auto &gid_1 = writer->declare_tile("gid_1", ckw::DataType::Int32);
    auto &gid_2 = writer->declare_tile("gid_2", ckw::DataType::Int32);

    writer->op_get_global_id(gid_0, 0);
    writer->op_get_global_id(gid_1, 1);
    writer->op_get_global_id(gid_2, 2);

    auto &bout = writer->declare_tile("bout", ckw::DataType::Int32);
    writer->op_binary_expression(bout, gid_2, ckw::BinaryOp::Div, dst_height); // gid_2 / h
    writer->op_binary_expression(yo, gid_2, ckw::BinaryOp::Mod, dst_height);   // gid_2 % h

    const int32_t dst_partial_n0_v = _dst->tensor_shape()[0] % n0;
    const int32_t dst_partial_m0_v = _dst->tensor_shape()[1] % m0;
    auto         &g_ind_0          = writer->declare_tile("g_ind_0", ckw::DataType::Int32);
    auto         &g_ind_1          = writer->declare_tile("g_ind_1", ckw::DataType::Int32);
    get_coord(writer, g_ind_0, gid_0, n0, dst_partial_n0_v, "dst_x_", const_0);
    get_coord(writer, g_ind_1, gid_1, m0, dst_partial_m0_v, "dst_y_", const_0);

    TensorTileSampler src_sampler;
    src_sampler.width(m0_a_val);
    src_sampler.height(n0_a_val);
    src_sampler.format(TensorSamplerFormat::C_W_H);
    src_sampler.address_mode_x(TensorSamplerAddressModeX::None);
    src_sampler.address_mode_y(TensorSamplerAddressModeY::Skip);
    src_sampler.address_mode_z(TensorSamplerAddressModeZ::Skip);

    TensorTileSampler wei_sampler;
    wei_sampler.width(m0_b_val);
    wei_sampler.height(n0);
    wei_sampler.format(TensorSamplerFormat::C_W_H);
    wei_sampler.address_mode_x(TensorSamplerAddressModeX::None);
    wei_sampler.address_mode_y(TensorSamplerAddressModeY::None);
    wei_sampler.address_mode_z(TensorSamplerAddressModeZ::None);

    TensorTileSampler dst_sampler;
    dst_sampler.width(n0);
    dst_sampler.height(m0);
    dst_sampler.format(TensorSamplerFormat::C_W_H);
    dst_sampler.address_mode_x(TensorSamplerAddressModeX::None);
    dst_sampler.address_mode_y(TensorSamplerAddressModeY::None);
    dst_sampler.address_mode_z(TensorSamplerAddressModeZ::Skip);
    dst_sampler.x(g_ind_0);
    dst_sampler.y(g_ind_1);
    dst_sampler.z(yo);
    dst_sampler.b(bout);

    if (!dst->has_tile())
    {
        auto &dst_tile = writer->declare_tile("dst_tile", ckw::TileInfo(to_ckw(_dst->data_type()), m0, n0));
        dst->init_virtual_tensor(dst_tile, dst_sampler);
    }
    auto &dst_tile = dst->tile();

    writer->op_assign(dst_tile, const_0);

    auto &xi = writer->declare_tile("xi", ckw::DataType::Int32);
    writer->op_binary_expression(xi, g_ind_1, ckw::BinaryOp::Mul, stride_x);
    writer->op_binary_expression(xi, xi, ckw::BinaryOp::Sub, pad_left);

    auto &yi = writer->declare_tile("yi", ckw::DataType::Int32);
    writer->op_binary_expression(yi, yo, ckw::BinaryOp::Mul, stride_y);
    writer->op_binary_expression(yi, yi, ckw::BinaryOp::Sub, pad_top);

    auto &a_x = writer->declare_tile("a_x", ckw::DataType::Int32);
    writer->op_binary_expression(a_x, g_ind_0, BinaryOp::Div, depth_multiplier);

    // src_tile
    auto &a = writer->declare_tile("a", ckw::TileInfo(to_ckw(_src->data_type()), m0_a_val, n0_a_val));
    // wei_tile
    auto &b = writer->declare_tile("b", ckw::TileInfo(to_ckw(_weight->data_type()), m0_b_val, n0));

    // Loop variables
    auto &yk = writer->declare_tile("yk", ckw::DataType::Int32);
    auto &xk = writer->declare_tile("xk", ckw::DataType::Int32);

    // Because 1x1 blocks are being used here, scalar values are being loaded from memory instead of using tiles, since tile vector access currently is not available. Hence the values are loaded in the inner loop.
    // This loop will be reworked.
    writer->op_assign(yk, const_0);
    writer->op_for_loop(yk, BinaryOp::Less, wei_height, yk, AssignmentOp::Increment, const_1,
                        [&]()
                        {
                            // xk = 0
                            writer->op_assign(xk, const_0);
                            writer->op_for_loop(
                                xk, BinaryOp::Less, wei_width, xk, AssignmentOp::Increment, const_1,
                                [&]()
                                {
                                    writer->op_assign(b, const_0);
                                    writer->op_assign(a, const_0);

                                    // src_tile loading
                                    auto &xi_curr = writer->declare_tile("xi_curr", ckw::DataType::Int32);
                                    writer->op_binary_expression(xi_curr, xi, BinaryOp::Add, xk);
                                    auto &a_y = writer->declare_tile("a_y", ckw::DataType::Int32);
                                    writer->op_binary_expression(a_y, yi, BinaryOp::Add, yk);
                                    src_sampler.x(a_x);
                                    src_sampler.y(xi_curr);
                                    src_sampler.z(a_y);
                                    src_sampler.b(bout);
                                    writer->op_load(a, src->tensor(), src_sampler);

                                    // wei_tile loading
                                    auto &b_y = writer->declare_tile("b_y", ckw::DataType::Int32);
                                    writer->op_binary_expression(b_y, wei_width, BinaryOp::Mul, yk);
                                    writer->op_binary_expression(b_y, b_y, BinaryOp::Add, xk);
                                    wei_sampler.x(g_ind_0);
                                    wei_sampler.y(b_y);
                                    wei_sampler.z(const_0);
                                    wei_sampler.b(const_0);
                                    writer->op_load(b, wei->tensor(), wei_sampler);

                                    // Do the accumulation
                                    auto &mul_result = writer->declare_tile("mul_results", a.data_type());
                                    writer->op_binary_expression(mul_result, a, BinaryOp::Mul, b);
                                    writer->op_binary_expression(dst_tile, dst_tile, BinaryOp::Add, mul_result);
                                });
                        });

    // Add Bias
    if (_bias && _bias->has_valid_id())
    {
        TensorTileSampler bias_sampler;
        bias_sampler.width(n0);
        bias_sampler.height(1);
        bias_sampler.format(TensorSamplerFormat::C_W_H);
        bias_sampler.address_mode_x(TensorSamplerAddressModeX::None);
        bias_sampler.address_mode_y(TensorSamplerAddressModeY::None);
        bias_sampler.address_mode_z(TensorSamplerAddressModeZ::None);
        bias_sampler.x(g_ind_0);
        bias_sampler.y(const_0);
        bias_sampler.z(const_0);
        bias_sampler.b(const_0);

        auto &bias_tile = writer->declare_tile("bias_tile", ckw::TileInfo(to_ckw(_bias->data_type()), 1, n0));
        writer->op_load(bias_tile, bia->tensor(), bias_sampler);
        writer->op_binary_expression(dst_tile, dst_tile, BinaryOp::Add, bias_tile);
    }
}

Window GpuCkwDepthwiseConv2d::get_window() const
{
    ARM_COMPUTE_ERROR_ON_MSG(_dst->tensor_shape().total_size() == 0U, "Destination tensor is not initialized");
    TensorShape output_shape = _dst->tensor_shape();
    // Currently only m0 and n0 = 1 are supported.
    Window win = calculate_max_window(output_shape, Steps(1U, 1U));
    return win.collapse(win, Window::DimZ);
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
