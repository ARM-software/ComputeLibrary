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

#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/GpuCkwMatMul.h"

#include "arm_compute/core/utils/helpers/AdjustVecSize.h"

#include "src/core/helpers/WindowHelpers.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/type_converter/Common.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/WriterHelper.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwKernelWriter.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwScopedKernelWriter.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwVariableTable.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGroup.h"
#include "support/StringSupport.h"

using namespace ckw;
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
    const auto root_window = comp_group.get_root_component()->ckw_component_driver()->get_window();

    GpuCkwComponentArgument *lhs =
        vtable.declare_variable(comp_group, writer, _lhs, TensorStorageType::ClBufferUint8Ptr, "lhs");
    GpuCkwComponentArgument *rhs =
        vtable.declare_variable(comp_group, writer, _rhs, TensorStorageType::ClBufferUint8Ptr, "rhs");
    GpuCkwComponentArgument *dst =
        vtable.declare_variable(comp_group, writer, _dst, TensorStorageType::ClBufferUint8Ptr, "dst");

    // Constants
    const int   height_idx = get_data_layout_dimension_index(_lhs->data_layout(), DataLayoutDimension::HEIGHT);
    const auto &rhs_h      = writer->declare_tile("rhs_h", static_cast<int32_t>(_rhs->dimension(height_idx)));
    const int   m          = static_cast<int>(_dst->dimension(1));
    const int   n          = static_cast<int>(_dst->dimension(0));
    const int   k =
        _attributes.adj_lhs() ? static_cast<int>(_lhs->tensor_shape().y()) : static_cast<int>(_lhs->tensor_shape().x());
    const int m0               = root_window.y().step();
    const int n0               = root_window.x().step();
    const int k0               = _settings.k0();
    const int partial_store_m0 = m % m0;
    const int partial_store_n0 = n % n0;

    const auto &const_1 = writer->declare_tile("1", 1);
    auto       &const_0 = writer->declare_tile("0", 0);
    auto       &k0_tile = writer->declare_tile("k0", k0);
    auto       &k_tile  = writer->declare_tile("k", k);

    auto &gid_0 = writer->declare_tile("gid_0", ckw::DataType::Int32);
    auto &gid_1 = writer->declare_tile("gid_1", ckw::DataType::Int32);
    auto &gid_2 = writer->declare_tile("gid_2", ckw::DataType::Int32);

    writer->op_get_global_id(gid_0, 0);
    writer->op_get_global_id(gid_1, 1);
    writer->op_get_global_id(gid_2, 2);

    auto &x = writer->declare_tile("x", ckw::DataType::Int32);
    auto &y = writer->declare_tile("y", ckw::DataType::Int32);
    auto &z = writer->declare_tile("z", ckw::DataType::Int32);

    get_coord(writer, x, gid_0, n0, partial_store_n0, "gid_x_", const_0);
    get_coord(writer, y, gid_1, m0, partial_store_m0, "gid_y_", const_0);
    get_coord(writer, z, gid_2, 1, 0, "gid_z_", const_0);

    TensorTileSampler lhs_sampler;
    lhs_sampler.height(m0);
    lhs_sampler.width(k0);
    lhs_sampler.format(TensorSamplerFormat::C_W_H);
    lhs_sampler.address_mode_x(TensorSamplerAddressModeX::None);
    lhs_sampler.address_mode_y(TensorSamplerAddressModeY::None);
    lhs_sampler.address_mode_z(TensorSamplerAddressModeZ::None);

    TensorTileSampler rhs_sampler;
    rhs_sampler.height(k0);
    rhs_sampler.width(n0);
    rhs_sampler.format(TensorSamplerFormat::C_WH_1);
    rhs_sampler.address_mode_x(TensorSamplerAddressModeX::None);
    rhs_sampler.address_mode_y(TensorSamplerAddressModeY::None);
    rhs_sampler.address_mode_z(TensorSamplerAddressModeZ::None);

    TensorTileSampler dst_sampler;
    dst_sampler.width(n0);
    dst_sampler.height(m0);
    dst_sampler.format(TensorSamplerFormat::C_W_H);
    dst_sampler.address_mode_x(TensorSamplerAddressModeX::OverlappingMin);
    dst_sampler.address_mode_y(TensorSamplerAddressModeY::None);
    dst_sampler.address_mode_z(TensorSamplerAddressModeZ::None);
    dst_sampler.x(x);
    dst_sampler.y(y);
    dst_sampler.z(z);
    dst_sampler.b(const_0);

    if (!dst->has_tile())
    {
        auto &dst_tile = writer->declare_tile("dst_tile", ckw::TileInfo(to_ckw(_dst->data_type()), m0, n0));
        dst->init_virtual_tensor(dst_tile, dst_sampler);
    }
    auto &dst_tile = dst->tile();

    // Initialize the accumulators
    writer->op_assign(dst_tile, const_0);

    auto &rhs_z = writer->declare_tile("rhs_z", ckw::DataType::Int32);
    writer->op_binary_expression(rhs_z, z, BinaryOp::Mul, rhs_h);

    auto &k_i     = writer->declare_tile("k_i", ckw::DataType::Int32);
    auto &k_limit = writer->declare_tile("k_limit", k - k0);

    auto &x_i = writer->declare_tile("x_i", ckw::DataType::Int32);
    writer->op_assign(x_i, const_0);

    writer->op_assign(k_i, const_0);

    // *INDENT-OFF*
    // clang-format off
    writer->op_for_loop(k_i, BinaryOp::LessEqual, k_limit, k_i, AssignmentOp::Increment, k0_tile,
        [&]()
        {
            //Initialize tiles
            // lhs_tile
            auto &a = writer->declare_tile("a", ckw::TileInfo(to_ckw(_lhs->data_type()), m0, k0));
            // rhs_tile
            auto &b = writer->declare_tile("b", ckw::TileInfo(to_ckw(_rhs->data_type()), n0, k0));
            writer->op_assign(a, const_0);
            writer->op_assign(b, const_0);

            // Loading the tiles
            // LHS
            lhs_sampler.x(x_i);
            lhs_sampler.y(y);
            lhs_sampler.z(z);
            lhs_sampler.b(const_0);
            writer->op_load(a, lhs->tensor(), lhs_sampler);

            // RHS
            auto &y_i = writer->declare_tile("y_i", ckw::DataType::Int32);
            writer->op_binary_expression(y_i, x, BinaryOp::Add, rhs_z);
            rhs_sampler.x(k_i);
            rhs_sampler.y(y_i);
            rhs_sampler.z(const_0);
            rhs_sampler.b(const_0);
            writer->op_load(b, rhs->tensor(), rhs_sampler);

            // Perform Matmul
            writer->op_binary_expression(dst_tile, a, BinaryOp::MatMul_Nt_T, b);
            writer->op_binary_expression(x_i, x_i, BinaryOp::Add, k0_tile);
        });
// *INDENT-ON*
    // clang-format on

    // Handling leftovers
    if (k % k0 != 0)
    {
        // *INDENT-OFF*
        // clang-format off
        writer->op_for_loop(k_i, BinaryOp::Less, k_tile, k_i, AssignmentOp::Increment, const_1,
            [&]()
            {
                //Initialize tiles
                // lhs_tile
                auto &a =
                    writer->declare_tile("a_leftover", ckw::TileInfo(to_ckw(_lhs->data_type()), m0, 1));
                // rhs_tile
                auto &b =
                    writer->declare_tile("b_leftover", ckw::TileInfo(to_ckw(_rhs->data_type()), n0, 1));
                writer->op_assign(a, const_0);
                writer->op_assign(b, const_0);

                // Loading the tiles
                // LHS
                lhs_sampler.x(x_i);
                lhs_sampler.y(y);
                lhs_sampler.z(z);
                lhs_sampler.b(const_0);
                writer->op_load(a, lhs->tensor(), lhs_sampler);

                // RHS
                auto &y_i = writer->declare_tile("y_i_leftover", ckw::DataType::Int32);
                writer->op_binary_expression(y_i, x, BinaryOp::Add, rhs_z);
                rhs_sampler.x(k_i);
                rhs_sampler.y(y_i);
                rhs_sampler.z(const_0);
                rhs_sampler.b(const_0);
                writer->op_load(b, rhs->tensor(), rhs_sampler);

                // Perform Matmul
                writer->op_binary_expression(dst_tile, a, BinaryOp::MatMul_Nt_T, b);
                writer->op_binary_expression(x_i, x_i, BinaryOp::Add, const_1);
            });
// *INDENT-ON*
        // clang-format on
    }
}

Window GpuCkwMatMul::get_window() const
{
    ARM_COMPUTE_ERROR_ON_MSG(_dst->tensor_shape().total_size() == 0U, "Destination tensor is not initialized");

    const int  m       = _dst->dimension(1);
    const int  n       = _dst->dimension(0);
    const bool adj_lhs = _attributes.adj_lhs();

    int m0 = adj_lhs ? adjust_vec_size(_settings.m0(), m) : std::min(_settings.m0(), m);
    int n0 = adjust_vec_size(_settings.n0(), n);

    // Configure kernel window
    Window win = calculate_max_window(_dst->tensor_shape(), Steps(n0, m0));
    win        = win.collapse(win, Window::DimZ);

    return win;
}

std::string GpuCkwMatMul::get_name(const ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);

    std::string kernel_name("mat_mul_native");

    const int m = _dst->dimension(1);
    const int n = _dst->dimension(0);
    const int k = _attributes.adj_lhs() ? _lhs->tensor_shape().y() : _lhs->tensor_shape().x();

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
