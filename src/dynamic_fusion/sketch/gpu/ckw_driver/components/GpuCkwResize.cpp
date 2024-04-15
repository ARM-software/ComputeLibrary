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

#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/GpuCkwResize.h"

#include "arm_compute/core/utils/helpers/AdjustVecSize.h"

#include "src/core/helpers/WindowHelpers.h"
#include "src/core/utils/ScaleUtils.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/type_converter/Common.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/WriterHelper.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwKernelWriter.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwScopedKernelWriter.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwVariableTable.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelArgument.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGroup.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
namespace
{
constexpr unsigned int opencl_vector_size_in_bytes = 16;
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
    const size_t width_idx  = get_data_layout_dimension_index(_dst->data_layout(), DataLayoutDimension::WIDTH);
    const size_t height_idx = get_data_layout_dimension_index(_dst->data_layout(), DataLayoutDimension::HEIGHT);

    const Window  root_window = comp_group.get_root_component()->ckw_component_driver()->get_window();
    const int32_t n0          = root_window.x().step();
    const int32_t m0          = root_window.y().step();
    const int32_t partial_n0  = _dst->dimension(0) % n0;

    GpuCkwComponentArgument *src =
        vtable.declare_variable(comp_group, writer, _src, TensorStorageType::ClBufferUint8Ptr, "src");
    GpuCkwComponentArgument *dst =
        vtable.declare_variable(comp_group, writer, _dst, TensorStorageType::ClBufferUint8Ptr, "dst");

    // Constants
    const float scale_x = scale_utils::calculate_resize_ratio(_src->dimension(width_idx), _dst->dimension(width_idx),
                                                              _attributes.align_corners());
    const float scale_y = scale_utils::calculate_resize_ratio(_src->dimension(height_idx), _dst->dimension(height_idx),
                                                              _attributes.align_corners());
    const auto &tile_scale_x = writer->declare_tile("scale_x", scale_x);
    const auto &tile_scale_y = writer->declare_tile("scale_y", scale_y);
    const auto &tile_0       = writer->declare_tile("0", 0);
    const auto &tile_half    = writer->declare_tile("half", 0.5f);
    const auto &tile_1       = writer->declare_tile("1", 1);
    const auto &tile_src_w   = writer->declare_tile("src_w", static_cast<int32_t>(_src->dimension(width_idx)));
    const auto &tile_src_h   = writer->declare_tile("src_h", static_cast<int32_t>(_src->dimension(height_idx)));
    const auto &tile_dst_h   = writer->declare_tile("dst_h", static_cast<int32_t>(_dst->dimension(height_idx)));

    const auto &tile_gid_0 = writer->declare_tile("gid_0", ckw::DataType::Int32);
    const auto &tile_gid_1 = writer->declare_tile("gid_1", ckw::DataType::Int32);
    const auto &tile_gid_2 = writer->declare_tile("gid_2", ckw::DataType::Int32);

    writer->op_get_global_id(tile_gid_0, 0);
    writer->op_get_global_id(tile_gid_1, 1);
    writer->op_get_global_id(tile_gid_2, 2);

    auto &tile_co = writer->declare_tile("co", ckw::DataType::Int32); // OFM
    auto &tile_xo = writer->declare_tile("xo", ckw::DataType::Int32); // WIDTH
    auto &tile_yo = writer->declare_tile("yo", ckw::DataType::Int32); // HEIGHT
    auto &tile_bo = writer->declare_tile("bo", ckw::DataType::Int32); // BATCH SIZE IDX

    // Get the boundary aware coordinates at each global dimension index
    get_coord(writer, tile_co, tile_gid_0, n0, partial_n0, tile_co.name() + "_dim0_", tile_0);
    get_coord(writer, tile_xo, tile_gid_1, 1, 0, tile_xo.name() + "_dim1_", tile_0);
    get_coord(writer, tile_yo, tile_gid_2, 1, 0, tile_yo.name() + "_dim2_", tile_0);
    get_coord(writer, tile_bo, tile_gid_2, 1, 0, tile_yo.name() + "_dim3_", tile_0);

    writer->op_binary_expression(tile_yo, tile_yo, BinaryOp::Mod, tile_dst_h);
    writer->op_binary_expression(tile_bo, tile_bo, BinaryOp::Div, tile_dst_h);

    const auto &tile_xi_f = writer->declare_tile("xi_f", ckw::DataType::Fp32);
    const auto &tile_yi_f = writer->declare_tile("yi_f", ckw::DataType::Fp32);

    switch (_attributes.sampling_policy())
    {
        case SamplingPolicy::TOP_LEFT:
            // xi_f = (xo * scale_x)
            // yi_f = (yo * scale_y)
            writer->op_binary_expression(tile_xi_f, tile_xo, BinaryOp::Mul, tile_scale_x);
            writer->op_binary_expression(tile_yi_f, tile_yo, BinaryOp::Mul, tile_scale_y);
            break;
        case SamplingPolicy::CENTER:
        {
            // xi_f = ((xo + 0.5f) * scale_x)
            // yi_f = ((yo + 0.5f) * scale_y)
            const auto &tile_xo_plus_half = writer->declare_tile("xo_plus_half", ckw::DataType::Fp32);
            const auto &tile_yo_plus_half = writer->declare_tile("yo_plus_half", ckw::DataType::Fp32);

            writer->op_binary_expression(tile_xo_plus_half, tile_xo, BinaryOp::Add, tile_half);
            writer->op_binary_expression(tile_yo_plus_half, tile_yo, BinaryOp::Add, tile_half);

            writer->op_binary_expression(tile_xi_f, tile_xo_plus_half, BinaryOp::Mul, tile_scale_x);
            writer->op_binary_expression(tile_yi_f, tile_yo_plus_half, BinaryOp::Mul, tile_scale_y);
        }
        break;
        default:
            ARM_COMPUTE_ERROR("Unsupported sampling policy");
    }

    if (_attributes.align_corners())
    {
        writer->op_unary_elementwise_function(tile_xi_f, UnaryFunction::Round, tile_xi_f);
        writer->op_unary_elementwise_function(tile_yi_f, UnaryFunction::Round, tile_yi_f);
    }

    // xi0 = clamp((int)xi_f, 0, (int)src_w - 1)
    // yi0 = clamp((int)yi_f, 0, (int)src_h - 1)
    const auto &tile_xi_f_int = writer->declare_tile("xi_f_int", ckw::DataType::Int32);
    const auto &tile_yi_f_int = writer->declare_tile("yi_f_int", ckw::DataType::Int32);

    writer->op_cast_expression(tile_xi_f_int, tile_xi_f, ckw::ConvertPolicy::None);
    writer->op_cast_expression(tile_yi_f_int, tile_yi_f, ckw::ConvertPolicy::None);

    const auto &tile_src_w_minus_1 = writer->declare_tile("src_w_minus_1", ckw::DataType::Int32);
    const auto &tile_src_h_minus_1 = writer->declare_tile("src_h_minus_1", ckw::DataType::Int32);

    writer->op_binary_expression(tile_src_w_minus_1, tile_src_w, BinaryOp::Sub, tile_1);
    writer->op_binary_expression(tile_src_h_minus_1, tile_src_h, BinaryOp::Sub, tile_1);

    auto &tile_xi0 = writer->declare_tile("xi0", ckw::DataType::Int32);
    auto &tile_yi0 = writer->declare_tile("yi0", ckw::DataType::Int32);

    writer->op_ternary_elementwise_function(tile_xi0, TernaryFunction::Clamp, tile_xi_f_int, tile_0,
                                            tile_src_w_minus_1);
    writer->op_ternary_elementwise_function(tile_yi0, TernaryFunction::Clamp, tile_yi_f_int, tile_0,
                                            tile_src_h_minus_1);

    TensorTileSampler src_sampler;
    src_sampler.x(tile_co);
    src_sampler.y(tile_xi0);
    src_sampler.z(tile_yi0);
    src_sampler.b(tile_bo);
    src_sampler.height(m0);
    src_sampler.width(n0);
    // We guarantee to not have out-of-bounds accesses
    src_sampler.format(TensorSamplerFormat::C_W_H);
    src_sampler.address_mode_x(TensorSamplerAddressModeX::None);
    src_sampler.address_mode_y(TensorSamplerAddressModeY::None);
    src_sampler.address_mode_z(TensorSamplerAddressModeZ::None);

    writer->op_load_once(src, src_sampler);
    auto &tile_src = src->tile();

    TensorTileSampler dst_sampler;
    dst_sampler.x(tile_co);
    dst_sampler.y(tile_xo);
    dst_sampler.z(tile_yo);
    dst_sampler.b(tile_bo);
    dst_sampler.height(m0);
    dst_sampler.width(n0);
    dst_sampler.format(TensorSamplerFormat::C_W_H);
    // Do not write to the same memory location with multiple threads
    dst_sampler.address_mode_x(TensorSamplerAddressModeX::OverlappingMin);
    dst_sampler.address_mode_y(TensorSamplerAddressModeY::None);
    dst_sampler.address_mode_z(TensorSamplerAddressModeZ::None);

    auto &tile_dst = writer->declare_tile("dst", TileInfo(to_ckw(_dst->data_type()), m0, n0));
    dst->init_virtual_tensor(tile_dst, dst_sampler);

    writer->op_assign(tile_dst, tile_src);
}

void GpuCkwResize::do_bilinear_resize(const ComponentGroup    &comp_group,
                                      GpuCkwVariableTable     &vtable,
                                      GpuCkwScopedKernelWriter writer) const
{
    const size_t width_idx  = get_data_layout_dimension_index(_dst->data_layout(), DataLayoutDimension::WIDTH);
    const size_t height_idx = get_data_layout_dimension_index(_dst->data_layout(), DataLayoutDimension::HEIGHT);

    const Window  root_window = comp_group.get_root_component()->ckw_component_driver()->get_window();
    const int32_t n0          = root_window.x().step();
    const int32_t m0          = root_window.y().step();
    const int32_t partial_n0  = _dst->dimension(0) % n0;

    GpuCkwComponentArgument *src =
        vtable.declare_variable(comp_group, writer, _src, TensorStorageType::ClBufferUint8Ptr, "src");
    GpuCkwComponentArgument *dst =
        vtable.declare_variable(comp_group, writer, _dst, TensorStorageType::ClBufferUint8Ptr, "dst");

    // Constants
    const float scale_x = scale_utils::calculate_resize_ratio(_src->dimension(width_idx), _dst->dimension(width_idx),
                                                              _attributes.align_corners());
    const float scale_y = scale_utils::calculate_resize_ratio(_src->dimension(height_idx), _dst->dimension(height_idx),
                                                              _attributes.align_corners());
    const auto &tile_scale_x = writer->declare_tile("scale_x", scale_x);
    const auto &tile_scale_y = writer->declare_tile("scale_y", scale_y);
    const auto &tile_0       = writer->declare_tile("0", 0);
    const auto &tile_half    = writer->declare_tile("half", 0.5f);
    const auto &tile_1       = writer->declare_tile("1", 1);
    const auto &tile_src_w   = writer->declare_tile("src_w", static_cast<int32_t>(_src->dimension(width_idx)));
    const auto &tile_src_h   = writer->declare_tile("src_h", static_cast<int32_t>(_src->dimension(height_idx)));
    const auto &tile_dst_h   = writer->declare_tile("dst_h", static_cast<int32_t>(_dst->dimension(height_idx)));

    const auto &tile_gid_0 = writer->declare_tile("gid_0", ckw::DataType::Int32);
    const auto &tile_gid_1 = writer->declare_tile("gid_1", ckw::DataType::Int32);
    const auto &tile_gid_2 = writer->declare_tile("gid_2", ckw::DataType::Int32);

    writer->op_get_global_id(tile_gid_0, 0);
    writer->op_get_global_id(tile_gid_1, 1);
    writer->op_get_global_id(tile_gid_2, 2);

    auto &tile_co = writer->declare_tile("co", ckw::DataType::Int32); // OFM
    auto &tile_xo = writer->declare_tile("xo", ckw::DataType::Int32); // WIDTH
    auto &tile_yo = writer->declare_tile("yo", ckw::DataType::Int32); // HEIGHT
    auto &tile_bo = writer->declare_tile("bo", ckw::DataType::Int32); // BATCH SIZE IDX

    // Get the boundary aware coordinates at each global dimension index
    get_coord(writer, tile_co, tile_gid_0, n0, partial_n0, tile_co.name() + "_dim0_", tile_0);
    get_coord(writer, tile_xo, tile_gid_1, 1, 0, tile_xo.name() + "_dim1_", tile_0);
    get_coord(writer, tile_yo, tile_gid_2, 1, 0, tile_yo.name() + "_dim2_", tile_0);
    get_coord(writer, tile_bo, tile_gid_2, 1, 0, tile_yo.name() + "_dim3_", tile_0);

    // yo = coord_dim2 % dst_h
    // bo = coord_dim2 / dst_h
    writer->op_binary_expression(tile_yo, tile_yo, BinaryOp::Mod, tile_dst_h);
    writer->op_binary_expression(tile_bo, tile_bo, BinaryOp::Div, tile_dst_h);

    const auto &tile_xi_f = writer->declare_tile("xi_f", ckw::DataType::Fp32);
    const auto &tile_yi_f = writer->declare_tile("yi_f", ckw::DataType::Fp32);

    switch (_attributes.sampling_policy())
    {
        case SamplingPolicy::TOP_LEFT:
            // xi_f = (xo * scale_x)
            // yi_f = (yo * scale_y)
            writer->op_binary_expression(tile_xi_f, tile_xo, BinaryOp::Mul, tile_scale_x);
            writer->op_binary_expression(tile_yi_f, tile_yo, BinaryOp::Mul, tile_scale_y);
            break;
        case SamplingPolicy::CENTER:
        {
            // xi_f = ((xo + 0.5f) * scale_x - 0.5f)
            // yi_f = ((yo + 0.5f) * scale_y - 0.5f)
            const auto &tile_xo_plus_half = writer->declare_tile("xo_plus_half", ckw::DataType::Fp32);
            const auto &tile_yo_plus_half = writer->declare_tile("yo_plus_half", ckw::DataType::Fp32);
            writer->op_binary_expression(tile_xo_plus_half, tile_xo, BinaryOp::Add, tile_half);
            writer->op_binary_expression(tile_yo_plus_half, tile_yo, BinaryOp::Add, tile_half);

            writer->op_binary_expression(tile_xi_f, tile_xo_plus_half, BinaryOp::Mul, tile_scale_x);
            writer->op_binary_expression(tile_yi_f, tile_yo_plus_half, BinaryOp::Mul, tile_scale_y);

            writer->op_binary_expression(tile_xi_f, tile_xi_f, BinaryOp::Sub, tile_half);
            writer->op_binary_expression(tile_yi_f, tile_yi_f, BinaryOp::Sub, tile_half);
        }
        break;
        default:
            ARM_COMPUTE_ERROR("Unsupported sampling policy");
    }

    // xi = (int)floor(xi_f);
    // yi = (int)floor(yi_f);
    const auto &tile_xi_f_floor = writer->declare_tile("xi_f_floor", ckw::DataType::Fp32);
    const auto &tile_yi_f_floor = writer->declare_tile("yi_f_floor", ckw::DataType::Fp32);
    writer->op_unary_elementwise_function(tile_xi_f_floor, UnaryFunction::Floor, tile_xi_f);
    writer->op_unary_elementwise_function(tile_yi_f_floor, UnaryFunction::Floor, tile_yi_f);

    const auto &tile_xi = writer->declare_tile("xi", ckw::DataType::Int32);
    const auto &tile_yi = writer->declare_tile("yi", ckw::DataType::Int32);
    writer->op_cast_expression(tile_xi, tile_xi_f_floor, ckw::ConvertPolicy::None);
    writer->op_cast_expression(tile_yi, tile_yi_f_floor, ckw::ConvertPolicy::None);

    // xi0  = clamp(xi, 0, (int)src_w - 1);
    // yi0  = clamp(yi, 0, (int)src_h - 1);
    // xi1  = clamp(xi + 1, 0, (int)src_w - 1);
    // yi1  = clamp(yi + 1, 0, (int)src_h - 1);
    const auto &tile_src_w_minus_1 = writer->declare_tile("src_w_minus_1", ckw::DataType::Int32);
    const auto &tile_src_h_minus_1 = writer->declare_tile("src_h_minus_1", ckw::DataType::Int32);
    writer->op_binary_expression(tile_src_w_minus_1, tile_src_w, BinaryOp::Sub, tile_1);
    writer->op_binary_expression(tile_src_h_minus_1, tile_src_h, BinaryOp::Sub, tile_1);

    const auto &tile_xi_plus_1 = writer->declare_tile("xi_plus_1", ckw::DataType::Int32);
    const auto &tile_yi_plus_1 = writer->declare_tile("yi_plus_1", ckw::DataType::Int32);
    writer->op_binary_expression(tile_xi_plus_1, tile_xi, BinaryOp::Add, tile_1);
    writer->op_binary_expression(tile_yi_plus_1, tile_yi, BinaryOp::Add, tile_1);

    auto &tile_xi0 = writer->declare_tile("xi0", ckw::DataType::Int32);
    auto &tile_yi0 = writer->declare_tile("yi0", ckw::DataType::Int32);
    auto &tile_xi1 = writer->declare_tile("xi1", ckw::DataType::Int32);
    auto &tile_yi1 = writer->declare_tile("yi1", ckw::DataType::Int32);

    writer->op_ternary_elementwise_function(tile_xi0, TernaryFunction::Clamp, tile_xi, tile_0, tile_src_w_minus_1);
    writer->op_ternary_elementwise_function(tile_yi0, TernaryFunction::Clamp, tile_yi, tile_0, tile_src_h_minus_1);
    writer->op_ternary_elementwise_function(tile_xi1, TernaryFunction::Clamp, tile_xi_plus_1, tile_0,
                                            tile_src_w_minus_1);
    writer->op_ternary_elementwise_function(tile_yi1, TernaryFunction::Clamp, tile_yi_plus_1, tile_0,
                                            tile_src_h_minus_1);

    TensorTileSampler in_sampler;
    in_sampler.x(tile_co);
    in_sampler.b(tile_bo);
    in_sampler.height(1);
    in_sampler.width(n0);
    // We guarantee to not have out-of-bounds accesses
    in_sampler.format(TensorSamplerFormat::C_W_H);
    in_sampler.address_mode_x(TensorSamplerAddressModeX::None);
    in_sampler.address_mode_y(TensorSamplerAddressModeY::None);
    in_sampler.address_mode_z(TensorSamplerAddressModeZ::None);

    TensorTileSampler in00_sampler = in_sampler;
    in00_sampler.y(tile_xi0);
    in00_sampler.z(tile_yi0);

    TensorTileSampler in01_sampler = in_sampler;
    in01_sampler.y(tile_xi1);
    in01_sampler.z(tile_yi0);

    TensorTileSampler in10_sampler = in_sampler;
    in10_sampler.y(tile_xi0);
    in10_sampler.z(tile_yi1);

    TensorTileSampler in11_sampler = in_sampler;
    in11_sampler.y(tile_xi1);
    in11_sampler.z(tile_yi1);

    auto &tile_in00 = writer->declare_tile("in00", TileInfo(to_ckw(_src->data_type()), 1, n0));
    auto &tile_in01 = writer->declare_tile("in01", TileInfo(to_ckw(_src->data_type()), 1, n0));
    auto &tile_in10 = writer->declare_tile("in10", TileInfo(to_ckw(_src->data_type()), 1, n0));
    auto &tile_in11 = writer->declare_tile("in11", TileInfo(to_ckw(_src->data_type()), 1, n0));

    writer->op_load(tile_in00, src->tensor(), in00_sampler);
    writer->op_load(tile_in01, src->tensor(), in01_sampler);
    writer->op_load(tile_in10, src->tensor(), in10_sampler);
    writer->op_load(tile_in11, src->tensor(), in11_sampler);

    TensorTileSampler dst_sampler;
    dst_sampler.x(tile_co);
    dst_sampler.y(tile_xo);
    dst_sampler.z(tile_yo);
    dst_sampler.b(tile_bo);
    dst_sampler.height(m0);
    dst_sampler.width(n0);
    dst_sampler.format(TensorSamplerFormat::C_W_H);
    // Do not write to the same memory location with multiple threads
    dst_sampler.address_mode_x(TensorSamplerAddressModeX::OverlappingMin);
    dst_sampler.address_mode_y(TensorSamplerAddressModeY::None);
    dst_sampler.address_mode_z(TensorSamplerAddressModeZ::None);

    auto &tile_dst = writer->declare_tile("dst", TileInfo(to_ckw(_dst->data_type()), m0, n0));
    dst->init_virtual_tensor(tile_dst, dst_sampler);

    // Weights of each nearest pixel
    const auto &tile_a  = writer->declare_tile("a", ckw::DataType::Fp32);
    const auto &tile_b  = writer->declare_tile("b", ckw::DataType::Fp32);
    const auto &tile_a1 = writer->declare_tile("a1", ckw::DataType::Fp32);
    const auto &tile_b1 = writer->declare_tile("b1", ckw::DataType::Fp32);

    // a = (xi_f - (float)xi)
    // b = (1.f - a)
    // a1 = (yi_f - (float)yi)
    // b1 = (1.f - a1)
    const auto &tile_xi_float = writer->declare_tile("xi_float", ckw::DataType::Fp32);
    const auto &tile_yi_float = writer->declare_tile("yi_float", ckw::DataType::Fp32);
    writer->op_cast_expression(tile_xi_float, tile_xi, ckw::ConvertPolicy::None);
    writer->op_cast_expression(tile_yi_float, tile_yi, ckw::ConvertPolicy::None);

    writer->op_binary_expression(tile_a, tile_xi_f, BinaryOp::Sub, tile_xi_float);
    writer->op_binary_expression(tile_b, tile_1, BinaryOp::Sub, tile_a);
    writer->op_binary_expression(tile_a1, tile_yi_f, BinaryOp::Sub, tile_yi_float);
    writer->op_binary_expression(tile_b1, tile_1, BinaryOp::Sub, tile_a1);

    if (is_data_type_float(_src->data_type()))
    {
        // Cast weights to source type
        const auto &tile_a_src_type  = writer->declare_tile("a_src_t", to_ckw(_src->data_type()));
        const auto &tile_b_src_type  = writer->declare_tile("b_src_t", to_ckw(_src->data_type()));
        const auto &tile_a1_src_type = writer->declare_tile("a1_src_t", to_ckw(_src->data_type()));
        const auto &tile_b1_src_type = writer->declare_tile("b1_src_t", to_ckw(_src->data_type()));

        writer->op_cast_expression(tile_a_src_type, tile_a, ckw::ConvertPolicy::None);
        writer->op_cast_expression(tile_b_src_type, tile_b, ckw::ConvertPolicy::None);
        writer->op_cast_expression(tile_a1_src_type, tile_a1, ckw::ConvertPolicy::None);
        writer->op_cast_expression(tile_b1_src_type, tile_b1, ckw::ConvertPolicy::None);

        // in00 * b * b1
        writer->op_binary_expression(tile_in00, tile_in00, BinaryOp::Mul, tile_b_src_type);
        writer->op_binary_expression(tile_in00, tile_in00, BinaryOp::Mul, tile_b1_src_type);

        // in01 * a * b1
        writer->op_binary_expression(tile_in01, tile_in01, BinaryOp::Mul, tile_a_src_type);
        writer->op_binary_expression(tile_in01, tile_in01, BinaryOp::Mul, tile_b1_src_type);

        // in10 * b * a1
        writer->op_binary_expression(tile_in10, tile_in10, BinaryOp::Mul, tile_b_src_type);
        writer->op_binary_expression(tile_in10, tile_in10, BinaryOp::Mul, tile_a1_src_type);

        // in11 * a * a1
        writer->op_binary_expression(tile_in11, tile_in11, BinaryOp::Mul, tile_a_src_type);
        writer->op_binary_expression(tile_in11, tile_in11, BinaryOp::Mul, tile_a1_src_type);

        // Summation of above terms
        writer->op_assign(tile_dst, tile_in00);
        writer->op_binary_expression(tile_dst, tile_dst, BinaryOp::Add, tile_in01);
        writer->op_binary_expression(tile_dst, tile_dst, BinaryOp::Add, tile_in10);
        writer->op_binary_expression(tile_dst, tile_dst, BinaryOp::Add, tile_in11);
    }
    else
    {
        // Cast to float
        const auto &tile_in00_f = writer->declare_tile("in00_f", TileInfo(ckw::DataType::Fp32, 1, n0));
        const auto &tile_in01_f = writer->declare_tile("in01_f", TileInfo(ckw::DataType::Fp32, 1, n0));
        const auto &tile_in10_f = writer->declare_tile("in10_f", TileInfo(ckw::DataType::Fp32, 1, n0));
        const auto &tile_in11_f = writer->declare_tile("in11_f", TileInfo(ckw::DataType::Fp32, 1, n0));
        writer->op_cast_expression(tile_in00_f, tile_in00, ckw::ConvertPolicy::None);
        writer->op_cast_expression(tile_in01_f, tile_in01, ckw::ConvertPolicy::None);
        writer->op_cast_expression(tile_in10_f, tile_in10, ckw::ConvertPolicy::None);
        writer->op_cast_expression(tile_in11_f, tile_in11, ckw::ConvertPolicy::None);

        // in00 * b * b1
        writer->op_binary_expression(tile_in00_f, tile_in00_f, BinaryOp::Mul, tile_b);
        writer->op_binary_expression(tile_in00_f, tile_in00_f, BinaryOp::Mul, tile_b1);

        // in01 * a * b1
        writer->op_binary_expression(tile_in01_f, tile_in01_f, BinaryOp::Mul, tile_a);
        writer->op_binary_expression(tile_in01_f, tile_in01_f, BinaryOp::Mul, tile_b1);

        // in10 * b * a1
        writer->op_binary_expression(tile_in10_f, tile_in10_f, BinaryOp::Mul, tile_b);
        writer->op_binary_expression(tile_in10_f, tile_in10_f, BinaryOp::Mul, tile_a1);

        // in11 * a * a1
        writer->op_binary_expression(tile_in11_f, tile_in11_f, BinaryOp::Mul, tile_a);
        writer->op_binary_expression(tile_in11_f, tile_in11_f, BinaryOp::Mul, tile_a1);

        // Summation of above terms
        writer->op_binary_expression(tile_in00_f, tile_in00_f, BinaryOp::Add, tile_in01_f);
        writer->op_binary_expression(tile_in00_f, tile_in00_f, BinaryOp::Add, tile_in10_f);
        writer->op_binary_expression(tile_in00_f, tile_in00_f, BinaryOp::Add, tile_in11_f);

        // Cast to destination type with saturation
        writer->op_cast_expression(tile_dst, tile_in00_f, ckw::ConvertPolicy::Saturate);
    }
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

    const unsigned int n0  = adjust_vec_size(opencl_vector_size_in_bytes / _src->element_size(), _src->dimension(0));
    Window             win = calculate_max_window(*_dst, Steps(n0));
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
