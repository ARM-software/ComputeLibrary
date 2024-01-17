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
#include "GpuCkwActivation.h"

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

GpuCkwActivation::GpuCkwActivation(ComponentId                      id,
                                   const ArgumentPack<ITensorInfo> &tensors,
                                   const Attributes                &attributes) // NOLINT
    : IGpuCkwComponentDriver{id, tensors}, _src{}, _dst{}, _attributes{attributes}
{
    _src = this->tensors().get_const_tensor(TensorType::ACL_SRC_0);
    _dst = this->tensors().get_const_tensor(TensorType::ACL_DST_0);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);
}

void GpuCkwActivation::write_component_code(const ComponentGroup    &comp_group,
                                            GpuCkwVariableTable     &vtable,
                                            GpuCkwScopedKernelWriter writer) const
{
    /********************************************************************************
     * 1 - Define tensors
     ********************************************************************************/
    GpuCkwComponentArgument *src = vtable.declare_variable(comp_group, writer, _src, "src");
    GpuCkwComponentArgument *dst = vtable.declare_variable(comp_group, writer, _dst, "dst");

    /********************************************************************************
     * 2 - Define CKW constants
     ********************************************************************************/
    const auto dst_h  = static_cast<int32_t>(_dst->dimension(1));
    const auto dst_dt = to_ckw(_dst->data_type());

    // CKW constants
    auto const_dst_h_i32 = writer->declare_constant_tile(ckw::ConstantData({{dst_h}}, ckw::DataType::Int32));
    auto const_pos_1_i32 = writer->declare_constant_tile(ckw::ConstantData({{1}}, ckw::DataType::Int32));
    auto const_0_i32     = writer->declare_constant_tile(ckw::ConstantData({{0}}, ckw::DataType::Int32));
    auto const_neg_1_fp  = writer->declare_constant_tile(ckw::ConstantData({{-1.0f}}, dst_dt));
    auto const_pos_1_fp  = writer->declare_constant_tile(ckw::ConstantData({{1.0f}}, dst_dt));
    auto const_0_fp      = writer->declare_constant_tile(ckw::ConstantData({{0.0f}}, dst_dt));
    auto const_A_fp      = writer->declare_constant_tile(ckw::ConstantData({{_attributes.a()}}, dst_dt));
    auto const_B_fp      = writer->declare_constant_tile(ckw::ConstantData({{_attributes.b()}}, dst_dt));

    /********************************************************************************
     * 3 - Define the compute block parameters and destination tile (if not root component)
     *     Bind the tile to the tensor to share it among different components and
     *     initialize the compute block parameters
     ********************************************************************************/
    // The compute block parameters depend on the employed tensor format

    // Destination compute block size
    int32_t dst_n0 = -1;
    int32_t dst_m0 = -1;

    // Destination compute block size left-over
    int32_t dst_n0_partial = -1;
    int32_t dst_m0_partial = -1;

    // Shift-back for the overlapping-min strategy
    int32_t dst_shift_back = -1;

    if (!dst->has_tile())
    {
        // If ROOT component, we use ckw::TensorSamplerFormat::Dim0_Dim1xDim2_1
        // as tensor format
        const auto root_window = comp_group.get_root_component()->ckw_component_driver()->get_window();

        dst_n0         = root_window.x().step();
        dst_m0         = root_window.y().step();
        dst_n0_partial = _dst->dimension(0) % dst_n0;
        dst_m0_partial = (_dst->dimension(1) * _dst->dimension(2)) % dst_m0;
        dst_shift_back = (dst_n0 - dst_n0_partial) % dst_n0;

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

        // Bind tile to the tensor
        dst->init_virtual_tensor(tile_dst, sampler_dst);
    }
    else
    {
        // dst_m0_partial depends on the TensorSamplerFormat
        dst_n0         = dst->tile().tile_info().width();
        dst_m0         = dst->tile().tile_info().height();
        dst_n0_partial = _dst->dimension(0) % dst_n0;

        ckw::TensorSampler sampler_dst = dst->tensor_sampler();

        if (sampler_dst.format() == ckw::TensorSamplerFormat::Dim0_Dim1xDim2_1)
        {
            dst_m0_partial = (_dst->dimension(1) * _dst->dimension(2)) % dst_m0;
        }
        else if (sampler_dst.format() == ckw::TensorSamplerFormat::Dim0_Dim1_Dim2)
        {
            dst_m0_partial = _dst->dimension(1) % dst_m0;
        }

        // Shift-back for the overlapping-min strategy
        dst_shift_back = (dst_n0 - dst_n0_partial) % dst_n0;
    }

    const auto &tile_dst = dst->tile();

    /********************************************************************************
     * 4 - Define the compute block parameters CKW constants
     ********************************************************************************/
    // Only now we can declare the N0 and M0 as constant
    auto const_dst_n0 = writer->declare_constant_tile(ckw::ConstantData({{dst_n0}}, ckw::DataType::Int32));
    auto const_dst_m0 = writer->declare_constant_tile(ckw::ConstantData({{dst_m0}}, ckw::DataType::Int32));
    auto const_dst_shift_back_n0 =
        writer->declare_constant_tile(ckw::ConstantData({{dst_shift_back}}, ckw::DataType::Int32));

    /********************************************************************************
     * 5 - Define the sampler for the input tensor
     ********************************************************************************/
    if (!src->has_tile())
    {
        // Sampler
        ckw::TensorSampler sampler_src = dst->tensor_sampler();

        auto tile_gid_0 = writer->declare_tile("gid_0_src", ckw::TileInfo(ckw::DataType::Int32));
        auto tile_gid_1 = writer->declare_tile("gid_1_src", ckw::TileInfo(ckw::DataType::Int32));
        auto tile_gid_2 = writer->declare_tile("gid_2_src", ckw::TileInfo(ckw::DataType::Int32));

        writer->op_get_global_id(tile_gid_0, 0);
        writer->op_get_global_id(tile_gid_1, 1);
        writer->op_get_global_id(tile_gid_2, 2);

        auto tile_nout0 = writer->declare_tile("nout0_src", ckw::TileInfo(ckw::DataType::Int32)); // OFM
        auto tile_mout0 =
            writer->declare_tile("mout0_src", ckw::TileInfo(ckw::DataType::Int32)); // WIDTH or WIDTH x HEIGHT
        auto tile_mout1 = writer->declare_tile("mout1_src", ckw::TileInfo(ckw::DataType::Int32)); // HEIGHT or 0
        auto tile_bout0 = writer->declare_tile("bout0_src", ckw::TileInfo(ckw::DataType::Int32)); // BATCH SIZE IDX

        get_coordinate_from_gws_overlapping_min(writer, tile_nout0, tile_gid_0, const_dst_n0, const_dst_shift_back_n0,
                                                const_0_i32);
        get_coordinate_from_gws(writer, tile_mout0, tile_gid_1, const_dst_m0);

        // Get the boundary aware coordinates at each global dimension index
        if (sampler_src.format() == ckw::TensorSamplerFormat::Dim0_Dim1xDim2_1)
        {
            writer->op_assign(tile_mout1, const_0_i32);
            get_coordinate_from_gws(writer, tile_bout0, tile_gid_2, const_pos_1_i32);
        }
        else if (sampler_src.format() == ckw::TensorSamplerFormat::Dim0_Dim1_Dim2)
        {
            writer->op_binary(tile_mout1, ckw::BinaryOp::Mod, tile_gid_2, const_dst_h_i32);
            writer->op_binary(tile_bout0, ckw::BinaryOp::Div, tile_gid_2, const_dst_h_i32);
        }

        auto tile_src = writer->declare_tile("src", ckw::TileInfo(dst_dt, dst_m0, dst_n0));

        writer->op_load(tile_src, src->tensor(), sampler_src, tile_nout0, tile_mout0, tile_mout1, tile_bout0);

        // Here, init_virtual_tensor() it is used to bring the tile_src outside the compound statement
        src->init_virtual_tensor(tile_src, sampler_src);
    }

    const auto &tile_src = src->tile();

    /********************************************************************************
     * 7 - Write the rest of the code
     ********************************************************************************/
    switch (_attributes.activation())
    {
        case ActivationLayerInfo::ActivationFunction::LOGISTIC:
        {
            // dst = src * -1
            writer->op_binary(tile_dst, ckw::BinaryOp::Mul, tile_src, const_neg_1_fp);
            // dst = exp(src * -1)
            writer->op_unary(tile_dst, ckw::UnaryOp::Exp, tile_dst);
            // dst = 1 + (exp(src * -1))
            writer->op_binary(tile_dst, ckw::BinaryOp::Add, tile_dst, const_pos_1_fp);
            // dst = 1 /  1 + (exp(src * -1))
            writer->op_binary(tile_dst, ckw::BinaryOp::Div, const_pos_1_fp, tile_dst);
            break;
        }
        case ActivationLayerInfo::ActivationFunction::TANH:
        {
            // dst = B_VAL * src
            writer->op_binary(tile_dst, ckw::BinaryOp::Mul, tile_src, const_B_fp);
            // dst = tanh(B_VAL * src)
            writer->op_unary(tile_dst, ckw::UnaryOp::Tanh, tile_dst);
            // dst = A_VAL * tanh(B_VAL * src)
            writer->op_binary(tile_dst, ckw::BinaryOp::Mul, tile_dst, const_A_fp);
            break;
        }
        case ActivationLayerInfo::ActivationFunction::RELU:
        {
            // dst = max(src, 0)
            writer->op_binary(tile_dst, ckw::BinaryOp::Max, tile_src, const_0_fp);
            break;
        }
        case ActivationLayerInfo::ActivationFunction::BOUNDED_RELU:
        {
            //dst = max(src, 0)
            writer->op_binary(tile_dst, ckw::BinaryOp::Max, tile_src, const_0_fp);
            //dst = min(max(src, 0), A_VAL)
            writer->op_binary(tile_dst, ckw::BinaryOp::Min, tile_dst, const_A_fp);
            break;
        }
        case ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU:
        {
            //dst = max(src, B_VAL)
            writer->op_binary(tile_dst, ckw::BinaryOp::Max, tile_src, const_B_fp);
            //dst = min(max(src, B_VAL), A_VAL)
            writer->op_binary(tile_dst, ckw::BinaryOp::Min, tile_dst, const_A_fp);
            break;
        }
        default:
            CKW_ASSERT(false);
            break;
    }
    ARM_COMPUTE_ERROR_ON_MSG(dst->has_tile() == false, "You must bind a tile before appending another component");
}

Window GpuCkwActivation::get_window() const
{
    ARM_COMPUTE_ERROR_ON_MSG(_dst->tensor_shape().total_size() == 0U, "Destination tensor is not initialized");

    TensorShape output_shape = _dst->tensor_shape();
    // Collapse Dim 1 (W) and Dim 2 (H) together, leave Dim 0 (C) unchanged
    // This is in line with the collapsing convention used by operators like Conv2d
    output_shape.collapse(2U, 1U);
    constexpr uint32_t vector_size_byte_opencl = 16;
    const uint32_t     num_elems_processed_per_iteration =
        adjust_vec_size(vector_size_byte_opencl / _dst->element_size(), _dst->dimension(0));
    Window win = calculate_max_window(output_shape, Steps(num_elems_processed_per_iteration));

    return win;
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
