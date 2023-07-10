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
#include "GpuCkwCast.h"

#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwKernelWriter.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwScopedKernelWriter.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Validate.h"
#include "ckw/TensorTileSampler.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGroup.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwVariableTable.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/TypeConverter.h"
#include <string>

using namespace ckw;
namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
namespace
{
/** Create a simple sampler from tile of dimension [m0, n0]
 */
inline TensorTileSampler create_sampler(GpuCkwScopedKernelWriter &writer, int32_t m0, int32_t n0)
{
    TensorTileSampler sampler;

    auto &gid_0 = writer->declare_tile("gid_0", ckw::DataType::Int32);
    auto &gid_1 = writer->declare_tile("gid_1", ckw::DataType::Int32);
    auto &gid_2 = writer->declare_tile("gid_2", ckw::DataType::Int32);

    auto &const_0 = writer->declare_tile("0", 0);
    writer->op_get_global_id(gid_0, 0);
    writer->op_get_global_id(gid_1, 1);
    writer->op_get_global_id(gid_2, 2);

    auto &x_coord = writer->declare_tile("x_coord", ckw::DataType::Int32);
    auto &y_coord = writer->declare_tile("y_coord", ckw::DataType::Int32);
    auto &m0_t    = writer->declare_tile("m0", m0);
    auto &n0_t    = writer->declare_tile("n0", n0);
    writer->op_binary_expression(x_coord, gid_0, BinaryOp::Mul, n0_t);
    writer->op_binary_expression(y_coord, gid_1, BinaryOp::Mul, m0_t);

    sampler.x(x_coord);
    sampler.y(y_coord);
    sampler.z(const_0); // 3rd dimension collapsed with 2nd dimension
    sampler.b(gid_2);

    sampler.width(n0);
    sampler.height(m0);

    sampler.format(TensorSamplerFormat::C_WH_1); // 3rd dimension collapsed with 2nd dimension
    sampler.address_mode_x(TensorSamplerAddressModeX::None);
    sampler.address_mode_y(TensorSamplerAddressModeY::ClampToBorder);
    sampler.address_mode_z(TensorSamplerAddressModeZ::Skip); // Dimensions higher than 3 not supported yet

    return sampler;
}
} // namespace

GpuCkwCast::GpuCkwCast(ComponentId                      id,
                                                 const ArgumentPack<ITensorInfo> &tensors,
                                                 const Attributes                &attributes)
    : IGpuCkwComponentDriver{ id, tensors },
      _src{},
      _dst{},
      _attributes{ attributes }
{
    _src = this->tensors().get_const_tensor(TensorType::ACL_SRC_0);
    _dst = this->tensors().get_const_tensor(TensorType::ACL_DST_0);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);
}

void GpuCkwCast::write_component_code(const ComponentGroup &comp_group, GpuCkwVariableTable &vtable, GpuCkwScopedKernelWriter writer) const
{
    const auto         root_window = comp_group.get_root_component()->ckw_component_driver()->get_window();
    const unsigned int n0          = root_window.x().step();
    const unsigned int m0          = root_window.y().step();

    GpuCkwComponentArgument *src = vtable.declare_variable(comp_group, writer, _src, "src");
    GpuCkwComponentArgument *dst = vtable.declare_variable(comp_group, writer, _dst, "dst");

    // Load the source tile and prepare the sampler.
    if(!src->has_tile())
    {
        const auto sampler = create_sampler(writer, m0, n0);
        writer->op_load_once(src, sampler);
    }
    else
    {
        const auto &sampler = src->tile_sampler();
        writer->op_load_once(src, sampler);
    }

    const auto &src_tile = src->tile();
    const auto &sampler  = src->tile_sampler();

    // Prepare the output tile.
    if(!dst->has_tile())
    {
        // Get Target datatype and convert it to ckw::DataType.
        ckw::DataType target_dt =  dynamic_fusion::to_ckw(_attributes.data_type());

        // Create dst_tile based on src_tile dimensions and with target DataType.
        const TileInfo src_tile_info = src_tile.tile_info();
        const TileInfo dst_tile_info = TileInfo(target_dt, src_tile_info.height(), src_tile_info.width());

        // Declare dst_tile
        auto &tile = writer->declare_tile("dst_tile", dst_tile_info);
        dst->init_virtual_tensor(tile, sampler);
    }

    const auto &dst_tile = dst->tile();

    // Check if this op is cast-down or cast-up
    const size_t src_size  = data_size_from_type(_src->data_type());
    const size_t dst_size  = data_size_from_type(_dst->data_type());
    const bool   cast_down = (src_size >= dst_size);

    if(cast_down && is_data_type_quantized(_src->data_type()))
    {
        const auto &constant_x80 = writer->declare_tile("0x80", 0x80);
        writer->op_binary_expression(src_tile, src_tile, BinaryOp::BitwiseXOR, constant_x80);
    }

    ckw::ConvertPolicy convert_policy = ckw::ConvertPolicy::None;

    if(cast_down && (is_data_type_float(_src->data_type()) || _attributes.convert_policy() == ConvertPolicy::SATURATE))
    {
        convert_policy = ckw::ConvertPolicy::Saturate;
    }

    writer->op_cast_expression(dst_tile, src_tile, convert_policy);
}

Window GpuCkwCast::get_window() const
{
    ARM_COMPUTE_ERROR_ON_MSG(_dst->tensor_shape().total_size() == 0U, "Destination tensor is not initialized");

    TensorShape output_shape = _dst->tensor_shape();
    // Collapse Dim 1 (W) and Dim 2 (H) together, leave Dim 0 (C) unchanged
    // This is in line with the collapsing convention used by operators like Conv2d
    output_shape.collapse(2U, 1U);
    constexpr unsigned int vector_size_byte_opencl = 16;
    const unsigned int num_elems_processed_per_iteration = adjust_vec_size(vector_size_byte_opencl / _dst->element_size(), _dst->dimension(0));
    Window             win                               = calculate_max_window(output_shape, Steps(num_elems_processed_per_iteration));

    return win;
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
