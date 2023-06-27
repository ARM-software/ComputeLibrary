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
#include "GpuCkwElementwiseBinary.h"

#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwKernelWriter.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwScopedKernelWriter.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Validate.h"
#include "ckw/TensorTileSampler.h"
#include "ckw/types/TensorSamplerTypes.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGroup.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwVariableTable.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/WriterHelper.h"
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
inline TensorTileSampler create_simple_sampler(GpuCkwScopedKernelWriter &writer, int32_t m0, int32_t n0)
{
    TensorTileSampler sampler;

    auto &gid_0 = writer->declare_tile("gid_0", ckw::DataType::Int32);
    auto &gid_1 = writer->declare_tile("gid_1", ckw::DataType::Int32);
    auto &gid_2 = writer->declare_tile("gid_2", ckw::DataType::Int32);

    auto &const_0 = writer->declare_tile("0", 0);

    writer->op_get_global_id(gid_0, 0);
    writer->op_get_global_id(gid_1, 1);
    writer->op_get_global_id(gid_2, 2);

    sampler.x(gid_0);
    sampler.y(gid_1);
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

GpuCkwElementwiseBinary::GpuCkwElementwiseBinary(ComponentId                      id,
                                                 const ArgumentPack<ITensorInfo> &tensors,
                                                 const Attributes                &attributes)
    : IGpuCkwComponentDriver{ id, tensors },
      _lhs{},
      _rhs{},
      _dst{}
{
    ARM_COMPUTE_UNUSED(attributes);

    _lhs = this->tensors().get_const_tensor(TensorType::ACL_SRC_0);
    _rhs = this->tensors().get_const_tensor(TensorType::ACL_SRC_1);
    _dst = this->tensors().get_const_tensor(TensorType::ACL_DST_0);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_lhs, _rhs, _dst);
}

void GpuCkwElementwiseBinary::write_component_code(const ComponentGroup &comp_group, GpuCkwVariableTable &vtable, GpuCkwScopedKernelWriter writer) const
{
    const auto         root_window = comp_group.get_root_component()->ckw_component_driver()->get_window();
    const unsigned int n0          = root_window.x().step();
    const unsigned int m0          = root_window.y().step();

    GpuCkwComponentArgument *lhs = vtable.declare_variable(comp_group, writer, _lhs, "lhs");
    GpuCkwComponentArgument *rhs = vtable.declare_variable(comp_group, writer, _rhs, "rhs");
    GpuCkwComponentArgument *dst = vtable.declare_variable(comp_group, writer, _dst, "dst");

    // Load the LHS and RHS tiles and prepare the tensor sampler.
    load_lhs_rhs_tiles_and_prepare_sampler(writer, lhs, rhs, m0, n0, create_simple_sampler);

    auto       &lhs_tile = lhs->tile();
    auto       &rhs_tile = rhs->tile();
    const auto &sampler  = lhs->tile_sampler();

    // Prepare the output tile.
    if(!dst->has_tile())
    {
        auto &tile = writer->declare_tile("dst_tile", lhs_tile.tile_info());
        dst->init_virtual_tensor(tile, sampler);
    }

    auto &dst_tile = dst->tile();

    // Perform the operation.
    writer->op_binary_expression(dst_tile, lhs_tile, BinaryOp::Add, rhs_tile);
}

Window GpuCkwElementwiseBinary::get_window() const
{
    ARM_COMPUTE_ERROR_ON_MSG(_dst->tensor_shape().total_size() == 0U, "Destination tensor is not initialized");

    TensorShape output_shape = _dst->tensor_shape();
    // Collapse Dim 1 (W) and Dim 2 (H) together, leave Dim 0 (C) unchanged
    // This is in line with the collapsing convention used by operators like Conv2d
    output_shape.collapse(2U, 1U);
    // constexpr unsigned int vector_size_byte_opencl = 16;
    // const unsigned int num_elems_processed_per_iteration = adjust_vec_size(vector_size_byte_opencl / _dst->element_size(), _dst->dimension(0));
    const unsigned int num_elems_processed_per_iteration = 1U; // Hard-coded for now
    Window             win                               = calculate_max_window(output_shape, Steps(num_elems_processed_per_iteration));

    return win;
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
