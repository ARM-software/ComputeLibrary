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

#include "arm_compute/core/Error.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/utils/StringUtils.h"
#include "arm_compute/core/Validate.h"
#include "ckw/TensorTileSampler.h"
#include "ckw/types/TensorSamplerTypes.h"

#include "src/core/helpers/WindowHelpers.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/type_converter/Common.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/type_converter/ElementwiseBinary.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/WriterHelper.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwKernelWriter.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwScopedKernelWriter.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwVariableTable.h"
#include "src/dynamic_fusion/sketch/gpu/components/utils/type_printer/ElementwiseBinary.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelArgument.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGroup.h"
#include "support/StringSupport.h"

#include <algorithm>
#include <string>

using namespace ckw;
namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
GpuCkwElementwiseBinary::GpuCkwElementwiseBinary(ComponentId                      id,
                                                 const ArgumentPack<ITensorInfo> &tensors,
                                                 const Attributes                &attributes)
    : IGpuCkwComponentDriver{id, tensors}, _lhs{}, _rhs{}, _dst{}, _attributes{attributes}
{
    _lhs = this->tensors().get_const_tensor(TensorType::ACL_SRC_0);
    _rhs = this->tensors().get_const_tensor(TensorType::ACL_SRC_1);
    _dst = this->tensors().get_const_tensor(TensorType::ACL_DST_0);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_lhs, _rhs, _dst);
}

void GpuCkwElementwiseBinary::write_component_code(const ComponentGroup    &comp_group,
                                                   GpuCkwVariableTable     &vtable,
                                                   GpuCkwScopedKernelWriter writer) const
{
    const auto root_window = comp_group.get_root_component()->ckw_component_driver()->get_window();
    const auto n0          = static_cast<int32_t>(root_window.x().step());
    const auto m0          = static_cast<int32_t>(root_window.y().step());

    GpuCkwComponentArgument *lhs =
        vtable.declare_variable(comp_group, writer, _lhs, TensorStorageType::ClBufferUint8Ptr, "lhs");
    GpuCkwComponentArgument *rhs =
        vtable.declare_variable(comp_group, writer, _rhs, TensorStorageType::ClBufferUint8Ptr, "rhs");
    GpuCkwComponentArgument *dst =
        vtable.declare_variable(comp_group, writer, _dst, TensorStorageType::ClBufferUint8Ptr, "dst");

    auto &gid_0 = writer->declare_tile("gid_0", ckw::DataType::Int32);
    auto &gid_1 = writer->declare_tile("gid_1", ckw::DataType::Int32);
    auto &gid_2 = writer->declare_tile("gid_2", ckw::DataType::Int32);

    writer->op_get_global_id(gid_0, 0);
    writer->op_get_global_id(gid_1, 1);
    writer->op_get_global_id(gid_2, 2);

    auto &const_0 = writer->declare_tile("0", 0);

    // Load the LHS and RHS tiles
    if (!lhs->has_tile())
    {
        auto sampler = create_boundary_aware_2d_sampler(writer, gid_0, gid_1, _lhs->dimension(0), _lhs->dimension(1),
                                                        n0, m0, "lhs_", const_0);
        sampler.format(TensorSamplerFormat::C_WH_1); // 3rd dimension collapsed with 2nd dimension
        sampler.z(const_0);
        sampler.b(gid_2);
        writer->op_load_once(lhs, sampler);
    }
    if (!rhs->has_tile())
    {
        auto sampler = create_boundary_aware_2d_sampler(writer, gid_0, gid_1, _rhs->dimension(0), _rhs->dimension(1),
                                                        n0, m0, "rhs_", const_0);
        sampler.format(TensorSamplerFormat::C_WH_1); // 3rd dimension collapsed with 2nd dimension
        sampler.z(const_0);
        sampler.b(gid_2);
        writer->op_load_once(rhs, sampler);
    }

    auto dst_sampler = create_boundary_aware_2d_sampler(writer, gid_0, gid_1, _dst->dimension(0), _dst->dimension(1),
                                                        n0, m0, "dst_", const_0);
    dst_sampler.format(TensorSamplerFormat::C_WH_1); // 3rd dimension collapsed with 2nd dimension
    dst_sampler.z(const_0);
    dst_sampler.b(gid_2);

    // Prepare the output tile.
    if (!dst->has_tile())
    {
        auto &tile = writer->declare_tile(
            "dst_tile", ckw::TileInfo(to_ckw(_dst->data_type()), dst_sampler.height(), dst_sampler.width()));
        dst->init_virtual_tensor(tile, dst_sampler);
    }

    auto &lhs_tile = lhs->tile();
    auto &rhs_tile = rhs->tile();
    auto &dst_tile = dst->tile();

    // Perform the operation.
    writer->op_binary_expression(dst_tile, lhs_tile, to_ckw(_attributes), rhs_tile);
}

Window GpuCkwElementwiseBinary::get_window() const
{
    ARM_COMPUTE_ERROR_ON_MSG(_dst->tensor_shape().total_size() == 0U, "Destination tensor is not initialized");

    TensorShape output_shape = _dst->tensor_shape();
    // Collapse Dim 1 (W) and Dim 2 (H) together, leave Dim 0 (C) unchanged
    // This is in line with the collapsing convention used by operators like Conv2d
    output_shape.collapse(2U, 1U);
    constexpr unsigned int vector_size_byte_opencl = 16;
    const unsigned int     num_elems_processed_per_iteration =
        adjust_vec_size(vector_size_byte_opencl / _dst->element_size(), _dst->dimension(0));
    Window win = calculate_max_window(output_shape, Steps(num_elems_processed_per_iteration));

    return win;
}

std::string GpuCkwElementwiseBinary::get_name(const ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);
    const std::vector<std::string> build_params = {
        "elementwise_binary",
        "op",
        to_string(_attributes.operation()),
        "dt",
        lower_string(string_from_data_type(_dst->data_type())),
    };
    return join(build_params, "_");
}

std::string GpuCkwElementwiseBinary::get_tuner_id(const ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);
    /// NOTE: Hardcoded for now, the parameters should ideally be exported by ckw (a selection of constant tiles)
    std::vector<std::string> build_params = {
        "elementwise_binary",
        "op",
        to_string(_attributes.operation()),
        "dt",
        lower_string(string_from_data_type(_dst->data_type())),
        "dst_dim0",
        support::cpp11::to_string(_dst->dimension(0)),
        "dst_dim1",
        support::cpp11::to_string(_dst->dimension(1)),
    };
    return join(build_params, "_");
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
