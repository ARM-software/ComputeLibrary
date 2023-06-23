/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#include "ClTemplateElementwiseBinary.h"

#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGroup.h"
#include "src/dynamic_fusion/sketch/gpu/components/cl/ClComponentElementwiseBinary.h"

#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/StringUtils.h"
#include "src/core/helpers/WindowHelpers.h"

#include "support/StringSupport.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
constexpr unsigned int vector_size_byte_opencl = 16;

ClTemplateElementwiseBinary::ClTemplateElementwiseBinary(ComponentId                      id,
                                                         const ArgumentPack<ITensorInfo> &tensors,
                                                         const Attributes                &attributes)
    : IGpuTemplateComponentWriter{ id, tensors },
      _lhs{},
      _rhs{},
      _dst{},
      _attributes{ attributes }
{
    _lhs = this->tensors().get_const_tensor(TensorType::ACL_SRC_0);
    _rhs = this->tensors().get_const_tensor(TensorType::ACL_SRC_1);
    _dst = this->tensors().get_const_tensor(TensorType::ACL_DST_0);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_lhs, _rhs, _dst);
}

std::string ClTemplateElementwiseBinary::get_name() const
{
    return "elementwise_binary";
}

std::string ClTemplateElementwiseBinary::get_component_code(const ComponentGroup &comp_group) const
{
    std::string code;
    const bool  is_root      = (comp_group.get_root_component()->id() == this->id());
    const bool  is_lhs_input = comp_group.is_input_tensor(_lhs);
    const bool  is_rhs_input = comp_group.is_input_tensor(_rhs);

    code =
R"_(
    //------------------ START KERNEL {{meta_kernel_id}} {{ELTWISE_OP}} ---------------------
)_";

    if(is_root)
    {
        code +=
R"_(
    TILE(uint, M0, 1, g_dst_indirect_y);
)_";
    }

    if(is_lhs_input)
    {
        code +=
R"_(
    TILE({{DATA_TYPE}}, {{lhs_m0}}, N0, {{lhs}});
)_";
    }

    if(is_rhs_input)
    {
        code +=
R"_(
    TILE({{DATA_TYPE}}, {{rhs_m0}}, N0, {{rhs}});
)_";
    }

    code +=
R"_(
    {
)_";

    if(is_lhs_input)
    {
        code +=
R"_(
        {{lhs}}_offset_first_element_in_bytes += g_ind_2 * {{lhs}}_stride_w;
        T_LOAD({{DATA_TYPE}}, {{lhs_m0}}, {{lhs_n0}}, BUFFER, {{lhs}}, {{lhs_start_ind_0}}, {{lhs_start_ind_1}}, 1, {{lhs}}_stride_y, {{lhs}});
)_";
    }

    if(is_rhs_input)
    {
        code +=
R"_(
        {{rhs}}_offset_first_element_in_bytes += g_ind_2 * {{rhs}}_stride_w;
        T_LOAD({{DATA_TYPE}}, {{rhs_m0}}, {{rhs_n0}}, BUFFER, {{rhs}}, {{rhs_start_ind_0}}, {{rhs_start_ind_1}}, 1, {{rhs}}_stride_y, {{rhs}});
)_";
    }

    code +=
R"_(
        T_ELTWISE_{{BROADCAST_OP}}{{ELTWISE_OP}}({{DATA_TYPE}}, M0, N0, {{lhs}}, {{rhs}}, {{dst}});
)_";

    if(is_root)
    {
        // Calculate the destination indirect Y
        code +=
R"_(
        LOOP_UNROLLING(int, i, 0, 1, M0,
        {
            g_dst_indirect_y[i].v = (uint)min(g_ind_1 + i, (int)({{arg_dst}}_w * {{arg_dst}}_h) - 1);
            g_dst_indirect_y[i].v += g_ind_2 * (int)({{arg_dst}}_w * {{arg_dst}}_h);
        })
)_";
    }

    code +=
R"_(
    }
    //------------------ END KERNEL {{meta_kernel_id}} {{ELTWISE_OP}} ---------------------
)_";

    return code;
}

void ClTemplateElementwiseBinary::declare_variables(GpuKernelVariableTable &vtable, const ComponentGroup &comp_group) const
{
    vtable.declare_variable(
        comp_group,
        _lhs,
        GpuKernelArgumentInfo(common_tensor_type),
        "lhs");

    vtable.declare_variable(
        comp_group,
        _rhs,
        GpuKernelArgumentInfo(common_tensor_type),
        "rhs");

    vtable.declare_variable(
        comp_group,
        _dst,
        GpuKernelArgumentInfo(common_tensor_type),
        "dst");
}

TagLUT ClTemplateElementwiseBinary::get_tag_lut(const GpuKernelVariableTable &vtable, const ComponentGroup &comp_group) const
{
    TagLUT lut{};

    // Local build options
    lut["meta_kernel_id"] = id();
    lut["DATA_TYPE"]      = get_cl_type_from_data_type(_lhs->data_type());
    // Arguments and global shared variables

    lut["lhs"]     = vtable.get_variable(_lhs);
    lut["rhs"]     = vtable.get_variable(_rhs);
    lut["dst"]     = vtable.get_variable(_dst);
    lut["arg_dst"] = vtable.get_variable(comp_group.get_any_dst_tensor());

    switch(_attributes.operation())
    {
        case Attributes::ElementwiseOp::Add:
            lut["ELTWISE_OP"] = "ADD";
            break;
        case Attributes::ElementwiseOp::Sub:
            lut["ELTWISE_OP"] = "SUB";
            break;
        case Attributes::ElementwiseOp::Mul:
            lut["ELTWISE_OP"] = "MUL";
            break;
        default:
            ARM_COMPUTE_ERROR("Arithmetic Operation not supported");
    }

    ARM_COMPUTE_ERROR_ON(
        comp_group.is_intermediate_tensor(_lhs) && detail::have_different_dimensions(_lhs->tensor_shape(), _dst->tensor_shape(), 0));
    ARM_COMPUTE_ERROR_ON(
        comp_group.is_intermediate_tensor(_rhs) && detail::have_different_dimensions(_rhs->tensor_shape(), _dst->tensor_shape(), 0));

    // Set broadcast parameters
    // PRE: All tensors are broadcast-compatible
    const auto &lhs_dims = _lhs->tensor_shape();
    const auto &rhs_dims = _rhs->tensor_shape();
    const auto &dst_dims = _dst->tensor_shape();

    const auto lhs_broadcast_x = dst_dims[0] != 1 && lhs_dims[0] == 1;
    const auto rhs_broadcast_x = dst_dims[0] != 1 && rhs_dims[0] == 1;
    const auto lhs_broadcast_y = dst_dims[1] != 1 && lhs_dims[1] == 1;
    const auto rhs_broadcast_y = dst_dims[1] != 1 && rhs_dims[1] == 1;
    const auto lhs_broadcast_z = dst_dims[2] != 1 && lhs_dims[2] == 1;
    const auto rhs_broadcast_z = dst_dims[2] != 1 && rhs_dims[2] == 1;

    const auto lhs_broadcast_yz = lhs_broadcast_y && lhs_broadcast_z;
    const auto rhs_broadcast_yz = rhs_broadcast_y && rhs_broadcast_z;

    lut["lhs_n0"]          = (lhs_broadcast_x) ? "1" : "N0";
    lut["lhs_start_ind_0"] = (lhs_broadcast_x) ? "0" : "g_ind_0";
    lut["rhs_n0"]          = (rhs_broadcast_x) ? "1" : "N0";
    lut["rhs_start_ind_0"] = (rhs_broadcast_x) ? "0" : "g_ind_0";

    lut["lhs_m0"]          = (lhs_broadcast_yz) ? "1" : "M0";
    lut["lhs_start_ind_1"] = (lhs_broadcast_yz) ? "0" : "g_ind_1";
    lut["rhs_m0"]          = (rhs_broadcast_yz) ? "1" : "M0";
    lut["rhs_start_ind_1"] = (rhs_broadcast_yz) ? "0" : "g_ind_1";

    lut["BROADCAST_OP"] = (lhs_broadcast_yz) ? "BROADCAST_LHS_X_" :
                          (rhs_broadcast_yz) ? "BROADCAST_RHS_X_" :
                                               "";

    return lut;
}

CLBuildOptions ClTemplateElementwiseBinary::get_build_options(const ComponentGroup &comp_group) const
{
    CLBuildOptions build_opts{};
    /// NOTE: For now tile sizes (n0, m0) are set by the execution window. This may change in the future
    const auto         root_window      = comp_group.get_root_component()->template_writer()->get_window();
    const unsigned int n0               = root_window.x().step();
    const unsigned int m0               = root_window.y().step();
    const unsigned int partial_store_n0 = _dst->dimension(0) % n0;

    build_opts.add_option("-DM0=" + support::cpp11::to_string(m0));
    build_opts.add_option("-DN0=" + support::cpp11::to_string(n0));
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(_lhs->data_type()));
    build_opts.add_option("-DPARTIAL_N0=" + support::cpp11::to_string(partial_store_n0));

    return build_opts;
}

std::string ClTemplateElementwiseBinary::get_config_id() const
{
    std::string config_id{};
    config_id += lower_string(string_from_data_type(_dst->data_type()));
    config_id += "_";
    config_id += support::cpp11::to_string(_dst->dimension(0));
    config_id += "_";
    config_id += support::cpp11::to_string(_dst->dimension(1));
    config_id += "_";
    config_id += lower_string(string_from_data_layout(_dst->data_layout()));

    return config_id;
}

std::set<std::string> ClTemplateElementwiseBinary::get_headers_list() const
{
    return std::set<std::string>{ "helpers.h", "tile_helpers.h" };
}

Window ClTemplateElementwiseBinary::get_window() const
{
    ARM_COMPUTE_ERROR_ON_MSG(_dst->tensor_shape().total_size() == 0U, "Destination tensor is not initialized");

    TensorShape output_shape = _dst->tensor_shape();
    // Collapse Dim 1 (W) and Dim 2 (H) together, leave Dim 0 (C) and upper dimensions unchanged
    // This is in line with the collapsing convention used by operators like Conv2d
    output_shape.collapse(2U, 1U);
    const unsigned int num_elems_processed_per_iteration = adjust_vec_size(vector_size_byte_opencl / _dst->element_size(), _dst->dimension(0));
    Window             win                               = calculate_max_window(output_shape, Steps(num_elems_processed_per_iteration));

    return win;
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
