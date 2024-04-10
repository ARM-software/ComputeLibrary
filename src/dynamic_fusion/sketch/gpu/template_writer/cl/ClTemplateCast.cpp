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
#include "ClTemplateCast.h"

#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/utils/StringUtils.h"

#include "src/core/helpers/WindowHelpers.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGroup.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
ClTemplateCast::ClTemplateCast(ComponentId id, const ArgumentPack<ITensorInfo> &tensors, const Attributes &attributes)
    : IGpuTemplateComponentWriter{id, tensors}, _src{}, _dst{}, _attributes{attributes}
{
    _src = this->tensors().get_const_tensor(TensorType::ACL_SRC_0);
    _dst = this->tensors().get_const_tensor(TensorType::ACL_DST_0);

    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);
}

std::string ClTemplateCast::get_name() const
{
    const size_t src_size = data_size_from_type(_src->data_type());
    const size_t dst_size = data_size_from_type(_dst->data_type());

    return (src_size >= dst_size) ? "cast_down" : "cast_up";
}

std::string ClTemplateCast::get_component_code(const ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);

    const std::string kernel_name = get_name();
    const auto        is_root     = (comp_group.get_root_component()->id() == this->id());

    std::string code = R"_(
//------------------ START KERNEL {{meta_kernel_id}} CAST ---------------------
)_";

    if (is_root)
    {
        code += R"_(
// IN_0(src)            {{src}}
// OUT(dst, accum)      {{dst}}

TILE(uint, M0, 1, g_dst_indirect_y);
{
    {{src}}_offset_first_element_in_bytes += get_global_id(2) * {{src}}_stride_z;

    TILE({{DATA_TYPE_IN}}, M0, N0, {{tmp}});
    T_LOAD({{DATA_TYPE_IN}}, M0, N0, BUFFER, {{src}}, g_ind_0, g_ind_1, 1, {{src}}_stride_y, {{tmp}});
)_";
    }

    code += R"_(
    LOOP_UNROLLING(int, m0, 0, 1, M0,
    {
)_";

    if (kernel_name == "cast_down" && is_data_type_quantized(_src->data_type()))
    {
        code += R"_(
    {{tmp}}[m0].v ^= (VEC_DATA_TYPE({{DATA_TYPE_IN}}, N0))0x80;
)_";
    }

    if (kernel_name == "cast_down" &&
        (is_data_type_float(_src->data_type()) || _attributes.convert_policy() == ConvertPolicy::SATURATE))
    {
        code += R"_(
    {{dst}}[m0].v = CONVERT_SAT({{tmp}}[m0].v, VEC_DATA_TYPE({{DATA_TYPE_OUT}}, N0));
)_";
    }
    else
    {
        code += R"_(
    {{dst}}[m0].v = CONVERT({{tmp}}[m0].v, VEC_DATA_TYPE({{DATA_TYPE_OUT}}, N0));
)_";
    }

    code += R"_(
    })
)_";

    if (is_root)
    {
        code += R"_(
    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        g_dst_indirect_y[i].v = (uint)min((int)(g_ind_1 + i), (int)({{arg_dst}}_w) - 1);
        g_dst_indirect_y[i].v += (int)(g_ind_2 % {{arg_dst}}_h) * (int)({{arg_dst}}_w);
        g_dst_indirect_y[i].v += (int)(g_ind_2 / {{arg_dst}}_h) * (int)({{arg_dst}}_w * {{arg_dst}}_h);
    })
}
)_";
    }

    code += R"_(
//------------------ END KERNEL {{meta_kernel_id}} CAST ---------------------
)_";

    return code;
}

void ClTemplateCast::declare_variables(GpuKernelVariableTable &vtable, const ComponentGroup &comp_group) const
{
    vtable.declare_variable(comp_group, _src, GpuKernelArgumentInfo(GpuKernelArgumentInfo::Type::Tensor_4D_t_Buffer),
                            "src");

    vtable.declare_variable(comp_group, _dst, GpuKernelArgumentInfo(GpuKernelArgumentInfo::Type::Tensor_4D_t_Buffer),
                            "dst");
}

TagLUT ClTemplateCast::get_tag_lut(const GpuKernelVariableTable &vtable, const ComponentGroup &comp_group) const
{
    const auto is_root = (comp_group.get_root_component()->id() == this->id());

    TagLUT lut{};

    // Arguments and global shared variables
    lut["src"] = vtable.get_variable(_src);
    lut["dst"] = vtable.get_variable(_dst);
    lut["tmp"] = (is_root) ? lut["src"].value + "_in_data" : lut["src"];

    const auto dst_argument = vtable.get_variable(comp_group.get_any_dst_tensor());
    lut["arg_dst"]          = dst_argument.uniq_name;

    // Local build options
    lut["meta_kernel_id"] = id();

    lut["DATA_TYPE_IN"]  = get_cl_type_from_data_type(_src->data_type());
    lut["DATA_TYPE_OUT"] = get_cl_type_from_data_type(_dst->data_type());

    return lut;
}

CLBuildOptions ClTemplateCast::get_build_options(const ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);

    const auto         root_window = comp_group.get_root_component()->template_writer()->get_window();
    const unsigned int n0          = root_window.x().step();
    const unsigned int m0          = root_window.y().step();

    // Set build options
    CLBuildOptions build_opts{};
    build_opts.add_option("-DN0=" + support::cpp11::to_string(n0));
    build_opts.add_option("-DPARTIAL_N0=" + support::cpp11::to_string(_src->dimension(0) % n0));
    build_opts.add_option("-DM0=" + support::cpp11::to_string(m0));

    return build_opts;
}

std::string ClTemplateCast::get_config_id() const
{
    std::string config_id{};

    config_id += "_";
    config_id += lower_string(string_from_data_type(_src->data_type()));
    config_id += "_";
    config_id += lower_string(string_from_data_type(_dst->data_type()));
    config_id += "_";
    config_id += support::cpp11::to_string(_src->dimension(0));
    config_id += "_";
    config_id += support::cpp11::to_string(_src->dimension(1));

    return config_id;
}

std::set<std::string> ClTemplateCast::get_headers_list() const
{
    return std::set<std::string>{"helpers.h", "tile_helpers.h"};
}

Window ClTemplateCast::get_window() const
{
    ARM_COMPUTE_ERROR_ON_MSG(_dst->tensor_shape().total_size() == 0U, "Destination tensor is not initialized");

    const unsigned int n0  = adjust_vec_size(16 / _dst->element_size(), _dst->dimension(0));
    Window             win = calculate_max_window(*_dst, Steps(n0));
    return win.collapse(win, Window::DimZ);
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
