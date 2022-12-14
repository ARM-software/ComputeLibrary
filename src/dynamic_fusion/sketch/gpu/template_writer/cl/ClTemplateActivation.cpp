/*
 * Copyright (c) 2022 Arm Limited.
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
#include "ClTemplateActivation.h"

#include "arm_compute/core/Utils.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGroup.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{

ClTemplateActivation::ClTemplateActivation(ComponentId                      id,
                                           const ArgumentPack<ITensorInfo> &tensors,
                                           const Attributes                &attributes)
    : IGpuTemplateComponentWriter{ id, tensors },
      _src{},
      _dst{},
      _attributes{ attributes }
{
    _src = this->tensors().get_const_tensor(TensorType::ACL_SRC);
    _dst = this->tensors().get_const_tensor(TensorType::ACL_DST);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);
}

std::string ClTemplateActivation::get_name() const
{
    return "activation";
}

std::string ClTemplateActivation::get_component_code(const ComponentGroup &comp_group) const
{
    std::string code;
    const bool  is_root = (comp_group.get_root_component()->id() == this->id());

    code = R"_(
//------------------ START KERNEL {{meta_kernel_id}} ---------------------
)_";
    if(is_root)
    {
        code += R"_(
// IN(src)              {{src}}
// OUT(dst, accum)      {{dst}}

TILE({{DATA_TYPE}}, M0, N0, {{dst}});
TILE(uint, M0, 1, g_dst_indirect_y);
{
    {{src}}_offset_first_element_in_bytes += g_ind_2 * {{src}}_stride_z;

    T_LOAD({{DATA_TYPE}}, M0, N0, {{TENSOR_TYPE}}, {{src}}, g_ind_0, g_ind_1, 1, {{src}}_stride_y, {{dst}});

    T_ACTIVATION({{DATA_TYPE}}, M0, N0, {{ACT}}, {{A_VAL}}, {{B_VAL}}, {{dst}}, {{dst}});
}

LOOP_UNROLLING(int, i, 0, 1, M0,
{
    g_dst_indirect_y[i].v = (uint)min((int)(g_ind_1 + i), (int)({{arg_dst}}_w) - 1);
    g_dst_indirect_y[i].v += (int)(g_ind_2 % {{arg_dst}}_h) * (int)({{arg_dst}}_w);
    g_dst_indirect_y[i].v += (int)(g_ind_2 / {{arg_dst}}_h) * (int)({{arg_dst}}_w * {{arg_dst}}_h);
})
)_";
    }
    else
    {
        code += R"_(
// IN/OUT(src, accum)   {{src}}

{
    T_ACTIVATION({{DATA_TYPE}}, M0, N0, {{ACT}}, {{A_VAL}}, {{B_VAL}}, {{src}}, {{src}});
}
)_";
    }
    code += R"_(
//------------------ END KERNEL {{meta_kernel_id}} ---------------------
)_";
    return code;
}

void ClTemplateActivation::declare_variables(GpuKernelVariableTable &vtable, const ComponentGroup &comp_group) const
{
    vtable.declare_variable(
        _src,
        GpuKernelArgumentInfo(GpuKernelArgumentInfo::Type::Tensor_4D_t_Buffer),
        comp_group.is_intermediate_tensor(_src),
        "src");

    vtable.declare_variable(
        _dst,
        GpuKernelArgumentInfo(GpuKernelArgumentInfo::Type::Tensor_4D_t_Buffer),
        comp_group.is_intermediate_tensor(_dst),
        "dst");
}

TagLUT ClTemplateActivation::get_tag_lut(const GpuKernelVariableTable &vtable, const ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);

    TagLUT lut{};
    // Arguments and global shared variables
    lut["src"] = vtable.get_variable(_src);
    lut["dst"] = vtable.get_variable(_dst);

    const auto dst_argument = vtable.get_variable(comp_group.get_any_dst_tensor());
    lut["arg_dst"]          = dst_argument.uniq_name;

    // Local build options
    lut["meta_kernel_id"] = id();
    lut["DATA_TYPE"]      = get_cl_type_from_data_type(_src->data_type());
    lut["TENSOR_TYPE"]    = "BUFFER";

    const auto f_act = lower_string(string_from_activation_func(_attributes.activation()));

    lut["ACT"]   = f_act;
    lut["A_VAL"] = float_to_string_with_full_precision(_attributes.a());
    lut["B_VAL"] = float_to_string_with_full_precision(_attributes.b());

    return lut;
}

CLBuildOptions ClTemplateActivation::get_build_options(const ComponentGroup &comp_group) const
{
    /// NOTE: For now tile sizes (n0, m0) are set by the execution window. This may change in the future
    const auto         root_window      = comp_group.get_root_component()->template_writer()->get_window();
    const unsigned int n0               = root_window.x().step();
    const unsigned int m0               = root_window.y().step();
    const unsigned int partial_store_n0 = _dst->dimension(0) % n0;

    CLBuildOptions build_opts;
    build_opts.add_option("-DN0=" + support::cpp11::to_string(n0));
    build_opts.add_option("-DM0=" + support::cpp11::to_string(m0));
    build_opts.add_option("-DPARTIAL_N0=" + support::cpp11::to_string(partial_store_n0));

    return build_opts;
}

std::string ClTemplateActivation::get_config_id() const
{
    std::string config_id{};
    config_id += "activation_";
    config_id += lower_string(string_from_data_type(_src->data_type()));
    config_id += "_";
    config_id += support::cpp11::to_string(_src->dimension(0));
    config_id += "_";
    config_id += support::cpp11::to_string(_src->dimension(1));
    return config_id;
}

std::set<std::string> ClTemplateActivation::get_headers_list() const
{
    return std::set<std::string>{ "helpers.h", "tile_helpers.h", "activation_float_helpers.h" };
}

Window ClTemplateActivation::get_window() const
{
    ARM_COMPUTE_ERROR_ON_MSG(_dst->tensor_shape().total_size() == 0U, "Destination tensor is not initialized");
    const unsigned int n0  = adjust_vec_size(16 / _src->element_size(), _src->dimension(0));
    Window             win = calculate_max_window(*_dst, Steps(n0));
    return win.collapse(win, Window::DimZ);
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
