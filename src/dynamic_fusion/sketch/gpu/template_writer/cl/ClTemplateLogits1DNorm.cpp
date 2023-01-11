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

#include "src/dynamic_fusion/sketch/gpu/template_writer/cl/ClTemplateLogits1DNorm.h"

#include "src/core/helpers/WindowHelpers.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGroup.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
ClTemplateLogits1DNorm::ClTemplateLogits1DNorm(ComponentId                      id,
                                               const ArgumentPack<ITensorInfo> &tensors,
                                               const Attributes                &attributes)
    : IGpuTemplateComponentWriter{ id, tensors },
      _src{},
      _sum{},
      _dst{},
      _attributes{ attributes }
{
    _src = this->tensors().get_const_tensor(TensorType::ACL_SRC_0);
    _sum = this->tensors().get_const_tensor(TensorType::ACL_SRC_1);
    _dst = this->tensors().get_const_tensor(TensorType::ACL_DST_0);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_src);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_sum);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_dst);
}

std::string ClTemplateLogits1DNorm::get_name() const
{
    return "logits_1d_norm";
}

std::string ClTemplateLogits1DNorm::get_component_code(const ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);

    std::string code = R"_(
//------------------ START KERNEL {{meta_kernel_id}} ---------------------
{
    const int x_offs = g_ind_0 * sizeof({{DATA_TYPE}});
    __global uchar *src_addr = {{src}}_ptr + {{src}}_offset_first_element_in_bytes + x_offs + g_ind_1 * {{src}}_stride_y + g_ind_2 * {{src}}_stride_z;
    __global uchar *dst_addr = {{dst}}_ptr + {{dst}}_offset_first_element_in_bytes + x_offs + g_ind_1 * {{dst}}_stride_y + g_ind_2 * {{dst}}_stride_z;
    Image sum = CONVERT_TENSOR3D_TO_IMAGE_STRUCT_NO_STEP({{sum}});
)_";
    // Load max value of 1D logits vector (row)
    code += R"_(
    {{DATA_TYPE}} sum_val = *((__global {{DATA_TYPE}} *)offset(&sum, 0, g_ind_1));
    VEC_DATA_TYPE({{DATA_TYPE}}, N0)
    data0 = VLOAD(N0)(0, (__global {{DATA_TYPE}} *)src_addr);
)_";

    if(_attributes.is_log_softmax())
    {
        code += R"_(
    sum_val = log(sum_val);
    data0 -= sum_val;
)_";
    }
    else
    {
        code += R"_(
    data0 /= sum_val;
)_";
    }

    code += R"_(
    STORE_VECTOR_SELECT(data, {{DATA_TYPE}}, dst_addr, N0, PARTIAL_N0, PARTIAL_N0 != 0 && g_ind_0 == 0);
}
//------------------ END KERNEL {{meta_kernel_id}} ---------------------
)_";

    return code;
}

void ClTemplateLogits1DNorm::declare_variables(GpuKernelVariableTable &vtable, const ComponentGroup &comp_group) const
{
    vtable.declare_variable(
        comp_group,
        _src,
        GpuKernelArgumentInfo(GpuKernelArgumentInfo::Type::Tensor_3D),
        "src");

    vtable.declare_variable(
        comp_group,
        _sum,
        GpuKernelArgumentInfo(GpuKernelArgumentInfo::Type::Tensor_3D),
        "sum");

    vtable.declare_variable(
        comp_group,
        _dst,
        GpuKernelArgumentInfo(GpuKernelArgumentInfo::Type::Tensor_3D),
        "dst");
}

TagLUT ClTemplateLogits1DNorm::get_tag_lut(const GpuKernelVariableTable &vtable, const ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);

    TagLUT lut{};

    // Arguments and global shared variables
    lut["src"] = vtable.get_variable(_src);
    lut["sum"] = vtable.get_variable(_sum);
    lut["dst"] = vtable.get_variable(_dst);

    // Local build options
    lut["meta_kernel_id"] = id();

    const DataType data_type = _src->data_type();

    lut["DATA_TYPE"] = get_cl_type_from_data_type(data_type);

    return lut;
}

CLBuildOptions ClTemplateLogits1DNorm::get_build_options(const ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);
    CLBuildOptions build_opts{};

    const auto         root_window = comp_group.get_root_component()->template_writer()->get_window();
    const unsigned int n0          = root_window.x().step();
    build_opts.add_option("-DN0=" + support::cpp11::to_string(n0));
    build_opts.add_option("-DPARTIAL_N0=" + support::cpp11::to_string((_src->dimension(0) % n0)));

    return build_opts;
}

std::string ClTemplateLogits1DNorm::get_config_id() const
{
    std::string config_id = get_name();

    config_id += "_";
    config_id += support::cpp11::to_string(_src->dimension(0));
    config_id += "_";
    config_id += string_from_data_type(_src->data_type());

    return config_id;
}

std::set<std::string> ClTemplateLogits1DNorm::get_headers_list() const
{
    return std::set<std::string>{ "helpers.h", "tile_helpers.h" };
}

Window ClTemplateLogits1DNorm::get_window() const
{
    ARM_COMPUTE_ERROR_ON_MSG(_dst->tensor_shape().total_size() == 0U, "Destination tensor is not initialized");
    constexpr unsigned int serial_vector_size = 16;
    const unsigned int vector_size = adjust_vec_size(serial_vector_size, _src->dimension(0));

    Window win = calculate_max_window(*_src, Steps(vector_size));
    return win.collapse(win, Window::DimZ);
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
