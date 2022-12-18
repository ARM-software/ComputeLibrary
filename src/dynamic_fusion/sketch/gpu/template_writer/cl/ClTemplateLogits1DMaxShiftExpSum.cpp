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

#include "src/dynamic_fusion/sketch/gpu/template_writer/cl/ClTemplateLogits1DMaxShiftExpSum.h"

#include "src/core/helpers/WindowHelpers.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGroup.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
ClTemplateLogits1DMaxShiftExpSum::ClTemplateLogits1DMaxShiftExpSum(ComponentId                      id,
                                                                   const ArgumentPack<ITensorInfo> &tensors,
                                                                   const Attributes                &attributes)
    : IGpuTemplateComponentWriter{ id, tensors },
      _src{},
      _sum{},
      _dst{},
      _attributes{ attributes }
{
    _src = this->tensors().get_const_tensor(TensorType::ACL_SRC_0);
    _sum = this->tensors().get_const_tensor(TensorType::ACL_DST_0);
    _dst = this->tensors().get_const_tensor(TensorType::ACL_DST_1);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _sum, _dst);
}

std::string ClTemplateLogits1DMaxShiftExpSum::get_name() const
{
    return "logits_1d_max_shift_exp_sum";
}

std::string ClTemplateLogits1DMaxShiftExpSum::get_component_code(const ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);

    std::string code = R"_(
//------------------ START KERNEL {{meta_kernel_id}} ---------------------
#define VEC_TYPE VEC_DATA_TYPE({{DATA_TYPE}}, N0)
#define SELECT_TYPE SELECT_VEC_DATA_TYPE({{DATA_TYPE}}, N0)
{
    __global uchar *src_addr = {{src}}_ptr + {{src}}_offset_first_element_in_bytes + g_ind_1 * {{src}}_stride_y + g_ind_2 * {{src}}_stride_z;
    __global uchar *dst_addr = {{dst}}_ptr + {{dst}}_offset_first_element_in_bytes + g_ind_1 * {{dst}}_stride_y + g_ind_2 * {{dst}}_stride_z;

    Image sum = CONVERT_TENSOR3D_TO_IMAGE_STRUCT({{sum}});
    VEC_TYPE max_val_vec = (VEC_TYPE)({{MINVAL}});
)_";

    const bool beta_defined = (_attributes.beta() != 1.f);

    if(beta_defined)
    {
        code += R"_(
    VEC_TYPE beta = (VEC_TYPE){{BETA}};
)_";
    }

    constexpr unsigned int _serial_vector_size = 8;
    const unsigned int     reduction_dim_size  = _src->dimension(0);
    const unsigned int     vector_size         = adjust_vec_size(_serial_vector_size, reduction_dim_size);
    const bool             non_multiple_of_n0  = ((reduction_dim_size % vector_size) != 0);

    if(non_multiple_of_n0)
    {
        code += R"_(
    VEC_TYPE data    = VLOAD(N0)(0, (__global {{DATA_TYPE}} *)src_addr);
    SELECT_TYPE widx = (SELECT_TYPE)PARTIAL_N0 > VEC_OFFS(SELECT_DATA_TYPE({{DATA_TYPE}}), N0);
    max_val_vec      = max(max_val_vec, select((VEC_TYPE)({{MINVAL}}), data, widx));
)_";
    }

    code += R"_(
    for(uint i = PARTIAL_N0; i < {{SRC_WIDTH}}; i += N0)
    {
        VEC_TYPE data = VLOAD(N0)(0, (__global {{DATA_TYPE}} *)(src_addr + i * sizeof({{DATA_TYPE}})));
        max_val_vec   = max(data, max_val_vec);
    }

    {{DATA_TYPE}} max_val = MAX_REDUCE(max_val_vec, N0);
    VEC_TYPE sum1D = 0;
)_";

    if(non_multiple_of_n0)
    {
        code += R"_(
    data -= max_val;
)_";
        if(beta_defined)
        {
            code += R"_(
    data *= beta;
)_";
        }

        if(_attributes.is_log_softmax())
        {
            code += R"_(
    VSTORE_PARTIAL(N0, PARTIAL_N0)
    (data, 0, (__global {{DATA_TYPE}} *)dst_addr);
    data = exp(data);
    data = select(0, data, widx);
)_";
        }
        else
        {
            code += R"_(
    data = exp(data);
    data = select(0, data, widx);
    VSTORE_PARTIAL(N0, PARTIAL_N0)
    (data, 0, (__global {{DATA_TYPE}} *)dst_addr);
)_";
        }

        code += R"_(
    sum1D += data;
)_";
    }
    else
    {
        code += R"_(
    for(uint i = PARTIAL_N0; i < {{SRC_WIDTH}}; i += N0)
    {
        VEC_TYPE data = VLOAD(N0)(0, (__global {{DATA_TYPE}} *)(src_addr + i * sizeof({{DATA_TYPE}})));
        data -= max_val;
)_";

        if(beta_defined)
        {
            code += R"_(
        data *= beta;
)_";
        }

        if(_attributes.is_log_softmax())
        {
            code += R"_(
        VSTORE(N0)
        (data, 0, (__global {{DATA_TYPE}} *)(dst_addr + i * sizeof({{DATA_TYPE}})));
        data = exp(data);
)_";
        }
        else
        {
            code += R"_(
        data = exp(data);
        VSTORE(N0)
        (data, 0, (__global {{DATA_TYPE}} *)(dst_addr + i * sizeof({{DATA_TYPE}})));
)_";
        }

        code += R"_(
        sum1D += data;
    }
)_";
    }

    code += R"_(
    *((__global {{DATA_TYPE}} *)sum.ptr) = SUM_REDUCE(sum1D, N0);
}
//------------------ END KERNEL {{meta_kernel_id}} ---------------------
)_";

    return code;
}

void ClTemplateLogits1DMaxShiftExpSum::declare_variables(GpuKernelVariableTable &vtable, const ComponentGroup &comp_group) const
{
    vtable.declare_variable(
        _src,
        GpuKernelArgumentInfo(GpuKernelArgumentInfo::Type::Tensor_4D_t_Buffer),
        comp_group.is_intermediate_tensor(_src),
        "src");

    vtable.declare_variable(
        _sum,
        GpuKernelArgumentInfo(GpuKernelArgumentInfo::Type::Tensor_4D_t_Buffer),
        comp_group.is_intermediate_tensor(_sum),
        "sum");

    vtable.declare_variable(
        _dst,
        GpuKernelArgumentInfo(GpuKernelArgumentInfo::Type::Tensor_4D_t_Buffer),
        comp_group.is_intermediate_tensor(_dst),
        "dst");
}

TagLUT ClTemplateLogits1DMaxShiftExpSum::get_tag_lut(const GpuKernelVariableTable &vtable, const ComponentGroup &comp_group) const
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
    lut["BETA"]      = float_to_string_with_full_precision(_attributes.beta());
    lut["MINVAL"]    = (data_type == DataType::F16) ? std::string("-HALF_MAX") : std::string("-FLT_MAX");
    lut["SRC_WIDTH"] = support::cpp11::to_string(_src->dimension(0));

    return lut;
}

CLBuildOptions ClTemplateLogits1DMaxShiftExpSum::get_build_options(const ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);
    CLBuildOptions build_opts{};

    constexpr unsigned int serial_vector_size = 8;
    const unsigned int     reduction_dim_size = _src->dimension(0);
    const unsigned int     vector_size        = adjust_vec_size(serial_vector_size, reduction_dim_size);

    build_opts.add_option("-DN0=" + support::cpp11::to_string(vector_size));
    build_opts.add_option("-DPARTIAL_N0=" + support::cpp11::to_string((reduction_dim_size % vector_size)));

    return build_opts;
}

std::string ClTemplateLogits1DMaxShiftExpSum::get_config_id() const
{
    std::string config_id = get_name();

    config_id += "_";
    config_id += support::cpp11::to_string(_src->dimension(0));
    config_id += "_";
    config_id += string_from_data_type(_src->data_type());

    return config_id;
}

std::set<std::string> ClTemplateLogits1DMaxShiftExpSum::get_headers_list() const
{
    return std::set<std::string>{ "helpers.h" };
}

Window ClTemplateLogits1DMaxShiftExpSum::get_window() const
{
    ARM_COMPUTE_ERROR_ON_MSG(_dst->tensor_shape().total_size() == 0U, "Destination tensor is not initialized");

    Window win = calculate_max_window(*_dst, Steps(_src->dimension(0)));
    return win.collapse(win, Window::DimZ);
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute