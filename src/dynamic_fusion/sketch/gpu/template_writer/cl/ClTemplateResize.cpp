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

#include "ClTemplateResize.h"

#include "src/core/helpers/WindowHelpers.h"
#include "src/core/utils/ScaleUtils.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGroup.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
ClTemplateResize::ClTemplateResize(ComponentId id, const ArgumentPack<ITensorInfo> &tensors, const ClTemplateResize::Attributes &attributes)
    : IGpuTemplateComponentWriter{ id, tensors }, _src{}, _dst{}, _attributes{ attributes }
{
    _src = this->tensors().get_const_tensor(TensorType::ACL_SRC_0);
    _dst = this->tensors().get_const_tensor(TensorType::ACL_DST_0);

    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);
}

std::string ClTemplateResize::get_name() const
{
    return _attributes.interpolation_policy() == InterpolationPolicy::BILINEAR ? "resize_bilinear" : "resize_nearest";
}

std::string ClTemplateResize::get_component_code(const IGpuTemplateComponentWriter::ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);

    std::string code = R"_(
//------------------ START KERNEL {{meta_kernel_id}} ---------------------
TILE({{DST_DATA_TYPE}}, 1, N0, {{dst}});
TILE(uint, 1, 1, g_dst_indirect_y);
{
    const int yo = g_ind_2 % {{arg_dst}}_h;
    const int bout = g_ind_2 / {{arg_dst}}_h;
)_";

    if(_attributes.interpolation_policy() == InterpolationPolicy::NEAREST_NEIGHBOR)
    {
        if(_attributes.sampling_policy() == SamplingPolicy::TOP_LEFT)
        {
            code += R"_(
    float xi_f = (g_ind_1 * SCALE_X);
    float yi_f = (yo * SCALE_Y);
)_";
        }
        else
        {
            code += R"_(
    float xi_f = ((g_ind_1 + 0.5f) * SCALE_X);
    float yi_f = ((yo + 0.5f) * SCALE_Y);
)_";
        }

        if(_attributes.align_corners())
        {
            code += R"_(
    xi_f = round(xi_f);
    yi_f = round(yi_f);
)_";
        }

        code += R"_(
    const int xi0 = clamp((int)xi_f, 0, (int){{src}}_w - 1);
    const int yi0 = clamp((int)yi_f, 0, (int){{src}}_h - 1);

    T_LOAD_NHWC_WITH_DILATION({{SRC_DATA_TYPE}}, 1, 1, N0, {{SRC_TENSOR_TYPE}}, {{src}}, bout, yi0, xi0, g_ind_0, {{src}}_w, {{src}}_h, 1, 1, false, {{dst}});
)_";
    }
    else if(_attributes.interpolation_policy() == InterpolationPolicy::BILINEAR)
    {
        if(_attributes.sampling_policy() == SamplingPolicy::TOP_LEFT)
        {
            code += R"_(
    float xi_f = (g_ind_1 * SCALE_X);
    float yi_f = (yo * SCALE_Y);
)_";
        }
        else
        {
            code += R"_(
    float xi_f = ((g_ind_1 + 0.5f) * SCALE_X - 0.5f);
    float yi_f = ((yo + 0.5f) * SCALE_Y - 0.5f);
)_";
        }

        code += R"_(
    const int xi = (int)floor(xi_f);
    const int yi = (int)floor(yi_f);

    TILE({{SRC_DATA_TYPE}}, 1, N0, in00);
    TILE({{SRC_DATA_TYPE}}, 1, N0, in01);
    TILE({{SRC_DATA_TYPE}}, 1, N0, in10);
    TILE({{SRC_DATA_TYPE}}, 1, N0, in11);

    in00[0].v = {{CONSTANT_VALUE}};
    in01[0].v = {{CONSTANT_VALUE}};
    in10[0].v = {{CONSTANT_VALUE}};
    in11[0].v = {{CONSTANT_VALUE}};

    const int xi0  = clamp(xi, 0, (int){{src}}_w - 1);
    const int yi0  = clamp(yi, 0, (int){{src}}_h - 1);
    const int xi1  = clamp(xi + 1, 0, (int){{src}}_w - 1);
    const int yi1  = clamp(yi + 1, 0, (int){{src}}_h - 1);

    T_LOAD_NHWC_WITH_DILATION({{SRC_DATA_TYPE}}, 1, 1, N0, {{SRC_TENSOR_TYPE}}, {{src}}, bout, yi0, xi0, g_ind_0, {{src}}_w, {{src}}_h, 1, 1, false, in00);
    T_LOAD_NHWC_WITH_DILATION({{SRC_DATA_TYPE}}, 1, 1, N0, {{SRC_TENSOR_TYPE}}, {{src}}, bout, yi0, xi1, g_ind_0, {{src}}_w, {{src}}_h, 1, 1, false, in01);
    T_LOAD_NHWC_WITH_DILATION({{SRC_DATA_TYPE}}, 1, 1, N0, {{SRC_TENSOR_TYPE}}, {{src}}, bout, yi1, xi0, g_ind_0, {{src}}_w, {{src}}_h, 1, 1, false, in10);
    T_LOAD_NHWC_WITH_DILATION({{SRC_DATA_TYPE}}, 1, 1, N0, {{SRC_TENSOR_TYPE}}, {{src}}, bout, yi1, xi1, g_ind_0, {{src}}_w, {{src}}_h, 1, 1, false, in11);
)_";

        if(is_data_type_float(_src->data_type()))
        {
            code += R"_(
    const {{SRC_DATA_TYPE}} a  = ({{SRC_DATA_TYPE}})(xi_f - (float)xi);
    const {{SRC_DATA_TYPE}} b  = ({{SRC_DATA_TYPE}})(1.f - a);
    const {{SRC_DATA_TYPE}} a1 = ({{SRC_DATA_TYPE}})(yi_f - (float)yi);
    const {{SRC_DATA_TYPE}} b1 = ({{SRC_DATA_TYPE}})(1.f - a1);

    // Calculate the output
    {{dst}}[0].v = ((in00[0].v * b * b1) + (in01[0].v * a * b1) + (in10[0].v * b * a1) + (in11[0].v * a * a1));
)_";
        }
        else
        {
            code += R"_(
    TILE(float, 1, N0, out_f);
    TILE(float, 1, N0, in00_f);
    TILE(float, 1, N0, in01_f);
    TILE(float, 1, N0, in10_f);
    TILE(float, 1, N0, in11_f);

    const float a  = (xi_f - (float)xi);
    const float b  = (1.f - a);
    const float a1 = (yi_f - (float)yi);
    const float b1 = (1.f - a1);
)_"
                    // Dequantize
                    R"_(
    LOOP_UNROLLING(int, n0, 0, 1, N0,
    {
        in00_f[0].s[n0] = ((float)in00[0].s[n0] - (float){{OFFSET}}) * (float){{SCALE}};
        in01_f[0].s[n0] = ((float)in01[0].s[n0] - (float){{OFFSET}}) * (float){{SCALE}};
        in10_f[0].s[n0] = ((float)in10[0].s[n0] - (float){{OFFSET}}) * (float){{SCALE}};
        in11_f[0].s[n0] = ((float)in11[0].s[n0] - (float){{OFFSET}}) * (float){{SCALE}};
    })
)_"
                    // Calculate the output in the floating-point domain
                    R"_(
    out_f[0].v = ((in00_f[0].v * b * b1) + (in01_f[0].v * a * b1) + (in10_f[0].v * b * a1) + (in11_f[0].v * a * a1));
)_"
                    // Quantize
                    R"_(
    LOOP_UNROLLING(int, n0, 0, 1, N0,
    {
        {{dst}}[0].s[n0] = CONVERT_SAT(out_f[0].s[n0] / (float){{SCALE}} + (float){{OFFSET}}, {{DST_DATA_TYPE}});
    })
)_";
        }
    }
    else
    {
        ARM_COMPUTE_ERROR("Unsupported interpolation policy");
    }

    code += R"_(
    g_dst_indirect_y[0].v = g_ind_1 + (yo * (int)({{arg_dst}}_w)) + bout * (int)({{arg_dst}}_w * {{arg_dst}}_h);
}
//------------------ END KERNEL {{meta_kernel_id}} ---------------------
)_";

    return code;
}

void ClTemplateResize::declare_variables(GpuKernelVariableTable &vtable, const IGpuTemplateComponentWriter::ComponentGroup &comp_group) const
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

TagLUT ClTemplateResize::get_tag_lut(const GpuKernelVariableTable &vtable, const IGpuTemplateComponentWriter::ComponentGroup &comp_group) const
{
    TagLUT lut{};

    // Arguments and global shared variables
    lut["src"] = vtable.get_variable(_src);
    lut["dst"] = vtable.get_variable(_dst);

    const auto dst_argument = vtable.get_variable(comp_group.get_any_dst_tensor());
    lut["arg_dst"]          = dst_argument.uniq_name;

    // Local build options
    lut["meta_kernel_id"]  = id();
    lut["SRC_DATA_TYPE"]   = get_cl_type_from_data_type(_src->data_type());
    lut["SRC_TENSOR_TYPE"] = "BUFFER";
    lut["DST_DATA_TYPE"]   = get_cl_type_from_data_type(_dst->data_type());
    lut["CONSTANT_VALUE"]  = string_from_pixel_value(0, _src->data_type());

    const bool is_qasymm_bilinear = is_data_type_quantized_asymmetric(_src->data_type())
                                    && _attributes.interpolation_policy() == InterpolationPolicy::BILINEAR;

    if(is_qasymm_bilinear)
    {
        const UniformQuantizationInfo qinfo = _src->quantization_info().uniform();
        lut["SCALE"]                        = support::cpp11::to_string(qinfo.scale);
        lut["OFFSET"]                       = support::cpp11::to_string(qinfo.offset);
    }
    else
    {
        lut["SCALE"]  = support::cpp11::to_string(1);
        lut["OFFSET"] = support::cpp11::to_string(0);
    }

    return lut;
}

CLBuildOptions ClTemplateResize::get_build_options(const IGpuTemplateComponentWriter::ComponentGroup &comp_group) const
{
    const Window       root_window = comp_group.get_root_component()->template_writer()->get_window();
    const unsigned int n0          = root_window.x().step();
    const unsigned int m0          = root_window.y().step();
    const unsigned int partial_n0  = _dst->dimension(0) % n0;

    const float scale_x = scale_utils::calculate_resize_ratio(_src->dimension(1), _dst->dimension(1), _attributes.align_corners());
    const float scale_y = scale_utils::calculate_resize_ratio(_src->dimension(2), _dst->dimension(2), _attributes.align_corners());

    CLBuildOptions build_opts;

    build_opts.add_option("-DN0=" + support::cpp11::to_string(n0));
    build_opts.add_option("-DM0=" + support::cpp11::to_string(m0));
    build_opts.add_option("-DPARTIAL_N0=" + support::cpp11::to_string(partial_n0));
    build_opts.add_option("-DSCALE_X=" + float_to_string_with_full_precision(scale_x));
    build_opts.add_option("-DSCALE_Y=" + float_to_string_with_full_precision(scale_y));

    return build_opts;
}

std::string ClTemplateResize::get_config_id() const
{
    std::string config_id{};

    config_id += "resize_";
    config_id += (_attributes.interpolation_policy() == InterpolationPolicy::NEAREST_NEIGHBOR ? "NEAREST_NEIGHBOR" : "");
    config_id += (_attributes.interpolation_policy() == InterpolationPolicy::BILINEAR ? "BILINEAR" : "");
    config_id += "_";
    config_id += (_attributes.sampling_policy() == SamplingPolicy::CENTER ? "center" : "topleft");
    config_id += "_";
    config_id += support::cpp11::to_string(_dst->dimension(0));
    config_id += "_";
    config_id += support::cpp11::to_string(_dst->dimension(1));
    config_id += "_";
    config_id += support::cpp11::to_string(_dst->dimension(2));
    config_id += "_";
    config_id += support::cpp11::to_string(_dst->dimension(3));

    return config_id;
}

std::set<std::string> ClTemplateResize::get_headers_list() const
{
    return std::set<std::string>{ "helpers.h", "tile_helpers.h" };
}

Window ClTemplateResize::get_window() const
{
    ARM_COMPUTE_ERROR_ON_MSG(_dst->tensor_shape().total_size() == 0U, "Destination tensor is not initialized");

    const unsigned int n0  = adjust_vec_size(16 / _src->element_size(), _src->dimension(0));
    Window             win = calculate_max_window(*_dst, Steps(n0));
    return win.collapse(win, Window::DimZ);
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
