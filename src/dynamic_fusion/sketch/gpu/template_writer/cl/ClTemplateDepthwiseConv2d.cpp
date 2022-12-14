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
#include "ClTemplateDepthwiseConv2d.h"

#include "src/core/helpers/WindowHelpers.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGroup.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
ClTemplateDepthwiseConv2d::ClTemplateDepthwiseConv2d(ComponentId                      id,
                                                     const ArgumentPack<ITensorInfo> &tensors,
                                                     const Attributes                &attributes,
                                                     const Settings                  &settings)
    : IGpuTemplateComponentWriter{ id, tensors },
      _src{},
      _weight{},
      _bias{},
      _dst{},
      _attributes{ attributes },
      _settings{ settings }
{
    _src    = this->tensors().get_const_tensor(TensorType::ACL_SRC_0);
    _weight = this->tensors().get_const_tensor(TensorType::ACL_SRC_1);
    if(this->tensors().get_const_tensor(TensorType::ACL_SRC_2))
    {
        _bias = this->tensors().get_const_tensor(TensorType::ACL_SRC_2);
    }
    _dst = this->tensors().get_const_tensor(TensorType::ACL_DST_0);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _weight, _dst);
}

std::string ClTemplateDepthwiseConv2d::get_name() const
{
    return "depthwise_conv2d";
}

std::string ClTemplateDepthwiseConv2d::get_component_code(const ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);

    constexpr int height_idx = 2; // Data Layout is NHWC

    std::string code = R"_(
//------------------ START KERNEL {{meta_kernel_id}} ---------------------
// IN_0(src)            {{src}}
// IN_1(wei)            {{weight}}
)_";

    if(_bias != nullptr && _bias->has_valid_id())
    {
        code += R"_(
// IN_1(bia)            {{bias}}
)_";
    }

    code += R"_(
// OUT(dst, accum)      {{dst}}

TILE({{ACC_DATA_TYPE}}, M0, N0, {{dst}});
TILE(uint, M0, 1, g_dst_indirect_y);

{
#define _IWEI_WIDTH {{WEI_WIDTH}}
#define _IWEI_HEIGHT {{WEI_HEIGHT}}
#define _IDST_WIDTH {{arg_dst}}_w
#define _IDST_HEIGHT {{arg_dst}}_h
#define _IM0_A M0_A
#define _IN0_A N0_A
#define _IM0_B _IWEI_WIDTH
#define _IN0_B N0
#define _IBOUNDARY_CHECK (!((_IWEI_WIDTH == 1 && _IWEI_HEIGHT == 1 && {{PAD_LEFT}} == 0 && {{PAD_TOP}} == 0 && M0 == 1)))
)_";

    code += R"_(
    const int yo = g_ind_2 % {{arg_dst}}_h;
    const int bout = g_ind_2 / {{arg_dst}}_h;
)_";

    code += R"_(

    int xi = g_ind_1 * {{STRIDE_X}};
    int yi = yo * {{STRIDE_Y}};
    xi -= {{PAD_LEFT}};
    yi -= {{PAD_TOP}};

    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        {{dst}}[i].v = 0;
    })
)_";

    if(_weight->dimension(height_idx) < 5)
    {
        code += R"_(
    LOOP_UNROLLING(int, yk, 0, 1, _IWEI_HEIGHT,
)_";
    }
    else
    {
        code += R"_(
    for(int yk = 0; yk < _IWEI_HEIGHT; ++yk)
)_";
    }

    code += R"_(
    {
        TILE({{SRC_DATA_TYPE}}, _IM0_A, _IN0_A, a);

        LOOP_UNROLLING(int, i, 0, 1, _IM0_A,
        {
            a[i].v = 0;
        })

        T_LOAD_NHWC_WITH_DILATION({{SRC_DATA_TYPE}}, 1, _IM0_A, _IN0_A, {{SRC_TENSOR_TYPE}}, {{src}}, bout, yi + yk * {{DILATION_Y}}, xi, (g_ind_0 / {{DEPTH_MULTIPLIER}}), {{src}}_w, {{src}}_h, {{DILATION_X}}, 1, _IBOUNDARY_CHECK, a);

        TILE({{WEI_DATA_TYPE}}, _IM0_B, _IN0_B, b);

        T_LOAD({{WEI_DATA_TYPE}}, _IM0_B, _IN0_B, {{WEI_TENSOR_TYPE}}, {{weight}}, g_ind_0, yk * _IM0_B, 1, {{weight}}_stride_y, b);

        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            LOOP_UNROLLING(int, xk, 0, 1, _IWEI_WIDTH,
            {
)_";

    if(!_settings.is_fma_available())
    {
        code += R"_(
                {{dst}}[m0].v += a[xk + m0].v * b[xk].v;
)_";
    }
    else
    {
        code += R"_(
                {{dst}}[m0].v = fma(a[xk + m0].v, b[xk].v, {{dst}}[m0].v);
)_";
    }

    code += R"_(
            })
        })
    }
)_";

    if(_weight->dimension(height_idx) < 5)
    {
        code += R"_(
    )
)_";
    }

    if(_bias && _bias->has_valid_id())
    {
        code += R"_(
        TILE({{BIA_DATA_TYPE}}, 1, N0, {{bias}});

        T_LOAD({{BIA_DATA_TYPE}}, 1, N0, BUFFER, {{bias}}, g_ind_0, 0, 0, 0, {{bias}});

        T_ELTWISE_BROADCAST_ADD_X({{ACC_DATA_TYPE}}, M0, N0, {{dst}}, {{bias}}, {{dst}});
)_";
    }

    code += R"_(
    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        g_dst_indirect_y[i].v = (uint)min((int)(g_ind_1 + i), (int)({{arg_dst}}_w) - 1);
        g_dst_indirect_y[i].v += (int)(g_ind_2 % {{arg_dst}}_h) * (int)({{arg_dst}}_w);
        g_dst_indirect_y[i].v += (int)(g_ind_2 / {{arg_dst}}_h) * (int)({{arg_dst}}_w * {{arg_dst}}_h);
    })
}
//------------------ END KERNEL {{meta_kernel_id}} ---------------------
)_";

    return code;
}

void ClTemplateDepthwiseConv2d::declare_variables(GpuKernelVariableTable &vtable, const ComponentGroup &comp_group) const
{
    const GpuKernelArgumentInfo::Type input_type = _settings.export_input_to_cl_image() ?
                                                       GpuKernelArgumentInfo::Type::Tensor_4D_t_Image :
                                                       GpuKernelArgumentInfo::Type::Tensor_4D_t_Buffer;

    vtable.declare_variable(
        _src,
        GpuKernelArgumentInfo(input_type),
        comp_group.is_intermediate_tensor(_src),
        "src");

    const GpuKernelArgumentInfo::Type weight_type = _settings.export_weights_to_cl_image() ?
                                                        GpuKernelArgumentInfo::Type::Tensor_4D_t_Image :
                                                        GpuKernelArgumentInfo::Type::Tensor_4D_t_Buffer;

    vtable.declare_variable(
        _weight,
        GpuKernelArgumentInfo(weight_type),
        comp_group.is_intermediate_tensor(_weight),
        "weight");

    if(_bias != nullptr && _bias->has_valid_id()) // optional bias
    {
        vtable.declare_variable(
            _bias,
            GpuKernelArgumentInfo(GpuKernelArgumentInfo::Type::Vector),
            comp_group.is_intermediate_tensor(_bias),
            "bias");
    }
    vtable.declare_variable(
        _dst,
        GpuKernelArgumentInfo(GpuKernelArgumentInfo::Type::Tensor_4D_t_Buffer),
        comp_group.is_intermediate_tensor(_dst),
        "dst");
}

TagLUT ClTemplateDepthwiseConv2d::get_tag_lut(const GpuKernelVariableTable &vtable, const ComponentGroup &comp_group) const
{
    TagLUT lut{};

    // Arguments and global shared variables
    lut["src"]    = vtable.get_variable(_src);
    lut["weight"] = vtable.get_variable(_weight);

    if(_bias != nullptr && _bias->has_valid_id()) // optional bias
    {
        lut["bias"]          = vtable.get_variable(_bias);
        lut["BIA_DATA_TYPE"] = get_cl_type_from_data_type(_bias->data_type());
    }
    lut["dst"] = vtable.get_variable(_dst);

    const auto dst_argument = vtable.get_variable(comp_group.get_any_dst_tensor());
    lut["arg_dst"]          = dst_argument.uniq_name;

    // Local build options
    lut["meta_kernel_id"] = id();
    lut["ACC_DATA_TYPE"]  = _src->data_type();
    lut["SRC_DATA_TYPE"]  = _src->data_type();
    lut["WEI_DATA_TYPE"]  = _weight->data_type();

    switch(vtable.get_variable(_src).kernel_argument_info.type)
    {
        case GpuKernelArgumentInfo::Type::Image_Export_To_ClImage2D:
        case GpuKernelArgumentInfo::Type::Image_3D_Export_To_ClImage2D:
        case GpuKernelArgumentInfo::Type::Tensor_4D_t_Image:
            lut["SRC_TENSOR_TYPE"] = "IMAGE";
            break;
        default:
            lut["SRC_TENSOR_TYPE"] = "BUFFER";
            break;
    }

    switch(vtable.get_variable(_weight).kernel_argument_info.type)
    {
        case GpuKernelArgumentInfo::Type::Image_Export_To_ClImage2D:
        case GpuKernelArgumentInfo::Type::Image_3D_Export_To_ClImage2D:
        case GpuKernelArgumentInfo::Type::Tensor_4D_t_Image:
            lut["WEI_TENSOR_TYPE"] = "IMAGE";
            break;
        default:
            lut["WEI_TENSOR_TYPE"] = "BUFFER";
            break;
    }

    // Data Layout is NHWC
    constexpr int width_idx  = 1;
    constexpr int height_idx = 2;

    lut["WEI_WIDTH"]  = _weight->dimension(width_idx);
    lut["WEI_HEIGHT"] = _weight->dimension(height_idx);

    lut["STRIDE_X"] = _attributes.stride().x();
    lut["STRIDE_Y"] = _attributes.stride().y();

    lut["PAD_LEFT"] = _attributes.pad().left;
    lut["PAD_TOP"]  = _attributes.pad().top;

    lut["DILATION_X"] = _attributes.dilation().x();
    lut["DILATION_Y"] = _attributes.dilation().y();

    lut["DEPTH_MULTIPLIER"] = _attributes.depth_multiplier();

    return lut;
}

CLBuildOptions ClTemplateDepthwiseConv2d::get_build_options(const ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);

    constexpr unsigned int width_idx = 1; // Data Layout is NHWC

    const unsigned int n0               = _settings.n0();
    const unsigned int m0               = _settings.m0();
    const unsigned int m0_a             = _weight->dimension(width_idx) + m0 - 1;
    const unsigned int n0_a             = _attributes.depth_multiplier() > 1 ? 1 : n0;
    const unsigned int partial_store_n0 = _dst->dimension(0) % n0;

    CLBuildOptions build_opts{};

    if(_settings.fast_relaxed_math())
    {
        build_opts.add_option("-cl-fast-relaxed-math");
    }
    else
    {
        // -cl-fast-relaxed-math also sets -cl-finite-math-only and -cl-unsafe-math-optimizations
        // to disable -cl-finite-math-only, we only include -cl-unsafe-math-optimizations
        build_opts.add_option("-cl-unsafe-math-optimizations");
    }

    build_opts.add_option("-DN0=" + support::cpp11::to_string(n0));
    build_opts.add_option("-DM0=" + support::cpp11::to_string(m0));
    build_opts.add_option("-DN0_A=" + support::cpp11::to_string(n0_a));
    build_opts.add_option("-DM0_A=" + support::cpp11::to_string(m0_a));
    build_opts.add_option("-DPARTIAL_N0=" + support::cpp11::to_string(partial_store_n0));

    return build_opts;
}

std::string ClTemplateDepthwiseConv2d::get_config_id() const
{
    std::string config_id{};

    config_id += support::cpp11::to_string(_src->dimension(0));
    config_id += "_";
    config_id += support::cpp11::to_string(_src->dimension(1));
    config_id += "_";
    config_id += support::cpp11::to_string(_src->dimension(2));
    config_id += "_";
    config_id += support::cpp11::to_string(_dst->dimension(0));
    config_id += "_";
    config_id += support::cpp11::to_string(_dst->dimension(1));
    config_id += "_";
    config_id += support::cpp11::to_string(_dst->dimension(2));
    config_id += "_";
    config_id += string_from_data_type(_src->data_type());

    return config_id;
}

std::set<std::string> ClTemplateDepthwiseConv2d::get_headers_list() const
{
    return std::set<std::string>{ "helpers.h", "tile_helpers.h" };
}

Window ClTemplateDepthwiseConv2d::get_window() const
{
    ARM_COMPUTE_ERROR_ON_MSG(_dst->tensor_shape().total_size() == 0U, "Destination tensor is not initialized");

    Window win = calculate_max_window(*_dst, Steps(_settings.n0(), _settings.m0()));
    return win.collapse(win, Window::DimZ);
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
