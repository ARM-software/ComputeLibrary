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
#include "ClTemplateDirectConv2d.h"

#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGroup.h"
#include "src/dynamic_fusion/sketch/gpu/components/cl/ClComponentDirectConv2d.h"

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
ClTemplateDirectConv2d::ClTemplateDirectConv2d(ComponentId                      id,
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

std::string ClTemplateDirectConv2d::get_name() const
{
    return "direct_conv2d";
}

std::string ClTemplateDirectConv2d::get_component_code(const ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);

    const auto channel_idx   = get_data_layout_dimension_index(_src->data_layout(), DataLayoutDimension::CHANNEL);
    const auto k0            = adjust_vec_size(_settings.direct_conv_descriptor().k0, _src->dimension(channel_idx));
    const bool leftover_loop = (_src->dimension(channel_idx) % k0) != 0;

    std::string code = R"_(
//------------------ START KERNEL {{meta_kernel_id}} ---------------------
// IN_0(src)            {{src}}
// IN_1(wei)            {{weight}}
)_";
    if(_bias && _bias->has_valid_id())
    {
        code += R"_(
// IN_1(bia)            {{bias}}
)_";
    }
    code += R"_(
// OUT(dst, accum)      {{dst}}

TILE(uint, M0, 1, g_dst_indirect_y);

{
#define _IWEI_WIDTH {{WEI_WIDTH}}
#define _IWEI_HEIGHT {{WEI_HEIGHT}}
#define _ISRC_WIDTH {{SRC_WIDTH}}
#define _ISRC_HEIGHT {{SRC_HEIGHT}}
#define _ISRC_CHANNELS {{SRC_CHANNELS}}
#define _IDST_WIDTH {{DST_WIDTH}}
#define _IDST_HEIGHT {{DST_HEIGHT}}
#define _IDST_CHANNELS {{DST_CHANNELS}}
#define _IY_MULTIPLIER (_IWEI_WIDTH * _IWEI_HEIGHT)

    TILE(int, M0, 1, xi);
    TILE(int, M0, 1, yi);

    // Convert the linear index to coordinate
    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        xi[0].s[i] = ((g_ind_1 + i) % _IDST_WIDTH) * {{STRIDE_X}};
        yi[0].s[i] = ((g_ind_1 + i) / _IDST_WIDTH) * {{STRIDE_Y}};
        xi[0].s[i] -= {{PAD_LEFT}};
        yi[0].s[i] -= {{PAD_TOP}};
    })

    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        {{dst}}[i].v = 0;
    })

    for(int i = 0; i < (_IWEI_WIDTH * _IWEI_HEIGHT); ++i)
    {
        int xk = i % _IWEI_WIDTH;
        int yk = i / _IWEI_WIDTH;

        TILE(int, 1, M0, my);

        LOOP_UNROLLING(int, i, 0, 1, M0,
        {
            int x_s    = xi[0].s[i] + xk;
            int y_s    = yi[0].s[i] + yk;
            my[0].s[i] = x_s + y_s *_ISRC_WIDTH;
            my[0].s[i] = my[0].s[i] + g_ind_2 * (int)(_ISRC_WIDTH * _ISRC_HEIGHT);
            my[0].s[i] = select(-1, my[0].s[i], x_s >= 0);
            my[0].s[i] = select(-1, my[0].s[i], x_s < _ISRC_WIDTH);
            my[0].s[i] = select(-1, my[0].s[i], y_s >= 0);
            my[0].s[i] = select(-1, my[0].s[i], y_s < _ISRC_HEIGHT);
        })

        int ck = 0;
        for(; ck <= (_ISRC_CHANNELS - K0); ck += K0)
        {
            TILE({{SRC_DATA_TYPE}}, M0, K0, a);
            TILE({{WEI_DATA_TYPE}}, N0, K0, b);

            LOOP_UNROLLING(int, i, 0, 1, M0,
            {
                a[i].v = {{ZERO_VALUE}};
            })

            LOOP_UNROLLING(int, i, 0, 1, N0,
            {
                b[i].v = {{ZERO_VALUE}};
            })

            T_LOAD2D_INDIRECT({{SRC_DATA_TYPE}}, M0, K0, {{SRC_TENSOR_TYPE}}, {{src}}, ck, {{src}}_stride_y, my, a);

            T_LOAD({{WEI_DATA_TYPE}}, N0, K0, {{WEI_TENSOR_TYPE}}, {{weight}}, ck, g_ind_0 * _IY_MULTIPLIER + i, _IY_MULTIPLIER, {{weight}}_stride_y, b);

            T_MMUL({{SRC_DATA_TYPE}}, {{WEI_DATA_TYPE}}, {{ACC_DATA_TYPE}}, M0, N0, K0, NT, T, a, b, {{dst}});
        }
)_";

    if(leftover_loop)
    {
        code += R"_(
        for(; ck < _ISRC_CHANNELS; ++ck)
        {
            TILE({{SRC_DATA_TYPE}}, M0, 1, a);
            TILE({{WEI_DATA_TYPE}}, N0, 1, b);

            LOOP_UNROLLING(int, i, 0, 1, M0,
            {
                a[i].v = {{ZERO_VALUE}};
            })

            LOOP_UNROLLING(int, i, 0, 1, N0,
            {
                b[i].v = {{ZERO_VALUE}};
            })

            T_LOAD2D_INDIRECT({{SRC_DATA_TYPE}}, M0, 1, {{SRC_TENSOR_TYPE}}, {{src}}, ck, {{src}}_stride_y, my, a);

            T_LOAD({{WEI_DATA_TYPE}}, N0, 1, BUFFER, {{weight}}, ck, g_ind_0 * _IY_MULTIPLIER + i, _IY_MULTIPLIER, {{weight}}_stride_y, b);

            T_MMUL({{SRC_DATA_TYPE}}, {{WEI_DATA_TYPE}}, {{ACC_DATA_TYPE}}, M0, N0, 1, NT, T, a, b, {{dst}});
        }
    )_";
}

code += R"_(
#undef _I_WEI_WIDTH
#undef _I_WEI_HEIGHT
#undef _ISRC_WIDTH
#undef _ISRC_HEIGHT
#undef _ISRC_CHANNELS
#undef _IDST_WIDTH
#undef _IDST_HEIGHT
#undef _IDST_CHANNELS
#undef _IY_MULTIPLIER

    }
)_";

    if(_bias && _bias->has_valid_id())
    {
        code += R"_(
        TILE({{BIA_DATA_TYPE}}, 1, N0, bias0);

        T_LOAD({{BIA_DATA_TYPE}}, 1, N0, BUFFER, {{bias}}, g_ind_0, 0, 1, 0, bias0);

        T_ELTWISE_BROADCAST_ADD_X({{ACC_DATA_TYPE}}, M0, N0, {{dst}}, bias0, {{dst}});
    )_";
}

code += R"_(
    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        g_dst_indirect_y[i].v = (uint)min(g_ind_1 + i, (int)({{DST_WIDTH}} * {{DST_HEIGHT}}) - 1);
        g_dst_indirect_y[i].v += g_ind_2 * (int)({{DST_WIDTH}} * {{DST_HEIGHT}});
    })
}
//------------------ END KERNEL {{meta_kernel_id}} ---------------------
)_";
    return code;
}

void ClTemplateDirectConv2d::declare_variables(GpuKernelVariableTable &vtable, const ComponentGroup &comp_group) const
{
    vtable.declare_variable(
        comp_group,
        _src,
        GpuKernelArgumentInfo(GpuKernelArgumentInfo::Type::Tensor_4D_t_Buffer),
        "src");

    const GpuKernelArgumentInfo::Type weight_type = _settings.export_to_cl_image() ? GpuKernelArgumentInfo::Type::Tensor_4D_t_Image : GpuKernelArgumentInfo::Type::Tensor_4D_t_Buffer;
    vtable.declare_variable(
        comp_group,
        _weight,
        GpuKernelArgumentInfo(weight_type),
        "weight");

    if(_bias && _bias->has_valid_id()) // optional bias
    {
        vtable.declare_variable(
            comp_group,
            _bias,
            GpuKernelArgumentInfo(GpuKernelArgumentInfo::Type::Vector),
            "bias");
    }
    vtable.declare_variable(
        comp_group,
        _dst,
        GpuKernelArgumentInfo(common_tensor_type),
        "dst");
}

TagLUT ClTemplateDirectConv2d::get_tag_lut(const GpuKernelVariableTable &vtable, const ComponentGroup &comp_group) const
{
    TagLUT lut{};
    // Arguments and global shared variables
    lut["src"]    = vtable.get_variable(_src);
    lut["weight"] = vtable.get_variable(_weight);

    if(_bias && _bias->has_valid_id()) // optional bias
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

    lut["SRC_TENSOR_TYPE"] = "BUFFER";
    switch(vtable.get_variable(_weight).kernel_argument_info.type)
    {
        case GpuKernelArgumentInfo::Type::Image_Export_To_ClImage2D:
        case GpuKernelArgumentInfo::Type::Image_3D_Export_To_ClImage2D:
        case GpuKernelArgumentInfo::Type::Tensor_4D_t_Image:
    {
        lut["WEI_TENSOR_TYPE"] = "IMAGE";
        break;
    }
        default:
    {
        lut["WEI_TENSOR_TYPE"] = "BUFFER";
        break;
    }
    }
    const auto width_idx  = 1;
    const auto height_idx = 2;
    const auto channel_idx = 0;

    lut["SRC_WIDTH"] = _src->dimension(width_idx);
    lut["SRC_HEIGHT"] = _src->dimension(height_idx);
    lut["SRC_CHANNELS"] = _src->dimension(channel_idx);

    lut["WEI_WIDTH"]      = _weight->dimension(width_idx);
    lut["WEI_HEIGHT"]     = _weight->dimension(height_idx);

    lut["DST_WIDTH"] = _dst->dimension(width_idx);
    lut["DST_HEIGHT"] = _dst->dimension(height_idx);
    lut["DST_CHANNELS"] = _dst->dimension(channel_idx);

    lut["STRIDE_X"] = _attributes.stride().x();
    lut["STRIDE_Y"] = _attributes.stride().y();

    lut["PAD_LEFT"] = _attributes.pad().left;
    lut["PAD_TOP"]  = _attributes.pad().top;

    lut["ZERO_VALUE"] = 0;

    return lut;
}

CLBuildOptions ClTemplateDirectConv2d::get_build_options(const ComponentGroup &comp_group) const
{
    const unsigned int channel_idx = get_data_layout_dimension_index(_src->data_layout(), DataLayoutDimension::CHANNEL);

    const auto         root_window      = comp_group.get_root_component()->template_writer()->get_window();
    const unsigned int n0               = root_window.x().step();
    const unsigned int m0               = root_window.y().step();
    const unsigned int k0               = adjust_vec_size(_settings.direct_conv_descriptor().k0, _src->dimension(channel_idx));
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
    build_opts.add_option("-DK0=" + support::cpp11::to_string(k0));
    build_opts.add_option("-DPARTIAL_N0=" + support::cpp11::to_string(partial_store_n0));

    return build_opts;
}

std::string ClTemplateDirectConv2d::get_config_id() const
{
    const DataType   data_type   = _src->data_type();
    const DataLayout data_layout = _src->data_layout();

    const unsigned int width_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int height_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    const unsigned int kernel_size = _weight->dimension(width_idx);

    std::string config_id{};
    config_id += lower_string(string_from_data_type(data_type));
    config_id += "_";
    config_id += support::cpp11::to_string(kernel_size);
    config_id += "_";
    config_id += support::cpp11::to_string(_attributes.stride().x());
    config_id += "_";
    config_id += support::cpp11::to_string(_attributes.stride().y());
    config_id += "_";
    config_id += support::cpp11::to_string(_dst->dimension(width_idx));
    config_id += "_";
    config_id += support::cpp11::to_string(_dst->dimension(height_idx));
    config_id += "_";
    config_id += lower_string(string_from_data_layout(data_layout));
    return config_id;
}

std::set<std::string> ClTemplateDirectConv2d::get_headers_list() const
{
    return std::set<std::string>{ "helpers.h", "tile_helpers.h" };
}

Window ClTemplateDirectConv2d::get_window() const
{
    ARM_COMPUTE_ERROR_ON_MSG(_dst->tensor_shape().total_size() == 0U, "Destination tensor is not initialized");

    const auto output_shape = _dst->tensor_shape();
    const auto desc         = _settings.direct_conv_descriptor();

    const unsigned int n0 = adjust_vec_size(desc.n0, output_shape[0]);
    const unsigned int m0 = adjust_vec_size(desc.m0, output_shape[1] * output_shape[2]);

    // Create and configure kernel window
    Window win = calculate_max_window(output_shape, Steps(n0, m0));

    const size_t dim_y_collapsed = ceil_to_multiple(output_shape[1] * output_shape[2], m0);
    win.set(Window::DimY, Window::Dimension(0, dim_y_collapsed, m0));
    win.set(Window::DimZ, Window::Dimension(0, output_shape.total_size_upper(3), 1));

    return win;
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
