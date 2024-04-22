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
#include "ClTemplateReshape.h"

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
constexpr unsigned int vector_size_byte_opencl = 16;

ClTemplateReshape::ClTemplateReshape(ComponentId id, const ArgumentPack<ITensorInfo> &tensors)
    : IGpuTemplateComponentWriter{id, tensors}, _src{}, _dst{}
{
    _src = this->tensors().get_const_tensor(TensorType::ACL_SRC_0);
    _dst = this->tensors().get_const_tensor(TensorType::ACL_DST_0);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);
}

std::string ClTemplateReshape::get_name() const
{
    return "reshape";
}

std::string ClTemplateReshape::get_component_code(const ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);
    std::string code;

    code = R"_(
//------------------ START KERNEL {{meta_kernel_id}} ---------------------

// IN(src)              {{src}}
// OUT(dst, accum)      {{dst}}

TILE(uint, M0, 1, g_dst_indirect_y);
{
    __global uchar * base_src_ptr = {{src}}_ptr + {{src}}_offset_first_element_in_bytes;
    const int tile_vertical_idx = g_ind_1 * {{arg_dst}}_c + g_ind_2 * {{arg_dst}}_c * {{arg_dst}}_w;
    LOOP_UNROLLING(int, _m0, 0, 1, M0,
    {
        const int row_idx = _m0 * {{arg_dst}}_c + tile_vertical_idx;
        const int tile_horizontal_idx = g_ind_0 + row_idx;
        LOOP_UNROLLING(int, _n0, 0, 1, N0,
        {
            {{src}}_ptr = base_src_ptr;
            const int linear_idx = tile_horizontal_idx + _n0;
            const int in_id_x = linear_idx % {{src}}_c;
            const int in_id_y = (linear_idx / {{src}}_c) % {{src}}_w;
            const int in_id_z = linear_idx / ({{src}}_c * {{src}}_w);
            {{src}}_ptr += in_id_x * sizeof({{DATA_TYPE}}) + in_id_y * {{src}}_stride_y + in_id_z * {{src}}_stride_z;
            {{dst}}[_m0].s[_n0] = *((__global {{DATA_TYPE}} *){{src}}_ptr);
        })
    })

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

void ClTemplateReshape::declare_variables(GpuKernelVariableTable &vtable, const ComponentGroup &comp_group) const
{
    vtable.declare_variable(comp_group, _src,
                            GpuKernelArgumentInfo(common_tensor_type), // GpuKernelArgumentInfo::Type::Image_3D
                            "src");

    vtable.declare_variable(comp_group, _dst, GpuKernelArgumentInfo(common_tensor_type), "dst");
}

TagLUT ClTemplateReshape::get_tag_lut(const GpuKernelVariableTable &vtable, const ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);
    TagLUT lut{};

    // Arguments and global shared variables
    lut["src"]            = vtable.get_variable(_src);
    lut["dst"]            = vtable.get_variable(_dst);
    lut["arg_dst"]        = vtable.get_variable(comp_group.get_any_dst_tensor());
    lut["meta_kernel_id"] = id();
    lut["DATA_TYPE"]      = get_cl_type_from_data_type(_dst->data_type());

    return lut;
}

CLBuildOptions ClTemplateReshape::get_build_options(const ComponentGroup &comp_group) const
{
    CLBuildOptions     build_opts{};
    const auto         root_window      = comp_group.get_root_component()->template_writer()->get_window();
    const unsigned int n0               = root_window.x().step();
    const unsigned int m0               = root_window.y().step();
    const unsigned int partial_store_n0 = _dst->dimension(0) % n0;
    build_opts.add_option("-DN0=" + support::cpp11::to_string(n0));
    build_opts.add_option("-DM0=" + support::cpp11::to_string(m0));
    build_opts.add_option("-DPARTIAL_N0=" + support::cpp11::to_string(partial_store_n0));

    return build_opts;
}

std::string ClTemplateReshape::get_config_id() const
{
    std::string config_id{};
    config_id += lower_string(string_from_data_type(_dst->data_type()));
    config_id += "_";
    config_id += support::cpp11::to_string(_dst->dimension(0));
    config_id += "_";
    config_id += support::cpp11::to_string(_dst->dimension(1));

    return config_id;
}

std::set<std::string> ClTemplateReshape::get_headers_list() const
{
    return std::set<std::string>{"helpers.h", "tile_helpers.h"};
}

Window ClTemplateReshape::get_window() const
{
    ARM_COMPUTE_ERROR_ON_MSG(_dst->tensor_shape().total_size() == 0U, "Destination tensor is not initialized");
    const unsigned int n0  = adjust_vec_size(vector_size_byte_opencl / _dst->element_size(), _dst->dimension(0));
    Window             win = calculate_max_window(*_dst, Steps(n0));
    return win.collapse(win, Window::DimZ);
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
