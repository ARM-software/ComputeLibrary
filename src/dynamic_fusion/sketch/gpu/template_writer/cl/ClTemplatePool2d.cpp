/*
 * Copyright (c) 2023-2024 Arm Limited.
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
#include "ClTemplatePool2d.h"

#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/StringUtils.h"

#include "src/core/helpers/WindowHelpers.h"
#include "src/dynamic_fusion/sketch/gpu/components/cl/ClComponentDirectConv2d.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGroup.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
namespace
{
// Shape indexes for NHWC Datalayout
constexpr static int32_t height_idx  = 2;
constexpr static int32_t width_idx   = 1;
constexpr static int32_t channel_idx = 0;
} // namespace
ClTemplatePool2d::ClTemplatePool2d(ComponentId                      id,
                                   const ArgumentPack<ITensorInfo> &tensors,
                                   const Attributes                &attributes,
                                   const Settings                  &settings)
    : IGpuTemplateComponentWriter{id, tensors}, _src{}, _dst{}, _attributes{attributes}, _settings{settings}
{
    _src = this->tensors().get_const_tensor(TensorType::ACL_SRC_0);
    _dst = this->tensors().get_const_tensor(TensorType::ACL_DST_0);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);
}

std::string ClTemplatePool2d::get_name() const
{
    return "pool2d";
}

std::string ClTemplatePool2d::get_component_code(const ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);

    // Condition to use 2x2 optimized kernel
    if (_attributes.pool_size() == Size2D(2, 2))
    {
        return get_2x2_kernel_code();
    }
    else
    {
        return get_MxN_kernel_code();
    }
}

std::string ClTemplatePool2d::get_MxN_kernel_code() const
{
    const auto pool_type          = _attributes.pool_type();
    const bool fp_mixed_precision = (_src->data_type() == DataType::F16) && pool_type != PoolingType::MAX;

    // Define pool op macro.
    std::string pool_op = (pool_type == PoolingType::AVG) ? R"_(#define POOL_OP(x,y) ((x) + (y)))_"
                                                          : R"_(#define POOL_OP(x,y) (fmax((x), (y))) )_";

    // Kernel start
    // Note: If C is not multiple of N0, we shift back of PARTIAL_N0 elements to compute the leftover elements for get_global_id(0) == 0
    // Note: If C is less than N0, N0 should be SHRINKED to the closest smaller N0. This operation is performed on the host side
    std::string code = R"_(
//------------------ START KERNEL {{meta_kernel_id}} ---------------------
// IN_0(src)            {{src}}
// OUT(dst, accum)      {{dst}}

{
    const int idx_out_c = g_ind_0;
    const int idx_out_w = g_ind_1;
)_";

    // Add macro for POOL_OP
    code += "\n" + pool_op + "\n";

    code += R"_(
    const int idx_out_h = g_ind_2 % {{DST_HEIGHT}};
    const int idx_out_n = g_ind_2 / {{DST_HEIGHT}};
)_";

    // Define common variables.
    code += R"_(
    __global unsigned char *in_base_ptr = {{src}}_ptr + {{src}}_offset_first_element_in_bytes + idx_out_c * sizeof({{DATA_TYPE}}) + idx_out_n * {{src}}_stride_w;

    __global unsigned char *out_base_ptr = {{dst}}_ptr + {{dst}}_offset_first_element_in_bytes + idx_out_c * sizeof({{DATA_TYPE}}) + idx_out_w * {{dst}}_stride_y + idx_out_h * {{dst}}_stride_z + idx_out_n * {{dst}}_stride_w;

    VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0)
    res0 = {{INITIAL_VALUE}};

    const int idx_in_w = idx_out_w * {{STRIDE_X}} - {{PAD_X}};
    const int idx_in_h = idx_out_h * {{STRIDE_Y}} - {{PAD_Y}};

    const int pool_x_s = max((int)0, -idx_in_w);
    const int pool_x_e = min((int){{POOL_SIZE_X}}, (int){{SRC_WIDTH}} - idx_in_w);
    const int pool_y_s = max((int)0, -idx_in_h);
    const int pool_y_e = min((int){{POOL_SIZE_Y}}, (int){{SRC_HEIGHT}} - idx_in_h);
)_";

    // Determine filter size depending on if padding is excluded or not
    if (_attributes.exclude_padding())
    {
        code += R"_(
    const int filter_size = (pool_y_e - pool_y_s) * (pool_x_e - pool_x_s);
)_";
    }
    else
    {
        code += R"_(
    const int filter_size = {{POOL_SIZE_X}} * {{POOL_SIZE_Y}};
)_";
    }

    // Loop through pool size
    // if global pooling
    if (_attributes.pool_size().x() == _src->dimension(width_idx) &&
        _attributes.pool_size().y() == _src->dimension(height_idx))
    {
        // Begin loop
        code += R"_(
    // Global pooling path
    for(int y = 0; y < {{POOL_SIZE_Y}}; ++y)
    {
    #pragma unroll 8
        for(int x = 0; x < {{POOL_SIZE_X}}; ++x)
        {
            VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0)
            data0;
)_";
    }
    else // if local pooling size
    {
        code += R"_(
    for(int y = pool_y_s; y < pool_y_e; ++y)
    {
    #pragma unroll 8
        for(int x = pool_x_s; x < pool_x_e; ++x)
        {
            VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0)
            data0;
)_";
    } // end else

    // if condition inside loop - use 32bit acc if mixed_precision.
    // End loop through pooling section.
    if (fp_mixed_precision)
    {
        // In case of FP_MIXED_PRECISION, ACC_DATA_TYPE is != DATA_TYPE
        code += R"_(
            data0 = CONVERT(VLOAD(N0)(0, (__global {{DATA_TYPE}} *)(in_base_ptr + (x + idx_in_w) * {{src}}_stride_y + (y + idx_in_h) * {{src}}_stride_z)), VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0));
            res0 = POOL_OP(res0, data0);
        }
    }
)_";
    }
    else // load data, compute result and end loop
    {
        code += R"_(
            data0 = VLOAD(N0)(0, (__global {{DATA_TYPE}} *)(in_base_ptr + (x + idx_in_w) * {{src}}_stride_y + (y + idx_in_h) * {{src}}_stride_z));
            res0 = POOL_OP(res0, data0);
        }
    }
)_";
    }

    // For Pool AVG ONLY, divide pool output by filter size
    if (pool_type == PoolingType::AVG)
    {
        code += R"_(
    res0 /= (VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0))filter_size;
)_";
    }

    // If mixed precision convert datatype before storing. Then end kernel.
    if (fp_mixed_precision)
    {
        code += R"_(
    VEC_DATA_TYPE({{DATA_TYPE}}, N0)
    res_converted0 = CONVERT(res0, VEC_DATA_TYPE({{DATA_TYPE}}, N0));
    STORE_VECTOR_SELECT(res_converted, {{DATA_TYPE}}, out_base_ptr, N0, PARTIAL_N0, (PARTIAL_N0 != 0) && g_ind_0 == 0);
)_";
    }
    else
    {
        // Store data
        code += R"_(
    STORE_VECTOR_SELECT(res, {{DATA_TYPE}}, out_base_ptr, N0, PARTIAL_N0, (PARTIAL_N0 != 0) && g_ind_0 == 0);
)_";
    }

    code += R"_(
//------------------ END KERNEL {{meta_kernel_id}} ---------------------
}
)_";

    return code;
}

std::string ClTemplatePool2d::get_2x2_kernel_code() const
{
    const auto  pool_type          = _attributes.pool_type();
    const bool  fp_mixed_precision = (_src->data_type() == DataType::F16) && pool_type != PoolingType::MAX;
    std::string pool_op            = (pool_type == PoolingType::AVG) ? R"_(#define POOL_OP(x,y) ((x) + (y)))_"
                                                                     : R"_(#define POOL_OP(x,y) (fmax((x), (y))) )_";

    std::string code = R"_(
//------------------ START KERNEL {{meta_kernel_id}} ---------------------
// IN_0(src)            {{src}}
// OUT(dst, accum)      {{dst}}

#define SELECT_TYPE SELECT_VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0)

{
    const int idx_out_c = g_ind_0;
    const int idx_out_w = g_ind_1;
)_";

    // Add pool op macro
    code += "\n" + pool_op + "\n";

    // If batch size != 1, the batch size dimension is collapsed over the height dimension
    code += R"_(
    const int idx_out_h = g_ind_2 % {{DST_HEIGHT}};
    const int idx_out_n = g_ind_2 / {{DST_HEIGHT}};
)_";

    code += R"_(
    const int idx_in_w = idx_out_w * {{STRIDE_X}} - {{PAD_X}};
    const int idx_in_h = idx_out_h * {{STRIDE_Y}} - {{PAD_Y}};

    __global unsigned char *in_base_ptr = {{src}}_ptr + {{src}}_offset_first_element_in_bytes + idx_out_c * sizeof({{DATA_TYPE}}) + idx_out_n * {{src}}_stride_w;
    __global unsigned char *out_base_ptr = {{dst}}_ptr + {{dst}}_offset_first_element_in_bytes + idx_out_c * sizeof({{DATA_TYPE}}) + idx_out_w * {{dst}}_stride_y + idx_out_h * {{dst}}_stride_z + idx_out_n *
                                           {{dst}}_stride_w;
    const int pool_x_s = max((int)0, -idx_in_w);
    const int pool_x_e = min((int)2, (int){{SRC_WIDTH}} - idx_in_w);
    const int pool_y_s = max((int)0, -idx_in_h);
    const int pool_y_e = min((int)2, (int){{SRC_HEIGHT}} - idx_in_h);

    const int filter_size = (pool_x_e - pool_x_s) * (pool_y_e - pool_y_s);
    const int x0 = pool_x_s + idx_in_w;
    const int y0 = pool_y_s + idx_in_h;
    const int x1 = pool_x_e - 1 + idx_in_w;
    const int y1 = pool_y_e - 1 + idx_in_h;

    REPEAT_VAR_INIT_TO_CONST(4, VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0), data, 0);
)_";

    if (fp_mixed_precision)
    {
        // In case of FP_MIXED_PRECISION, ACC_DATA_TYPE is != DATA_TYPE
        code += R"_(
    data0 = CONVERT(VLOAD(N0)(0, (__global {{DATA_TYPE}} *)(in_base_ptr + x0 * {{src}}_stride_y + y0 * {{src}}_stride_z)), VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0));
    data1 = CONVERT(VLOAD(N0)(0, (__global {{DATA_TYPE}} *)(in_base_ptr + x1 * {{src}}_stride_y + y0 * {{src}}_stride_z)), VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0));
    data2 = CONVERT(VLOAD(N0)(0, (__global {{DATA_TYPE}} *)(in_base_ptr + x0 * {{src}}_stride_y + y1 * {{src}}_stride_z)), VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0));
    data3 = CONVERT(VLOAD(N0)(0, (__global {{DATA_TYPE}} *)(in_base_ptr + x1 * {{src}}_stride_y + y1 * {{src}}_stride_z)), VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0));
)_";
    }
    else
    {
        code += R"_(
    data0         = VLOAD(N0)(0, (__global {{DATA_TYPE}} *)(in_base_ptr + x0 * {{src}}_stride_y + y0 * {{src}}_stride_z));
    data1         = VLOAD(N0)(0, (__global {{DATA_TYPE}} *)(in_base_ptr + x1 * {{src}}_stride_y + y0 * {{src}}_stride_z));
    data2         = VLOAD(N0)(0, (__global {{DATA_TYPE}} *)(in_base_ptr + x0 * {{src}}_stride_y + y1 * {{src}}_stride_z));
    data3         = VLOAD(N0)(0, (__global {{DATA_TYPE}} *)(in_base_ptr + x1 * {{src}}_stride_y + y1 * {{src}}_stride_z));
)_";
    }

    if (pool_type != PoolingType::MAX)
    {
        // Make invalid the values loaded if the x or y coordinate was clamped (out-of-bound)
        code += R"_(
    if(filter_size != 4)
    {
        SELECT_TYPE cond_w_s = (SELECT_TYPE)idx_in_w < (SELECT_TYPE)0;
        SELECT_TYPE cond_w_e = (SELECT_TYPE)idx_in_w >= (SELECT_TYPE)({{SRC_WIDTH}} - 1);
        SELECT_TYPE cond_h_s = (SELECT_TYPE)idx_in_h < (SELECT_TYPE)0;
        SELECT_TYPE cond_h_e = (SELECT_TYPE)idx_in_h >= (SELECT_TYPE)({{SRC_HEIGHT}} - 1);

        data0 = select(data0, (VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0)){{INITIAL_VALUE}}, (SELECT_TYPE)(cond_w_s | cond_h_s));
        data1 = select(data1, (VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0)){{INITIAL_VALUE}}, (SELECT_TYPE)(cond_w_e | cond_h_s));
        data2 = select(data2, (VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0)){{INITIAL_VALUE}}, (SELECT_TYPE)(cond_w_s | cond_h_e));
        data3 = select(data3, (VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0)){{INITIAL_VALUE}}, (SELECT_TYPE)(cond_w_e | cond_h_e));
    }
)_";
    }

    code += R"_(
    VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0)
    res0 = data0;
    res0 = POOL_OP(res0, data1);
    res0 = POOL_OP(res0, data2);
    res0 = POOL_OP(res0, data3);
)_";

    if (pool_type == PoolingType::AVG)
    {
        // If avg pooling divide result accordingly.
        if (_attributes.exclude_padding())
        {
            code += R"_(
    res0 /= (VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0))filter_size;
)_";
        }
        else
        {
            code += R"_(
    res0 /= (VEC_DATA_TYPE({{ACC_DATA_TYPE}}, N0))4;
)_";
        }
    }

    // Store result
    if (fp_mixed_precision)
    {
        code += R"_(
    VEC_DATA_TYPE({{DATA_TYPE}}, N0)
    res_converted0 = CONVERT(res0, VEC_DATA_TYPE({{DATA_TYPE}}, N0));
    STORE_VECTOR_SELECT(res_converted, {{DATA_TYPE}}, out_base_ptr, N0, PARTIAL_N0, (PARTIAL_N0 != 0) && g_ind_0 == 0);
)_";
    }
    else
    {
        code += R"_(
    STORE_VECTOR_SELECT(res, {{DATA_TYPE}}, out_base_ptr, N0, PARTIAL_N0, (PARTIAL_N0 != 0) && g_ind_0 == 0);
)_";
    }

    code += R"_(
    //------------------ END KERNEL {{meta_kernel_id}} ---------------------
}
#undef SELECT_TYPE
)_";

    return code;
}

void ClTemplatePool2d::declare_variables(GpuKernelVariableTable &vtable, const ComponentGroup &comp_group) const
{
    vtable.declare_variable(comp_group, _src, GpuKernelArgumentInfo(GpuKernelArgumentInfo::Type::Tensor_4D_t_Buffer),
                            "src");

    vtable.declare_variable(comp_group, _dst, GpuKernelArgumentInfo(GpuKernelArgumentInfo::Type::Tensor_4D_t_Buffer),
                            "dst");
}

TagLUT ClTemplatePool2d::get_tag_lut(const GpuKernelVariableTable &vtable, const ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);

    TagLUT lut{};
    // Arguments and global shared variables
    lut["src"] = vtable.get_variable(_src);
    lut["dst"] = vtable.get_variable(_dst);

    // Local build options
    lut["meta_kernel_id"] = id();

    // Retrieve relevant data
    const auto padding   = _attributes.pad();
    const auto stride    = _attributes.stride();
    const auto pool_size = _attributes.pool_size();
    const auto data_type = _src->data_type();
    const auto use_fp_mixed_precision =
        (_src->data_type() == DataType::F16) && _attributes.pool_type() != PoolingType::MAX;
    const std::string max_initial_value =
        _settings.use_inf_as_limit() ? "(-INFINITY)"
                                     : float_to_string_with_full_precision(std::numeric_limits<float>::lowest());

    // pool specific
    lut["STRIDE_X"]    = stride.x();
    lut["STRIDE_Y"]    = stride.y();
    lut["PAD_X"]       = padding.left;
    lut["PAD_Y"]       = padding.top;
    lut["POOL_SIZE_X"] = pool_size.width;
    lut["POOL_SIZE_Y"] = pool_size.height;

    // Datatypes and variables
    lut["ACC_DATA_TYPE"] = get_cl_type_from_data_type(
        (use_fp_mixed_precision) ? (DataType::F32) : (data_type)); // Type of accumulators to use.
    lut["DATA_TYPE"]     = get_cl_type_from_data_type(data_type);
    lut["SRC_WIDTH"]     = _src->dimension(width_idx);
    lut["SRC_HEIGHT"]    = _src->dimension(height_idx);
    lut["INITIAL_VALUE"] = (_attributes.pool_type() == PoolingType::MAX) ? max_initial_value : std::string("0");

    // Tensor specific data
    lut["DST_HEIGHT"] = _dst->dimension(height_idx);

    return lut;
}

CLBuildOptions ClTemplatePool2d::get_build_options(const ComponentGroup &comp_group) const
{
    const auto         root_window      = comp_group.get_root_component()->template_writer()->get_window();
    const unsigned int n0               = root_window.x().step();
    const unsigned int partial_store_n0 = _dst->dimension(0) % n0;

    CLBuildOptions build_opts{};
    build_opts.add_option("-DN0=" + support::cpp11::to_string(n0));
    build_opts.add_option("-DPARTIAL_N0=" + support::cpp11::to_string(partial_store_n0));

    return build_opts;
}

std::string ClTemplatePool2d::get_config_id() const
{
    const DataType   data_type   = _src->data_type();
    const DataLayout data_layout = _src->data_layout();

    std::string config_id{};
    config_id += "pooling_layer_2d_";
    config_id += lower_string(string_from_data_type(data_type));
    config_id += "_";
    config_id += lower_string(string_from_data_layout(data_layout));
    config_id += "_";
    config_id += support::cpp11::to_string(_dst->dimension(width_idx));
    config_id += "_";
    config_id += support::cpp11::to_string(_dst->dimension(height_idx));
    config_id += "_";
    config_id += support::cpp11::to_string(_dst->dimension(channel_idx));

    return config_id;
}

std::set<std::string> ClTemplatePool2d::get_headers_list() const
{
    return std::set<std::string>{"helpers.h", "tile_helpers.h", "repeat.h"};
}

Window ClTemplatePool2d::get_window() const
{
    ARM_COMPUTE_ERROR_ON_MSG(_dst->tensor_shape().total_size() == 0U, "Destination tensor is not initialized");
    const auto         output_shape = _dst->tensor_shape();
    const unsigned int vec_size = adjust_vec_size(((_dst->data_type() == DataType::F32) ? 2 : 4), _dst->dimension(0));

    // Create and configure kernel window
    auto win = calculate_max_window(output_shape, Steps(vec_size));
    win      = win.collapse_if_possible(win, Window::DimZ); // collapse window on batch size.
    return win;
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
