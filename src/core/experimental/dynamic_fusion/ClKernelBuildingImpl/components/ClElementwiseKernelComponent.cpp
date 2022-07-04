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
#ifdef ENABLE_EXPERIMENTAL_DYNAMIC_FUSION

#include "src/core/experimental/dynamic_fusion/ClKernelBuildingImpl/components/ClElementwiseKernelComponent.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
ComponentType ClElementwiseKernelComponent::get_component_type() const
{
    return ComponentType::Simple;
}

std::set<std::string> ClElementwiseKernelComponent::get_headers_list() const
{
    return std::set<std::string> { "common/experimental/gemm_fused_post_ops/fp_mixed_precision_helpers.h", "tile_helpers.h" };
}

Window ClElementwiseKernelComponent::get_window() const
{
    const ITensorInfo *lhs_info = _blueprint->impl().get_kernel_argument_info(_lhs.arg_id);
    const ITensorInfo *rhs_info = _blueprint->impl().get_kernel_argument_info(_rhs.arg_id);
    ITensorInfo       *dst_info = _blueprint->impl().get_kernel_argument_info(_blueprint->impl().get_dst_id());

    ARM_COMPUTE_ERROR_ON_NULLPTR(lhs_info, rhs_info, dst_info);

    const std::pair<TensorShape, ValidRegion> broadcast_pair = ITensorInfo::broadcast_shape_and_valid_region(*lhs_info, *rhs_info);
    const TensorShape &out_shape = broadcast_pair.first;

    auto_init_if_empty(*dst_info, out_shape, 1, lhs_info->data_type());

    TensorShape output_shape = dst_info->tensor_shape();
    // Collapse Dim 1 (W) and Dim 2 (H) together, leave Dim 0 (C) and upper dimensions unchanged
    // This is in line with the collapsing convention used by Conv2d
    output_shape.collapse(2U, 1U);
    const unsigned int vector_size_byte_opencl           = 16;
    const unsigned int num_elems_processed_per_iteration = adjust_vec_size(vector_size_byte_opencl / dst_info->element_size(), dst_info->dimension(0));
    Window             win                               = calculate_max_window(output_shape, Steps(num_elems_processed_per_iteration));

    return win;
}

std::string ClElementwiseKernelComponent::get_component_code() const
{
    std::string code;
    const bool  is_root = _blueprint->impl().group(_lhs.arg_id) == SharedVarGroup::Argument && _blueprint->impl().group(_rhs.arg_id) == SharedVarGroup::Argument;

    if(is_root)
    {
        return R"_(
    //------------------ START KERNEL {{meta_kernel_id}} ELTWISE_OP ---------------------
    // IN_0(LHS)            {{lhs}}
    // IN_1(RHS)            {{rhs}}
    // OUT(dst, accum)      {{dst}}

    // dst = lhs + rhs (mix-precision, broadcast, boundary aware)
    TILE({{DATA_TYPE}}, M0, N0, {{dst}});
    {
        TILE({{DATA_TYPE}}, M0, N0, lhs_tile);
        TILE({{DATA_TYPE}}, M0, N0, rhs_tile);

        // Since mout maps to dimensions 1 (y) and dimension 2 (z) of the input tensor because of the collapsed window, bout maps to dimension 3 (w)
        {{lhs}}_offset_first_element_in_bytes += bout * {{lhs}}_stride_w;
        {{rhs}}_offset_first_element_in_bytes += bout * {{rhs}}_stride_w;

        T_LOAD({{DATA_TYPE}}, M0, N0, BUFFER, {{lhs}}, cout, mout, 1, {{lhs}}_stride_y, lhs_tile);
        T_LOAD({{DATA_TYPE}}, {{rhs_m0}}, {{rhs_n0}}, BUFFER, {{rhs}}, {{rhs_start_x}}, {{rhs_start_y}}, 1, {{rhs}}_stride_y, rhs_tile);

#if defined(IS_BROADCAST)
        T_ELTWISE_BROADCAST_{{ELTWISE_OP}}_X({{DATA_TYPE}}, M0, N0, lhs_tile, rhs_tile, {{dst}});
#else // !defined(IS_BROADCAST)
        T_ELTWISE_{{ELTWISE_OP}}({{DATA_TYPE}}, M0, N0, lhs_tile, rhs_tile, {{dst}});
#endif // defined(IS_BROADCAST)

    }
    //------------------ END KERNEL {{meta_kernel_id}} ELTWISE_OP ---------------------
)_";
    }
    else
    {
        return R"_(
    //------------------ START KERNEL {{meta_kernel_id}} ELTWISE_OP ---------------------
    // IN_0/Out(Accumulator)   {{acc}}
    // IN_1(Addend)        {{addend}}

    // acc = addend + acc (mix-precision, broadcast, boundary aware)
    {
        TILE({{DATA_TYPE}}, M0, N0, addend_tile);

        T_LOAD({{DATA_TYPE}}, {{rhs_m0}}, {{rhs_n0}}, BUFFER, {{addend}}, {{rhs_start_x}}, {{rhs_start_y}}, 1, {{addend}}_stride_y, addend_tile);

#if defined(IS_BROADCAST)
        T_ELTWISE_BROADCAST_{{ELTWISE_OP}}_X({{DATA_TYPE}}, M0, N0, {{acc}}, addend_tile, {{acc}});
#else // !defined(IS_BROADCAST)
        T_ELTWISE_{{ELTWISE_OP}}({{DATA_TYPE}}, M0, N0, {{acc}}, addend_tile, {{acc}});
#endif // defined(IS_BROADCAST)
    }
    //------------------ END KERNEL {{meta_kernel_id}} ELTWISE_OP ---------------------
)_";
    }
}

CLBuildOptions ClElementwiseKernelComponent::generate_build_options() const
{
    const auto t_rhs_info = _blueprint->impl().get_kernel_argument_info(_rhs.arg_id);
    const auto t_dst_info = _blueprint->impl().get_kernel_argument_info(_blueprint->impl().get_dst_id());

    CLBuildOptions     build_opts{};
    const auto         n0               = _blueprint->impl().get_execution_window().x().step();
    const auto         m0               = _blueprint->impl().get_execution_window().y().step();
    const unsigned int partial_store_n0 = t_dst_info->dimension(0) % n0;
    const bool         is_broadcast     = t_rhs_info->tensor_shape() != t_dst_info->tensor_shape();

    build_opts.add_option("-DM0=" + support::cpp11::to_string(m0));
    build_opts.add_option("-DN0=" + support::cpp11::to_string(n0));
    build_opts.add_option("-DPARTIAL_N0=" + support::cpp11::to_string(partial_store_n0));
    build_opts.add_option_if(is_broadcast, "-DIS_BROADCAST");

    return build_opts;
}

std::string ClElementwiseKernelComponent::generate_config_id() const
{
    auto        t_dst_info = _blueprint->impl().get_kernel_argument_info(_blueprint->impl().get_dst_id());
    std::string config_id{};
    config_id += lower_string(string_from_data_type(t_dst_info->data_type()));
    config_id += "_";
    config_id += support::cpp11::to_string(t_dst_info->dimension(0));
    config_id += "_";
    config_id += support::cpp11::to_string(t_dst_info->dimension(1));
    config_id += "_";
    config_id += lower_string(string_from_data_layout(t_dst_info->data_layout()));
    return config_id;
}

void ClElementwiseKernelComponent::allocate_shared_vars(SharedVarTable &vtable) const
{
    const bool is_root = _blueprint->impl().group(_lhs.arg_id) == SharedVarGroup::Argument && _blueprint->impl().group(_rhs.arg_id) == SharedVarGroup::Argument;
    vtable.add(_lhs, _blueprint->impl().group(_lhs.arg_id), ClKernelArgDescriptor(_lhs.arg_id, ClKernelTensorArgType::Tensor_4D_t_Buffer), "lhs");
    vtable.add(_rhs, _blueprint->impl().group(_rhs.arg_id), ClKernelArgDescriptor(_rhs.arg_id, ClKernelTensorArgType::Tensor_4D_t_Buffer), "rhs");
    if(is_root)
    {
        vtable.add(_dst, _blueprint->impl().group(_dst.arg_id), ClKernelArgDescriptor(_dst.arg_id, ClKernelTensorArgType::Tensor_4D_t_Buffer), "dst");
    }
}

ClElementwiseKernelComponent::TagLUT ClElementwiseKernelComponent::get_tag_lut(const SharedVarTable &vtable) const
{
    TagLUT     lut{};
    const auto t_dst_info = _blueprint->impl().get_kernel_argument_info(_blueprint->impl().get_dst_id());
    const auto t_rhs_info = _blueprint->impl().get_kernel_argument_info(_rhs.arg_id);
    // Arguments and global shared variables
    const bool is_root = _blueprint->impl().group(_lhs.arg_id) == SharedVarGroup::Argument && _blueprint->impl().group(_rhs.arg_id) == SharedVarGroup::Argument;
    if(is_root)
    {
        lut["lhs"] = vtable.get(_lhs);
        lut["rhs"] = vtable.get(_rhs);
        lut["dst"] = vtable.get(_dst);
    }
    else
    {
        // Determine which link is the accumulator
        Link accumulator;
        Link addend;
        if(_blueprint->impl().group(_lhs.arg_id) == SharedVarGroup::Automatic)
        {
            accumulator = _lhs;
            addend      = _rhs;
        }
        else if(_blueprint->impl().group(_rhs.arg_id) == SharedVarGroup::Automatic)
        {
            accumulator = _rhs;
            addend      = _lhs;
        }
        else
        {
            ARM_COMPUTE_ERROR("Invalid elementwise component linking");
        }
        lut["acc"]    = vtable.get(accumulator);
        lut["addend"] = vtable.get(addend);
    }
    // Local build options
    lut["meta_kernel_id"] = id();
    lut["DATA_TYPE"]      = get_cl_type_from_data_type(t_dst_info->data_type());

    switch(_desc.eltwise.op)
    {
        case ArithmeticOperation::DIV:
            lut["ELTWISE_OP"] = "DIV";
            break;
        case ArithmeticOperation::ADD:
            lut["ELTWISE_OP"] = "ADD";
            break;
        default:
            ARM_COMPUTE_ERROR("Arithmetic Operation not supported");
    }

    // Set broadcast parameters
    // PRE: All tensors are broadcast-compatible
    const bool is_broadcast = t_rhs_info->tensor_shape() != t_dst_info->tensor_shape();
    if(is_broadcast)
    {
        // Note that n0 maps to input tensor dimension 0, m0 maps to input dimensions 1 and 2 because of our collapse strategy
        if(t_rhs_info->dimension(0) == 1U && t_rhs_info->dimension(1) == 1U && t_rhs_info->dimension(2) == 1U) // Broadcast in X, Y, Z: collapsed rhs win [M0xN0] = [1x1]
        {
            lut["rhs_m0"]      = "1";
            lut["rhs_n0"]      = "1";
            lut["rhs_start_y"] = "0";
            lut["rhs_start_x"] = "0";
        }
        else if(t_rhs_info->dimension(1) == 1U && t_rhs_info->dimension(2) == 1U) // Broadcast in Y and Z: collapsed rhs win [M0xN0] = [1xN]
        {
            lut["rhs_m0"]      = "1";
            lut["rhs_n0"]      = "N0";
            lut["rhs_start_y"] = "0";
            lut["rhs_start_x"] = "cout";
        }
        else
        {
            ARM_COMPUTE_ERROR("Only support rhs broadcasting in all X, Y, Z dimensions, or just in Y and Z dimensions");
        }
    }
    else
    {
        lut["rhs_m0"]      = "M0";
        lut["rhs_n0"]      = "N0";
        lut["rhs_start_y"] = "mout";
        lut["rhs_start_x"] = "cout";
    }
    return lut;
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */