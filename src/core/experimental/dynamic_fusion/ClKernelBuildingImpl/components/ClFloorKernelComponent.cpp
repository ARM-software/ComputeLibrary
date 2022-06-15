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

#include "src/core/experimental/dynamic_fusion/ClKernelBuildingImpl/components/ClFloorKernelComponent.h"
#include "arm_compute/core/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
ComponentType ClFloorKernelComponent::get_component_type() const
{
    return ComponentType::Simple;
}

std::set<std::string> ClFloorKernelComponent::get_headers_list() const
{
    return std::set<std::string> { "common/experimental/gemm_fused_post_ops/fp_mixed_precision_helpers.h", "tile_helpers.h" };
}

Window ClFloorKernelComponent::get_window() const
{
    const ITensorInfo *src_info = _blueprint->impl().get_kernel_argument_info(_src.arg_id);
    ITensorInfo       *dst_info = _blueprint->impl().get_kernel_argument_info(_blueprint->impl().get_dst_id());

    ARM_COMPUTE_ERROR_ON_NULLPTR(src_info, dst_info);
    auto_init_if_empty(*dst_info, src_info->tensor_shape(), 1, src_info->data_type());

    const unsigned int vector_size_byte_opencl           = 16;
    const unsigned int num_elems_processed_per_iteration = adjust_vec_size(vector_size_byte_opencl / dst_info->element_size(), dst_info->dimension(0));
    Window             win                               = calculate_max_window(*dst_info, Steps(num_elems_processed_per_iteration));

    return win;
}

std::string ClFloorKernelComponent::get_component_code() const
{
    return R"_(
    //------------------ START KERNEL {{meta_kernel_id}} FLOOR ---------------------
    // IN_0(src)            {{src}}
    // OUT(dst, accum)      {{dst}}
    TILE({{DATA_TYPE}}, M0, N0, {{dst}});
    {
        TILE({{DATA_TYPE}}, M0, N0, src_tile);

        T_LOAD({{DATA_TYPE}}, M0, N0, BUFFER, {{src}}, cout, mout, 1, {{src}}_stride_y, src_tile);
        T_FLOOR({{DATA_TYPE}}, M0, N0, src_tile, {{dst}});
    }

    //------------------ END KERNEL {{meta_kernel_id}} FLOOR ---------------------
)_";
}

CLBuildOptions ClFloorKernelComponent::generate_build_options() const
{
    CLBuildOptions build_opts{};

    const auto n0 = _blueprint->impl().get_execution_window().x().step();
    const auto m0 = _blueprint->impl().get_execution_window().y().step();

    const auto         dst_info         = _blueprint->impl().get_kernel_argument_info(_blueprint->impl().get_dst_id());
    const unsigned int partial_store_n0 = dst_info->dimension(0) % n0;
    build_opts.add_option("-DM0=" + support::cpp11::to_string(m0));
    build_opts.add_option("-DN0=" + support::cpp11::to_string(n0));
    build_opts.add_option("-DPARTIAL_N0=" + support::cpp11::to_string(partial_store_n0));

    return build_opts;
}

std::string ClFloorKernelComponent::generate_config_id() const
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

void ClFloorKernelComponent::allocate_shared_vars(SharedVarTable &vtable) const
{
    vtable.add(_src, _blueprint->impl().group(_src.arg_id), ClKernelArgDescriptor(_src.arg_id, ClKernelTensorArgType::Tensor_4D_t_Buffer), "src");
    vtable.add(_dst, _blueprint->impl().group(_dst.arg_id), ClKernelArgDescriptor(_dst.arg_id, ClKernelTensorArgType::Tensor_4D_t_Buffer), "dst");
}

ClFloorKernelComponent::TagLUT ClFloorKernelComponent::get_tag_lut(const SharedVarTable &vtable) const
{
    TagLUT     lut{};
    const auto t_dst_info = _blueprint->impl().get_kernel_argument_info(_blueprint->impl().get_dst_id());
    // Arguments and global shared variables
    lut["src"]            = vtable.get(_src);
    lut["dst"]            = vtable.get(_dst);
    lut["meta_kernel_id"] = id();
    lut["DATA_TYPE"]      = get_cl_type_from_data_type(t_dst_info->data_type());
    return lut;
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */