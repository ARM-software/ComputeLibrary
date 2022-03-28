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
#if defined(ENABLE_EXPERIMENTAL_DYNAMIC_FUSION)

#include "src/core/experimental/dynamic_fusion/ClKernelBuildingImpl/components/ClStoreKernelComponents.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
ComponentType ClStoreBlockBoundaryAwareKernelComponent::get_component_type() const
{
    return ComponentType::Store;
}

std::string ClStoreBlockBoundaryAwareKernelComponent::get_component_code() const
{
    return R"_(
    //------------------ START KERNEL {{meta_kernel_id}} STORE ---------------------

    __global uchar *dst_addr = {{dst}}_ptr + {{dst}}_offset_first_element_in_bytes + (g_x * (uint)N0 * sizeof(DATA_TYPE)) + (COMPUTE_M0_START_ROW(g_y, M0, PARTIAL_STORE_M0) * {{dst}}_stride_y);

#if defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += g_z * {{dst}}_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_OUTPUT_AS_3D)

    // Add offset for batched GEMM
    dst_addr += g_z * {{dst}}_stride_z;

#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, {{src}}, dst_addr, {{dst}}_stride_y, g_zout, PARTIAL_STORE_M0, PARTIAL_STORE_N0, g_cond_y, g_cond_x);

    //------------------ END KERNEL {{meta_kernel_id}} STORE ---------------------

)_";
}

CLBuildOptions ClStoreBlockBoundaryAwareKernelComponent::generate_build_options() const
{
    auto t_dst_info = _blueprint->impl().get_kernel_argument_info(_blueprint->impl().get_dst_id());
    auto tile_info  = _blueprint->impl().get_tile_info();

    CLBuildOptions build_opts{};

    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(t_dst_info->data_type()));
    build_opts.add_option("-DM0=" + support::cpp11::to_string(tile_info.tile_dims.y()));
    build_opts.add_option("-DN0=" + support::cpp11::to_string(tile_info.tile_dims.x()));
    build_opts.add_option("-DPARTIAL_STORE_M0=" + support::cpp11::to_string(tile_info.boundaries.y() % tile_info.tile_dims.y()));
    build_opts.add_option("-DPARTIAL_STORE_N0=" + support::cpp11::to_string(tile_info.boundaries.x() % tile_info.tile_dims.x()));

    return build_opts;
}

ClStoreBlockBoundaryAwareKernelComponent::TagLUT ClStoreBlockBoundaryAwareKernelComponent::allocate_vars(SharedVarTable &vtable) const
{
    return {
        { "meta_kernel_id", id() },
        { "src", vtable.add(_src, ClKernelArgRuntimeDescriptor(_src.arg_id, TensorArgType::Image_3D), "src") },
        { "dst", vtable.add(_dst, ClKernelArgRuntimeDescriptor(_dst.arg_id, TensorArgType::Image_3D), "dst") },
    };
}

ComponentType ClStoreIndirectWidthSelectKernelComponent::get_component_type() const
{
    return ComponentType::Store;
}

std::string ClStoreIndirectWidthSelectKernelComponent::get_component_code() const
{
    return R"_(
    //------------------ START KERNEL {{meta_kernel_id}} STORE ---------------------

    TILE(uint, M0, 1, dst_indirect_y);

    // Calculate the destination indirect Y
    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        dst_indirect_y[i].v = (uint)min(mout + i, (int)({{dst_w}} * {{dst_h}}) - 1);
        dst_indirect_y[i].v += bout * (int)({{dst_w}} * {{dst_h}});
    })

    T_STORE_INDIRECT_WIDTH_SELECT({{DST_DATA_TYPE}}, M0, N0, PARTIAL_N0, {{DST_TENSOR_TYPE}}, {{dst}}, cout, {{dst}}_stride_y, PARTIAL_N0 != 0 && g_cond_x, {{src}}, dst_indirect_y);

    //------------------ END KERNEL {{meta_kernel_id}} STORE ---------------------

)_";
}

CLBuildOptions ClStoreIndirectWidthSelectKernelComponent::generate_build_options() const
{
    CLBuildOptions build_opts{};

    return build_opts;
}

ClStoreIndirectWidthSelectKernelComponent::TagLUT ClStoreIndirectWidthSelectKernelComponent::allocate_vars(SharedVarTable &vtable) const
{
    TagLUT lut{};

    lut["meta_kernel_id"] = id();
    lut["src"]            = vtable.add(_src, ClKernelArgRuntimeDescriptor(_src.arg_id, TensorArgType::Image_3D), "src");
    lut["dst"]            = vtable.add(_dst, ClKernelArgRuntimeDescriptor(_dst.arg_id, TensorArgType::Tensor_4D_t_Buffer), "dst");

    // Local build options
    auto dst_info = _blueprint->impl().get_kernel_argument_info(_blueprint->impl().get_dst_id());

    lut["dst_w"] = dst_info->dimension(1);
    lut["dst_h"] = dst_info->dimension(2);

    lut["DST_TENSOR_TYPE"] = "BUFFER";
    lut["DST_DATA_TYPE"]   = dst_info->data_type();

    return lut;
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif // defined(ENABLE_EXPERIMENTAL_DYNAMIC_FUSION)