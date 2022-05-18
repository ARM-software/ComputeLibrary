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
    // auto tile_info  = _blueprint->impl().get_tile_info();

    CLBuildOptions build_opts{};

    const auto n0         = _blueprint->impl().get_execution_window().x().step();
    const auto m0         = _blueprint->impl().get_execution_window().y().step();
    const auto partial_m0 = t_dst_info->dimension(0) % m0;
    const auto partial_n0 = t_dst_info->dimension(1) % n0;

    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(t_dst_info->data_type()));
    build_opts.add_option("-DM0=" + support::cpp11::to_string(m0));
    build_opts.add_option("-DN0=" + support::cpp11::to_string(n0));
    build_opts.add_option("-DPARTIAL_STORE_M0=" + support::cpp11::to_string(partial_m0));
    build_opts.add_option("-DPARTIAL_STORE_N0=" + support::cpp11::to_string(partial_n0));

    return build_opts;
}

void ClStoreBlockBoundaryAwareKernelComponent::allocate_shared_vars(SharedVarTable &vtable) const
{
    vtable.add(_src, _blueprint->impl().group(_src.arg_id), ClKernelArgDescriptor(_src.arg_id, ClKernelTensorArgType::Image_3D), "src");
    vtable.add(_dst, _blueprint->impl().group(_dst.arg_id), ClKernelArgDescriptor(_dst.arg_id, ClKernelTensorArgType::Image_3D), "dst");
}

ClStoreBlockBoundaryAwareKernelComponent::TagLUT ClStoreBlockBoundaryAwareKernelComponent::get_tag_lut(const SharedVarTable &vtable) const
{
    return {
        { "meta_kernel_id", id() },
        { "src", vtable.get(_src) },
        { "dst", vtable.get(_dst) },
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
    {
    #define _IDST_WIDTH {{dst}}_w
    #define _IDST_HEIGHT {{dst}}_h
        TILE(uint, M0, 1, dst_indirect_y);

        // Calculate the destination indirect Y
        LOOP_UNROLLING(int, i, 0, 1, M0,
        {
            dst_indirect_y[i].v = (uint)min(mout + i, (int)(_IDST_WIDTH * _IDST_HEIGHT) - 1);
            dst_indirect_y[i].v += bout * (int)(_IDST_WIDTH * _IDST_HEIGHT);
        })

        bool x_cond = PARTIAL_N0 != 0 && get_global_id(0) == 0;

        T_STORE_INDIRECT_WIDTH_SELECT({{DST_DATA_TYPE}}, M0, N0, PARTIAL_N0, {{DST_TENSOR_TYPE}}, {{dst}}, cout, {{dst}}_stride_y, x_cond, {{src}}, dst_indirect_y);

    #undef _IDST_WIDTH
    #undef _IDST_HEIGHT
        //------------------ END KERNEL {{meta_kernel_id}} STORE ---------------------
    }

)_";
}

CLBuildOptions ClStoreIndirectWidthSelectKernelComponent::generate_build_options() const
{
    CLBuildOptions build_opts{};

    return build_opts;
}

void ClStoreIndirectWidthSelectKernelComponent::allocate_shared_vars(SharedVarTable &vtable) const
{
    vtable.add(_src, _blueprint->impl().group(_src.arg_id), ClKernelArgDescriptor(_src.arg_id, ClKernelTensorArgType::Tensor_4D_t_Buffer), "src");
    vtable.add(_dst, _blueprint->impl().group(_dst.arg_id), ClKernelArgDescriptor(_dst.arg_id, ClKernelTensorArgType::Tensor_4D_t_Buffer), "dst");
}

ClStoreIndirectWidthSelectKernelComponent::TagLUT ClStoreIndirectWidthSelectKernelComponent::get_tag_lut(const SharedVarTable &vtable) const
{
    TagLUT lut{};

    // Arguments and global shared variables
    lut["src"] = vtable.get(_src);
    lut["dst"] = vtable.get(_dst);

    // Local build options
    lut["meta_kernel_id"]  = id();
    lut["DST_TENSOR_TYPE"] = "BUFFER";
    const auto dst_info    = _blueprint->impl().get_kernel_argument_info(_blueprint->impl().get_dst_id());
    lut["DST_DATA_TYPE"]   = dst_info->data_type();

    return lut;
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */