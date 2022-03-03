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
ClStoreBlockBoundaryAwareKernelComponent::TagLUT ClStoreBlockBoundaryAwareKernelComponent::allocate_vars(SharedVarTable &vtable) const
{
    return {
        { "meta_kernel_id", id() },
        { "src", vtable.add(_src, ClKernelArgRuntimeDescriptor(_src.arg_id, TensorArgType::Image_3D), "src") },
        { "dst", vtable.add(_dst, ClKernelArgRuntimeDescriptor(_dst.arg_id, TensorArgType::Image_3D), "dst") },
    };
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif // defined(ENABLE_EXPERIMENTAL_DYNAMIC_FUSION)