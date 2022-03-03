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

#include "src/core/experimental/dynamic_fusion/ClKernelBuildingImpl/components/ClElementwiseAddKernelComponent.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
ComponentType ClElementwiseAddKernelComponent::get_component_type() const
{
    return ComponentType::Simple;
}

std::set<std::string> ClElementwiseAddKernelComponent::get_headers_list() const
{
    return std::set<std::string> { "gemm_helpers.h", "repeat.h" };
}

std::string ClElementwiseAddKernelComponent::get_component_code() const
{
    std::string code;
    return R"_(
    //------------------ START KERNEL {{meta_kernel_id}} ELTWISE_ADD ---------------------
    // IN_0(Accumulator)   {{acc}}
    // IN_1(Addend)                {{addend}}

    // c = addend + c (mix-precision, broadcast, boundary aware)
    {
        __global uchar *addend_addr = {{addend}}_ptr + {{addend}}_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE)) + (COMPUTE_M0_START_ROW(g_y, M0, PARTIAL_STORE_M0) * {{addend}}_stride_y) + get_global_id(2) * {{addend}}_stride_z; \
        LOAD_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, addend, addend_addr, 0, {{addend}}_stride_y, g_zero, PARTIAL_LOAD_M0, PARTIAL_LOAD_N0, PARTIAL_COND_Y, PARTIAL_COND_X);                                                                                        \
        MIXED_PRECISION_ELTWISE_OP_BLOCK(ADD_X_POS_0, M0, N0, {{acc}}, addend, DATA_TYPE_ACCUMULATOR, addend_hp);
    }
    //------------------ END KERNEL {{meta_kernel_id}} ELTWISE_ADD ---------------------

)_";
}
ClElementwiseAddKernelComponent::TagLUT ClElementwiseAddKernelComponent::allocate_vars(SharedVarTable &vtable) const
{
    // Determine which argument is the accumulator
    Link accumulator;
    Link addend;
    if(_lhs.group == SharedVarGroup::Automatic)
    {
        accumulator = _lhs;
        addend      = _rhs;
    }
    else if(_rhs.group == SharedVarGroup::Automatic)
    {
        accumulator = _rhs;
        addend      = _lhs;
    }
    else
    {
        ARM_COMPUTE_ERROR("Invalid elementwise component linking");
    }
    return {
        { "meta_kernel_id", id() },
        { "acc", vtable.add(accumulator, ClKernelArgRuntimeDescriptor(accumulator.arg_id, TensorArgType::Image_3D), "add_acc") },
        { "addend", vtable.add(addend, ClKernelArgRuntimeDescriptor(addend.arg_id, TensorArgType::Image_3D), "add_addend") },
        // {"dst", vtable.add(_dst, ClKernelArgRuntimeDescriptor(_dst.arg_id, TensorArgType::Image_3D), "dst")}, // dst is needed for the root version and/or non-inplace version should we need one
    };
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif // defined(ENABLE_EXPERIMENTAL_DYNAMIC_FUSION)