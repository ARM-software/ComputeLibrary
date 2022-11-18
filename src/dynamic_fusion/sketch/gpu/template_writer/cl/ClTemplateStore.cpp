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
#include "ClTemplateStore.h"

#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGroup.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
ClTemplateStore::ClTemplateStore(ComponentId id, const ArgumentPack<ITensorInfo> &tensors)
    : IGpuTemplateComponentWriter{ id, tensors }, _src{}, _dst{}
{
    _src = this->tensors().get_const_tensor(TensorType::ACL_SRC_0);
    _dst = this->tensors().get_const_tensor(TensorType::ACL_DST_0);
}

std::string ClTemplateStore::get_name() const
{
    return "store";
}

std::string ClTemplateStore::get_component_code(const ComponentGroup &comp_group) const
{
    ARM_COMPUTE_UNUSED(comp_group);
    return R"_(
//------------------ START KERNEL {{meta_kernel_id}} STORE ---------------------
{
// This also follows NHWC layout
// g_ind_0 maps to global_id(0) maps to Channel
// g_ind_1 maps to global_id(1) maps to Height and Weight (Collapsed Window)
// g_ind_2 maps to global_id(2) maps to N / Batch
#define _IDST_WIDTH {{dst}}_w
#define _IDST_HEIGHT {{dst}}_h
    TILE(uint, M0, 1, dst_indirect_y);

    // Calculate the destination indirect Y
    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        dst_indirect_y[i].v = (uint)min(g_ind_1 + i, (int)(_IDST_WIDTH * _IDST_HEIGHT) - 1);
        dst_indirect_y[i].v += g_ind_2 * (int)(_IDST_WIDTH * _IDST_HEIGHT);
    })

    bool x_cond = PARTIAL_N0 != 0 && get_global_id(0) == 0;

    T_STORE_INDIRECT_WIDTH_SELECT({{DST_DATA_TYPE}}, M0, N0, PARTIAL_N0, {{DST_TENSOR_TYPE}}, {{dst}}, g_ind_0, {{dst}}_stride_y, x_cond, {{src}}, dst_indirect_y);

#undef _IDST_WIDTH
#undef _IDST_HEIGHT
    //------------------ END KERNEL {{meta_kernel_id}} STORE ---------------------
}

)_";
}

void ClTemplateStore::declare_variables(GpuKernelVariableTable &vtable, const ComponentGroup &comp_group) const
{
    // ARM_COMPUTE_UNUSED(comp_group)
    vtable.declare_variable(
        _src,
        GpuKernelArgumentInfo(GpuKernelArgumentInfo::Type::Tensor_4D_t_Buffer),
        comp_group.is_intermediate_tensor(_src),
        "src");
    vtable.declare_variable(
        _dst,
        GpuKernelArgumentInfo(GpuKernelArgumentInfo::Type::Tensor_4D_t_Buffer),
        comp_group.is_intermediate_tensor(_dst),
        "dst");
}

TagLUT ClTemplateStore::get_tag_lut(const GpuKernelVariableTable &vtable, const ComponentGroup &comp_group) const
{
    TagLUT lut{};

    // Arguments and global shared variables
    lut["src"] = vtable.get_variable(_src);
    lut["dst"] = vtable.get_variable(_dst);

    // Local build options
    lut["meta_kernel_id"]  = id();
    lut["DST_TENSOR_TYPE"] = "BUFFER";
    const auto dst_info    = comp_group.get_dst_tensors()[0];
    lut["DST_DATA_TYPE"]   = dst_info->data_type();

    return lut;
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
