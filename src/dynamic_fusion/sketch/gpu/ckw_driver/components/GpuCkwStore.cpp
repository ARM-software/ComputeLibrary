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
#include "GpuCkwStore.h"

#include "arm_compute/core/Error.h"
#include "compute_kernel_writer/include/acl/AclKernelWriter.h"
#include "compute_kernel_writer/include/acl/AclScopedKernelWriter.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwVariableTable.h"
#include <string>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
GpuCkwStore::GpuCkwStore(ComponentId id, const ArgumentPack<ITensorInfo> &tensors)
    : IGpuCkwComponentDriver{ id, tensors }, _src{}, _dst{}
{
    _src = this->tensors().get_const_tensor(TensorType::ACL_SRC_0);
    _dst = this->tensors().get_const_tensor(TensorType::ACL_DST_0);
}
void GpuCkwStore::write_component_code(const ComponentGroup &comp_group, GpuCkwVariableTable &vtable, AclScopedKernelWriter writer) const
{
    auto src = vtable.declare_variable(comp_group, writer, _src, "src");
    auto dst = vtable.declare_variable(comp_group, writer, _dst, "dst");

    auto       &src_tile   = src->tile();
    const auto &sampler    = src->tile_sampler();
    auto       &dst_tensor = dst->tensor();

    writer->op_store(dst_tensor, src_tile, sampler);
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
