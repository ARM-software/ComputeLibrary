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
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwVariableTable.h"

#include "src/dynamic_fusion/sketch/gpu/GpuKernelArgument.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGroup.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwKernelWriter.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwScopedKernelWriter.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/type_converter/Common.h"
#include <sstream>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
GpuCkwComponentArgument *GpuCkwVariableTable::declare_variable(const GpuKernelComponentGroup &comp_group, GpuCkwScopedKernelWriter &writer, const ITensorInfo *tensor, TensorStorageType storage,
                                                               const std::string &alias)
{
    ARM_COMPUTE_ERROR_ON_MSG(!tensor->has_valid_id(), "Tensor info with valid id expected");

    // Do not re-declare if the variable associated with the tensor has already been declared
    auto it = _vars.find(tensor->id());

    if(it != _vars.end())
    {
        return &it->second;
    }
    if(comp_group.is_intermediate_tensor(tensor))
    {
        // Create a virtual tensor variable
        GpuCkwComponentArgument var;
        auto                  &&inserted = _vars.emplace(tensor->id(), var);
        return &(inserted.first->second);
    }
    else
    {
        // Create a user tensor variable
        std::stringstream ss;
        ss << alias << "_t" << abs(tensor->id());
        const auto              uniq_name = ss.str();
        GpuCkwComponentArgument var{ writer->declare_tensor_argument(uniq_name, to_ckw(*tensor), to_ckw(storage)) };
        auto                  &&inserted = _vars.emplace(tensor->id(), var);
        return &(inserted.first->second);
    }
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
