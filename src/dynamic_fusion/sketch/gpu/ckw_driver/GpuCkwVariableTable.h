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
#ifndef ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_GPUCKWVARIABLETABLE
#define ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_GPUCKWVARIABLETABLE

#include "acl/AclComponentArgument.h"
#include "arm_compute/core/ITensorInfo.h"

#include <map>

class AclScopedKernelWriter;

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
class GpuKernelComponentGroup;

/** A table of all the variables used in the kernel.
 *
 * It determines whether we create an virtual tensor var or a user tensor var
 * It avoids duplicating variables for the same tensors (Tensors with the same id)
 * Each kernel has exactly one variable table.
 */
class GpuCkwVariableTable
{
public:
    /** Declare a kernel component variable(argument) for the corresponding tensor info.
     *
     * @param[in] comp_group Component group the tensor belongs to
     * @param[in] writer     Compute Kernel Writer
     * @param[in] tensor     Tensor info with which the new variable is associated
     * @param[in] alias      Alias for the variable. Will be used as part of the variable name
     *
     * @return AclComponentArgument*
     */
    AclComponentArgument *declare_variable(const GpuKernelComponentGroup &comp_group, AclScopedKernelWriter &writer, const ITensorInfo *tensor, const std::string &alias = "unnamed");

private:
    std::map<ITensorInfo::Id, AclComponentArgument> _vars{};
};

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_GPUCKWVARIABLETABLE */
