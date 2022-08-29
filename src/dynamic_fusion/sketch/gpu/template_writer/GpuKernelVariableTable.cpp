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
#include "GpuKernelVariableTable.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/ITensorInfo.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
void GpuKernelVariableTable::declare_variable(const ITensorInfo *tensor, GpuKernelArgumentInfo argument_info, bool is_interm, const std::string &alias)
{
    ARM_COMPUTE_ERROR_ON_MSG(!tensor->has_valid_id(), "Tensor info with valid id expected");
    // Do not re-declare if the variable associated with the tensor has already been declared
    if(get_variable(tensor).has_valid_id())
    {
        ARM_COMPUTE_ERROR_ON(!(get_variable(tensor).kernel_argument_info == argument_info));
        return;
    }
    // Declare variable associated with the tensor
    std::stringstream ss;
    ss << alias << "_t" << tensor->id();
    const auto     uniq_name = ss.str();
    TensorVariable var{ tensor->id(), uniq_name, argument_info };

    if(is_interm)
    {
        _interm_var = var;
        _interm_tensors.insert(tensor->id());
    }
    else
    {
        _vars.emplace(tensor->id(), var);
    }
}

GpuKernelVariableTable::TensorVariable GpuKernelVariableTable::get_variable(const ITensorInfo *tensor) const
{
    const TensorVariable empty_var{};
    if(_vars.find(tensor->id()) != _vars.end())
    {
        return _vars.at(tensor->id());
    }
    if(_interm_tensors.find(tensor->id()) != _interm_tensors.end())
    {
        return _interm_var;
    }
    return empty_var;
}

GpuKernelVariableTable::VariableList GpuKernelVariableTable::get_variable_list(const std::vector<const ITensorInfo *> &tensors) const
{
    VariableList vars{};
    for(const auto &tensor : tensors)
    {
        if(!tensor->has_valid_id())
        {
            continue;
        }
        vars.push_back(get_variable(tensor));
    }
    return vars;
}

TagVal::TagVal(const GpuKernelVariableTable::TensorVariable &var)
    : value{ var.uniq_name }
{
}

TagVal::TagVal(const std::string &val)
    : value{ val }
{
}

TagVal::TagVal(const char *val)
    : value{ std::string(val) }
{
}

TagVal::TagVal(const DataType &data_type)
    : value{ get_cl_type_from_data_type(data_type) }
{
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
