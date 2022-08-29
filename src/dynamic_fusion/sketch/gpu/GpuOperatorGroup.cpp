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
#include "src/dynamic_fusion/sketch/gpu/GpuOperatorGroup.h"

#include "arm_compute/core/Validate.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
namespace
{
std::vector<DependencyGraph::TensorId> get_tensor_ids(const std::vector<const ITensorInfo *> tensors)
{
    std::vector<DependencyGraph::TensorId> tensor_ids{};
    std::transform(
        std::begin(tensors), std::end(tensors),
        std::back_inserter(tensor_ids),
        [](const auto & t)
    {
        return t->id();
    });
    return tensor_ids;
}

} // namespace

Operator::Operator(OperatorId id, GpuOperatorType operator_type, const ArgumentPack<ITensorInfo> &tensors)
    : _id{ id }, _operator_type{ operator_type }, _tensors{ tensors }
{
}

OperatorId Operator::id() const
{
    return _id;
}

GpuOperatorType Operator::operator_type() const
{
    return _operator_type;
}

ArgumentPack<ITensorInfo> Operator::tensors() const
{
    return _tensors;
}

bool GpuOperatorGroup::try_add_operator(const Operator &op) const
{
    const auto src_tensor_ids = get_tensor_ids(op.tensors().get_const_src_tensors());
    const auto dst_tensor_ids = get_tensor_ids(op.tensors().get_const_dst_tensors());
    // Constraint 1
    if(!_graph.try_add_operator_as_linear(op.id(), src_tensor_ids, dst_tensor_ids))
    {
        return false;
    }
    // Constraint 2
    if(_operators.size() >= max_fused_operators)
    {
        return false;
    }
    // Constraint 3.1: Pattern: (Unfusable)
    if(_operators.size() > 0 && get_root_operator()->operator_type() == GpuOperatorType::Unfusable)
    {
        return false;
    }
    // Constraint 3.2
    if(_operators.size() > 0 && (op.operator_type() != GpuOperatorType::Simple))
    {
        return false;
    }
    // Constraint 4
    if(op.operator_type() != GpuOperatorType::Unfusable && op.tensors().get_const_dst_tensors().size() != 1U)
    {
        return false;
    }
    // Constraint 5
    if(_operators.size() > 0)
    {
        const auto root_dst_tensors = get_root_operator()->tensors().get_const_dst_tensors();
        ARM_COMPUTE_ERROR_ON(root_dst_tensors.empty());
        const auto first_dst_tensor = root_dst_tensors[0];
        const auto dst_tensors      = op.tensors().get_const_dst_tensors();
        for(const auto &t : root_dst_tensors)
        {
            if(detail::have_different_dimensions(t->tensor_shape(), first_dst_tensor->tensor_shape(), 0))
            {
                return false;
            }
        }
        for(const auto &t : dst_tensors)
        {
            if(detail::have_different_dimensions(t->tensor_shape(), first_dst_tensor->tensor_shape(), 0))
            {
                return false;
            }
        }
    }
    // Constraint 6
    if(_operators.size() > 0)
    {
        const auto root_dst_tensors = get_root_operator()->tensors().get_const_dst_tensors();
        ARM_COMPUTE_ERROR_ON(root_dst_tensors.empty());
        const auto first_dst_tensor_layout = root_dst_tensors[0]->data_layout();
        const auto dst_tensors             = op.tensors().get_const_dst_tensors();
        for(const auto &t : root_dst_tensors)
        {
            if(t->data_layout() != first_dst_tensor_layout)
            {
                return false;
            }
        }
        for(const auto &t : dst_tensors)
        {
            if(t->data_layout() != first_dst_tensor_layout)
            {
                return false;
            }
        }
    }
    return true;
}
void GpuOperatorGroup::add_operator(const Operator &op)
{
    ARM_COMPUTE_ERROR_ON(!try_add_operator(op));
    const auto src_tensor_ids = get_tensor_ids(op.tensors().get_const_src_tensors());
    const auto dst_tensor_ids = get_tensor_ids(op.tensors().get_const_dst_tensors());
    _graph.add_operator_as_linear(op.id(), src_tensor_ids, dst_tensor_ids);
    _operators[op.id()] = op;
}
Operator GpuOperatorGroup::new_operator(const GpuOperatorType &operator_type, const ArgumentPack<ITensorInfo> &tensors) const
{
    auto new_id = static_cast<OperatorId>(_operators.size());
    return Operator{ new_id, operator_type, tensors };
}
const Operator *GpuOperatorGroup::get_root_operator() const
{
    const auto roots = _graph.get_root_ops();
    ARM_COMPUTE_ERROR_ON(roots.size() > 1);
    if(roots.empty())
    {
        return nullptr;
    }
    return &_operators.at(roots[0]);
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
