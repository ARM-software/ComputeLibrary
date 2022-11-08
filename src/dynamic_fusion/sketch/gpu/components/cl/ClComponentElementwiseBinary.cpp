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
#include "ClComponentElementwiseBinary.h"

#include "arm_compute/core/Validate.h"
#include "arm_compute/dynamic_fusion/sketch/OperatorAttributes.h"
#include "src/core/CL/CLValidate.h"
#include "src/dynamic_fusion/sketch/gpu/template_writer/cl/ClTemplateElementwiseBinary.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
namespace
{
std::set<ElementwiseBinaryCommonAttributes::ElementwiseOp> supported_ops
{
    ElementwiseBinaryCommonAttributes::ElementwiseOp::ADD
};
}

Status ClComponentElementwiseBinary::validate(const ArgumentPack<ITensorInfo> &tensors, const ElementwiseBinaryCommonAttributes &attributes)
{
    const auto lhs = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const auto rhs = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    const auto dst = tensors.get_const_tensor(TensorType::ACL_DST_0);

    // Check operator type
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(supported_ops.find(attributes.operation()) == supported_ops.end(), "Provided Elementwise operation not supported.");

    // Check validity
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(lhs, rhs, dst);

    //Check data type for different elementwise operators
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(lhs, 1, DataType::F32, DataType::F16, DataType::S32, DataType::S16, DataType::U8);

    const bool rhs_in_place = (rhs == dst);
    const bool lhs_in_place = (lhs == dst);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(rhs_in_place && lhs_in_place, "Both LHS and RHS cannot be in-place at same time for any elementwise operation.");

    // dst shape is correct
    const TensorShape out_shape = TensorShape::broadcast_shape(lhs->tensor_shape(), rhs->tensor_shape());
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, dst->tensor_shape(), 0), "Wrong shape for dst.");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((!rhs_in_place && !lhs_in_place) && detail::have_different_dimensions(lhs->tensor_shape(), dst->tensor_shape(), 0),
                                    "Only the rhs operand can be broadcast to match the accumulator's (lhs) shape");
    // Matching data type
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(lhs, rhs);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(lhs, dst);

    // Matching data layout
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(lhs, rhs);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(lhs, dst);

    // Batching case not supported yet
    const size_t idx_batch = get_data_layout_dimension_index(lhs->data_layout(), DataLayoutDimension::BATCHES);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((lhs->tensor_shape()[idx_batch] != 1) || (rhs->tensor_shape()[idx_batch] != 1) || (dst->tensor_shape()[idx_batch] != 1), "Batching case not supported yet");

    // All tensor infos are initialized
    ARM_COMPUTE_RETURN_ERROR_ON(lhs->tensor_shape().total_size() == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(rhs->tensor_shape().total_size() == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(dst->tensor_shape().total_size() == 0);

    // Device requirements are met
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(lhs);

    return Status{};
}

ClComponentElementwiseBinary::ClComponentElementwiseBinary(
    ComponentId                      id,
    const Properties                &properties,
    const ArgumentPack<ITensorInfo> &tensors,
    const Attributes                &attributes)
    : IGpuKernelComponent{ id, properties, tensors },
      _component_writer{ std::make_unique<ClTemplateElementwiseBinary>(id, tensors, attributes) }
{
}
ClComponentElementwiseBinary::~ClComponentElementwiseBinary()
{
}
const IGpuTemplateComponentWriter *ClComponentElementwiseBinary::template_writer() const
{
    return _component_writer.get();
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
