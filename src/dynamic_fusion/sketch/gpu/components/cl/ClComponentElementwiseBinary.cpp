/*
 * Copyright (c) 2022-2023 Arm Limited.
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

#include "src/core/CL/CLValidate.h"
#ifndef ACL_INTERNAL_TEST_CKW_IN_DF
#include "src/dynamic_fusion/sketch/gpu/template_writer/cl/ClTemplateElementwiseBinary.h"
#else //ACL_INTERNAL_TEST_CKW_IN_DF
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/GpuCkwElementwiseBinary.h"
#endif //ACL_INTERNAL_TEST_CKW_IN_DF

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
namespace
{
std::set<ElementwiseBinaryCommonAttributes::ElementwiseOp> supported_ops{
    ElementwiseBinaryCommonAttributes::ElementwiseOp::Add, ElementwiseBinaryCommonAttributes::ElementwiseOp::Sub,
    ElementwiseBinaryCommonAttributes::ElementwiseOp::Mul};
}

Status ClComponentElementwiseBinary::validate(const ArgumentPack<ITensorInfo>         &tensors,
                                              const ElementwiseBinaryCommonAttributes &attributes)
{
    const auto lhs = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const auto rhs = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    const auto dst = tensors.get_const_tensor(TensorType::ACL_DST_0);

    // Check operator type
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(supported_ops.find(attributes.operation()) == supported_ops.end(),
                                    "Provided Elementwise operation not supported.");

    // Check validity
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(lhs, rhs, dst);

    //Check data type for different elementwise operators
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(lhs, 1, DataType::F32, DataType::F16, DataType::S32,
                                                         DataType::S16, DataType::U8);

    // dst shape is correct
    const TensorShape out_shape = TensorShape::broadcast_shape(lhs->tensor_shape(), rhs->tensor_shape());
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, dst->tensor_shape(), 0),
                                    "Wrong shape for dst.");

    const auto &lhs_shape = lhs->tensor_shape();
    const auto &rhs_shape = rhs->tensor_shape();
    const auto &dst_shape = dst->tensor_shape();

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(lhs_shape, dst_shape, 0) &&
                                        detail::have_different_dimensions(rhs_shape, dst_shape, 0),
                                    "Only LHS or RHS can be broadcasting, not both.");

    // Dimension Y and Z are collapsed together in the current kernel implementation,
    // hence they cannot be independently broadcast or non-broadcast.
    // See: ClTemplateElementwiseBinary::get_window
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((lhs_shape[1] != dst_shape[1] || rhs_shape[1] != dst_shape[1]) !=
                                        (lhs_shape[2] != dst_shape[2] || rhs_shape[2] != dst_shape[2]),
                                    "Dimension Y and Z must both be either broadcast or non-broadcast.");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(lhs_shape, dst_shape, 3),
                                    "LHS broadcast in dimension 3 or higher is not supported.");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(rhs_shape, dst_shape, 3),
                                    "RHS broadcast in dimension 3 or higher is not supported.");

    // Matching data type
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(lhs, rhs);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(lhs, dst);

    // Matching data layout
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(lhs, rhs);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(lhs, dst);

    // All tensor infos are initialized
    ARM_COMPUTE_RETURN_ERROR_ON(lhs->tensor_shape().total_size() == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(rhs->tensor_shape().total_size() == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(dst->tensor_shape().total_size() == 0);

    // Device requirements are met
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(lhs);

    return Status{};
}

ClComponentElementwiseBinary::~ClComponentElementwiseBinary()
{
}
ClComponentElementwiseBinary::ClComponentElementwiseBinary(ComponentId                      id,
                                                           const Properties                &properties,
                                                           const ArgumentPack<ITensorInfo> &tensors,
                                                           const Attributes                &attributes)
    : IGpuKernelComponent{id, properties, tensors},
#ifndef ACL_INTERNAL_TEST_CKW_IN_DF
      _component_writer{std::make_unique<ClTemplateElementwiseBinary>(id, tensors, attributes)}
#else  //ACL_INTERNAL_TEST_CKW_IN_DF
      _component_writer{std::make_unique<GpuCkwElementwiseBinary>(id, tensors, attributes)}
#endif //ACL_INTERNAL_TEST_CKW_IN_DF
{
}

#ifndef ACL_INTERNAL_TEST_CKW_IN_DF
const IGpuTemplateComponentWriter *ClComponentElementwiseBinary::template_writer() const
#else  //ACL_INTERNAL_TEST_CKW_IN_DF
const IGpuCkwComponentDriver *ClComponentElementwiseBinary::ckw_component_driver() const
#endif //ACL_INTERNAL_TEST_CKW_IN_DF
{
    return _component_writer.get();
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
