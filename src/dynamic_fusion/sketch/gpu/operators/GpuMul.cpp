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
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuMul.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h"

#include "src/dynamic_fusion/sketch/gpu/operators/internal/GpuElementwiseBinaryCommon.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
Status GpuMul::validate_op(const GpuWorkloadSketch &sketch,
                           const ITensorInfo       *lhs,
                           const ITensorInfo       *rhs)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(lhs, rhs);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(lhs, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(lhs->data_type() != rhs->data_type(), "Input tensors must be the same data type");

    // Set the elementwise operation to Mul then call the elementwise common validate_op
    ElementwiseBinaryCommonAttributes common_attributes{};
    common_attributes.operation(ElementwiseBinaryCommonAttributes::ElementwiseOp::Mul);
    return GpuElementwiseBinaryCommon::validate_op(sketch, lhs, rhs, common_attributes);
}

Status GpuMul::is_supported_op(const GpuWorkloadContext &context,
                               const ITensorInfo        *lhs,
                               const ITensorInfo        *rhs)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(lhs, rhs);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(lhs, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(lhs->data_type() != rhs->data_type(), "Input tensors must be the same data type");

    // Set the elementwise operation to Mul then call the elementwise common is_supported_op
    ElementwiseBinaryCommonAttributes common_attributes{};
    common_attributes.operation(ElementwiseBinaryCommonAttributes::ElementwiseOp::Mul);
    return GpuElementwiseBinaryCommon::is_supported_op(context, lhs, rhs, common_attributes);
}

ITensorInfo *GpuMul::create_op(GpuWorkloadSketch &sketch,
                               ITensorInfo       *lhs,
                               ITensorInfo       *rhs)
{
    // Set the elementwise operation to Mul then call the elementwise common create_op
    ElementwiseBinaryCommonAttributes common_attributes{};
    common_attributes.operation(ElementwiseBinaryCommonAttributes::ElementwiseOp::Mul);
    return GpuElementwiseBinaryCommon::create_op(sketch, lhs, rhs, common_attributes);
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
