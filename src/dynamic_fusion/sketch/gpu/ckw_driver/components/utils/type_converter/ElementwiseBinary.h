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
#ifndef ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_COMPONENTS_UTILS_TYPE_CONVERTER_ELEMENTWISEBINARY
#define ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_COMPONENTS_UTILS_TYPE_CONVERTER_ELEMENTWISEBINARY

#include "ckw/types/Operators.h"

#include "src/dynamic_fusion/sketch/gpu/operators/internal/GpuElementwiseBinaryCommon.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
inline ckw::BinaryOp to_ckw(const ElementwiseBinaryCommonAttributes &attributes)
{
    switch (attributes.operation())
    {
        case ElementwiseBinaryCommonAttributes::ElementwiseOp::Add:
            return ckw::BinaryOp::Add;
        case ElementwiseBinaryCommonAttributes::ElementwiseOp::Sub:
            return ckw::BinaryOp::Sub;
        case ElementwiseBinaryCommonAttributes::ElementwiseOp::Div:
            return ckw::BinaryOp::Div;
        case ElementwiseBinaryCommonAttributes::ElementwiseOp::Mul:
            return ckw::BinaryOp::Mul;
        case ElementwiseBinaryCommonAttributes::ElementwiseOp::Min:
        case ElementwiseBinaryCommonAttributes::ElementwiseOp::Max:
        case ElementwiseBinaryCommonAttributes::ElementwiseOp::Power:
        case ElementwiseBinaryCommonAttributes::ElementwiseOp::Prelu:
        case ElementwiseBinaryCommonAttributes::ElementwiseOp::SquaredDiff:
        default:
            ARM_COMPUTE_ERROR("Cannot convert ElementwiseBinaryCommonAttributes to corresponding ckw::BinaryOp");
    }
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif /* ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_COMPONENTS_UTILS_TYPE_CONVERTER_ELEMENTWISEBINARY */
