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

#include "src/dynamic_fusion/sketch/gpu/components/cl/ClComponentLogits1DMaxShiftExpSum.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/SoftmaxAttributes.h"
#include "src/core/CL/CLValidate.h"
#include "src/dynamic_fusion/sketch/gpu/template_writer/cl/ClTemplateLogits1DMaxShiftExpSum.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
Status ClComponentLogits1DMaxShiftExpSum::validate(
    const Properties                &properties,
    const ArgumentPack<ITensorInfo> &tensors,
    const Attributes                &attributes)
{
    ARM_COMPUTE_UNUSED(properties, attributes);

    const ITensorInfo *src = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const ITensorInfo *sum = tensors.get_const_tensor(TensorType::ACL_DST_0);
    const ITensorInfo *dst = tensors.get_const_tensor(TensorType::ACL_DST_1);

    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(sum);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(dst);

    // 1. Check validity
    // All tensor infos are initialized
    ARM_COMPUTE_RETURN_ERROR_ON(src->tensor_shape().total_size() == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(sum->tensor_shape().total_size() == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(dst->tensor_shape().total_size() == 0);

    // Check for mismatches in shapes and data types
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst, sum);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, dst);

    // Device requirements are met
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src);

    // 2. Check support level
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F16, DataType::F32);

    return Status{};
}

ClComponentLogits1DMaxShiftExpSum::ClComponentLogits1DMaxShiftExpSum(ComponentId                      id,
                                                                     const Properties                &properties,
                                                                     const ArgumentPack<ITensorInfo> &tensors,
                                                                     const Attributes                &attributes)
    : IGpuKernelComponent{ id, properties, tensors },
      _component_writer{ std::make_unique<ClTemplateLogits1DMaxShiftExpSum>(id, tensors, attributes) }
{
}

ClComponentLogits1DMaxShiftExpSum::~ClComponentLogits1DMaxShiftExpSum()
{
}

const IGpuTemplateComponentWriter *ClComponentLogits1DMaxShiftExpSum::template_writer() const
{
    return _component_writer.get();
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
