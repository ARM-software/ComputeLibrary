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
#include "ClComponentReshape.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Validate.h"

#include "src/core/CL/CLValidate.h"
#include "src/dynamic_fusion/sketch/gpu/template_writer/cl/ClTemplateReshape.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
Status ClComponentReshape::validate(const ArgumentPack<ITensorInfo> &tensors)
{
    const auto src = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const auto dst = tensors.get_const_tensor(TensorType::ACL_DST_0);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON(src->tensor_shape().total_size() == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(dst->tensor_shape().total_size() == 0);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON(src->tensor_shape().total_size() != dst->tensor_shape().total_size());

    return Status{};
}

ClComponentReshape::ClComponentReshape(ComponentId                      id,
                                       const Properties                &properties,
                                       const ArgumentPack<ITensorInfo> &tensors)
    : IGpuKernelComponent{id, properties, tensors}, _component_writer{std::make_unique<ClTemplateReshape>(id, tensors)}
{
}
ClComponentReshape::~ClComponentReshape()
{
}
const IGpuTemplateComponentWriter *ClComponentReshape::template_writer() const
{
    return _component_writer.get();
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
