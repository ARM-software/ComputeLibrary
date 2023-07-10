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
#include "ClComponentActivation.h"

#include "src/core/CL/CLValidate.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/GpuCkwActivation.h"
#include "src/dynamic_fusion/sketch/gpu/template_writer/cl/ClTemplateActivation.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
Status ClComponentActivation::validate(const Properties                &properties,
                                       const ArgumentPack<ITensorInfo> &tensors,
                                       const Attributes                &attributes)
{
    ARM_COMPUTE_UNUSED(properties, attributes);

    const ITensorInfo *const src = tensors.get_const_tensor(TensorType::ACL_SRC);
    const ITensorInfo *const dst = tensors.get_const_tensor(TensorType::ACL_DST);

    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);

    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F16, DataType::F32);

    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(src, dst);

    // All tensor infos are initialized
    ARM_COMPUTE_RETURN_ERROR_ON(src->tensor_shape().total_size() == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(dst->tensor_shape().total_size() == 0);

    // Device requirements are met
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(dst);

    return Status{};
}

ClComponentActivation::ClComponentActivation(ComponentId                            id,
                                             const IGpuKernelComponent::Properties &properties,
                                             const ArgumentPack<ITensorInfo>       &tensors,
                                             const Attributes                      &attributes)
    : IGpuKernelComponent{ id, properties, tensors },
      _component_writer{ std::make_unique<ClTemplateActivation>(id, tensors, attributes) },
      _ckw_driver{ std::make_unique<GpuCkwActivation>(id, tensors, attributes) }
{
}

ClComponentActivation::~ClComponentActivation()
{
}

const IGpuTemplateComponentWriter *ClComponentActivation::template_writer() const
{
    return _component_writer.get();
}

const IGpuCkwComponentDriver *ClComponentActivation::ckw_component_driver() const
{
    return _ckw_driver.get();
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
