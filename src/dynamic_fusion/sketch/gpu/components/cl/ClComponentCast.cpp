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
#include "ClComponentCast.h"

#include "arm_compute/core/Error.h"
#include "src/core/CL/CLValidate.h"
#include "src/dynamic_fusion/sketch/ArgumentPack.h"
#include "src/dynamic_fusion/sketch/gpu/template_writer/cl/ClTemplateCast.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
Status ClComponentCast::validate(
    const Properties                &properties,
    const ArgumentPack<ITensorInfo> &tensors,
    const Attributes                &attributes,
    const Settings                  &settings)
{
    ARM_COMPUTE_UNUSED(properties, attributes, settings);

    const ITensorInfo *src = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const ITensorInfo *dst = tensors.get_const_tensor(TensorType::ACL_DST_0);

    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(dst);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON(src == dst);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(src->data_type() == attributes.data_type(), "input and target data types should be different");

    // Validate in case of configured dst
    if(dst->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(dst->data_type() != attributes.data_type(), "dst and target data types should be same");
    }

    return Status{};
}
ClComponentCast::ClComponentCast(ComponentId                      id,
                                 const Properties                &properties,
                                 const ArgumentPack<ITensorInfo> &tensors,
                                 const Attributes                &attributes,
                                 const Settings                  &settings)
    : IGpuKernelComponent{ id, properties, tensors },
      _component_writer{ std::make_unique<ClTemplateCast>(id, tensors, attributes) }
{
    ARM_COMPUTE_UNUSED(attributes, settings);
}
ClComponentCast::~ClComponentCast()
{
}
const IGpuTemplateComponentWriter *ClComponentCast::template_writer() const
{
    return _component_writer.get();
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
