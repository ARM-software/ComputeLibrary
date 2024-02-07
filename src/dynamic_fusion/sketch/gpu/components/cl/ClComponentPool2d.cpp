/*
 * Copyright (c) 2023-2024 Arm Limited.
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
#include "ClComponentPool2d.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/Pool2dAttributes.h"

#include "src/core/CL/CLValidate.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/GpuCkwPool2d.h"
#include "src/dynamic_fusion/utils/Utils.h"

#include <memory>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
Status ClComponentPool2d::validate(const Properties                &properties,
                                   const ArgumentPack<ITensorInfo> &tensors,
                                   const Attributes                &attributes,
                                   const Settings                  &settings)
{
    ARM_COMPUTE_UNUSED(properties, settings);
    const auto src = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const auto dst = tensors.get_const_tensor(TensorType::ACL_DST_0);

    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_ON_MSG((attributes.pool_type() != PoolingType::AVG && attributes.pool_type() != PoolingType::MAX),
                             "Unsupported Pooling type");

    // 1. Check validity
    // Check if pooling is valid
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(
        is_pool_region_entirely_outside_input(convert_pool_attr_to_pool_info(attributes, true)),
        "Pooling region that is entirely outside input tensor is unsupported");

    // Matching data type
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);

    // Matching data layout
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(src, dst);

    // All tensor infos are initialized
    ARM_COMPUTE_RETURN_ERROR_ON(src->tensor_shape().total_size() == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(dst->tensor_shape().total_size() == 0);

    // Device requirements are met
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src);

    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(
        dst->tensor_shape(),
        misc::shape_calculator::compute_pool_shape(*src, convert_pool_attr_to_pool_info(attributes, true)));

    // 2. Check support level
    // Data type
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F16, DataType::F32);

    // Data layout
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(src, DataLayout::NHWC);

    return Status{};
}

ClComponentPool2d::ClComponentPool2d(ComponentId                      id,
                                     const Properties                &properties,
                                     const ArgumentPack<ITensorInfo> &tensors,
                                     const Attributes                &attributes,
                                     const Settings                  &settings)
    : IGpuKernelComponent{id, properties, tensors},
      _component_writer{std::make_unique<GpuCkwPool2d>(id, tensors, attributes, settings)}
{
}
ClComponentPool2d::~ClComponentPool2d()
{
}
const IGpuCkwComponentDriver *ClComponentPool2d::ckw_component_driver() const
{
    return _component_writer.get();
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
