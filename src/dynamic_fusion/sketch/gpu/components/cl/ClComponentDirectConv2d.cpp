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
#include "ClComponentDirectConv2d.h"

#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/dynamic_fusion/sketch/OperatorAttributes.h"
#include "src/core/CL/CLValidate.h"
#include "src/dynamic_fusion/sketch/gpu/template_writer/cl/ClTemplateDirectConv2d.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
ClComponentDirectConv2dSettings &ClComponentDirectConv2dSettings::export_to_cl_image(bool cl_image)
{
    _export_to_cl_image = cl_image;
    return *this;
}

bool ClComponentDirectConv2dSettings::export_to_cl_image() const
{
    return _export_to_cl_image;
}

ClComponentDirectConv2dSettings &ClComponentDirectConv2dSettings::fast_relaxed_math(bool fast_relaxed_math)
{
    _fast_relaxed_math = fast_relaxed_math;
    return *this;
}

bool ClComponentDirectConv2dSettings::fast_relaxed_math() const
{
    return _fast_relaxed_math;
}

ClComponentDirectConv2dSettings &ClComponentDirectConv2dSettings::direct_conv_descriptor(const DirectConvComputeKernelInfo &desc)
{
    _desc = desc;
    return *this;
}

DirectConvComputeKernelInfo ClComponentDirectConv2dSettings::direct_conv_descriptor() const
{
    return _desc;
}

Status ClComponentDirectConv2d::validate(
    const Properties                &properties,
    const ArgumentPack<ITensorInfo> &tensors,
    const Attributes                &attributes,
    const Settings                  &settings)
{
    ARM_COMPUTE_UNUSED(properties);
    const auto src = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const auto wei = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    const auto bia = tensors.get_const_tensor(TensorType::ACL_SRC_2);
    const auto dst = tensors.get_const_tensor(TensorType::ACL_DST_0);

    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, wei, dst);

    // 1. Check validity
    // Matching data type
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, wei);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
    if(bia != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, bia);
    }

    // Matching data layout
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(src, wei);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(src, dst);
    if(bia != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(src, bia);
    }

    // All tensor infos are initialized
    ARM_COMPUTE_RETURN_ERROR_ON(src->tensor_shape().total_size() == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(wei->tensor_shape().total_size() == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(dst->tensor_shape().total_size() == 0);
    if(bia != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(bia->tensor_shape().total_size() == 0);
    }
    // Device requirements are met
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src);
    // wei shape is correct
    const DataLayout data_layout = src->data_layout();
    const int        channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(wei->dimension(channel_idx) != src->dimension(channel_idx), "Weights feature map dimension should match the respective src's one");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(wei->num_dimensions() > 4, "Weights can be at most 4 dimensional");

    // dst shape is correct
    PadStrideInfo legacy_pad_stride(attributes.stride().x(), attributes.stride().y(), attributes.pad().left, attributes.pad().right, attributes.pad().top,
                                    attributes.pad().bottom, DimensionRoundingType{});
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(),
                                                       misc::shape_calculator::compute_deep_convolution_shape(*src, *wei, legacy_pad_stride));

    // bia shape is correct
    if(bia != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(bia->dimension(0) != wei->dimension(3),
                                        "Biases size and number of dst feature maps should match");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(bia->num_dimensions() > 1,
                                        "Biases should be one dimensional");
    }

    // 2. Check support level
    // Data type
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F16, DataType::F32);
    // Data layout
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(src, DataLayout::NHWC);

    const auto desc = settings.direct_conv_descriptor();
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(desc.n0 != 1 && desc.n0 != 2 && desc.n0 != 3 && desc.n0 != 4 && desc.n0 != 8 && desc.n0 != 16,
                                    "N0 can only be: 1, 2, 3, 4, 8, and 16");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(desc.k0 != 1 && desc.k0 != 2 && desc.k0 != 3 && desc.k0 != 4 && desc.k0 != 8 && desc.k0 != 16,
                                    "K0 can only be: 1, 2, 3, 4, 8, and 16");
    return Status{};
}

ClComponentDirectConv2d::ClComponentDirectConv2d(
    ComponentId                      id,
    const Properties                &properties,
    const ArgumentPack<ITensorInfo> &tensors,
    const Attributes                &attributes,
    const Settings                  &settings)
    : IGpuKernelComponent{ id, properties, tensors },
      _component_writer{ std::make_unique<ClTemplateDirectConv2d>(id, tensors, attributes, settings) }
{
}
ClComponentDirectConv2d::~ClComponentDirectConv2d()
{
}
const IGpuTemplateComponentWriter *ClComponentDirectConv2d::template_writer() const
{
    return _component_writer.get();
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
