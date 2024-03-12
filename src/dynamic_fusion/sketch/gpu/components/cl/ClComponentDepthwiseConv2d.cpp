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
#include "ClComponentDepthwiseConv2d.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/DepthwiseConv2dAttributes.h"

#include "src/core/CL/CLValidate.h"
#ifndef ACL_INTERNAL_TEST_CKW_IN_DF
#include "src/dynamic_fusion/sketch/gpu/template_writer/cl/ClTemplateDepthwiseConv2d.h"
#else //ACL_INTERNAL_TEST_CKW_IN_DF
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/GpuCkwDepthwiseConv2d.h"
#endif //ACL_INTERNAL_TEST_CKW_IN_DF

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
using Settings = ClComponentDepthwiseConv2dSettings;

Settings &Settings::export_input_to_cl_image(bool cl_image)
{
    _export_input_to_cl_image = cl_image;
    return *this;
}

bool Settings::export_input_to_cl_image() const
{
    return _export_input_to_cl_image;
}

Settings &Settings::export_weights_to_cl_image(bool cl_image)
{
    _export_weights_to_cl_image = cl_image;
    return *this;
}

bool Settings::export_weights_to_cl_image() const
{
    return _export_weights_to_cl_image;
}

Settings &Settings::fast_relaxed_math(bool fast_relaxed_math)
{
    _fast_relaxed_math = fast_relaxed_math;
    return *this;
}

bool Settings::fast_relaxed_math() const
{
    return _fast_relaxed_math;
}

Settings &Settings::is_fma_available(bool is_fma_available)
{
    _is_fma_available = is_fma_available;
    return *this;
}

bool Settings::is_fma_available() const
{
    return _is_fma_available;
}

Settings &Settings::n0(unsigned int n0)
{
    _n0 = n0;
    return *this;
}

unsigned int Settings::n0() const
{
    return _n0;
}

Settings &Settings::m0(unsigned int m0)
{
    _m0 = m0;
    return *this;
}

unsigned int Settings::m0() const
{
    return _m0;
}

Status ClComponentDepthwiseConv2d::validate(const Properties                &properties,
                                            const ArgumentPack<ITensorInfo> &tensors,
                                            const Attributes                &attributes,
                                            const Settings                  &settings)
{
    ARM_COMPUTE_UNUSED(properties, settings);
    const auto src = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const auto wei = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    const auto bia = tensors.get_const_tensor(TensorType::ACL_SRC_2);
    const auto dst = tensors.get_const_tensor(TensorType::ACL_DST_0);

    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, wei, dst);

    // 1. Check validity
    // Matching data type
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, wei);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
    if (bia != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, bia);
    }

    // Matching data layout
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(src, wei);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(src, dst);
    if (bia != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(src, bia);
    }

    // All tensor infos are initialized
    ARM_COMPUTE_RETURN_ERROR_ON(src->tensor_shape().total_size() == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(wei->tensor_shape().total_size() == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(dst->tensor_shape().total_size() == 0);
    if (bia != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(bia->tensor_shape().total_size() == 0);
    }
    // Device requirements are met
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src);
    // wei shape is correct
    const DataLayout data_layout = src->data_layout();
    const size_t     channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    ARM_COMPUTE_RETURN_ERROR_ON(wei->dimension(channel_idx) !=
                                (src->dimension(channel_idx) * attributes.depth_multiplier()));
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(wei->num_dimensions() > 3, "Weights can be at most 3 dimensional");

    // dst shape is correct
    const PadStrideInfo pad_stride_info =
        PadStrideInfo(attributes.stride().x(), attributes.stride().y(), attributes.pad().left, attributes.pad().right,
                      attributes.pad().top, attributes.pad().bottom, attributes.dimension_rounding_type());
    const ConvolutionInfo conv_info{pad_stride_info, attributes.depth_multiplier(), ActivationLayerInfo(),
                                    attributes.dilation()};
    const TensorShape output_shape = misc::shape_calculator::compute_depthwise_convolution_shape(*src, *wei, conv_info);

    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(), output_shape);

    // Check strides and dilation
    ARM_COMPUTE_RETURN_ERROR_ON(conv_info.pad_stride_info.stride().first < 1);
    ARM_COMPUTE_RETURN_ERROR_ON(conv_info.pad_stride_info.stride().second < 1);
    ARM_COMPUTE_RETURN_ERROR_ON((conv_info.dilation.x() < 1) || (conv_info.dilation.y() < 1));
    ARM_COMPUTE_RETURN_ERROR_ON(conv_info.pad_stride_info.stride().first > 1 && settings.m0() != 1);
    ARM_COMPUTE_RETURN_ERROR_ON(conv_info.dilation.x() > 1 && settings.m0() != 1);

    if (conv_info.depth_multiplier > 1 && settings.n0() > 1)
    {
        ARM_COMPUTE_RETURN_ERROR_ON((conv_info.depth_multiplier % settings.n0()) != 0);
    }

    // Check export weights to cl image
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((settings.export_weights_to_cl_image() == true) &&
                                        (export_to_cl_image(wei) == false),
                                    "Weights cannot be exported to cl_image!");
    ARM_COMPUTE_RETURN_ERROR_ON((settings.export_weights_to_cl_image() == true) && ((settings.n0() % 4) != 0));

    ARM_COMPUTE_RETURN_ERROR_ON(wei->dimension(channel_idx) !=
                                (src->dimension(channel_idx) * conv_info.depth_multiplier));

    // bia shape is correct
    if (bia != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(bia->dimension(0) != output_shape[channel_idx],
                                        "Biases size and number of dst feature maps should match");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(bia->num_dimensions() > 1, "Biases should be one dimensional");
    }

    // 2. Check support level
    // Data type
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F16, DataType::F32);
    // Data layout
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(src, DataLayout::NHWC);
    // Texture in the input tensor
    ARM_COMPUTE_RETURN_ERROR_ON((settings.export_input_to_cl_image() == true));

    return Status{};
}

ClComponentDepthwiseConv2d::ClComponentDepthwiseConv2d(ComponentId                      id,
                                                       const Properties                &properties,
                                                       const ArgumentPack<ITensorInfo> &tensors,
                                                       const Attributes                &attributes,
                                                       const Settings                  &settings)
    : IGpuKernelComponent{id, properties, tensors},
#ifndef ACL_INTERNAL_TEST_CKW_IN_DF
      _component_writer{std::make_unique<ClTemplateDepthwiseConv2d>(id, tensors, attributes, settings)}
#else  //ACL_INTERNAL_TEST_CKW_IN_DF
      _component_writer{std::make_unique<GpuCkwDepthwiseConv2d>(id, tensors, attributes, settings)}
#endif //ACL_INTERNAL_TEST_CKW_IN_DF
{
    ARM_COMPUTE_UNUSED(attributes, settings);
}
ClComponentDepthwiseConv2d::~ClComponentDepthwiseConv2d()
{
}
#ifndef ACL_INTERNAL_TEST_CKW_IN_DF
const IGpuTemplateComponentWriter *ClComponentDepthwiseConv2d::template_writer() const
#else  //ACL_INTERNAL_TEST_CKW_IN_DF
const IGpuCkwComponentDriver *ClComponentDepthwiseConv2d::ckw_component_driver() const
#endif //ACL_INTERNAL_TEST_CKW_IN_DF
{
    return _component_writer.get();
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
