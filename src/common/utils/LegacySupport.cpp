/*
 * Copyright (c) 2021, 2023 Arm Limited.
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
#include "src/common/utils/LegacySupport.h"

#include "arm_compute/function_info/ActivationLayerInfo.h"

namespace arm_compute
{
namespace detail
{
namespace
{
DataType convert_to_legacy_data_type(AclDataType data_type)
{
    switch (data_type)
    {
        case AclDataType::AclFloat32:
            return DataType::F32;
        case AclDataType::AclFloat16:
            return DataType::F16;
        case AclDataType::AclBFloat16:
            return DataType::BFLOAT16;
        default:
            return DataType::UNKNOWN;
    }
}

AclDataType convert_to_c_data_type(DataType data_type)
{
    switch (data_type)
    {
        case DataType::F32:
            return AclDataType::AclFloat32;
        case DataType::F16:
            return AclDataType::AclFloat16;
        case DataType::BFLOAT16:
            return AclDataType::AclBFloat16;
        default:
            return AclDataType::AclDataTypeUnknown;
    }
}

TensorShape create_legacy_tensor_shape(int32_t ndims, int32_t *shape)
{
    TensorShape legacy_shape{};
    for (int32_t d = 0; d < ndims; ++d)
    {
        legacy_shape.set(d, shape[d], false);
    }
    return legacy_shape;
}
int32_t *create_tensor_shape_array(const TensorInfo &info)
{
    const auto num_dims = info.num_dimensions();
    if (num_dims <= 0)
    {
        return nullptr;
    }

    int32_t *shape_array = new int32_t[num_dims];

    for (size_t d = 0; d < num_dims; ++d)
    {
        shape_array[d] = info.tensor_shape()[d];
    }

    return shape_array;
}
} // namespace

TensorInfo convert_to_legacy_tensor_info(const AclTensorDescriptor &desc)
{
    TensorInfo legacy_desc;
    legacy_desc.init(create_legacy_tensor_shape(desc.ndims, desc.shape), 1,
                     convert_to_legacy_data_type(desc.data_type));
    return legacy_desc;
}

AclTensorDescriptor convert_to_descriptor(const TensorInfo &info)
{
    const auto          num_dims = info.num_dimensions();
    AclTensorDescriptor desc{static_cast<int32_t>(num_dims), create_tensor_shape_array(info),
                             convert_to_c_data_type(info.data_type()), nullptr, 0};
    return desc;
}

ActivationLayerInfo convert_to_activation_info(const AclActivationDescriptor &desc)
{
    ActivationLayerInfo::ActivationFunction act;
    switch (desc.type)
    {
        case AclActivationType::AclIdentity:
            act = ActivationLayerInfo::ActivationFunction::IDENTITY;
            break;
        case AclActivationType::AclLogistic:
            act = ActivationLayerInfo::ActivationFunction::LOGISTIC;
            break;
        case AclActivationType::AclTanh:
            act = ActivationLayerInfo::ActivationFunction::TANH;
            break;
        case AclActivationType::AclRelu:
            act = ActivationLayerInfo::ActivationFunction::RELU;
            break;
        case AclActivationType::AclBoundedRelu:
            act = ActivationLayerInfo::ActivationFunction::BOUNDED_RELU;
            break;
        case AclActivationType::AclLuBoundedRelu:
            act = ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU;
            break;
        case AclActivationType::AclLeakyRelu:
            act = ActivationLayerInfo::ActivationFunction::LEAKY_RELU;
            break;
        case AclActivationType::AclSoftRelu:
            act = ActivationLayerInfo::ActivationFunction::SOFT_RELU;
            break;
        case AclActivationType::AclElu:
            act = ActivationLayerInfo::ActivationFunction::ELU;
            break;
        case AclActivationType::AclAbs:
            act = ActivationLayerInfo::ActivationFunction::ABS;
            break;
        case AclActivationType::AclSquare:
            act = ActivationLayerInfo::ActivationFunction::SQUARE;
            break;
        case AclActivationType::AclSqrt:
            act = ActivationLayerInfo::ActivationFunction::SQRT;
            break;
        case AclActivationType::AclLinear:
            act = ActivationLayerInfo::ActivationFunction::LINEAR;
            break;
        case AclActivationType::AclHardSwish:
            act = ActivationLayerInfo::ActivationFunction::HARD_SWISH;
            break;
        default:
            return ActivationLayerInfo();
    }

    return ActivationLayerInfo(act, desc.a, desc.b);
}
} // namespace detail
} // namespace arm_compute
