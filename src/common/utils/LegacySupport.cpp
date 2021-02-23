/*
 * Copyright (c) 2021 Arm Limited.
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

namespace arm_compute
{
namespace detail
{
namespace
{
DataType data_type_mapper(AclDataType data_type)
{
    switch(data_type)
    {
        case AclDataType::AclFloat32:
            return DataType::F32;
        case AclDataType::AclFloat16:
            return DataType::F16;
        case AclDataType::AclBFloat16:
            return DataType::BFLOAT16;
        default:
            return DataType::UNKNOWN;
            ;
    }
}

TensorShape tensor_shape_mapper(int32_t ndims, int32_t *shape)
{
    TensorShape legacy_shape{};
    for(int32_t d = 0; d < ndims; ++d)
    {
        legacy_shape.set(d, shape[d], false);
    }
    return legacy_shape;
}
} // namespace

TensorInfo convert_to_legacy_tensor_info(const AclTensorDescriptor &desc)
{
    TensorInfo legacy_desc;
    legacy_desc.init(tensor_shape_mapper(desc.ndims, desc.shape), 1, data_type_mapper(desc.data_type));
    return legacy_desc;
}
} // namespace detail
} // namespace arm_compute
