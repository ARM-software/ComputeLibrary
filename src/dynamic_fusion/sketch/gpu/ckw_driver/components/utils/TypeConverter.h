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
#ifndef ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_COMPONENTS_UTILS_TYPECONVERTER
#define ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_COMPONENTS_UTILS_TYPECONVERTER

#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "ckw/TensorInfo.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
inline ckw::DataType to_ckw(DataType dt)
{
    switch(dt)
    {
        case DataType::F32:
            return ckw::DataType::Fp32;
        case DataType::F16:
            return ckw::DataType::Fp16;
        case DataType::S32:
            return ckw::DataType::Int32;
        case DataType::S16:
            return ckw::DataType::Int16;
        case DataType::S8:
        case DataType::QASYMM8_SIGNED:
            return ckw::DataType::Int8;
        case DataType::U32:
            return ckw::DataType::Uint32;
        case DataType::U16:
            return ckw::DataType::Uint16;
        case DataType::U8:
        case DataType::QASYMM8:
            return ckw::DataType::Uint8;
        default:
            return ckw::DataType::Unknown;
    }
}

inline ckw::TensorShape to_ckw(const TensorShape &shape)
{
    ARM_COMPUTE_ERROR_ON(shape.num_max_dimensions < std::tuple_size<ckw::TensorShape> {});
    ARM_COMPUTE_ERROR_ON(std::tuple_size<ckw::TensorShape> {} != 5);
    /// NOTE: Overflow danger. Use size_t?
    return ckw::TensorShape
    {
        static_cast<int32_t>(shape[0]),
        static_cast<int32_t>(shape[1]),
        static_cast<int32_t>(shape[2]),
        static_cast<int32_t>(shape[3]),
        static_cast<int32_t>(shape[4])
    };
}
inline ckw::TensorDataLayout to_ckw(DataLayout dl)
{
    switch(dl)
    {
        case DataLayout::NHWC:
            return ckw::TensorDataLayout::Nhwc;
        case DataLayout::NDHWC:
            return ckw::TensorDataLayout::Ndhwc;
        default:
            return ckw::TensorDataLayout::Unknown;
    }
}
inline ckw::TensorInfo to_ckw(const ITensorInfo &tensor_info)
{
    return ckw::TensorInfo
    {
        to_ckw(tensor_info.data_type()),
        to_ckw(tensor_info.tensor_shape()),
        to_ckw(tensor_info.data_layout()),
        tensor_info.id()
    };
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_COMPONENTS_UTILS_TYPECONVERTER */
