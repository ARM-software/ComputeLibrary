/*
 * Copyright (c) 2022-2024 Arm Limited.
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
#ifndef ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUKERNELARGUMENT_H
#define ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUKERNELARGUMENT_H

#include "arm_compute/core/TensorInfo.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Describe how the tensor runtime memory can be accessed
 *
 * Please see documentation under @ref GpuKernelArgumentBinding
 */
enum class TensorStorageType
{
    Unknown,
    ClBufferUint8Ptr,
    ClImage2dReadOnly,
    ClImage2dWriteOnly,
};

/** Describe additional runtime information about the tensor
 *
 * Please see documentation under @ref GpuKernelArgumentBinding
 */
enum class TensorComponentType
{
    Unknown,
    OffsetFirstElement,
    Stride0,
    Stride1,
    Stride2,
    Stride3,
    Stride4,
    Dim0,
    Dim1,
    Dim2,
    Dim3,
    Dim4,
    Dim1xDim2,
    Dim2xDim3,
    Dim1xDim2xDim3,
};

/** Describe how to extract information from a runtime Gpu tensor, and set it as an argument to a gpu kernel at runtime
 *
 * A kernel argument is just an argument to the gpu kernel as shown in the argument list below. This contrasts with a "workload argument" which is a tensor (@ref GpuWorkloadArgument)
 * void kernel(arg0, arg1, ... argN)
 *
 * In a kernel generated using dynamic fusion (@ref GpuKernelSourceCode), every kernel argument describes part of a tensor.
 * A tensor is described as: **storages** followed by **components**
 *
 * A storage (@ref TensorStorageType) describes how the tensor runtime memory can be accessed (e.g. via a global uint8 pointer to a CL buffer)
 * A component (@ref TensorComponentType) describes additional runtime information about the tensor (e.g. the dimensions of the tensor)
 *
 * The arguments are arranged in the order of use in the generated kernel code:
 *
 *  arg0   , arg1      , arg2      ,                         ...,                         , argN
 *  storage, component0, component1, ..., componentX, storage, component0, component1, ..., componentY
 * |                   tensor0                       |                    tensor1                    |
 *
 * An example argument list:
 *
 * void kernel(
 *  image2d_t       t0_image,               // TensorStorageType::ClImage2dReadOnly
 *  uint8_t*        t0_ptr,                 // TensorStorageType::ClBufferUint8Ptr
 *  uint            t0_dim0,                // TensorComponentType::Dim0
 *  uint            t0_stride1,             // TensorComponentType::Stride1
 *  image2d_t       t1_ptr,                 // TensorStorageType::ClImage2dReadOnly
 *  uint            t1_dim1xdim2,           // TensorComponentType::Dim1xDim2
 *  uint            t1_stride1,             // TensorComponentType::Stride1
 *  uint            t1_stride2,             // TensorComponentType:Stride2
 * )
 *
 */
class GpuKernelArgumentBinding
{
public:
    enum class Type : int32_t
    {
        TensorStorage,  /** @ref TensorStorageType */
        TensorComponent /** @ref TensorComponentType */
    };
    GpuKernelArgumentBinding(ITensorInfo::Id id, TensorStorageType storage)
        : _type{Type::TensorStorage}, _id{id}, _value{}
    {
        _value.tensor_storage_type = storage;
    }
    GpuKernelArgumentBinding(ITensorInfo::Id id, TensorComponentType component)
        : _type{Type::TensorComponent}, _id{id}, _value{}
    {
        _value.tensor_component_type = component;
    }
    /** Storage type of the tensor
     */
    TensorStorageType tensor_storage_type() const
    {
        ARM_COMPUTE_ERROR_ON(_type != Type::TensorStorage);
        return _value.tensor_storage_type;
    }
    /** Component of the tensor
     */
    TensorComponentType tensor_component_type() const
    {
        ARM_COMPUTE_ERROR_ON(_type != Type::TensorComponent);
        return _value.tensor_component_type;
    }
    /** Id of the tensor this kernel argument belongs to
     */
    ITensorInfo::Id id() const
    {
        return _id;
    }
    /** Type of the kernel argument
     */
    Type type() const
    {
        return _type;
    }

private:
    Type            _type;
    ITensorInfo::Id _id;
    union Value
    {
        TensorStorageType   tensor_storage_type;
        TensorComponentType tensor_component_type;
    };
    Value _value;
};

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif // ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUKERNELARGUMENT_H
