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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUKERNELARGUMENT
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUKERNELARGUMENT

#include "arm_compute/core/TensorInfo.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Contain information required to set up a kernel argument at run time
 * @deprecated To be removed along with ClTemplateWriter
 */
struct GpuKernelArgumentInfo
{
    /** Enumerate all the tensor arguments variants used by all kernel implementations.  */
    enum class Type : int
    {
        Scalar,

        Vector,

        Image,
        Image_Reinterpret_As_3D,
        Image_Export_To_ClImage2D,

        Image_3D, // 3D Tensor represented as a 2D Image + stride_z
        Image_3D_Export_To_ClImage2D,

        Tensor_3D,
        Tensor_4D,
        Tensor_4D_t_Buffer,
        Tensor_4D_t_Image,

        Tensor_Special_0,
    };
    /** Default constructor */
    GpuKernelArgumentInfo() = default;
    /** Constructor */
    GpuKernelArgumentInfo(Type type)
        : type{ type }
    {
    }
    Type type{ Type::Tensor_4D_t_Buffer };
};
bool operator==(const GpuKernelArgumentInfo &info0, const GpuKernelArgumentInfo &info1);
/** Kernel argument information linked with its corresponding @ref ITensorInfo
 * @deprecated To be removed along with ClTemplateWriter
 */
class GpuKernelArgument
{
public:
    /** Constructor
     *
     * @param[in] tensor_info     Associated @ref ITensorInfo
     * @param[in] kernel_arg_info Associated @ref GpuKernelArgumentInfo
     */
    GpuKernelArgument(const ITensorInfo           &tensor_info,
                      const GpuKernelArgumentInfo &kernel_arg_info)
        : _tensor_info{ tensor_info },
          _kernel_arg_info{ kernel_arg_info }
    {
    }
    /** Get workload tensor id */
    ITensorInfo::Id id() const
    {
        return _tensor_info.id();
    }
    /** Get associated @ref ITensorInfo */
    ITensorInfo *tensor_info()
    {
        return &_tensor_info;
    }
    /** Get associated @ref ITensorInfo */
    const ITensorInfo *tensor_info() const
    {
        return &_tensor_info;
    }
    /** Get associated @ref GpuKernelArgumentInfo */
    GpuKernelArgumentInfo *kernel_argument_info()
    {
        return &_kernel_arg_info;
    }
    /** Get associated @ref GpuKernelArgumentInfo */
    const GpuKernelArgumentInfo *kernel_argument_info() const
    {
        return &_kernel_arg_info;
    }
    /** Check if the associated workload tensor has valid id
     *
     * @return true if has valid id
     * @return false  otherwise
     */
    bool has_valid_id() const
    {
        return _tensor_info.has_valid_id();
    }

private:
    TensorInfo            _tensor_info{};
    GpuKernelArgumentInfo _kernel_arg_info{};
};
#ifdef ACL_INTERNAL_TEST_CKW_IN_DF
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
        : _type{ Type::TensorStorage }, _id{ id }, _value{}
    {
        _value.tensor_storage_type = storage;
    }
    GpuKernelArgumentBinding(ITensorInfo::Id id, TensorComponentType component)
        : _type{ Type::TensorComponent }, _id{ id }, _value{}
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
#endif // ACL_INTERNAL_TEST_CKW_IN_DF

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUKERNELARGUMENT */
