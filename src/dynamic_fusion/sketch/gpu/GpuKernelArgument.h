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
        Tensor_4D_t_Image
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
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUKERNELARGUMENT */
