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

#ifndef ACL_SRC_DYNAMIC_FUSION_RUNTIME_GPU_CL_CKW_DRIVER_GPUCKWKERNELARGUMENTSHELPERS
#define ACL_SRC_DYNAMIC_FUSION_RUNTIME_GPU_CL_CKW_DRIVER_GPUCKWKERNELARGUMENTSHELPERS

#include "arm_compute/core/CL/ICLTensor.h"

#include "src/dynamic_fusion/sketch/gpu/GpuKernelArgument.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Select a Compute Kernel Writer tensor component from a tensor and add to the kernel's arguments at the specified index idx.
 *
 * @param[in,out] kernel    OpenCL kernel to configure with the provided argument.
 * @param[in,out] idx       Index at which to add the argument.
 * @param[in]     tensor    Tensor from which to access the tensor component.
 * @param[in]     component Tensor component to select such as tensor dimensions, strides, etc.
 */
void cl_add_tensor_component_argument(cl::Kernel         &kernel,
                                      unsigned int       &idx,
                                      const ICLTensor    *tensor,
                                      TensorComponentType component);

/** Add an OpenCL buffer object to the kernel's arguments at the specified index @p idx.
 *
 * @param[in,out] kernel OpenCL kernel to configure with the provided argument.
 * @param[in,out] idx    Index at which to add the argument.
 * @param[in]     buffer OpenCL buffer containing the tensor's data.
 */
void cl_add_buffer_argument(cl::Kernel &kernel, unsigned int &idx, const cl::Buffer &buffer);

/** Add an OpenCL image object to the kernel's arguments at the specified index @p idx.
 *
 * @param[in,out] kernel OpenCL kernel to configure with the provided argument.
 * @param[in,out] idx    Index at which to add the argument.
 * @param[in]     image  OpenCL image containing the image's data.
 */
void cl_add_texture_argument(cl::Kernel &kernel, unsigned int &idx, const cl::Image &image);

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif /* ACL_SRC_DYNAMIC_FUSION_RUNTIME_GPU_CL_CKW_DRIVER_GPUCKWKERNELARGUMENTSHELPERS */
