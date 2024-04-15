/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CL_HELPERS_H
#define ARM_COMPUTE_CL_HELPERS_H

#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/runtime/CL/CLTypes.h"
#include "arm_compute/runtime/IScheduler.h"

namespace arm_compute
{
// Forward declarations
class CLRuntimeContext;
class ICLKernel;

/** This function creates an OpenCL context and a device.
 *
 * @note In debug builds, the function will automatically enable cl_arm_printf if the driver/device supports it.
 *
 * @param[in] cl_backend_type The OpenCL backend type to use.
 *
 * @return A std::tuple where the first element is the opencl context, the second element is the opencl device
 *         and the third one an error code. The error code will be CL_SUCCESS upon successful creation, otherwise
 *         a value telling why the function failed.
 */
std::tuple<cl::Context, cl::Device, cl_int> create_opencl_context_and_device(CLBackendType cl_backend_type);
/** Schedules a kernel using the context if not nullptr else uses the legacy scheduling flow.
 *
 * @param[in] ctx    Context to use.
 * @param[in] kernel Kernel to schedule.
 * @param[in] flush  (Optional) Specifies if the command queue will be flushed after running the kernel.
 */
void schedule_kernel_on_ctx(CLRuntimeContext *ctx, ICLKernel *kernel, bool flush = true);

/** This function selects the OpenCL platform based on the backend type.
 *
 * @param[in] cl_backend_type The OpenCL backend type to use.
 *
 * @return A cl::Platform object.
 */
cl::Platform select_preferable_platform(CLBackendType cl_backend_type);
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_HELPERS_H */
