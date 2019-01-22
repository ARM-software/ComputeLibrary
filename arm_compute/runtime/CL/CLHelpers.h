/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_CL_HELPERS_H__
#define __ARM_COMPUTE_CL_HELPERS_H__

#include "arm_compute/core/CL/OpenCL.h"

namespace arm_compute
{
/** This function creates an OpenCL context and a device.
 *
 * @note In debug builds, the function will automatically enable cl_arm_printf if the driver/device supports it.
 *
 * @return A std::tuple where the first element is the opencl context, the second element is the opencl device
 *         and the third one an error code. The error code will be CL_SUCCESS upon successful creation, otherwise
 *         a value telling why the function failed.
 */
std::tuple<cl::Context, cl::Device, cl_int> create_opencl_context_and_device();
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CL_HELPERS_H__ */
