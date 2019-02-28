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

#include "arm_compute/runtime/CL/CLHelpers.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/Error.h"

namespace
{
#if defined(ARM_COMPUTE_ASSERTS_ENABLED)
void printf_callback(const char *buffer, unsigned int len, size_t complete, void *user_data)
{
    printf("%.*s", len, buffer);
}
#endif /* defined(ARM_COMPUTE_ASSERTS_ENABLED) */

/** This initialises the properties vector with the configuration to be used when creating the opencl context
 *
 * @param[in] platform The opencl platform used to create the context
 * @param[in] device   The opencl device to be used to create the context
 * @param[in] prop     An array of properties to be initialised
 *
 * @note In debug builds, this function will enable cl_arm_printf if it's supported.
 *
 * @return A pointer to the context properties which can be used to create an opencl context
 */

void initialise_context_properties(const cl::Platform &platform, const cl::Device &device, cl_context_properties prop[7])
{
    ARM_COMPUTE_UNUSED(device);
#if defined(ARM_COMPUTE_ASSERTS_ENABLED)
    // Query devices in the context for cl_arm_printf support
    if(arm_compute::device_supports_extension(device, "cl_arm_printf"))
    {
        // Create a cl_context with a printf_callback and user specified buffer size.
        cl_context_properties properties_printf[] =
        {
            CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform()),
            // Enable a printf callback function for this context.
            CL_PRINTF_CALLBACK_ARM, reinterpret_cast<cl_context_properties>(printf_callback),
            // Request a minimum printf buffer size of 4MB for devices in the
            // context that support this extension.
            CL_PRINTF_BUFFERSIZE_ARM, 0x1000,
            0
        };
        std::copy_n(properties_printf, 7, prop);
    }
    else
#endif // defined(ARM_COMPUTE_ASSERTS_ENABLED)
    {
        cl_context_properties properties[] =
        {
            CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform()),
            0
        };
        std::copy_n(properties, 3, prop);
    };
}
} //namespace

namespace arm_compute
{
std::tuple<cl::Context, cl::Device, cl_int>
create_opencl_context_and_device()
{
    ARM_COMPUTE_ERROR_ON(!opencl_is_available());
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    ARM_COMPUTE_ERROR_ON_MSG(platforms.size() == 0, "Couldn't find any OpenCL platform");
    cl::Platform            p = platforms[0];
    cl::Device              device;
    std::vector<cl::Device> platform_devices;
    p.getDevices(CL_DEVICE_TYPE_DEFAULT, &platform_devices);
    ARM_COMPUTE_ERROR_ON_MSG(platform_devices.size() == 0, "Couldn't find any OpenCL device");
    device                              = platform_devices[0];
    cl_int                err           = CL_SUCCESS;
    cl_context_properties properties[7] = { 0, 0, 0, 0, 0, 0, 0 };
    initialise_context_properties(p, device, properties);
    cl::Context cl_context = cl::Context(device, properties, nullptr, nullptr, &err);
    ARM_COMPUTE_ERROR_ON_MSG(err != CL_SUCCESS, "Failed to create OpenCL context");
    return std::make_tuple(cl_context, device, err);
}
} // namespace arm_compute
