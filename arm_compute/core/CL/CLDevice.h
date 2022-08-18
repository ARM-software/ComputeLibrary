/*
 * Copyright (c) 2020-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CLDEVICE_H
#define ARM_COMPUTE_CLDEVICE_H

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLTypes.h"
#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/IDevice.h"

#include <set>
#include <sstream>
#include <string>

namespace arm_compute
{
/**  OpenCL device type class
 *
 *   Initializes and stores all the information about a cl device,
 *   working mainly as a cache mechanism.
 * */
class CLDevice : public IDevice
{
public:
    /** Default Constructor */
    CLDevice()
        : _device(cl::Device()), _options()
    {
    }

    /** Constructor
     *
     * @param[in] cl_device OpenCL device
     */
    CLDevice(const cl::Device &cl_device)
        : _device(), _options()
    {
        _device = cl_device;

        // Get device target
        std::string device_name = _device.getInfo<CL_DEVICE_NAME>();
        _options.gpu_target     = get_target_from_name(device_name);

        // Fill extensions
        std::string extensions = _device.getInfo<CL_DEVICE_EXTENSIONS>();

        std::istringstream iss(extensions);
        for(std::string s; iss >> s;)
        {
            _options.extensions.insert(s);
        }

        // SW workaround for G76
        if(_options.gpu_target == GPUTarget::G76)
        {
            _options.extensions.insert("cl_arm_integer_dot_product_int8");
        }

        // Get device version
        _options.version = get_cl_version(_device);

        // Get compute units
        _options.compute_units = _device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

        // Get device version
        _options.device_version = _device.getInfo<CL_DEVICE_VERSION>();
    }

    /** Returns the GPU target of the cl device
     *
     * @return The GPU target
     */
    const GPUTarget &target() const
    {
        return _options.gpu_target;
    }

    /** Returns the number of compute units available
     *
     * @return Number of compute units
     */
    size_t compute_units() const
    {
        return _options.compute_units;
    }

    /** Returns the underlying cl device object
     *
     * @return A cl device
     */
    const cl::Device &cl_device() const
    {
        return _device;
    }

    /** Returns the device's CL version
     *
     * @return CLVersion of the device
     */
    CLVersion version() const
    {
        return _options.version;
    }

    /** Returns the device version as a string
     *
     * @return CLVersion of the device
     */
    std::string device_version() const
    {
        return _options.device_version;
    }

    // Inherrited methods
    DeviceType type() const override
    {
        return DeviceType::CL;
    }

    bool supported(const std::string &extension) const override
    {
        return _options.extensions.count(extension) != 0;
    }

    /** Returns whether non-uniform workgroup is supported and the build options.
     *
     * If the feature is supported, the appropriate build options will be
     * appended to the specified string.
     *
     * @return A tuple (supported, build_options) indicating whether the feature
     *         is supported and the corresponding build options to enable it.
     */
    std::tuple<bool, std::string> is_non_uniform_workgroup_supported() const
    {
        if(version() == CLVersion::CL30 && get_cl_non_uniform_work_group_supported(_device))
        {
            return {true, " -cl-std=CL3.0 "};
        }
        else if(version() == CLVersion::CL20)
        {
            return {true, " -cl-std=CL2.0 "};
        }
        else if(supported("cl_arm_non_uniform_work_group_size"))
        {
            return {true, " -cl-arm-non-uniform-work-group-size "};
        }

        return {false, ""};
    }

private:
    cl::Device             _device;  /**< OpenCL device. */
    struct CLDeviceOptions _options; /**< OpenCL device options */
};

} // namespace arm_compute

#endif /* ARM_COMPUTE_CLDEVICE_H */
