/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLTypes.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Types.h"

#include <map>
#include <vector>

namespace
{
arm_compute::GPUTarget get_bifrost_target(const std::string &name)
{
    arm_compute::GPUTarget target = arm_compute::GPUTarget::MIDGARD;

    if(name == "G7")
    {
        target = arm_compute::GPUTarget::G70;
    }

    return target;
}

arm_compute::GPUTarget get_midgard_target(const std::string &name)
{
    arm_compute::GPUTarget target = arm_compute::GPUTarget::MIDGARD;

    if(name == "T6")
    {
        target = arm_compute::GPUTarget::T600;
    }
    else if(name == "T7")
    {
        target = arm_compute::GPUTarget::T700;
    }
    else if(name == "T8")
    {
        target = arm_compute::GPUTarget::T800;
    }

    return target;
}
} // namespace

namespace arm_compute
{
std::string get_cl_type_from_data_type(const DataType &dt)
{
    switch(dt)
    {
        case DataType::U8:
            return "uchar";
        case DataType::QS8:
            return "qs8";
        case DataType::S8:
            return "char";
        case DataType::U16:
            return "ushort";
        case DataType::S16:
            return "short";
        case DataType::U32:
            return "uint";
        case DataType::S32:
            return "int";
        case DataType::U64:
            return "ulong";
        case DataType::S64:
            return "long";
        case DataType::F16:
            return "half";
        case DataType::F32:
            return "float";
        default:
            ARM_COMPUTE_ERROR("Unsupported input data type.");
            return "";
    }
}

const std::string &string_from_target(GPUTarget target)
{
    static std::map<GPUTarget, const std::string> gpu_target_map =
    {
        { GPUTarget::MIDGARD, "midgard" },
        { GPUTarget::BIFROST, "bifrost" },
        { GPUTarget::T600, "t600" },
        { GPUTarget::T700, "t700" },
        { GPUTarget::T800, "t800" },
        { GPUTarget::G70, "g70" }
    };

    return gpu_target_map[target];
}

GPUTarget get_target_from_device(cl::Device &device)
{
    const std::string name_mali("Mali-");
    GPUTarget         target{ GPUTarget::MIDGARD };

    size_t            name_size = 0;
    std::vector<char> name;

    // Query device name size
    cl_int err = clGetDeviceInfo(device.get(), CL_DEVICE_NAME, 0, nullptr, &name_size);
    ARM_COMPUTE_ERROR_ON_MSG((err != 0) || (name_size == 0), "clGetDeviceInfo failed to return valid information");
    // Resize vector
    name.resize(name_size);
    // Query device name
    err = clGetDeviceInfo(device.get(), CL_DEVICE_NAME, name_size, name.data(), nullptr);
    ARM_COMPUTE_ERROR_ON_MSG(err != 0, "clGetDeviceInfo failed to return valid information");
    ARM_COMPUTE_UNUSED(err);

    std::string name_str(name.begin(), name.end());
    auto        pos = name_str.find(name_mali);

    if(pos != std::string::npos)
    {
        ARM_COMPUTE_ERROR_ON_MSG((pos + name_mali.size() + 2) > name_str.size(), "Device name is shorter than expected.");
        std::string sub_name = name_str.substr(pos + name_mali.size(), 2);

        if(sub_name[0] == 'G')
        {
            target = get_bifrost_target(sub_name);
        }
        else if(sub_name[0] == 'T')
        {
            target = get_midgard_target(sub_name);
        }
        else
        {
            ARM_COMPUTE_INFO("Mali GPU unknown. Target is set to the default one.");
        }
    }
    else
    {
        ARM_COMPUTE_INFO("Can't find valid Mali GPU. Target is set to the default one.");
    }

    return target;
}

GPUTarget get_arch_from_target(GPUTarget target)
{
    return (target & GPUTarget::GPU_ARCH_MASK);
}
} // namespace arm_compute
