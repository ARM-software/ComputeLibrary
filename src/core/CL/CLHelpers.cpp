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
#include "arm_compute/core/Log.h"
#include "arm_compute/core/Types.h"

#include <map>
#include <regex>
#include <vector>

namespace
{
arm_compute::GPUTarget get_bifrost_target(const std::string &version)
{
    if(version == "70")
    {
        return arm_compute::GPUTarget::G70;
    }
    else
    {
        return arm_compute::GPUTarget::BIFROST;
    }
}

arm_compute::GPUTarget get_midgard_target(const std::string &version)
{
    switch(version[0])
    {
        case '6':
            return arm_compute::GPUTarget::T600;
        case '7':
            return arm_compute::GPUTarget::T700;
        case '8':
            return arm_compute::GPUTarget::T800;
        default:
            return arm_compute::GPUTarget::MIDGARD;
    }
}

bool extension_support(const cl::Device &device, const char *extension_name)
{
    std::string extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
    auto        pos        = extensions.find(extension_name);
    return (pos != std::string::npos);
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
        case DataType::QASYMM8:
            return "uchar";
        case DataType::U16:
            return "ushort";
        case DataType::S16:
            return "short";
        case DataType::QS16:
            return "qs16";
        case DataType::U32:
            return "uint";
        case DataType::S32:
            return "int";
        case DataType::QS32:
            return "qs32";
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

std::string get_data_size_from_data_type(const DataType &dt)
{
    switch(dt)
    {
        case DataType::U8:
        case DataType::QS8:
        case DataType::S8:
        case DataType::QASYMM8:
            return "8";
        case DataType::U16:
        case DataType::S16:
        case DataType::QS16:
        case DataType::F16:
            return "16";
        case DataType::U32:
        case DataType::S32:
        case DataType::F32:
            return "32";
        case DataType::U64:
        case DataType::S64:
            return "64";
        default:
            ARM_COMPUTE_ERROR("Unsupported input data type.");
            return "0";
    }
}

std::string get_underlying_cl_type_from_data_type(const DataType &dt)
{
    switch(dt)
    {
        case DataType::QS8:
            return "char";
        case DataType::QS16:
            return "short";
        case DataType::QS32:
            return "int";
        default:
            return get_cl_type_from_data_type(dt);
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
    size_t name_size = 0;

    // Query device name size
    cl_int err = clGetDeviceInfo(device.get(), CL_DEVICE_NAME, 0, nullptr, &name_size);
    ARM_COMPUTE_ERROR_ON_MSG((err != 0) || (name_size == 0), "clGetDeviceInfo failed to return valid information");
    ARM_COMPUTE_UNUSED(err);

    std::vector<char> name_buffer(name_size);

    // Query device name
    err = clGetDeviceInfo(device.get(), CL_DEVICE_NAME, name_size, name_buffer.data(), nullptr);
    ARM_COMPUTE_ERROR_ON_MSG(err != 0, "clGetDeviceInfo failed to return valid information");
    ARM_COMPUTE_UNUSED(err);

    std::regex  mali_regex(R"(Mali-([TG])(\d+))");
    std::string device_name(name_buffer.begin(), name_buffer.end());
    std::smatch name_parts;
    const bool  found_mali = std::regex_search(device_name, name_parts, mali_regex);

    if(!found_mali)
    {
        ARM_COMPUTE_LOG_INFO_MSG_CORE("Can't find valid Mali GPU. Target is set to MIDGARD.");
        return GPUTarget::MIDGARD;
    }

    const char         target  = name_parts.str(1)[0];
    const std::string &version = name_parts.str(2);

    switch(target)
    {
        case 'T':
            return get_midgard_target(version);
        case 'G':
            return get_bifrost_target(version);
        default:
            ARM_COMPUTE_LOG_INFO_MSG_CORE("Mali GPU unknown. Target is set to the default one.");
            return GPUTarget::MIDGARD;
    }
}

GPUTarget get_arch_from_target(GPUTarget target)
{
    return (target & GPUTarget::GPU_ARCH_MASK);
}

bool non_uniform_workgroup_support(const cl::Device &device)
{
    return extension_support(device, "cl_arm_non_uniform_work_group_size");
}

bool fp16_support(const cl::Device &device)
{
    return extension_support(device, "cl_khr_fp16");
}

CLVersion get_cl_version(const cl::Device &device)
{
    std::vector<char> version;
    size_t            version_size = 0;
    cl_int            err          = clGetDeviceInfo(device.get(), CL_DEVICE_VERSION, 0, nullptr, &version_size);
    ARM_COMPUTE_ERROR_ON_MSG((err != 0) || (version_size == 0), "clGetDeviceInfo failed to return valid information");
    ARM_COMPUTE_UNUSED(err);

    // Resize vector
    version.resize(version_size);
    // Query version
    err = clGetDeviceInfo(device.get(), CL_DEVICE_VERSION, version_size, version.data(), nullptr);
    ARM_COMPUTE_ERROR_ON_MSG(err != 0, "clGetDeviceInfo failed to return valid information");
    ARM_COMPUTE_UNUSED(err);

    std::string version_str(version.begin(), version.end());
    if(version_str.find("OpenCL 2") != std::string::npos)
    {
        return CLVersion::CL20;
    }
    else if(version_str.find("OpenCL 1.2") != std::string::npos)
    {
        return CLVersion::CL12;
    }
    else if(version_str.find("OpenCL 1.1") != std::string::npos)
    {
        return CLVersion::CL11;
    }
    else if(version_str.find("OpenCL 1.0") != std::string::npos)
    {
        return CLVersion::CL10;
    }

    return CLVersion::UNKNOWN;
}

} // namespace arm_compute
