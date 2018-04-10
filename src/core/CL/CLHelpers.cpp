/*
 * Copyright (c) 2016-2018 ARM Limited.
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
    if(version == "G71")
    {
        return arm_compute::GPUTarget::G71;
    }
    else if(version == "G72")
    {
        return arm_compute::GPUTarget::G72;
    }
    else if(version == "G51")
    {
        return arm_compute::GPUTarget::G51;
    }
    else if(version == "G51BIG")
    {
        return arm_compute::GPUTarget::G51BIG;
    }
    else if(version == "G51LIT")
    {
        return arm_compute::GPUTarget::G51LIT;
    }
    else if(version == "TNOX")
    {
        return arm_compute::GPUTarget::TNOX;
    }
    else if(version == "TTRX")
    {
        return arm_compute::GPUTarget::TTRX;
    }
    else if(version == "TBOX")
    {
        return arm_compute::GPUTarget::TBOX;
    }
    else
    {
        return arm_compute::GPUTarget::BIFROST;
    }
}

arm_compute::GPUTarget get_midgard_target(const std::string &version)
{
    if(version == "T600")
    {
        return arm_compute::GPUTarget::T600;
    }
    else if(version == "T700")
    {
        return arm_compute::GPUTarget::T700;
    }
    else if(version == "T800")
    {
        return arm_compute::GPUTarget::T800;
    }
    else
    {
        return arm_compute::GPUTarget::MIDGARD;
    }
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
        { GPUTarget::G71, "g71" },
        { GPUTarget::G72, "g72" },
        { GPUTarget::G51, "g51" },
        { GPUTarget::G51BIG, "g51big" },
        { GPUTarget::G51LIT, "g51lit" },
        { GPUTarget::TNOX, "tnox" },
        { GPUTarget::TTRX, "ttrx" },
        { GPUTarget::TBOX, "tbox" }
    };

    return gpu_target_map[target];
}

GPUTarget get_target_from_name(const std::string &device_name)
{
    std::regex  mali_regex(R"(Mali-(.*))");
    std::smatch name_parts;
    const bool  found_mali = std::regex_search(device_name, name_parts, mali_regex);

    if(!found_mali)
    {
        ARM_COMPUTE_LOG_INFO_MSG_CORE("Can't find valid Mali GPU. Target is set to UNKNOWN.");
        return GPUTarget::UNKNOWN;
    }

    const char         target  = name_parts.str(1)[0];
    const std::string &version = name_parts.str(1);

    std::regex future_regex(R"(.*X)");
    const bool is_future_bifrost = std::regex_search(version, future_regex);

    if(target == 'G' || is_future_bifrost)
    {
        return get_bifrost_target(version);
    }
    else if(target == 'T')
    {
        return get_midgard_target(version);
    }
    else
    {
        ARM_COMPUTE_LOG_INFO_MSG_CORE("Mali GPU unknown. Target is set to the default one.");
        return GPUTarget::UNKNOWN;
    }
}

GPUTarget get_target_from_device(cl::Device &device)
{
    // Query device name size
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();

    return get_target_from_name(device_name);
}

GPUTarget get_arch_from_target(GPUTarget target)
{
    return (target & GPUTarget::GPU_ARCH_MASK);
}

bool non_uniform_workgroup_support(const cl::Device &device)
{
    return device_supports_extension(device, "cl_arm_non_uniform_work_group_size");
}

bool fp16_support(const cl::Device &device)
{
    return device_supports_extension(device, "cl_khr_fp16");
}

CLVersion get_cl_version(const cl::Device &device)
{
    std::string version_str = device.getInfo<CL_DEVICE_VERSION>();
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


bool device_supports_extension(const cl::Device &device, const char *extension_name)
{
    std::string extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
    auto        pos        = extensions.find(extension_name);
    return (pos != std::string::npos);
}

} // namespace arm_compute
