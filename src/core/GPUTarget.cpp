/*
 * Copyright (c) 2018-2022 Arm Limited.
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
#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/Log.h"

#include <map>
#include <regex>

namespace
{
arm_compute::GPUTarget get_valhall_target(const std::string &version)
{
    if(version.find("G77") != std::string::npos)
    {
        return arm_compute::GPUTarget::G77;
    }
    if(version.find("G78") != std::string::npos)
    {
        return arm_compute::GPUTarget::G78;
    }
    else if(version.find("G710") != std::string::npos)
    {
        return arm_compute::GPUTarget::G710;
    }
    else if(version.find("G57") != std::string::npos)
    {
        return arm_compute::GPUTarget::G57;
    }
    else
    {
        return arm_compute::GPUTarget::UNKNOWN;
    }
}

arm_compute::GPUTarget get_bifrost_target(const std::string &version)
{
    if(version.find("G71") != std::string::npos)
    {
        return arm_compute::GPUTarget::G71;
    }
    else if(version.find("G72") != std::string::npos)
    {
        return arm_compute::GPUTarget::G72;
    }
    else if(version.find("G51BIG") != std::string::npos)
    {
        return arm_compute::GPUTarget::G51BIG;
    }
    else if(version.find("G51LIT") != std::string::npos)
    {
        return arm_compute::GPUTarget::G51LIT;
    }
    else if(version.find("G51") != std::string::npos)
    {
        return arm_compute::GPUTarget::G51;
    }
    else if(version.find("G52LIT") != std::string::npos)
    {
        return arm_compute::GPUTarget::G52LIT;
    }
    else if(version.find("G52") != std::string::npos)
    {
        return arm_compute::GPUTarget::G52;
    }
    else if(version.find("G76") != std::string::npos)
    {
        return arm_compute::GPUTarget::G76;
    }
    else if(version.find("G31") != std::string::npos)
    {
        return arm_compute::GPUTarget::G31;
    }
    else
    {
        return arm_compute::GPUTarget::UNKNOWN;
    }
}

arm_compute::GPUTarget get_midgard_target(const std::string &version)
{
    if(version.find("T600") != std::string::npos)
    {
        return arm_compute::GPUTarget::T600;
    }
    else if(version.find("T700") != std::string::npos)
    {
        return arm_compute::GPUTarget::T700;
    }
    else if(version.find("T800") != std::string::npos)
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
const std::string &string_from_target(GPUTarget target)
{
    static std::map<GPUTarget, const std::string> gpu_target_map =
    {
        { GPUTarget::MIDGARD, "midgard" },
        { GPUTarget::BIFROST, "bifrost" },
        { GPUTarget::VALHALL, "valhall" },
        { GPUTarget::T600, "t600" },
        { GPUTarget::T700, "t700" },
        { GPUTarget::T800, "t800" },
        { GPUTarget::G71, "g71" },
        { GPUTarget::G72, "g72" },
        { GPUTarget::G51, "g51" },
        { GPUTarget::G51BIG, "g51big" },
        { GPUTarget::G51LIT, "g51lit" },
        { GPUTarget::G52, "g52" },
        { GPUTarget::G52LIT, "g52lit" },
        { GPUTarget::G76, "g76" },
        { GPUTarget::G77, "g77" },
        { GPUTarget::G78, "g78" },
        { GPUTarget::G710, "g710" },
        { GPUTarget::G57, "g57" }
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
        ARM_COMPUTE_LOG_INFO_MSG_CORE("Can't find valid Arm® Mali™ GPU. Target is set to default.");
        return GPUTarget::MIDGARD;
    }

    const char         target  = name_parts.str(1)[0];
    const std::string &version = name_parts.str(1);

    std::regex future_regex(R"(.*X)");
    const bool is_future_gpu = std::regex_search(version, future_regex);

    // Work-out gpu target
    GPUTarget gpu_target;
    if(target == 'G' || is_future_gpu)
    {
        // Check for Valhall or Bifrost
        gpu_target = get_valhall_target(version);
        if(gpu_target == GPUTarget::UNKNOWN)
        {
            gpu_target = get_bifrost_target(version);
        }

        // Default GPUTarget
        if(gpu_target == GPUTarget::UNKNOWN)
        {
            gpu_target = GPUTarget::VALHALL;
        }
    }
    else if(target == 'T')
    {
        gpu_target = get_midgard_target(version);
    }
    else
    {
        gpu_target = GPUTarget::UNKNOWN;
    }

    // Report in case of unknown target
    if(gpu_target == GPUTarget::UNKNOWN)
    {
        ARM_COMPUTE_LOG_INFO_MSG_CORE("Arm® Mali™ Mali GPU unknown. Target is set to the default one. (BIFROST)");
        return GPUTarget::BIFROST;
    }

    return gpu_target;
}

GPUTarget get_arch_from_target(GPUTarget target)
{
    return (target & GPUTarget::GPU_ARCH_MASK);
}
} // namespace arm_compute
