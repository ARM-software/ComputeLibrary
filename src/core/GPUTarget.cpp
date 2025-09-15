/*
 * Copyright (c) 2018-2025 Arm Limited.
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
#include <string>

namespace arm_compute
{

namespace
{
GPUTarget get_gpu_target(const std::string &full_name, const std::string &base_name)
{
    const static std::map<const std::string, GPUTarget> name_to_target_map = {
        // Midgard
        {"T600", GPUTarget::T600}, //
        {"T700", GPUTarget::T700}, //
        {"T800", GPUTarget::T800}, //
        // Bifrost
        {"G31", GPUTarget::G31},       //
        {"G51", GPUTarget::G51},       //
        {"G51LIT", GPUTarget::G51LIT}, //
        {"G51BIG", GPUTarget::G51BIG}, //
        {"G71", GPUTarget::G71},       //
        {"G52", GPUTarget::G52},       //
        {"G52LIT", GPUTarget::G52LIT}, //
        {"G72", GPUTarget::G72},       //
        {"G76", GPUTarget::G76},       //
        // Vallhall
        {"G57", GPUTarget::G57},              //
        {"G77", GPUTarget::G77},              //
        {"G68", GPUTarget::G68},              //
        {"G78", GPUTarget::G78},              //
        {"G78AE", GPUTarget::G78AE},          //
        {"G310", GPUTarget::G310},            //
        {"G510", GPUTarget::G510},            //
        {"G610", GPUTarget::G610},            //
        {"G710", GPUTarget::G710},            //
        {"G615", GPUTarget::G615},            //
        {"G715", GPUTarget::G715},            //
        {"Immortalis-G715", GPUTarget::G715}, //
        // 5th Gen
        {"G620", GPUTarget::G620},            //
        {"G720", GPUTarget::G720},            //
        {"Immortalis-G720", GPUTarget::G720}, //

    };
    // Try full name with variant.
    auto it = name_to_target_map.find(full_name);
    if (it != name_to_target_map.end())
    {
        return it->second;
    }
    // Try name without variant.
    it = name_to_target_map.find(base_name);
    if (it != name_to_target_map.end())
    {
        return it->second;
    }
    // Try architecture name only.
    if (!base_name.empty())
    {
        if (base_name[0] == 'G')
        {
            return GPUTarget::FIFTHGEN;
        }
        else if (base_name[0] == 'T')
        {
            return GPUTarget::MIDGARD;
        }
    }

    // Othewise, assume it's bifrost.
    ARM_COMPUTE_LOG_INFO_MSG_CORE("Arm® Mali™ Mali GPU unknown. Target is set to the default one. (FIFTHGEN)");
    return GPUTarget::FIFTHGEN;
}
} // namespace

const std::string &string_from_target(GPUTarget target)
{
    static std::map<GPUTarget, const std::string> gpu_target_map = {
        {GPUTarget::MIDGARD, "midgard"},  {GPUTarget::BIFROST, "bifrost"}, {GPUTarget::VALHALL, "valhall"},
        {GPUTarget::FIFTHGEN, "5th Gen"},

        {GPUTarget::T600, "t600"},        {GPUTarget::T700, "t700"},       {GPUTarget::T800, "t800"},
        {GPUTarget::G71, "g71"},          {GPUTarget::G72, "g72"},         {GPUTarget::G51, "g51"},
        {GPUTarget::G51BIG, "g51big"},    {GPUTarget::G51LIT, "g51lit"},   {GPUTarget::G31, "g31"},
        {GPUTarget::G76, "g76"},          {GPUTarget::G52, "g52"},         {GPUTarget::G52LIT, "g52lit"},
        {GPUTarget::G77, "g77"},          {GPUTarget::G57, "g57"},         {GPUTarget::G78, "g78"},
        {GPUTarget::G68, "g68"},          {GPUTarget::G78AE, "g78ae"},     {GPUTarget::G710, "g710"},
        {GPUTarget::G610, "g610"},        {GPUTarget::G510, "g510"},       {GPUTarget::G310, "g310"},
        {GPUTarget::G715, "g715"},        {GPUTarget::G615, "g615"},       {GPUTarget::G720, "g720"},
        {GPUTarget::G620, "g620"}};

    return gpu_target_map[target];
}

GPUTarget get_target_from_name(const std::string &device_name)
{
    std::regex mali_regex(R"(Mali-(([A-Za-z]+\d*)\w*))");
    std::regex immortalis_regex(R"(Immortalis-(([A-Za-z]+\d*)\w*))");

    std::smatch name_parts_mali;
    std::smatch name_parts_immortalis;

    const bool found_mali       = std::regex_search(device_name, name_parts_mali, mali_regex);
    const bool found_immortalis = std::regex_search(device_name, name_parts_immortalis, immortalis_regex);

    if (!found_mali && !found_immortalis)
    {
        ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Can't find valid Arm® Mali™ GPU device name: %s. "
                                                  "Target is set to default: %s",
                                                  device_name.c_str(), string_from_target(GPUTarget::FIFTHGEN).c_str());
        return GPUTarget::FIFTHGEN;
    }

    std::string full_name;
    std::string base_name;

    if (found_mali)
    {
        full_name = name_parts_mali.str(1);
        base_name = name_parts_mali.str(2);
    }
    else
    {
        full_name = name_parts_immortalis.str(1);
        base_name = name_parts_immortalis.str(2);
    }

    const auto gpu_target = get_gpu_target(full_name, base_name);
    ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Found Arm® Mali™ GPU device name %s. Target is set to %s.",
                                              full_name.c_str(), string_from_target(gpu_target).c_str());
    return gpu_target;
}

GPUTarget get_arch_from_target(GPUTarget target)
{
    return (target & GPUTarget::GPU_ARCH_MASK);
}
} // namespace arm_compute
