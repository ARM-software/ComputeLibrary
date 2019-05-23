/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/runtime/CPUUtils.h"

#include "arm_compute/core/CPP/CPPTypes.h"
#include "arm_compute/core/Error.h"
#include "support/ToolchainSupport.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <sched.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef BARE_METAL
/* C++ std::regex takes up a lot of space in the standalone builds */
#include <regex.h>
#include <thread>
#endif /* BARE_METAL */

#if !defined(BARE_METAL) && (defined(__arm__) || defined(__aarch64__))
#include <sys/auxv.h>

/* Get HWCAP bits from asm/hwcap.h */
#include <asm/hwcap.h>
#endif /* !BARE_METAL */

/* Make sure the bits we care about are defined, just in case asm/hwcap.h is
 * out of date (or for bare metal mode) */
#ifndef HWCAP_ASIMDHP
#define HWCAP_ASIMDHP (1 << 10) // NOLINT
#endif                          /* HWCAP_ASIMDHP */

#ifndef HWCAP_CPUID
#define HWCAP_CPUID (1 << 11) // NOLINT
#endif                        /* HWCAP_CPUID */

#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20) // NOLINT
#endif                          /* HWCAP_ASIMDDP */

namespace
{
using namespace arm_compute;

#if !defined(BARE_METAL) && (defined(__arm__) || defined(__aarch64__))

bool model_supports_dot(CPUModel model)
{
    switch(model)
    {
        case CPUModel::GENERIC_FP16_DOT:
        case CPUModel::A55r1:
            return true;
        default:
            return false;
    }
}

bool model_supports_fp16(CPUModel model)
{
    switch(model)
    {
        case CPUModel::GENERIC_FP16:
        case CPUModel::GENERIC_FP16_DOT:
        case CPUModel::A55r1:
            return true;
        default:
            return false;
    }
}

/* Convert an MIDR register value to a CPUModel enum value. */
CPUModel midr_to_model(const unsigned int midr)
{
    CPUModel model = CPUModel::GENERIC;

    // Unpack variant and CPU ID
    const int implementer = (midr >> 24) & 0xFF;
    const int variant     = (midr >> 20) & 0xF;
    const int cpunum      = (midr >> 4) & 0xFFF;

    if(implementer == 0x41) // Arm CPUs
    {
        // Only CPUs we have code paths for are detected.  All other CPUs can be safely classed as "GENERIC"
        switch(cpunum)
        {
            case 0xd03: // A53
            case 0xd04: // A35
                model = CPUModel::A53;
                break;
            case 0xd05: // A55
                if(variant != 0)
                {
                    model = CPUModel::A55r1;
                }
                else
                {
                    model = CPUModel::A55r0;
                }
                break;
            case 0xd0a: // A75
                if(variant != 0)
                {
                    model = CPUModel::GENERIC_FP16_DOT;
                }
                else
                {
                    model = CPUModel::GENERIC_FP16;
                }
                break;
            case 0xd0b: // A76
            case 0xd06:
            case 0xd0c:
            case 0xd0d:
                model = CPUModel::GENERIC_FP16_DOT;
                break;
            default:
                model = CPUModel::GENERIC;
                break;
        }
    }
    else if(implementer == 0x48)
    {
        // Only CPUs we have code paths for are detected.  All other CPUs can be safely classed as "GENERIC"
        switch(cpunum)
        {
            case 0xd40: // A76
                model = CPUModel::GENERIC_FP16_DOT;
                break;
            default:
                model = CPUModel::GENERIC;
                break;
        }
    }

    return model;
}

void populate_models_cpuid(std::vector<CPUModel> &cpusv)
{
    // If the CPUID capability is present, MIDR information is provided in /sys. Use that to populate the CPU model table.
    uint32_t i = 0;
    for(auto &c : cpusv)
    {
        std::stringstream str;
        str << "/sys/devices/system/cpu/cpu" << i++ << "/regs/identification/midr_el1";
        std::ifstream file;
        file.open(str.str(), std::ios::in);
        if(file.is_open())
        {
            std::string line;
            if(bool(getline(file, line)))
            {
                const uint32_t midr = support::cpp11::stoul(line, nullptr, support::cpp11::NumericBase::BASE_16);
                c                   = midr_to_model(midr & 0xffffffff);
            }
        }
    }
}

void populate_models_cpuinfo(std::vector<CPUModel> &cpusv)
{
    regex_t proc_regex;
    regex_t imp_regex;
    regex_t var_regex;
    regex_t part_regex;
    regex_t rev_regex;

    memset(&proc_regex, 0, sizeof(regex_t));
    memset(&imp_regex, 0, sizeof(regex_t));
    memset(&var_regex, 0, sizeof(regex_t));
    memset(&part_regex, 0, sizeof(regex_t));
    memset(&rev_regex, 0, sizeof(regex_t));

    int ret_status = 0;
    // If "long-form" cpuinfo is present, parse that to populate models.
    ret_status |= regcomp(&proc_regex, R"(^processor.*([[:digit:]]+)$)", REG_EXTENDED);
    ret_status |= regcomp(&imp_regex, R"(^CPU implementer.*0x(..)$)", REG_EXTENDED);
    ret_status |= regcomp(&var_regex, R"(^CPU variant.*0x(.)$)", REG_EXTENDED);
    ret_status |= regcomp(&part_regex, R"(^CPU part.*0x(...)$)", REG_EXTENDED);
    ret_status |= regcomp(&rev_regex, R"(^CPU revision.*([[:digit:]]+)$)", REG_EXTENDED);
    ARM_COMPUTE_UNUSED(ret_status);
    ARM_COMPUTE_ERROR_ON_MSG(ret_status != 0, "Regex compilation failed.");

    std::ifstream file;
    file.open("/proc/cpuinfo", std::ios::in);

    if(file.is_open())
    {
        std::string line;
        int         midr   = 0;
        int         curcpu = -1;

        while(bool(getline(file, line)))
        {
            std::array<regmatch_t, 2> match;
            ret_status = regexec(&proc_regex, line.c_str(), 2, match.data(), 0);
            if(ret_status == 0)
            {
                std::string id     = line.substr(match[1].rm_so, (match[1].rm_eo - match[1].rm_so));
                int         newcpu = support::cpp11::stoi(id, nullptr);

                if(curcpu >= 0 && midr == 0)
                {
                    // Matched a new CPU ID without any description of the previous one - looks like old format.
                    return;
                }

                if(curcpu >= 0)
                {
                    cpusv[curcpu] = midr_to_model(midr);
                }

                midr   = 0;
                curcpu = newcpu;

                continue;
            }

            ret_status = regexec(&imp_regex, line.c_str(), 2, match.data(), 0);
            if(ret_status == 0)
            {
                std::string subexp = line.substr(match[1].rm_so, (match[1].rm_eo - match[1].rm_so));
                int         impv   = support::cpp11::stoi(subexp, nullptr, support::cpp11::NumericBase::BASE_16);
                midr |= (impv << 24);

                continue;
            }

            ret_status = regexec(&var_regex, line.c_str(), 2, match.data(), 0);
            if(ret_status == 0)
            {
                std::string subexp = line.substr(match[1].rm_so, (match[1].rm_eo - match[1].rm_so));
                int         varv   = support::cpp11::stoi(subexp, nullptr, support::cpp11::NumericBase::BASE_16);
                midr |= (varv << 20);

                continue;
            }

            ret_status = regexec(&part_regex, line.c_str(), 2, match.data(), 0);
            if(ret_status == 0)
            {
                std::string subexp = line.substr(match[1].rm_so, (match[1].rm_eo - match[1].rm_so));
                int         partv  = support::cpp11::stoi(subexp, nullptr, support::cpp11::NumericBase::BASE_16);
                midr |= (partv << 4);

                continue;
            }

            ret_status = regexec(&rev_regex, line.c_str(), 2, match.data(), 0);
            if(ret_status == 0)
            {
                std::string subexp = line.substr(match[1].rm_so, (match[1].rm_eo - match[1].rm_so));
                int         regv   = support::cpp11::stoi(subexp, nullptr);
                midr |= (regv);
                midr |= (0xf << 16);

                continue;
            }
        }

        if(curcpu >= 0)
        {
            cpusv[curcpu] = midr_to_model(midr);
        }
    }

    // Free allocated memory
    regfree(&proc_regex);
    regfree(&imp_regex);
    regfree(&var_regex);
    regfree(&part_regex);
    regfree(&rev_regex);
}

int get_max_cpus()
{
    int           max_cpus = 1;
    std::ifstream CPUspresent;
    CPUspresent.open("/sys/devices/system/cpu/present", std::ios::in);
    bool success = false;

    if(CPUspresent.is_open())
    {
        std::string line;

        if(bool(getline(CPUspresent, line)))
        {
            /* The content of this file is a list of ranges or single values, e.g.
                 * 0-5, or 1-3,5,7 or similar.  As we are interested in the
                 * max valid ID, we just need to find the last valid
                 * delimiter ('-' or ',') and parse the integer immediately after that.
                 */
            auto startfrom = line.begin();

            for(auto i = line.begin(); i < line.end(); ++i)
            {
                if(*i == '-' || *i == ',')
                {
                    startfrom = i + 1;
                }
            }

            line.erase(line.begin(), startfrom);

            max_cpus = support::cpp11::stoi(line, nullptr) + 1;
            success  = true;
        }
    }

    // Return std::thread::hardware_concurrency() as a fallback.
    if(!success)
    {
        max_cpus = std::thread::hardware_concurrency();
    }
    return max_cpus;
}
#endif /* !defined(BARE_METAL) && (defined(__arm__) || defined(__aarch64__)) */

} // namespace

namespace arm_compute
{
void get_cpu_configuration(CPUInfo &cpuinfo)
{
#if !defined(BARE_METAL) && (defined(__arm__) || defined(__aarch64__))
    bool cpuid               = false;
    bool hwcaps_fp16_support = false;
    bool hwcaps_dot_support  = false;

    const uint32_t hwcaps = getauxval(AT_HWCAP);

    if((hwcaps & HWCAP_CPUID) != 0)
    {
        cpuid = true;
    }

    if((hwcaps & HWCAP_ASIMDHP) != 0)
    {
        hwcaps_fp16_support = true;
    }

#if defined(__aarch64__)
    if((hwcaps & HWCAP_ASIMDDP) != 0)
    {
        hwcaps_dot_support = true;
    }
#endif /* defined(__aarch64__) */

    const unsigned int max_cpus = get_max_cpus();
    cpuinfo.set_cpu_num(max_cpus);
    std::vector<CPUModel> percpu(max_cpus, CPUModel::GENERIC);
    if(cpuid)
    {
        populate_models_cpuid(percpu);
    }
    else
    {
        populate_models_cpuinfo(percpu);
    }
    int j(0);
    // Update dot product and FP16 support if one of the CPUs support these features
    // We assume that the system does not have mixed architectures
    bool one_supports_dot  = false;
    bool one_supports_fp16 = false;
    for(const auto &v : percpu)
    {
        one_supports_dot  = one_supports_dot || model_supports_dot(v);
        one_supports_fp16 = one_supports_fp16 || model_supports_fp16(v);
        cpuinfo.set_cpu_model(j++, v);
    }
    cpuinfo.set_dotprod(one_supports_dot || hwcaps_dot_support);
    cpuinfo.set_fp16(one_supports_fp16 || hwcaps_fp16_support);
#else  /* !defined(BARE_METAL) && (defined(__arm__) || defined(__aarch64__)) */
    ARM_COMPUTE_UNUSED(cpuinfo);
#endif /* !defined(BARE_METAL) && (defined(__arm__) || defined(__aarch64__)) */
}

unsigned int get_threads_hint()
{
    unsigned int num_threads_hint = 1;

#ifndef BARE_METAL
    std::map<std::string, unsigned int> cpu_part_occurrence_map;

    // CPU part regex
    regex_t cpu_part_rgx;
    memset(&cpu_part_rgx, 0, sizeof(regex_t));
    int ret_status = regcomp(&cpu_part_rgx, R"(.*CPU part.+/?\:[[:space:]]+([[:alnum:]]+).*)", REG_EXTENDED);
    ARM_COMPUTE_UNUSED(ret_status);
    ARM_COMPUTE_ERROR_ON_MSG(ret_status != 0, "Regex compilation failed.");

    // Read cpuinfo and get occurrence of each core
    std::ifstream cpuinfo;
    cpuinfo.open("/proc/cpuinfo", std::ios::in);
    if(cpuinfo.is_open())
    {
        std::string line;
        while(bool(getline(cpuinfo, line)))
        {
            std::array<regmatch_t, 2> match;
            ret_status = regexec(&cpu_part_rgx, line.c_str(), 2, match.data(), 0);
            if(ret_status == 0)
            {
                std::string cpu_part = line.substr(match[1].rm_so, (match[1].rm_eo - match[1].rm_so));
                if(cpu_part_occurrence_map.find(cpu_part) != cpu_part_occurrence_map.end())
                {
                    cpu_part_occurrence_map[cpu_part]++;
                }
                else
                {
                    cpu_part_occurrence_map[cpu_part] = 1;
                }
            }
        }
    }
    regfree(&cpu_part_rgx);

    // Get min number of threads
    auto min_common_cores = std::min_element(cpu_part_occurrence_map.begin(), cpu_part_occurrence_map.end(),
                                             [](const std::pair<std::string, unsigned int> &p1, const std::pair<std::string, unsigned int> &p2)
    {
        return p1.second < p2.second;
    });

    // Set thread hint
    num_threads_hint = cpu_part_occurrence_map.empty() ? std::thread::hardware_concurrency() : min_common_cores->second;
#endif /* BARE_METAL */

    return num_threads_hint;
}

} // namespace arm_compute
