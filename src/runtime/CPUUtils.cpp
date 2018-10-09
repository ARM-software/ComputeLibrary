/*
 * Copyright (c) 2018 ARM Limited.
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
#include <regex>
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
#define HWCAP_ASIMDHP (1 << 10)
#endif /* HWCAP_ASIMDHP */

#ifndef HWCAP_CPUID
#define HWCAP_CPUID (1 << 11)
#endif /* HWCAP_CPUID */

#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif /* HWCAP_ASIMDDP */

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
            case 0xd06: // Helios FIXME: Unreleased CPU, remove before release
            case 0xd0c: // Ares   FIXME: Unreleased CPU, remove before release
            case 0xd0d: // Deimos FIXME: Unreleased CPU, remove before release
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
                const unsigned long midr = support::cpp11::stoul(line, nullptr, support::cpp11::NumericBase::BASE_16);
                c                        = midr_to_model(midr & 0xffffffff);
            }
        }
    }
}

void populate_models_cpuinfo(std::vector<CPUModel> &cpusv)
{
    // If "long-form" cpuinfo is present, parse that to populate models.
    std::regex proc_regex("^processor.*(\\d+)$");
    std::regex imp_regex("^CPU implementer.*0x(..)$");
    std::regex var_regex("^CPU variant.*0x(.)$");
    std::regex part_regex("^CPU part.*0x(...)$");
    std::regex rev_regex("^CPU revision.*(\\d+)$");

    std::ifstream file;
    file.open("/proc/cpuinfo", std::ios::in);

    if(file.is_open())
    {
        std::string line;
        int         midr   = 0;
        int         curcpu = -1;

        while(bool(getline(file, line)))
        {
            std::smatch match;

            if(std::regex_match(line, match, proc_regex))
            {
                std::string id     = match[1];
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

            if(std::regex_match(line, match, imp_regex))
            {
                int impv = support::cpp11::stoi(match[1], nullptr, support::cpp11::NumericBase::BASE_16);
                midr |= (impv << 24);
                continue;
            }

            if(std::regex_match(line, match, var_regex))
            {
                int varv = support::cpp11::stoi(match[1], nullptr, support::cpp11::NumericBase::BASE_16);
                midr |= (varv << 20);
                continue;
            }

            if(std::regex_match(line, match, part_regex))
            {
                int partv = support::cpp11::stoi(match[1], nullptr, support::cpp11::NumericBase::BASE_16);
                midr |= (partv << 4);
                continue;
            }

            if(std::regex_match(line, match, rev_regex))
            {
                int regv = support::cpp11::stoi(match[1], nullptr);
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
}

int get_max_cpus()
{
    int max_cpus = 1;
#if !defined(BARE_METAL) && (defined(__arm__) || defined(__aarch64__))
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
#endif /* BARE_METAL */
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
    // Update dot product and FP16 support if all CPUs support these features:
    bool all_support_dot  = true;
    bool all_support_fp16 = true;
    for(const auto &v : percpu)
    {
        all_support_dot &= model_supports_dot(v);
        all_support_fp16 &= model_supports_fp16(v);
        cpuinfo.set_cpu_model(j++, v);
    }
    cpuinfo.set_dotprod(all_support_dot || hwcaps_dot_support);
    cpuinfo.set_fp16(all_support_fp16 || hwcaps_fp16_support);
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
    std::regex  cpu_part_rgx(R"(.*CPU part.+?(?=:).+?(?=\w+)(\w+).*)");
    std::smatch cpu_part_match;

    // Read cpuinfo and get occurrence of each core
    std::ifstream cpuinfo;
    cpuinfo.open("/proc/cpuinfo", std::ios::in);
    if(cpuinfo.is_open())
    {
        std::string line;
        while(bool(getline(cpuinfo, line)))
        {
            if(std::regex_search(line.cbegin(), line.cend(), cpu_part_match, cpu_part_rgx))
            {
                std::string cpu_part = cpu_part_match[1];
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
