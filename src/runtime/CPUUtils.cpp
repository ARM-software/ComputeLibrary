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
struct PerCPUData
{
    CPUModel     model     = CPUModel::GENERIC;
    unsigned int midr      = 0;
    bool         model_set = false;
};

/* Convert an MIDR register value to a CPUModel enum value. */
CPUModel midr_to_model(const unsigned int midr)
{
    CPUModel model;

    // Unpack variant and CPU ID
    const int variant = (midr >> 20) & 0xF;
    const int cpunum  = (midr >> 4) & 0xFFF;

    // Only CPUs we have code paths for are detected.  All other CPUs can be safely classed as "GENERIC"
    switch(cpunum)
    {
        case 0xd03:
            model = CPUModel::A53;
            break;

        case 0xd05:
            if(variant != 0)
            {
                model = CPUModel::A55r1;
            }
            else
            {
                model = CPUModel::A55r0;
            }
            break;

        default:
            model = CPUModel::GENERIC;
            break;
    }

    return model;
}

void populate_models_cpuid(std::vector<PerCPUData> &cpusv)
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
                const unsigned long midr = support::cpp11::stoul(line, nullptr, 16);
                c.midr                   = (midr & 0xffffffff);
                c.model                  = midr_to_model(c.midr);
                c.model_set              = true;
            }
        }
    }
}

void populate_models_cpuinfo(std::vector<PerCPUData> &cpusv)
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
                int         newcpu = support::cpp11::stoi(id, nullptr, 0);

                if(curcpu >= 0 && midr == 0)
                {
                    // Matched a new CPU ID without any description of the previous one - looks like old format.
                    return;
                }

                if(curcpu >= 0)
                {
                    cpusv[curcpu].midr      = midr;
                    cpusv[curcpu].model     = midr_to_model(midr);
                    cpusv[curcpu].model_set = true;
                }

                midr   = 0;
                curcpu = newcpu;

                continue;
            }

            if(std::regex_match(line, match, imp_regex))
            {
                int impv = support::cpp11::stoi(match[1], nullptr, 16);
                midr |= (impv << 24);
                continue;
            }

            if(std::regex_match(line, match, var_regex))
            {
                int varv = support::cpp11::stoi(match[1], nullptr, 16);
                midr |= (varv << 16);
                continue;
            }

            if(std::regex_match(line, match, part_regex))
            {
                int partv = support::cpp11::stoi(match[1], nullptr, 16);
                midr |= (partv << 4);
                continue;
            }

            if(std::regex_match(line, match, rev_regex))
            {
                int regv = support::cpp11::stoi(match[1], nullptr, 10);
                midr |= (regv);
                midr |= (0xf << 16);
                continue;
            }
        }

        if(curcpu >= 0)
        {
            cpusv[curcpu].midr      = midr;
            cpusv[curcpu].model     = midr_to_model(midr);
            cpusv[curcpu].model_set = true;
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

            max_cpus = support::cpp11::stoi(line, nullptr, 0) + 1;
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
    bool cpuid        = false;
    bool fp16_support = false;
    bool dot_support  = false;

    const uint32_t hwcaps = getauxval(AT_HWCAP);

    if((hwcaps & HWCAP_CPUID) != 0)
    {
        cpuid = true;
    }

    if((hwcaps & HWCAP_ASIMDHP) != 0)
    {
        fp16_support = true;
    }

    if((hwcaps & HWCAP_ASIMDDP) != 0)
    {
        dot_support = true;
    }

#ifdef __aarch64__
    /* Pre-4.15 kernels don't have the ASIMDDP bit.
     *
     * Although the CPUID bit allows us to read the feature register
     * directly, the kernel quite sensibly masks this to only show
     * features known by it to be safe to show to userspace.  As a
     * result, pre-4.15 kernels won't show the relevant bit in the
     * feature registers either.
     *
     * So for now, use a whitelist of CPUs known to support the feature.
     */
    if(!dot_support && cpuid)
    {
        /* List of CPUs with dot product support:         A55r1       A75r1       A75r2  */
        const unsigned int dotprod_whitelist_masks[]  = { 0xfff0fff0, 0xfff0fff0, 0xfff0fff0, 0 };
        const unsigned int dotprod_whitelist_values[] = { 0x4110d050, 0x4110d0a0, 0x4120d0a0, 0 };

        unsigned long cpuid;

        __asm __volatile(
            "mrs %0, midr_el1\n"
            : "=r"(cpuid)
            :
            : );

        for(int i = 0; dotprod_whitelist_values[i] != 0; i++)
        {
            if((cpuid & dotprod_whitelist_masks[i]) == dotprod_whitelist_values[i])
            {
                dot_support = true;
                break;
            }
        }
    }
#endif /* __aarch64__ */
    const unsigned int max_cpus = get_max_cpus();
    cpuinfo.set_cpu_num(max_cpus);
    cpuinfo.set_fp16(fp16_support);
    cpuinfo.set_dotprod(dot_support);
    std::vector<PerCPUData> percpu(max_cpus);
    if(cpuid)
    {
        populate_models_cpuid(percpu);
    }
    else
    {
        populate_models_cpuinfo(percpu);
    }
    int j(0);
    for(const auto &v : percpu)
    {
        cpuinfo.set_cpu_model(j++, v.model);
    }
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
