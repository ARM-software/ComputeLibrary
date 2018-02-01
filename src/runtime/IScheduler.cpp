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
#include "arm_compute/runtime/IScheduler.h"

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

namespace
{
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

unsigned int get_cpu_impl()
{
#ifndef BARE_METAL
    int fd = open("/proc/cpuinfo", 0); // NOLINT
    std::array<char, 3000> buff{ {} };
    char *pos     = nullptr;
    char *end     = nullptr;
    bool  foundid = false;

    int cpu = sched_getcpu();

    if(fd == -1)
    {
        return 0;
    }

    int charsread = read(fd, buff.data(), 3000);
    pos           = buff.data();
    end           = buff.data() + charsread;

    close(fd);

    /* So, to date I've encountered two formats for /proc/cpuinfo.
     *
     * One of them just lists processor : n  for each processor (with no
     * other info), then at the end lists part information for the current
     * CPU.
     *
     * The other has an entire clause (including part number info) for each
     * CPU in the system, with "processor : n" headers.
     *
     * We can cope with either of these formats by waiting to see
     * "processor: n" (where n = our CPU ID), and then looking for the next
     * "CPU part" field.
     */
    while(pos < end)
    {
        if(foundid && strncmp(pos, "CPU part", 8) == 0)
        {
            /* Found part number */
            pos += 11;

            for(char *ch = pos; ch < end; ch++)
            {
                if(*ch == '\n')
                {
                    *ch = '\0';
                    break;
                }
            }

            return strtoul(pos, nullptr, 0);
        }

        if(strncmp(pos, "processor", 9) == 0)
        {
            /* Found processor ID, see if it's ours. */
            pos += 11;

            for(char *ch = pos; ch < end; ch++)
            {
                if(*ch == '\n')
                {
                    *ch = '\0';
                    break;
                }
            }

            int num = strtol(pos, nullptr, 0);

            if(num == cpu)
            {
                foundid = true;
            }
        }

        while(pos < end)
        {
            char ch = *pos++;
            if(ch == '\n' || ch == '\0')
            {
                break;
            }
        }
    }
#endif /* BARE_METAL */

    return 0;
}
} // namespace

namespace arm_compute
{
IScheduler::IScheduler()
{
    // Work out the best possible number of execution threads
    _num_threads_hint = get_threads_hint();

    // Work out the CPU implementation
    switch(get_cpu_impl())
    {
        case 0xd0f:
            _info.CPU = CPUTarget::A55_DOT;
            break;
        case 0xd03:
            _info.CPU = CPUTarget::A53;
            break;
        default:
#ifdef __arm__
            _info.CPU = CPUTarget::ARMV7;
#elif __aarch64__
            _info.CPU = CPUTarget::ARMV8;
#else  /* __arm__ || __aarch64__ */
            _info.CPU = CPUTarget::INTRINSICS;
#endif /* __arm__ || __aarch64__ */
            break;
    }

    _info.L1_size = 31000;
    _info.L2_size = 500000;
}

void IScheduler::set_target(CPUTarget target)
{
    _info.CPU = target;
}

CPUInfo IScheduler::cpu_info() const
{
    return _info;
}

unsigned int IScheduler::num_threads_hint() const
{
    return _num_threads_hint;
}
} // namespace arm_compute
