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
#include "arm_compute/core/CPP/CPPTypes.h"
#include "arm_compute/core/Error.h"
#include "support/ToolchainSupport.h"

#ifndef BARE_METAL
#include <fstream>
#include <regex>
#include <sstream>
#endif // ifndef BARE_METAL

namespace
{
void parse_mem_info(size_t &total, size_t &free, size_t &buffer)
{
    free   = 0;
    total  = 0;
    buffer = 0;
#ifndef BARE_METAL
    size_t        memcache = 0;
    size_t        memfree  = 0;
    std::ifstream meminfo_f;
    meminfo_f.open("/proc/meminfo", std::ios::in);
    if(meminfo_f.is_open())
    {
        std::stringstream str_stream;
        str_stream << meminfo_f.rdbuf();
        const std::string str = str_stream.str();
        try
        {
            std::smatch match;
            if(std::regex_search(str, match, std::regex("MemTotal: (.*)kB")) && match.size() > 1)
            {
                const std::string result = match.str(1);
                total                    = arm_compute::support::cpp11::stoul(result, nullptr);
            }
            if(std::regex_search(str, match, std::regex("MemFree: (.*)kB")) && match.size() > 1)
            {
                const std::string result = match.str(1);
                memfree                  = arm_compute::support::cpp11::stoul(result, nullptr);
            }
            if(std::regex_search(str, match, std::regex("Buffers: (.*)kB")) && match.size() > 1)
            {
                const std::string result = match.str(1);
                buffer                   = arm_compute::support::cpp11::stoul(result, nullptr);
            }
            if(std::regex_search(str, match, std::regex("Cached: (.*)kB")) && match.size() > 1)
            {
                const std::string result = match.str(1);
                memcache                 = arm_compute::support::cpp11::stoul(result, nullptr);
            }
            free = memfree + (buffer + memcache);
        }
        catch(std::regex_error &e)
        {
            // failed parsing /proc/meminfo
            // return 0s on all fields
        }
    }
#endif // ifndef BARE_METAL
}

} // namespace

namespace arm_compute
{
void MEMInfo::set_policy(MemoryPolicy policy)
{
    _policy = policy;
}

MemoryPolicy MEMInfo::get_policy()
{
    return _policy;
}
MemoryPolicy MEMInfo::_policy = { MemoryPolicy::NORMAL };

MEMInfo::MEMInfo()
    : _total(0), _free(0), _buffer(0)
{
    parse_mem_info(_total, _free, _buffer);
}

size_t MEMInfo::get_total_in_kb() const
{
    return _total;
}

} // namespace arm_compute
