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
#include <iterator>
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
        std::string line;
        while(bool(getline(meminfo_f, line)))
        {
            std::istringstream       iss(line);
            std::vector<std::string> tokens((std::istream_iterator<std::string>(iss)),
                                            std::istream_iterator<std::string>());
            if(tokens[0] == "MemTotal:")
            {
                total = arm_compute::support::cpp11::stoul(tokens[1], nullptr);
            }
            else if(tokens[0] == "MemFree:")
            {
                memfree = arm_compute::support::cpp11::stoul(tokens[1], nullptr);
            }
            else if(tokens[0] == "Buffers:")
            {
                buffer = arm_compute::support::cpp11::stoul(tokens[1], nullptr);
            }
            else if(tokens[0] == "Cached:")
            {
                memcache = arm_compute::support::cpp11::stoul(tokens[1], nullptr);
            }
        }
        free = memfree + (buffer + memcache);
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
