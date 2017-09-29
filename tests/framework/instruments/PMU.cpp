/*
 * Copyright (c) 2017 ARM Limited.
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
#include "PMU.h"

#include <asm/unistd.h>
#include <cstring>
#include <stdexcept>
#include <sys/ioctl.h>

namespace arm_compute
{
namespace test
{
namespace framework
{
PMU::PMU()
    : _perf_config()
{
    _perf_config.type = PERF_TYPE_HARDWARE;
    _perf_config.size = sizeof(perf_event_attr);

    // Start disabled
    _perf_config.disabled = 1;
    // The inherit bit specifies that this counter should count events of child
    // tasks as well as the task specified
    _perf_config.inherit = 1;
    // Enables saving of event counts on context switch for inherited tasks
    _perf_config.inherit_stat = 1;
}

PMU::PMU(uint64_t config)
    : PMU()
{
    open(config);
}

PMU::~PMU()
{
    close();
}

void PMU::open(uint64_t config)
{
    _perf_config.config = config;
    open(_perf_config);
}

void PMU::open(const perf_event_attr &perf_config)
{
    // Measure this process/thread (+ children) on any CPU
    _fd = syscall(__NR_perf_event_open, &perf_config, 0, -1, -1, 0);

    ARM_COMPUTE_ERROR_ON_MSG(_fd < 0, "perf_event_open failed");

    const int result = ioctl(_fd, PERF_EVENT_IOC_ENABLE, 0);
    if(result == -1)
    {
        ARM_COMPUTE_ERROR("Failed to enable PMU counter: %d", errno);
    }
}

void PMU::close()
{
    if(_fd != -1)
    {
        ::close(_fd);
        _fd = -1;
    }
}

void PMU::reset()
{
    const int result = ioctl(_fd, PERF_EVENT_IOC_RESET, 0);
    if(result == -1)
    {
        ARM_COMPUTE_ERROR("Failed to reset PMU counter: %d", errno);
    }
}
} // namespace framework
} // namespace test
} // namespace arm_compute
