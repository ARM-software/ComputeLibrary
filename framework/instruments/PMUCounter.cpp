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
#include "PMUCounter.h"

#define _GNU_SOURCE 1
#include <asm/unistd.h>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <linux/hw_breakpoint.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <unistd.h>
#undef _GNU_SOURCE

#include <stdexcept>

namespace arm_compute
{
namespace test
{
namespace framework
{
CycleCounter::CycleCounter()
{
    const pid_t pid = getpid();

    struct perf_event_attr perf_config; //NOLINT
    memset(&perf_config, 0, sizeof(struct perf_event_attr));

    perf_config.config = PERF_COUNT_HW_CPU_CYCLES;
    perf_config.size   = sizeof(struct perf_event_attr);
    perf_config.type   = PERF_TYPE_HARDWARE;
    // The inherit bit specifies that this counter should count events of child
    // tasks as well as the task specified
    perf_config.inherit = 1;
    // Enables saving of event counts on context switch for inherited tasks
    perf_config.inherit_stat = 1;

    _fd = syscall(__NR_perf_event_open, &perf_config, pid, -1, -1, 0);

    if(_fd < 0)
    {
        throw std::runtime_error("perf_event_open for cycles failed");
    }
}

std::string CycleCounter::id() const
{
    return "Cycle Counter";
}

void CycleCounter::start()
{
    ioctl(_fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(_fd, PERF_EVENT_IOC_ENABLE, 0);
}

void CycleCounter::stop()
{
    ioctl(_fd, PERF_EVENT_IOC_DISABLE, 0);
    read(_fd, &_cycles, sizeof(_cycles));
}

Instrument::Measurement CycleCounter::measurement() const
{
    return Measurement(_cycles, "cycles");
}

InstructionCounter::InstructionCounter()
{
    const pid_t pid = getpid();

    struct perf_event_attr perf_config; //NOLINT
    memset(&perf_config, 0, sizeof(struct perf_event_attr));

    perf_config.config = PERF_COUNT_HW_INSTRUCTIONS;
    perf_config.size   = sizeof(struct perf_event_attr);
    perf_config.type   = PERF_TYPE_HARDWARE;
    // The inherit bit specifies that this counter should count events of child
    // tasks as well as the task specified
    perf_config.inherit = 1;
    // Enables saving of event counts on context switch for inherited tasks
    perf_config.inherit_stat = 1;

    _fd = syscall(__NR_perf_event_open, &perf_config, pid, -1, -1, 0);

    if(_fd < 0)
    {
        throw std::runtime_error("perf_event_open for instructions failed");
    }
}

std::string InstructionCounter::id() const
{
    return "Instruction Counter";
}

void InstructionCounter::start()
{
    ioctl(_fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(_fd, PERF_EVENT_IOC_ENABLE, 0);
}

void InstructionCounter::stop()
{
    ioctl(_fd, PERF_EVENT_IOC_DISABLE, 0);
    read(_fd, &_instructions, sizeof(_instructions));
}

Instrument::Measurement InstructionCounter::measurement() const
{
    return Measurement(_instructions, "instructions");
}
} // namespace framework
} // namespace test
} // namespace arm_compute
