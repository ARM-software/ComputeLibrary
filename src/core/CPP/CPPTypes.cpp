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

#ifndef BARE_METAL
#include <sched.h>
#endif /* defined(BARE_METAL) */

using namespace arm_compute;

void CPUInfo::set_fp16(const bool fp16)
{
    _fp16 = fp16;
}

void CPUInfo::set_dotprod(const bool dotprod)
{
    _dotprod = dotprod;
}

void CPUInfo::set_cpu_model(unsigned int cpuid, CPUModel model)
{
    ARM_COMPUTE_ERROR_ON(cpuid >= _percpu.size());
    if(_percpu.size() > cpuid)
    {
        _percpu[cpuid] = model;
    }
}

bool CPUInfo::has_fp16() const
{
    return _fp16;
}

bool CPUInfo::has_dotprod() const
{
    return _dotprod;
}

CPUModel CPUInfo::get_cpu_model(unsigned int cpuid) const
{
    if(cpuid < _percpu.size())
    {
        return _percpu[cpuid];
    }
    return CPUModel::GENERIC;
}

unsigned int CPUInfo::get_L1_cache_size() const
{
    return _L1_cache_size;
}

void CPUInfo::set_L1_cache_size(unsigned int size)
{
    _L1_cache_size = size;
}

unsigned int CPUInfo::get_L2_cache_size() const
{
    return _L2_cache_size;
}

void CPUInfo::set_L2_cache_size(unsigned int size)
{
    _L2_cache_size = size;
}

void CPUInfo::set_cpu_num(unsigned int cpu_count)
{
    _percpu.resize(cpu_count);
}

CPUInfo::CPUInfo()
    : _percpu(1)
{
    // The core library knows nothing about the CPUs so we set only 1 CPU to be generic.
    // The runtime NESCheduler will initialise this vector with the correct CPU models.
    // See void detect_cpus_configuration(CPUInfo &cpuinfo) in CPPUtils.h
    _percpu[0] = CPUModel::GENERIC;
}

CPUModel CPUInfo::get_cpu_model() const
{
#if defined(BARE_METAL) || (!defined(__arm__) && !defined(__aarch64__))
    return get_cpu_model(0);
#else  /* defined(BARE_METAL) || (!defined(__arm__) && !defined(__aarch64__)) */
    return get_cpu_model(sched_getcpu());
#endif /* defined(BARE_METAL) || (!defined(__arm__) && !defined(__aarch64__)) */
}
