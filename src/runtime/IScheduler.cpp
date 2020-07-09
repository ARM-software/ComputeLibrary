/*
 * Copyright (c) 2016-2020 Arm Limited.
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

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/CPUUtils.h"

namespace arm_compute
{
IScheduler::IScheduler()
    : _cpu_info()
{
    get_cpu_configuration(_cpu_info);
    // Work out the best possible number of execution threads
    _num_threads_hint = get_threads_hint();
}

CPUInfo &IScheduler::cpu_info()
{
    return _cpu_info;
}

void IScheduler::set_num_threads_with_affinity(unsigned int num_threads, BindFunc func)
{
    ARM_COMPUTE_UNUSED(num_threads, func);
    ARM_COMPUTE_ERROR("Feature for affinity setting is not implemented");
}

unsigned int IScheduler::num_threads_hint() const
{
    return _num_threads_hint;
}
void IScheduler::run_tagged_workloads(std::vector<Workload> &workloads, const char *tag)
{
    ARM_COMPUTE_UNUSED(tag);
    run_workloads(workloads);
}

} // namespace arm_compute
