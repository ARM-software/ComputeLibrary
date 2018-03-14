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
#ifndef __ARM_COMPUTE_RUNTIME_CPU_UTILS_H__
#define __ARM_COMPUTE_RUNTIME_CPU_UTILS_H__

namespace arm_compute
{
class CPUInfo;
/** This function will try to detect the CPU configuration on the system and will fill
 *  the cpuinfo object accordingly to reflect this.
 *
 * @param[out] cpuinfo @ref CPUInfo to be used to hold the system's cpu configuration.
 */
void get_cpu_configuration(CPUInfo &cpuinfo);
/** Some systems have both big and small cores, this fuction computes the minimum number of cores
 *  that are exactly the same on the system. To maximize performance the library attempts to process
 *  workloads concurrently using as many threads as big cores are available on the system.
 *
 * @return The minumum number of common cores.
 */
unsigned int get_threads_hint();
}
#endif /* __ARM_COMPUTE_RUNTIME_CPU_UTILS_H__ */
