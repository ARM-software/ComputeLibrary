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
#ifndef __ARM_COMPUTE_ISCHEDULER_H__
#define __ARM_COMPUTE_ISCHEDULER_H__

#include "arm_compute/core/CPP/CPPTypes.h"

namespace arm_compute
{
class ICPPKernel;

/** Scheduler interface to run kernels */
class IScheduler
{
public:
    /** Default constructor. */
    IScheduler();

    /** Destructor. */
    virtual ~IScheduler() = default;

    /** Sets the number of threads the scheduler will use to run the kernels.
     *
     * @param[in] num_threads If set to 0, then one thread per CPU core available on the system will be used, otherwise the number of threads specified.
     */
    virtual void set_num_threads(unsigned int num_threads) = 0;

    /** Returns the number of threads that the SingleThreadScheduler has in his pool.
     *
     * @return Number of threads available in SingleThreadScheduler.
     */
    virtual unsigned int num_threads() const = 0;

    /** Runs the kernel in the same thread as the caller synchronously.
     *
     * @param[in] kernel          Kernel to execute.
     * @param[in] split_dimension Dimension along which to split the kernel's execution window.
     */
    virtual void schedule(ICPPKernel *kernel, unsigned int split_dimension) = 0;

    /** Sets the target CPU architecture.
     *
     * @param[in] target Target CPU.
     */
    void set_target(CPUTarget target);

    /** Get CPU info.
     *
     * @return CPU info.
     */
    CPUInfo cpu_info() const;

protected:
    CPUInfo _info{};
};
}
#endif /* __ARM_COMPUTE_ISCHEDULER_H__ */
