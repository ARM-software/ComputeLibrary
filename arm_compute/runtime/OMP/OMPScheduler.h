/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_OMPSCHEDULER_H
#define ARM_COMPUTE_OMPSCHEDULER_H

#include "arm_compute/runtime/IScheduler.h"

namespace arm_compute
{
/** Pool of threads to automatically split a kernel's execution among several threads. */
class OMPScheduler final : public IScheduler
{
public:
    /** Constructor. */
    OMPScheduler();
    /** Sets the number of threads the scheduler will use to run the kernels.
     *
     * @param[in] num_threads If set to 0, then the number returned by omp_get_max_threads() will be used, otherwise the number of threads specified.
     */
    void set_num_threads(unsigned int num_threads) override;
    /** Returns the number of threads that the OMPScheduler has in its pool.
     *
     * @return Number of threads available in OMPScheduler.
     */
    unsigned int num_threads() const override;
    /** Multithread the execution of the passed kernel if possible.
     *
     * The kernel will run on a single thread if any of these conditions is true:
     * - ICPPKernel::is_parallelisable() returns false
     * - The scheduler has been initialized with only one thread.
     *
     * @param[in] kernel Kernel to execute.
     * @param[in] hints  Hints for the scheduler.
     */
    void schedule(ICPPKernel *kernel, const Hints &hints) override;

    /** Multithread the execution of the passed kernel if possible.
     *
     * The kernel will run on a single thread if any of these conditions is true:
     * - ICPPKernel::is_parallelisable() returns false
     * - The scheduler has been initialized with only one thread.
     *
     * @param[in] kernel  Kernel to execute.
     * @param[in] hints   Hints for the scheduler.
     * @param[in] window  Window to use for kernel execution.
     * @param[in] tensors Vector containing the tensors to operate on.
     */
    void schedule_op(ICPPKernel *kernel, const Hints &hints, const Window &window, ITensorPack &tensors) override;

protected:
    /** Execute all the passed workloads
     *
     * @note there is no guarantee regarding the order in which the workloads will be executed or whether or not they will be executed in parallel.
     *
     * @param[in] workloads Array of workloads to run
     */
    void run_workloads(std::vector<Workload> &workloads) override;

private:
    unsigned int _num_threads;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_OMPSCHEDULER_H */
