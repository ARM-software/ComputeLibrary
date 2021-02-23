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
#ifndef ARM_COMPUTE_SINGLETHREADSCHEDULER_H
#define ARM_COMPUTE_SINGLETHREADSCHEDULER_H

#include "arm_compute/runtime/IScheduler.h"

namespace arm_compute
{
/** Pool of threads to automatically split a kernel's execution among several threads. */
class SingleThreadScheduler final : public IScheduler
{
public:
    /** Constructor. */
    SingleThreadScheduler() = default;
    /** Sets the number of threads the scheduler will use to run the kernels.
     *
     * @param[in] num_threads This is ignored for this scheduler as the number of threads is always one.
     */
    void set_num_threads(unsigned int num_threads) override;
    /** Returns the number of threads that the SingleThreadScheduler has, which is always 1.
     *
     * @return Number of threads available in SingleThreadScheduler.
     */
    unsigned int num_threads() const override;
    /** Runs the kernel in the same thread as the caller synchronously.
     *
     * @param[in] kernel Kernel to execute.
     * @param[in] hints  Hints for the scheduler.
     */
    void schedule(ICPPKernel *kernel, const Hints &hints) override;
    /** Runs the kernel in the same thread as the caller synchronously.
     *
     * @param[in] kernel  Kernel to execute.
     * @param[in] hints   Hints for the scheduler.
     * @param[in] window  Window to use for kernel execution.
     * @param[in] tensors Vector containing the tensors to operate on.
     */
    void schedule_op(ICPPKernel *kernel, const Hints &hints, const Window &window, ITensorPack &tensors) override;

protected:
    /** Will run the workloads sequentially and in order.
     *
     * @param[in] workloads Workloads to run
     */
    void run_workloads(std::vector<Workload> &workloads) override;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_SINGLETHREADSCHEDULER_H */
