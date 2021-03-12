/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CPPSCHEDULER_H
#define ARM_COMPUTE_CPPSCHEDULER_H

#include "arm_compute/core/experimental/Types.h"
#include "arm_compute/runtime/IScheduler.h"

#include <memory>

namespace arm_compute
{
/** C++11 implementation of a pool of threads to automatically split a kernel's execution among several threads.
 *
 * It has 2 scheduling modes: Linear or Fanout (please refer to the implementation for details)
 * The mode is selected automatically based on the runtime environment. However it can be forced via an environment
 * variable ARM_COMPUTE_CPP_SCHEDULER_MODE. e.g.:
 * ARM_COMPUTE_CPP_SCHEDULER_MODE=linear      # Force select the linear scheduling mode
 * ARM_COMPUTE_CPP_SCHEDULER_MODE=fanout      # Force select the fanout scheduling mode
*/
class CPPScheduler final : public IScheduler
{
public:
    /** Constructor: create a pool of threads. */
    CPPScheduler();
    /** Default destructor */
    ~CPPScheduler();

    /** Access the scheduler singleton
     *
     * @note this method has been deprecated and will be remover in future releases
     * @return The scheduler
     */
    static CPPScheduler &get();

    // Inherited functions overridden
    void set_num_threads(unsigned int num_threads) override;
    void set_num_threads_with_affinity(unsigned int num_threads, BindFunc func) override;
    unsigned int num_threads() const override;
    void schedule(ICPPKernel *kernel, const Hints &hints) override;
    void schedule_op(ICPPKernel *kernel, const Hints &hints, const Window &window, ITensorPack &tensors) override;

protected:
    /** Will run the workloads in parallel using num_threads
     *
     * @param[in] workloads Workloads to run
     */
    void run_workloads(std::vector<Workload> &workloads) override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPPSCHEDULER_H */
