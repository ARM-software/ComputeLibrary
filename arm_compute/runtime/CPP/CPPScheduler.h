/*
 * Copyright (c) 2016-2020 ARM Limited.
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
/** C++11 implementation of a pool of threads to automatically split a kernel's execution among several threads. */
class CPPScheduler final : public IScheduler
{
public:
    /** Constructor: create a pool of threads. */
    CPPScheduler();
    /** Default destructor */
    ~CPPScheduler();
    /** Sets the number of threads the scheduler will use to run the kernels.
     *
     * @param[in] num_threads If set to 0, then the maximum number of threads supported by C++11 will be used, otherwise the number of threads specified.
     */
    void set_num_threads(unsigned int num_threads) override;
    /** Returns the number of threads that the CPPScheduler has in his pool.
     *
     * @return Number of threads available in CPPScheduler.
     */
    unsigned int num_threads() const override;

    /** Access the scheduler singleton
     *
     * @note this method has been deprecated and will be remover in the upcoming releases
     * @return The scheduler
     */
    static CPPScheduler &get();
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
     * @param[in] inputs  Vector that contains the input tensors.
     * @param[in] outputs Vector that contains the output tensors.
     */
    void schedule_op(ICPPKernel *kernel, const Hints &hints, std::vector<InputOperatorTensors *> &inputs, std::vector<OutputOperatorTensors *> &outputs) override;

protected:
    /** Will run the workloads in parallel using num_threads
     *
     * @param[in] workloads Workloads to run
     */
    void run_workloads(std::vector<Workload> &workloads) override;

private:
    void schedule_common(ICPPKernel *kernel, const Hints &hints, std::vector<InputOperatorTensors *> &inputs, std::vector<OutputOperatorTensors *> &outputs);
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPPSCHEDULER_H */
