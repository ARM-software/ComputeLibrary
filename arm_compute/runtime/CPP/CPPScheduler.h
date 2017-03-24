/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_CPPSCHEDULER_H__
#define __ARM_COMPUTE_CPPSCHEDULER_H__

#include <cstddef>
#include <memory>

namespace arm_compute
{
class ICPPKernel;
class Thread;

/** Pool of threads to automatically split a kernel's execution among several threads. */
class CPPScheduler
{
private:
    /** Constructor: create a pool of threads. */
    CPPScheduler();

public:
    /** Force the re-creation of the pool of threads to use the specified number of threads.
     *
     * @param[in] num_threads If set to 0, then std::thread::hardware_concurrency() threads will be used, otherwise the number of threads specified.
     */
    void force_number_of_threads(int num_threads);
    /** Returns the number of threads that the CPPScheduler has in his pool.
     *
     * @return Number of threads available in CPPScheduler.
     */
    int num_threads() const
    {
        return _num_threads;
    }
    /** Access the scheduler singleton
     *
     * @return The scheduler
     */
    static CPPScheduler &get();
    /** Multithread the execution of the passed kernel if possible.
     *
     * The kernel will run on a single thread if any of these conditions is true:
     * - ICPPKernel::is_parallelisable() returns false
     * - The scheduler has been initialized with only one thread.
     *
     * @param[in] kernel          Kernel to execute.
     * @param[in] split_dimension Dimension along which to split the kernel's execution window (By default 1/Y)
     */
    void multithread(ICPPKernel *kernel, size_t split_dimension = 1);

private:
    int _num_threads;
    std::unique_ptr<Thread[], void (*)(Thread *)> _threads;
};
}
#endif /* __ARM_COMPUTE_CPPSCHEDULER_H__ */
