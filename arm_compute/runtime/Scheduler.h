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
#ifndef __ARM_COMPUTE_SCHEDULER_H__
#define __ARM_COMPUTE_SCHEDULER_H__

#include "arm_compute/runtime/IScheduler.h"
#include <memory>

namespace arm_compute
{
/** Configurable scheduler which supports multiple multithreading APIs and choosing between different schedulers at runtime. */
class Scheduler
{
public:
    enum class Type
    {
        ST,    // Single thread.
        CPP,   // C++11 threads.
        OMP,   // OpenMP.
        CUSTOM // Provided by the user.
    };
    /** Sets the user defined scheduler and makes it the active scheduler.
     *
     * @param[in] scheduler A shared pointer to a custom scheduler implemented by the user.
     */
    static void set(std::shared_ptr<IScheduler> &scheduler);
    /** Access the scheduler singleton.
     *
     * @return A reference to the scheduler object.
     */
    static IScheduler &get();
    /** Set the active scheduler.
     *
     * Only one scheduler can be enabled at any time.
     *
     * @param[in] t the type of the scheduler to be enabled.
     */
    static void set(Type t);
    /** Returns the type of the active scheduler.
     *
     * @return The current scheduler's type.
     */
    static Type get_type();
    /** Returns true if the given scheduler type is supported. False otherwise.
     *
     * @return true if the given scheduler type is supported. False otherwise.
     */
    static bool is_available(Type t);

private:
    static Type                        _scheduler_type;
    static std::shared_ptr<IScheduler> _custom_scheduler;
    Scheduler();
};
}
#endif /* __ARM_COMPUTE_SCHEDULER_H__ */
