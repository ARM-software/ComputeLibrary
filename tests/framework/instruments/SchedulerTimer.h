/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_SCHEDULER_TIMER
#define ARM_COMPUTE_TEST_SCHEDULER_TIMER

#include "Instrument.h"
#include "arm_compute/runtime/Scheduler.h"
#include <list>

namespace arm_compute
{
namespace test
{
namespace framework
{
/** Instrument creating measurements based on the information returned by clGetEventProfilingInfo for each OpenCL kernel executed*/
class SchedulerTimer : public Instrument
{
public:
    /** Construct a Scheduler timer.
     *
     * @param[in] scale_factor Measurement scale factor.
     */
    SchedulerTimer(ScaleFactor scale_factor);

    /** Prevent instances of this class from being copy constructed */
    SchedulerTimer(const SchedulerTimer &) = delete;
    /** Prevent instances of this class from being copied */
    SchedulerTimer &operator=(const SchedulerTimer &) = delete;

    std::string                 id() const override;
    void                        start() override;
    void                        stop() override;
    Instrument::MeasurementsMap measurements() const override;

    /** Kernel information */
    struct kernel_info
    {
        Instrument::MeasurementsMap measurements{}; /**< Time it took the kernel to run */
        std::string                 name{};         /**< Kernel name */
    };

private:
    std::list<kernel_info> _kernels;
    IScheduler            *_real_scheduler;
    Scheduler::Type        _real_scheduler_type;
    ScaleFactor            _scale_factor;
};
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SCHEDULER_TIMER */
