/*
 * Copyright (c) 2017-2019,2021 Arm Limited.
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
#include "arm_compute/graph/Workload.h"
#include "arm_compute/runtime/Scheduler.h"

#include <list>
#include <memory>
#include <vector>

namespace arm_compute
{
namespace test
{
namespace framework
{
/** Scheduler user interface  */
class ISchedulerUser
{
public:
    /** Default Destructor */
    virtual ~ISchedulerUser() = default;
    /** Intercept the scheduler used by
     *
     * @param interceptor Intercept the scheduler used by the scheduler user.
     */
    virtual void intercept_scheduler(std::unique_ptr<IScheduler> interceptor) = 0;
    /** Restore the original scheduler */
    virtual void restore_scheduler() = 0;
    /** Real scheduler accessor
     *
     * @return The real scheduler
     */
    virtual IScheduler *scheduler() = 0;
};

/** Instrument creating measurements based on the information returned by clGetEventProfilingInfo for each OpenCL kernel executed*/
template <bool output_timestamps>
class SchedulerClock : public Instrument
{
public:
    using LayerData = std::map<std::string, std::string>;
    /** Construct a Scheduler timer.
     *
     * @param[in] scale_factor Measurement scale factor.
     */
    SchedulerClock(ScaleFactor scale_factor);
    /** Prevent instances of this class from being copy constructed */
    SchedulerClock(const SchedulerClock &) = delete;
    /** Prevent instances of this class from being copied */
    SchedulerClock &operator=(const SchedulerClock &) = delete;
    /** Use the default move assignment operator */
    SchedulerClock &operator=(SchedulerClock &&) = default;
    /** Use the default move constructor */
    SchedulerClock(SchedulerClock &&) = default;
    /** Use the default destructor */
    ~SchedulerClock() = default;

    // Inherited overridden methods
    std::string                 id() const override;
    void                        test_start() override;
    void                        start() override;
    void                        test_stop() override;
    Instrument::MeasurementsMap measurements() const override;
    std::string                 instrument_header() const override;

    /** Kernel information */
    struct kernel_info
    {
        Instrument::MeasurementsMap measurements{}; /**< Time it took the kernel to run */
        std::string                 name{};         /**< Kernel name */
        std::string                 prefix{};       /**< Kernel prefix */
    };

private:
    std::list<kernel_info>                       _kernels;
    std::map<std::string, LayerData>             _layer_data_map;
    IScheduler                                  *_real_scheduler;
    Scheduler::Type                              _real_scheduler_type;
    std::function<decltype(graph::execute_task)> _real_graph_function;
    ScaleFactor                                  _scale_factor;
    std::shared_ptr<IScheduler>                  _interceptor;
    std::vector<ISchedulerUser *>                _scheduler_users;
};

using SchedulerTimer      = SchedulerClock<false>;
using SchedulerTimestamps = SchedulerClock<true>;

} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SCHEDULER_TIMER */
