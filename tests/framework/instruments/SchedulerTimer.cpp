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
#include "SchedulerTimer.h"

#include "WallClockTimer.h"
#include "arm_compute/core/CPP/ICPPKernel.h"

namespace arm_compute
{
namespace test
{
namespace framework
{
std::string SchedulerTimer::id() const
{
    return "SchedulerTimer";
}

class Interceptor final : public IScheduler
{
public:
    /** Default constructor. */
    Interceptor(std::list<SchedulerTimer::kernel_info> &kernels, IScheduler &real_scheduler, ScaleFactor scale_factor)
        : _kernels(kernels), _real_scheduler(real_scheduler), _timer(scale_factor)
    {
    }

    void set_num_threads(unsigned int num_threads) override
    {
        _real_scheduler.set_num_threads(num_threads);
    }

    unsigned int num_threads() const override
    {
        return _real_scheduler.num_threads();
    }

    void schedule(ICPPKernel *kernel, unsigned int split_dimension) override
    {
        _timer.start();
        _real_scheduler.schedule(kernel, split_dimension);
        _timer.stop();

        SchedulerTimer::kernel_info info;
        info.name         = kernel->name();
        info.measurements = _timer.measurements();
        _kernels.push_back(std::move(info));
    }

private:
    std::list<SchedulerTimer::kernel_info> &_kernels;
    IScheduler                             &_real_scheduler;
    WallClockTimer                          _timer;
};

SchedulerTimer::SchedulerTimer(ScaleFactor scale_factor)
    : _kernels(), _real_scheduler(nullptr), _real_scheduler_type(), _scale_factor(scale_factor)
{
}

void SchedulerTimer::start()
{
    ARM_COMPUTE_ERROR_ON(_real_scheduler != nullptr);
    _real_scheduler_type = Scheduler::get_type();
    //Note: We can't currently replace a custom scheduler
    if(_real_scheduler_type != Scheduler::Type::CUSTOM)
    {
        _real_scheduler  = &Scheduler::get();
        auto interceptor = std::make_shared<Interceptor>(_kernels, *_real_scheduler, _scale_factor);
        Scheduler::set(std::static_pointer_cast<IScheduler>(interceptor));
    }
    _kernels.clear();
}

void SchedulerTimer::stop()
{
    // Restore real scheduler
    Scheduler::set(_real_scheduler_type);
    _real_scheduler = nullptr;
}

Instrument::MeasurementsMap SchedulerTimer::measurements() const
{
    MeasurementsMap measurements;
    unsigned int    kernel_number = 0;
    for(auto kernel : _kernels)
    {
        measurements.emplace(kernel.name + " #" + support::cpp11::to_string(kernel_number++), kernel.measurements.begin()->second);
    }

    return measurements;
}
} // namespace framework
} // namespace test
} // namespace arm_compute
