/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/core/utils/misc/Cast.h"
#include "arm_compute/graph/INode.h"

namespace arm_compute
{
namespace test
{
namespace framework
{
template <bool output_timestamps>
std::string    SchedulerClock<output_timestamps>::id() const
{
    if(output_timestamps)
    {
        return "SchedulerTimestamps";
    }
    else
    {
        return "SchedulerTimer";
    }
}

template <bool    output_timestamps>
class Interceptor final : public IScheduler
{
public:
    /** Default constructor. */
    Interceptor(std::list<struct SchedulerClock<output_timestamps>::kernel_info> &kernels, IScheduler &real_scheduler, ScaleFactor scale_factor)
        : _kernels(kernels), _real_scheduler(real_scheduler), _timer(scale_factor), _prefix()
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

    void set_prefix(const std::string &prefix)
    {
        _prefix = prefix;
    }

    void schedule(ICPPKernel *kernel, const Hints &hints) override
    {
        _timer.start();
        _real_scheduler.schedule(kernel, hints.split_dimension());
        _timer.stop();

        typename SchedulerClock<output_timestamps>::kernel_info info;
        info.name         = kernel->name();
        info.prefix       = _prefix;
        info.measurements = _timer.measurements();
        _kernels.push_back(std::move(info));
    }

    void run_tagged_workloads(std::vector<Workload> &workloads, const char *tag) override
    {
        _timer.start();
        _real_scheduler.run_tagged_workloads(workloads, tag);
        _timer.stop();

        typename SchedulerClock<output_timestamps>::kernel_info info;
        info.name         = tag != nullptr ? tag : "Unknown";
        info.prefix       = _prefix;
        info.measurements = _timer.measurements();
        _kernels.push_back(std::move(info));
    }

protected:
    void run_workloads(std::vector<Workload> &workloads) override
    {
        ARM_COMPUTE_UNUSED(workloads);
        ARM_COMPUTE_ERROR("Can't be reached");
    }

private:
    std::list<struct SchedulerClock<output_timestamps>::kernel_info> &_kernels;
    IScheduler                                                       &_real_scheduler;
    WallClock<output_timestamps>                                      _timer;
    std::string                                                       _prefix;
};

template <bool output_timestamps>
SchedulerClock<output_timestamps>::SchedulerClock(ScaleFactor scale_factor)
    : _kernels(), _real_scheduler(nullptr), _real_scheduler_type(), _real_graph_function(nullptr), _scale_factor(scale_factor), _interceptor(nullptr)
{
}

template <bool output_timestamps>
void           SchedulerClock<output_timestamps>::test_start()
{
    // Start intercepting tasks:
    ARM_COMPUTE_ERROR_ON(_real_graph_function != nullptr);
    _real_graph_function  = graph::TaskExecutor::get().execute_function;
    auto task_interceptor = [this](graph::ExecutionTask & task)
    {
        Interceptor<output_timestamps> *scheduler = nullptr;
        if(dynamic_cast<Interceptor<output_timestamps> *>(this->_interceptor.get()) != nullptr)
        {
            scheduler = arm_compute::utils::cast::polymorphic_downcast<Interceptor<output_timestamps> *>(_interceptor.get());
            if(task.node != nullptr && !task.node->name().empty())
            {
                scheduler->set_prefix(task.node->name() + "/");
            }
            else
            {
                scheduler->set_prefix("");
            }
        }

        this->_real_graph_function(task);

        if(scheduler != nullptr)
        {
            scheduler->set_prefix("");
        }
    };

    ARM_COMPUTE_ERROR_ON(_real_scheduler != nullptr);
    _real_scheduler_type = Scheduler::get_type();
    //Note: We can't currently replace a custom scheduler
    if(_real_scheduler_type != Scheduler::Type::CUSTOM)
    {
        _real_scheduler = &Scheduler::get();
        _interceptor    = std::make_shared<Interceptor<output_timestamps>>(_kernels, *_real_scheduler, _scale_factor);
        Scheduler::set(std::static_pointer_cast<IScheduler>(_interceptor));
        graph::TaskExecutor::get().execute_function = task_interceptor;
    }
}

template <bool output_timestamps>
void           SchedulerClock<output_timestamps>::start()
{
    _kernels.clear();
}

template <bool output_timestamps>
void           SchedulerClock<output_timestamps>::test_stop()
{
    // Restore real scheduler
    Scheduler::set(_real_scheduler_type);
    _real_scheduler                             = nullptr;
    _interceptor                                = nullptr;
    graph::TaskExecutor::get().execute_function = _real_graph_function;
    _real_graph_function                        = nullptr;
}

template <bool              output_timestamps>
Instrument::MeasurementsMap SchedulerClock<output_timestamps>::measurements() const
{
    MeasurementsMap measurements;
    unsigned int    kernel_number = 0;
    for(auto kernel : _kernels)
    {
        std::string name = kernel.prefix + kernel.name + " #" + support::cpp11::to_string(kernel_number++);
        if(output_timestamps)
        {
            ARM_COMPUTE_ERROR_ON(kernel.measurements.size() != 2);
            for(auto const &m : kernel.measurements)
            {
                if(m.first.find("[start]") != std::string::npos)
                {
                    measurements.emplace("[start]" + name, m.second);
                }
                else if(m.first.find("[end]") != std::string::npos)
                {
                    measurements.emplace("[end]" + name, m.second);
                }
                else
                {
                    ARM_COMPUTE_ERROR("Measurement not handled");
                }
            }
        }
        else
        {
            measurements.emplace(name, kernel.measurements.begin()->second);
        }
    }

    return measurements;
}

} // namespace framework
} // namespace test
} // namespace arm_compute

template class arm_compute::test::framework::SchedulerClock<true>;
template class arm_compute::test::framework::SchedulerClock<false>;
