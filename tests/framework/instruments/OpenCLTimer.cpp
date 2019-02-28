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
#include "OpenCLTimer.h"

#include "../Framework.h"
#include "../Utils.h"

#include "arm_compute/graph/INode.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#ifndef ARM_COMPUTE_CL
#error "You can't use OpenCLTimer without OpenCL"
#endif /* ARM_COMPUTE_CL */

namespace arm_compute
{
namespace test
{
namespace framework
{
template <bool output_timestamps>
std::string    OpenCLClock<output_timestamps>::id() const
{
    if(output_timestamps)
    {
        return "OpenCLTimestamps";
    }
    else
    {
        return "OpenCLTimer";
    }
}

template <bool output_timestamps>
OpenCLClock<output_timestamps>::OpenCLClock(ScaleFactor scale_factor)
    : _kernels(), _real_function(nullptr), _real_graph_function(nullptr), _prefix(), _timer_enabled(false)
{
    auto                        q     = CLScheduler::get().queue();
    cl_command_queue_properties props = q.getInfo<CL_QUEUE_PROPERTIES>();
    if((props & CL_QUEUE_PROFILING_ENABLE) == 0)
    {
        CLScheduler::get().set_queue(cl::CommandQueue(CLScheduler::get().context(), props | CL_QUEUE_PROFILING_ENABLE));
    }

    switch(scale_factor)
    {
        case ScaleFactor::NONE:
            _scale_factor = 1.f;
            _unit         = "ns";
            break;
        case ScaleFactor::TIME_US:
            _scale_factor = 1000.f;
            _unit         = "us";
            break;
        case ScaleFactor::TIME_MS:
            _scale_factor = 1000000.f;
            _unit         = "ms";
            break;
        case ScaleFactor::TIME_S:
            _scale_factor = 1000000000.f;
            _unit         = "s";
            break;
        default:
            ARM_COMPUTE_ERROR("Invalid scale");
    }
}

template <bool output_timestamps>
void           OpenCLClock<output_timestamps>::test_start()
{
    // Start intercepting enqueues:
    ARM_COMPUTE_ERROR_ON(_real_function != nullptr);
    ARM_COMPUTE_ERROR_ON(_real_graph_function != nullptr);
    _real_function       = CLSymbols::get().clEnqueueNDRangeKernel_ptr;
    _real_graph_function = graph::TaskExecutor::get().execute_function;
    auto interceptor     = [this](
                               cl_command_queue command_queue,
                               cl_kernel        kernel,
                               cl_uint          work_dim,
                               const size_t    *gwo,
                               const size_t    *gws,
                               const size_t    *lws,
                               cl_uint          num_events_in_wait_list,
                               const cl_event * event_wait_list,
                               cl_event *       event)
    {
        if(this->_timer_enabled)
        {
            kernel_info       info;
            cl::Kernel        cpp_kernel(kernel, true);
            std::stringstream ss;
            ss << this->_prefix << cpp_kernel.getInfo<CL_KERNEL_FUNCTION_NAME>();
            if(gws != nullptr)
            {
                ss << " GWS[" << gws[0] << "," << gws[1] << "," << gws[2] << "]";
            }
            if(lws != nullptr)
            {
                ss << " LWS[" << lws[0] << "," << lws[1] << "," << lws[2] << "]";
            }
            info.name = ss.str();
            cl_event tmp;
            cl_int   retval = this->_real_function(command_queue, kernel, work_dim, gwo, gws, lws, num_events_in_wait_list, event_wait_list, &tmp);
            info.event      = tmp;
            this->_kernels.push_back(std::move(info));

            if(event != nullptr)
            {
                //return cl_event from the intercepted call
                clRetainEvent(tmp);
                *event = tmp;
            }
            return retval;
        }
        else
        {
            return this->_real_function(command_queue, kernel, work_dim, gwo, gws, lws, num_events_in_wait_list, event_wait_list, event);
        }
    };

    // Start intercepting tasks:
    auto task_interceptor = [this](graph::ExecutionTask & task)
    {
        if(task.node != nullptr && !task.node->name().empty())
        {
            this->_prefix = task.node->name() + "/";
        }
        else
        {
            this->_prefix = "";
        }
        this->_real_graph_function(task);
        this->_prefix = "";
    };

    CLSymbols::get().clEnqueueNDRangeKernel_ptr = interceptor;
    graph::TaskExecutor::get().execute_function = task_interceptor;
}

template <bool output_timestamps>
void           OpenCLClock<output_timestamps>::start()
{
    _kernels.clear();
    _timer_enabled = true;
}
template <bool output_timestamps>
void           OpenCLClock<output_timestamps>::stop()
{
    _timer_enabled = false;
}

template <bool output_timestamps>
void           OpenCLClock<output_timestamps>::test_stop()
{
    // Restore real function
    CLSymbols::get().clEnqueueNDRangeKernel_ptr = _real_function;
    graph::TaskExecutor::get().execute_function = _real_graph_function;
    _real_graph_function                        = nullptr;
    _real_function                              = nullptr;
}

template <bool              output_timestamps>
Instrument::MeasurementsMap OpenCLClock<output_timestamps>::measurements() const
{
    MeasurementsMap measurements;
    unsigned int    kernel_number = 0;
    for(auto kernel : _kernels)
    {
        cl_ulong queued, flushed, start, end;
        kernel.event.getProfilingInfo(CL_PROFILING_COMMAND_QUEUED, &queued);
        kernel.event.getProfilingInfo(CL_PROFILING_COMMAND_SUBMIT, &flushed);
        kernel.event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
        kernel.event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
        std::string name = kernel.name + " #" + support::cpp11::to_string(kernel_number++);

        if(output_timestamps)
        {
            measurements.emplace("[start]" + name, Measurement(start / static_cast<cl_ulong>(_scale_factor), _unit));
            measurements.emplace("[queued]" + name, Measurement(queued / static_cast<cl_ulong>(_scale_factor), _unit));
            measurements.emplace("[flushed]" + name, Measurement(flushed / static_cast<cl_ulong>(_scale_factor), _unit));
            measurements.emplace("[end]" + name, Measurement(end / static_cast<cl_ulong>(_scale_factor), _unit));
        }
        else
        {
            measurements.emplace(name, Measurement((end - start) / _scale_factor, _unit));
        }
    }

    return measurements;
}

template <bool              output_timestamps>
Instrument::MeasurementsMap OpenCLClock<output_timestamps>::test_measurements() const
{
    MeasurementsMap measurements;

    if(output_timestamps)
    {
        // The OpenCL clock and the wall clock are not in sync, so we use
        // this trick to calculate the offset between the two clocks:
        ::cl::Event event;
        cl_ulong    now_gpu;

        // Enqueue retrieve current CPU clock and enqueue a dummy marker
        std::chrono::high_resolution_clock::time_point now_cpu = std::chrono::high_resolution_clock::now();
        CLScheduler::get().queue().enqueueMarker(&event);

        CLScheduler::get().queue().finish();
        //Access the time at which the marker was enqueued:
        event.getProfilingInfo(CL_PROFILING_COMMAND_QUEUED, &now_gpu);

        measurements.emplace("Now Wall clock", Measurement(now_cpu.time_since_epoch().count() / 1000, "us"));
        measurements.emplace("Now OpenCL", Measurement(now_gpu / static_cast<cl_ulong>(_scale_factor), _unit));
    }

    return measurements;
}

} // namespace framework
} // namespace test
} // namespace arm_compute

template class arm_compute::test::framework::OpenCLClock<true>;
template class arm_compute::test::framework::OpenCLClock<false>;
