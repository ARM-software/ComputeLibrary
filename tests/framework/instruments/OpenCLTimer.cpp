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
#include "OpenCLTimer.h"

#include "../Framework.h"
#include "../Utils.h"

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
std::string OpenCLTimer::id() const
{
    return "OpenCLTimer";
}

/* Function to be used to intercept kernel enqueues and store their OpenCL Event */
class Interceptor
{
public:
    explicit Interceptor(OpenCLTimer &timer)
        : _timer(timer)
    {
    }

    cl_int operator()(
        cl_command_queue command_queue,
        cl_kernel        kernel,
        cl_uint          work_dim,
        const size_t    *gwo,
        const size_t    *gws,
        const size_t    *lws,
        cl_uint          num_events_in_wait_list,
        const cl_event *event_wait_list,
        cl_event        *event)
    {
        ARM_COMPUTE_ERROR_ON_MSG(event != nullptr, "Not supported");
        ARM_COMPUTE_UNUSED(event);

        OpenCLTimer::kernel_info info;
        cl::Kernel               cpp_kernel(kernel, true);
        std::stringstream        ss;
        ss << cpp_kernel.getInfo<CL_KERNEL_FUNCTION_NAME>();
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
        cl_int   retval = _timer.real_function(command_queue, kernel, work_dim, gwo, gws, lws, num_events_in_wait_list, event_wait_list, &tmp);
        info.event      = tmp;
        _timer.kernels.push_back(std::move(info));
        return retval;
    }

private:
    OpenCLTimer &_timer;
};

OpenCLTimer::OpenCLTimer(ScaleFactor scale_factor)
    : real_function(CLSymbols::get().clEnqueueNDRangeKernel_ptr)
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

void OpenCLTimer::start()
{
    kernels.clear();
    // Start intercepting enqueues:
    CLSymbols::get().clEnqueueNDRangeKernel_ptr = Interceptor(*this);
}

void OpenCLTimer::stop()
{
    // Restore real function
    CLSymbols::get().clEnqueueNDRangeKernel_ptr = real_function;
}

Instrument::MeasurementsMap OpenCLTimer::measurements() const
{
    MeasurementsMap measurements;
    unsigned int    kernel_number = 0;
    for(auto kernel : kernels)
    {
        //cl_int status = kernel.event.getInfo(<CL_EVENT_COMMAND_EXECUTION_STATUS>();
        cl_ulong queued = kernel.event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
        cl_ulong submit = kernel.event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
        cl_ulong start  = kernel.event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong end    = kernel.event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

        std::list<std::string> raw_data =
        {
            "queued", support::cpp11::to_string(queued),
            "submit", support::cpp11::to_string(submit),
            "start", support::cpp11::to_string(start),
            "end", support::cpp11::to_string(end),
        };
        measurements.emplace(kernel.name + " #" + support::cpp11::to_string(kernel_number++), Measurement((end - start) / _scale_factor, _unit, raw_data));
    }

    return measurements;
}
} // namespace framework
} // namespace test
} // namespace arm_compute
