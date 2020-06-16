/*
 * Copyright (c) 2017-2020 ARM Limited.
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
#include "arm_compute/runtime/OMP/OMPScheduler.h"

#include "arm_compute/core/CPP/ICPPKernel.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/runtime/CPUUtils.h"

#include <omp.h>

namespace arm_compute
{
OMPScheduler::OMPScheduler() // NOLINT
    : _num_threads(omp_get_max_threads())
{
}

unsigned int OMPScheduler::num_threads() const
{
    return _num_threads;
}

void OMPScheduler::set_num_threads(unsigned int num_threads)
{
    const unsigned int num_cores = omp_get_max_threads();
    _num_threads                 = (num_threads == 0) ? num_cores : num_threads;
}

void OMPScheduler::schedule(ICPPKernel *kernel, const Hints &hints)
{
    ARM_COMPUTE_ERROR_ON_MSG(!kernel, "The child class didn't set the kernel");
    ARM_COMPUTE_ERROR_ON_MSG(hints.strategy() == StrategyHint::DYNAMIC,
                             "Dynamic scheduling is not supported in OMPScheduler");

    const Window      &max_window     = kernel->window();
    const unsigned int num_iterations = max_window.num_iterations(hints.split_dimension());
    const unsigned int num_threads    = std::min(num_iterations, _num_threads);

    if(!kernel->is_parallelisable() || num_threads == 1)
    {
        ThreadInfo info;
        info.cpu_info = &_cpu_info;
        kernel->run(max_window, info);
    }
    else
    {
        const unsigned int                num_windows = num_threads;
        std::vector<IScheduler::Workload> workloads(num_windows);
        for(unsigned int t = 0; t < num_windows; t++)
        {
            //Capture 't' by copy, all the other variables by reference:
            workloads[t] = [t, &hints, &max_window, &num_windows, &kernel](const ThreadInfo & info)
            {
                Window win = max_window.split_window(hints.split_dimension(), t, num_windows);
                win.validate();
                kernel->run(win, info);
            };
        }
        run_workloads(workloads);
    }
}

void OMPScheduler::schedule_op(ICPPKernel *kernel, const Hints &hints, const std::vector<InputTensor> &inputs, const std::vector<OutputTensor> &outputs)
{
    ARM_COMPUTE_ERROR_ON_MSG(!kernel, "The child class didn't set the kernel");
    ARM_COMPUTE_ERROR_ON_MSG(hints.strategy() == StrategyHint::DYNAMIC,
                             "Dynamic scheduling is not supported in OMPScheduler");

    const Window      &max_window     = kernel->window();
    const unsigned int num_iterations = max_window.num_iterations(hints.split_dimension());
    const unsigned int num_threads    = std::min(num_iterations, _num_threads);

    if(!kernel->is_parallelisable() || num_threads == 1)
    {
        ThreadInfo info;
        info.cpu_info = &_cpu_info;
        kernel->run_op(inputs, outputs, max_window, info);
    }
    else
    {
        const unsigned int                num_windows = num_threads;
        std::vector<IScheduler::Workload> workloads(num_windows);
        for(unsigned int t = 0; t < num_windows; t++)
        {
            //Capture 't' by copy, all the other variables by reference:
            workloads[t] = [t, &hints, &max_window, &num_windows, &kernel, &inputs, &outputs](const ThreadInfo & info)
            {
                Window win = max_window.split_window(hints.split_dimension(), t, num_windows);
                win.validate();
                kernel->run_op(inputs, outputs, win, info);
            };
        }
        run_workloads(workloads);
    }
}
#ifndef DOXYGEN_SKIP_THIS
void OMPScheduler::run_workloads(std::vector<arm_compute::IScheduler::Workload> &workloads)
{
    const unsigned int num_threads = std::min(_num_threads, static_cast<unsigned int>(workloads.size()));
    if(num_threads < 1)
    {
        return;
    }

    ThreadInfo info;
    info.cpu_info    = &_cpu_info;
    info.num_threads = num_threads;
    #pragma omp parallel firstprivate(info) num_threads(num_threads)
    {
        const int tid  = omp_get_thread_num();
        info.thread_id = tid;
        workloads[tid](info);
    }
}
#endif /* DOXYGEN_SKIP_THIS */
} // namespace arm_compute
