/*
 * Copyright (c) 2017-2025 Arm Limited.
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

#include <omp.h>

namespace arm_compute
{
#if !defined(_WIN64) && !defined(BARE_METAL) && !defined(__APPLE__) && !defined(__OpenBSD__) && \
    (defined(__arm__) || defined(__aarch64__)) && defined(__ANDROID__)
OMPScheduler::OMPScheduler() // NOLINT
    : _num_threads(cpu_info().get_cpu_num_excluding_little()),
      _nonlittle_num_cpus(cpu_info().get_cpu_num_excluding_little())
{
}
#else  /* !defined(_WIN64) && !defined(BARE_METAL) && !defined(__APPLE__) && !defined(__OpenBSD__) && \
    (defined(__arm__) || defined(__aarch64__)) && defined(__ANDROID__)*/
OMPScheduler::OMPScheduler() // NOLINT
    : _num_threads(omp_get_max_threads()), _nonlittle_num_cpus(cpu_info().get_cpu_num_excluding_little())
{
}
#endif /* !defined(_WIN64) && !defined(BARE_METAL) && !defined(__APPLE__) && !defined(__OpenBSD__) && \
    (defined(__arm__) || defined(__aarch64__)) && defined(__ANDROID__)*/

unsigned int OMPScheduler::num_threads() const
{
    return _num_threads;
}

void OMPScheduler::set_num_threads(unsigned int num_threads)
{
    const unsigned int num_cores = omp_get_max_threads();
#if !defined(_WIN64) && !defined(BARE_METAL) && !defined(__APPLE__) && !defined(__OpenBSD__) && \
    (defined(__arm__) || defined(__aarch64__)) && defined(__ANDROID__)
    const unsigned int adjusted_num_threads = std::min(_nonlittle_num_cpus, num_threads);
    _num_threads                            = (num_threads == 0) ? num_cores : adjusted_num_threads;
#else  /* !defined(_WIN64) && !defined(BARE_METAL) && !defined(__APPLE__) && !defined(__OpenBSD__) && \
    (defined(__arm__) || defined(__aarch64__)) && defined(__ANDROID__)*/
    _num_threads = (num_threads == 0) ? num_cores : num_threads;
#endif /* !defined(_WIN64) && !defined(BARE_METAL) && !defined(__APPLE__) && !defined(__OpenBSD__) && \
    (defined(__arm__) || defined(__aarch64__)) && defined(__ANDROID__)*/
}

void OMPScheduler::schedule(ICPPKernel *kernel, const Hints &hints)
{
    ITensorPack tensors;
    schedule_common(kernel, hints, kernel->window(), tensors);
}

void OMPScheduler::schedule_op(ICPPKernel *kernel, const Hints &hints, const Window &window, ITensorPack &tensors)
{
    // The rest of the logic in this function does not handle the
    // split_dimensions_all case so we defer to IScheduler::schedule_common()
    if (hints.split_dimension() == IScheduler::split_dimensions_all)
    {
        return schedule_common(kernel, hints, window, tensors);
    }

    ARM_COMPUTE_ERROR_ON_MSG(!kernel, "The child class didn't set the kernel");
    ARM_COMPUTE_ERROR_ON_MSG(hints.strategy() == StrategyHint::DYNAMIC,
                             "Dynamic scheduling is not supported in OMPScheduler");

    const Window      &max_window     = window;
    const unsigned int num_iterations = max_window.num_iterations(hints.split_dimension());
    const unsigned int mws            = kernel->get_mws(CPUInfo::get(), _num_threads);

    // Ensure each thread has mws amount of work to do (i.e. ceil(num_iterations / mws) threads)
    const unsigned int candidate_num_threads = (num_iterations + mws - 1) / mws;

    // Cap the number of threads to be spawn with the size of the thread pool
    const unsigned int num_threads = std::min(candidate_num_threads, _num_threads);

    if (!kernel->is_parallelisable() || num_threads == 1)
    {
        ThreadInfo info;
        info.cpu_info = &cpu_info();
        kernel->run_op(tensors, max_window, info);
    }
    else
    {
        const unsigned int                num_windows = num_threads;
        std::vector<IScheduler::Workload> workloads(num_windows);
        for (unsigned int t = 0; t < num_windows; t++)
        {
            //Capture 't' by copy, all the other variables by reference:
            workloads[t] = [t, &hints, &max_window, &num_windows, &kernel, &tensors](const ThreadInfo &info)
            {
                Window win = max_window.split_window(hints.split_dimension(), t, num_windows);
                win.validate();
                kernel->run_op(tensors, win, info);
            };
        }
        run_workloads(workloads);
    }
}
#ifndef DOXYGEN_SKIP_THIS
void OMPScheduler::run_workloads(std::vector<arm_compute::IScheduler::Workload> &workloads)
{
    const unsigned int amount_of_work     = static_cast<unsigned int>(workloads.size());
    const unsigned int num_threads_to_use = std::min(_num_threads, amount_of_work);

    if (num_threads_to_use < 1)
    {
        return;
    }

    ThreadInfo info;
    info.cpu_info    = &cpu_info();
    info.num_threads = num_threads_to_use;

    ARM_COMPUTE_ERROR_ON(amount_of_work > _num_threads);

#if !defined(__ANDROID__)
    // Use fixed number of omp threads in the thread pool because changing this
    // in-between kernel execution negatively affects the scheduler performance,
    // possibly switching between X and Y number of threads, causing reconfiguration
    // of the synchronization mechanism. This has been only tested in a subset of
    // operating systems, thus we limit the change using guards.
    const unsigned int omp_num_threads = _num_threads;
#else  /* !__ANDROID__ */
    const unsigned int omp_num_threads = num_threads_to_use;
#endif /* __ANDROID__ */

#pragma omp parallel for firstprivate(info) num_threads(omp_num_threads) default(shared) proc_bind(close) \
    schedule(static, 1)
    for (unsigned int wid = 0; wid < amount_of_work; ++wid)
    {
        info.thread_id = wid;
        workloads[wid](info);
    }
}
#endif /* DOXYGEN_SKIP_THIS */
} // namespace arm_compute
