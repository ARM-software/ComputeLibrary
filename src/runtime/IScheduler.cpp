/*
 * Copyright (c) 2016-2022 Arm Limited.
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
#include "arm_compute/runtime/IScheduler.h"

#include "arm_compute/core/CPP/ICPPKernel.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Log.h"
#include "arm_compute/core/Window.h"
#include "src/common/cpuinfo/CpuInfo.h"
#include "src/runtime/SchedulerUtils.h"

namespace arm_compute
{
IScheduler::IScheduler()
{
    // Work out the best possible number of execution threads
    _num_threads_hint = cpuinfo::num_threads_hint();
}

CPUInfo &IScheduler::cpu_info()
{
    return CPUInfo::get();
}

void IScheduler::set_num_threads_with_affinity(unsigned int num_threads, BindFunc func)
{
    ARM_COMPUTE_UNUSED(num_threads, func);
    ARM_COMPUTE_ERROR("Feature for affinity setting is not implemented");
}

unsigned int IScheduler::num_threads_hint() const
{
    return _num_threads_hint;
}

void IScheduler::schedule_common(ICPPKernel *kernel, const Hints &hints, const Window &window, ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(!kernel, "The child class didn't set the kernel");
#ifndef BARE_METAL
    const Window &max_window = window;
    if(hints.split_dimension() == IScheduler::split_dimensions_all)
    {
        /*
         * if the split dim is size_t max then this signals we should parallelise over
         * all dimensions
         */
        const std::size_t m = max_window.num_iterations(Window::DimX);
        const std::size_t n = max_window.num_iterations(Window::DimY);

        //in c++17 this can be swapped for   auto [ m_threads, n_threads ] = split_2d(...
        unsigned m_threads, n_threads;
        std::tie(m_threads, n_threads) = scheduler_utils::split_2d(this->num_threads(), m, n);

        std::vector<IScheduler::Workload> workloads;
        for(unsigned int ni = 0; ni != n_threads; ++ni)
        {
            for(unsigned int mi = 0; mi != m_threads; ++mi)
            {
                workloads.push_back(
                    [ni, mi, m_threads, n_threads, &max_window, &kernel](const ThreadInfo & info)
                {
                    //narrow the window to our mi-ni workload
                    Window win = max_window.split_window(Window::DimX, mi, m_threads)
                                 .split_window(Window::DimY, ni, n_threads);

                    win.validate();

                    Window thread_locator;
                    thread_locator.set(Window::DimX, Window::Dimension(mi, m_threads));
                    thread_locator.set(Window::DimY, Window::Dimension(ni, n_threads));

                    thread_locator.validate();

                    kernel->run_nd(win, info, thread_locator);
                });
            }
        }
        run_workloads(workloads);
    }
    else
    {
        const unsigned int num_iterations = max_window.num_iterations(hints.split_dimension());
        const unsigned int num_threads    = std::min(num_iterations, this->num_threads());

        if(num_iterations == 0)
        {
            return;
        }

        if(!kernel->is_parallelisable() || num_threads == 1)
        {
            ThreadInfo info;
            info.cpu_info = &cpu_info();
            if(tensors.empty())
            {
                kernel->run(max_window, info);
            }
            else
            {
                kernel->run_op(tensors, max_window, info);
            }
        }
        else
        {
            unsigned int num_windows = 0;
            switch(hints.strategy())
            {
                case StrategyHint::STATIC:
                    num_windows = num_threads;
                    break;
                case StrategyHint::DYNAMIC:
                {
                    const unsigned int granule_threshold = (hints.threshold() <= 0) ? num_threads : static_cast<unsigned int>(hints.threshold());
                    // Make sure we don't use some windows which are too small as this might create some contention on the ThreadFeeder
                    num_windows = num_iterations > granule_threshold ? granule_threshold : num_iterations;
                    break;
                }
                default:
                    ARM_COMPUTE_ERROR("Unknown strategy");
            }
            // Make sure the smallest window is larger than minimim workload size
            num_windows = adjust_num_of_windows(max_window, hints.split_dimension(), num_windows, *kernel, cpu_info());

            std::vector<IScheduler::Workload> workloads(num_windows);
            for(unsigned int t = 0; t < num_windows; ++t)
            {
                //Capture 't' by copy, all the other variables by reference:
                workloads[t] = [t, &hints, &max_window, &num_windows, &kernel, &tensors](const ThreadInfo & info)
                {
                    Window win = max_window.split_window(hints.split_dimension(), t, num_windows);
                    win.validate();

                    if(tensors.empty())
                    {
                        kernel->run(win, info);
                    }
                    else
                    {
                        kernel->run_op(tensors, win, info);
                    }
                };
            }
            run_workloads(workloads);
        }
    }
#else  /* !BARE_METAL */
    ARM_COMPUTE_UNUSED(kernel, hints, window, tensors);
#endif /* !BARE_METAL */
}

void IScheduler::run_tagged_workloads(std::vector<Workload> &workloads, const char *tag)
{
    ARM_COMPUTE_UNUSED(tag);
    run_workloads(workloads);
}

std::size_t IScheduler::adjust_num_of_windows(const Window &window, std::size_t split_dimension, std::size_t init_num_windows, const ICPPKernel &kernel, const CPUInfo &cpu_info)
{
    // Mitigation of the narrow split issue, which occurs when the split dimension is too small to split (hence "narrow").
    if(window.num_iterations(split_dimension) < init_num_windows )
    {
        auto recommended_split_dim = Window::DimX;
        for(std::size_t dims = Window::DimY; dims <= Window::DimW; ++dims)
        {
            if(window.num_iterations(recommended_split_dim) < window.num_iterations(dims))
            {
                recommended_split_dim = dims;
            }
        }
        ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("%zu dimension is not a suitable dimension to split the workload. Recommended: %zu recommended_split_dim", split_dimension,
                                                  recommended_split_dim);
    }

    for(auto t = init_num_windows; t > 0; --t) // Trying the highest number of windows ,init_num_windows, first
    {
        // Try splitting the workload into t, subject to each subworkload size <= mws.
        if((window.num_iterations(split_dimension) / kernel.get_mws(cpu_info, t)) >= t)
        {
            if(t != init_num_windows)
            {
                ARM_COMPUTE_LOG_INFO_MSG_CORE("The scheduler is using a different thread count than the one assigned by the user.");
            }
            return t;
        }
    }
    ARM_COMPUTE_LOG_INFO_MSG_CORE("The scheduler is using single thread instead of the thread count assigned by the user.");
    return 1; //  If the workload is so small that it can't be split, we should run a single thread
}

} // namespace arm_compute
