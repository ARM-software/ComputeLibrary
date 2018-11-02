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
#include "arm_compute/runtime/OMP/OMPScheduler.h"

#include "arm_compute/core/CPP/ICPPKernel.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/runtime/CPUUtils.h"

#include <omp.h>

using namespace arm_compute;

OMPScheduler &OMPScheduler::get()
{
    static OMPScheduler scheduler;
    return scheduler;
}

OMPScheduler::OMPScheduler() // NOLINT
    : _num_threads(omp_get_max_threads())
{
    get_cpu_configuration(_cpu_info);
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

void OMPScheduler::schedule(ICPPKernel *kernel, unsigned int split_dimension)
{
    ARM_COMPUTE_ERROR_ON_MSG(!kernel, "The child class didn't set the kernel");

    ThreadInfo info;
    info.cpu_info = &_cpu_info;

    const Window      &max_window     = kernel->window();
    const unsigned int num_iterations = max_window.num_iterations(split_dimension);
    info.num_threads                  = std::min(num_iterations, _num_threads);

    if(!kernel->is_parallelisable() || info.num_threads == 1)
    {
        kernel->run(max_window, info);
    }
    else
    {
        #pragma omp parallel firstprivate(info) num_threads(info.num_threads)
        {
            const int tid  = omp_get_thread_num();
            Window    win  = max_window.split_window(split_dimension, tid, info.num_threads);
            info.thread_id = tid;
            kernel->run(win, info);
        }
    }
}
