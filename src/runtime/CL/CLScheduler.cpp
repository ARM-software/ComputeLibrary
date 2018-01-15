/*
 * Copyright (c) 2016-2018 ARM Limited.
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
#include "arm_compute/runtime/CL/CLScheduler.h"

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/runtime/CL/CLTuner.h"

using namespace arm_compute;

std::once_flag CLScheduler::_initialize_symbols;

CLScheduler::CLScheduler()
    : _context(), _queue(), _target(GPUTarget::MIDGARD), _is_initialised(false), _cl_tuner()
{
}

CLScheduler &CLScheduler::get()
{
    std::call_once(_initialize_symbols, opencl_is_available);
    static CLScheduler scheduler;
    return scheduler;
}

void CLScheduler::enqueue(ICLKernel &kernel, bool flush)
{
    ARM_COMPUTE_ERROR_ON_MSG(!_is_initialised,
                             "The CLScheduler is not initialised yet! Please call the CLScheduler::get().default_init(), \
                             or CLScheduler::get()::init() and CLKernelLibrary::get()::init() function before running functions!");

    // Tune the kernel if the CLTuner has been provided
    if(_cl_tuner != nullptr)
    {
        // Tune the OpenCL kernel
        _cl_tuner->tune_kernel(kernel);
    }

    // Run kernel
    kernel.run(kernel.window(), _queue);

    if(flush)
    {
        _queue.flush();
    }
}
