/*
 * Copyright (c) 2016-2019 ARM Limited.
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

#include "arm_compute/runtime/CL/CLHelpers.h"

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/runtime/CL/CLTuner.h"
#include "arm_compute/runtime/CL/tuners/Tuners.h"

using namespace arm_compute;

std::once_flag CLScheduler::_initialize_symbols;

CLScheduler::CLScheduler()
    : _context(), _queue(), _target(GPUTarget::MIDGARD), _is_initialised(false), _cl_tuner(nullptr), _cl_default_static_tuner(nullptr)
{
}

CLScheduler &CLScheduler::get()
{
    std::call_once(_initialize_symbols, opencl_is_available);
    static CLScheduler scheduler;
    return scheduler;
}

void CLScheduler::default_init_with_context(cl::Device &device, cl::Context &ctx, ICLTuner *cl_tuner)
{
    if(!_is_initialised)
    {
        cl::CommandQueue queue = cl::CommandQueue(ctx, device);
        CLKernelLibrary::get().init("./cl_kernels/", ctx, device);
        init(ctx, queue, device, cl_tuner);
        _cl_default_static_tuner = tuners::TunerFactory::create_tuner(_target);
        _cl_tuner                = (cl_tuner == nullptr) ? _cl_default_static_tuner.get() : cl_tuner;
    }
}

void CLScheduler::default_init(ICLTuner *cl_tuner)
{
    if(!_is_initialised)
    {
        cl::Context ctx;
        cl::Device  dev;
        cl_int      err;
        std::tie(ctx, dev, err) = create_opencl_context_and_device();
        ARM_COMPUTE_ERROR_ON_MSG(err != CL_SUCCESS, "Failed to create OpenCL context");
        cl::CommandQueue queue = cl::CommandQueue(ctx, dev);
        CLKernelLibrary::get().init("./cl_kernels/", ctx, dev);
        init(ctx, queue, dev, cl_tuner);
        // Create a default static tuner and set if none was provided
        _cl_default_static_tuner = tuners::TunerFactory::create_tuner(_target);
    }

    // Set CL tuner
    _cl_tuner = (cl_tuner == nullptr) ? _cl_default_static_tuner.get() : cl_tuner;
}

void CLScheduler::set_context(cl::Context context)
{
    _context = std::move(context);
    CLKernelLibrary::get().set_context(_context);
}

void CLScheduler::init(cl::Context context, cl::CommandQueue queue, const cl::Device &device, ICLTuner *cl_tuner)
{
    set_context(std::move(context));
    _queue          = std::move(queue);
    _target         = get_target_from_device(device);
    _is_initialised = true;
    _cl_tuner       = cl_tuner;
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
        _cl_tuner->tune_kernel_dynamic(kernel);
    }

    // Run kernel
    kernel.run(kernel.window(), _queue);

    if(flush)
    {
        _queue.flush();
    }
}
