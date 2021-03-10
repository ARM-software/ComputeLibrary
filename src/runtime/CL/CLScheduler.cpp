/*
 * Copyright (c) 2016-2021 Arm Limited.
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

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/runtime/CL/CLHelpers.h"
#include "arm_compute/runtime/CL/CLTuner.h"
#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
cl::Context &CLScheduler::context()
{
    ARM_COMPUTE_ERROR_ON(!_is_initialised);
    _context = CLKernelLibrary::get().context();
    return _context;
}

cl::CommandQueue &CLScheduler::queue()
{
    ARM_COMPUTE_ERROR_ON(!_is_initialised);
    return _queue;
}

GPUTarget CLScheduler::target() const
{
    return _target;
}

CLGEMMHeuristicsHandle *CLScheduler::gemm_heuristics() const
{
    return _gemm_heuristics;
}

void CLScheduler::set_queue(cl::CommandQueue queue)
{
    _queue = std::move(queue);
}

void CLScheduler::set_target(GPUTarget target)
{
    _target = target;
}

void CLScheduler::set_tuner(ICLTuner *tuner)
{
    _cl_tuner = tuner;
}

void CLScheduler::sync()
{
    _queue.finish();
}

cl::Event CLScheduler::enqueue_sync_event()
{
    cl::Event event;
    _queue.enqueueMarker(&event);
    return event;
}

void CLScheduler::tune_kernel_static(ICLKernel &kernel)
{
    if(_cl_tuner != nullptr)
    {
        _cl_tuner->tune_kernel_static(kernel);
    }
}

bool CLScheduler::is_initialised() const
{
    return _is_initialised;
}

std::once_flag CLScheduler::_initialize_symbols;

CLScheduler::CLScheduler()
    : _context(), _queue(), _target(GPUTarget::MIDGARD), _is_initialised(false), _cl_tuner(nullptr), _gemm_heuristics(nullptr)
{
}

CLScheduler &CLScheduler::get()
{
    std::call_once(_initialize_symbols, opencl_is_available);
    static CLScheduler scheduler;
    return scheduler;
}

void CLScheduler::default_init_with_context(cl::Device &device, cl::Context &ctx, ICLTuner *cl_tuner, CLGEMMHeuristicsHandle *gemm_h)
{
    if(!_is_initialised)
    {
        const std::string cl_kernels_folder("./cl_kernels/");
        cl::CommandQueue  queue = cl::CommandQueue(ctx, device);
        CLKernelLibrary::get().init(cl_kernels_folder, ctx, device);
        init(ctx, queue, device, cl_tuner, gemm_h);
        _cl_tuner = cl_tuner;
    }
}

void CLScheduler::default_init(ICLTuner *cl_tuner, CLGEMMHeuristicsHandle *gemm_h)
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
        init(ctx, queue, dev, cl_tuner, gemm_h);
    }

    // Set CL tuner
    _cl_tuner = cl_tuner;
}

void CLScheduler::set_context(cl::Context context)
{
    _context = std::move(context);
    CLKernelLibrary::get().set_context(_context);
}

void CLScheduler::init(cl::Context context, cl::CommandQueue queue, const cl::Device &device, ICLTuner *cl_tuner, CLGEMMHeuristicsHandle *gemm_h)
{
    set_context(std::move(context));
    _queue           = std::move(queue);
    _target          = get_target_from_device(device);
    _is_initialised  = true;
    _cl_tuner        = cl_tuner;
    _gemm_heuristics = gemm_h;
}

void CLScheduler::enqueue_common(ICLKernel &kernel, ITensorPack &tensors, bool flush)
{
    ARM_COMPUTE_ERROR_ON_MSG(!_is_initialised,
                             "The CLScheduler is not initialised yet! Please call the CLScheduler::get().default_init(), \
                             or CLScheduler::get()::init() and CLKernelLibrary::get()::init() function before running functions!");

    const bool inject_memory = !tensors.empty();

    // Tune the kernel if the CLTuner has been provided
    if(_cl_tuner != nullptr)
    {
        inject_memory ? _cl_tuner->tune_kernel_dynamic(kernel, tensors) : _cl_tuner->tune_kernel_dynamic(kernel);
    }

    // Run kernel
    inject_memory ? kernel.run_op(tensors, kernel.window(), _queue) : kernel.run(kernel.window(), _queue);

    if(flush)
    {
        _queue.flush();
    }
}

void CLScheduler::enqueue(ICLKernel &kernel, bool flush)
{
    ITensorPack pack;
    enqueue_common(kernel, pack, flush);
}

void CLScheduler::enqueue_op(ICLKernel &kernel, ITensorPack &tensors, bool flush)
{
    enqueue_common(kernel, tensors, flush);
}
} // namespace arm_compute
