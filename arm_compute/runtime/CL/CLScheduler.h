/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLSCHEDULER_H__
#define __ARM_COMPUTE_CLSCHEDULER_H__

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLTypes.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ICLKernel;

/** Provides global access to a CL context and command queue. */
class CLScheduler
{
private:
    /** Constructor */
    CLScheduler();

public:
    /** Access the scheduler singleton.
     *
     * @return The scheduler
     */
    static CLScheduler &get();
    /** Initialises the context and command queue used by the scheduler to default values
     *  and sets a default device and kernel path for the @ref CLKernelLibrary.
     */
    void default_init()
    {
        CLKernelLibrary::get().init("./cl_kernels/", cl::Context::getDefault(), cl::Device::getDefault());
        init(cl::Context::getDefault(), cl::CommandQueue::getDefault(), cl::Device::getDefault());
    }
    /** Schedule the execution of the passed kernel if possible.
     *
     * @param[in] kernel Kernel to execute.
     * @param[in] flush  (Optional) Specifies if the command queue will be flushed after running the kernel.
     */
    void enqueue(ICLKernel &kernel, bool flush = true);

    /** Initialises the context and command queue to be used by the scheduler.
     *
     * @param[in] context A CL context.
     * @param[in] queue   A CL command queue.
     * @param[in] device  A CL device.
     */
    void init(cl::Context context = cl::Context::getDefault(), cl::CommandQueue queue = cl::CommandQueue::getDefault(),
              cl::Device device = cl::Device::getDefault())
    {
        _context = std::move(context);
        _queue   = std::move(queue);
        _target  = get_target_from_device(device);
    }

    /** Accessor for the associated CL context.
     *
     * @return A CL context.
     */
    cl::Context &context()
    {
        return _context;
    }

    /** Accessor to set the CL context to be used by the scheduler.
     *
     * @param[in] context A CL context.
     */
    void set_context(cl::Context context)
    {
        _context = std::move(context);
    }

    /** Accessor for the associated CL command queue.
     *
     * @return A CL command queue.
     */
    cl::CommandQueue &queue()
    {
        return _queue;
    }

    /** Get the target GPU.
     *
     * @return The target GPU.
     */
    GPUTarget target() const
    {
        return _target;
    }

    /** Accessor to set the CL command queue to be used by the scheduler.
     *
     * @param[in] queue A CL command queue.
     */
    void set_queue(cl::CommandQueue queue)
    {
        _queue = std::move(queue);
    }

    /** Accessor to set target GPU to be used by the scheduler.
     *
     * @param[in] target The target GPU.
     */
    void set_target(GPUTarget target)
    {
        _target = target;
    }

    /** Blocks until all commands in the associated command queue have finished. */
    void sync()
    {
        _queue.finish();
    }

    /** Enqueues a marker into the associated command queue and return the event.
     *
     * @return An event that can be waited on to block the executing thread.
     */
    cl::Event enqueue_sync_event()
    {
        cl::Event event;
        _queue.enqueueMarker(&event);

        return event;
    }

private:
    cl::Context      _context;
    cl::CommandQueue _queue;
    GPUTarget        _target;
};
}
#endif /* __ARM_COMPUTE_CLSCHEDULER_H__ */
