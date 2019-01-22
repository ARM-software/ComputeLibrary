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
#ifndef __ARM_COMPUTE_CLSCHEDULER_H__
#define __ARM_COMPUTE_CLSCHEDULER_H__

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLTypes.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/ICLTuner.h"

namespace arm_compute
{
class ICLKernel;

/** Provides global access to a CL context and command queue. */
class CLScheduler
{
private:
    /** Constructor */
    CLScheduler();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLScheduler(const CLScheduler &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLScheduler &operator=(const CLScheduler &) = delete;

public:
    /** Access the scheduler singleton.
     *
     * @return The scheduler
     */
    static CLScheduler &get();
    /** Initialises the context and command queue used by the scheduler to default values
     *  and sets a default device and kernel path for the @ref CLKernelLibrary.
     *
     * @param[in] cl_tuner (Optional) Pointer to ICLTuner (default=nullptr)
     */
    void default_init(ICLTuner *cl_tuner = nullptr);
    /** Initialises the scheduler with context and device provided by the user
     *
     * @param[in] device   OpenCL device to be used
     * @param[in] ctx      OpenCL ctx to be used
     * @param[in] cl_tuner (Optional) Pointer to ICLTuner (default=nullptr)
     */
    void default_init_with_context(cl::Device &device, cl::Context &ctx, ICLTuner *cl_tuner = nullptr);

    /** Schedule the execution of the passed kernel if possible.
     *
     * @param[in] kernel Kernel to execute.
     * @param[in] flush  (Optional) Specifies if the command queue will be flushed after running the kernel.
     */
    void enqueue(ICLKernel &kernel, bool flush = true);

    /** Initialises the context and command queue to be used by the scheduler.
     *
     * @param[in] context  A CL context.
     * @param[in] queue    A CL command queue.
     * @param[in] device   A CL device.
     * @param[in] cl_tuner (Optional) Pointer to OpenCL tuner (default=nullptr)
     *                     Note: It is caller's responsibility to release the allocated memory for CLTuner
     */
    void init(cl::Context context, cl::CommandQueue queue, const cl::Device &device, ICLTuner *cl_tuner = nullptr);

    /** Accessor for the associated CL context.
     *
     * @return A CL context.
     */
    cl::Context &context()
    {
        ARM_COMPUTE_ERROR_ON(!_is_initialised);
        _context = CLKernelLibrary::get().context();
        return _context;
    }

    /** Accessor for the associated CL command queue.
     *
     * @return A CL command queue.
     */
    cl::CommandQueue &queue()
    {
        ARM_COMPUTE_ERROR_ON(!_is_initialised);
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

    /** Accessor to set the CL context to be used by the scheduler.
     *
     * @param[in] context A CL context.
     */
    void set_context(cl::Context context);

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

    /** Accessor to set the CL tuner to be used by the scheduler.
     *
     * @param[in] tuner A CL tuner
     */
    void set_tuner(ICLTuner *tuner)
    {
        _cl_tuner = tuner;
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

    /** Tunes OpenCL kernel
     *
     * @param[in] kernel Kernel to tune
     */
    void tune_kernel_static(ICLKernel &kernel)
    {
        if(_cl_tuner != nullptr)
        {
            _cl_tuner->tune_kernel_static(kernel);
        }
    }

    bool is_initialised() const
    {
        return _is_initialised;
    }

private:
    /** Flag to ensure symbols initialisation is happening before Scheduler creation */
    static std::once_flag _initialize_symbols;

    cl::Context               _context;
    cl::CommandQueue          _queue;
    GPUTarget                 _target;
    bool                      _is_initialised;
    ICLTuner                 *_cl_tuner;
    std::unique_ptr<ICLTuner> _cl_default_static_tuner;
};
}
#endif /* __ARM_COMPUTE_CLSCHEDULER_H__ */
