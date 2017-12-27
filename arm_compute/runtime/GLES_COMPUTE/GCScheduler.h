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

#ifndef __ARM_COMPUTE_GCSCHEDULER_H__
#define __ARM_COMPUTE_GCSCHEDULER_H__

#include "arm_compute/core/GLES_COMPUTE/IGCKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class IGCKernel;

/** Provides global access to a OpenGL ES context and command queue. */
class GCScheduler
{
public:
    /** Access the scheduler singleton.
     *
     * @return The scheduler
     */
    static GCScheduler &get();

    /** Initialises the context and command queue used by the scheduler to default values
     *  and sets a default device and kernel path for the @ref GCKernelLibrary.
     */
    void default_init();

    /** Schedule the execution of the passed kernel if possible.
     *
     * @param[in] kernel Kernel to execute.
     * @param[in] flush  (Optional) Specifies if the command queue will be flushed after running the kernel.
     */
    void dispatch(IGCKernel &kernel, bool flush = true);

    /** Initialises the display and context to be used by the scheduler.
     *
     * @param[in] dpy The EGL display connection
     * @param[in] ctx The EGL rendering context
     */
    void init(EGLDisplay dpy, EGLContext ctx);

    /** Defines a barrier ordering memory transactions. */
    void memory_barrier();

private:
    /** Constructor */
    GCScheduler();
    /** Destructor */
    ~GCScheduler();
    /** Prevent instances of this class from being copied */
    GCScheduler(const GCScheduler &) = delete;
    /** Prevent instances of this class from being copied */
    GCScheduler &operator=(const GCScheduler &) = delete;

    /** Set up EGL context */
    void setup_context();

    /** Flag to ensure symbols initialisation is happening before Scheduler creation */
    static std::once_flag _initialize_symbols;

    EGLDisplay _display; /**< Underlying EGL Display. */
    EGLContext _context; /**< Underlying EGL Context. */
};
}
#endif /* __ARM_COMPUTE_GCSCHEDULER_H__ */
