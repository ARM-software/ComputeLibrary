/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef ARM_COMPUTE_GCRUNTIME_CONTEXT_H
#define ARM_COMPUTE_GCRUNTIME_CONTEXT_H

#include "arm_compute/core/GLES_COMPUTE/GCCoreRuntimeContext.h"
#include "arm_compute/core/GLES_COMPUTE/GCKernelLibrary.h"
#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCScheduler.h"
#include "arm_compute/runtime/IScheduler.h"
#include "arm_compute/runtime/RuntimeContext.h"

namespace arm_compute
{
/** Runtime context */
class GCRuntimeContext : public RuntimeContext
{
public:
    /** Default Constructor */
    GCRuntimeContext();
    /** Destructor */
    ~GCRuntimeContext() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCRuntimeContext(const GCRuntimeContext &) = delete;
    /** Default move constructor */
    GCRuntimeContext(GCRuntimeContext &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCRuntimeContext &operator=(const GCRuntimeContext &) = delete;
    /** Default move assignment operator */
    GCRuntimeContext &operator=(GCRuntimeContext &&) = default;
    /** CPU Scheduler setter */
    void set_gpu_scheduler(GCScheduler *scheduler);

    // Inherited overridden methods
    GCScheduler          *gpu_scheduler();
    GCKernelLibrary      &kernel_library();
    GCCoreRuntimeContext *core_runtime_context();

private:
    std::unique_ptr<GCScheduler> _gpu_owned_scheduler{ nullptr };
    GCScheduler                 *_gpu_scheduler{ nullptr };
    GCKernelLibrary              _kernel_lib{};
    GCCoreRuntimeContext         _core_context{};
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_GCRUNTIME_CONTEXT_H */
