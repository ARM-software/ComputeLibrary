/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLRUNTIME_CONTEXT_H
#define ARM_COMPUTE_CLRUNTIME_CONTEXT_H

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTuner.h"
#include "arm_compute/runtime/CL/CLTypes.h"
#include "arm_compute/runtime/IScheduler.h"
#include "arm_compute/runtime/RuntimeContext.h"

namespace arm_compute
{
/** Runtime context */
class CLRuntimeContext : public RuntimeContext
{
public:
    /** Default Constructor */
    CLRuntimeContext();
    /** Destructor */
    ~CLRuntimeContext() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLRuntimeContext(const CLRuntimeContext &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLRuntimeContext &operator=(const CLRuntimeContext &) = delete;
    /** CPU Scheduler setter */
    void set_gpu_scheduler(CLScheduler *scheduler);

    // Inherited overridden methods
    CLScheduler     *gpu_scheduler();
    CLKernelLibrary &kernel_library();

private:
    std::unique_ptr<CLScheduler> _gpu_owned_scheduler{ nullptr };
    CLScheduler                 *_gpu_scheduler{ nullptr };
    CLTuner                      _tuner{ false };
    CLSymbols                    _symbols{};
    CLBackendType                _backend_type{ CLBackendType::Native };
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLRUNTIME_CONTEXT_H */
