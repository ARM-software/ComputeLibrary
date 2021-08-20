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
#include "arm_compute/runtime/CL/CLRuntimeContext.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLHelpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

namespace arm_compute
{
CLRuntimeContext::CLRuntimeContext()
    : _gpu_owned_scheduler(std::make_unique<CLScheduler>()), _gpu_scheduler(_gpu_owned_scheduler.get()), _symbols(), _backend_type()
{
    _symbols.load_default();
    auto ctx_dev_err = create_opencl_context_and_device(_backend_type);
    ARM_COMPUTE_ERROR_ON_MSG(std::get<2>(ctx_dev_err) != CL_SUCCESS, "Failed to create OpenCL context");
    auto             ctx   = std::get<0>(ctx_dev_err);
    auto             dev   = std::get<1>(ctx_dev_err);
    cl::CommandQueue queue = cl::CommandQueue(ctx, dev);
    _gpu_owned_scheduler->init(ctx, queue, dev, &_tuner);
    const std::string cl_kernels_folder("./cl_kernels");
    CLKernelLibrary::get().init(cl_kernels_folder, ctx, dev);
}

CLKernelLibrary &CLRuntimeContext::kernel_library()
{
    return CLKernelLibrary::get();
}

void CLRuntimeContext::set_gpu_scheduler(CLScheduler *scheduler)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(scheduler);
    _gpu_scheduler = scheduler;
}

CLScheduler *CLRuntimeContext::gpu_scheduler()
{
    return _gpu_scheduler;
}

} // namespace arm_compute
