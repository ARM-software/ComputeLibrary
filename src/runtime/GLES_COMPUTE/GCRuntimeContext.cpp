/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#include "arm_compute/runtime/GLES_COMPUTE/GCRuntimeContext.h"

#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCHelpers.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCScheduler.h"

namespace arm_compute
{
GCRuntimeContext::GCRuntimeContext()
    : _gpu_owned_scheduler(std::make_unique<GCScheduler>()),
      _gpu_scheduler(_gpu_owned_scheduler.get()),
      _core_context()
{
    auto attrs   = create_opengl_display_and_context();
    auto display = std::get<0>(attrs);
    auto ctx     = std::get<1>(attrs);

    _gpu_owned_scheduler->default_init_with_context(display, ctx);
    _kernel_lib.init("./cs_shaders/", display, ctx);

    _core_context = GCCoreRuntimeContext(&_kernel_lib);
}

GCKernelLibrary &GCRuntimeContext::kernel_library()
{
    return _kernel_lib;
}

GCCoreRuntimeContext *GCRuntimeContext::core_runtime_context()
{
    return &_core_context;
}

void GCRuntimeContext::set_gpu_scheduler(GCScheduler *scheduler)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(scheduler);
    _gpu_scheduler = scheduler;
}

GCScheduler *GCRuntimeContext::gpu_scheduler()
{
    return _gpu_scheduler;
}
} // namespace arm_compute
