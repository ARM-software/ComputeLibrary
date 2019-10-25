/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/core/GLES_COMPUTE/GCHelpers.h"

#include "arm_compute/core/GLES_COMPUTE/GCCoreRuntimeContext.h"

namespace arm_compute
{
GPUTarget get_target_from_device()
{
    const std::string device_name = reinterpret_cast<const char *>(glGetString(GL_RENDERER));

    return get_target_from_name(device_name);
}

GCKernel create_opengl_kernel(GCCoreRuntimeContext *ctx, const std::string &kernel_name, const std::set<std::string> &build_opts)
{
    if(ctx && ctx->kernel_library())
    {
        // New api going through the core context
        return ctx->kernel_library()->create_kernel(kernel_name, build_opts);
    }
    else
    {
        // Legacy code through the singleton
        return GCKernelLibrary::get().create_kernel(kernel_name, build_opts);
    }
}
} // namespace arm_compute
