/*
 * Copyright (c) 2019 Arm Limited.
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
#include "arm_compute/core/CL/CLCoreRuntimeContext.h"

namespace arm_compute
{
cl::Context CLCoreRuntimeContext::context()
{
    return _ctx;
}

cl::CommandQueue CLCoreRuntimeContext::queue()
{
    return _queue;
}

CLCoreRuntimeContext::CLCoreRuntimeContext()
    : _kernel_lib(nullptr), _ctx(), _queue()
{
}

CLCoreRuntimeContext::CLCoreRuntimeContext(CLKernelLibrary *kernel_lib, cl::Context ctx, cl::CommandQueue queue)
    : _kernel_lib(kernel_lib), _ctx(ctx), _queue(queue)
{
}

CLKernelLibrary *CLCoreRuntimeContext::kernel_library() const
{
    return _kernel_lib;
}
} // namespace arm_compute
