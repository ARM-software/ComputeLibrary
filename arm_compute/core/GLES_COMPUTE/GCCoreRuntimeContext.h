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
#ifndef __ARM_COMPUTE_GCCORERUNTIME_CONTEXT_H__
#define __ARM_COMPUTE_GCCORERUNTIME_CONTEXT_H__

#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"

namespace arm_compute
{
// Forward declarations
class GCKernelLibrary;

/** Core runtime context for OpenGL ES */
class GCCoreRuntimeContext final
{
public:
    /** Legacy constructor */
    GCCoreRuntimeContext();

    /** Constructor */
    GCCoreRuntimeContext(GCKernelLibrary *kernel_lib);
    /** Destructor */
    ~GCCoreRuntimeContext() = default;
    /** Default copy constructor */
    GCCoreRuntimeContext(const GCCoreRuntimeContext &) = default;
    /** Default move constructor */
    GCCoreRuntimeContext(GCCoreRuntimeContext &&) = default;
    /** Default copy assignment */
    GCCoreRuntimeContext &operator=(const GCCoreRuntimeContext &) = default;
    /** Default move assignment operator */
    GCCoreRuntimeContext &operator=(GCCoreRuntimeContext &&) = default;
    /** Kernel Library accessor
     *
     * @return The kernel library instance used by the core context
     */
    GCKernelLibrary *kernel_library() const;

private:
    GCKernelLibrary *_kernel_lib{ nullptr };
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_GCCORERUNTIME_CONTEXT_H__ */
