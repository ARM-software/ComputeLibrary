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
#ifndef ARM_COMPUTE_CLCORERUNTIME_CONTEXT_H
#define ARM_COMPUTE_CLCORERUNTIME_CONTEXT_H

#include "arm_compute/core/CL/OpenCL.h"

namespace arm_compute
{
// Forward declarations
class CLKernelLibrary;

/** Core runtime context for OpenCL */
class CLCoreRuntimeContext final
{
public:
    /** Legacy constructor */
    CLCoreRuntimeContext();

    /** Constructor */
    CLCoreRuntimeContext(CLKernelLibrary *kernel_lib, cl::Context ctx, cl::CommandQueue queue);
    /** Destructor */
    ~CLCoreRuntimeContext() = default;
    /** Default copy constructor */
    CLCoreRuntimeContext(const CLCoreRuntimeContext &) = default;
    /** Default move constructor */
    CLCoreRuntimeContext(CLCoreRuntimeContext &&) = default;
    /** Default copy assignment */
    CLCoreRuntimeContext &operator=(const CLCoreRuntimeContext &) = default;
    /** Default move assignment operator */
    CLCoreRuntimeContext &operator=(CLCoreRuntimeContext &&) = default;
    /** Kernel Library accessor
     *
     * @return The kernel library instance used by the core context
     */
    CLKernelLibrary *kernel_library() const;
    /** OpenCL context accessor
     *
     * @return The OpenCL context used by the core context
     */
    cl::Context context();
    /** OpenCL command queue accessor
     *
     * @return The OpenCL queue used by the core context
     */
    cl::CommandQueue queue();

private:
    CLKernelLibrary *_kernel_lib{ nullptr };
    cl::Context      _ctx{};
    cl::CommandQueue _queue{};
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLCORERUNTIME_CONTEXT_H */
