/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_ICPPKERNEL_H
#define ARM_COMPUTE_ICPPKERNEL_H

#include "arm_compute/core/CPP/CPPTypes.h"
#include "arm_compute/core/IKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/experimental/Types.h"

namespace arm_compute
{
class Window;
class ITensor;

/** Common interface for all kernels implemented in C++ */
class ICPPKernel : public IKernel
{
public:
    /** Default destructor */
    virtual ~ICPPKernel() = default;

    /** Execute the kernel on the passed window
     *
     * @warning If is_parallelisable() returns false then the passed window must be equal to window()
     *
     * @note The window has to be a region within the window returned by the window() method
     *
     * @note The width of the window has to be a multiple of num_elems_processed_per_iteration().
     *
     * @param[in] window Region on which to execute the kernel. (Must be a region of the window returned by window())
     * @param[in] info   Info about executing thread and CPU.
     */
    virtual void run(const Window &window, const ThreadInfo &info)
    {
        ARM_COMPUTE_UNUSED(window, info);
        ARM_COMPUTE_ERROR("default implementation of legacy run() virtual member function invoked");
    }

    /** legacy compatibility layer for implemantions which do not support thread_locator
     * In these cases we simply narrow the interface down the legacy version
     *
     * @param[in] window         Region on which to execute the kernel. (Must be a region of the window returned by window())
     * @param[in] info           Info about executing thread and CPU.
     * @param[in] thread_locator Specifies "where" the current thread is in the multi-dimensional space
     */
    virtual void run_nd(const Window &window, const ThreadInfo &info, const Window &thread_locator)
    {
        ARM_COMPUTE_UNUSED(thread_locator);
        run(window, info);
    }

    /** Execute the kernel on the passed window
     *
     * @warning If is_parallelisable() returns false then the passed window must be equal to window()
     *
     * @note The window has to be a region within the window returned by the window() method
     *
     * @note The width of the window has to be a multiple of num_elems_processed_per_iteration().
     *
     * @param[in] inputs  A vector containing the input tensors.
     * @param[in] outputs A vector containing the output tensors.
     * @param[in] window  Region on which to execute the kernel. (Must be a region of the window returned by window())
     * @param[in] info    Info about executing thread and CPU.
     */
    virtual void run_op(const InputTensorMap &inputs, const OutputTensorMap &outputs, const Window &window, const ThreadInfo &info)
    {
        ARM_COMPUTE_UNUSED(inputs, outputs, window, info);
    }

    /** Name of the kernel
     *
     * @return Kernel name
     */
    virtual const char *name() const = 0;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_ICPPKERNEL_H */
