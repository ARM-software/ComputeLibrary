/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_ICPPKERNEL_H__
#define __ARM_COMPUTE_ICPPKERNEL_H__

#include "arm_compute/core/CPP/CPPTypes.h"
#include "arm_compute/core/IKernel.h"

namespace arm_compute
{
class Window;

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
    virtual void run(const Window &window, const ThreadInfo &info) = 0;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_ICPPKERNEL_H__ */
