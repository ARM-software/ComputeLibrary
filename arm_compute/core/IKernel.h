/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_IKERNEL_H
#define ARM_COMPUTE_IKERNEL_H

#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"

namespace arm_compute
{
/** Common information for all the kernels */
class IKernel
{
public:
    /** Constructor */
    IKernel();
    /** Destructor */
    virtual ~IKernel() = default;
    /** Indicates whether or not the kernel is parallelisable
     *
     * If the kernel is parallelisable then the window returned by window() can be split into sub-windows
     * which can then be run in parallel.
     *
     * If the kernel is not parallelisable then only the window returned by window() can be passed to run()
     *
     * @return True if the kernel is parallelisable
     */
    virtual bool is_parallelisable() const;
    /** The size of the border for that kernel
     *
     * @return The width in number of elements of the border.
     */
    virtual BorderSize border_size() const;
    /** The maximum window the kernel can be executed on
     *
     * @return The maximum window the kernel can be executed on.
     */
    const Window &window() const;
    /** Function to check if the embedded window of this kernel has been configured
     *
     * @return True if the windows has been configured
     */
    bool is_window_configured() const;

protected:
    /** Configure the kernel's window
     *
     * @param[in] window The maximum window which will be returned by window()
     */
    void configure(const Window &window);

private:
    Window _window;
};
}
#endif /*ARM_COMPUTE_IKERNEL_H */
