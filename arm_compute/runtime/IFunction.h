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
#ifndef __ARM_COMPUTE_IFUNCTION_H__
#define __ARM_COMPUTE_IFUNCTION_H__

namespace arm_compute
{
/** Base class for all functions */
class IFunction
{
public:
    /** Run the kernels contained in the function
     *
     * For NEON kernels:
     * - Multi-threading is used for the kernels which are parallelisable.
     * - By default std::thread::hardware_concurrency() threads are used.
     *
     * @note @ref CPPScheduler::set_num_threads() can be used to manually set the number of threads
     *
     * For OpenCL kernels:
     * - All the kernels are enqueued on the queue associated with CLScheduler.
     * - The queue is then flushed.
     *
     * @note The function will not block until the kernels are executed. It is the user's responsibility to wait.
     */
    virtual void run() = 0;
    /** Destructor
     *
     */
    virtual ~IFunction() = default;
};
}
#endif /*__ARM_COMPUTE_IFUNCTION_H__ */
