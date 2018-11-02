/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_MUTEX_H__
#define __ARM_COMPUTE_MUTEX_H__

#include <mutex>

namespace arm_compute
{
#ifndef NO_MULTI_THREADING
/** Wrapper of Mutex data-object */
using Mutex = std::mutex;
#else  /* NO_MULTI_THREADING */
/** Wrapper implementation of Mutex data-object */
class Mutex
{
public:
    /** Default constructor */
    Mutex() = default;
    /** Default destructor */
    ~Mutex() = default;

    /** Lock */
    void lock() {};

    /** Unlock */
    void unlock() {};

    /** Try the lock.
     *
     * @return true.
     */
    bool try_lock()
    {
        return true;
    }
};
#endif /* NO_MULTI_THREADING */
}
#endif /* __ARM_COMPUTE_MUTEX_H__ */
