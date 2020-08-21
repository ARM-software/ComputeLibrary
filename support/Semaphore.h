/*
 * Copyright (c) 2017-2018 Arm Limited.
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
#ifndef __ARM_COMPUTE_UTILS_SEMAMPHORE_H__
#define __ARM_COMPUTE_UTILS_SEMAMPHORE_H__

#include "Mutex.h"
#include "support/Mutex.h"
#include <condition_variable>

namespace arm_compute
{
#ifndef NO_MULTI_THREADING
/** Semamphore class */
class Semaphore
{
public:
    /** Default Constuctor
     *
     * @param[in] value Semaphore initial value
     */
    Semaphore(int value = 0)
        : _value(value), _m(), _cv()
    {
    }
    /** Signals a semaphore */
    inline void signal()
    {
        {
            std::lock_guard<std::mutex> lock(_m);
            ++_value;
        }
        _cv.notify_one();
    }
    /** Waits on a semaphore */
    inline void wait()
    {
        std::unique_lock<std::mutex> lock(_m);
        _cv.wait(lock, [this]()
        {
            return _value > 0;
        });
        --_value;
    }

private:
    int                     _value;
    arm_compute::Mutex      _m;
    std::condition_variable _cv;
};
#else  /* NO_MULTI_THREADING */
/** Empty semamphore class */
class Semaphore
{
public:
    Semaphore(int value = 0)
        : _value(value)
    {
        (void)_value;
    }
    /** Signals a semaphore */
    inline void signal()
    {
        (void)_value;
    }
    /** Waits on a semaphore */
    inline void wait()
    {
        (void)_value;
    }

private:
    int _value;
};
#endif /* NO_MULTI_THREADING */
} // arm_compute
#endif /* __ARM_COMPUTE_UTILS_SEMAMPHORE_H__ */
