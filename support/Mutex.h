/*
 * Copyright (c) 2017-2019 Arm Limited.
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

/** Wrapper of lock_guard data-object */
template <typename Mutex>
using lock_guard = std::lock_guard<Mutex>;

/** Wrapper of lock_guard data-object */
template <typename Mutex>
using unique_lock = std::unique_lock<Mutex>;
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

/** Wrapper implementation of lock-guard data-object */
template <typename Mutex>
class lock_guard
{
public:
    typedef Mutex mutex_type;

public:
    explicit lock_guard(Mutex &m_)
        : m(m_)
    {
    }
    ~lock_guard()
    {
    }
    lock_guard(const lock_guard &) = delete;

private:
    mutex_type &m;
};

/** Wrapper implementation of unique-lock data-object */
template <typename Mutex>
class unique_lock
{
public:
    typedef Mutex mutex_type;

public:
    unique_lock() noexcept : m(nullptr)
    {
    }
    explicit unique_lock(mutex_type &m)
        : m(&m)
    {
    }
    unique_lock(const unique_lock &) = delete;
    unique_lock(unique_lock &&)      = default;
    unique_lock &operator=(const unique_lock &) = delete;
    unique_lock &operator=(unique_lock &&) = default;
    ~unique_lock()                         = default;
    void lock()
    {
    }
    bool try_lock()
    {
        return true;
    }
    void unlock()
    {
    }

private:
    mutex_type *m;
};
#endif /* NO_MULTI_THREADING */
}
#endif /* __ARM_COMPUTE_MUTEX_H__ */
