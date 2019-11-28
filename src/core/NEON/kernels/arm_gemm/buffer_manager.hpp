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
#pragma once

#include <cstdlib>
#include <vector>

#ifndef NO_MULTI_THREADING
#include <atomic>
#include <mutex>

#define USE_SEMAPHORE

#ifdef USE_SEMAPHORE
#include <condition_variable>
#endif

#endif

namespace arm_gemm {

#ifndef NO_MULTI_THREADING
enum class BufferStatus {
    IDLE,
    POPULATING,
    BUSY
};

class Buffer {
private:
    const int                _maxusers;    // Maximum permissible threads.
    void * const             _storage;     // Storage for buffer content.

    int                      _numusers;    // Actual number of threads (might be lower).

    volatile BufferStatus    _status = BufferStatus::IDLE; // Status
    std::atomic_int          _users = { };   // How many users are still using the buffer.
    volatile int             _index = 0;   // Which block of data currently resides in the buffer.

    std::mutex               _lock = { };
#ifdef USE_SEMAPHORE
    std::condition_variable  _cv = { };
#endif

    template <typename T>
    void populate_buffer(T func) {
        func(_storage);

        /* Now mark it as ready. */
#ifdef USE_SEMAPHORE
        {
            std::unique_lock<std::mutex> ul(_lock);
            _status = BufferStatus::BUSY;
            _cv.notify_all();
        }
#else
        _status = BufferStatus::BUSY;
#endif
    }

public:
    Buffer(Buffer &) = delete;
    Buffer &operator= (Buffer &) = delete;

    Buffer(void *storage, int maxusers) : _maxusers(maxusers), _storage(storage), _numusers(maxusers) {
        _status = BufferStatus::IDLE;
    }

    /* Try and populate the given index.
     * Wait if the buffer is busy with previous index, then:
     *
     * If the buffer is idle, grab it and populate it.
     * If it's already being populated by another thread or is ready, return.
     */
    template <typename T>
    void try_populate(const int index, T func) {
        for (;;) {
#ifdef USE_SEMAPHORE
            /* If it's busy with a previous index, wait on the semaphore. */
            if ((_status == BufferStatus::BUSY) && (_index != index)) {
                std::unique_lock<std::mutex> ul(_lock);

                if ((_status == BufferStatus::BUSY) && (_index != index)) {
                    _cv.wait(ul);
                }
            }
#endif
            /* Return if another thread is populating it already. */
            if ((_index == index) &&
                ((_status == BufferStatus::POPULATING) || (_status == BufferStatus::BUSY))) {
                return;
            }

            if (_status == BufferStatus::IDLE) {
                std::lock_guard<std::mutex> guard(_lock);

                /* If the buffer is still idle, we can grab it and populate it. */
                if (_status == BufferStatus::IDLE) {
                    _status = BufferStatus::POPULATING;
                    _index = index;
                    _users = _numusers;
                    break;
                }
            }
        }

        /* If we get here, fill in the buffer. */
        populate_buffer(func);
    }

    template <typename T>
    void *get(const int index, T func) {
        // Loop until we achieve something.
        for (;;) {
            // If the index is correct and the buffer status is busy then we can
            // just return the content.  No locking is needed here as the index
            // cannot change (and status cannot change from BUSY) until all
            // users have finished.
            if ((_index == index) && (_status == BufferStatus::BUSY)) {
                return _storage;
            }

            /* If the buffer still has some previous content, or is being
             * populated, we can wait with the semaphore.  */
#ifdef USE_SEMAPHORE
            if (((_status == BufferStatus::BUSY) && (_index != index)) ||
                 (_status == BufferStatus::POPULATING)) {
                std::unique_lock<std::mutex> ul(_lock);

                if (((_status == BufferStatus::BUSY) && (_index != index)) ||
                     (_status == BufferStatus::POPULATING)) {
                    _cv.wait(ul);
                }
            }
#endif

            // If it's idle, we need to populate it.  The IDLE->POPULATING
            // transition requires the lock.
            if (_status == BufferStatus::IDLE) {
                std::lock_guard<std::mutex> guard(_lock);

                /* If it's still idle, grab it.  Otherwise drop through and
                 * we'll do something else next time through the loop.  */
                if (_status == BufferStatus::IDLE) {
                    _status = BufferStatus::POPULATING;
                    _index = index;
                    _users = _numusers;
                    break;
                }
            }
        }

        /* If we get here we need to populate the buffer. */
        populate_buffer(func);

        return _storage;
    }

    /* Threads call this when they have finished processing a buffer.  We
     * simply (atomically) decrement the user count, and if it's hit zero we
     * flag the buffer as idle.
     */
    void release(void) {
        if (--_users == 0) {
#ifdef USE_SEMAPHORE
            std::unique_lock<std::mutex> ul(_lock);
            _status = BufferStatus::IDLE;
            /* We notify all waiters as we expect one to do the populating
             * and any others to go and process and earlier block.  */
            _cv.notify_all();
#else
            _status = BufferStatus::IDLE;
#endif
        }
    }

    /* This is called to change the number of users. */
    void set_numusers(int numusers) {
        _numusers = std::min(numusers, _maxusers);
    }
};


class BufferManager {
private:
    /* This has to be a vector of Buffer *, because a Buffer cannot be moved
     * or copied due to atomic members. */
    std::vector<Buffer *> _buffers = { };
    const int _maxthreads;
    void * const _storage;

public:
    BufferManager(BufferManager &) = delete;
    BufferManager & operator=(BufferManager &) = delete;

    // Say how much storage is needed.
    static inline size_t get_storage_requirement(const int maxthreads, const size_t buffersize) {
        return buffersize * ((maxthreads == 1) ? 1 : 3);
    }

    BufferManager(const int maxthreads, const size_t buffersize, void *storage) : _maxthreads(maxthreads), _storage(storage) {
        const int numbuffers = (maxthreads == 1) ? 1 : 3;

        /* We don't need any Buffer objects in single thread mode. */
        if (_maxthreads == 1) {
            return;
        }

        /* Use intptr_t to avoid performing arithmetic on a void * */
        intptr_t storage_int = reinterpret_cast<intptr_t>(_storage);

        for (int i=0; i<numbuffers; i++) {
            _buffers.push_back(new Buffer(reinterpret_cast<void *>(storage_int), _maxthreads));
            storage_int += buffersize;
        }
    }

    ~BufferManager() {
        while (_buffers.size()) {
            delete _buffers.back();
            _buffers.pop_back();
        }
    }

    template <typename T>
    void *get(const int index, T func) {
        /* In single thread mode, we just directly call the populating
         * function on the (single) buffer, otherwise forward to the
         * relevant Buffer.  */
        if (_maxthreads==1) {
            func(_storage);
            return _storage;
        } else {
            return _buffers[index % _buffers.size()]->get(index, func);
        }
    }

    template <typename T>
    void try_populate(const int index, T func) {
        /* No need for this in single thread mode. */
        if (_maxthreads==1) {
            return;
        }

        _buffers[index % _buffers.size()]->try_populate(index, func);
    }

    void release(const int index) {
        /* No need for this in single thread mode. */
        if (_maxthreads==1) {
            return;
        }

        _buffers[index % _buffers.size()]->release();
    }

    void set_nthreads(int threads) {
        if (_maxthreads==1) {
            return;
        }

        for(unsigned int i=0; i<_buffers.size(); i++) {
            _buffers[i]->set_numusers(threads);
        }
    }
};

#else

/* Trivial implementation if threading is disabled at compile time.
 *
 * Here, we only need storage for a single buffer.  The 'get' method needs
 * to call the supplied function to populate the buffer and then return it.
 * All the other methods do nothing.
 */

class BufferManager {
private:
    void * const _storage;

public:
    BufferManager(BufferManager &) = delete;
    BufferManager & operator=(BufferManager &) = delete;

    BufferManager(const int maxthreads, const size_t buffersize, void *storage) : _storage(storage) {
        UNUSED(maxthreads);
        UNUSED(buffersize);
    }

    ~BufferManager() { }

    // Say how much storage is needed.
    static inline size_t get_storage_requirement(const int maxthreads, const size_t buffersize) {
        UNUSED(maxthreads);
        return buffersize;
    }

    template <typename T>
    void try_populate(const int index, T func) {
         UNUSED(index);
         UNUSED(func);
    }

    void release(const int index) {
         UNUSED(index);
    }

    template <typename T>
    void *get(const int index, T func) {
        UNUSED(index);
        func(_storage);
        return _storage;
    }

    void set_nthreads(int) { }
};

#endif

} // namespace arm_gemm
