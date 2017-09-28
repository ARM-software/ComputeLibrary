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
#include "arm_compute/runtime/CPP/CPPScheduler.h"

#include "arm_compute/core/CPP/ICPPKernel.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <system_error>
#include <thread>

namespace arm_compute
{
class Thread
{
public:
    /** Start a new thread. */
    Thread();

    Thread(const Thread &) = delete;
    Thread &operator=(const Thread &) = delete;
    Thread(Thread &&)                 = delete;
    Thread &operator=(Thread &&) = delete;

    /** Destructor. Make the thread join. */
    ~Thread();

    /** Request the worker thread to start executing the given kernel
     * This function will return as soon as the kernel has been sent to the worker thread.
     * wait() needs to be called to ensure the execution is complete.
     */
    void start(ICPPKernel *kernel, const Window &window, const ThreadInfo &info);

    /** Wait for the current kernel execution to complete. */
    void wait();

    /** Function ran by the worker thread. */
    void worker_thread();

private:
    std::thread             _thread;
    ICPPKernel             *_kernel{ nullptr };
    Window                  _window;
    ThreadInfo              _info;
    std::mutex              _m;
    std::condition_variable _cv;
    bool                    _wait_for_work{ false };
    bool                    _job_complete{ true };
    std::exception_ptr      _current_exception;
};

Thread::Thread()
    : _thread(), _window(), _info(), _m(), _cv(), _current_exception(nullptr)
{
    _thread = std::thread(&Thread::worker_thread, this);
}

Thread::~Thread()
{
    // Make sure worker thread has ended
    if(_thread.joinable())
    {
        start(nullptr, Window(), ThreadInfo());
        _thread.join();
    }
}

void Thread::start(ICPPKernel *kernel, const Window &window, const ThreadInfo &info)
{
    _kernel = kernel;
    _window = window;
    _info   = info;

    {
        std::lock_guard<std::mutex> lock(_m);
        _wait_for_work = true;
        _job_complete  = false;
    }
    _cv.notify_one();
}

void Thread::wait()
{
    {
        std::unique_lock<std::mutex> lock(_m);
        _cv.wait(lock, [&] { return _job_complete; });
    }

    if(_current_exception)
    {
        std::rethrow_exception(_current_exception);
    }
}

void Thread::worker_thread()
{
    while(true)
    {
        std::unique_lock<std::mutex> lock(_m);
        _cv.wait(lock, [&] { return _wait_for_work; });
        _wait_for_work = false;

        _current_exception = nullptr;

        // Time to exit
        if(_kernel == nullptr)
        {
            return;
        }

        try
        {
            _window.validate();
            _kernel->run(_window, _info);
        }
        catch(...)
        {
            _current_exception = std::current_exception();
        }

        _job_complete = true;
        lock.unlock();
        _cv.notify_one();
    }
}

CPPScheduler &CPPScheduler::get()
{
    static CPPScheduler scheduler;
    return scheduler;
}

CPPScheduler::CPPScheduler()
    : _num_threads(std::thread::hardware_concurrency()),
      _threads(_num_threads - 1)
{
}

void CPPScheduler::set_num_threads(unsigned int num_threads)
{
    _num_threads = num_threads == 0 ? std::thread::hardware_concurrency() : num_threads;
    _threads.resize(_num_threads - 1);
}

unsigned int CPPScheduler::num_threads() const
{
    return _num_threads;
}

void CPPScheduler::schedule(ICPPKernel *kernel, unsigned int split_dimension)
{
    ARM_COMPUTE_ERROR_ON_MSG(!kernel, "The child class didn't set the kernel");

    /** [Scheduler example] */
    ThreadInfo info;
    info.cpu_info = _info;

    const Window      &max_window     = kernel->window();
    const unsigned int num_iterations = max_window.num_iterations(split_dimension);
    info.num_threads                  = std::min(num_iterations, _num_threads);

    if(num_iterations == 0)
    {
        return;
    }

    if(!kernel->is_parallelisable() || info.num_threads == 1)
    {
        kernel->run(max_window, info);
    }
    else
    {
        int  t         = 0;
        auto thread_it = _threads.begin();

        for(; t < info.num_threads - 1; ++t, ++thread_it)
        {
            Window win     = max_window.split_window(split_dimension, t, info.num_threads);
            info.thread_id = t;
            thread_it->start(kernel, win, info);
        }

        // Run last part on main thread
        Window win     = max_window.split_window(split_dimension, t, info.num_threads);
        info.thread_id = t;
        kernel->run(win, info);

        try
        {
            for(auto &thread : _threads)
            {
                thread.wait();
            }
        }
        catch(const std::system_error &e)
        {
            std::cerr << "Caught system_error with code " << e.code() << " meaning " << e.what() << '\n';
        }
    }
    /** [Scheduler example] */
}
} // namespace arm_compute
