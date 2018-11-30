/*
 * Copyright (c) 2016-2018 ARM Limited.
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
#include "arm_compute/runtime/CPUUtils.h"

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <system_error>
#include <thread>

namespace arm_compute
{
namespace
{
class ThreadFeeder
{
public:
    /** Constructor
     *
     * @param[in] start First value that will be returned by the feeder
     * @param[in] end   End condition (The last value returned by get_next() will be end - 1)
     */
    explicit ThreadFeeder(unsigned int start = 0, unsigned int end = 0)
        : _atomic_counter(start), _end(end)
    {
    }
    /** Return the next element in the range if there is one.
     *
     * @param[out] next Will contain the next element if there is one.
     *
     * @return False if the end of the range has been reached and next wasn't set.
     */
    bool get_next(unsigned int &next)
    {
        next = atomic_fetch_add_explicit(&_atomic_counter, 1u, std::memory_order_relaxed);
        return next < _end;
    }

private:
    std::atomic_uint   _atomic_counter;
    const unsigned int _end;
};

/** Execute workloads[info.thread_id] first, then call the feeder to get the index of the next workload to run.
 *
 * Will run workloads until the feeder reaches the end of its range.
 *
 * @param[in]     workloads The array of workloads
 * @param[in,out] feeder    The feeder indicating which workload to execute next.
 * @param[in]     info      Threading and CPU info.
 */
void process_workloads(std::vector<IScheduler::Workload> &workloads, ThreadFeeder &feeder, const ThreadInfo &info)
{
    unsigned int workload_index = info.thread_id;
    do
    {
        ARM_COMPUTE_ERROR_ON(workload_index >= workloads.size());
        workloads[workload_index](info);
    }
    while(feeder.get_next(workload_index));
}

} //namespace

class CPPScheduler::Thread
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

    /** Request the worker thread to start executing workloads.
     *
     * The thread will start by executing workloads[info.thread_id] and will then call the feeder to
     * get the index of the following workload to run.
     *
     * @note This function will return as soon as the workloads have been sent to the worker thread.
     * wait() needs to be called to ensure the execution is complete.
     */
    void start(std::vector<IScheduler::Workload> *workloads, ThreadFeeder &feeder, const ThreadInfo &info);

    /** Wait for the current kernel execution to complete. */
    void wait();

    /** Function ran by the worker thread. */
    void worker_thread();

private:
    std::thread                        _thread{};
    ThreadInfo                         _info{};
    std::vector<IScheduler::Workload> *_workloads{ nullptr };
    ThreadFeeder                      *_feeder{ nullptr };
    std::mutex                         _m{};
    std::condition_variable            _cv{};
    bool                               _wait_for_work{ false };
    bool                               _job_complete{ true };
    std::exception_ptr                 _current_exception{ nullptr };
};

CPPScheduler::Thread::Thread()
{
    _thread = std::thread(&Thread::worker_thread, this);
}

CPPScheduler::Thread::~Thread()
{
    // Make sure worker thread has ended
    if(_thread.joinable())
    {
        ThreadFeeder feeder;
        start(nullptr, feeder, ThreadInfo());
        _thread.join();
    }
}

void CPPScheduler::Thread::start(std::vector<IScheduler::Workload> *workloads, ThreadFeeder &feeder, const ThreadInfo &info)
{
    _workloads = workloads;
    _feeder    = &feeder;
    _info      = info;
    {
        std::lock_guard<std::mutex> lock(_m);
        _wait_for_work = true;
        _job_complete  = false;
    }
    _cv.notify_one();
}

void CPPScheduler::Thread::wait()
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

void CPPScheduler::Thread::worker_thread()
{
    while(true)
    {
        std::unique_lock<std::mutex> lock(_m);
        _cv.wait(lock, [&] { return _wait_for_work; });
        _wait_for_work = false;

        _current_exception = nullptr;

        // Time to exit
        if(_workloads == nullptr)
        {
            return;
        }

#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
        try
        {
#endif /* ARM_COMPUTE_EXCEPTIONS_ENABLED */
            process_workloads(*_workloads, *_feeder, _info);

#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
        }
        catch(...)
        {
            _current_exception = std::current_exception();
        }
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
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
    : _num_threads(num_threads_hint()),
      _threads(_num_threads - 1)
{
}

void CPPScheduler::set_num_threads(unsigned int num_threads)
{
    _num_threads = num_threads == 0 ? num_threads_hint() : num_threads;
    _threads.resize(_num_threads - 1);
}

unsigned int CPPScheduler::num_threads() const
{
    return _num_threads;
}

#ifndef DOXYGEN_SKIP_THIS
void CPPScheduler::run_workloads(std::vector<IScheduler::Workload> &workloads)
{
    const unsigned int num_threads = std::min(_num_threads, static_cast<unsigned int>(workloads.size()));
    if(num_threads < 1)
    {
        return;
    }
    ThreadFeeder feeder(num_threads, workloads.size());
    ThreadInfo   info;
    info.cpu_info          = &_cpu_info;
    info.num_threads       = num_threads;
    unsigned int t         = 0;
    auto         thread_it = _threads.begin();
    for(; t < num_threads - 1; ++t, ++thread_it)
    {
        info.thread_id = t;
        thread_it->start(&workloads, feeder, info);
    }

    info.thread_id = t;
    process_workloads(workloads, feeder, info);
#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    try
    {
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
        for(auto &thread : _threads)
        {
            thread.wait();
        }
#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    }
    catch(const std::system_error &e)
    {
        std::cerr << "Caught system_error with code " << e.code() << " meaning " << e.what() << '\n';
    }
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
}
#endif /* DOXYGEN_SKIP_THIS */

void CPPScheduler::schedule(ICPPKernel *kernel, const Hints &hints)
{
    ARM_COMPUTE_ERROR_ON_MSG(!kernel, "The child class didn't set the kernel");

    const Window      &max_window     = kernel->window();
    const unsigned int num_iterations = max_window.num_iterations(hints.split_dimension());
    const unsigned int num_threads    = std::min(num_iterations, _num_threads);

    if(num_iterations == 0)
    {
        return;
    }

    if(!kernel->is_parallelisable() || num_threads == 1)
    {
        ThreadInfo info;
        info.cpu_info = &_cpu_info;
        kernel->run(max_window, info);
    }
    else
    {
        unsigned int num_windows = 0;
        switch(hints.strategy())
        {
            case StrategyHint::STATIC:
                num_windows = num_threads;
                break;
            case StrategyHint::DYNAMIC:
            {
                // Make sure we don't use some windows which are too small as this might create some contention on the ThreadFeeder
                const unsigned int max_iterations = static_cast<unsigned int>(_num_threads) * 3;
                num_windows                       = num_iterations > max_iterations ? max_iterations : num_iterations;
                break;
            }
            default:
                ARM_COMPUTE_ERROR("Unknown strategy");
        }
        std::vector<IScheduler::Workload> workloads(num_windows);
        for(unsigned int t = 0; t < num_windows; t++)
        {
            //Capture 't' by copy, all the other variables by reference:
            workloads[t] = [t, &hints, &max_window, &num_windows, &kernel](const ThreadInfo & info)
            {
                Window win = max_window.split_window(hints.split_dimension(), t, num_windows);
                win.validate();
                kernel->run(win, info);
            };
        }
        run_workloads(workloads);
    }
}
} // namespace arm_compute
