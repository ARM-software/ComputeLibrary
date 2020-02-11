/*
 * Copyright (c) 2016-2020 ARM Limited.
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
#include "support/Mutex.h"

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <list>
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

/** Given two dimensions and a maxium number of threads to utilise, calcualte the best
 * combination of threads that fit in (mutliplied together) max_threads.
 *
 * This algorithm assumes that work in either of the dimensions is equally difficult
 * to compute
 *
 * @returns [m_nthreads, n_nthreads] A pair of the threads that should be used in each dimension
 */
std::pair<unsigned, unsigned> split_2d(unsigned max_threads, std::size_t m, std::size_t n)
{
    /*
     * We want the same ratio of threads in M & N to the ratio of m and n problem size
     *
     * Therefore:    mt/nt == m/n    where mt*nt == max_threads
     *
     *             max_threads/nt = mt    &    (max_threads/nt) * (m/n) = nt
     *          nt^2 = max_threads * (m/n)
     *          nt = sqrt( max_threads * (m/n) )
     */
    //ratio of m to n in problem dimensions
    double ratio = m / static_cast<double>(n);

    // nt = sqrt(max_threads * (m / n) )
    const unsigned adjusted = std::round(
                    std::sqrt(max_threads * ratio));

    //find the nearest factor of max_threads
    for(unsigned i = 0; i!= adjusted; ++i)
    {
        //try down
        const unsigned adj_down = adjusted - i;
        if(max_threads % adj_down == 0)
        {
            return { adj_down, max_threads / adj_down };
        }

        //try up
        const unsigned adj_up = adjusted + i;
        if(max_threads % adj_up == 0)
        {
            return { adj_up, max_threads / adj_up };
        }
    }

    //we didn't find anything so lets bail out with maxes biased to the largest dimension
    if(m > n)
    {
         return{ std::min<unsigned>(m, max_threads), 1 };
    }
    else
    {
        return{ 1, std::min<unsigned>(n, max_threads) };
    }
}

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

struct CPPScheduler::Impl final
{
    explicit Impl(unsigned int thread_hint)
        : _num_threads(thread_hint), _threads(_num_threads - 1)
    {
    }
    void set_num_threads(unsigned int num_threads, unsigned int thead_hint)
    {
        _num_threads = num_threads == 0 ? thead_hint : num_threads;
        _threads.resize(_num_threads - 1);
    }
    unsigned int num_threads() const
    {
        return _num_threads;
    }

    void run_workloads(std::vector<IScheduler::Workload> &workloads);

    class Thread;

    unsigned int       _num_threads;
    std::list<Thread>  _threads;
    arm_compute::Mutex _run_workloads_mutex{};
};

class CPPScheduler::Impl::Thread final
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

CPPScheduler::Impl::Thread::Thread()
{
    _thread = std::thread(&Thread::worker_thread, this);
}

CPPScheduler::Impl::Thread::~Thread()
{
    // Make sure worker thread has ended
    if(_thread.joinable())
    {
        ThreadFeeder feeder;
        start(nullptr, feeder, ThreadInfo());
        _thread.join();
    }
}

void CPPScheduler::Impl::Thread::start(std::vector<IScheduler::Workload> *workloads, ThreadFeeder &feeder, const ThreadInfo &info)
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

void CPPScheduler::Impl::Thread::wait()
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

void CPPScheduler::Impl::Thread::worker_thread()
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

/*
 * This singleton has been deprecated and will be removed in the next release
 */
CPPScheduler &CPPScheduler::get()
{
    static CPPScheduler scheduler;
    return scheduler;
}

CPPScheduler::CPPScheduler()
    : _impl(support::cpp14::make_unique<Impl>(num_threads_hint()))
{
}

CPPScheduler::~CPPScheduler() = default;

void CPPScheduler::set_num_threads(unsigned int num_threads)
{
    // No changes in the number of threads while current workloads are running
    arm_compute::lock_guard<std::mutex> lock(_impl->_run_workloads_mutex);
    _impl->set_num_threads(num_threads, num_threads_hint());
}

unsigned int CPPScheduler::num_threads() const
{
    return _impl->num_threads();
}

#ifndef DOXYGEN_SKIP_THIS
void CPPScheduler::run_workloads(std::vector<IScheduler::Workload> &workloads)
{
    // Mutex to ensure other threads won't interfere with the setup of the current thread's workloads
    // Other thread's workloads will be scheduled after the current thread's workloads have finished
    // This is not great because different threads workloads won't run in parallel but at least they
    // won't interfere each other and deadlock.
    arm_compute::lock_guard<std::mutex> lock(_impl->_run_workloads_mutex);
    const unsigned int                  num_threads = std::min(_impl->num_threads(), static_cast<unsigned int>(workloads.size()));
    if(num_threads < 1)
    {
        return;
    }
    ThreadFeeder feeder(num_threads, workloads.size());
    ThreadInfo   info;
    info.cpu_info          = &_cpu_info;
    info.num_threads       = num_threads;
    unsigned int t         = 0;
    auto         thread_it = _impl->_threads.begin();
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
        for(auto &thread : _impl->_threads)
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

    if(hints.split_dimension() == IScheduler::split_dimensions_all)
    {
        /*
         * if the split dim is size_t max then this signals we should parallelise over
         * all dimensions
         */
        const std::size_t m = max_window.num_iterations(Window::DimX);
        const std::size_t n = max_window.num_iterations(Window::DimY);

       //in c++17 this can be swapped for   auto [ m_threads, n_threads ] = split_2d(...
        unsigned m_threads, n_threads;
        std::tie(m_threads, n_threads) = split_2d(_impl->_num_threads, m, n);

        std::vector<IScheduler::Workload> workloads;
        for(unsigned int ni  = 0; ni != n_threads; ++ni)
        {
            for(unsigned int mi  = 0; mi != m_threads; ++mi)
            {
                workloads.push_back(
                    [ ni, mi, m_threads, n_threads, &max_window, &kernel ]
                    (const ThreadInfo & info)
                    {
                        //narrow the window to our mi-ni workload
                        Window win = max_window.split_window(Window::DimX, mi, m_threads)
                                               .split_window(Window::DimY, ni, n_threads);

                        win.validate();

                        Window thread_locator;
                        thread_locator.set(Window::DimX, Window::Dimension(mi, m_threads));
                        thread_locator.set(Window::DimY, Window::Dimension(ni, n_threads));

                        thread_locator.validate();

                        kernel->run_nd(win, info, thread_locator);
                    }
                );
            }
        }
        run_workloads(workloads);
    }
    else
    {
        const unsigned int num_iterations = max_window.num_iterations(hints.split_dimension());
        const unsigned int num_threads    = std::min(num_iterations, _impl->_num_threads);

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
                    const unsigned int granule_threshold = (hints.threshold() <= 0) ? num_threads : static_cast<unsigned int>(hints.threshold());
                    // Make sure we don't use some windows which are too small as this might create some contention on the ThreadFeeder
                    num_windows = num_iterations > granule_threshold ? granule_threshold : num_iterations;
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
}
} // namespace arm_compute
