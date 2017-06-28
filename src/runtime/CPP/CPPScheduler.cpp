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

#include <iostream>
#include <semaphore.h>
#include <system_error>
#include <thread>

using namespace arm_compute;

class arm_compute::Thread
{
public:
    /** Start a new thread
     */
    Thread();
    Thread(const Thread &) = delete;
    Thread &operator=(const Thread &) = delete;
    Thread(Thread &&)                 = delete;
    Thread &operator=(Thread &&) = delete;
    /** Make the thread join
     */
    ~Thread();
    /** Request the worker thread to start executing the given kernel
     * This function will return as soon as the kernel has been sent to the worker thread.
     * wait() needs to be called to ensure the execution is complete.
     */
    void start(ICPPKernel *kernel, const Window &window);
    /** Wait for the current kernel execution to complete
     */
    void wait();
    /** Function ran by the worker thread
     */
    void worker_thread();

private:
    std::thread        _thread;
    ICPPKernel        *_kernel{ nullptr };
    Window             _window;
    sem_t              _wait_for_work;
    sem_t              _job_complete;
    std::exception_ptr _current_exception;
};

Thread::Thread()
    : _thread(), _window(), _wait_for_work(), _job_complete(), _current_exception(nullptr)
{
    int ret = sem_init(&_wait_for_work, 0, 0);
    ARM_COMPUTE_ERROR_ON(ret < 0);
    ARM_COMPUTE_UNUSED(ret);

    ret = sem_init(&_job_complete, 0, 0);
    ARM_COMPUTE_ERROR_ON(ret < 0);
    ARM_COMPUTE_UNUSED(ret);

    _thread = std::thread(&Thread::worker_thread, this);
}

Thread::~Thread()
{
    ARM_COMPUTE_ERROR_ON(!_thread.joinable());

    start(nullptr, Window());
    _thread.join();

    int ret = sem_destroy(&_wait_for_work);
    ARM_COMPUTE_ERROR_ON(ret < 0);
    ARM_COMPUTE_UNUSED(ret);

    ret = sem_destroy(&_job_complete);
    ARM_COMPUTE_ERROR_ON(ret < 0);
    ARM_COMPUTE_UNUSED(ret);
}

void Thread::start(ICPPKernel *kernel, const Window &window)
{
    _kernel = kernel;
    _window = window;
    int ret = sem_post(&_wait_for_work);
    ARM_COMPUTE_UNUSED(ret);
    ARM_COMPUTE_ERROR_ON(ret < 0);
}

void Thread::wait()
{
    int ret = sem_wait(&_job_complete);
    ARM_COMPUTE_UNUSED(ret);
    ARM_COMPUTE_ERROR_ON(ret < 0);
    if(_current_exception)
    {
        std::rethrow_exception(_current_exception);
    }
}

void Thread::worker_thread()
{
    while(sem_wait(&_wait_for_work) >= 0)
    {
        _current_exception = nullptr;
        // Time to exit
        if(_kernel == nullptr)
        {
            return;
        }

        try
        {
            _window.validate();
            _kernel->run(_window);
        }
        catch(...)
        {
            _current_exception = std::current_exception();
        }
        int ret = sem_post(&_job_complete);
        ARM_COMPUTE_UNUSED(ret);
        ARM_COMPUTE_ERROR_ON(ret < 0);
    }

    ARM_COMPUTE_ERROR("Wait failed");
}

namespace
{
void delete_threads(Thread *t)
{
    delete[] t;
}
} // namespace

CPPScheduler &CPPScheduler::get()
{
    static CPPScheduler scheduler;
    return scheduler;
}

CPPScheduler::CPPScheduler()
    : _num_threads(std::thread::hardware_concurrency()),
      _threads(std::unique_ptr<Thread[], void(*)(Thread *)>(new Thread[std::thread::hardware_concurrency() - 1], delete_threads)),
      _target(CPUTarget::INTRINSICS)
{
}

void CPPScheduler::set_num_threads(unsigned int num_threads)
{
    const unsigned int num_cores = std::thread::hardware_concurrency();
    _num_threads                 = num_threads == 0 ? num_cores : num_threads;
}

unsigned int CPPScheduler::num_threads() const
{
    return _num_threads;
}

void CPPScheduler::set_target(CPUTarget target)
{
    _target = target;
}

CPUTarget CPPScheduler::target() const
{
    return _target;
}

void CPPScheduler::schedule(ICPPKernel *kernel, unsigned int split_dimension)
{
    ARM_COMPUTE_ERROR_ON_MSG(!kernel, "The child class didn't set the kernel");

    /** [Scheduler example] */
    const Window      &max_window     = kernel->window();
    const unsigned int num_iterations = max_window.num_iterations(split_dimension);
    const unsigned int num_threads    = std::min(num_iterations, _num_threads);

    if(!kernel->is_parallelisable() || 1 == num_threads)
    {
        kernel->run(max_window);
    }
    else
    {
        for(unsigned int t = 0; t < num_threads; ++t)
        {
            Window win = max_window.split_window(split_dimension, t, num_threads);
            win.set_thread_id(t);
            win.set_num_threads(num_threads);

            if(t != num_threads - 1)
            {
                _threads[t].start(kernel, win);
            }
            else
            {
                kernel->run(win);
            }
        }

        try
        {
            for(unsigned int t = 1; t < num_threads; ++t)
            {
                _threads[t - 1].wait();
            }
        }
        catch(const std::system_error &e)
        {
            std::cout << "Caught system_error with code " << e.code() << " meaning " << e.what() << '\n';
        }
    }
    /** [Scheduler example] */
}
