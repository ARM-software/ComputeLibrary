/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/runtime/Scheduler.h"

#include "arm_compute/core/Error.h"
#if ARM_COMPUTE_CPP_SCHEDULER
#include "arm_compute/runtime/CPP/CPPScheduler.h"
#endif /* ARM_COMPUTE_CPP_SCHEDULER */

#include "arm_compute/runtime/SingleThreadScheduler.h"

#if ARM_COMPUTE_OPENMP_SCHEDULER
#include "arm_compute/runtime/OMP/OMPScheduler.h"
#endif /* ARM_COMPUTE_OPENMP_SCHEDULER */

using namespace arm_compute;

#if !ARM_COMPUTE_CPP_SCHEDULER && ARM_COMPUTE_OPENMP_SCHEDULER
Scheduler::Type Scheduler::_scheduler_type = Scheduler::Type::OMP;
#elif ARM_COMPUTE_CPP_SCHEDULER && !ARM_COMPUTE_OPENMP_SCHEDULER
Scheduler::Type Scheduler::_scheduler_type = Scheduler::Type::CPP;
#elif ARM_COMPUTE_CPP_SCHEDULER && ARM_COMPUTE_OPENMP_SCHEDULER
Scheduler::Type Scheduler::_scheduler_type = Scheduler::Type::CPP;
#else  /* ARM_COMPUTE_*_SCHEDULER */
Scheduler::Type Scheduler::_scheduler_type = Scheduler::Type::ST;
#endif /* ARM_COMPUTE_*_SCHEDULER */

void Scheduler::set(Type t)
{
    ARM_COMPUTE_ERROR_ON(!Scheduler::is_available(t));
    _scheduler_type = t;
}

bool Scheduler::is_available(Type t)
{
    switch(t)
    {
        case Type::ST:
        {
            return true;
        }
        case Type::CPP:
        {
#if ARM_COMPUTE_CPP_SCHEDULER
            return true;
#else  /* ARM_COMPUTE_CPP_SCHEDULER */
            return false;
#endif /* ARM_COMPUTE_CPP_SCHEDULER */
        }
        case Type::OMP:
        {
#if ARM_COMPUTE_OPENMP_SCHEDULER
            return true;
#else  /* ARM_COMPUTE_OPENMP_SCHEDULER */
            return false;
#endif /* ARM_COMPUTE_OPENMP_SCHEDULER */
        }
        case Type::CUSTOM:
        {
            return _custom_scheduler != nullptr;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Invalid Scheduler type");
            return false;
        }
    }
}

Scheduler::Type Scheduler::get_type()
{
    return _scheduler_type;
}

IScheduler &Scheduler::get()
{
    switch(_scheduler_type)
    {
        case Type::ST:
        {
            return SingleThreadScheduler::get();
        }
        case Type::CPP:
        {
#if ARM_COMPUTE_CPP_SCHEDULER
            return CPPScheduler::get();
#else  /* ARM_COMPUTE_CPP_SCHEDULER */
            ARM_COMPUTE_ERROR("Recompile with cppthreads=1 to use C++11 scheduler.");
#endif /* ARM_COMPUTE_CPP_SCHEDULER */
            break;
        }
        case Type::OMP:
        {
#if ARM_COMPUTE_OPENMP_SCHEDULER
            return OMPScheduler::get();
#else  /* ARM_COMPUTE_OPENMP_SCHEDULER */
            ARM_COMPUTE_ERROR("Recompile with openmp=1 to use openmp scheduler.");
#endif /* ARM_COMPUTE_OPENMP_SCHEDULER */
            break;
        }
        case Type::CUSTOM:
        {
            if(_custom_scheduler == nullptr)
            {
                ARM_COMPUTE_ERROR("No custom scheduler has been setup. Call set(std::shared_ptr<IScheduler> &scheduler) before Scheduler::get()");
            }
            else
            {
                return *_custom_scheduler;
            }
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Invalid Scheduler type");
            break;
        }
    }
    return SingleThreadScheduler::get();
}

std::shared_ptr<IScheduler> Scheduler::_custom_scheduler = nullptr;

void Scheduler::set(std::shared_ptr<IScheduler> &scheduler)
{
    _custom_scheduler = scheduler;
    set(Type::CUSTOM);
}
