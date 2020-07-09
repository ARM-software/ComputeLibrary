/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#include "arm_compute/runtime/SchedulerFactory.h"

#include "support/MemorySupport.h"

#include "arm_compute/core/Error.h"
#if ARM_COMPUTE_CPP_SCHEDULER
#include "arm_compute/runtime/CPP/CPPScheduler.h"
#endif /* ARM_COMPUTE_CPP_SCHEDULER */

#include "arm_compute/runtime/SingleThreadScheduler.h"

#if ARM_COMPUTE_OPENMP_SCHEDULER
#include "arm_compute/runtime/OMP/OMPScheduler.h"
#endif /* ARM_COMPUTE_OPENMP_SCHEDULER */

namespace arm_compute
{
#if !ARM_COMPUTE_CPP_SCHEDULER && ARM_COMPUTE_OPENMP_SCHEDULER
const SchedulerFactory::Type SchedulerFactory::_default_type = SchedulerFactory::Type::OMP;
#elif ARM_COMPUTE_CPP_SCHEDULER && !ARM_COMPUTE_OPENMP_SCHEDULER
const SchedulerFactory::Type SchedulerFactory::_default_type = SchedulerFactory::Type::CPP;
#elif ARM_COMPUTE_CPP_SCHEDULER && ARM_COMPUTE_OPENMP_SCHEDULER
const SchedulerFactory::Type SchedulerFactory::_default_type = SchedulerFactory::Type::CPP;
#else  /* ARM_COMPUTE_*_SCHEDULER */
const SchedulerFactory::Type SchedulerFactory::_default_type = SchedulerFactory::Type::ST;
#endif /* ARM_COMPUTE_*_SCHEDULER */

std::unique_ptr<IScheduler> SchedulerFactory::create(Type type)
{
    switch(type)
    {
        case Type::ST:
        {
            return support::cpp14::make_unique<SingleThreadScheduler>();
        }
        case Type::CPP:
        {
#if ARM_COMPUTE_CPP_SCHEDULER
            return support::cpp14::make_unique<CPPScheduler>();
#else  /* ARM_COMPUTE_CPP_SCHEDULER */
            ARM_COMPUTE_ERROR("Recompile with cppthreads=1 to use C++11 scheduler.");
#endif /* ARM_COMPUTE_CPP_SCHEDULER */
        }
        case Type::OMP:
        {
#if ARM_COMPUTE_OPENMP_SCHEDULER
            return support::cpp14::make_unique<OMPScheduler>();
#else  /* ARM_COMPUTE_OPENMP_SCHEDULER */
            ARM_COMPUTE_ERROR("Recompile with openmp=1 to use openmp scheduler.");
#endif /* ARM_COMPUTE_OPENMP_SCHEDULER */
        }
        default:
        {
            break;
        }
    }
    ARM_COMPUTE_ERROR("Invalid Scheduler type");
}
} // namespace arm_compute
