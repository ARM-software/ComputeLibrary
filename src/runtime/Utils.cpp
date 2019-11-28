/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/runtime/Utils.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"

#include <map>
#include <string>

namespace arm_compute
{
#ifndef DOXYGEN_SKIP_THIS
static const std::string information =
#include "arm_compute_version.embed"
    ;
#endif /* DOXYGEN_SKIP_THIS */

const std::string &string_from_scheduler_type(Scheduler::Type t)
{
    static std::map<Scheduler::Type, const std::string> scheduler_type_map =
    {
        { Scheduler::Type::ST, "Single Thread" },
        { Scheduler::Type::CPP, "C++11 Threads" },
        { Scheduler::Type::OMP, "OpenMP Threads" },
        { Scheduler::Type::CUSTOM, "Custom" }
    };

    return scheduler_type_map[t];
}

void schedule_kernel_on_ctx(IRuntimeContext *ctx, ICPPKernel *kernel, const IScheduler::Hints &hints)
{
    if(ctx)
    {
        ARM_COMPUTE_ERROR_ON(ctx->scheduler() == nullptr);
        ctx->scheduler()->schedule(kernel, hints);
    }
    else
    {
        NEScheduler::get().schedule(kernel, hints);
    }
}
} // namespace arm_compute
