/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "src/runtime/Utils.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"

#include <cmath>
#include <map>
#include <string>

namespace arm_compute
{
namespace utils
{
#ifndef DOXYGEN_SKIP_THIS
static const std::string information =
#include "arm_compute_version.embed"
    ;
#endif /* DOXYGEN_SKIP_THIS */

const std::string &string_from_scheduler_type(Scheduler::Type t)
{
    static std::map<Scheduler::Type, const std::string> scheduler_type_map = {{Scheduler::Type::ST, "Single Thread"},
                                                                              {Scheduler::Type::CPP, "C++11 Threads"},
                                                                              {Scheduler::Type::OMP, "OpenMP Threads"},
                                                                              {Scheduler::Type::CUSTOM, "Custom"}};

    return scheduler_type_map[t];
}

void schedule_kernel_on_ctx(IRuntimeContext *ctx, ICPPKernel *kernel, const IScheduler::Hints &hints)
{
    if (ctx)
    {
        ARM_COMPUTE_ERROR_ON(ctx->scheduler() == nullptr);
        ctx->scheduler()->schedule(kernel, hints);
    }
    else
    {
        NEScheduler::get().schedule(kernel, hints);
    }
}

unsigned int calculate_number_of_stages_only_x_axis(size_t input_x_dimension, unsigned int axis)
{
    // We need only 1 stage for all axis except x-axis
    if (axis != 0)
    {
        return 1;
    }
    // Calculate number of WGs. 16 elements per thread, 8 threads per WG
    const auto num_of_wg = static_cast<unsigned int>(ceil(input_x_dimension / 128.f));

    // Calculate number of stages. First stage performs op and the rest reduction sum
    // depending on the size of the input. Last stage should have only 1 WG.
    const unsigned int num_of_stages = num_of_wg / 128 + 2;
    return num_of_stages;
}
} // namespace utils
} // namespace arm_compute
