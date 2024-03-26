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
#ifndef SRC_RUNTIME_UTILS_H
#define SRC_RUNTIME_UTILS_H

#include "arm_compute/runtime/IRuntimeContext.h"
#include "arm_compute/runtime/Scheduler.h"

#include <string>

namespace arm_compute
{
namespace utils
{
/** Convert a Scheduler::Type into a string.
 *
 * @param[in] t @ref Scheduler::Type to be translated to string.
 *
 * @return The string describing the scheduler type.
 */
const std::string &string_from_scheduler_type(Scheduler::Type t);

/** Schedules a kernel using the context if not nullptr else uses the legacy scheduling flow.
 *
 * @param[in] ctx    Context to use.
 * @param[in] kernel Kernel to schedule.
 * @param[in] hints  Hints to use.
 */
void schedule_kernel_on_ctx(IRuntimeContext *ctx, ICPPKernel *kernel, const IScheduler::Hints &hints);

/** Calculate number of stages for parallel implementations
 *
 * @param[in] input_x_dimension input tensor x dimension
 * @param[in] axis              axis to be used
 */
unsigned int calculate_number_of_stages_only_x_axis(size_t input_x_dimension, unsigned int axis);
} // namespace utils
} // namespace arm_compute
#endif /* SRC_RUNTIME_UTILS_H */
