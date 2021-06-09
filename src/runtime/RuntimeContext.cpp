/*
 * Copyright (c) 2019, 2021 Arm Limited.
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
#include "arm_compute/runtime/RuntimeContext.h"

#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/SchedulerFactory.h"

namespace arm_compute
{
RuntimeContext::RuntimeContext()
    : _owned_scheduler(SchedulerFactory::create()), _scheduler(_owned_scheduler.get())
{
}

void RuntimeContext::set_scheduler(IScheduler *scheduler)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(scheduler);
    _scheduler = scheduler;
}

IScheduler *RuntimeContext::scheduler()
{
    return _scheduler;
}

IAssetManager *RuntimeContext::asset_manager()
{
    return nullptr;
}
} // namespace arm_compute
