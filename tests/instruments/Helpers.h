/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_INSTRUMENTS_UTILS_H
#define ARM_COMPUTE_TEST_INSTRUMENTS_UTILS_H

#include "arm_compute/runtime/RuntimeContext.h"
#include "tests/framework/instruments/Instruments.h"

namespace arm_compute
{
namespace test
{
class ContextSchedulerUser : public framework::ISchedulerUser
{
public:
    /** Default Constructor
     *
     * @param[in] ctx Runtime context to track
     */
    ContextSchedulerUser(RuntimeContext *ctx)
        : _ctx(ctx), _scheduler_to_use(nullptr), _real_scheduler(nullptr), _interceptor(nullptr)
    {
        ARM_COMPUTE_ERROR_ON(ctx == nullptr);
        _real_scheduler   = _ctx->scheduler();
        _scheduler_to_use = _real_scheduler;
    }
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    ContextSchedulerUser(const ContextSchedulerUser &) = delete;
    /** Default move constructor */
    ContextSchedulerUser(ContextSchedulerUser &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    ContextSchedulerUser &operator=(const ContextSchedulerUser &) = delete;
    /** Default move assignment operator */
    ContextSchedulerUser &operator=(ContextSchedulerUser &&) = default;

    // Overridden inherited methods
    void intercept_scheduler(std::unique_ptr<IScheduler> interceptor)
    {
        if(interceptor != nullptr)
        {
            _interceptor      = std::move(interceptor);
            _scheduler_to_use = _interceptor.get();
            _ctx->set_scheduler(_scheduler_to_use);
        }
    }
    void restore_scheduler()
    {
        _interceptor      = nullptr;
        _scheduler_to_use = _real_scheduler;
        _ctx->set_scheduler(_scheduler_to_use);
    }
    IScheduler *scheduler()
    {
        return _real_scheduler;
    }

private:
    RuntimeContext             *_ctx;
    IScheduler                 *_scheduler_to_use;
    IScheduler                 *_real_scheduler;
    std::unique_ptr<IScheduler> _interceptor;
};
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_INSTRUMENTS_UTILS_H */
