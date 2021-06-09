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
#ifndef ARM_COMPUTE_RUNTIME_CONTEXT_H
#define ARM_COMPUTE_RUNTIME_CONTEXT_H

#include "arm_compute/runtime/IRuntimeContext.h"

#include <memory>

namespace arm_compute
{
/** Runtime context */
class RuntimeContext : public IRuntimeContext
{
public:
    /** Default Constructor */
    RuntimeContext();
    /** Destructor */
    ~RuntimeContext() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    RuntimeContext(const RuntimeContext &) = delete;
    /** Default move constructor */
    RuntimeContext(RuntimeContext &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    RuntimeContext &operator=(const RuntimeContext &) = delete;
    /** Default move assignment operator */
    RuntimeContext &operator=(RuntimeContext &&) = default;
    /** CPU Scheduler setter */
    void set_scheduler(IScheduler *scheduler);

    // Inherited overridden methods
    IScheduler    *scheduler() override;
    IAssetManager *asset_manager() override;

private:
    std::unique_ptr<IScheduler> _owned_scheduler{ nullptr };
    IScheduler                 *_scheduler{ nullptr };
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_RUNTIME_CONTEXT_H */
