/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef SRC_CPU_CPUQUEUE_H
#define SRC_CPU_CPUQUEUE_H

#include "src/common/IQueue.h"

#include "arm_compute/runtime/IScheduler.h"

namespace arm_compute
{
namespace cpu
{
/** CPU queue implementation class */
class CpuQueue final : public IQueue
{
public:
    /** Construct a new CpuQueue object
     *
     * @param[in] ctx     Context to be used
     * @param[in] options Command queue options
     */
    CpuQueue(IContext *ctx, const AclQueueOptions *options);
    /** Return legacy scheduler
     *
     * @return arm_compute::IScheduler&
     */
    arm_compute::IScheduler &scheduler();

    // Inherited functions overridden
    StatusCode finish() override;
};
} // namespace cpu
} // namespace arm_compute
#endif /* SRC_CPU_CPUQUEUE_H */
