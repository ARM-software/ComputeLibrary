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
#ifndef SRC_GPU_CLQUEUE_H
#define SRC_GPU_CLQUEUE_H

#include "src/common/IQueue.h"

#include "arm_compute/runtime/CL/CLScheduler.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class CLTuner;

namespace gpu
{
namespace opencl
{
/** OpenCL queue implementation class */
class ClQueue final : public IQueue
{
public:
    /** Construct a new CpuQueue object
     *
     * @param[in] ctx     Context to be used
     * @param[in] options Command queue options
     */
    ClQueue(IContext *ctx, const AclQueueOptions *options);

    /** Return legacy scheduler
     *
     * @return arm_compute::IScheduler&
     */
    arm_compute::CLScheduler &scheduler();

    /** Underlying cl command queue accessor
     *
     * @return the cl command queue used
     */
    ::cl::CommandQueue cl_queue();

    /** Update/inject an underlying cl command queue object
     *
     * @warning Command queue needs to come from the same context as the AclQueue
     *
     * @param[in] queue Underlying cl command queue to be used
     *
     * @return true if the queue was set successfully else falseS
     */
    bool set_cl_queue(::cl::CommandQueue queue);

    // Inherited functions overridden
    StatusCode finish() override;

private:
    std::unique_ptr<CLTuner> _tuner;
};
} // namespace opencl
} // namespace gpu
} // namespace arm_compute
#endif /* SRC_GPU_CLQUEUE_H */
