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
#include "src/gpu/cl/ClQueue.h"

#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTuner.h"

namespace arm_compute
{
namespace gpu
{
namespace opencl
{
namespace
{
CLTunerMode map_tuner_mode(AclTuningMode mode)
{
    switch(mode)
    {
        case AclRapid:
            return CLTunerMode::RAPID;
            break;
        case AclNormal:
            return CLTunerMode::NORMAL;
            break;
        case AclExhaustive:
            return CLTunerMode::EXHAUSTIVE;
            break;
        default:
            ARM_COMPUTE_ERROR("Invalid tuner mode");
            break;
    }
}

std::unique_ptr<CLTuner> populate_tuner(const AclQueueOptions *options)
{
    if(options == nullptr || options->mode == AclTuningModeNone)
    {
        return nullptr;
    }

    CLTuningInfo tune_info;
    tune_info.tuner_mode = map_tuner_mode(options->mode);
    tune_info.tune_wbsm  = false;

    return std::make_unique<CLTuner>(true /* tune_new_kernels */, tune_info);
}
} // namespace

ClQueue::ClQueue(IContext *ctx, const AclQueueOptions *options)
    : IQueue(ctx), _tuner(nullptr)
{
    _tuner = populate_tuner(options);
}

arm_compute::CLScheduler &ClQueue::scheduler()
{
    return arm_compute::CLScheduler::get();
}

::cl::CommandQueue ClQueue::cl_queue()
{
    return arm_compute::CLScheduler::get().queue();
}

bool ClQueue::set_cl_queue(::cl::CommandQueue queue)
{
    // TODO: Check queue is from the same context
    arm_compute::CLScheduler::get().set_queue(queue);
    return true;
}

StatusCode ClQueue::finish()
{
    arm_compute::CLScheduler::get().queue().finish();
    return StatusCode::Success;
}

} // namespace opencl
} // namespace gpu
} // namespace arm_compute
