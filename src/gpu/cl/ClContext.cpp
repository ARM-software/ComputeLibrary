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
#include "src/gpu/cl/ClContext.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"

#include "src/gpu/cl/ClQueue.h"
#include "src/gpu/cl/ClTensor.h"

namespace arm_compute
{
namespace gpu
{
namespace opencl
{
namespace
{
mlgo::MLGOHeuristics populate_mlgo(const char *filename)
{
    bool                 status = false;
    mlgo::MLGOHeuristics heuristics;

    if (filename != nullptr)
    {
        status = heuristics.reload_from_file(filename);
    }
    return status ? std::move(heuristics) : mlgo::MLGOHeuristics();
}
} // namespace

ClContext::ClContext(const AclContextOptions *options)
    : IContext(Target::GpuOcl), _mlgo_heuristics(), _cl_ctx(), _cl_dev()
{
    if (options != nullptr)
    {
        _mlgo_heuristics = populate_mlgo(options->kernel_config_file);
    }
    _cl_ctx = CLKernelLibrary::get().context();
    _cl_dev = CLKernelLibrary::get().get_device();
}

const mlgo::MLGOHeuristics &ClContext::mlgo() const
{
    return _mlgo_heuristics;
}

::cl::Context ClContext::cl_ctx()
{
    return _cl_ctx;
}

::cl::Device ClContext::cl_dev()
{
    return _cl_dev;
}

bool ClContext::set_cl_ctx(::cl::Context ctx)
{
    if (this->refcount() == 0)
    {
        _cl_ctx = ctx;
        CLScheduler::get().set_context(ctx);
        return true;
    }
    return false;
}

ITensorV2 *ClContext::create_tensor(const AclTensorDescriptor &desc, bool allocate)
{
    ClTensor *tensor = new ClTensor(this, desc);
    if (tensor != nullptr && allocate)
    {
        tensor->allocate();
    }
    return tensor;
}

IQueue *ClContext::create_queue(const AclQueueOptions *options)
{
    return new ClQueue(this, options);
}
} // namespace opencl
} // namespace gpu
} // namespace arm_compute
