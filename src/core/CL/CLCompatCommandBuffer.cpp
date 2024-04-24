/*
 * Copyright (c) 2023 Arm Limited.
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

#include "src/core/CL/CLCompatCommandBuffer.h"

#include "arm_compute/core/Error.h"

#include "src/core/CL/CLUtils.h"

namespace arm_compute
{

CLCompatCommandBuffer::CLCompatCommandBuffer(cl_command_queue queue) : _queue(queue)
{
}

CLCompatCommandBuffer::~CLCompatCommandBuffer()
{
}

void CLCompatCommandBuffer::add_kernel(cl_kernel          kernel,
                                       const cl::NDRange &offset,
                                       const cl::NDRange &global,
                                       const cl::NDRange &local)
{
    ARM_COMPUTE_ERROR_ON(state() != State::Created);

    _kernel_cmds.push_back(KernelCommand{kernel, offset, global, local, {}});
}

void CLCompatCommandBuffer::add_mutable_argument_generic(cl_uint arg_idx, const void *value, size_t size)
{
    ARM_COMPUTE_ERROR_ON(state() != State::Created);
    ARM_COMPUTE_ERROR_ON(_kernel_cmds.empty());

    _kernel_cmds.back().mutable_args.push_back(cl_mutable_dispatch_arg_khr{arg_idx, size, value});
}

void CLCompatCommandBuffer::finalize()
{
    ARM_COMPUTE_ERROR_ON(state() != State::Created);

    _kernel_cmds.shrink_to_fit();

    for (auto &cmd : _kernel_cmds)
    {
        cmd.mutable_args.shrink_to_fit();
    }

    state(State::Finalized);
}

void CLCompatCommandBuffer::update()
{
    ARM_COMPUTE_ERROR_ON(state() != State::Finalized);

    // Nothing to do here - The kernel arguments will be updated when each command is enqueued.
}

void CLCompatCommandBuffer::enqueue()
{
    ARM_COMPUTE_ERROR_ON(state() != State::Finalized);

    for (const auto &cmd : _kernel_cmds)
    {
        for (const auto &arg : cmd.mutable_args)
        {
            const auto error = clSetKernelArg(cmd.kernel, arg.arg_index, arg.arg_size, arg.arg_value);

            handle_cl_error("clSetKernelArg", error);
        }

        const auto error =
            clEnqueueNDRangeKernel(_queue, cmd.kernel, static_cast<cl_uint>(cmd.global.dimensions()),
                                   cmd.offset.dimensions() != 0 ? cmd.offset.get() : nullptr, cmd.global.get(),
                                   cmd.local.dimensions() != 0 ? cmd.local.get() : nullptr, 0, nullptr, nullptr);

        handle_cl_error("clEnqueueNDRangeKernel", error);
    }
}

bool CLCompatCommandBuffer::is_finalized() const
{
    return state() == State::Finalized;
}

} // namespace arm_compute
