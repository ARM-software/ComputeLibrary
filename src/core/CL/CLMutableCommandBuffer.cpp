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

#include "src/core/CL/CLMutableCommandBuffer.h"

#include "arm_compute/core/Error.h"

#include "src/common/utils/Log.h"
#include "src/core/CL/CLUtils.h"

namespace arm_compute
{

CLMutableCommandBuffer::CLMutableCommandBuffer(cl_command_queue queue) : CLCommandBuffer()
{
    cl_int status = CL_SUCCESS;

    cl_command_buffer_properties_khr properties[] = {
        CL_COMMAND_BUFFER_FLAGS_KHR,
        CL_COMMAND_BUFFER_MUTABLE_KHR,
        0,
    };

    _cb = clCreateCommandBufferKHR(1, &queue, properties, &status);
    handle_cl_error("clCreateCommandBufferKHR", status);
}

CLMutableCommandBuffer::~CLMutableCommandBuffer()
{
    const auto status = clReleaseCommandBufferKHR(_cb);
    if (status != CL_SUCCESS)
    {
        const std::string error_message = "clReleaseCommandBufferKHR - Error code: " + std::to_string(status);
        ARM_COMPUTE_LOG_ERROR_ACL(error_message);
    }
}

void CLMutableCommandBuffer::add_kernel(cl_kernel          kernel,
                                        const cl::NDRange &offset,
                                        const cl::NDRange &global,
                                        const cl::NDRange &local)
{
    ARM_COMPUTE_ERROR_ON(state() != State::Created);

    cl_mutable_command_khr mutable_handle = nullptr;

    cl_ndrange_kernel_command_properties_khr properties[] = {
        CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR,
        CL_MUTABLE_DISPATCH_ARGUMENTS_KHR,
        0,
    };

    const auto error = clCommandNDRangeKernelKHR(
        _cb, nullptr, properties, kernel, global.dimensions(), offset.dimensions() != 0 ? offset.get() : nullptr,
        global.get(), local.dimensions() != 0 ? local.get() : nullptr, 0, nullptr, nullptr, &mutable_handle);

    handle_cl_error("clCommandNDRangeKernelKHR", error);

    cl_mutable_dispatch_config_khr mut_dispatch_cfg{};
    mut_dispatch_cfg.type    = CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR;
    mut_dispatch_cfg.command = mutable_handle;

    _mut_dispatch_cfgs.emplace_back(mut_dispatch_cfg);
}

void CLMutableCommandBuffer::add_mutable_argument_generic(cl_uint arg_idx, const void *value, size_t size)
{
    ARM_COMPUTE_ERROR_ON(state() != State::Created);

    cl_mutable_dispatch_arg_khr cfg{};
    cfg.arg_index = arg_idx;
    cfg.arg_size  = size;
    cfg.arg_value = value;

    _mut_arg_cfgs.emplace_back(cfg);
    ++_mut_dispatch_cfgs.back().num_args;
}

void CLMutableCommandBuffer::finalize()
{
    ARM_COMPUTE_ERROR_ON(state() != State::Created);

    const auto error = clFinalizeCommandBufferKHR(_cb);
    handle_cl_error("clFinalizeCommandBufferKHR", error);

    state(State::Finalized);

    _mut_dispatch_cfgs.shrink_to_fit();
    _mut_arg_cfgs.shrink_to_fit();

    size_t arg_no = 0;

    for (auto &mut_dispatch_cfg : _mut_dispatch_cfgs)
    {
        ARM_COMPUTE_ERROR_ON(arg_no >= _mut_arg_cfgs.size());
        mut_dispatch_cfg.arg_list = &_mut_arg_cfgs[arg_no];

        arg_no += mut_dispatch_cfg.num_args;
    }

    _mut_cfg.type                  = CL_STRUCTURE_TYPE_MUTABLE_BASE_CONFIG_KHR;
    _mut_cfg.next                  = nullptr;
    _mut_cfg.num_mutable_dispatch  = _mut_dispatch_cfgs.size();
    _mut_cfg.mutable_dispatch_list = &_mut_dispatch_cfgs[0];
}

void CLMutableCommandBuffer::update()
{
    ARM_COMPUTE_ERROR_ON(state() != State::Finalized);

    const auto error = clUpdateMutableCommandsKHR(_cb, &_mut_cfg);

    handle_cl_error("clUpdateMutableCommandsKHR", error);
}

void CLMutableCommandBuffer::enqueue()
{
    ARM_COMPUTE_ERROR_ON(state() != State::Finalized);

    const auto error = clEnqueueCommandBufferKHR(0, nullptr, _cb, 0, nullptr, nullptr);

    handle_cl_error("clEnqueueCommandBufferKHR", error);
}

bool CLMutableCommandBuffer::is_finalized() const
{
    return state() == State::Finalized;
}

} // namespace arm_compute
