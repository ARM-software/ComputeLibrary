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

#ifndef ACL_SRC_CORE_CL_CLCOMPATCOMMANDBUFFER_H
#define ACL_SRC_CORE_CL_CLCOMPATCOMMANDBUFFER_H

#include "src/core/CL/CLCommandBuffer.h"

#include <vector>

namespace arm_compute
{

/** Command buffer implementation for platform without mutable dispatch command buffer extension. */
class CLCompatCommandBuffer final : public CLCommandBuffer
{
public:
    /** Create a new command buffer targeting the specified command queue.
     *
     * @param[in] queue The command queue to execute the command buffer.
     */
    CLCompatCommandBuffer(cl_command_queue queue);

    /** Destructor. */
    virtual ~CLCompatCommandBuffer();

    /** Disallow copy constructor. */
    CLCompatCommandBuffer(const CLCompatCommandBuffer &) = delete;

    /** Disallow copy assignment. */
    CLCompatCommandBuffer &operator=(const CLCompatCommandBuffer &) = delete;

    /** Disallow move constructor. */
    CLCompatCommandBuffer(CLCompatCommandBuffer &&) = delete;

    /** Disallow move assignment. */
    CLCompatCommandBuffer &operator=(CLCompatCommandBuffer &&) = delete;

    void add_kernel(cl_kernel kernel, const cl::NDRange &offset, const cl::NDRange &global, const cl::NDRange &local) override;

    void finalize() override;

    void update() override;

    void enqueue() override;

    bool is_finalized() const override;

protected:
    void add_mutable_argument_generic(cl_uint arg_idx, const void *value, size_t size) override;

private:
    struct KernelCommand
    {
        cl_kernel   kernel;
        cl::NDRange offset;
        cl::NDRange global;
        cl::NDRange local;

        std::vector<cl_mutable_dispatch_arg_khr> mutable_args;
    };

private:
    cl_command_queue           _queue{};
    std::vector<KernelCommand> _kernel_cmds{};
};

} // namespace arm_compute

#endif // ACL_SRC_CORE_CL_CLCOMPATCOMMANDBUFFER_H
