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

#include "src/core/CL/CLCommandBuffer.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"

#include "src/core/CL/CLCompatCommandBuffer.h"
#include "src/core/CL/CLMutableCommandBuffer.h"

namespace arm_compute
{

std::unique_ptr<CLCommandBuffer> CLCommandBuffer::create(cl_command_queue queue)
{
    const auto &cl_device            = CLKernelLibrary::get().get_device();
    const auto  has_mutable_dispatch = command_buffer_mutable_dispatch_supported(cl_device);

    if(has_mutable_dispatch)
    {
        return std::make_unique<CLMutableCommandBuffer>(queue);
    }
    else
    {
        return std::make_unique<CLCompatCommandBuffer>(queue);
    }
}

CLCommandBuffer::CLCommandBuffer()  = default;
CLCommandBuffer::~CLCommandBuffer() = default;

CLCommandBuffer::State CLCommandBuffer::state() const
{
    return _state;
}

CLCommandBuffer &CLCommandBuffer::state(CLCommandBuffer::State state)
{
    _state = state;

    return *this;
}

} // namespace arm_compute
