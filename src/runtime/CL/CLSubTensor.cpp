/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/runtime/CL/CLSubTensor.h"

#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

using namespace arm_compute;

CLSubTensor::CLSubTensor()
    : _parent(nullptr), _info()
{
}

CLSubTensor::CLSubTensor(ICLTensor *parent, const TensorShape &tensor_shape, const Coordinates &coords, bool extend_parent)
    : _parent(nullptr), _info()
{
    ARM_COMPUTE_ERROR_ON(parent == nullptr);
    _info   = SubTensorInfo(parent->info(), tensor_shape, coords, extend_parent);
    _parent = parent;
}

ITensorInfo *CLSubTensor::info() const
{
    return &_info;
}

ITensorInfo *CLSubTensor::info()
{
    return &_info;
}

const cl::Buffer &CLSubTensor::cl_buffer() const
{
    ARM_COMPUTE_ERROR_ON(_parent == nullptr);
    return _parent->cl_buffer();
}

CLQuantization CLSubTensor::quantization() const
{
    return _parent->quantization();
}

ICLTensor *CLSubTensor::parent()
{
    return _parent;
}

void CLSubTensor::map(bool blocking)
{
    ICLTensor::map(CLScheduler::get().queue(), blocking);
}

void CLSubTensor::unmap()
{
    ICLTensor::unmap(CLScheduler::get().queue());
}

uint8_t *CLSubTensor::do_map(cl::CommandQueue &q, bool blocking)
{
    ARM_COMPUTE_ERROR_ON(cl_buffer().get() == nullptr);
    return static_cast<uint8_t *>(q.enqueueMapBuffer(cl_buffer(), blocking ? CL_TRUE : CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, 0, info()->total_size()));
}

void CLSubTensor::do_unmap(cl::CommandQueue &q)
{
    ARM_COMPUTE_ERROR_ON(cl_buffer().get() == nullptr);
    q.enqueueUnmapMemObject(cl_buffer(), buffer());
}
