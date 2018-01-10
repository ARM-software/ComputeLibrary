/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/runtime/SubTensor.h"

#include "arm_compute/core/Error.h"

using namespace arm_compute;

SubTensor::SubTensor(ITensor *parent, const TensorShape &tensor_shape, const Coordinates &coords, bool extend_parent)
    : _parent(nullptr), _info()
{
    ARM_COMPUTE_ERROR_ON(parent == nullptr);
    _info   = SubTensorInfo(parent->info(), tensor_shape, coords, extend_parent);
    _parent = parent;
}

ITensorInfo *SubTensor::info() const
{
    return &_info;
}

ITensorInfo *SubTensor::info()
{
    return &_info;
}

uint8_t *SubTensor::buffer() const
{
    ARM_COMPUTE_ERROR_ON(_parent == nullptr);
    return _parent->buffer();
}

ITensor *SubTensor::parent()
{
    return _parent;
}
