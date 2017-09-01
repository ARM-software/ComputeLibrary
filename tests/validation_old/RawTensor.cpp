/*
 * Copyright (c) 2017 ARM Limited.
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
#include "RawTensor.h"

namespace arm_compute
{
namespace test
{
RawTensor::RawTensor(TensorShape shape, Format format, int fixed_point_position)
    : SimpleTensor(shape, format, fixed_point_position)
{
    _buffer = support::cpp14::make_unique<uint8_t[]>(SimpleTensor::num_elements() * SimpleTensor::num_channels() * SimpleTensor::element_size());
}

RawTensor::RawTensor(TensorShape shape, DataType data_type, int num_channels, int fixed_point_position)
    : SimpleTensor(shape, data_type, num_channels, fixed_point_position)
{
    _buffer = support::cpp14::make_unique<uint8_t[]>(SimpleTensor::num_elements() * SimpleTensor::num_channels() * SimpleTensor::element_size());
}

RawTensor::RawTensor(const RawTensor &tensor)
    : SimpleTensor(tensor.shape(), tensor.data_type(), tensor.num_channels(), tensor.fixed_point_position())
{
    _format = tensor.format();
    _buffer = support::cpp14::make_unique<uint8_t[]>(num_elements() * num_channels() * element_size());
    std::copy_n(tensor.data(), num_elements() * num_channels() * element_size(), _buffer.get());
}

RawTensor &RawTensor::operator=(RawTensor tensor)
{
    swap(*this, tensor);

    return *this;
}

const void *RawTensor::operator()(const Coordinates &coord) const
{
    return _buffer.get() + coord2index(_shape, coord) * element_size();
}

void *RawTensor::operator()(const Coordinates &coord)
{
    return _buffer.get() + coord2index(_shape, coord) * element_size();
}
} // namespace test
} // namespace arm_compute
