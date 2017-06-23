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

#include "Utils.h"

#include "arm_compute/core/Utils.h"

#include <algorithm>
#include <array>
#include <functional>
#include <stdexcept>
#include <utility>

namespace arm_compute
{
namespace test
{
RawTensor::RawTensor(TensorShape shape, Format format, int fixed_point_position)
    : _buffer(nullptr),
      _shape(shape),
      _format(format),
      _fixed_point_position(fixed_point_position)
{
    _buffer = ::arm_compute::test::cpp14::make_unique<BufferType[]>(size());
}

RawTensor::RawTensor(TensorShape shape, DataType data_type, int num_channels, int fixed_point_position)
    : _buffer(nullptr),
      _shape(shape),
      _data_type(data_type),
      _num_channels(num_channels),
      _fixed_point_position(fixed_point_position)
{
    _buffer = ::arm_compute::test::cpp14::make_unique<BufferType[]>(size());
}

RawTensor::RawTensor(const RawTensor &tensor)
    : _buffer(nullptr),
      _shape(tensor.shape()),
      _format(tensor.format()),
      _fixed_point_position(tensor.fixed_point_position())
{
    _buffer = ::arm_compute::test::cpp14::make_unique<BufferType[]>(tensor.size());
    std::copy(tensor.data(), tensor.data() + size(), _buffer.get());
}

RawTensor &RawTensor::operator=(RawTensor tensor)
{
    swap(*this, tensor);

    return *this;
}

RawTensor::BufferType &RawTensor::operator[](size_t offset)
{
    return _buffer[offset];
}

const RawTensor::BufferType &RawTensor::operator[](size_t offset) const
{
    return _buffer[offset];
}

TensorShape RawTensor::shape() const
{
    return _shape;
}

size_t RawTensor::element_size() const
{
    return num_channels() * element_size_from_data_type(data_type());
}

int RawTensor::fixed_point_position() const
{
    return _fixed_point_position;
}

size_t RawTensor::size() const
{
    const size_t size = std::accumulate(_shape.cbegin(), _shape.cend(), 1, std::multiplies<size_t>());
    return size * element_size();
}

Format RawTensor::format() const
{
    return _format;
}

DataType RawTensor::data_type() const
{
    if(_format != Format::UNKNOWN)
    {
        return data_type_from_format(_format);
    }
    else
    {
        return _data_type;
    }
}

int RawTensor::num_channels() const
{
    switch(_format)
    {
        case Format::U8:
        case Format::S16:
        case Format::U16:
        case Format::S32:
        case Format::U32:
            return 1;
        case Format::RGB888:
            return 3;
        case Format::UNKNOWN:
            return _num_channels;
        default:
            ARM_COMPUTE_ERROR("NOT SUPPORTED!");
    }
}

int RawTensor::num_elements() const
{
    return _shape.total_size();
}

const RawTensor::BufferType *RawTensor::data() const
{
    return _buffer.get();
}

RawTensor::BufferType *RawTensor::data()
{
    return _buffer.get();
}

const RawTensor::BufferType *RawTensor::operator()(const Coordinates &coord) const
{
    return _buffer.get() + coord2index(_shape, coord) * element_size();
}

RawTensor::BufferType *RawTensor::operator()(const Coordinates &coord)
{
    return _buffer.get() + coord2index(_shape, coord) * element_size();
}

void swap(RawTensor &tensor1, RawTensor &tensor2)
{
    // Use unqualified call to swap to enable ADL. But make std::swap available
    // as backup.
    using std::swap;
    swap(tensor1._shape, tensor2._shape);
    swap(tensor1._format, tensor2._format);
    swap(tensor1._data_type, tensor2._data_type);
    swap(tensor1._num_channels, tensor2._num_channels);
    swap(tensor1._buffer, tensor2._buffer);
}
} // namespace test
} // namespace arm_compute
