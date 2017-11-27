/*
 * Copyright (c) 2017, 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_TEST_SIMPLE_TENSOR_H__
#define __ARM_COMPUTE_TEST_SIMPLE_TENSOR_H__

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "support/ToolchainSupport.h"
#include "tests/IAccessor.h"
#include "tests/Utils.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>

namespace arm_compute
{
namespace test
{
class RawTensor;

/** Simple tensor object that stores elements in a consecutive chunk of memory.
 *
 * It can be created by either loading an image from a file which also
 * initialises the content of the tensor or by explcitly specifying the size.
 * The latter leaves the content uninitialised.
 *
 * Furthermore, the class provides methods to convert the tensor's values into
 * different image format.
 */
template <typename T>
class SimpleTensor : public IAccessor
{
public:
    /** Create an uninitialised tensor. */
    SimpleTensor() = default;

    /** Create an uninitialised tensor of the given @p shape and @p format.
     *
     * @param[in] shape                Shape of the new raw tensor.
     * @param[in] format               Format of the new raw tensor.
     * @param[in] fixed_point_position (Optional) Number of bits for the fractional part of the fixed point numbers
     */
    SimpleTensor(TensorShape shape, Format format, int fixed_point_position = 0);

    /** Create an uninitialised tensor of the given @p shape and @p data type.
     *
     * @param[in] shape                Shape of the new raw tensor.
     * @param[in] data_type            Data type of the new raw tensor.
     * @param[in] num_channels         (Optional) Number of channels (default = 1).
     * @param[in] fixed_point_position (Optional) Number of bits for the fractional part of the fixed point numbers (default = 0).
     * @param[in] quantization_info    (Optional) Quantization info for asymmetric quantization (default = empty).
     */
    SimpleTensor(TensorShape shape, DataType data_type,
                 int num_channels         = 1,
                 int fixed_point_position = 0, QuantizationInfo quantization_info = QuantizationInfo());

    /** Create a deep copy of the given @p tensor.
     *
     * @param[in] tensor To be copied tensor.
     */
    SimpleTensor(const SimpleTensor &tensor);

    /** Create a deep copy of the given @p tensor.
     *
     * @param[in] tensor To be copied tensor.
     */
    SimpleTensor &operator        =(SimpleTensor tensor);
    SimpleTensor(SimpleTensor &&) = default;
    ~SimpleTensor()               = default;

    using value_type = T;
    using Buffer     = std::unique_ptr<value_type[]>;

    friend class RawTensor;

    /** Return value at @p offset in the buffer.
     *
     * @param[in] offset Offset within the buffer.
     */
    T &operator[](size_t offset);

    /** Return constant value at @p offset in the buffer.
     *
     * @param[in] offset Offset within the buffer.
     */
    const T &operator[](size_t offset) const;

    /** Shape of the tensor. */
    TensorShape shape() const override;

    /** Size of each element in the tensor in bytes. */
    size_t element_size() const override;

    /** Total size of the tensor in bytes. */
    size_t size() const override;

    /** Image format of the tensor. */
    Format format() const override;

    /** Data type of the tensor. */
    DataType data_type() const override;

    /** Number of channels of the tensor. */
    int num_channels() const override;

    /** Number of elements of the tensor. */
    int num_elements() const override;

    /** Available padding around the tensor. */
    PaddingSize padding() const override;

    /** The number of bits for the fractional part of the fixed point numbers. */
    int fixed_point_position() const override;

    /** Quantization info in case of asymmetric quantized type */
    QuantizationInfo quantization_info() const override;

    /** Constant pointer to the underlying buffer. */
    const T *data() const;

    /** Pointer to the underlying buffer. */
    T *data();

    /** Read only access to the specified element.
     *
     * @param[in] coord Coordinates of the desired element.
     *
     * @return A pointer to the desired element.
     */
    const void *operator()(const Coordinates &coord) const override;

    /** Access to the specified element.
     *
     * @param[in] coord Coordinates of the desired element.
     *
     * @return A pointer to the desired element.
     */
    void *operator()(const Coordinates &coord) override;

    /** Swaps the content of the provided tensors.
     *
     * @param[in, out] tensor1 Tensor to be swapped.
     * @param[in, out] tensor2 Tensor to be swapped.
     */
    template <typename U>
    friend void swap(SimpleTensor<U> &tensor1, SimpleTensor<U> &tensor2);

protected:
    Buffer           _buffer{ nullptr };
    TensorShape      _shape{};
    Format           _format{ Format::UNKNOWN };
    DataType         _data_type{ DataType::UNKNOWN };
    int              _num_channels{ 0 };
    int              _fixed_point_position{ 0 };
    QuantizationInfo _quantization_info{};
};

template <typename T>
SimpleTensor<T>::SimpleTensor(TensorShape shape, Format format, int fixed_point_position)
    : _buffer(nullptr),
      _shape(shape),
      _format(format),
      _fixed_point_position(fixed_point_position),
      _quantization_info()
{
    _num_channels = num_channels();
    _buffer       = support::cpp14::make_unique<T[]>(num_elements() * _num_channels);
}

template <typename T>
SimpleTensor<T>::SimpleTensor(TensorShape shape, DataType data_type, int num_channels, int fixed_point_position, QuantizationInfo quantization_info)
    : _buffer(nullptr),
      _shape(shape),
      _data_type(data_type),
      _num_channels(num_channels),
      _fixed_point_position(fixed_point_position),
      _quantization_info(quantization_info)
{
    _buffer = support::cpp14::make_unique<T[]>(num_elements() * this->num_channels());
}

template <typename T>
SimpleTensor<T>::SimpleTensor(const SimpleTensor &tensor)
    : _buffer(nullptr),
      _shape(tensor.shape()),
      _format(tensor.format()),
      _data_type(tensor.data_type()),
      _num_channels(tensor.num_channels()),
      _fixed_point_position(tensor.fixed_point_position()),
      _quantization_info(tensor.quantization_info())
{
    _buffer = support::cpp14::make_unique<T[]>(tensor.num_elements() * num_channels());
    std::copy_n(tensor.data(), num_elements() * num_channels(), _buffer.get());
}

template <typename T>
SimpleTensor<T> &SimpleTensor<T>::operator=(SimpleTensor tensor)
{
    swap(*this, tensor);

    return *this;
}

template <typename T>
T &SimpleTensor<T>::operator[](size_t offset)
{
    return _buffer[offset];
}

template <typename T>
const T &SimpleTensor<T>::operator[](size_t offset) const
{
    return _buffer[offset];
}

template <typename T>
TensorShape SimpleTensor<T>::shape() const
{
    return _shape;
}

template <typename T>
size_t SimpleTensor<T>::element_size() const
{
    return num_channels() * element_size_from_data_type(data_type());
}

template <typename T>
int SimpleTensor<T>::fixed_point_position() const
{
    return _fixed_point_position;
}

template <typename T>
QuantizationInfo SimpleTensor<T>::quantization_info() const
{
    return _quantization_info;
}

template <typename T>
size_t SimpleTensor<T>::size() const
{
    const size_t size = std::accumulate(_shape.cbegin(), _shape.cend(), 1, std::multiplies<size_t>());
    return size * element_size();
}

template <typename T>
Format SimpleTensor<T>::format() const
{
    return _format;
}

template <typename T>
DataType SimpleTensor<T>::data_type() const
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

template <typename T>
int SimpleTensor<T>::num_channels() const
{
    switch(_format)
    {
        case Format::U8:
        case Format::S16:
        case Format::U16:
        case Format::S32:
        case Format::U32:
        case Format::F32:
            return 1;
        case Format::RGB888:
            return 3;
        case Format::UNKNOWN:
            return _num_channels;
        default:
            ARM_COMPUTE_ERROR("NOT SUPPORTED!");
    }
}

template <typename T>
int SimpleTensor<T>::num_elements() const
{
    return _shape.total_size();
}

template <typename T>
PaddingSize SimpleTensor<T>::padding() const
{
    return PaddingSize(0);
}

template <typename T>
const T *SimpleTensor<T>::data() const
{
    return _buffer.get();
}

template <typename T>
T *SimpleTensor<T>::data()
{
    return _buffer.get();
}

template <typename T>
const void *SimpleTensor<T>::operator()(const Coordinates &coord) const
{
    return _buffer.get() + coord2index(_shape, coord) * _num_channels;
}

template <typename T>
void *SimpleTensor<T>::operator()(const Coordinates &coord)
{
    return _buffer.get() + coord2index(_shape, coord) * _num_channels;
}

template <typename U>
void swap(SimpleTensor<U> &tensor1, SimpleTensor<U> &tensor2)
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
#endif /* __ARM_COMPUTE_TEST_SIMPLE_TENSOR_H__ */
