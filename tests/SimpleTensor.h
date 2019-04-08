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
     * @param[in] shape  Shape of the new raw tensor.
     * @param[in] format Format of the new raw tensor.
     */
    SimpleTensor(TensorShape shape, Format format);

    /** Create an uninitialised tensor of the given @p shape and @p data type.
     *
     * @param[in] shape             Shape of the new raw tensor.
     * @param[in] data_type         Data type of the new raw tensor.
     * @param[in] num_channels      (Optional) Number of channels (default = 1).
     * @param[in] quantization_info (Optional) Quantization info for asymmetric quantization (default = empty).
     * @param[in] data_layout       (Optional) Data layout of the tensor (default = NCHW).
     */
    SimpleTensor(TensorShape shape, DataType data_type,
                 int              num_channels      = 1,
                 QuantizationInfo quantization_info = QuantizationInfo(),
                 DataLayout       data_layout       = DataLayout::NCHW);

    /** Create a deep copy of the given @p tensor.
     *
     * @param[in] tensor To be copied tensor.
     */
    SimpleTensor(const SimpleTensor &tensor);

    /** Create a deep copy of the given @p tensor.
     *
     * @param[in] tensor To be copied tensor.
     *
     * @return a copy of the given tensor.
     */
    SimpleTensor &operator=(SimpleTensor tensor);
    /** Allow instances of this class to be move constructed */
    SimpleTensor(SimpleTensor &&) = default;
    /** Default destructor. */
    ~SimpleTensor() = default;

    /** Tensor value type */
    using value_type = T;
    /** Tensor buffer pointer type */
    using Buffer = std::unique_ptr<value_type[]>;

    friend class RawTensor;

    /** Return value at @p offset in the buffer.
     *
     * @param[in] offset Offset within the buffer.
     *
     * @return value in the buffer.
     */
    T &operator[](size_t offset);

    /** Return constant value at @p offset in the buffer.
     *
     * @param[in] offset Offset within the buffer.
     *
     * @return constant value in the buffer.
     */
    const T &operator[](size_t offset) const;

    /** Shape of the tensor.
     *
     * @return the shape of the tensor.
     */
    TensorShape shape() const override;
    /** Size of each element in the tensor in bytes.
     *
     * @return the size of each element in the tensor in bytes.
     */
    size_t element_size() const override;
    /** Total size of the tensor in bytes.
     *
     * @return the total size of the tensor in bytes.
     */
    size_t size() const override;
    /** Image format of the tensor.
     *
     * @return the format of the tensor.
     */
    Format format() const override;
    /** Data layout of the tensor.
     *
     * @return the data layout of the tensor.
     */
    DataLayout data_layout() const override;
    /** Data type of the tensor.
     *
     * @return the data type of the tensor.
     */
    DataType data_type() const override;
    /** Number of channels of the tensor.
     *
     * @return the number of channels of the tensor.
     */
    int num_channels() const override;
    /** Number of elements of the tensor.
     *
     * @return the number of elements of the tensor.
     */
    int num_elements() const override;
    /** Available padding around the tensor.
     *
     * @return the available padding around the tensor.
     */
    PaddingSize padding() const override;
    /** Quantization info in case of asymmetric quantized type
     *
     * @return
     */
    QuantizationInfo quantization_info() const override;

    /** Constant pointer to the underlying buffer.
     *
     * @return a constant pointer to the data.
     */
    const T *data() const;

    /** Pointer to the underlying buffer.
     *
     * @return a pointer to the data.
     */
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
    QuantizationInfo _quantization_info{};
    DataLayout       _data_layout{ DataLayout::UNKNOWN };
};

template <typename T1, typename T2>
SimpleTensor<T1> copy_tensor(const SimpleTensor<T2> &tensor)
{
    SimpleTensor<T1> st(tensor.shape(), tensor.data_type(),
                        tensor.num_channels(),
                        tensor.quantization_info(),
                        tensor.data_layout());
    for(size_t n = 0; n < size_t(st.num_elements()); n++)
    {
        st.data()[n] = static_cast<T1>(tensor.data()[n]);
    }
    return st;
}

template <typename T1, typename T2, typename std::enable_if<std::is_same<T1, T2>::value, int>::type = 0>
SimpleTensor<T1> copy_tensor(const SimpleTensor<half> &tensor)
{
    SimpleTensor<T1> st(tensor.shape(), tensor.data_type(),
                        tensor.num_channels(),
                        tensor.quantization_info(),
                        tensor.data_layout());
    memcpy((void *)st.data(), (const void *)tensor.data(), size_t(st.num_elements() * sizeof(T1)));
    return st;
}

template < typename T1, typename T2, typename std::enable_if < (std::is_same<T1, half>::value || std::is_same<T2, half>::value), int >::type = 0 >
SimpleTensor<T1> copy_tensor(const SimpleTensor<half> &tensor)
{
    SimpleTensor<T1> st(tensor.shape(), tensor.data_type(),
                        tensor.num_channels(),
                        tensor.quantization_info(),
                        tensor.data_layout());
    for(size_t n = 0; n < size_t(st.num_elements()); n++)
    {
        st.data()[n] = half_float::detail::half_cast<T1, T2>(tensor.data()[n]);
    }
    return st;
}

template <typename T>
SimpleTensor<T>::SimpleTensor(TensorShape shape, Format format)
    : _buffer(nullptr),
      _shape(shape),
      _format(format),
      _quantization_info(),
      _data_layout(DataLayout::NCHW)
{
    _num_channels = num_channels();
    _buffer       = support::cpp14::make_unique<T[]>(num_elements() * _num_channels);
}

template <typename T>
SimpleTensor<T>::SimpleTensor(TensorShape shape, DataType data_type, int num_channels, QuantizationInfo quantization_info, DataLayout data_layout)
    : _buffer(nullptr),
      _shape(shape),
      _data_type(data_type),
      _num_channels(num_channels),
      _quantization_info(quantization_info),
      _data_layout(data_layout)
{
    _buffer = support::cpp14::make_unique<T[]>(this->_shape.total_size() * _num_channels);
}

template <typename T>
SimpleTensor<T>::SimpleTensor(const SimpleTensor &tensor)
    : _buffer(nullptr),
      _shape(tensor.shape()),
      _format(tensor.format()),
      _data_type(tensor.data_type()),
      _num_channels(tensor.num_channels()),
      _quantization_info(tensor.quantization_info()),
      _data_layout(tensor.data_layout())
{
    _buffer = support::cpp14::make_unique<T[]>(tensor.num_elements() * _num_channels);
    std::copy_n(tensor.data(), this->_shape.total_size() * _num_channels, _buffer.get());
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
DataLayout SimpleTensor<T>::data_layout() const
{
    return _data_layout;
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
        case Format::U16:
        case Format::S16:
        case Format::U32:
        case Format::S32:
        case Format::F16:
        case Format::F32:
            return 1;
        // Because the U and V channels are subsampled
        // these formats appear like having only 2 channels:
        case Format::YUYV422:
        case Format::UYVY422:
            return 2;
        case Format::UV88:
            return 2;
        case Format::RGB888:
            return 3;
        case Format::RGBA8888:
            return 4;
        case Format::UNKNOWN:
            return _num_channels;
        //Doesn't make sense for planar formats:
        case Format::NV12:
        case Format::NV21:
        case Format::IYUV:
        case Format::YUV444:
        default:
            return 0;
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
    swap(tensor1._quantization_info, tensor2._quantization_info);
    swap(tensor1._buffer, tensor2._buffer);
}
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_SIMPLE_TENSOR_H__ */
