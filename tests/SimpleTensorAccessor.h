/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_TEST_SIMPLE_TENSOR_ACCESSOR_H__
#define __ARM_COMPUTE_TEST_SIMPLE_TENSOR_ACCESSOR_H__

#include "SimpleTensor.h"
#include "tests/IAccessor.h"

namespace arm_compute
{
namespace test
{
/** Accessor implementation for @ref SimpleTensor objects. */
template <typename T>
class SimpleTensorAccessor : public IAccessor
{
public:
    /** Create an accessor for the given @p tensor.
     *
     * @param[in, out] tensor To be accessed tensor.
     */
    SimpleTensorAccessor(SimpleTensor<T> &tensor);

    /** Prevent instances of this class from being copy constructed */
    SimpleTensorAccessor(const SimpleTensorAccessor &) = delete;
    /** Prevent instances of this class from being copied */
    SimpleTensorAccessor &operator=(const SimpleTensorAccessor &) = delete;
    /** Allow instances of this class to be move constructed */
    SimpleTensorAccessor(SimpleTensorAccessor &&) = default;
    /** Allow instances of this class to be moved */
    SimpleTensorAccessor &operator=(SimpleTensorAccessor &&) = default;

    /** Get the tensor data.
     *
     * @return a constant pointer to the tensor data.
     */
    const void *data() const;
    /** Get the tensor data.
     *
     * @return a pointer to the tensor data.
     */
    void *data();

    // Inherited methods overridden:
    TensorShape      shape() const override;
    size_t           element_size() const override;
    size_t           size() const override;
    Format           format() const override;
    DataLayout       data_layout() const override;
    DataType         data_type() const override;
    int              num_channels() const override;
    int              num_elements() const override;
    PaddingSize      padding() const override;
    QuantizationInfo quantization_info() const override;
    const void *operator()(const Coordinates &coord) const override;
    void *operator()(const Coordinates &coord) override;

private:
    SimpleTensor<T> &_tensor;
};

template <typename T>
inline SimpleTensorAccessor<T>::SimpleTensorAccessor(SimpleTensor<T> &tensor)
    : _tensor{ tensor }
{
}

template <typename T>
inline TensorShape SimpleTensorAccessor<T>::shape() const
{
    return _tensor.shape();
}

template <typename T>
inline size_t SimpleTensorAccessor<T>::element_size() const
{
    return _tensor.element_size();
}

template <typename T>
inline size_t SimpleTensorAccessor<T>::size() const
{
    return _tensor.num_elements() * _tensor.element_size();
}

template <typename T>
inline Format SimpleTensorAccessor<T>::format() const
{
    return _tensor.format();
}

template <typename T>
inline DataLayout SimpleTensorAccessor<T>::data_layout() const
{
    return _tensor.data_layout();
}

template <typename T>
inline DataType SimpleTensorAccessor<T>::data_type() const
{
    return _tensor.data_type();
}

template <typename T>
inline int SimpleTensorAccessor<T>::num_channels() const
{
    return _tensor.num_channels();
}

template <typename T>
inline int SimpleTensorAccessor<T>::num_elements() const
{
    return _tensor.num_elements();
}

template <typename T>
inline PaddingSize SimpleTensorAccessor<T>::padding() const
{
    return _tensor.padding();
}

template <typename T>
inline QuantizationInfo SimpleTensorAccessor<T>::quantization_info() const
{
    return _tensor.quantization_info();
}

template <typename T>
inline const void *SimpleTensorAccessor<T>::data() const
{
    return _tensor.data();
}

template <typename T>
inline void *SimpleTensorAccessor<T>::data()
{
    return _tensor.data();
}

template <typename T>
inline const void *SimpleTensorAccessor<T>::operator()(const Coordinates &coord) const
{
    return _tensor(coord);
}

template <typename T>
inline void *SimpleTensorAccessor<T>::operator()(const Coordinates &coord)
{
    return _tensor(coord);
}
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_SIMPLE_TENSOR_ACCESSOR_H__ */
