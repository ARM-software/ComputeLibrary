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
#ifndef __ARM_COMPUTE_TEST_ACCESSOR_H__
#define __ARM_COMPUTE_TEST_ACCESSOR_H__

#include "arm_compute/runtime/Tensor.h"
#include "tests/IAccessor.h"

namespace arm_compute
{
namespace test
{
/** Accessor implementation for @ref Tensor objects. */
class Accessor : public IAccessor
{
public:
    /** Create an accessor for the given @p tensor.
     *
     * @param[in, out] tensor To be accessed tensor.
     */
    Accessor(ITensor &tensor);

    /** Prevent instances of this class from being copy constructed */
    Accessor(const Accessor &) = delete;
    /** Prevent instances of this class from being copied */
    Accessor &operator=(const Accessor &) = delete;
    /** Allow instances of this class to be move constructed */
    Accessor(Accessor &&) = default;
    /** Allow instances of this class to be moved */
    Accessor &operator=(Accessor &&) = default;

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
    ITensor &_tensor;
};

inline Accessor::Accessor(ITensor &tensor)
    : _tensor{ tensor }
{
}

inline TensorShape Accessor::shape() const
{
    return _tensor.info()->tensor_shape();
}

inline size_t Accessor::element_size() const
{
    return _tensor.info()->element_size();
}

inline size_t Accessor::size() const
{
    return _tensor.info()->total_size();
}

inline Format Accessor::format() const
{
    return _tensor.info()->format();
}

inline DataLayout Accessor::data_layout() const
{
    return _tensor.info()->data_layout();
}

inline DataType Accessor::data_type() const
{
    return _tensor.info()->data_type();
}

inline int Accessor::num_channels() const
{
    return _tensor.info()->num_channels();
}

inline int Accessor::num_elements() const
{
    return _tensor.info()->tensor_shape().total_size();
}

inline PaddingSize Accessor::padding() const
{
    return _tensor.info()->padding();
}

inline QuantizationInfo Accessor::quantization_info() const
{
    return _tensor.info()->quantization_info();
}

inline const void *Accessor::data() const
{
    return _tensor.buffer();
}

inline void *Accessor::data()
{
    return _tensor.buffer();
}

inline const void *Accessor::operator()(const Coordinates &coord) const
{
    return _tensor.ptr_to_element(coord);
}

inline void *Accessor::operator()(const Coordinates &coord)
{
    return _tensor.ptr_to_element(coord);
}
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_ACCESSOR_H__ */
