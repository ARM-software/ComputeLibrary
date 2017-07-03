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
#ifndef __ARM_COMPUTE_TEST_NEON_NEACCESSOR_H__
#define __ARM_COMPUTE_TEST_NEON_NEACCESSOR_H__

#include "IAccessor.h"

#include "arm_compute/runtime/Tensor.h"

namespace arm_compute
{
namespace test
{
namespace neon
{
/** Accessor implementation for @ref Tensor objects. */
class NEAccessor : public IAccessor
{
public:
    /** Create an accessor for the given @p tensor.
     *
     * @param[in, out] tensor To be accessed tensor.
     */
    NEAccessor(Tensor &tensor);

    NEAccessor(const NEAccessor &) = delete;
    NEAccessor &operator=(const NEAccessor &) = delete;
    NEAccessor(NEAccessor &&)                 = default;
    NEAccessor &operator=(NEAccessor &&) = default;

    TensorShape shape() const override;
    size_t      element_size() const override;
    size_t      size() const override;
    Format      format() const override;
    DataType    data_type() const override;
    int         num_channels() const override;
    int         num_elements() const override;
    int         fixed_point_position() const override;
    const void *operator()(const Coordinates &coord) const override;
    void *operator()(const Coordinates &coord) override;

private:
    Tensor &_tensor;
};

inline NEAccessor::NEAccessor(Tensor &tensor)
    : _tensor{ tensor }
{
}

inline TensorShape NEAccessor::shape() const
{
    return _tensor.info()->tensor_shape();
}

inline size_t NEAccessor::element_size() const
{
    return _tensor.info()->element_size();
}

inline size_t NEAccessor::size() const
{
    return _tensor.info()->total_size();
}

inline Format NEAccessor::format() const
{
    return _tensor.info()->format();
}

inline DataType NEAccessor::data_type() const
{
    return _tensor.info()->data_type();
}

inline int NEAccessor::num_channels() const
{
    return _tensor.info()->num_channels();
}

inline int NEAccessor::num_elements() const
{
    return _tensor.info()->tensor_shape().total_size();
}

inline int NEAccessor::fixed_point_position() const
{
    return _tensor.info()->fixed_point_position();
}

inline const void *NEAccessor::operator()(const Coordinates &coord) const
{
    return _tensor.ptr_to_element(coord);
}

inline void *NEAccessor::operator()(const Coordinates &coord)
{
    return _tensor.ptr_to_element(coord);
}
} // namespace neon
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_NEON_NEACCESSOR_H__ */
