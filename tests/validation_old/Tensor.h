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
#ifndef __ARM_COMPUTE_TEST_TENSOR_H__
#define __ARM_COMPUTE_TEST_TENSOR_H__

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename T>
class Tensor
{
public:
    Tensor()
        : _shape(), _dt(DataType::UNKNOWN), _fixed_point_position(0), _ptr(nullptr), _ptr_const(nullptr) {};

    Tensor(TensorShape shape, DataType dt, int fixed_point_position, T *ptr)
        : _shape(shape), _dt(dt), _fixed_point_position(fixed_point_position), _ptr(ptr), _ptr_const(nullptr) {};

    Tensor(TensorShape shape, DataType dt, int fixed_point_position, const T *ptr)
        : _shape(shape), _dt(dt), _fixed_point_position(fixed_point_position), _ptr(nullptr), _ptr_const(ptr) {};

    Tensor(const Tensor &tensor) = delete;
    Tensor &operator=(const Tensor &) = delete;
    Tensor(Tensor &&)                 = default;
    Tensor &operator=(Tensor &&) = default;

    ~Tensor() = default;

    T &operator[](size_t offset)
    {
        ARM_COMPUTE_ERROR_ON(_ptr == nullptr);

        return _ptr[offset];
    }

    const T &operator[](size_t offset) const
    {
        const T *ptr = (_ptr_const != nullptr) ? _ptr_const : _ptr;

        ARM_COMPUTE_ERROR_ON(ptr == nullptr);

        return ptr[offset]; // NOLINT
    }

    int num_elements() const
    {
        return std::accumulate(_shape.cbegin(), _shape.cend(), 1, std::multiplies<int>());
    }

    TensorShape shape() const
    {
        return _shape;
    }

    DataType data_type() const
    {
        return _dt;
    }

    int fixed_point_position() const
    {
        return _fixed_point_position;
    }

    const T *data() const
    {
        return (_ptr_const != nullptr) ? _ptr_const : _ptr;
    }

    T *data()
    {
        return _ptr;
    }

    const T *data_const() const
    {
        return _ptr_const;
    }

private:
    TensorShape _shape;
    DataType    _dt;
    int         _fixed_point_position;
    T          *_ptr;
    const T    *_ptr_const;
};
} // namespace validation
} // test
} // arm_compute

#endif /* __ARM_COMPUTE_TEST_TENSOR_H__ */
