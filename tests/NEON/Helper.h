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
#ifndef __ARM_COMPUTE_TEST_NEON_HELPER_H__
#define __ARM_COMPUTE_TEST_NEON_HELPER_H__

#include "Globals.h"
#include "TensorLibrary.h"

#include "arm_compute/runtime/Array.h"
#include "arm_compute/runtime/Tensor.h"

namespace arm_compute
{
namespace test
{
namespace neon
{
/** Helper to create an empty tensor.
 *
 * @param[in] shape                Desired shape.
 * @param[in] data_type            Desired data type.
 * @param[in] num_channels         (Optional) It indicates the number of channels for each tensor element
 * @param[in] fixed_point_position (Optional) Fixed point position that expresses the number of bits for the fractional part of the number when the tensor's data type is QS8 or QS16.
 *
 * @return Empty @ref Tensor with the specified shape and data type.
 */
inline Tensor create_tensor(const TensorShape &shape, DataType data_type, int num_channels = 1, int fixed_point_position = 0)
{
    Tensor tensor;
    tensor.allocator()->init(TensorInfo(shape, num_channels, data_type, fixed_point_position));

    return tensor;
}

/** Helper to create an empty tensor.
 *
 * @param[in] name                 File name from which to get the dimensions.
 * @param[in] data_type            Desired data type.
 * @param[in] fixed_point_position (Optional) Number of bits for the fractional part of the fixed point numbers
 *
 * @return Empty @ref Tensor with the specified shape and data type.
 */
inline Tensor create_tensor(const std::string &name, DataType data_type, int fixed_point_position = 0)
{
    constexpr unsigned int num_channels = 1;

    const RawTensor &raw = library->get(name);

    Tensor tensor;
    tensor.allocator()->init(TensorInfo(raw.shape(), num_channels, data_type, fixed_point_position));

    return tensor;
}

template <typename T>
Array<T> create_array(const std::vector<T> &v)
{
    const size_t v_size = v.size();
    Array<T>     array(v_size);

    array.resize(v_size);
    std::copy(v.begin(), v.end(), array.buffer());

    return array;
}
} // namespace neon
} // namespace test
} // namespace arm_compute
#endif
