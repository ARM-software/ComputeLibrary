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
#include "ReshapeLayer.h"

#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> reshape_layer(const SimpleTensor<T> &src, const TensorShape &output_shape)
{
    ARM_COMPUTE_ERROR_ON(src.shape().total_size() != output_shape.total_size());

    SimpleTensor<T> dst(output_shape, src.data_type());
    std::copy_n(src.data(), src.num_elements(), dst.data());
    return dst;
}

template SimpleTensor<uint8_t> reshape_layer(const SimpleTensor<uint8_t> &src, const TensorShape &output_shape);
template SimpleTensor<int8_t> reshape_layer(const SimpleTensor<int8_t> &src, const TensorShape &output_shape);
template SimpleTensor<uint16_t> reshape_layer(const SimpleTensor<uint16_t> &src, const TensorShape &output_shape);
template SimpleTensor<int16_t> reshape_layer(const SimpleTensor<int16_t> &src, const TensorShape &output_shape);
template SimpleTensor<uint32_t> reshape_layer(const SimpleTensor<uint32_t> &src, const TensorShape &output_shape);
template SimpleTensor<int32_t> reshape_layer(const SimpleTensor<int32_t> &src, const TensorShape &output_shape);
template SimpleTensor<half> reshape_layer(const SimpleTensor<half> &src, const TensorShape &output_shape);
template SimpleTensor<float> reshape_layer(const SimpleTensor<float> &src, const TensorShape &output_shape);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
