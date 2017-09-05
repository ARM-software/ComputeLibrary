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
#include "FlattenLayer.h"

#include "tests/validation/FixedPoint.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> flatten_layer(const SimpleTensor<T> &src)
{
    TensorShape shape_flatten(src.shape());
    shape_flatten.set(0, src.shape()[0] * src.shape()[1] * src.shape()[2]);
    shape_flatten.remove_dimension(1);
    shape_flatten.remove_dimension(1);
    SimpleTensor<T> dst(shape_flatten, src.data_type(), 1, src.fixed_point_position());

    // Note: Since the reference implementation does not use padding bytes, we can copy directly the content of the source tensor
    std::copy(src.data(), src.data() + src.num_elements(), dst.data());

    return dst;
}

template SimpleTensor<float> flatten_layer(const SimpleTensor<float> &src);
template SimpleTensor<half> flatten_layer(const SimpleTensor<half> &src);
template SimpleTensor<qint8_t> flatten_layer(const SimpleTensor<qint8_t> &src);
template SimpleTensor<qint16_t> flatten_layer(const SimpleTensor<qint16_t> &src);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
