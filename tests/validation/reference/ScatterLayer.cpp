/*
 * Copyright (c) 2024 Arm Limited.
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
#include "ScatterLayer.h"
#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{

template <typename T>
SimpleTensor<T> scatter_layer_internal(const SimpleTensor<T> &src, const SimpleTensor<T> &updates, const SimpleTensor<uint32_t> &indices, const TensorShape &out_shape, const ScatterInfo &info)
{
    ARM_COMPUTE_UNUSED(src);
    ARM_COMPUTE_UNUSED(updates);
    ARM_COMPUTE_UNUSED(indices);
    ARM_COMPUTE_UNUSED(info);
    // Unimplemented reference.
    SimpleTensor<T> dst{ out_shape, src.data_type(), 1 };
    return dst;
}

template <typename T>
SimpleTensor<T> scatter_layer(const SimpleTensor<T> &src, const SimpleTensor<T> &updates, const SimpleTensor<uint32_t> &indices, const TensorShape &out_shape, const ScatterInfo &info)
{
    return scatter_layer_internal<T>(src, updates, indices, out_shape, info);
}

template SimpleTensor<float> scatter_layer(const SimpleTensor<float> &src, const SimpleTensor<float> &updates, const SimpleTensor<uint32_t> &indices, const TensorShape &out_shape, const ScatterInfo &info);

} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
