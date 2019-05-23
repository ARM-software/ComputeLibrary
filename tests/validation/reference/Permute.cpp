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
#include "Permute.h"

#include "arm_compute/core/Types.h"
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
SimpleTensor<T> permute(const SimpleTensor<T> &src, PermutationVector perm)
{
    // Permute shapes
    TensorShape dst_shape = src.shape();
    permute(dst_shape, perm);

    // Create reference
    SimpleTensor<T> dst{ dst_shape, src.data_type(), src.num_channels(), src.quantization_info() };

    // Compute reference
    for(int i = 0; i < src.num_elements(); ++i)
    {
        const Coordinates src_coords = index2coord(src.shape(), i);
        Coordinates       dst_coords = src_coords;
        permute(dst_coords, perm);

        std::copy_n(static_cast<const T *>(src(src_coords)), src.num_channels(), static_cast<T *>(dst(dst_coords)));
    }

    return dst;
}

template SimpleTensor<int8_t> permute(const SimpleTensor<int8_t> &src, PermutationVector perm);
template SimpleTensor<uint8_t> permute(const SimpleTensor<uint8_t> &src, PermutationVector perm);
template SimpleTensor<int16_t> permute(const SimpleTensor<int16_t> &src, PermutationVector perm);
template SimpleTensor<uint16_t> permute(const SimpleTensor<uint16_t> &src, PermutationVector perm);
template SimpleTensor<uint32_t> permute(const SimpleTensor<uint32_t> &src, PermutationVector perm);
template SimpleTensor<float> permute(const SimpleTensor<float> &src, PermutationVector perm);
template SimpleTensor<half> permute(const SimpleTensor<half> &src, PermutationVector perm);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
