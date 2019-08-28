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
#include "SoftmaxLayer.h"

#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type>
SimpleTensor<T> softmax_layer(const SimpleTensor<T> &src, float beta, size_t axis)
{
    // Create reference
    SimpleTensor<T> dst{ src.shape(), src.data_type(), 1 };

    // Compute reference. Lower dims are the collapsing of the first axis
    // dimensions (i.e., the flattened dimension of each batch). The upper dims are
    // instead the batches we want to normalize

    int lower_dims = 1;
    for(size_t i = 0; i < axis; i++)
    {
        lower_dims *= src.shape()[i];
    }

    int upper_dims = 1;
    for(size_t i = axis; i < TensorShape::num_max_dimensions; i++)
    {
        upper_dims *= src.shape()[i];
    }

    for(int r = 0; r < upper_dims; ++r)
    {
        const T *src_row_ptr = src.data() + r * lower_dims;
        T       *dst_row_ptr = dst.data() + r * lower_dims;

        // Find max
        const T max = *std::max_element(src_row_ptr, src_row_ptr + lower_dims);

        // Regularize
        T sum(0.f);
        std::transform(src_row_ptr, src_row_ptr + lower_dims, dst_row_ptr, [&sum, max, beta](T val)
        {
            const T res(std::exp((val - max) * beta));
            sum += res;
            return res;
        });

        // Normalize
        std::transform(dst_row_ptr, dst_row_ptr + lower_dims, dst_row_ptr, [sum](T val)
        {
            return val / sum;
        });
    }

    return dst;
}

template <typename T, typename std::enable_if<std::is_same<T, uint8_t>::value, int>::type>
SimpleTensor<T> softmax_layer(const SimpleTensor<T> &src, float beta, size_t axis)
{
    // Note: Output quantization info should always have scale = 1/256 and offset = 0
    const QuantizationInfo output_quantization_info = QuantizationInfo(1.f / 256, 0);

    SimpleTensor<float> src_tmp = convert_from_asymmetric(src);
    SimpleTensor<float> dst_tmp = softmax_layer<float>(src_tmp, beta, axis);
    SimpleTensor<T>     dst     = convert_to_asymmetric<uint8_t>(dst_tmp, output_quantization_info);
    return dst;
}

template SimpleTensor<float> softmax_layer(const SimpleTensor<float> &src, float beta, size_t axis);
template SimpleTensor<half> softmax_layer(const SimpleTensor<half> &src, float beta, size_t axis);
template SimpleTensor<uint8_t> softmax_layer(const SimpleTensor<uint8_t> &src, float beta, size_t axis);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
