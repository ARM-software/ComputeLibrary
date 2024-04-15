/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "NormalizePlanarYUVLayer.h"

#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
// NormalizePlanarYUV Layer for floating point type
template <typename T>
SimpleTensor<T> normalize_planar_yuv_layer(const SimpleTensor<T> &src, const SimpleTensor<T> &mean, const SimpleTensor<T> &std)
{
    SimpleTensor<T> result(src.shape(), src.data_type());

    const auto cols       = static_cast<int>(src.shape()[0]);
    const auto rows       = static_cast<int>(src.shape()[1]);
    const auto depth      = static_cast<int>(src.shape()[2]);
    const int  upper_dims = src.shape().total_size() / (cols * rows * depth);

    for(int r = 0; r < upper_dims; ++r)
    {
        for(int i = 0; i < depth; ++i)
        {
            for(int k = 0; k < rows; ++k)
            {
                for(int l = 0; l < cols; ++l)
                {
                    const int pos = l + k * cols + i * rows * cols + r * cols * rows * depth;
                    result[pos]   = (src[pos] - mean[i]) / std[i];
                }
            }
        }
    }
    return result;
}

template <>
SimpleTensor<uint8_t> normalize_planar_yuv_layer<uint8_t>(const SimpleTensor<uint8_t> &src, const SimpleTensor<uint8_t> &mean, const SimpleTensor<uint8_t> &std)
{
    SimpleTensor<float>   src_tmp  = convert_from_asymmetric(src);
    SimpleTensor<float>   mean_tmp = convert_from_asymmetric(mean);
    SimpleTensor<float>   std_tmp  = convert_from_asymmetric(std);
    SimpleTensor<float>   dst_tmp  = normalize_planar_yuv_layer<float>(src_tmp, mean_tmp, std_tmp);
    SimpleTensor<uint8_t> dst      = convert_to_asymmetric<uint8_t>(dst_tmp, src.quantization_info());
    return dst;
}

template <>
SimpleTensor<int8_t> normalize_planar_yuv_layer<int8_t>(const SimpleTensor<int8_t> &src, const SimpleTensor<int8_t> &mean, const SimpleTensor<int8_t> &std)
{
    SimpleTensor<float>  src_tmp  = convert_from_asymmetric(src);
    SimpleTensor<float>  mean_tmp = convert_from_asymmetric(mean);
    SimpleTensor<float>  std_tmp  = convert_from_asymmetric(std);
    SimpleTensor<float>  dst_tmp  = normalize_planar_yuv_layer<float>(src_tmp, mean_tmp, std_tmp);
    SimpleTensor<int8_t> dst      = convert_to_asymmetric<int8_t>(dst_tmp, src.quantization_info());
    return dst;
}

template SimpleTensor<half> normalize_planar_yuv_layer(const SimpleTensor<half> &src, const SimpleTensor<half> &mean, const SimpleTensor<half> &std);
template SimpleTensor<float> normalize_planar_yuv_layer(const SimpleTensor<float> &src, const SimpleTensor<float> &mean, const SimpleTensor<float> &std);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
