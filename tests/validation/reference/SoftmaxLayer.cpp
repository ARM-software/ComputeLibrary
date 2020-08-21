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
#include "SoftmaxLayer.h"

#include "arm_compute/core/Helpers.h"
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
SimpleTensor<T> softmax_layer_generic(const SimpleTensor<T> &src, float beta, int32_t reduce_end_axis, bool is_log)
{
    // Create reference
    SimpleTensor<T> dst{ src.shape(), src.data_type(), 1 };

    // Convert reduce-before axis (inclusive) to first n axes to reduce
    const size_t first_n_reduce_axes = dim_index_2_num_dims(reduce_end_axis, static_cast<int32_t>(src.shape().num_dimensions()));

    // Compute reference. Lower dims are the collapsing of the first axis
    // dimensions (i.e., the flattened dimension of each batch). The upper dims are
    // instead the batches we want to normalize

    const int lower_dims = src.shape().total_size_lower(first_n_reduce_axes);

    const int upper_dims = src.shape().total_size_upper(first_n_reduce_axes);

#if defined(_OPENMP)
    #pragma omp parallel for
#endif /* _OPENMP */
    for(int r = 0; r < upper_dims; ++r)
    {
        const T *src_row_ptr = src.data() + r * lower_dims;
        T       *dst_row_ptr = dst.data() + r * lower_dims;

        // Find max
        const T max = *std::max_element(src_row_ptr, src_row_ptr + lower_dims);

        // Regularize
        T sum(0.f);
        std::transform(src_row_ptr, src_row_ptr + lower_dims, dst_row_ptr, [&sum, max, beta, is_log](T val)
        {
            T res{ (val - max) *beta };

            if(is_log)
            {
                sum += std::exp(res);
            }
            else
            {
                res = std::exp(res);
                sum += res;
            }
            return res;
        });

        // Normalize
        std::transform(dst_row_ptr, dst_row_ptr + lower_dims, dst_row_ptr, [sum, is_log](T val)
        {
            if(is_log)
            {
                return val - static_cast<T>(std::log(sum));
            }
            else
            {
                return val / sum;
            }
        });
    }

    return dst;
}

template SimpleTensor<float> softmax_layer_generic(const SimpleTensor<float> &src, float beta, int32_t reduce_end_axis, bool is_log);
template SimpleTensor<half> softmax_layer_generic(const SimpleTensor<half> &src, float beta, int32_t reduce_end_axis, bool is_log);

template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type>
SimpleTensor<T> softmax_layer(const SimpleTensor<T> &src, float beta, int32_t reduce_end_axis)
{
    return softmax_layer_generic<T>(src, beta, reduce_end_axis, false);
}

template < typename T, typename std::enable_if < std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value, int >::type >
SimpleTensor<T> softmax_layer(const SimpleTensor<T> &src, float beta, int32_t reduce_end_axis)
{
    const QuantizationInfo output_quantization_info = arm_compute::get_softmax_output_quantization_info(src.data_type(), false);

    SimpleTensor<float> src_tmp = convert_from_asymmetric(src);
    SimpleTensor<float> dst_tmp = softmax_layer<float>(src_tmp, beta, reduce_end_axis);
    SimpleTensor<T>     dst     = convert_to_asymmetric<T>(dst_tmp, output_quantization_info);
    return dst;
}

template SimpleTensor<float> softmax_layer(const SimpleTensor<float> &src, float beta, int32_t reduce_end_axis);
template SimpleTensor<half> softmax_layer(const SimpleTensor<half> &src, float beta, int32_t reduce_end_axis);
template SimpleTensor<uint8_t> softmax_layer(const SimpleTensor<uint8_t> &src, float beta, int32_t reduce_end_axis);
template SimpleTensor<int8_t> softmax_layer(const SimpleTensor<int8_t> &src, float beta, int32_t reduce_end_axis);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
