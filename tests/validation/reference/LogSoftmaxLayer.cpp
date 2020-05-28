/*
 * Copyright (c) 2019-2020 ARM Limited.
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
#include "LogSoftmaxLayer.h"
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
SimpleTensor<T> log_softmax_layer(const SimpleTensor<T> &src, float beta, int32_t reduce_end_axis)
{
    return softmax_layer_generic<T>(src, beta, reduce_end_axis, true);
}

template < typename T, typename std::enable_if < std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value, int >::type >
SimpleTensor<T> log_softmax_layer(const SimpleTensor<T> &src, float beta, int32_t reduce_end_axis)
{
    const QuantizationInfo output_quantization_info = arm_compute::get_softmax_output_quantization_info(src.data_type(), true);

    SimpleTensor<float> src_tmp = convert_from_asymmetric(src);
    SimpleTensor<float> dst_tmp = log_softmax_layer<float>(src_tmp, beta, reduce_end_axis);
    SimpleTensor<T>     dst     = convert_to_asymmetric<T>(dst_tmp, output_quantization_info);
    return dst;
}

template SimpleTensor<float> log_softmax_layer(const SimpleTensor<float> &src, float beta, int32_t reduce_end_axis);
template SimpleTensor<half> log_softmax_layer(const SimpleTensor<half> &src, float beta, int32_t reduce_end_axis);
template SimpleTensor<uint8_t> log_softmax_layer(const SimpleTensor<uint8_t> &src, float beta, int32_t reduce_end_axis);
template SimpleTensor<int8_t> log_softmax_layer(const SimpleTensor<int8_t> &src, float beta, int32_t reduce_end_axis);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
