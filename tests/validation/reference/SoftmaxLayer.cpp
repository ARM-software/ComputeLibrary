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
#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type>
SimpleTensor<T> softmax_layer_generic(const SimpleTensor<T> &src, float beta, int32_t axis, bool is_log)
{
    // Create reference
    SimpleTensor<T> dst{ src.shape(), src.data_type(), 1 };

    const int32_t n_dims = static_cast<int32_t>(src.shape().num_dimensions());
    ARM_COMPUTE_ERROR_ON(axis < -n_dims || axis >= n_dims);

    const unsigned int actual_axis = static_cast<unsigned int>(wrap_around(axis, n_dims));
    Window             window;
    window.use_tensor_dimensions(src.shape());
    const unsigned int axis_dimension = src.shape()[actual_axis];
    window.set(actual_axis, Window::Dimension(0, 1, 1));

    execute_window_loop(window, [&](const Coordinates & id)
    {
        // Find max along axis
        Coordinates offset(id);
        offset.set(actual_axis, 0);
        T max = *reinterpret_cast<const T *>(src(offset));
        for(unsigned int axis_id = 1; axis_id < axis_dimension; ++axis_id)
        {
            offset.set(actual_axis, axis_id);
            const T val = *reinterpret_cast<const T *>(src(offset));
            if(val > max)
            {
                max = val;
            }
        }

        // Regularize
        T sum(0.f);
        for(unsigned int axis_id = 0; axis_id < axis_dimension; ++axis_id)
        {
            offset.set(actual_axis, axis_id);
            const T val = *reinterpret_cast<const T *>(src(offset));
            T       res{ (val - max) *beta };
            if(is_log)
            {
                sum += std::exp(res);
            }
            else
            {
                res = std::exp(res);
                sum += res;
            }
            *reinterpret_cast<T *>(dst(offset)) = res;
        }

        // Normalize
        for(unsigned int axis_id = 0; axis_id < axis_dimension; ++axis_id)
        {
            offset.set(actual_axis, axis_id);
            const T val = *reinterpret_cast<const T *>(dst(offset));
            if(is_log)
            {
                *reinterpret_cast<T *>(dst(offset)) = val - static_cast<T>(std::log(sum));
            }
            else
            {
                *reinterpret_cast<T *>(dst(offset)) = val / sum;
            }
        }
    });
    return dst;
}

template SimpleTensor<float> softmax_layer_generic(const SimpleTensor<float> &src, float beta, int32_t axis, bool is_log);
template SimpleTensor<half> softmax_layer_generic(const SimpleTensor<half> &src, float beta, int32_t axis, bool is_log);

template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type>
SimpleTensor<T> softmax_layer(const SimpleTensor<T> &src, float beta, int32_t axis, bool is_log)
{
    return softmax_layer_generic<T>(src, beta, axis, is_log);
}

template < typename T, typename std::enable_if < std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value, int >::type >
SimpleTensor<T> softmax_layer(const SimpleTensor<T> &src, float beta, int32_t axis, bool is_log)
{
    const QuantizationInfo output_quantization_info = arm_compute::get_softmax_output_quantization_info(src.data_type(), is_log);

    SimpleTensor<float> src_tmp = convert_from_asymmetric(src);
    SimpleTensor<float> dst_tmp = softmax_layer<float>(src_tmp, beta, axis, is_log);
    SimpleTensor<T>     dst     = convert_to_asymmetric<T>(dst_tmp, output_quantization_info);
    return dst;
}

template SimpleTensor<float> softmax_layer(const SimpleTensor<float> &src, float beta, int32_t axis, bool is_log);
template SimpleTensor<half> softmax_layer(const SimpleTensor<half> &src, float beta, int32_t axis, bool is_log);
template SimpleTensor<uint8_t> softmax_layer(const SimpleTensor<uint8_t> &src, float beta, int32_t axis, bool is_log);
template SimpleTensor<int8_t> softmax_layer(const SimpleTensor<int8_t> &src, float beta, int32_t axis, bool is_log);

} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
