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
#include "ActivationLayer.h"

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
SimpleTensor<T> activation_layer(const SimpleTensor<T> &src, ActivationLayerInfo info, const QuantizationInfo &oq_info)
{
    ARM_COMPUTE_UNUSED(oq_info);

    // Create reference
    SimpleTensor<T> dst{ src.shape(), src.data_type(), 1 };

    // Compute reference
    const T a(info.a());
    const T b(info.b());

    for(int i = 0; i < src.num_elements(); ++i)
    {
        dst[i] = activate_float<T>(src[i], a, b, info.activation());
    }

    return dst;
}

template <>
SimpleTensor<uint8_t> activation_layer<uint8_t>(const SimpleTensor<uint8_t> &src, ActivationLayerInfo info, const QuantizationInfo &oq_info)
{
    const QuantizationInfo dst_qinfo = oq_info.empty() ? src.quantization_info() : oq_info;

    SimpleTensor<float>   src_tmp = convert_from_asymmetric(src);
    SimpleTensor<float>   dst_tmp = activation_layer<float>(src_tmp, info);
    SimpleTensor<uint8_t> dst     = convert_to_asymmetric(dst_tmp, dst_qinfo);
    return dst;
}

template <>
SimpleTensor<int16_t> activation_layer<int16_t>(const SimpleTensor<int16_t> &src, ActivationLayerInfo info, const QuantizationInfo &oq_info)
{
    const QuantizationInfo dst_qinfo = oq_info.empty() ? src.quantization_info() : oq_info;

    SimpleTensor<float>   src_tmp = convert_from_symmetric(src);
    SimpleTensor<float>   dst_tmp = activation_layer<float>(src_tmp, info);
    SimpleTensor<int16_t> dst     = convert_to_symmetric<int16_t>(dst_tmp, dst_qinfo);
    return dst;
}

template SimpleTensor<float> activation_layer(const SimpleTensor<float> &src, ActivationLayerInfo info, const QuantizationInfo &oq_info);
template SimpleTensor<half> activation_layer(const SimpleTensor<half> &src, ActivationLayerInfo info, const QuantizationInfo &oq_info);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
