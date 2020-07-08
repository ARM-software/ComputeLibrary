/*
 * Copyright (c) 2020 Arm Limited.
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
#include "MaxUnpoolingLayer.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
using namespace arm_compute::misc::shape_calculator;

template <typename T>
SimpleTensor<T> max_unpooling_layer_internal(const SimpleTensor<T> &src, const PoolingLayerInfo &info,
                                             const QuantizationInfo &output_qinfo, SimpleTensor<uint32_t> &indices,
                                             TensorShape output_shape, DataLayout data_layout)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_UNUSED(output_qinfo);
    ARM_COMPUTE_UNUSED(data_layout);
    // Create reference
    SimpleTensor<T> dst{ output_shape, src.data_type(), 1 };
    ARM_COMPUTE_ERROR_ON(indices.shape().total_size() == 0);
    std::fill_n(dst.data(), dst.num_elements(), 0);
    const auto w_indices = static_cast<int>(indices.shape()[0]);
    const auto h_indices = static_cast<int>(indices.shape()[1]);
    const auto z_indices = static_cast<int>(indices.shape()[2]);
    const auto b_indices = static_cast<int>(indices.shape()[3]);
    const auto w_dst     = static_cast<int>(dst.shape()[0]);
    const auto h_dst     = static_cast<int>(dst.shape()[1]);
    const auto z_dst     = static_cast<int>(dst.shape()[2]);
    for(int b = 0; b < b_indices; ++b)
    {
        for(int r = 0; r < z_indices; ++r)
        {
            for(int h = 0; h < h_indices; ++h)
            {
                for(int w = 0; w < w_indices; ++w)
                {
                    const uint32_t index_into_dst = indices[b * z_indices * h_indices * w_indices + r * h_indices * w_indices + h * w_indices + w];
                    const auto     input_val      = src[b * z_indices * h_indices * w_indices + r * h_indices * w_indices + h * w_indices + w];
                    auto          *ptr            = &dst[b * z_dst * h_dst * w_dst];
                    ptr[index_into_dst]           = input_val;
                }
            }
        }
    }
    return dst;
}

template <>
SimpleTensor<uint8_t> max_unpooling_layer<uint8_t>(
    const SimpleTensor<uint8_t> &src, const PoolingLayerInfo &info,
    const QuantizationInfo &output_qinfo, SimpleTensor<uint32_t> &indices,
    TensorShape output_shape, DataLayout data_layout)

{
    SimpleTensor<float>   src_tmp = convert_from_asymmetric(src);
    SimpleTensor<float>   dst_tmp = max_unpooling_layer_internal<float>(src_tmp, info, output_qinfo, indices, output_shape, data_layout);
    SimpleTensor<uint8_t> dst     = convert_to_asymmetric<uint8_t>(dst_tmp, output_qinfo);
    return dst;
}

template <typename T>
SimpleTensor<T> max_unpooling_layer(const SimpleTensor<T> &src, const PoolingLayerInfo &info,
                                    const QuantizationInfo &output_qinfo, SimpleTensor<uint32_t> &indices,
                                    TensorShape output_shape, DataLayout data_layout)
{
    return max_unpooling_layer_internal<T>(src, info, output_qinfo, indices, output_shape, data_layout);
}

template SimpleTensor<float> max_unpooling_layer(const SimpleTensor<float> &src, const PoolingLayerInfo &info,
                                                 const QuantizationInfo &output_qinfo, SimpleTensor<uint32_t> &indices,
                                                 TensorShape output_shape, DataLayout data_layout);

} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
