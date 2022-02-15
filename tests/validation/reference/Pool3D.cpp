/*
 * Copyright (c) 2022 Arm Limited.
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

#include "Pool3D.h"
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
SimpleTensor<T> pool3d_internal(const SimpleTensor<T> &src, const Pool3DInfo &pool3d_info, SimpleTensor<uint32_t> *indices)
{
    TensorShape     pooled_shape = compute_pool3d_shape(src.shape(), pool3d_info);
    SimpleTensor<T> dst{ pooled_shape, src.data_type(), 1 };

    if(indices != nullptr)
    {
        *indices = SimpleTensor<uint32_t> { pooled_shape, DataType::U32, 1 };
    }

    const int idx_channel = 0;
    const int idx_width   = 1;
    const int idx_height  = 2;
    const int idx_depth   = 3;
    const int idx_batch   = 4;

    const int pool_size_width  = pool3d_info.is_global_pooling ? src.shape()[idx_width] : pool3d_info.pool_size.width;
    const int pool_size_height = pool3d_info.is_global_pooling ? src.shape()[idx_height] : pool3d_info.pool_size.height;
    const int pool_size_depth  = pool3d_info.is_global_pooling ? src.shape()[idx_depth] : pool3d_info.pool_size.depth;

    const int pool_stride_width  = static_cast<int>(pool3d_info.strides.width);
    const int pool_stride_height = static_cast<int>(pool3d_info.strides.height);
    const int pool_stride_depth  = static_cast<int>(pool3d_info.strides.depth);

    const int pad_left  = static_cast<int>(pool3d_info.padding.left);
    const int pad_top   = static_cast<int>(pool3d_info.padding.top);
    const int pad_front = static_cast<int>(pool3d_info.padding.front);

    const int pad_right  = static_cast<int>(pool3d_info.padding.right);
    const int pad_bottom = static_cast<int>(pool3d_info.padding.bottom);
    const int pad_back   = static_cast<int>(pool3d_info.padding.back);

    const int num_channels = static_cast<int>(src.shape()[idx_channel]);
    const int num_batches  = static_cast<int>(src.shape()[idx_batch]);

    ARM_COMPUTE_ERROR_ON(num_channels != static_cast<int>(dst.shape()[idx_channel]));
    ARM_COMPUTE_ERROR_ON(num_batches != static_cast<int>(dst.shape()[idx_batch]));

    const int w_src = static_cast<int>(src.shape()[idx_width]);
    const int h_src = static_cast<int>(src.shape()[idx_height]);
    const int d_src = static_cast<int>(src.shape()[idx_depth]);
    const int w_dst = static_cast<int>(dst.shape()[idx_width]);
    const int h_dst = static_cast<int>(dst.shape()[idx_height]);
    const int d_dst = static_cast<int>(dst.shape()[idx_depth]);

    const bool exclude_padding = pool3d_info.exclude_padding;

    const int height_stride_src = num_channels * w_src;
    const int depth_stride_src  = height_stride_src * h_src;
    const int batch_stride_src  = depth_stride_src * d_src;
    const int height_stride_dst = num_channels * w_dst;
    const int depth_stride_dst  = height_stride_dst * h_dst;
    const int batch_stride_dst  = depth_stride_dst * d_dst;

    for(int b = 0; b < num_batches; ++b)
    {
        const int batch_offset_dst = b * batch_stride_dst;
        const int batch_offset_src = b * batch_stride_src;
        for(int c = 0; c < num_channels; ++c)
        {
            for(int d = 0; d < d_dst; ++d)
            {
                const int depth_offset_dst = d * depth_stride_dst;
                for(int h = 0; h < h_dst; ++h)
                {
                    const int height_offset_dst = h * height_stride_dst;
                    for(int w = 0; w < w_dst; ++w)
                    {
                        int wstart = w * pool_stride_width - pad_left;
                        int hstart = h * pool_stride_height - pad_top;
                        int dstart = d * pool_stride_depth - pad_front;
                        int wend   = std::min(wstart + pool_size_width, w_src + pad_right);
                        int hend   = std::min(hstart + pool_size_height, h_src + pad_bottom);
                        int dend   = std::min(dstart + pool_size_depth, d_src + pad_back);

                        // this may not be equal to pool_w * pool_h * pool_d because of
                        // DimensionRoundingType choice (CEIL)
                        int pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);

                        // limit [start, end) to [0, w_src)
                        wstart = std::max(wstart, 0);
                        hstart = std::max(hstart, 0);
                        dstart = std::max(dstart, 0);
                        wend   = std::min(wend, w_src);
                        hend   = std::min(hend, h_src);
                        dend   = std::min(dend, d_src);

                        auto max_val = -std::numeric_limits<T>::infinity();
                        int  max_index{ 0 };
                        T    avg_val = static_cast<T>(0.f);
                        T    l2_val  = static_cast<T>(0.f);

                        if(exclude_padding)
                        {
                            pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
                        }

                        for(int z = dstart; z < dend; ++z)
                        {
                            const int depth_offset_src = z * depth_stride_src;
                            for(int y = hstart; y < hend; ++y)
                            {
                                const int height_offset_src = y * height_stride_src;
                                for(int x = wstart; x < wend; ++x)
                                {
                                    const auto val = static_cast<T>(
                                                         src[batch_offset_src + depth_offset_src + height_offset_src + x * num_channels + c]);

                                    if(val > max_val)
                                    {
                                        max_val   = val;
                                        max_index = coord2index(src.shape(), Coordinates(c, x, y, z, 0));
                                    }

                                    avg_val += val;
                                    l2_val += val * val;
                                }
                            }
                        }

                        avg_val /= pool_size;
                        l2_val = static_cast<T>(std::sqrt(l2_val / pool_size));

                        int dst_index = batch_offset_dst + depth_offset_dst + height_offset_dst + w * num_channels + c;
                        switch(pool3d_info.pool_type)
                        {
                            case PoolingType::MAX:
                                dst[dst_index] = static_cast<T>(max_val);
                                break;
                            case PoolingType::AVG:
                                dst[dst_index] = static_cast<T>(avg_val);
                                break;
                            case PoolingType::L2:
                                dst[dst_index] = static_cast<T>(l2_val);
                                break;
                            default:
                                ARM_COMPUTE_ERROR("Pooling Type should be either MAX, AVG or L2");
                        }

                        if(indices != nullptr)
                        {
                            (*indices)[dst_index] = max_index;
                        }
                    }
                }
            }
        }
    }

    return dst;
}

template SimpleTensor<float> pool3d(const SimpleTensor<float> &src, const Pool3DInfo &pool3d_info, const QuantizationInfo &output_qinfo, SimpleTensor<uint32_t> *indices);
template SimpleTensor<half> pool3d(const SimpleTensor<half> &src, const Pool3DInfo &pool3d_info, const QuantizationInfo &output_qinfo, SimpleTensor<uint32_t> *indices);

template <typename T>
SimpleTensor<T> pool3d(const SimpleTensor<T> &src, const Pool3DInfo &pool3d_info, const QuantizationInfo &output_qinfo, SimpleTensor<uint32_t> *indices)
{
    ARM_COMPUTE_UNUSED(output_qinfo);
    return pool3d_internal<T>(src, pool3d_info, indices);
}

template <>
SimpleTensor<int8_t> pool3d<int8_t>(const SimpleTensor<int8_t> &src, const Pool3DInfo &pool3d_info, const QuantizationInfo &output_qinfo, SimpleTensor<uint32_t> *indices)
{
    SimpleTensor<float> src_tmp = convert_from_asymmetric(src);
    SimpleTensor<float> dst_tmp = pool3d_internal<float>(src_tmp, pool3d_info, indices);
    return convert_to_asymmetric<int8_t>(dst_tmp, output_qinfo);
}

template <>
SimpleTensor<uint8_t> pool3d<uint8_t>(const SimpleTensor<uint8_t> &src, const Pool3DInfo &pool3d_info, const QuantizationInfo &output_qinfo, SimpleTensor<uint32_t> *indices)
{
    SimpleTensor<float> src_tmp = convert_from_asymmetric(src);
    SimpleTensor<float> dst_tmp = pool3d_internal<float>(src_tmp, pool3d_info, indices);
    return convert_to_asymmetric<uint8_t>(dst_tmp, output_qinfo);
}

} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute