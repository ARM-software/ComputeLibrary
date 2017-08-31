/*
 * Copyright (c) 2017 ARM Limited.
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
#include "PoolingLayer.h"

#include "arm_compute/core/Types.h"
#include "tests/validation/FixedPoint.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
namespace
{
TensorShape calculate_output_shape(TensorShape shape, PoolingLayerInfo info)
{
    TensorShape dst_shape = shape;
    const std::pair<unsigned int, unsigned int> scaled_dims = arm_compute::scaled_dimensions(shape.x(),
                                                                                             shape.y(),
                                                                                             info.pool_size(),
                                                                                             info.pool_size(),
                                                                                             info.pad_stride_info());
    dst_shape.set(0, scaled_dims.first);
    dst_shape.set(1, scaled_dims.second);

    return dst_shape;
}
} // namespace

template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type>
SimpleTensor<T> pooling_layer(const SimpleTensor<T> &src, PoolingLayerInfo info)
{
    const int   pool_size     = info.pool_size();
    PoolingType type          = info.pool_type();
    int         pool_stride_x = info.pad_stride_info().stride().first;
    int         pool_stride_y = info.pad_stride_info().stride().second;
    int         pad_x         = info.pad_stride_info().pad().first;
    int         pad_y         = info.pad_stride_info().pad().second;

    const auto w_src      = static_cast<int>(src.shape()[0]);
    const auto h_src      = static_cast<int>(src.shape()[1]);
    const int  upper_dims = src.shape().total_size() / (w_src * h_src);

    // Create reference
    SimpleTensor<T> dst{ calculate_output_shape(src.shape(), info), src.data_type(), 1, src.fixed_point_position() };

    const auto w_dst = static_cast<int>(dst.shape()[0]);
    const auto h_dst = static_cast<int>(dst.shape()[1]);

    if(type == PoolingType::MAX)
    {
        for(int r = 0; r < upper_dims; ++r)
        {
            for(int h = 0; h < h_dst; ++h)
            {
                for(int w = 0; w < w_dst; ++w)
                {
                    int wstart = w * pool_stride_x - pad_x;
                    int hstart = h * pool_stride_y - pad_y;
                    int wend   = std::min(wstart + pool_size, w_src);
                    int hend   = std::min(hstart + pool_size, h_src);
                    wstart     = std::max(wstart, 0);
                    hstart     = std::max(hstart, 0);

                    T max_val = std::numeric_limits<T>::lowest();
                    for(int y = hstart; y < hend; ++y)
                    {
                        for(int x = wstart; x < wend; ++x)
                        {
                            const T val = src[r * h_src * w_src + y * w_src + x];
                            if(val > max_val)
                            {
                                max_val = val;
                            }
                        }
                    }

                    dst[r * h_dst * w_dst + h * w_dst + w] = max_val;
                }
            }
        }
    }
    else // Average pooling
    {
        for(int r = 0; r < upper_dims; ++r)
        {
            for(int h = 0; h < h_dst; ++h)
            {
                for(int w = 0; w < w_dst; ++w)
                {
                    T   avg_val(0);
                    int wstart = w * pool_stride_x - pad_x;
                    int hstart = h * pool_stride_y - pad_y;
                    int wend   = std::min(wstart + pool_size, w_src + pad_x);
                    int hend   = std::min(hstart + pool_size, h_src + pad_y);
                    int pool   = (hend - hstart) * (wend - wstart);
                    wstart     = std::max(wstart, 0);
                    hstart     = std::max(hstart, 0);
                    wend       = std::min(wend, w_src);
                    hend       = std::min(hend, h_src);

                    for(int y = hstart; y < hend; ++y)
                    {
                        for(int x = wstart; x < wend; ++x)
                        {
                            avg_val += src[r * h_src * w_src + y * w_src + x];
                        }
                    }
                    dst[r * h_dst * w_dst + h * w_dst + w] = avg_val / pool;
                }
            }
        }
    }

    return dst;
}

template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type>
SimpleTensor<T> pooling_layer(const SimpleTensor<T> &src, PoolingLayerInfo info)
{
    const int   pool_size     = info.pool_size();
    PoolingType type          = info.pool_type();
    int         pool_stride_x = info.pad_stride_info().stride().first;
    int         pool_stride_y = info.pad_stride_info().stride().second;
    int         pad_x         = info.pad_stride_info().pad().first;
    int         pad_y         = info.pad_stride_info().pad().second;

    const auto w_src      = static_cast<int>(src.shape()[0]);
    const auto h_src      = static_cast<int>(src.shape()[1]);
    const int  upper_dims = src.shape().total_size() / (w_src * h_src);

    // Create reference
    SimpleTensor<T> dst{ calculate_output_shape(src.shape(), info), src.data_type(), 1, src.fixed_point_position() };

    const auto w_dst = static_cast<int>(dst.shape()[0]);
    const auto h_dst = static_cast<int>(dst.shape()[1]);

    if(type == PoolingType::MAX)
    {
        for(int r = 0; r < upper_dims; ++r)
        {
            for(int h = 0; h < h_dst; ++h)
            {
                for(int w = 0; w < w_dst; ++w)
                {
                    int wstart = w * pool_stride_x - pad_x;
                    int hstart = h * pool_stride_y - pad_y;
                    int wend   = std::min(wstart + pool_size, w_src);
                    int hend   = std::min(hstart + pool_size, h_src);
                    wstart     = std::max(wstart, 0);
                    hstart     = std::max(hstart, 0);

                    T max_val = std::numeric_limits<T>::lowest();
                    for(int y = hstart; y < hend; ++y)
                    {
                        for(int x = wstart; x < wend; ++x)
                        {
                            const T val = src[r * h_src * w_src + y * w_src + x];
                            if(val > max_val)
                            {
                                max_val = val;
                            }
                        }
                    }

                    dst[r * h_dst * w_dst + h * w_dst + w] = max_val;
                }
            }
        }
    }
    else // Average pooling
    {
        for(int r = 0; r < upper_dims; ++r)
        {
            for(int h = 0; h < h_dst; ++h)
            {
                for(int w = 0; w < w_dst; ++w)
                {
                    int wstart = w * pool_stride_x - pad_x;
                    int hstart = h * pool_stride_y - pad_y;
                    int wend   = std::min(wstart + pool_size, w_src + pad_x);
                    int hend   = std::min(hstart + pool_size, h_src + pad_y);
                    int pool   = (hend - hstart) * (wend - wstart);
                    wstart     = std::max(wstart, 0);
                    hstart     = std::max(hstart, 0);
                    wend       = std::min(wend, w_src);
                    hend       = std::min(hend, h_src);

                    using namespace fixed_point_arithmetic;

                    const int            fixed_point_position = src.fixed_point_position();
                    const fixed_point<T> invpool_fp(1.f / static_cast<float>(pool), fixed_point_position);
                    fixed_point<T>       avg_val(0, fixed_point_position, true);

                    for(int y = hstart; y < hend; ++y)
                    {
                        for(int x = wstart; x < wend; ++x)
                        {
                            const fixed_point<T> in_fp(src[r * h_src * w_src + y * w_src + x], fixed_point_position, true);
                            avg_val = add(avg_val, in_fp);
                        }
                    }
                    dst[r * h_dst * w_dst + h * w_dst + w] = mul(avg_val, invpool_fp).raw();
                }
            }
        }
    }

    return dst;
}

template SimpleTensor<float> pooling_layer(const SimpleTensor<float> &src, PoolingLayerInfo info);
template SimpleTensor<half> pooling_layer(const SimpleTensor<half> &src, PoolingLayerInfo info);
template SimpleTensor<qint8_t> pooling_layer(const SimpleTensor<qint8_t> &src, PoolingLayerInfo info);
template SimpleTensor<qint16_t> pooling_layer(const SimpleTensor<qint16_t> &src, PoolingLayerInfo info);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
