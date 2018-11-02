/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "tests/validation/FixedPoint.h"
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

template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type>
SimpleTensor<T> pooling_layer(const SimpleTensor<T> &src, const PoolingLayerInfo &info)
{
    ARM_COMPUTE_ERROR_ON(info.is_global_pooling() && (src.shape().x() != src.shape().y()));

    // Create reference
    SimpleTensor<T> dst{ compute_pool_shape(TensorInfo(src.shape(), 1, src.data_type(), src.fixed_point_position()), info), src.data_type(), 1, src.fixed_point_position() };

    const int   pool_size_x     = info.is_global_pooling() ? src.shape().x() : info.pool_size().width;
    const int   pool_size_y     = info.is_global_pooling() ? src.shape().y() : info.pool_size().height;
    PoolingType type            = info.pool_type();
    int         pool_stride_x   = info.pad_stride_info().stride().first;
    int         pool_stride_y   = info.pad_stride_info().stride().second;
    int         pad_left        = info.pad_stride_info().pad_left();
    int         pad_top         = info.pad_stride_info().pad_top();
    int         pad_right       = info.pad_stride_info().pad_right();
    int         pad_bottom      = info.pad_stride_info().pad_bottom();
    bool        exclude_padding = info.exclude_padding();

    const auto w_src      = static_cast<int>(src.shape()[0]);
    const auto h_src      = static_cast<int>(src.shape()[1]);
    const int  upper_dims = src.shape().total_size() / (w_src * h_src);

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
                    int wstart = w * pool_stride_x - pad_left;
                    int hstart = h * pool_stride_y - pad_top;
                    int wend   = std::min(wstart + pool_size_x, w_src);
                    int hend   = std::min(hstart + pool_size_y, h_src);
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
    else // Average or l2 pooling
    {
        for(int r = 0; r < upper_dims; ++r)
        {
            for(int h = 0; h < h_dst; ++h)
            {
                for(int w = 0; w < w_dst; ++w)
                {
                    T   avg_val(0);
                    int wstart = w * pool_stride_x - pad_left;
                    int hstart = h * pool_stride_y - pad_top;
                    int wend   = std::min(wstart + pool_size_x, w_src + pad_right);
                    int hend   = std::min(hstart + pool_size_y, h_src + pad_bottom);
                    int pool   = (hend - hstart) * (wend - wstart);
                    wstart     = std::max(wstart, 0);
                    hstart     = std::max(hstart, 0);
                    wend       = std::min(wend, w_src);
                    hend       = std::min(hend, h_src);
                    // Exclude padding pixels from the average
                    if(exclude_padding)
                    {
                        pool = (hend - hstart) * (wend - wstart);
                    }

                    if(type == PoolingType::AVG)
                    {
                        for(int y = hstart; y < hend; ++y)
                        {
                            for(int x = wstart; x < wend; ++x)
                            {
                                avg_val += src[r * h_src * w_src + y * w_src + x];
                            }
                        }
                        dst[r * h_dst * w_dst + h * w_dst + w] = avg_val / pool;
                    }
                    else
                    {
                        for(int y = hstart; y < hend; ++y)
                        {
                            for(int x = wstart; x < wend; ++x)
                            {
                                const T val = src[r * h_src * w_src + y * w_src + x];
                                avg_val += val * val;
                            }
                        }
                        dst[r * h_dst * w_dst + h * w_dst + w] = std::sqrt(avg_val / pool);
                    }
                }
            }
        }
    }

    return dst;
}

template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type>
SimpleTensor<T> pooling_layer(const SimpleTensor<T> &src, const PoolingLayerInfo &info)
{
    ARM_COMPUTE_ERROR_ON(info.is_global_pooling() && (src.shape().x() != src.shape().y()));

    const auto w_src      = static_cast<int>(src.shape()[0]);
    const auto h_src      = static_cast<int>(src.shape()[1]);
    const int  upper_dims = src.shape().total_size() / (w_src * h_src);

    const int   pool_size_x     = info.is_global_pooling() ? src.shape().x() : info.pool_size().width;
    const int   pool_size_y     = info.is_global_pooling() ? src.shape().y() : info.pool_size().height;
    PoolingType type            = info.pool_type();
    int         pool_stride_x   = info.pad_stride_info().stride().first;
    int         pool_stride_y   = info.pad_stride_info().stride().second;
    int         pad_left        = info.pad_stride_info().pad_left();
    int         pad_top         = info.pad_stride_info().pad_top();
    int         pad_right       = info.pad_stride_info().pad_right();
    int         pad_bottom      = info.pad_stride_info().pad_bottom();
    bool        exclude_padding = info.exclude_padding();

    // Create reference
    SimpleTensor<T> dst{ compute_pool_shape(TensorInfo(src.shape(), 1, src.data_type(), src.fixed_point_position()), info), src.data_type(), 1, src.fixed_point_position() };

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
                    int wstart = w * pool_stride_x - pad_left;
                    int hstart = h * pool_stride_y - pad_top;
                    int wend   = std::min(wstart + pool_size_x, w_src);
                    int hend   = std::min(hstart + pool_size_y, h_src);
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
    else // Average or l2 pooling
    {
        for(int r = 0; r < upper_dims; ++r)
        {
            for(int h = 0; h < h_dst; ++h)
            {
                for(int w = 0; w < w_dst; ++w)
                {
                    int wstart = w * pool_stride_x - pad_left;
                    int hstart = h * pool_stride_y - pad_top;
                    int wend   = std::min(wstart + pool_size_x, w_src + pad_right);
                    int hend   = std::min(hstart + pool_size_y, h_src + pad_bottom);
                    int pool   = (hend - hstart) * (wend - wstart);
                    wstart     = std::max(wstart, 0);
                    hstart     = std::max(hstart, 0);
                    wend       = std::min(wend, w_src);
                    hend       = std::min(hend, h_src);
                    // Exclude padding pixels from the average
                    if(exclude_padding)
                    {
                        pool = (hend - hstart) * (wend - wstart);
                    }

                    using namespace fixed_point_arithmetic;

                    const int            fixed_point_position = src.fixed_point_position();
                    const fixed_point<T> const_1(1, fixed_point_position);
                    const fixed_point<T> invpool_fp(1.f / static_cast<float>(pool), fixed_point_position);
                    fixed_point<T>       avg_val(0, fixed_point_position, true);

                    if(type == PoolingType::AVG)
                    {
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
                    else
                    {
                        for(int y = hstart; y < hend; ++y)
                        {
                            for(int x = wstart; x < wend; ++x)
                            {
                                const fixed_point<T> in_fp(src[r * h_src * w_src + y * w_src + x], fixed_point_position, true);
                                avg_val = add(avg_val, mul(in_fp, in_fp));
                            }
                        }
                        auto res                               = div(const_1, (inv_sqrt(mul(avg_val, invpool_fp))));
                        dst[r * h_dst * w_dst + h * w_dst + w] = res.raw();
                    }
                }
            }
        }
    }

    return dst;
}

template <>
SimpleTensor<uint8_t> pooling_layer<uint8_t>(const SimpleTensor<uint8_t> &src, const PoolingLayerInfo &info)
{
    SimpleTensor<float>   src_tmp = convert_from_asymmetric(src);
    SimpleTensor<float>   dst_tmp = pooling_layer<float>(src_tmp, info);
    SimpleTensor<uint8_t> dst     = convert_to_asymmetric(dst_tmp, src.quantization_info());
    return dst;
}

template SimpleTensor<float> pooling_layer(const SimpleTensor<float> &src, const PoolingLayerInfo &info);
template SimpleTensor<half> pooling_layer(const SimpleTensor<half> &src, const PoolingLayerInfo &info);
template SimpleTensor<qint8_t> pooling_layer(const SimpleTensor<qint8_t> &src, const PoolingLayerInfo &info);
template SimpleTensor<qint16_t> pooling_layer(const SimpleTensor<qint16_t> &src, const PoolingLayerInfo &info);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
