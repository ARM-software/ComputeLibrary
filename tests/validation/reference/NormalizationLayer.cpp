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
#include "NormalizationLayer.h"

#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> normalization_layer(const SimpleTensor<T> &src, NormalizationLayerInfo info)
{
    // Create reference
    SimpleTensor<T> dst{ src.shape(), src.data_type(), 1 };

    // Compute reference
    const uint32_t norm_size = info.norm_size();
    NormType       type      = info.type();
    float          beta      = info.beta();
    uint32_t       kappa     = info.kappa();

    const int cols       = src.shape()[0];
    const int rows       = src.shape()[1];
    const int depth      = src.shape()[2];
    int       upper_dims = src.shape().total_size() / (cols * rows);

    float coeff       = info.scale_coeff();
    int   radius_cols = norm_size / 2;

    // IN_MAP_1D and CROSS_MAP normalize over a single axis only
    int radius_rows = (NormType::IN_MAP_2D == type) ? norm_size / 2 : 0;

    if(info.is_cross_map())
    {
        // Remove also depth from upper dimensions since it is the dimension we
        // want to use for normalization
        upper_dims /= depth;

        for(int r = 0; r < upper_dims; ++r)
        {
            for(int i = 0; i < rows; ++i)
            {
                for(int k = 0; k < cols; ++k)
                {
                    for(int l = 0; l < depth; ++l)
                    {
                        float accumulated_scale = 0.f;

                        for(int j = -radius_cols; j <= radius_cols; ++j)
                        {
                            const int z = l + j;

                            if(z >= 0 && z < depth)
                            {
                                const T value = src[k + i * cols + z * rows * cols + r * cols * rows * depth];
                                accumulated_scale += value * value;
                            }
                        }

                        dst[k + i * cols + l * rows * cols + r * cols * rows * depth] = kappa + accumulated_scale * coeff;
                    }
                }
            }
        }
    }
    else
    {
        for(int r = 0; r < upper_dims; ++r)
        {
            for(int i = 0; i < rows; ++i)
            {
                for(int k = 0; k < cols; ++k)
                {
                    float accumulated_scale = 0.f;

                    for(int j = -radius_rows; j <= radius_rows; ++j)
                    {
                        const int y = i + j;
                        for(int l = -radius_cols; l <= radius_cols; ++l)
                        {
                            const int x = k + l;

                            if((x >= 0 && y >= 0) && (x < cols && y < rows))
                            {
                                const T value = src[x + y * cols + r * cols * rows];
                                accumulated_scale += value * value;
                            }
                        }
                    }

                    dst[k + i * cols + r * cols * rows] = kappa + accumulated_scale * coeff;
                }
            }
        }
    }

    if(beta == 1.f)
    {
        for(int i = 0; i < dst.num_elements(); ++i)
        {
            dst[i] = src[i] / dst[i];
        }
    }
    else if(beta == 0.5f)
    {
        for(int i = 0; i < dst.num_elements(); ++i)
        {
            dst[i] = src[i] / std::sqrt(dst[i]);
        }
    }
    else
    {
        for(int i = 0; i < dst.num_elements(); ++i)
        {
            dst[i] = src[i] * std::exp(std::log(dst[i]) * -beta);
        }
    }

    return dst;
}

template SimpleTensor<float> normalization_layer(const SimpleTensor<float> &src, NormalizationLayerInfo info);
template SimpleTensor<half> normalization_layer(const SimpleTensor<half> &src, NormalizationLayerInfo info);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
