/*
 * Copyright (c) 2019 ARM Limited.
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
#include "InstanceNormalizationLayer.h"

#include "tests/validation/Helpers.h"

#include <algorithm>
#include <cmath>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> instance_normalization(const SimpleTensor<T> &src, float gamma, float beta, float epsilon)
{
    SimpleTensor<T> dst{ src.shape(), src.data_type() };

    //NCHW
    const size_t w_size = src.shape()[0];
    const size_t h_size = src.shape()[1];
    const size_t c_size = src.shape()[2];
    const size_t n_size = src.shape()[3];

    for(size_t n_i = 0; n_i < n_size; ++n_i)
    {
        for(size_t c_i = 0; c_i < c_size; ++c_i)
        {
            float sum_h_w = 0;
            //Compute mean
            for(size_t h_i = 0; h_i < h_size; ++h_i)
            {
                for(size_t w_i = 0; w_i < w_size; ++w_i)
                {
                    sum_h_w += src[coord2index(src.shape(), Coordinates(w_i, h_i, c_i, n_i))];
                }
            }
            const float mean_h_w = sum_h_w / (h_size * w_size);

            //Compute variance
            float partial_var_h_w = 0;
            for(size_t h_i = 0; h_i < h_size; ++h_i)
            {
                for(size_t w_i = 0; w_i < w_size; ++w_i)
                {
                    partial_var_h_w += std::pow(src[coord2index(src.shape(), Coordinates(w_i, h_i, c_i, n_i))] - mean_h_w, 2);
                }
            }
            const float var_h_w = partial_var_h_w / (h_size * w_size);

            //Apply mean
            for(size_t h_i = 0; h_i < h_size; ++h_i)
            {
                for(size_t w_i = 0; w_i < w_size; ++w_i)
                {
                    //Compute output
                    size_t index = coord2index(src.shape(), Coordinates(w_i, h_i, c_i, n_i));
                    dst[index]   = (src[index] - mean_h_w) * gamma / std::sqrt(var_h_w + epsilon) + beta;
                }
            }
        }
    }
    return dst;
}

template SimpleTensor<float> instance_normalization(const SimpleTensor<float> &src, float gamma, float beta, float epsilon);
template SimpleTensor<half> instance_normalization(const SimpleTensor<half> &src, float gamma, float beta, float epsilon);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
