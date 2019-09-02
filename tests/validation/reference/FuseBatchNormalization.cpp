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
#include "FuseBatchNormalization.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
void fuse_batch_normalization_dwc_layer(const SimpleTensor<T> &w, const SimpleTensor<T> &mean, const SimpleTensor<T> &var, SimpleTensor<T> &w_fused, SimpleTensor<T> &b_fused, const SimpleTensor<T> &b,
                                        const SimpleTensor<T> &beta, const SimpleTensor<T> &gamma, float epsilon)
{
    const auto *w_data = w.data();
    const auto *b_data = b.data();

    auto *w_fused_data = w_fused.data();
    auto *b_fused_data = b_fused.data();

    const unsigned int width  = w.shape()[0];
    const unsigned int height = w.shape()[1];
    const unsigned int dim2   = w.shape()[2];

    for(unsigned int b = 0; b < dim2; ++b)
    {
        const auto mean_val  = mean.data()[b];
        const auto var_val   = var.data()[b];
        const auto beta_val  = beta.data()[b];
        const auto gamma_val = gamma.data()[b];

        for(unsigned int i = 0; i < width * height; ++i)
        {
            unsigned int index = i + b * width * height;

            w_fused_data[index] = (gamma_val * (w_data[index])) / sqrt(var_val + epsilon);
        }

        b_fused_data[b] = (b_data[b] - mean_val) / sqrt(var_val + epsilon) * gamma_val + beta_val;
    }
}

template <typename T>
void fuse_batch_normalization_conv_layer(const SimpleTensor<T> &w, const SimpleTensor<T> &mean, const SimpleTensor<T> &var, SimpleTensor<T> &w_fused, SimpleTensor<T> &b_fused,
                                         const SimpleTensor<T> &b,
                                         const SimpleTensor<T> &beta, const SimpleTensor<T> &gamma, float epsilon)
{
    const auto *w_data = w.data();
    const auto *b_data = b.data();

    auto *w_fused_data = w_fused.data();
    auto *b_fused_data = b_fused.data();

    const unsigned int width  = w.shape()[0];
    const unsigned int height = w.shape()[1];
    const unsigned int dim2   = w.shape()[2];
    const unsigned int dim3   = w.shape()[3];

    for(unsigned int b = 0; b < dim3; ++b)
    {
        const auto mean_val  = mean.data()[b];
        const auto var_val   = var.data()[b];
        const auto beta_val  = beta.data()[b];
        const auto gamma_val = gamma.data()[b];

        for(unsigned int i = 0; i < width * height * dim2; ++i)
        {
            unsigned int index = i + b * width * height * dim2;

            w_fused_data[index] = (gamma_val * (w_data[index])) / sqrt(var_val + epsilon);
        }

        b_fused_data[b] = (b_data[b] - mean_val) / sqrt(var_val + epsilon) * gamma_val + beta_val;
    }
}

template void fuse_batch_normalization_dwc_layer(const SimpleTensor<float> &w, const SimpleTensor<float> &mean, const SimpleTensor<float> &var, SimpleTensor<float> &w_fused,
                                                 SimpleTensor<float> &b_fused, const SimpleTensor<float> &b, const SimpleTensor<float> &beta, const SimpleTensor<float> &gamma, float epsilon);
template void fuse_batch_normalization_dwc_layer(const SimpleTensor<half> &w, const SimpleTensor<half> &mean, const SimpleTensor<half> &var, SimpleTensor<half> &w_fused, SimpleTensor<half> &b_fused,
                                                 const SimpleTensor<half> &b, const SimpleTensor<half> &beta, const SimpleTensor<half> &gamma, float epsilon);
template void fuse_batch_normalization_conv_layer(const SimpleTensor<float> &w, const SimpleTensor<float> &mean, const SimpleTensor<float> &var, SimpleTensor<float> &w_fused,
                                                  SimpleTensor<float> &b_fused, const SimpleTensor<float> &b, const SimpleTensor<float> &beta, const SimpleTensor<float> &gamma, float epsilon);
template void fuse_batch_normalization_conv_layer(const SimpleTensor<half> &w, const SimpleTensor<half> &mean, const SimpleTensor<half> &var, SimpleTensor<half> &w_fused, SimpleTensor<half> &b_fused,
                                                  const SimpleTensor<half> &b, const SimpleTensor<half> &beta, const SimpleTensor<half> &gamma, float epsilon);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
