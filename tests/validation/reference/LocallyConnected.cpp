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
#include "LocallyConnected.h"

#include "tests/validation/Helpers.h"
#include "tests/validation/reference/Convolution3d.h"
#include "tests/validation/reference/Utils.h"

#include "tests/framework/Asserts.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T, typename TB>
SimpleTensor<T> locally_connected(const SimpleTensor<T> &src, const SimpleTensor<T> &weights, const SimpleTensor<TB> &bias, const TensorShape &output_shape, const PadStrideInfo &info)
{
    // Create reference
    SimpleTensor<T> dst{ output_shape, src.data_type(), 1, src.quantization_info() };

    // Compute reference
    const int width_in  = src.shape().x();
    const int height_in = src.shape().y();
    const int depth_in  = src.shape().z();

    const int width_out  = dst.shape().x();
    const int height_out = dst.shape().y();
    const int depth_out  = dst.shape().z();

    const int width_weights  = weights.shape().x();
    const int height_weights = weights.shape().y();
    const int depth_weights  = weights.shape().z();

    const int pad_left  = info.pad_left();
    const int pad_top   = info.pad_top();
    const int stride_xi = info.stride().first;
    const int stride_yi = info.stride().second;

    auto output_wh = scaled_dimensions(width_in, height_in, width_weights, height_weights, info);

    const int start_xi    = width_weights / 2 - pad_left;
    const int start_yi    = height_weights / 2 - pad_top;
    const int end_xi      = output_wh.first * stride_xi;
    const int end_yi      = output_wh.second * stride_yi;
    const int num_batches = src.shape().total_size() / (width_in * height_in * depth_in);

    for(int r = 0; r < num_batches; ++r)
    {
        int count = 0;
        for(int yi = start_yi; yi < start_yi + end_yi; yi += stride_yi)
        {
            for(int xi = start_xi; xi < start_xi + end_xi; xi += stride_xi)
            {
                for(int ofm = 0; ofm < depth_out; ++ofm)
                {
                    // Compute input and output offsets
                    const int offset_in  = r * width_in * height_in * depth_in;
                    const int xo         = (xi - start_xi) / stride_xi;
                    const int yo         = (yi - start_yi) / stride_yi;
                    const int offset_out = xo + yo * width_out + ofm * width_out * height_out + r * width_out * height_out * depth_out;

                    ARM_COMPUTE_ASSERT(xo < width_out);
                    ARM_COMPUTE_ASSERT(yo < height_out);

                    // Compute 3D convolution
                    convolution_3d::detail::convolution3d(src, weights, bias, dst,
                                                          offset_in, count * width_weights * height_weights * depth_weights, count, offset_out,
                                                          xi, yi,
                                                          width_in, height_in, depth_in,
                                                          width_weights, height_weights);
                    count++;
                }
            }
        }
    }

    return dst;
}

// Locally Connected only supports F32
template SimpleTensor<float> locally_connected(const SimpleTensor<float> &src, const SimpleTensor<float> &weights, const SimpleTensor<float> &bias, const TensorShape &output_shape,
                                               const PadStrideInfo &info);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
