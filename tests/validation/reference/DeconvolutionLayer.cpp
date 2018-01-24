/*
 * Copyright (c) 2017, 2018 ARM Limited.
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
#include "ConvolutionLayer.h"

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
template <typename T>
SimpleTensor<T> deconvolution_layer(const SimpleTensor<T> &src, const SimpleTensor<T> &weights, const SimpleTensor<T> &bias, const TensorShape &output_shape,
                                    const PadStrideInfo &info, const std::pair<unsigned int, unsigned int> &a)
{
    // Create reference
    const int   stride_x     = info.stride().first;
    const int   stride_y     = info.stride().second;
    TensorShape scaled_shape = src.shape();
    int         out_x        = src.shape().x() + (src.shape().x() - 1) * (stride_x - 1) + a.first + 2 * info.pad().first;
    int         out_y        = src.shape().y() + (src.shape().y() - 1) * (stride_y - 1) + a.second + 2 * info.pad().second;
    scaled_shape.set(0, out_x);
    scaled_shape.set(1, out_y);
    SimpleTensor<T> scaled{ scaled_shape, src.data_type(), 1, src.fixed_point_position() };

    const int width_in      = src.shape().x();
    const int height_in     = src.shape().y();
    const int width_scaled  = scaled.shape().x();
    const int height_scaled = scaled.shape().y();
    const int num_2d_slices = src.shape().total_size() / (width_in * height_in);
    const int ax            = a.first;  // The number of zeros added to right edge of the input.
    const int ay            = a.second; // The number of zeros added to top edge of the input.
    ARM_COMPUTE_ERROR_ON(info.pad().first > (weights.shape().x() - 1));

    ARM_COMPUTE_ERROR_ON_MSG(ax > stride_x - 1, "ax must be smaller than stride_x");
    ARM_COMPUTE_ERROR_ON_MSG(ay > stride_y - 1, "ay must be smaller than stride_y");

    for(int j = 0; j < scaled.num_elements(); ++j)
    {
        scaled[j] = T(0);
    }

    for(int slice = 0; slice < num_2d_slices; ++slice)
    {
        const int offset_slice_in  = slice * width_in * height_in;
        const int offset_slice_out = slice * width_scaled * height_scaled;
        const int start_x          = info.pad().first;
        const int start_y          = ay + info.pad().second;
        const int end_y            = height_scaled - info.pad().second;
        const int end_x            = width_scaled - ax - info.pad().first;

        for(int yi = start_y, in_y = 0; yi < end_y; yi += stride_y, in_y++)
        {
            for(int xi = start_x, in_x = 0; xi < end_x; xi += stride_x, in_x++)
            {
                const T *in  = src.data() + offset_slice_in + in_y * width_in + in_x;
                T       *out = scaled.data() + offset_slice_out + xi + yi * width_scaled;
                *out         = *in;
            }
        }
    }

    const PadStrideInfo conv_info(1, 1, 0, 0, 0, 0, DimensionRoundingType::CEIL);
    return convolution_layer(scaled, weights, bias, output_shape, conv_info);
}

template SimpleTensor<float> deconvolution_layer(const SimpleTensor<float> &src, const SimpleTensor<float> &weights, const SimpleTensor<float> &bias, const TensorShape &output_shape,
                                                 const PadStrideInfo &info, const std::pair<unsigned int, unsigned int> &a);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
