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
    TensorShape scaled_shape = src.shape();
    scaled_shape.set(0, output_shape.x());
    scaled_shape.set(1, output_shape.y());
    SimpleTensor<T> scaled{ scaled_shape, src.data_type(), 1, src.fixed_point_position() };

    const int          width_in      = src.shape().x();
    const int          height_in     = src.shape().y();
    const int          width_scaled  = scaled.shape().x();
    const int          height_scaled = scaled.shape().y();
    const int          num_2d_slices = src.shape().total_size() / (width_in * height_in);
    const auto         width_ratio   = static_cast<float>(width_in) / static_cast<float>(width_scaled);
    const auto         height_ratio  = static_cast<float>(height_in) / static_cast<float>(height_scaled);
    const int          ax            = a.first;  // The number of zeros added to right edge of the input.
    const int          ay            = a.second; // The number of zeros added to bottom edge of the input.
    const unsigned int kernel_size   = weights.shape().x();
    ARM_COMPUTE_ERROR_ON(info.pad().first > (kernel_size - 1));
    const int transposed_convolution_padx = kernel_size - info.pad().first - 1;
    const int transposed_convolution_pady = kernel_size - info.pad().second - 1;
    const int stridex                     = info.stride().first;
    const int stridey                     = info.stride().second;

    for(int j = 0; j < scaled.num_elements(); ++j)
    {
        scaled[j] = T(0);
    }

    for(int slice = 0; slice < num_2d_slices; ++slice)
    {
        const int offset_slice_in  = slice * width_in * height_in;
        const int offset_slice_out = slice * width_scaled * height_scaled;
        for(int yi = ay; yi < height_scaled; yi += stridey)
        {
            for(int xi = transposed_convolution_padx; xi < width_scaled; xi += stridex)
            {
                const float x_src     = (xi + 0.5f) * width_ratio - 0.5f;
                const float y_src     = (yi + 0.5f) * height_ratio - 0.5f;
                T          *out       = scaled.data() + offset_slice_out + xi + yi * width_scaled;
                const bool  in_bounds = x_src > -1 && y_src > -1 && x_src < width_in && y_src < height_in;
                const bool  in_axy    = xi < transposed_convolution_padx || xi >= (width_scaled - ax)  // this is checking if the x coordinate is in the padded left/right area
                                        || yi < ay || yi >= (height_scaled - transposed_convolution_pady); // like above but top and bottom padding in the upscaled XY plane
                if(!in_axy)
                {
                    if(in_bounds)
                    {
                        const int in_scaled_x = support::cpp11::round(x_src);
                        const int in_scaled_y = support::cpp11::round(y_src);
                        const T *in          = src.data() + offset_slice_in + in_scaled_x + in_scaled_y * width_in;
                        *out                  = *in;
                    }
                    else
                    {
                        *out = T(0);
                    }
                }
            }
        }
    }
    const PadStrideInfo conv_info(1, 1, 1, 1, DimensionRoundingType::CEIL);
    return convolution_layer(scaled, weights, bias, output_shape, conv_info);
}

template SimpleTensor<float> deconvolution_layer(const SimpleTensor<float> &src, const SimpleTensor<float> &weights, const SimpleTensor<float> &bias, const TensorShape &output_shape,
                                                 const PadStrideInfo &info, const std::pair<unsigned int, unsigned int> &a);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
