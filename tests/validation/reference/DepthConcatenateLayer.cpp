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
#include "DepthConcatenateLayer.h"

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
SimpleTensor<T> depthconcatenate_layer(const std::vector<SimpleTensor<T>> &srcs)
{
    // Create reference
    std::vector<TensorShape> shapes;

    for(const auto &src : srcs)
    {
        shapes.emplace_back(src.shape());
    }

    DataType        dst_type  = srcs.empty() ? DataType::UNKNOWN : srcs[0].data_type();
    TensorShape     dst_shape = calculate_depth_concatenate_shape(shapes);
    SimpleTensor<T> dst(dst_shape, dst_type);

    // Compute reference
    int       depth_offset = 0;
    const int width_out    = dst.shape().x();
    const int height_out   = dst.shape().y();
    const int depth_out    = dst.shape().z();
    const int out_stride_z = width_out * height_out;
    const int batches      = dst.shape().total_size_upper(3);

    // Set output tensor to 0
    std::fill_n(dst.data(), dst.num_elements(), 0);

    for(const auto &src : srcs)
    {
        ARM_COMPUTE_ERROR_ON(depth_offset >= depth_out);
        ARM_COMPUTE_ERROR_ON(batches != static_cast<int>(src.shape().total_size_upper(3)));

        const int width  = src.shape().x();
        const int height = src.shape().y();
        const int depth  = src.shape().z();
        const int x_diff = (width_out - width) / 2;
        const int y_diff = (height_out - height) / 2;

        const T *src_ptr = src.data();

        for(int b = 0; b < batches; ++b)
        {
            const size_t offset_to_first_element = b * out_stride_z * depth_out + depth_offset * out_stride_z + y_diff * width_out + x_diff;

            for(int d = 0; d < depth; ++d)
            {
                for(int r = 0; r < height; ++r)
                {
                    std::copy(src_ptr, src_ptr + width, dst.data() + offset_to_first_element + d * out_stride_z + r * width_out);
                    src_ptr += width;
                }
            }
        }

        depth_offset += depth;
    }

    return dst;
}

template SimpleTensor<float> depthconcatenate_layer(const std::vector<SimpleTensor<float>> &srcs);
template SimpleTensor<half> depthconcatenate_layer(const std::vector<SimpleTensor<half>> &srcs);
template SimpleTensor<qint8_t> depthconcatenate_layer(const std::vector<SimpleTensor<qint8_t>> &srcs);
template SimpleTensor<qint16_t> depthconcatenate_layer(const std::vector<SimpleTensor<qint16_t>> &srcs);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
