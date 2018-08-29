/*
 * Copyright (c) 2018 ARM Limited.
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
#include "WidthConcatenateLayer.h"

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
SimpleTensor<T> widthconcatenate_layer(const std::vector<SimpleTensor<T>> &srcs)
{
    // Create reference
    std::vector<TensorShape> shapes;

    for(const auto &src : srcs)
    {
        shapes.emplace_back(src.shape());
    }

    DataType        dst_type  = srcs.empty() ? DataType::UNKNOWN : srcs[0].data_type();
    TensorShape     dst_shape = calculate_width_concatenate_shape(shapes);
    SimpleTensor<T> dst(dst_shape, dst_type);

    // Compute reference
    int       width_offset = 0;
    const int width_out    = dst.shape().x();

    // Set output tensor to 0
    std::fill_n(dst.data(), dst.num_elements(), 0);

    for(const auto &src : srcs)
    {
        ARM_COMPUTE_ERROR_ON(width_offset >= width_out);

        const int width  = src.shape().x();
        const int height = src.shape().y();
        const int depth  = src.shape().z();

        const T *src_ptr = src.data();
        T       *dst_ptr = dst.data();

        for(int d = 0; d < depth; ++d)
        {
            for(int r = 0; r < height; ++r)
            {
                int offset = d * height + r;
                std::copy(src_ptr, src_ptr + width, dst_ptr + width_offset + offset * width_out);
                src_ptr += width;
            }
        }

        width_offset += width;
    }

    return dst;
}

template SimpleTensor<float> widthconcatenate_layer(const std::vector<SimpleTensor<float>> &srcs);
template SimpleTensor<half> widthconcatenate_layer(const std::vector<SimpleTensor<half>> &srcs);
template SimpleTensor<uint8_t> widthconcatenate_layer(const std::vector<SimpleTensor<uint8_t>> &srcs);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
