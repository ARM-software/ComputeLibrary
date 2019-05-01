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
#include "ConcatenateLayer.h"

#include "tests/validation/Helpers.h"
#include "tests/validation/reference/Permute.h"

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
template <typename T>
SimpleTensor<T> widthconcatenate_layer(const std::vector<SimpleTensor<T>> &srcs, SimpleTensor<T> &dst)
{
    // Create reference
    std::vector<TensorShape> shapes;
    shapes.reserve(srcs.size());
    for(const auto &src : srcs)
    {
        shapes.emplace_back(src.shape());
    }
    // Compute reference
    int       width_offset = 0;
    const int width_out    = dst.shape().x();
    // Set output tensor to 0
    std::fill_n(dst.data(), dst.num_elements(), 0);
    for(const auto &src : srcs)
    {
        ARM_COMPUTE_ERROR_ON(width_offset >= width_out);

        const int width      = src.shape().x();
        const int height     = src.shape().y();
        const int depth      = src.shape().z();
        const int upper_dims = src.shape().total_size() / (width * height * depth);

        const T *src_ptr = src.data();
        T       *dst_ptr = dst.data();

        for(int u = 0; u < upper_dims; ++u)
        {
            for(int d = 0; d < depth; ++d)
            {
                for(int r = 0; r < height; ++r)
                {
                    const int offset = u * height * depth + d * height + r;
                    if(src.data_type() == DataType::QASYMM8 && src.quantization_info() != dst.quantization_info())
                    {
                        std::transform(src_ptr, src_ptr + width, dst_ptr + width_offset + offset * width_out, [src, dst](T t)
                        {
                            const float dequantized_input = src.quantization_info().dequantize(t);
                            return dst.quantization_info().quantize(dequantized_input, RoundingPolicy::TO_NEAREST_UP);
                        });
                        src_ptr += width;
                    }
                    else
                    {
                        std::copy(src_ptr, src_ptr + width, dst_ptr + width_offset + offset * width_out);
                        src_ptr += width;
                    }
                }
            }
        }
        width_offset += width;
    }
    return dst;
}

template SimpleTensor<float> widthconcatenate_layer(const std::vector<SimpleTensor<float>> &srcs, SimpleTensor<float> &dst);
template SimpleTensor<half> widthconcatenate_layer(const std::vector<SimpleTensor<half>> &srcs, SimpleTensor<half> &dst);
template SimpleTensor<uint8_t> widthconcatenate_layer(const std::vector<SimpleTensor<uint8_t>> &srcs, SimpleTensor<uint8_t> &dst);
} // namespace

template <typename T>
SimpleTensor<T> concatenate_layer(std::vector<SimpleTensor<T>> &srcs, SimpleTensor<T> &dst, unsigned int axis)
{
    switch(axis)
    {
        case Window::DimX:
        {
            return widthconcatenate_layer(srcs, dst);
        }
        case Window::DimY:
        {
            for(auto &t : srcs)
            {
                t = reference::permute<T>(t, PermutationVector(1U, 0U));
            }
            dst = reference::permute<T>(dst, PermutationVector(1U, 0U));
            return reference::permute<T>(widthconcatenate_layer(srcs, dst), PermutationVector(1U, 0U));
        }
        case Window::DimZ:
        {
            for(auto &t : srcs)
            {
                t = reference::permute<T>(t, PermutationVector(2U, 1U, 0U));
            }
            dst = reference::permute<T>(dst, PermutationVector(2U, 1U, 0U));
            return reference::permute<T>(widthconcatenate_layer(srcs, dst), PermutationVector(2U, 1U, 0U));
        }
        default:
        {
            ARM_COMPUTE_ERROR("Not supported");
            return dst;
        }
    }
}

template SimpleTensor<float> concatenate_layer(std::vector<SimpleTensor<float>> &srcs, SimpleTensor<float> &dst, unsigned int axis);
template SimpleTensor<half> concatenate_layer(std::vector<SimpleTensor<half>> &srcs, SimpleTensor<half> &dst, unsigned int axis);
template SimpleTensor<uint8_t> concatenate_layer(std::vector<SimpleTensor<uint8_t>> &srcs, SimpleTensor<uint8_t> &dst, unsigned int axis);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
