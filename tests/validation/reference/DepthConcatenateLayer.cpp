/*
 * Copyright (c) 2017-2019 ARM Limited.
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
SimpleTensor<T> depthconcatenate_layer(const std::vector<SimpleTensor<T>> &srcs, SimpleTensor<T> &dst)
{
    // Create reference
    std::vector<TensorShape> shapes;

    for(const auto &src : srcs)
    {
        shapes.emplace_back(src.shape());
    }

    // Compute reference
    int       depth_offset                = 0;
    const int width_out                   = dst.shape().x();
    const int height_out                  = dst.shape().y();
    const int depth_out                   = dst.shape().z();
    const int out_stride_z                = width_out * height_out;
    const int batches                     = dst.shape().total_size_upper(3);
    auto have_different_quantization_info = [&](const SimpleTensor<T> &tensor)
    {
        return tensor.quantization_info() != dst.quantization_info();
    };
    if(srcs[0].data_type() == DataType::QASYMM8 && std::any_of(srcs.cbegin(), srcs.cend(), have_different_quantization_info))
    {
        for(int b = 0; b < batches; ++b)
        {
            // input tensors can have smaller width and height than the output, so for each output's slice we need to requantize 0 (as this is the value
            // used in NEFillBorderKernel by NEDepthConcatenateLayer) using the corresponding quantization info for that particular slice/input tensor.
            int slice = 0;
            for(const auto &src : srcs)
            {
                auto       ptr_slice = static_cast<T *>(dst(Coordinates(0, 0, slice, b)));
                const auto num_elems_in_slice((dst.num_elements() / depth_out) * src.shape().z());
                std::transform(ptr_slice, ptr_slice + num_elems_in_slice, ptr_slice, [src, dst](T t)
                {
                    return dst.quantization_info().quantize(src.quantization_info().dequantize(0), RoundingPolicy::TO_NEAREST_UP);
                });
                slice += src.shape().z();
            }
        }
    }
    else
    {
        std::fill_n(dst.data(), dst.num_elements(), 0);
    }

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
                    if(src.data_type() == DataType::QASYMM8 && src.quantization_info() != dst.quantization_info())
                    {
                        std::transform(src_ptr, src_ptr + width, dst.data() + offset_to_first_element + d * out_stride_z + r * width_out, [src, dst](T t)
                        {
                            const float dequantized_input = src.quantization_info().dequantize(t);
                            return dst.quantization_info().quantize(dequantized_input, RoundingPolicy::TO_NEAREST_UP);
                        });
                        src_ptr += width;
                    }
                    else
                    {
                        std::copy(src_ptr, src_ptr + width, dst.data() + offset_to_first_element + d * out_stride_z + r * width_out);
                        src_ptr += width;
                    }
                }
            }
        }

        depth_offset += depth;
    }

    return dst;
}

template SimpleTensor<uint8_t> depthconcatenate_layer(const std::vector<SimpleTensor<uint8_t>> &srcs, SimpleTensor<uint8_t> &dst);
template SimpleTensor<float> depthconcatenate_layer(const std::vector<SimpleTensor<float>> &srcs, SimpleTensor<float> &dst);
template SimpleTensor<half> depthconcatenate_layer(const std::vector<SimpleTensor<half>> &srcs, SimpleTensor<half> &dst);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
