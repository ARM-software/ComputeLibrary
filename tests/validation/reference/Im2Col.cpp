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
#include "Im2Col.h"

#include "Permute.h"

#include "arm_compute/core/Types.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/Utils.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
void im2col_nchw(const SimpleTensor<T> &src, SimpleTensor<T> &dst, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias)
{
    // Create reference
    const int pad_x         = conv_info.pad().first;
    const int pad_y         = conv_info.pad().second;
    const int stride_x      = conv_info.stride().first;
    const int stride_y      = conv_info.stride().second;
    const int kernel_width  = kernel_dims.width;
    const int kernel_height = kernel_dims.height;
    const int src_width     = src.shape().x();
    const int src_height    = src.shape().y();
    const int src_depth     = src.shape().z();
    const int batches       = src.shape().total_size_upper(3);
    const int pad_val       = is_data_type_quantized_asymmetric(src.data_type()) ? src.quantization_info().offset : 0;

    int dst_idx = 0;
    for(int b = 0; b < batches; ++b)
    {
        for(int y = -pad_y; y <= (src_height + pad_y - kernel_height); y += stride_y)
        {
            for(int x = -pad_x; x <= (src_width + pad_x - kernel_width); x += stride_x)
            {
                for(int z = 0; z < src_depth; ++z)
                {
                    for(int patch_y = y; patch_y < (y + kernel_height); ++patch_y)
                    {
                        for(int patch_x = x; patch_x < (x + kernel_width); ++patch_x)
                        {
                            dst[dst_idx++] = tensor_elem_at(src, Coordinates(patch_x, patch_y, z, b), BorderMode::CONSTANT, static_cast<T>(pad_val));
                        }
                    }
                }

                if(has_bias)
                {
                    dst[dst_idx++] = static_cast<T>(1);
                }
            }
        }
    }
}

template <typename T>
SimpleTensor<T> im2col(const SimpleTensor<T> &src, const TensorShape &dst_shape, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias)
{
    SimpleTensor<T> dst{ dst_shape, src.data_type(), 1, src.fixed_point_position(), src.quantization_info() };

    if(src.data_layout() == DataLayout::NHWC)
    {
        SimpleTensor<T> src_nchw = reference::permute<T>(src, PermutationVector(1U, 2U, 0U));
        SimpleTensor<T> dst_nchw = reference::permute<T>(dst, PermutationVector(1U, 2U, 0U));

        im2col_nchw(src_nchw, dst_nchw, kernel_dims, conv_info, has_bias);

        return reference::permute<T>(dst_nchw, PermutationVector(2U, 0U, 1U));
    }

    im2col_nchw(src, dst, kernel_dims, conv_info, has_bias);

    return dst;
}

template SimpleTensor<uint8_t> im2col(const SimpleTensor<uint8_t> &src, const TensorShape &output_shape, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias);
template SimpleTensor<half> im2col(const SimpleTensor<half> &src, const TensorShape &output_shape, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias);
template SimpleTensor<float> im2col(const SimpleTensor<float> &src, const TensorShape &output_shape, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
