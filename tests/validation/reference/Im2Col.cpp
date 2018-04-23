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
    ARM_COMPUTE_ERROR_ON(src.data_layout() != DataLayout::NCHW);
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
void im2col_nhwc(const SimpleTensor<T> &src, SimpleTensor<T> &dst, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias)
{
    ARM_COMPUTE_ERROR_ON(src.data_layout() != DataLayout::NHWC);
    const int pad_x         = conv_info.pad().first;
    const int pad_y         = conv_info.pad().second;
    const int stride_x      = conv_info.stride().first;
    const int stride_y      = conv_info.stride().second;
    const int kernel_width  = kernel_dims.width;
    const int kernel_height = kernel_dims.height;
    const int src_width     = src.shape().y();
    const int src_height    = src.shape().z();
    const int src_depth     = src.shape().x();
    const int batches       = src.shape().total_size_upper(3);
    const int pad_val       = is_data_type_quantized_asymmetric(src.data_type()) ? src.quantization_info().offset : 0;
    int       dst_idx       = 0;
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
                            dst[dst_idx++] = tensor_elem_at(src, Coordinates(z, patch_x, patch_y, b), BorderMode::CONSTANT, static_cast<T>(pad_val));
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
void im2col(const SimpleTensor<T> &src, SimpleTensor<T> &dst, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias)
{
    switch(src.data_layout())
    {
        case DataLayout::NCHW:
        {
            im2col_nchw(src, dst, kernel_dims, conv_info, has_bias);
            break;
        }
        case DataLayout::NHWC:
        {
            im2col_nhwc(src, dst, kernel_dims, conv_info, has_bias);
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Not supported.");
            break;
        }
    }
}

template void im2col(const SimpleTensor<uint8_t> &src, SimpleTensor<uint8_t> &dst, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias);
template void im2col(const SimpleTensor<half> &src, SimpleTensor<half> &dst, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias);
template void im2col(const SimpleTensor<float> &src, SimpleTensor<float> &dst, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
