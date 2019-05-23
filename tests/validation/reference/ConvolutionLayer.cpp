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
#include "ConvolutionLayer.h"

#include "tests/validation/Helpers.h"
#include "tests/validation/reference/Convolution3d.h"
#include "tests/validation/reference/Permute.h"
#include "tests/validation/reference/Utils.h"
#include "tests/validation/reference/UtilsQuantizedAsymm.h"

#include "tests/framework/Asserts.h"

#include "arm_compute/core/utils/quantization/AsymmHelpers.h"

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
} // namespace

template <typename T, typename TB>
SimpleTensor<T> convolution_layer_nchw(const SimpleTensor<T> &src, const SimpleTensor<T> &weights, const SimpleTensor<TB> &bias, SimpleTensor<T> &dst, const PadStrideInfo &info,
                                       const Size2D &dilation, unsigned int num_groups)
{
    ARM_COMPUTE_ERROR_ON((src.shape()[2] / num_groups) != weights.shape()[2]);

    // Compute reference
    const int width_in       = src.shape().x();
    const int height_in      = src.shape().y();
    const int depth_in       = src.shape().z();
    const int width_out      = dst.shape().x();
    const int height_out     = dst.shape().y();
    const int depth_out      = dst.shape().z();
    const int width_weights  = weights.shape().x();
    const int height_weights = weights.shape().y();
    const int depth_weights  = weights.shape().z();
    const int pad_left       = info.pad_left();
    const int pad_top        = info.pad_top();
    const int stride_xi      = info.stride().first;
    const int stride_yi      = info.stride().second;

    auto output_wh = scaled_dimensions(width_in, height_in, width_weights, height_weights, info, dilation);

    const int start_xi    = (dilation.x() * (width_weights - 1) + 1) / 2 - pad_left;
    const int start_yi    = (dilation.y() * (height_weights - 1) + 1) / 2 - pad_top;
    const int end_xi      = output_wh.first * stride_xi;
    const int end_yi      = output_wh.second * stride_yi;
    const int num_batches = src.shape().total_size() / (width_in * height_in * depth_in);

    for(int r = 0; r < num_batches; ++r)
    {
        for(int yi = start_yi; yi < start_yi + end_yi; yi += stride_yi)
        {
            for(int xi = start_xi; xi < start_xi + end_xi; xi += stride_xi)
            {
                for(int group = 0; group < static_cast<int>(num_groups); ++group)
                {
                    for(int ofm = 0; ofm < static_cast<int>(depth_out / num_groups); ++ofm)
                    {
                        // Compute input and output offsets
                        const int offset_in  = r * width_in * height_in * depth_in + (group * (depth_in / num_groups) * width_in * height_in);
                        const int xo         = (xi - start_xi) / stride_xi;
                        const int yo         = (yi - start_yi) / stride_yi;
                        const int offset_out = xo + yo * width_out + ((ofm + group * (depth_out / num_groups)) * width_out * height_out) + (r * width_out * height_out * depth_out);
                        const int offset_w   = (ofm + group * (depth_out / num_groups)) * width_weights * height_weights * depth_weights;
                        const int offset_b   = (ofm + group * (depth_out / num_groups));

                        ARM_COMPUTE_ASSERT(xo < width_out);
                        ARM_COMPUTE_ASSERT(yo < height_out);

                        // Compute 3D convolution
                        convolution_3d::detail::convolution3d(src, weights, bias, dst,
                                                              offset_in, offset_w, offset_b, offset_out,
                                                              xi, yi,
                                                              width_in, height_in, (depth_in / num_groups),
                                                              width_weights, height_weights, dilation.x(), dilation.y());
                    }
                }
            }
        }
    }

    return dst;
}
template <typename T, typename TB>
SimpleTensor<T> convolution_layer(const SimpleTensor<T> &src, const SimpleTensor<T> &weights, const SimpleTensor<TB> &bias, const TensorShape &output_shape, const PadStrideInfo &info,
                                  const Size2D &dilation, unsigned int num_groups, QuantizationInfo out_quant_info)
{
    // if no explicit quantization has been set you the same as src
    if(out_quant_info == QuantizationInfo())
    {
        out_quant_info = src.quantization_info();
    }
    // Create reference
    SimpleTensor<T> dst{ output_shape, src.data_type(), 1, out_quant_info };

    if(src.data_layout() == DataLayout::NHWC)
    {
        SimpleTensor<T> src_nchw     = reference::permute<T>(src, PermutationVector(1U, 2U, 0U));
        SimpleTensor<T> weights_nchw = reference::permute<T>(weights, PermutationVector(1U, 2U, 0U));
        SimpleTensor<T> dst_nchw     = reference::permute<T>(dst, PermutationVector(1U, 2U, 0U));

        return reference::permute<T>(convolution_layer_nchw(src_nchw, weights_nchw, bias, dst_nchw, info, dilation, num_groups), PermutationVector(2U, 0U, 1U));
    }
    else
    {
        return convolution_layer_nchw(src, weights, bias, dst, info, dilation, num_groups);
    }
}

template SimpleTensor<float> convolution_layer(const SimpleTensor<float> &src, const SimpleTensor<float> &weights, const SimpleTensor<float> &bias, const TensorShape &output_shape,
                                               const PadStrideInfo &info, const Size2D &dilation, unsigned int num_groups, QuantizationInfo out_quant_info);
template SimpleTensor<half> convolution_layer(const SimpleTensor<half> &src, const SimpleTensor<half> &weights, const SimpleTensor<half> &bias, const TensorShape &output_shape,
                                              const PadStrideInfo &info, const Size2D &dilation, unsigned int num_groups, QuantizationInfo out_quant_info);
template SimpleTensor<uint8_t> convolution_layer(const SimpleTensor<uint8_t> &src, const SimpleTensor<uint8_t> &weights, const SimpleTensor<int32_t> &bias, const TensorShape &output_shape,
                                                 const PadStrideInfo &info, const Size2D &dilation, unsigned int num_groups, QuantizationInfo out_quant_info);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
