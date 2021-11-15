/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#pragma once

#include <cstdint>

namespace arm_gemm
{
/*
 * Parameter set for "convolution" type GEMM.
 *
 * For a "convolution" GEMM, the GEMM parameters (M, K) are specified as if
 * an im2row had been performed on the input tensor to generate the operand
 * matrix, but instead this structure describes the convolution parameters
 * such that this can be done on the fly.
 *
 * The parameters describe the convolution details - the notional shape of
 * the input and output tensors, whether padding is to be applied, the size
 * of the kernel and a constant value to be used for padding (needed for
 * quantized tensors).
 *
 * The second part describes the layout of the input tensor in memory, which
 * is assumed to be in NHWC format.  This consists of a base pointer and
 * strides for columns, rows and batches.  'multis' are not supported for
 * convolution type GEMMs.
 */
struct ConvolutionParameters
{
    int64_t input_width;
    int64_t input_height;
    int64_t input_channels;
    int64_t kernel_width;
    int64_t kernel_height;
    int64_t output_width;
    int64_t output_height;
    int64_t output_stride_w;
    int64_t output_stride_h;
    //          output_channels not included as they do not affect the input.
    int64_t padding_top;
    int64_t padding_left;
    float   padding_value;
};

} // namespace arm_gemm
