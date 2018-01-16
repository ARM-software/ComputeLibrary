/*
 * Copyright (c) 2017-2018 ARM Limited.
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

#include "helpers_asymm.h"

#if defined(CONV_STRIDE_X)

#if CONV_STRIDE_X == 1
#define convolution1x3 convolution1x3_stride_1
#elif CONV_STRIDE_X == 2
#define convolution1x3 convolution1x3_stride_2
#elif CONV_STRIDE_X == 3
#define convolution1x3 convolution1x3_stride_3
#else /* CONV_STRIDE_X */
#error "Stride not supported"
#endif /* CONV_STRIDE_X */

/** Compute a 1D horizontal convolution of size 3 and stride 1 for uchar type.
 *
 * @param[in] left_pixel    Pointer to the left pixel.
 * @param[in] left_coeff    Weight of the left pixel
 * @param[in] middle_coeff  Weight of the middle pixel
 * @param[in] right_coeff   Weight of the right pixel
 * @param[in] input_offset  Quantized offset of zero point of the input tensor data range
 * @param[in] weight_offset Quantized offset of zero point of the weights tensor data range
 *
 * @return a int8 containing 8 convoluted values.
 */
inline int8 convolution1x3_stride_1(__global const uchar *left_pixel,
                                    const int             left_coeff,
                                    const int             middle_coeff,
                                    const int             right_coeff,
                                    const int             input_offset,
                                    const int             weight_offset)
{
    int8 temp0 = CONVERT(vload8(0, left_pixel), int8);
    int2 temp1 = CONVERT(vload2(0, (left_pixel + 8 * sizeof(uchar))), int2);

    int8 left   = CONVERT(temp0.s01234567, int8);
    int8 middle = CONVERT((int8)(temp0.s1234, temp0.s567, temp1.s0), int8);
    int8 right  = CONVERT((int8)(temp0.s2345, temp0.s67, temp1.s01), int8);

    return (left + input_offset) * (int8)(left_coeff + weight_offset) + (middle + input_offset) * (int8)(middle_coeff + weight_offset) + (right + input_offset) * (int8)(right_coeff + weight_offset);
}

/** Compute a 1D horizontal convolution of size 3 and stride 2 for uchar type.
 *
 * @param[in] left_pixel    Pointer to the left pixel.
 * @param[in] left_coeff    Weight of the left pixel
 * @param[in] middle_coeff  Weight of the middle pixel
 * @param[in] right_coeff   Weight of the right pixel
 * @param[in] input_offset  Quantized offset of zero point of the input tensor data range
 * @param[in] weight_offset Quantized offset of zero point of the weights tensor data range
 *
 * @return a int8 containing 8 convoluted values.
 */
inline int8 convolution1x3_stride_2(__global const uchar *left_pixel,
                                    const int             left_coeff,
                                    const int             middle_coeff,
                                    const int             right_coeff,
                                    const int             input_offset,
                                    const int             weight_offset)
{
    int16 temp0 = CONVERT(vload16(0, left_pixel), int16);
    int   temp1 = CONVERT(*(left_pixel + 16 * sizeof(uchar)), int);

    int8 left   = CONVERT(temp0.s02468ace, int8);
    int8 middle = CONVERT(temp0.s13579bdf, int8);
    int8 right  = CONVERT((int8)(temp0.s2468, temp0.sace, temp1), int8);

    return (left + input_offset) * (int8)(left_coeff + weight_offset) + (middle + input_offset) * (int8)(middle_coeff + weight_offset) + (right + input_offset) * (int8)(right_coeff + weight_offset);
}

/** Compute a 1D horizontal convolution of size 3 and stride 3 for uchar type.
 *
 * @param[in] left_pixel    Pointer to the left pixel.
 * @param[in] left_coeff    Weight of the left pixel
 * @param[in] middle_coeff  Weight of the middle pixel
 * @param[in] right_coeff   Weight of the right pixel
 * @param[in] input_offset  Quantized offset of zero point of the input tensor data range
 * @param[in] weight_offset Quantized offset of zero point of the weights tensor data range
 *
 * @return a int8 containing 8 convoluted values.
 */
inline int8 convolution1x3_stride_3(__global const uchar *left_pixel,
                                    const int             left_coeff,
                                    const int             middle_coeff,
                                    const int             right_coeff,
                                    const int             input_offset,
                                    const int             weight_offset)
{
    int16 temp0 = CONVERT(vload16(0, left_pixel), int16);
    int8  temp1 = CONVERT(vload8(0, (left_pixel + 16 * sizeof(uchar))), int8);

    int8 left   = CONVERT((int8)(temp0.s0369, temp0.scf, temp1.s25), int8);
    int8 middle = CONVERT((int8)(temp0.s147a, temp0.sd, temp1.s036), int8);
    int8 right  = CONVERT((int8)(temp0.s258b, temp0.se, temp1.s147), int8);

    return (left + input_offset) * (int8)(left_coeff + weight_offset) + (middle + input_offset) * (int8)(middle_coeff + weight_offset) + (right + input_offset) * (int8)(right_coeff + weight_offset);
}

/** Apply a 3x3 convolution matrix to a single channel QASYMM8 input image and return the result.
 *
 * Convolution matrix layout:
 *
 * [ mat0, mat1, mat2 ]\n
 * [ mat3, mat4, mat5 ]\n
 * [ mat6, mat7, mat8 ]\n
 *
 * @param[in] src               A pointer to source Image structure
 * @param[in] mat0              Coefficient from the convolution matrix
 * @param[in] mat1              Coefficient from the convolution matrix
 * @param[in] mat2              Coefficient from the convolution matrix
 * @param[in] mat3              Coefficient from the convolution matrix
 * @param[in] mat4              Coefficient from the convolution matrix
 * @param[in] mat5              Coefficient from the convolution matrix
 * @param[in] mat6              Coefficient from the convolution matrix
 * @param[in] mat7              Coefficient from the convolution matrix
 * @param[in] mat8              Coefficient from the convolution matrix
 * @param[in] input_offset      Quantized offset of zero point of the input tensor data range
 * @param[in] weight_offset     Quantized offset of zero point of the weights tensor data range
 * @param[in] output_offset     Quantized offset of zero point of the output tensor data range
 * @param[in] output_multiplier Output scale multiplier
 * @param[in] output_shift      Output scale divisor exponent
 * @param[in] bias              (Optional) Bias value
 *
 * @return a uchar8 containing 8 convoluted values.
 */
inline uchar8 convolution3x3(
    Image      *src,
    const uchar mat0, const uchar mat1, const uchar mat2,
    const uchar mat3, const uchar mat4, const uchar mat5,
    const uchar mat6, const uchar mat7, const uchar mat8,
    const int input_offset, const int weight_offset, const int output_offset,
    const int output_multiplier, const int output_shift
#if defined(HAS_BIAS)
    ,
    const int bias
#endif //defined(HAS_BIAS)
)
{
    int8 pixels;

    pixels = convolution1x3(offset(src, 0, 0), mat0, mat1, mat2, input_offset, weight_offset);
    pixels += convolution1x3(offset(src, 0, 1), mat3, mat4, mat5, input_offset, weight_offset);
    pixels += convolution1x3(offset(src, 0, 2), mat6, mat7, mat8, input_offset, weight_offset);
#if defined(HAS_BIAS)
    pixels += (int8)(bias);
#endif //defined(HAS_BIAS)

    pixels = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(pixels, output_multiplier, output_shift, 8);
    pixels = pixels + output_offset;
    pixels = clamp(pixels, 0, 255);

    return CONVERT(pixels, uchar8);
}

/** This function computes the horizontal integral of the image.
 *
 * @param[in] src_ptr                               Pointer to the source image. Supported data types: QASYMM8
 * @param[in] src_stride_x                          Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                          Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source image
 * @param[in] src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                            src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_ptr                               Pointer to the destination tensor. Supported data types: QASYMM8
 * @param[in] dst_stride_x                          Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                            dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                          Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                            dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                          Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                            dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
 * @param[in] weights_ptr                           Pointer to the weights tensor. Supported data types: QASYMM8
 * @param[in] weights_stride_x                      Stride of the weights tensor in X dimension (in bytes)
 * @param[in] weights_step_x                        weights_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] weights_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
 * @param[in] weights_step_y                        weights_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_stride_z                      Stride of the weights tensor in Z dimension (in bytes)
 * @param[in] weights_step_z                        weights_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_offset_first_element_in_bytes The offset of the first element in the weights tensor
 * @param[in] biases_ptr                            (Optional) Pointer to the biases vector. Supported data types: QASYMM8
 * @param[in] biases_stride_x                       (Optional) Stride of the biases vector in X dimension (in bytes)
 * @param[in] biases_step_x                         (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] biases_offset_first_element_in_bytes  (Optional) The offset of the first element in the biases vector
 * @param[in] input_offset                          Quantized offset of zero point of the input tensor data range
 * @param[in] weight_offset                         Quantized offset of zero point of the weights tensor data range
 * @param[in] output_offset                         Quantized offset of zero point of the output tensor data range
 * @param[in] output_multiplier                     Output scale multiplier
 * @param[in] output_shift                          Output scale divisor exponent
 */

__kernel void depthwise_convolution_3x3_quantized(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(weights),
#if defined(HAS_BIAS)
    VECTOR_DECLARATION(biases),
#endif //defined(HAS_BIAS)
    int input_offset,
    int weight_offset,
    int output_offset,
    int output_multiplier,
    int output_shift)
{
    Image    src     = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(src);
    Image    dst     = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);
    Tensor3D weights = CONVERT_TO_TENSOR3D_STRUCT(weights);
#if defined(HAS_BIAS)
    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);
#endif //defined(HAS_BIAS)

    uchar3 offset          = (uchar3)(0, 1, 2) * (uchar3)weights_stride_y;
    uchar3 weights_values0 = vload3(0, weights.ptr + offset.s0);
    uchar3 weights_values1 = vload3(0, weights.ptr + offset.s1);
    uchar3 weights_values2 = vload3(0, weights.ptr + offset.s2);

#if defined(HAS_BIAS)
    int bias_value = *((__global int *)(vector_offset(&biases, get_global_id(2))));
#endif //defined(HAS_BIAS)

    uchar8 pixels = convolution3x3(&src, weights_values0.s0, weights_values0.s1, weights_values0.s2,
                                   weights_values1.s0, weights_values1.s1, weights_values1.s2,
                                   weights_values2.s0, weights_values2.s1, weights_values2.s2,
                                   input_offset, weight_offset, output_offset,
                                   output_multiplier, output_shift
#if defined(HAS_BIAS)
                                   ,
                                   bias_value
#endif //defined(HAS_BIAS)
                                  );

    vstore8(pixels, 0, dst.ptr);
}

#endif //defined(CONV_STRIDE_X)
