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

#include "helpers.h"

#if CONV_STRIDE_X == 1
#define convolution1x3 convolution1x3_stride_1
#elif CONV_STRIDE_X == 2
#define convolution1x3 convolution1x3_stride_2
#elif CONV_STRIDE_X == 3
#define convolution1x3 convolution1x3_stride_3
#else /* CONV_STRIDE_X */
#error "Stride not supported"
#endif /* CONV_STRIDE_X */

/** Compute a 1D horizontal convolution of size 3 and stride 1 for floating point type.
 *
 * @param[in] left_pixel   Pointer to the left pixel.
 * @param[in] left_coeff   Weight of the left pixel
 * @param[in] middle_coeff Weight of the middle pixel
 * @param[in] right_coeff  Weight of the right pixel
 *
 * @return a float2 containing 2 convoluted values.
 */
inline float2 convolution1x3_stride_1(__global const uchar *left_pixel,
                                      const float           left_coeff,
                                      const float           middle_coeff,
                                      const float           right_coeff)
{
    float4 temp = vload4(0, (__global float *)left_pixel);

    float2 left   = CONVERT(temp.s01, float2);
    float2 middle = CONVERT(temp.s12, float2);
    float2 right  = CONVERT(temp.s23, float2);

    return left * (float2)left_coeff + middle * (float2)middle_coeff + right * (float2)right_coeff;
}

/** Compute a 1D horizontal convolution of size 3 and stride 2 for floating point type.
 *
 * @param[in] left_pixel   Pointer to the left pixel.
 * @param[in] left_coeff   Weight of the left pixel
 * @param[in] middle_coeff Weight of the middle pixel
 * @param[in] right_coeff  Weight of the right pixel
 *
 * @return a float2 containing 2 convoluted values.
 */
inline float2 convolution1x3_stride_2(__global const uchar *left_pixel,
                                      const float           left_coeff,
                                      const float           middle_coeff,
                                      const float           right_coeff)
{
    float4 temp0 = vload4(0, (__global float *)left_pixel);
    float  temp1 = *((__global float *)(left_pixel + 4 * sizeof(float)));

    float2 left   = CONVERT(temp0.s02, float2);
    float2 middle = CONVERT(temp0.s13, float2);
    float2 right  = CONVERT((float2)(temp0.s2, temp1), float2);

    return left * (float2)left_coeff + middle * (float2)middle_coeff + right * (float2)right_coeff;
}

/** Compute a 1D horizontal convolution of size 3 and stride 3 for floating point type.
 *
 * @param[in] left_pixel   Pointer to the left pixel.
 * @param[in] left_coeff   Weight of the left pixel
 * @param[in] middle_coeff Weight of the middle pixel
 * @param[in] right_coeff  Weight of the right pixel
 *
 * @return a float2 containing 2 convoluted values.
 */
inline float2 convolution1x3_stride_3(__global const uchar *left_pixel,
                                      const float           left_coeff,
                                      const float           middle_coeff,
                                      const float           right_coeff)
{
    float4 temp0 = vload4(0, (__global float *)left_pixel);
    float2 temp1 = vload2(0, (__global float *)(left_pixel + 4 * sizeof(float)));

    float2 left   = CONVERT(temp0.s03, float2);
    float2 middle = CONVERT((float2)(temp0.s1, temp1.s0), float2);
    float2 right  = CONVERT((float2)(temp0.s2, temp1.s1), float2);

    return left * (float2)left_coeff + middle * (float2)middle_coeff + right * (float2)right_coeff;
}

/** Apply a 3x3 convolution matrix to a single channel F32 input image and return the result.
 *
 * Convolution matrix layout:
 *
 * [ mat0, mat1, mat2 ]\n
 * [ mat3, mat4, mat5 ]\n
 * [ mat6, mat7, mat8 ]\n
 *
 * @param[in] src  A pointer to source Image structure
 * @param[in] mat0 Coefficient from the convolution matrix
 * @param[in] mat1 Coefficient from the convolution matrix
 * @param[in] mat2 Coefficient from the convolution matrix
 * @param[in] mat3 Coefficient from the convolution matrix
 * @param[in] mat4 Coefficient from the convolution matrix
 * @param[in] mat5 Coefficient from the convolution matrix
 * @param[in] mat6 Coefficient from the convolution matrix
 * @param[in] mat0 Coefficient from the convolution matrix
 * @param[in] mat7 Coefficient from the convolution matrix
 * @param[in] mat8 Coefficient from the convolution matrix
 *
 * @return a float2 containing 2 convoluted values.
 */
inline float2 convolution3x3(
    Image      *src,
    const float mat0, const float mat1, const float mat2,
    const float mat3, const float mat4, const float mat5,
    const float mat6, const float mat7, const float mat8)
{
    float2 pixels;

    pixels = convolution1x3(offset(src, 0, 0), mat0, mat1, mat2);
    pixels += convolution1x3(offset(src, 0, 1), mat3, mat4, mat5);
    pixels += convolution1x3(offset(src, 0, 2), mat6, mat7, mat8);

    return pixels;
}

/** This function computes the horizontal integral of the image.
  *
  * @param[in] src_ptr                               Pointer to the source image. Supported data types: U8
  * @param[in] src_stride_x                          Stride of the source image in X dimension (in bytes)
  * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
  * @param[in] src_stride_y                          Stride of the source image in Y dimension (in bytes)
  * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
  * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source image
  * @param[in] src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
  * @param[in] src_step_z                            src_stride_z * number of elements along Y processed per workitem(in bytes)
  * @param[in] dst_ptr                               Pointer to the destination tensor. Supported data types: F16/F32
  * @param[in] dst_stride_x                          Stride of the destination tensor in X dimension (in bytes)
  * @param[in] dst_step_x                            dst_stride_x * number of elements along X processed per workitem(in bytes)
  * @param[in] dst_stride_y                          Stride of the destination tensor in Y dimension (in bytes)
  * @param[in] dst_step_y                            dst_stride_y * number of elements along Y processed per workitem(in bytes)
  * @param[in] dst_stride_z                          Stride of the destination tensor in Z dimension (in bytes)
  * @param[in] dst_step_z                            dst_stride_z * number of elements along Y processed per workitem(in bytes)
  * @param[in] dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
  * @param[in] weights_ptr                           Pointer to the weights tensor. Supported data types: F16/F32
  * @param[in] weights_stride_x                      Stride of the weights tensor in X dimension (in bytes)
  * @param[in] weights_step_x                        weights_stride_x * number of elements along X processed per workitem(in bytes)
  * @param[in] weights_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
  * @param[in] weights_step_y                        weights_stride_y * number of elements along Y processed per workitem(in bytes)
  * @param[in] weights_stride_z                      Stride of the weights tensor in Z dimension (in bytes)
  * @param[in] weights_step_z                        weights_stride_z * number of elements along Y processed per workitem(in bytes)
  * @param[in] weights_offset_first_element_in_bytes The offset of the first element in the weights tensor
  */

__kernel void depthwise_convolution_3x3(TENSOR3D_DECLARATION(src), TENSOR3D_DECLARATION(dst), TENSOR3D_DECLARATION(weights))
{
    Image    src     = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(src);
    Image    dst     = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);
    Tensor3D weights = CONVERT_TO_TENSOR3D_STRUCT(weights);

    uchar3 offset          = (uchar3)(0, 1, 2) * (uchar3)weights_stride_y;
    float3 weights_values0 = vload3(0, (__global float *)(weights.ptr + offset.s0));
    float3 weights_values1 = vload3(0, (__global float *)(weights.ptr + offset.s1));
    float3 weights_values2 = vload3(0, (__global float *)(weights.ptr + offset.s2));

    float2 pixels = convolution3x3(&src, weights_values0.s0, weights_values0.s1, weights_values0.s2,
                                   weights_values1.s0, weights_values1.s1, weights_values1.s2,
                                   weights_values2.s0, weights_values2.s1, weights_values2.s2);

    vstore2(pixels, 0, (__global float *)dst.ptr);
}