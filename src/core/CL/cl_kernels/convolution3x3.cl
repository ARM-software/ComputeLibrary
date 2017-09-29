/*
 * Copyright (c) 2016, 2017 ARM Limited.
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

#ifndef DATA_TYPE
#define DATA_TYPE short
#endif /* DATA_TYPE */

#ifndef DATA_TYPE_OUT
#define DATA_TYPE_OUT uchar
#endif /* DATA_TYPE_OUT */

/** Compute a 1D horizontal convolution of size 3 for 8 bytes assuming the input is made of 1 channel of 1 byte (i.e 8 pixels).
 *
 * @param[in] left_pixel   Pointer to the left pixel.
 * @param[in] left_coeff   Weight of the left pixel
 * @param[in] middle_coeff Weight of the middle pixel
 * @param[in] right_coeff  Weight of the right pixel
 *
 * @return a short8 containing 8 convoluted values.
 */
inline VEC_DATA_TYPE(DATA_TYPE, 8) convolution1x3(__global const uchar *left_pixel,
                                                  const short left_coeff,
                                                  const short middle_coeff,
                                                  const short right_coeff)
{
    uchar16 temp = vload16(0, left_pixel);
    VEC_DATA_TYPE(DATA_TYPE, 8)
    left = CONVERT(temp.s01234567, VEC_DATA_TYPE(DATA_TYPE, 8));
    VEC_DATA_TYPE(DATA_TYPE, 8)
    middle = CONVERT(temp.s12345678, VEC_DATA_TYPE(DATA_TYPE, 8));
    VEC_DATA_TYPE(DATA_TYPE, 8)
    right = CONVERT(temp.s23456789, VEC_DATA_TYPE(DATA_TYPE, 8));

    return left * (VEC_DATA_TYPE(DATA_TYPE, 8))left_coeff + middle * (VEC_DATA_TYPE(DATA_TYPE, 8))middle_coeff + right * (VEC_DATA_TYPE(DATA_TYPE, 8))right_coeff;
}

/** Apply a 3x3 convolution matrix to a single channel U8 input image and return the result.
 *
 * Convolution matrix layout:
 *
 * [ mat0, mat1, mat2 ]\n
 * [ mat3, mat4, mat5 ]\n
 * [ mat6, mat7, mat8 ]\n
 *
 * @param[in] src   A pointer to source Image structure
 * @param[in] mat0  Coefficient from the convolution matrix
 * @param[in] mat1  Coefficient from the convolution matrix
 * @param[in] mat2  Coefficient from the convolution matrix
 * @param[in] mat3  Coefficient from the convolution matrix
 * @param[in] mat4  Coefficient from the convolution matrix
 * @param[in] mat5  Coefficient from the convolution matrix
 * @param[in] mat6  Coefficient from the convolution matrix
 * @param[in] mat0  Coefficient from the convolution matrix
 * @param[in] mat7  Coefficient from the convolution matrix
 * @param[in] mat8  Coefficient from the convolution matrix
 * @param[in] scale Convolution matrix scale (Sum of the coefficients, or 1 if the sum is 0)
 *
 * @return a short8 containing 8 convoluted and scaled values.
 */
inline VEC_DATA_TYPE(DATA_TYPE, 8) convolution3x3(
    Image      *src,
    const short mat0, const short mat1, const short mat2,
    const short mat3, const short mat4, const short mat5,
    const short mat6, const short mat7, const short mat8, uint scale)
{
    // Output pixels
    VEC_DATA_TYPE(DATA_TYPE, 8)
    pixels;

    // Row 0
    pixels = convolution1x3(offset(src, -1, -1), mat0, mat1, mat2);
    // Row
    pixels += convolution1x3(offset(src, -1, 0), mat3, mat4, mat5);
    // Row 2
    pixels += convolution1x3(offset(src, -1, 1), mat6, mat7, mat8);

    // Divide by the scale
    return pixels / (VEC_DATA_TYPE(DATA_TYPE, 8))scale;
}

#ifndef DYNAMIC_MATRIX_CONVOLUTION

/** Apply a 3x3 static convolution matrix to a single channel U8 input image and output a single channel image.
 *
 * @attention The matrix coefficients(MAT0, MAT1, ... MAT8, SCALE), DATA_TYPE, and DATA_TYPE_OUT need to be passed at compile time.\n
 * e.g. -DMAT0=1 -DMAT2=2, ...-DMAT8=8, -DSCALE=1, -DDATA_TYPE=int, -DDATA_TYPE_OUT=int
 *
 * @param[in]  src_ptr                           Pointer to the source image
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported data types: U8, S16
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void convolution3x3_static(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    VEC_DATA_TYPE(DATA_TYPE, 8)
    pixels = convolution3x3(&src,
                            MAT0, MAT1, MAT2, MAT3, MAT4, MAT5, MAT6, MAT7, MAT8, SCALE);

    // Store the result as is in dst
    vstore8(CONVERT_SAT(pixels, VEC_DATA_TYPE(DATA_TYPE_OUT, 8)), 0, (__global DATA_TYPE_OUT *)dst.ptr);
}

#endif // DYNAMIC_MATRIX_CONVOLUTION
