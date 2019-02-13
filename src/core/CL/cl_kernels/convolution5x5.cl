/*
 * Copyright (c) 2016-2019 ARM Limited.
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

#ifndef COMPUTE_TYPE
#define COMPUTE_TYPE int
#endif /* COMPUTE_TYPE */

#ifndef DATA_TYPE_OUT
#define DATA_TYPE_OUT uchar
#endif /* DATA_TYPE_OUT */

/** Compute a 1D horizontal convolution of size 5 for 8 bytes assuming the input is made of 1 channel of 1 byte (i.e 8 pixels).
 *
 * @param[in] left_pixel   Pointer to the left pixel
 * @param[in] left1_coeff  Weight of the most left pixel
 * @param[in] left2_coeff  Weight of the left pixel
 * @param[in] middle_coeff Weight of the middle pixel
 * @param[in] right1_coeff Weight of the right pixel
 * @param[in] right2_coeff Weight of the most right pixel
 *
 * @return a short8 containing 8 convoluted values.
 */
VEC_DATA_TYPE(DATA_TYPE, 8)
convolution1x5(
    __global const uchar *left_pixel,
    const short           left1_coeff,
    const short           left2_coeff,
    const short           middle_coeff,
    const short           right1_coeff,
    const short           right2_coeff)
{
    uchar16 temp = vload16(0, left_pixel);

    VEC_DATA_TYPE(DATA_TYPE, 8)
    left1 = CONVERT(temp.s01234567, VEC_DATA_TYPE(DATA_TYPE, 8));
    VEC_DATA_TYPE(DATA_TYPE, 8)
    left2 = CONVERT(temp.s12345678, VEC_DATA_TYPE(DATA_TYPE, 8));
    VEC_DATA_TYPE(DATA_TYPE, 8)
    middle = CONVERT(temp.s23456789, VEC_DATA_TYPE(DATA_TYPE, 8));
    VEC_DATA_TYPE(DATA_TYPE, 8)
    right1 = CONVERT(temp.s3456789a, VEC_DATA_TYPE(DATA_TYPE, 8));
    VEC_DATA_TYPE(DATA_TYPE, 8)
    right2 = CONVERT(temp.s456789ab, VEC_DATA_TYPE(DATA_TYPE, 8));

    return left1 * (VEC_DATA_TYPE(DATA_TYPE, 8))left1_coeff + left2 * (VEC_DATA_TYPE(DATA_TYPE, 8))left2_coeff
           + middle * (VEC_DATA_TYPE(DATA_TYPE, 8))middle_coeff + right1 * (VEC_DATA_TYPE(DATA_TYPE, 8))right1_coeff + right2 * (VEC_DATA_TYPE(DATA_TYPE, 8))right2_coeff;
}

/** Compute a 1D vertical convolution of size 5 for 8 bytes assuming the input is made of 1 channel of 1 byte (i.e 8 pixels).
 *
 * @param[in] src          Pointer to source image.
 * @param[in] up1_coeff    Weight of the most up pixel
 * @param[in] up2_coeff    Weight of the up pixel
 * @param[in] middle_coeff Weight of the middle pixel
 * @param[in] down1_coeff  Weight of the down pixel
 * @param[in] down2_coeff  Weight of the most down pixel
 *
 * @return a short8 containing 8 convoluted values.
 */
VEC_DATA_TYPE(COMPUTE_TYPE, 8)
convolution5x1(
    Image      *src,
    const short up1_coeff,
    const short up2_coeff,
    const short middle_coeff,
    const short down1_coeff,
    const short down2_coeff)
{
    VEC_DATA_TYPE(COMPUTE_TYPE, 8)
    val;
    VEC_DATA_TYPE(COMPUTE_TYPE, 8)
    out = (VEC_DATA_TYPE(COMPUTE_TYPE, 8))0;

    val = CONVERT(vload8(0, (__global DATA_TYPE *)offset(src, 0, -2)), VEC_DATA_TYPE(COMPUTE_TYPE, 8));
    out += val * (VEC_DATA_TYPE(COMPUTE_TYPE, 8))up1_coeff;

    val = CONVERT(vload8(0, (__global DATA_TYPE *)offset(src, 0, -1)), VEC_DATA_TYPE(COMPUTE_TYPE, 8));
    out += val * (VEC_DATA_TYPE(COMPUTE_TYPE, 8))up2_coeff;

    val = CONVERT(vload8(0, (__global DATA_TYPE *)offset(src, 0, 0)), VEC_DATA_TYPE(COMPUTE_TYPE, 8));
    out += val * (VEC_DATA_TYPE(COMPUTE_TYPE, 8))middle_coeff;

    val = CONVERT(vload8(0, (__global DATA_TYPE *)offset(src, 0, 1)), VEC_DATA_TYPE(COMPUTE_TYPE, 8));
    out += val * (VEC_DATA_TYPE(COMPUTE_TYPE, 8))down1_coeff;

    val = CONVERT(vload8(0, (__global DATA_TYPE *)offset(src, 0, 2)), VEC_DATA_TYPE(COMPUTE_TYPE, 8));
    out += val * (VEC_DATA_TYPE(COMPUTE_TYPE, 8))down2_coeff;

    return out;
}

/** Apply a 5x5 convolution matrix to a single channel U8 input image and return the result.
 *
 * Convolution matrix layout:\n
 * [  mat0,  mat1,  mat2,  mat3 , mat4 ]\n
 * [  mat5,  mat6,  mat7,  mat8,  mat9 ]\n
 * [ mat10, mat11, mat12, mat13, mat14 ]\n
 * [ mat15, mat16, mat17, mat18, mat19 ]\n
 * [ mat20, mat21, mat22, mat23, mat24 ]
 *
 * @param[in] src   A pointer to source Image structure.
 * @param[in] mat0  Coefficient from the convolution matrix
 * @param[in] mat1  Coefficient from the convolution matrix
 * @param[in] mat2  Coefficient from the convolution matrix
 * @param[in] mat3  Coefficient from the convolution matrix
 * @param[in] mat4  Coefficient from the convolution matrix
 * @param[in] mat5  Coefficient from the convolution matrix
 * @param[in] mat6  Coefficient from the convolution matrix
 * @param[in] mat7  Coefficient from the convolution matrix
 * @param[in] mat8  Coefficient from the convolution matrix
 * @param[in] mat9  Coefficient from the convolution matrix
 * @param[in] mat10 Coefficient from the convolution matrix
 * @param[in] mat11 Coefficient from the convolution matrix
 * @param[in] mat12 Coefficient from the convolution matrix
 * @param[in] mat13 Coefficient from the convolution matrix
 * @param[in] mat14 Coefficient from the convolution matrix
 * @param[in] mat15 Coefficient from the convolution matrix
 * @param[in] mat16 Coefficient from the convolution matrix
 * @param[in] mat17 Coefficient from the convolution matrix
 * @param[in] mat18 Coefficient from the convolution matrix
 * @param[in] mat19 Coefficient from the convolution matrix
 * @param[in] mat20 Coefficient from the convolution matrix
 * @param[in] mat21 Coefficient from the convolution matrix
 * @param[in] mat22 Coefficient from the convolution matrix
 * @param[in] mat23 Coefficient from the convolution matrix
 * @param[in] mat24 Coefficient from the convolution matrix
 * @param[in] scale Convolution matrix scale (Sum of the coefficients, or 1 if the sum is 0)
 *
 * @return a short8 containing 8 convoluted and scaled values.
 */
short8 convolution5x5(
    Image      *src,
    const short mat0, const short mat1, const short mat2, const short mat3, const short mat4,
    const short mat5, const short mat6, const short mat7, const short mat8, const short mat9,
    const short mat10, const short mat11, const short mat12, const short mat13, const short mat14,
    const short mat15, const short mat16, const short mat17, const short mat18, const short mat19,
    const short mat20, const short mat21, const short mat22, const short mat23, const short mat24,
    uint scale)
{
    VEC_DATA_TYPE(DATA_TYPE, 8)
    pixels;

    pixels = convolution1x5(offset(src, -2, -2), mat0, mat1, mat2, mat3, mat4);
    pixels += convolution1x5(offset(src, -2, -1), mat5, mat6, mat7, mat8, mat9);
    pixels += convolution1x5(offset(src, -2, 0), mat10, mat11, mat12, mat13, mat14);
    pixels += convolution1x5(offset(src, -2, 1), mat15, mat16, mat17, mat18, mat19);
    pixels += convolution1x5(offset(src, -2, 2), mat20, mat21, mat22, mat23, mat24);

    if(scale > 0)
    {
        pixels /= (VEC_DATA_TYPE(DATA_TYPE, 8))scale;
    }

    return convert_short8_sat(pixels);
}

#ifndef DYNAMIC_MATRIX_CONVOLUTION

/** Apply a 1x5 static convolution matrix to a single channel U8 input image and output a single temporary channel image(Support U16, S16, S32).
 *
 * @attention The matrix coefficients (MAT0, MAT1, MAT2, MAT3, MAT4) and DATA_TYPE need to be passed at compile time:\n
 * e.g. -DMAT0=1 -DMAT2=2, -DMAT3=3, -DMAT4=4, -DDATA_TYPE=int
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: U8
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported data types: U16, S16, S32
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void convolution_separable1x5_static(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Output pixels
    VEC_DATA_TYPE(DATA_TYPE, 8)
    pixels = convolution1x5(offset(&src, -2, 0), MAT0, MAT1, MAT2, MAT3, MAT4);

    // Store result in dst
    vstore8(pixels, 0, (__global DATA_TYPE *)dst.ptr);
}

/** Apply a 5x1 static convolution matrix to a single channel U8 input image and output a single channel image.
 *
 * @attention The matrix coefficients (MAT5, MAT6, MAT7, MAT8, MAT9, SCALE), COMPUTE_TYPE and DATA_TYPE_OUT need to be passed at compile time:\n
 * e.g. -DMAT5=1 -DMAT6=2, -DMAT7=3, -DMAT8=4, -DMAT9=5, -DSCALE=6, -DCOMPUTE_TYPE=int, -DDATA_TYPE_OUT=int
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: U16, S16, S32
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
__kernel void convolution_separable5x1_static(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Output pixels
    VEC_DATA_TYPE(COMPUTE_TYPE, 8)
    pixels = convolution5x1(&src, MAT5, MAT6, MAT7, MAT8, MAT9);

    // Divide by the scale
    pixels /= (VEC_DATA_TYPE(COMPUTE_TYPE, 8))SCALE;

    // Store result in dst
    vstore8(CONVERT_SAT(pixels, VEC_DATA_TYPE(DATA_TYPE_OUT, 8)), 0, (__global DATA_TYPE_OUT *)dst.ptr);
}

/** Apply a static 5x5 convolution matrix to a single channel U8 input image and output a single channel image including borders
 *
 * @attention The matrix coefficients(MAT0, MAT1, ... MAT24, SCALE), DATA_TYPE_OUT need to be passed at compile time:\n
 * e.g. -DMAT0=1 -DMAT1=2, ... -DMAT24=24, -DSCALE=6, -DDATA_TYPE_OUT=int
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: U8
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
__kernel void convolution5x5_static(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    short8 pixels = convolution5x5(&src,
                                   MAT0, MAT1, MAT2, MAT3, MAT4, MAT5, MAT6, MAT7, MAT8, MAT9, MAT10, MAT11, MAT12, MAT13,
                                   MAT14, MAT15, MAT16, MAT17, MAT18, MAT19, MAT20, MAT21, MAT22, MAT23, MAT24, SCALE);

    // Store the result as is in dst
    vstore8(CONVERT_SAT(pixels, VEC_DATA_TYPE(DATA_TYPE_OUT, 8)), 0, (__global DATA_TYPE_OUT *)dst.ptr);
}

#endif // DYNAMIC_MATRIX_CONVOLUTION
