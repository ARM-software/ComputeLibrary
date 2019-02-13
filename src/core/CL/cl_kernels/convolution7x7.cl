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

/** Compute a 1D horizontal convolution of size 7 for 8 bytes assuming the input is made of 1 channel of 1 byte (i.e 8 pixels).
 *
 * @param[in] left_pixel   Pointer to the left pixel
 * @param[in] left1_coeff  Weight of the most left pixel
 * @param[in] left2_coeff  Weight of the second left pixel
 * @param[in] left3_coeff  Weight of the left pixel
 * @param[in] middle_coeff Weight of the middle pixel
 * @param[in] right1_coeff Weight of the right pixel
 * @param[in] right2_coeff Weight of the second right pixel
 * @param[in] right3_coeff Weight of the most right pixel
 *
 * @return a short8 containing 8 convoluted values.
 */
VEC_DATA_TYPE(DATA_TYPE, 8)
convolution1x7(
    __global const uchar *left_pixel,
    const short           left1_coeff,
    const short           left2_coeff,
    const short           left3_coeff,
    const short           middle_coeff,
    const short           right1_coeff,
    const short           right2_coeff,
    const short           right3_coeff)
{
    uchar16 temp = vload16(0, left_pixel);

    VEC_DATA_TYPE(DATA_TYPE, 8)
    left1 = CONVERT(temp.s01234567, VEC_DATA_TYPE(DATA_TYPE, 8));
    VEC_DATA_TYPE(DATA_TYPE, 8)
    left2 = CONVERT(temp.s12345678, VEC_DATA_TYPE(DATA_TYPE, 8));
    VEC_DATA_TYPE(DATA_TYPE, 8)
    left3 = CONVERT(temp.s23456789, VEC_DATA_TYPE(DATA_TYPE, 8));
    VEC_DATA_TYPE(DATA_TYPE, 8)
    middle = CONVERT(temp.s3456789a, VEC_DATA_TYPE(DATA_TYPE, 8));
    VEC_DATA_TYPE(DATA_TYPE, 8)
    right1 = CONVERT(temp.s456789ab, VEC_DATA_TYPE(DATA_TYPE, 8));
    VEC_DATA_TYPE(DATA_TYPE, 8)
    right2 = CONVERT(temp.s56789abc, VEC_DATA_TYPE(DATA_TYPE, 8));
    VEC_DATA_TYPE(DATA_TYPE, 8)
    right3 = CONVERT(temp.s6789abcd, VEC_DATA_TYPE(DATA_TYPE, 8));

    return left1 * (VEC_DATA_TYPE(DATA_TYPE, 8))left1_coeff + left2 * (VEC_DATA_TYPE(DATA_TYPE, 8))left2_coeff + left3 * (VEC_DATA_TYPE(DATA_TYPE, 8))left3_coeff + middle * (VEC_DATA_TYPE(DATA_TYPE,
            8))middle_coeff + right1 * (VEC_DATA_TYPE(DATA_TYPE, 8))right1_coeff + right2 * (VEC_DATA_TYPE(DATA_TYPE, 8))right2_coeff + right3 * (VEC_DATA_TYPE(DATA_TYPE, 8))right3_coeff;
}

/** Compute a 1D vertical convolution of size 7 for 8 bytes assuming the input is made of 1 channel of 1 byte (i.e 8 pixels).
 *
 * @param[in] src          Pointer to source image.
 * @param[in] up1_coeff    Weight of the most up pixel
 * @param[in] up2_coeff    Weight of the second up pixel
 * @param[in] up3_coeff    Weight of the up pixel
 * @param[in] middle_coeff Weight of the middle pixel
 * @param[in] down1_coeff  Weight of the down pixel
 * @param[in] down2_coeff  Weight of the second down pixel
 * @param[in] down3_coeff  Weight of the third down pixel
 *
 * @return a short8 containing 8 convoluted values.
 */
VEC_DATA_TYPE(COMPUTE_TYPE, 8)
convolution7x1(
    Image      *src,
    const short up1_coeff,
    const short up2_coeff,
    const short up3_coeff,
    const short middle_coeff,
    const short down1_coeff,
    const short down2_coeff,
    const short down3_coeff)
{
    VEC_DATA_TYPE(COMPUTE_TYPE, 8)
    val;
    VEC_DATA_TYPE(COMPUTE_TYPE, 8)
    out = (VEC_DATA_TYPE(COMPUTE_TYPE, 8))0;

    val = CONVERT(vload8(0, (__global DATA_TYPE *)offset(src, 0, -3)), VEC_DATA_TYPE(COMPUTE_TYPE, 8));
    out += val * (VEC_DATA_TYPE(COMPUTE_TYPE, 8))up1_coeff;

    val = CONVERT(vload8(0, (__global DATA_TYPE *)offset(src, 0, -2)), VEC_DATA_TYPE(COMPUTE_TYPE, 8));
    out += val * (VEC_DATA_TYPE(COMPUTE_TYPE, 8))up2_coeff;

    val = CONVERT(vload8(0, (__global DATA_TYPE *)offset(src, 0, -1)), VEC_DATA_TYPE(COMPUTE_TYPE, 8));
    out += val * (VEC_DATA_TYPE(COMPUTE_TYPE, 8))up3_coeff;

    val = CONVERT(vload8(0, (__global DATA_TYPE *)offset(src, 0, 0)), VEC_DATA_TYPE(COMPUTE_TYPE, 8));
    out += val * (VEC_DATA_TYPE(COMPUTE_TYPE, 8))middle_coeff;

    val = CONVERT(vload8(0, (__global DATA_TYPE *)offset(src, 0, 1)), VEC_DATA_TYPE(COMPUTE_TYPE, 8));
    out += val * (VEC_DATA_TYPE(COMPUTE_TYPE, 8))down1_coeff;

    val = CONVERT(vload8(0, (__global DATA_TYPE *)offset(src, 0, 2)), VEC_DATA_TYPE(COMPUTE_TYPE, 8));
    out += val * (VEC_DATA_TYPE(COMPUTE_TYPE, 8))down2_coeff;

    val = CONVERT(vload8(0, (__global DATA_TYPE *)offset(src, 0, 3)), VEC_DATA_TYPE(COMPUTE_TYPE, 8));
    out += val * (VEC_DATA_TYPE(COMPUTE_TYPE, 8))down3_coeff;

    return out;
}

/** Apply a 7x7 convolution matrix to a single channel U8 input image and return the result.
 *
 * Convolution matrix layout:\n
 * [  mat0,  mat1,  mat2,  mat3 , mat4,  mat5,  mat6 ]\n
 * [  mat7,  mat8,  mat9,  mat10, mat11, mat12, mat13 ]\n
 * [  mat14, mat15, mat16, mat17, mat18, mat19, mat20 ]\n
 * [  mat21, mat22, mat23, mat24, mat25, mat26, mat27 ]\n
 * [  mat28, mat29, mat30, mat31, mat32, mat33, mat34 ]\n
 * [  mat35, mat36, mat37, mat38, mat39, mat40, mat41 ]\n
 * [  mat42, mat43, mat44, mat45, mat46, mat47, mat48 ]
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
 * @param[in] mat25 Coefficient from the convolution matrix
 * @param[in] mat26 Coefficient from the convolution matrix
 * @param[in] mat27 Coefficient from the convolution matrix
 * @param[in] mat28 Coefficient from the convolution matrix
 * @param[in] mat29 Coefficient from the convolution matrix
 * @param[in] mat30 Coefficient from the convolution matrix
 * @param[in] mat31 Coefficient from the convolution matrix
 * @param[in] mat32 Coefficient from the convolution matrix
 * @param[in] mat33 Coefficient from the convolution matrix
 * @param[in] mat34 Coefficient from the convolution matrix
 * @param[in] mat35 Coefficient from the convolution matrix
 * @param[in] mat36 Coefficient from the convolution matrix
 * @param[in] mat37 Coefficient from the convolution matrix
 * @param[in] mat38 Coefficient from the convolution matrix
 * @param[in] mat39 Coefficient from the convolution matrix
 * @param[in] mat40 Coefficient from the convolution matrix
 * @param[in] mat41 Coefficient from the convolution matrix
 * @param[in] mat42 Coefficient from the convolution matrix
 * @param[in] mat43 Coefficient from the convolution matrix
 * @param[in] mat44 Coefficient from the convolution matrix
 * @param[in] mat45 Coefficient from the convolution matrix
 * @param[in] mat46 Coefficient from the convolution matrix
 * @param[in] mat47 Coefficient from the convolution matrix
 * @param[in] mat48 Coefficient from the convolution matrix
 * @param[in] scale Convolution matrix scale (Sum of the coefficients, or 1 if the sum is 0)
 *
 */
short8 convolution7x7(
    Image      *src,
    const short mat0, const short mat1, const short mat2, const short mat3, const short mat4,
    const short mat5, const short mat6, const short mat7, const short mat8, const short mat9,
    const short mat10, const short mat11, const short mat12, const short mat13, const short mat14,
    const short mat15, const short mat16, const short mat17, const short mat18, const short mat19,
    const short mat20, const short mat21, const short mat22, const short mat23, const short mat24,
    const short mat25, const short mat26, const short mat27, const short mat28, const short mat29,
    const short mat30, const short mat31, const short mat32, const short mat33, const short mat34,
    const short mat35, const short mat36, const short mat37, const short mat38, const short mat39,
    const short mat40, const short mat41, const short mat42, const short mat43, const short mat44,
    const short mat45, const short mat46, const short mat47, const short mat48, uint scale)
{
    VEC_DATA_TYPE(DATA_TYPE, 8)
    pixels;

    pixels = convolution1x7(offset(src, -3, -3), mat0, mat1, mat2, mat3, mat4, mat5, mat6);
    pixels += convolution1x7(offset(src, -3, -2), mat7, mat8, mat9, mat10, mat11, mat12, mat13);
    pixels += convolution1x7(offset(src, -3, -1), mat14, mat15, mat16, mat17, mat18, mat19, mat20);
    pixels += convolution1x7(offset(src, -3, 0), mat21, mat22, mat23, mat24, mat25, mat26, mat27);
    pixels += convolution1x7(offset(src, -3, 1), mat28, mat29, mat30, mat31, mat32, mat33, mat34);
    pixels += convolution1x7(offset(src, -3, 2), mat35, mat36, mat37, mat38, mat39, mat40, mat41);
    pixels += convolution1x7(offset(src, -3, 3), mat42, mat43, mat44, mat45, mat46, mat47, mat48);

    if(scale > 0)
    {
        pixels /= (VEC_DATA_TYPE(DATA_TYPE, 8))scale;
    }

    return convert_short8_sat(pixels);
}

#ifndef DYNAMIC_MATRIX_CONVOLUTION

/** Apply a 1x7 static convolution matrix to a single channel U8 input image and output a single temporary channel image.
 *
 * @attention The matrix coefficients (MAT0, MAT1, MAT2, MAT3, MAT4, MAT5, MAT6) and DATA_TYPE need to be passed at compile time:\n
 * e.g. -DMAT0=1 -DMAT1=2, ... -DMAT6=6, -DDATA_TYPE=int
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
__kernel void convolution_separable1x7_static(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Output pixels
    VEC_DATA_TYPE(DATA_TYPE, 8)
    pixels = convolution1x7(offset(&src, -3, 0), MAT0, MAT1, MAT2, MAT3, MAT4, MAT5, MAT6);

    // Store result in dst
    vstore8(pixels, 0, (__global DATA_TYPE *)dst.ptr);
}

/** Apply a 7x1 static convolution matrix to a single channel U8 input image and output a single channel image.
 *
 * @attention The matrix coefficients (MAT7, MAT8, MAT9, MAT10, MAT11, MAT12, MAT13, SCALE), COMPUTE_TYPE and DATA_TYPE_OUT need to be passed at compile time:\n
 * e.g. -DMAT0=7 -DMAT1=8, ... -DMAT24=13, -DSCALE=6, -DCOMPUTE_TYPE=int, -DDATA_TYPE_OUT=int
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
__kernel void convolution_separable7x1_static(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Output pixels
    VEC_DATA_TYPE(COMPUTE_TYPE, 8)
    pixels = convolution7x1(&src, MAT7, MAT8, MAT9, MAT10, MAT11, MAT12, MAT13);

    // Divide by the scale
    pixels /= (VEC_DATA_TYPE(COMPUTE_TYPE, 8))SCALE;

    // Store result in dst
    vstore8(CONVERT_SAT(pixels, VEC_DATA_TYPE(DATA_TYPE_OUT, 8)), 0, (__global DATA_TYPE_OUT *)dst.ptr);
}

/** Apply a static 7x7 convolution matrix to a single channel U8 input image and output a single channel U8 image including the borders.
 *
 * @attention The matrix coefficients(MAT0, MAT1, ... MAT48, SCALE), DATA_TYPE_OUT need to be passed at compile time:\n
 * e.g. -DMAT0=7 -DMAT1=8, ... -DMAT48=48, -DSCALE=6, -DDATA_TYPE_OUT=int
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
__kernel void convolution7x7_static(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    short8 pixels = convolution7x7(&src,
                                   MAT0, MAT1, MAT2, MAT3, MAT4, MAT5, MAT6, MAT7, MAT8, MAT9, MAT10, MAT11, MAT12, MAT13,
                                   MAT14, MAT15, MAT16, MAT17, MAT18, MAT19, MAT20, MAT21, MAT22, MAT23, MAT24, MAT25,
                                   MAT26, MAT27, MAT28, MAT29, MAT30, MAT31, MAT32, MAT33, MAT34, MAT35, MAT36, MAT37,
                                   MAT38, MAT39, MAT40, MAT41, MAT42, MAT43, MAT44, MAT45, MAT46, MAT47, MAT48, SCALE);

    // Clamp results to [ 0, 255 ] and store them in dst
    vstore8(CONVERT_SAT(pixels, VEC_DATA_TYPE(DATA_TYPE_OUT, 8)), 0, (__global DATA_TYPE_OUT *)dst.ptr);
}

#endif // DYNAMIC_MATRIX_CONVOLUTION
