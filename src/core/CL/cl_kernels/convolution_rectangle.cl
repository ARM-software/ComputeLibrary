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
#include "convolution3x3.cl"
#include "convolution5x5.cl"
#include "convolution7x7.cl"
#include "convolution9x9.cl"
#include "helpers.h"

#define MAT_INDEX(i) MAT##i

#ifndef DATA_TYPE
#define DATA_TYPE short
#endif /* DATA_TYPE */

#ifndef COMPUTE_TYPE
#define COMPUTE_TYPE int
#endif /* COMPUTE_TYPE */

#ifndef DATA_TYPE_OUT
#define DATA_TYPE_OUT uchar
#endif /* DATA_TYPE_OUT */

#ifndef DYNAMIC_MATRIX_CONVOLUTION

/** Apply a rectangle matrix to a single channel U8 input image and output a single channel image including borders
 *
 * @attention The matrix coefficients(MAT0, MAT1, ... MAT80, SCALE), MATRIX_WIDTH, MATRIX_HEIGHT, COMPUTE_TYPE, DATA_TYPE, DATA_TYPE_OUT need to be passed at compile time:\n
 * e.g. -DMAT0=0 -DMAT1=1, ... -DMAT80=80, -DSCALE=6, -DMATRIX_WIDTH=3, -DMATRIX_HEIGHT=5, -DCOMPUTE_TYPE=int, -DDATA_TYPE=int, -DDATA_TYPE_OUT=int
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
__kernel void convolution_rectangle(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    short matrix_coeff[81] =
    {
        MAT0, MAT1, MAT2, MAT3, MAT4, MAT5, MAT6, MAT7, MAT8,
        MAT9, MAT10, MAT11, MAT12, MAT13, MAT14, MAT15, MAT16, MAT17,
        MAT18, MAT19, MAT20, MAT21, MAT22, MAT23, MAT24, MAT25, MAT26,
        MAT27, MAT28, MAT29, MAT30, MAT31, MAT32, MAT33, MAT34, MAT35,
        MAT36, MAT37, MAT38, MAT39, MAT40, MAT41, MAT42, MAT43, MAT44,
        MAT45, MAT46, MAT47, MAT48, MAT49, MAT50, MAT51, MAT52, MAT53,
        MAT54, MAT55, MAT56, MAT57, MAT58, MAT59, MAT60, MAT61, MAT62,
        MAT63, MAT64, MAT65, MAT66, MAT67, MAT68, MAT69, MAT70, MAT71,
        MAT72, MAT73, MAT74, MAT75, MAT76, MAT77, MAT78, MAT79, MAT80
    };

    VEC_DATA_TYPE(DATA_TYPE, 8)
    pixels = (VEC_DATA_TYPE(DATA_TYPE, 8))0;

    for(int i = 0; i < MATRIX_HEIGHT; i++)
    {
#if MATRIX_WIDTH == 3
        pixels += convolution1x3(offset(&src, -1, -(MATRIX_HEIGHT / 2) + i), matrix_coeff[0 + i * 3], matrix_coeff[1 + i * 3],
                                 matrix_coeff[2 + i * 3]);
#endif /* MATRIX_WIDTH */

#if MATRIX_WIDTH == 5
        pixels += convolution1x5(offset(&src, -2, -(MATRIX_HEIGHT / 2) + i), matrix_coeff[0 + i * 5], matrix_coeff[1 + i * 5],
                                 matrix_coeff[2 + i * 5], matrix_coeff[3 + i * 5], matrix_coeff[4 + i * 5]);
#endif /* MATRIX_WIDTH */

#if MATRIX_WIDTH == 7
        pixels += convolution1x7(offset(&src, -3, -(MATRIX_HEIGHT / 2) + i), matrix_coeff[0 + i * 7], matrix_coeff[1 + i * 7],
                                 matrix_coeff[2 + i * 7], matrix_coeff[3 + i * 7], matrix_coeff[4 + i * 7],
                                 matrix_coeff[5 + i * 7], matrix_coeff[6 + i * 7]);
#endif /* MATRIX_WIDTH */

#if MATRIX_WIDTH == 9
        pixels += convolution1x9(offset(&src, -4, -(MATRIX_HEIGHT / 2) + i), matrix_coeff[0 + i * 9], matrix_coeff[1 + i * 9],
                                 matrix_coeff[2 + i * 9], matrix_coeff[3 + i * 9], matrix_coeff[4 + i * 9],
                                 matrix_coeff[5 + i * 9], matrix_coeff[6 + i * 9], matrix_coeff[7 + i * 9], matrix_coeff[8 + i * 9]);
#endif /* MATRIX_WIDTH */
    }

    pixels /= (VEC_DATA_TYPE(DATA_TYPE, 8))SCALE;

    // Store the result as is in dst
    vstore8(CONVERT_SAT(pixels, VEC_DATA_TYPE(DATA_TYPE_OUT, 8)), 0, ((__global DATA_TYPE_OUT *)dst.ptr));
}

#endif /* not DYNAMIC_MATRIX_CONVOLUTION */
