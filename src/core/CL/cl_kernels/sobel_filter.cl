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

/***********************************************/
/*   Begin implementation of Sobel3x3 filter   */
/***********************************************/

/** This OpenCL kernel that computes a Sobel3x3 filter.
 *
 * @attention To enable computation of the X gradient -DGRAD_X must be passed at compile time, while computation of the Y gradient
 * is performed when -DGRAD_Y is used. You can use both when computation of both gradients is required.
 *
 * @param[in]  src_ptr                              Pointer to the source image. Supported data types: U8
 * @param[in]  src_stride_x                         Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                           src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                         Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                           src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes    The offset of the first element in the source image
 * @param[out] dst_gx_ptr                           Pointer to the destination image. Supported data types: S16
 * @param[in]  dst_gx_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_gx_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_gx_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_gx_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_gx_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[out] dst_gy_ptr                           Pointer to the destination image. Supported data types: S16
 * @param[in]  dst_gy_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_gy_step_x                        dst_gy_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_gy_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_gy_step_y                        dst_gy_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_gy_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void sobel3x3(
    IMAGE_DECLARATION(src)
#ifdef GRAD_X
    ,
    IMAGE_DECLARATION(dst_gx)
#endif /* GRAD_X */
#ifdef GRAD_Y
    ,
    IMAGE_DECLARATION(dst_gy)
#endif /* GRAD_Y */
)
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
#ifdef GRAD_X
    Image dst_gx = CONVERT_TO_IMAGE_STRUCT(dst_gx);
#endif /* GRAD_X */
#ifdef GRAD_Y
    Image dst_gy = CONVERT_TO_IMAGE_STRUCT(dst_gy);
#endif /* GRAD_Y */

    // Output pixels
#ifdef GRAD_X
    short8 gx = (short8)0;
#endif /* GRAD_X */
#ifdef GRAD_Y
    short8 gy = (short8)0;
#endif /* GRAD_Y */

    // Row0
    uchar16 temp   = vload16(0, offset(&src, -1, -1));
    short8  left   = convert_short8(temp.s01234567);
    short8  middle = convert_short8(temp.s12345678);
    short8  right  = convert_short8(temp.s23456789);
#ifdef GRAD_X
    gx += left * (short8)(-1);
    gx += right * (short8)(+1);
#endif /* GRAD_X */
#ifdef GRAD_Y
    gy += left * (short8)(-1);
    gy += middle * (short8)(-2);
    gy += right * (short8)(-1);
#endif /* GRAD_Y */

    // Row1
    temp  = vload16(0, offset(&src, -1, 0));
    left  = convert_short8(temp.s01234567);
    right = convert_short8(temp.s23456789);
#ifdef GRAD_X
    gx += left * (short8)(-2);
    gx += right * (short8)(+2);
#endif /* GRAD_X */

    // Row2
    temp   = vload16(0, offset(&src, -1, 1));
    left   = convert_short8(temp.s01234567);
    middle = convert_short8(temp.s12345678);
    right  = convert_short8(temp.s23456789);
#ifdef GRAD_X
    gx += left * (short8)(-1);
    gx += right * (short8)(+1);
#endif /* GRAD_X */
#ifdef GRAD_Y
    gy += left * (short8)(+1);
    gy += middle * (short8)(+2);
    gy += right * (short8)(+1);
#endif /* GRAD_Y */

    // Store results
#ifdef GRAD_X
    vstore8(gx, 0, ((__global short *)dst_gx.ptr));
#endif /* GRAD_X */
#ifdef GRAD_Y
    vstore8(gy, 0, ((__global short *)dst_gy.ptr));
#endif /* GRAD_Y */
}

/**********************************************/
/*    End implementation of Sobel3x3 filter   */
/**********************************************/

/***********************************************/
/*   Begin implementation of Sobel5x5 filter   */
/***********************************************/

/** Compute a 1D horizontal sobel filter 1x5 for 8 bytes assuming the input is made of 1 channel of 1 byte (i.e 8 pixels).
 *
 * @param[in] src             Pointer to source image.
 * @param[in] left1_coeff_gx  Weight of the most left pixel for gx
 * @param[in] left2_coeff_gx  Weight of the left pixel for gx
 * @param[in] middle_coeff_gx Weight of the middle pixel for gx
 * @param[in] right1_coeff_gx Weight of the right pixel for gx
 * @param[in] right2_coeff_gx Weight of the most right pixel for gx
 * @param[in] left1_coeff_gy  Weight of the most left pixel for gy
 * @param[in] left2_coeff_gy  Weight of the left pixel for gy
 * @param[in] middle_coeff_gy Weight of the middle pixel for gy
 * @param[in] right1_coeff_gy Weight of the right pixel for gy
 * @param[in] right2_coeff_gy Weight of the most right pixel for gy
 *
 * @return a short16 containing short8 gx and short8 gy values.
 */
short16 sobel1x5(
    Image      *src,
    const short left1_coeff_gx,
    const short left2_coeff_gx,
    const short middle_coeff_gx,
    const short right1_coeff_gx,
    const short right2_coeff_gx,
    const short left1_coeff_gy,
    const short left2_coeff_gy,
    const short middle_coeff_gy,
    const short right1_coeff_gy,
    const short right2_coeff_gy)
{
    uchar16 temp = vload16(0, offset(src, -2, 0));
    short8  gx   = 0;
    short8  gy   = 0;
    short8  val;

    val = convert_short8(temp.s01234567);
    gx += val * (short8)left1_coeff_gx;
    gy += val * (short8)left1_coeff_gy;

    val = convert_short8(temp.s12345678);
    gx += val * (short8)left2_coeff_gx;
    gy += val * (short8)left2_coeff_gy;

    val = convert_short8(temp.s23456789);
    gx += val * (short8)middle_coeff_gx;
    gy += val * (short8)middle_coeff_gy;

    val = convert_short8(temp.s3456789a);
    gx += val * (short8)right1_coeff_gx;
    gy += val * (short8)right1_coeff_gy;

    val = convert_short8(temp.s456789ab);
    gx += val * (short8)right2_coeff_gx;
    gy += val * (short8)right2_coeff_gy;

    return (short16)(gx, gy);
}

/** Compute a 1D vertical sobel filter 5x1 for 8 bytes assuming the input is made of 1 channel of 1 byte (i.e 8 pixels).
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
short8 sobel5x1(
    Image      *src,
    const short up1_coeff,
    const short up2_coeff,
    const short middle_coeff,
    const short down1_coeff,
    const short down2_coeff)
{
    short8 val;
    short8 out = (short8)0;

    val = vload8(0, (__global short *)offset(src, 0, -2));
    out += val * (short8)up1_coeff;

    val = vload8(0, (__global short *)offset(src, 0, -1));
    out += val * (short8)up2_coeff;

    val = vload8(0, (__global short *)offset(src, 0, 0));
    out += val * (short8)middle_coeff;

    val = vload8(0, (__global short *)offset(src, 0, 1));
    out += val * (short8)down1_coeff;

    val = vload8(0, (__global short *)offset(src, 0, 2));
    out += val * (short8)down2_coeff;

    return (short8)(out);
}

/** Apply a 1x5 sobel matrix to a single channel U8 input image and output two temporary channel S16 images.
 *
 * @attention To enable computation of the X gradient -DGRAD_X must be passed at compile time, while computation of the Y gradient
 * is performed when -DGRAD_Y is used. You can use both when computation of both gradients is required.
 *
 * @param[in]  src_ptr                              Pointer to the source image.. Supported data types: U8
 * @param[in]  src_stride_x                         Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                           src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                         Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                           src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes    The offset of the first element in the source image
 * @param[out] dst_gx_ptr                           Pointer to the destination image.. Supported data types: S16
 * @param[in]  dst_gx_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_gx_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_gx_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_gx_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_gx_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[out] dst_gy_ptr                           Pointer to the destination image. Supported data types: S16
 * @param[in]  dst_gy_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_gy_step_x                        dst_gy_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_gy_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_gy_step_y                        dst_gy_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_gy_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void sobel_separable1x5(
    IMAGE_DECLARATION(src)
#ifdef GRAD_X
    ,
    IMAGE_DECLARATION(dst_gx)
#endif /* GRAD_X */
#ifdef GRAD_Y
    ,
    IMAGE_DECLARATION(dst_gy)
#endif /* GRAD_Y */
)
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
#ifdef GRAD_X
    Image dst_gx = CONVERT_TO_IMAGE_STRUCT(dst_gx);
#endif /* GRAD_X */
#ifdef GRAD_Y
    Image dst_gy = CONVERT_TO_IMAGE_STRUCT(dst_gy);
#endif /* GRAD_Y */

    // Output pixels
    short16 gx_gy = sobel1x5(&src,
                             -1, -2, 0, 2, 1,
                             1, 4, 6, 4, 1);

    // Store result in dst
#ifdef GRAD_X
    vstore8(gx_gy.s01234567, 0, ((__global short *)dst_gx.ptr));
#endif /* GRAD_X */
#ifdef GRAD_Y
    vstore8(gx_gy.s89ABCDEF, 0, ((__global short *)dst_gy.ptr));
#endif /* GRAD_Y */
}

/** Apply a 5x1 convolution matrix to two single channel S16 input temporary images
 *  and output two single channel S16 images.
 *
 * @attention To enable computation of the X gradient -DGRAD_X must be passed at compile time, while computation of the Y gradient
 * is performed when -DGRAD_Y is used. You can use both when computation of both gradients is required.
 *
 * @param[in]  src_x_ptr                            Pointer to the source image.. Supported data types: S16
 * @param[in]  src_x_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  src_x_step_x                         src_x_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_x_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_x_step_y                         src_x_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_x_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] dst_gx_ptr                           Pointer to the destination image. Supported data types: S16
 * @param[in]  dst_gx_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_gx_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_gx_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_gx_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_gx_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[in]  src_y_ptr                            Pointer to the source image. Supported data types: S16
 * @param[in]  src_y_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  src_y_step_x                         src_y_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_y_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_y_step_y                         src_y_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_y_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] dst_gy_ptr                           Pointer to the destination image. Supported data types: S16
 * @param[in]  dst_gy_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_gy_step_x                        dst_gy_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_gy_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_gy_step_y                        dst_gy_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_gy_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[in]  dummy                                Dummy parameter to easy conditional inclusion
 */
__kernel void sobel_separable5x1(
#ifdef GRAD_X
    IMAGE_DECLARATION(src_x),
    IMAGE_DECLARATION(dst_gx),
#endif /* GRAD_X */
#ifdef GRAD_Y
    IMAGE_DECLARATION(src_y),
    IMAGE_DECLARATION(dst_gy),
#endif /* GRAD_Y */
    int dummy)
{
#ifdef GRAD_X
    Image src_x  = CONVERT_TO_IMAGE_STRUCT(src_x);
    Image dst_gx = CONVERT_TO_IMAGE_STRUCT(dst_gx);
#endif /* GRAD_X */
#ifdef GRAD_Y
    Image src_y  = CONVERT_TO_IMAGE_STRUCT(src_y);
    Image dst_gy = CONVERT_TO_IMAGE_STRUCT(dst_gy);
#endif /* GRAD_Y */

#ifdef GRAD_X
    short8 gx = sobel5x1(&src_x,
                         1, 4, 6, 4, 1);
    vstore8(gx, 0, ((__global short *)dst_gx.ptr));
#endif /* GRAD_X */
#ifdef GRAD_Y
    short8 gy = sobel5x1(&src_y,
                         -1, -2, 0, 2, 1);
    vstore8(gy, 0, ((__global short *)dst_gy.ptr));
#endif /* GRAD_Y */
}

/**********************************************/
/*    End implementation of Sobel5x5 filter   */
/**********************************************/

/***********************************************/
/*   Begin implementation of Sobel7x7 filter   */
/***********************************************/

/* Sobel 1x7 horizontal X / 7x1 vertical Y coefficients */
#define X0 -1
#define X1 -4
#define X2 -5
#define X3 0
#define X4 5
#define X5 4
#define X6 1

/* Sobel 1x7 vertical X / 7x1 horizontal Y coefficients */
#define Y0 1
#define Y1 6
#define Y2 15
#define Y3 20
#define Y4 15
#define Y5 6
#define Y6 1

/* Calculates single horizontal iteration. */
#define SOBEL1x1_HOR(src, gx, gy, idx)                               \
    {                                                                \
        int8 val = convert_int8(vload8(0, offset(src, idx - 3, 0))); \
        gx += val * X##idx;                                          \
        gy += val * Y##idx;                                          \
    }

/* Calculates single vertical iteration. */
#define SOBEL1x1_VERT(src, g, direction, idx)                          \
    {                                                                  \
        int8 val = vload8(0, (__global int *)offset(src, 0, idx - 3)); \
        g += val * (int8)direction##idx;                               \
    }

/* Calculates a 1x7 horizontal iteration. */
#define SOBEL1x7(ptr, gx, gy)                        \
    SOBEL1x1_HOR(ptr, gx, gy, 0)                     \
    SOBEL1x1_HOR(ptr, gx, gy, 1)                 \
    SOBEL1x1_HOR(ptr, gx, gy, 2)             \
    SOBEL1x1_HOR(ptr, gx, gy, 3)         \
    SOBEL1x1_HOR(ptr, gx, gy, 4)     \
    SOBEL1x1_HOR(ptr, gx, gy, 5) \
    SOBEL1x1_HOR(ptr, gx, gy, 6)

/* Calculates a 7x1 vertical iteration. */
#define SOBEL7x1(ptr, g, direction)                         \
    SOBEL1x1_VERT(ptr, g, direction, 0)                     \
    SOBEL1x1_VERT(ptr, g, direction, 1)                 \
    SOBEL1x1_VERT(ptr, g, direction, 2)             \
    SOBEL1x1_VERT(ptr, g, direction, 3)         \
    SOBEL1x1_VERT(ptr, g, direction, 4)     \
    SOBEL1x1_VERT(ptr, g, direction, 5) \
    SOBEL1x1_VERT(ptr, g, direction, 6)

/** Apply a 1x7 sobel matrix to a single channel U8 input image and output two temporary channel S16 images and leave the borders undefined.
 *
 * @attention To enable computation of the X gradient -DGRAD_X must be passed at compile time, while computation of the Y gradient
 * is performed when -DGRAD_Y is used. You can use both when computation of both gradients is required.
 *
 * @param[in]  src_ptr                              Pointer to the source image. Supported data types: U8
 * @param[in]  src_stride_x                         Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                           src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                         Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                           src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes    The offset of the first element in the source image
 * @param[out] dst_gx_ptr                           Pointer to the destination image. Supported data types: S32
 * @param[in]  dst_gx_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_gx_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_gx_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_gx_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_gx_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[out] dst_gy_ptr                           Pointer to the destination image. Supported data types: S32
 * @param[in]  dst_gy_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_gy_step_x                        dst_gy_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_gy_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_gy_step_y                        dst_gy_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_gy_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void sobel_separable1x7(
    IMAGE_DECLARATION(src)
#ifdef GRAD_X
    ,
    IMAGE_DECLARATION(dst_gx)
#endif /* GRAD_X */
#ifdef GRAD_Y
    ,
    IMAGE_DECLARATION(dst_gy)
#endif /* GRAD_Y */
)
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
#ifdef GRAD_X
    Image dst_gx = CONVERT_TO_IMAGE_STRUCT(dst_gx);
#endif /* GRAD_X */
#ifdef GRAD_Y
    Image dst_gy = CONVERT_TO_IMAGE_STRUCT(dst_gy);
#endif /* GRAD_Y */
    int8 gx = (int8)0;
    int8 gy = (int8)0;

    SOBEL1x7(&src, gx, gy);

    // Store result in dst
#ifdef GRAD_X
    vstore8(gx, 0, ((__global int *)dst_gx.ptr));
#endif /* GRAD_X */
#ifdef GRAD_Y
    vstore8(gy, 0, ((__global int *)dst_gy.ptr));
#endif /* GRAD_Y */
}

/** Apply a 7x1 convolution matrix to two single channel S16 input temporary images and output two single channel S16 images and leave the borders undefined.
 *
 * @attention To enable computation of the X gradient -DGRAD_X must be passed at compile time, while computation of the Y gradient
 * is performed when -DGRAD_Y is used. You can use both when computation of both gradients is required.
 *
 * @param[in]  src_x_ptr                            Pointer to the source image. Supported data types: S32
 * @param[in]  src_x_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  src_x_step_x                         src_x_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_x_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_x_step_y                         src_x_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_x_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] dst_gx_ptr                           Pointer to the destination image. Supported data types: S16
 * @param[in]  dst_gx_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_gx_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_gx_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_gx_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_gx_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[in]  src_y_ptr                            Pointer to the source image. Supported data types: S32
 * @param[in]  src_y_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  src_y_step_x                         src_y_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_y_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_y_step_y                         src_y_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_y_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] dst_gy_ptr                           Pointer to the destination image. Supported data types: S16
 * @param[in]  dst_gy_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_gy_step_x                        dst_gy_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_gy_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_gy_step_y                        dst_gy_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_gy_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[in]  dummy                                Dummy parameter to easy conditional inclusion
 */
__kernel void sobel_separable7x1(
#ifdef GRAD_X
    IMAGE_DECLARATION(src_x),
    IMAGE_DECLARATION(dst_gx),
#endif /* GRAD_X */
#ifdef GRAD_Y
    IMAGE_DECLARATION(src_y),
    IMAGE_DECLARATION(dst_gy),
#endif /* GRAD_Y */
    int dummy)
{
#ifdef GRAD_X
    Image src_x  = CONVERT_TO_IMAGE_STRUCT(src_x);
    Image dst_gx = CONVERT_TO_IMAGE_STRUCT(dst_gx);
#endif /* GRAD_X */
#ifdef GRAD_Y
    Image src_y  = CONVERT_TO_IMAGE_STRUCT(src_y);
    Image dst_gy = CONVERT_TO_IMAGE_STRUCT(dst_gy);
#endif /* GRAD_Y */

    // Output pixels
#ifdef GRAD_X
    int8 gx = 0;
    SOBEL7x1(&src_x, gx, Y);
    vstore8(gx, 0, (__global int *)dst_gx.ptr);
#endif /* GRAD_X */
#ifdef GRAD_Y
    int8 gy = 0;
    SOBEL7x1(&src_y, gy, X);
    vstore8(gy, 0, (__global int *)dst_gy.ptr);
#endif /* GRAD_Y */
}

/**********************************************/
/*    End implementation of Sobel7x7 filter   */
/**********************************************/
