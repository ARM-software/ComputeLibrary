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
#include "types.h"

/* The map table to retrieve the 16 texels in the Bresenham circle of radius 3 with center in P.
 *
 *      . . F 0 1 . . .
 *      . E . . . 2 . .
 *      D . . . . . 3 .
 *      C . . P . . 4 .
 *      B . . . . . 5 .
 *      . A . . . 6 . .
 *      . . 9 8 7 . . .
 */
constant int offsets_s[16][2] =
{
    { 0, -3 },  // 0
    { 1, -3 },  // 1
    { 2, -2 },  // 2
    { 3, -1 },  // 3
    { 3, 0 },   // 4
    { 3, 1 },   // 5
    { 2, 2 },   // 6
    { 1, 3 },   // 7
    { 0, 3 },   // 8
    { -1, 3 },  // 9
    { -2, 2 },  // A
    { -3, 1 },  // B
    { -3, 0 },  // C
    { -3, -1 }, // D
    { -2, -2 }, // E
    { -1, -3 }, // F
};

/** Load a pixel and set the mask values.
 *
 * @param[in]  ptr         The pointer to the starting address of source image
 * @param[in]  a           Index to indicate the position in the Bresenham circle
 * @param[in]  stride      Stride of source image in x dimension
 * @param[in]  dark        The left end of the threshold range
 * @param[in]  bright      The right end of the threshold range
 * @param[out] dark_mask   The bit-set mask records dark pixels. Its bit is set as 1 if the corresponding pixel is dark
 * @param[out] bright_mask The bit-set mask records bright pixels. Its bit is set as 1 if the corresponding pixel is bright
 *
 */
#define LOAD_AND_SET_MASK(ptr, a, stride, dark, bright, dark_mask, bright_mask) \
    {                                                                           \
        unsigned char pixel;                                                    \
        pixel = *(ptr + (int)stride * offsets_s[a][1] + offsets_s[a][0]);       \
        dark_mask |= (pixel < dark) << a;                                       \
        bright_mask |= (pixel > bright) << a;                                   \
    }

/** Checks if a pixel is a corner. Pixel is considerred as a corner if the 9 continuous pixels in the Bresenham circle are bright or dark.
 *
 * @param[in]  bright_mask The mask recording postions of bright pixels
 * @param[in]  dark_mask   The mask recording postions of dark pixels
 * @param[out] isCorner    Indicate whether candidate pixel is corner
 */
#define CHECK_CORNER(bright_mask, dark_mask, isCorner)    \
    {                                                     \
        for(int i = 0; i < 16; i++)                       \
        {                                                 \
            isCorner |= ((bright_mask & 0x1FF) == 0x1FF); \
            isCorner |= ((dark_mask & 0x1FF) == 0x1FF);   \
            if(isCorner)                                  \
            {                                             \
                break;                                    \
            }                                             \
            bright_mask >>= 1;                            \
            dark_mask >>= 1;                              \
        }                                                 \
    }

/* Calculate pixel's strength */
uchar compute_strength(uchar candidate_pixel, __global unsigned char *ptr, unsigned int stride, unsigned char threshold)
{
    short a = threshold;
    short b = 255;
    while(b - a > 1)
    {
        uchar        c           = convert_uchar_sat((a + b) / 2);
        unsigned int bright_mask = 0;
        unsigned int dark_mask   = 0;

        unsigned char p_bright = add_sat(candidate_pixel, c);
        unsigned char p_dark   = sub_sat(candidate_pixel, c);

        bool isCorner = 0;

        for(uint i = 0; i < 16; i++)
        {
            LOAD_AND_SET_MASK(ptr, i, stride, p_dark, p_bright, dark_mask, bright_mask)
        }

        bright_mask |= (bright_mask << 16);
        dark_mask |= (dark_mask << 16);
        CHECK_CORNER(bright_mask, dark_mask, isCorner);

        if(isCorner)
        {
            a = convert_short(c);
        }
        else
        {
            b = convert_short(c);
        }
    }
    return a;
}

/** Fast corners implementation. Calculates and returns the strength of each pixel.
 *
 * The algorithm loops through the 16 pixels in the Bresenham circle and set low 16 bit of masks if corresponding pixel is bright
 * or dark. It then copy the low 16 bit to the high 16 bit of the masks. Right shift the bit to check whether the 9 continuous bits
 * from the LSB are set.
 *
 * @param[in]  input_ptr                            Pointer to the first source image. Supported data types: U8
 * @param[in]  input_stride_x                       Stride of the first source image in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the first source image in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the first source image
 * @param[out] output_ptr                           Pointer to the first source image. Supported data types: U8
 * @param[in]  output_stride_x                      Stride of the first source image in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the first source image in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the first source image
 * @param[in]  threshold_value                      Threshold value.
 *
 */
__kernel void fast_corners(
    IMAGE_DECLARATION(input),
    IMAGE_DECLARATION(output),
    float threshold_value)
{
    Image in  = CONVERT_TO_IMAGE_STRUCT(input);
    Image out = CONVERT_TO_IMAGE_STRUCT(output);

    const unsigned char threshold = (uchar)threshold_value;

    unsigned int bright_mask = 0;
    unsigned int dark_mask   = 0;

    unsigned char isCorner = 0;

    unsigned char p        = *in.ptr;
    unsigned char p_bright = add_sat(p, threshold);
    unsigned char p_dark   = sub_sat(p, threshold);

    LOAD_AND_SET_MASK(in.ptr, 0, input_stride_y, p_dark, p_bright, dark_mask, bright_mask)
    LOAD_AND_SET_MASK(in.ptr, 4, input_stride_y, p_dark, p_bright, dark_mask, bright_mask)
    LOAD_AND_SET_MASK(in.ptr, 8, input_stride_y, p_dark, p_bright, dark_mask, bright_mask)
    LOAD_AND_SET_MASK(in.ptr, 12, input_stride_y, p_dark, p_bright, dark_mask, bright_mask)

    if(((bright_mask | dark_mask) & 0x1111) == 0)
    {
        *out.ptr = 0;
        return;
    }

    LOAD_AND_SET_MASK(in.ptr, 1, input_stride_y, p_dark, p_bright, dark_mask, bright_mask)
    LOAD_AND_SET_MASK(in.ptr, 2, input_stride_y, p_dark, p_bright, dark_mask, bright_mask)
    LOAD_AND_SET_MASK(in.ptr, 3, input_stride_y, p_dark, p_bright, dark_mask, bright_mask)
    LOAD_AND_SET_MASK(in.ptr, 5, input_stride_y, p_dark, p_bright, dark_mask, bright_mask)
    LOAD_AND_SET_MASK(in.ptr, 6, input_stride_y, p_dark, p_bright, dark_mask, bright_mask)
    LOAD_AND_SET_MASK(in.ptr, 7, input_stride_y, p_dark, p_bright, dark_mask, bright_mask)
    LOAD_AND_SET_MASK(in.ptr, 9, input_stride_y, p_dark, p_bright, dark_mask, bright_mask)
    LOAD_AND_SET_MASK(in.ptr, 10, input_stride_y, p_dark, p_bright, dark_mask, bright_mask)
    LOAD_AND_SET_MASK(in.ptr, 11, input_stride_y, p_dark, p_bright, dark_mask, bright_mask)
    LOAD_AND_SET_MASK(in.ptr, 13, input_stride_y, p_dark, p_bright, dark_mask, bright_mask)
    LOAD_AND_SET_MASK(in.ptr, 14, input_stride_y, p_dark, p_bright, dark_mask, bright_mask)
    LOAD_AND_SET_MASK(in.ptr, 15, input_stride_y, p_dark, p_bright, dark_mask, bright_mask)

    bright_mask |= (bright_mask << 16);
    dark_mask |= (dark_mask << 16);

    CHECK_CORNER(bright_mask, dark_mask, isCorner)

    if(!isCorner)
    {
        *out.ptr = 0;
        return;
    }

#ifdef USE_MAXSUPPRESSION
    *out.ptr = compute_strength(p, in.ptr, input_stride_y, threshold);
#else  /* USE_MAXSUPPRESSION */
    *out.ptr = 1;
#endif /* USE_MAXSUPPRESSION */
}

/** Copy result to Keypoint buffer and count number of corners
 *
 * @param[in]  input_ptr                           Pointer to the image with calculated strenghs. Supported data types: U8
 * @param[in]  input_stride_x                      Stride of the first source image in X dimension (in bytes)
 * @param[in]  input_step_x                        input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                      Stride of the first source image in Y dimension (in bytes)
 * @param[in]  input_step_y                        input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes The offset of the first element in the first source image
 * @param[in]  max_num_points                      The maximum number of keypoints the array can hold
 * @param[out] offset                              The number of skipped pixels in x dimension
 * @param[out] num_of_points                       Number of points found
 * @param[out] out                                 The keypoints found
 *
 */
__kernel void copy_to_keypoint(
    IMAGE_DECLARATION(input),
    uint     max_num_points,
    uint     offset,
    __global uint *num_of_points,
    __global Keypoint *out)
{
#ifndef UPDATE_NUMBER
    if(*num_of_points >= max_num_points)
    {
        return;
    }
#endif /* UPDATE_NUMBER */

    Image in = CONVERT_TO_IMAGE_STRUCT(input);

    uchar value = *in.ptr;

    if(value > 0)
    {
        int id = atomic_inc(num_of_points);
        if(id < max_num_points)
        {
            out[id].strength        = value;
            out[id].x               = get_global_id(0) + offset;
            out[id].y               = get_global_id(1) + offset;
            out[id].tracking_status = 1;
        }
    }
}
