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
#include "types.h"

#ifndef DATA_TYPE_MIN
#define DATA_TYPE_MIN 0x0
#endif /* DATA_TYPE_MIN */

#ifndef DATA_TYPE_MAX
#define DATA_TYPE_MAX 0xFF
#endif /* DATA_TYPE_MAX */

__constant VEC_DATA_TYPE(DATA_TYPE, 16) type_min = (VEC_DATA_TYPE(DATA_TYPE, 16))(DATA_TYPE_MIN);
__constant VEC_DATA_TYPE(DATA_TYPE, 16) type_max = (VEC_DATA_TYPE(DATA_TYPE, 16))(DATA_TYPE_MAX);
__constant uint16 idx16 = (uint16)(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

/** This function identifies the min and maximum value of an input image.
 *
 * @note Input image data type must be passed as a preprocessor argument using -DDATA_TYPE.
 * Moreover, the minimum and maximum value of the given data type must be provided using -DDATA_TYPE_MIN and -DDATA_TYPE_MAX respectively.
 * @note In case image width is not a multiple of 16 then -DNON_MULTIPLE_OF_16 must be passed.
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: U8
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] min_max                           Pointer to buffer with minimum value in position 0 and maximum value in position 1
 * @param[in]  width                             Input image width
 */
__kernel void minmax(
    IMAGE_DECLARATION(src),
    __global int *min_max,
    uint          width)
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);

    // Initialize local minimum and local maximum
    VEC_DATA_TYPE(DATA_TYPE, 16)
    local_min = type_max;
    VEC_DATA_TYPE(DATA_TYPE, 16)
    local_max = type_min;

    // Calculate min/max of row
    uint width4 = width >> 4;
    for(uint i = 0; i < width4; i++)
    {
        VEC_DATA_TYPE(DATA_TYPE, 16)
        data      = vload16(0, (__global DATA_TYPE *)offset(&src, i << 4, 0));
        local_min = min(data, local_min);
        local_max = max(data, local_max);
    }

#ifdef NON_MULTIPLE_OF_16
    // Handle non multiple of 16
    VEC_DATA_TYPE(DATA_TYPE, 16)
    data = vload16(0, (__global DATA_TYPE *)offset(&src, width4 << 4, 0));
    VEC_DATA_TYPE(DATA_TYPE, 16)
    widx      = CONVERT(((uint16)(width4 << 4) + idx16) < width, VEC_DATA_TYPE(DATA_TYPE, 16));
    local_max = max(local_max, select(type_min, data, widx));
    local_min = min(local_min, select(type_max, data, widx));
#endif /* NON_MULTIPLE_OF_16 */

    // Perform min/max reduction
    local_min.s01234567 = min(local_min.s01234567, local_min.s89ABCDEF);
    local_max.s01234567 = max(local_max.s01234567, local_max.s89ABCDEF);

    local_min.s0123 = min(local_min.s0123, local_min.s4567);
    local_max.s0123 = max(local_max.s0123, local_max.s4567);

    local_min.s01 = min(local_min.s01, local_min.s23);
    local_max.s01 = max(local_max.s01, local_max.s23);

    local_min.s0 = min(local_min.s0, local_min.s1);
    local_max.s0 = max(local_max.s0, local_max.s1);

    // Update global min/max
    atomic_min(&min_max[0], local_min.s0);
    atomic_max(&min_max[1], local_max.s0);
}

/** This function counts the min and max occurrences in an image and tags their position.
 *
 * @note -DCOUNT_MIN_MAX should be specified if we want to count the occurrences of the minimum and maximum values.
 * @note -DLOCATE_MIN and/or -DLOCATE_MAX should be specified if we want to store the position of each occurrence on the given array.
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: U8
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]  min_max                           Pointer to buffer with minimum value in position 0 and maximum value in position 1
 * @param[out] min_max_count                     Pointer to buffer with minimum value occurrences in position 0 and maximum value occurrences in position 1
 * @param[out] min_loc                           Array that holds the location of the minimum value occurrences
 * @param[in]  max_min_loc_count                 The maximum number of min value occurrences coordinates the array can hold
 * @param[out] max_loc                           Array that holds the location of the maximum value occurrences
 * @param[in]  max_max_loc_count                 The maximum number of max value occurrences coordinates the array can hold
 */
__kernel void minmaxloc(
    IMAGE_DECLARATION(src),
    __global int *min_max,
    __global uint *min_max_count
#ifdef LOCATE_MIN
    ,
    __global Coordinates2D *min_loc, uint max_min_loc_count
#endif /* LOCATE_MIN */
#ifdef LOCATE_MAX
    ,
    __global Coordinates2D *max_loc, uint max_max_loc_count
#endif /* LOCATE_MAX */
)
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);

    DATA_TYPE value = *((__global DATA_TYPE *)src.ptr);
#ifdef COUNT_MIN_MAX
    if(value == min_max[0])
    {
        uint idx = atomic_inc(&min_max_count[0]);
#ifdef LOCATE_MIN
        if(idx < max_min_loc_count)
        {
            min_loc[idx].x = get_global_id(0);
            min_loc[idx].y = get_global_id(1);
        }
#endif /* LOCATE_MIN */
    }
    if(value == min_max[1])
    {
        uint idx = atomic_inc(&min_max_count[1]);
#ifdef LOCATE_MAX
        if(idx < max_max_loc_count)
        {
            max_loc[idx].x = get_global_id(0);
            max_loc[idx].y = get_global_id(1);
        }
#endif /* LOCATE_MAX */
    }
#endif /* COUNT_MIN_MAX */
}
