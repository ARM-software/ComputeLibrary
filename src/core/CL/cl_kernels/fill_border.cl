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

#if defined(FIXED_POINT_POSITION)
#include "fixed_point.h"
#endif /* FIXED_POINT_POSITION */

/** Fill N pixel of the padding edge of a single channel image by replicating the closest valid pixel.
 *
 * @attention  The DATA_TYPE needs to be passed at the compile time.
 * e.g. -DDATA_TYPE=int
 *
 * @attention  The border size for top, bottom, left, right needs to be passed at the compile time.
 * e.g. --DBORDER_SIZE_TOP=0 -DBORDER_SIZE_BOTTOM=2 -DBORDER_SIZE_LEFT=0 -DBORDER_SIZE_RIGHT=2
 *
 * @param[in,out] buf_ptr                           Pointer to the source image. Supported data types: U8, U16, S16, U32, S32, F32
 * @param[in]     buf_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]     buf_step_x                        buf_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     buf_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]     buf_step_y                        buf_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     buf_stride_z                      Stride between images if batching images (in bytes)
 * @param[in]     buf_step_z                        buf_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     buf_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]     width                             Width of the valid region of the image
 * @param[in]     height                            Height of the valid region of the image
 * @param[in]     start_pos                         XY coordinate indicating the start point of the valid region
 */
__kernel void fill_image_borders_replicate(
    TENSOR3D_DECLARATION(buf),
    uint width,
    uint height,
    int2 start_pos)
{
    Image buf = CONVERT_TENSOR3D_TO_IMAGE_STRUCT_NO_STEP(buf);

    // Update pointer to point to the starting point of the valid region
    buf.ptr += start_pos.y * buf.stride_y + start_pos.x * buf.stride_x;

    const int total_width = BORDER_SIZE_LEFT + width + BORDER_SIZE_RIGHT;
    const int gid0        = get_global_id(0);
    const int gidH        = gid0 - total_width;
    const int gidW        = gid0 - BORDER_SIZE_LEFT;

    if(gidH >= 0)
    {
        // Handle left border
        DATA_TYPE left_val = *(__global DATA_TYPE *)offset(&buf, 0, gidH);
        for(int i = -BORDER_SIZE_LEFT; i < 0; ++i)
        {
            *(__global DATA_TYPE *)offset(&buf, i, gidH) = left_val;
        }
        // Handle right border
        DATA_TYPE right_val = *(__global DATA_TYPE *)offset(&buf, width - 1, gidH);
        for(int i = 0; i < BORDER_SIZE_RIGHT; ++i)
        {
            *(__global DATA_TYPE *)offset(&buf, width + i, gidH) = right_val;
        }
    }
    else
    {
        // Get value for corners
        int val_idx = gidW;
        if(gidW < 0 || gidW > (width - 1))
        {
            val_idx = gidW < 0 ? 0 : width - 1;
        }

        // Handle top border
        DATA_TYPE top_val = *(__global DATA_TYPE *)offset(&buf, val_idx, 0);
        for(int i = -BORDER_SIZE_TOP; i < 0; ++i)
        {
            *(__global DATA_TYPE *)offset(&buf, gidW, i) = top_val;
        }
        // Handle bottom border
        DATA_TYPE bottom_val = *(__global DATA_TYPE *)offset(&buf, val_idx, height - 1);
        for(int i = 0; i < BORDER_SIZE_BOTTOM; ++i)
        {
            *(__global DATA_TYPE *)offset(&buf, gidW, height + i) = bottom_val;
        }
    }
}

/** Fill N pixels of the padding edge of a single channel image with a constant value.
 *
 * @attention  The DATA_TYPE needs to be passed at the compile time.
 * e.g. -DDATA_TYPE=int
 *
 * @attention  The border size for top, bottom, left, right needs to be passed at the compile time.
 * e.g. --DBORDER_SIZE_TOP=0 -DBORDER_SIZE_BOTTOM=2 -DBORDER_SIZE_LEFT=0 -DBORDER_SIZE_RIGHT=2
 *
 * @param[out] buf_ptr                           Pointer to the source image. Supported data types: U8, U16, S16, U32, S32, F32
 * @param[in]  buf_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  buf_step_x                        buf_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  buf_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  buf_step_y                        buf_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  buf_stride_z                      Stride between images if batching images (in bytes)
 * @param[in]  buf_step_z                        buf_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  buf_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]  width                             Width of the valid region of the image
 * @param[in]  height                            Height of the valid region of the image
 * @param[in]  start_pos                         XY coordinate indicating the start point of the valid region
 * @param[in]  constant_value                    Constant value to use to fill the edges
 */
__kernel void fill_image_borders_constant(
    TENSOR3D_DECLARATION(buf),
    uint      width,
    uint      height,
    int2      start_pos,
    DATA_TYPE constant_value)
{
    Image buf = CONVERT_TENSOR3D_TO_IMAGE_STRUCT_NO_STEP(buf);

    // Update pointer to point to the starting point of the valid region
    buf.ptr += start_pos.y * buf.stride_y + start_pos.x * buf.stride_x;

    const int total_width = BORDER_SIZE_LEFT + width + BORDER_SIZE_RIGHT;
    const int gid0        = get_global_id(0);
    const int gidH        = gid0 - total_width;
    const int gidW        = gid0 - BORDER_SIZE_LEFT;

    if(gidH >= 0)
    {
        // Handle left border
        for(int i = -BORDER_SIZE_LEFT; i < 0; ++i)
        {
            *(__global DATA_TYPE *)offset(&buf, i, gidH) = constant_value;
        }
        // Handle right border
        for(int i = 0; i < BORDER_SIZE_RIGHT; ++i)
        {
            *(__global DATA_TYPE *)offset(&buf, width + i, gidH) = constant_value;
        }
    }
    else
    {
        // Handle top border
        for(int i = -BORDER_SIZE_TOP; i < 0; ++i)
        {
            *(__global DATA_TYPE *)offset(&buf, gidW, i) = constant_value;
        }
        // Handle bottom border
        for(int i = 0; i < BORDER_SIZE_BOTTOM; ++i)
        {
            *(__global DATA_TYPE *)offset(&buf, gidW, height + i) = constant_value;
        }
    }
}
