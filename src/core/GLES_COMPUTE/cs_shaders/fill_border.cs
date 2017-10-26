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
layout(local_size_x = LOCAL_SIZE_X, local_size_y = LOCAL_SIZE_Y, local_size_z = LOCAL_SIZE_Z) in;
#include "helpers.h"

#if defined(DATA_TYPE_FP32)
#ifdef FILL_IMAGE_BORDERS_REPLICATE
BUFFER_DECLARATION(buf, 1, float, restrict);
layout(std140) uniform shader_params
{
    TENSOR3D_PARAM_DECLARATION(buf);
    uint width;
    uint height;
    int  start_pos_x;
    int  start_pos_y;
};

/** Fill N pixel of the padding edge of a single channel image by replicating the closest valid pixel.
 *
 * @attention  The border size for top, bottom, left, right needs to be passed at the compile time.
 * e.g. BORDER_SIZE_TOP=0 BORDER_SIZE_BOTTOM=2 BORDER_SIZE_LEFT=0 BORDER_SIZE_RIGHT=2
 *
 * @param[in,out] buf_ptr                           Pointer to the source image. Supported data types: F32
 * @param[in]     buf_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]     buf_step_x                        buf_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     buf_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]     buf_step_y                        buf_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     buf_stride_z                      Stride between images if batching images (in bytes)
 * @param[in]     buf_step_z                        buf_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     buf_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]     width                             Width of the valid region of the image
 * @param[in]     height                            Height of the valid region of the image
 * @param[in]     start_pos_x                       X coordinate indicating the start point of the valid region
 * @param[in]     start_pos_y                       Y coordinate indicating the start point of the valid region
 */
void main()
{
    Image buf = CONVERT_TENSOR3D_TO_IMAGE_STRUCT_NO_STEP(buf);

    // Update pointer to point to the starting point of the valid region
    buf.current_offset = uint(int(buf.current_offset) + ((start_pos_y * int(buf_stride_y) + start_pos_x * int(buf_stride_x)) >> 2));

    int total_width = BORDER_SIZE_LEFT + int(width) + BORDER_SIZE_RIGHT;
    int gid0        = int(gl_GlobalInvocationID.x);
    int gidH        = gid0 - total_width;
    int gidW        = gid0 - BORDER_SIZE_LEFT;

    if(gidH >= 0)
    {
        // Handle left border
        float left_val = LOAD4(buf, offset(buf, 0, gidH));
        for(int i = -BORDER_SIZE_LEFT; i < 0; ++i)
        {
            STORE4(buf, offset(buf, i, gidH), left_val);
        }
        // Handle right border
        float right_val = LOAD4(buf, offset(buf, int(width) - 1, gidH));
        for(int i = 0; i < BORDER_SIZE_RIGHT; ++i)
        {
            STORE4(buf, offset(buf, int(width) + i, gidH), right_val);
        }
    }
    else
    {
        // Get value for corners
        int val_idx = gidW;
        if(gidW < 0 || gidW > (int(width) - 1))
        {
            val_idx = gidW < 0 ? 0 : int(width) - 1;
        }

        // Handle top border
        float top_val = LOAD4(buf, offset(buf, val_idx, 0));
        for(int i = -BORDER_SIZE_TOP; i < 0; ++i)
        {
            STORE4(buf, offset(buf, gidW, i), top_val);
        }
        // Handle bottom border
        float bottom_val = LOAD4(buf, offset(buf, val_idx, int(height) - 1));
        for(int i = 0; i < BORDER_SIZE_BOTTOM; ++i)
        {
            STORE4(buf, offset(buf, gidW, int(height) + i), bottom_val);
        }
    }
}
#endif /* FILL_IMAGE_BORDERS_REPLICATE */

#ifdef FILL_IMAGE_BORDERS_CONSTANT
BUFFER_DECLARATION(buf, 1, float, writeonly);
layout(std140) uniform shader_params
{
    TENSOR3D_PARAM_DECLARATION(buf);
    uint  width;
    uint  height;
    int   start_pos_x;
    int   start_pos_y;
    float constant_value;
};

/** Fill N pixels of the padding edge of a single channel image with a constant value.
 *
 * @attention  The border size for top, bottom, left, right needs to be passed at the compile time.
 * e.g. BORDER_SIZE_TOP=0 BORDER_SIZE_BOTTOM=2 BORDER_SIZE_LEFT=0 BORDER_SIZE_RIGHT=2
 *
 * @param[out] buf_ptr                           Pointer to the source image. Supported data types: F32
 * @param[in]  buf_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  buf_step_x                        buf_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  buf_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  buf_step_y                        buf_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  buf_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]  width                             Width of the valid region of the image
 * @param[in]  height                            Height of the valid region of the image
 * @param[in]  start_pos_x                       X coordinate indicating the start point of the valid region
 * @param[in]  start_pos_y                       Y coordinate indicating the start point of the valid region
 * @param[in]  constant_value                    Constant value to use to fill the edges
 */
void main()
{
    Image buf = CONVERT_TENSOR3D_TO_IMAGE_STRUCT_NO_STEP(buf);

    // Update pointer to point to the starting point of the valid region
    buf.current_offset = uint(int(buf.current_offset) + ((start_pos_y * int(buf_stride_y) + start_pos_x * int(buf_stride_x)) >> 2));

    int total_width = BORDER_SIZE_LEFT + int(width) + BORDER_SIZE_RIGHT;
    int gid0        = int(gl_GlobalInvocationID.x);
    int gidH        = gid0 - total_width;
    int gidW        = gid0 - BORDER_SIZE_LEFT;

    if(gidH >= 0)
    {
        // Handle left border
        for(int i = -BORDER_SIZE_LEFT; i < 0; ++i)
        {
            STORE1(buf, offset(buf, i, gidH), constant_value);
        }
        // Handle right border
        for(int i = 0; i < BORDER_SIZE_RIGHT; ++i)
        {
            STORE1(buf, offset(buf, int(width) + i, gidH), constant_value);
        }
    }
    else
    {
        // Handle top border
        for(int i = -BORDER_SIZE_TOP; i < 0; ++i)
        {
            STORE1(buf, offset(buf, gidW, i), constant_value);
        }
        // Handle bottom border
        for(int i = 0; i < BORDER_SIZE_BOTTOM; ++i)
        {
            STORE1(buf, offset(buf, gidW, int(height) + i), constant_value);
        }
    }
}
#endif /* FILL_IMAGE_BORDERS_CONSTANT */

#elif defined(DATA_TYPE_FP16)
precision mediump float;

#ifdef FILL_IMAGE_BORDERS_REPLICATE
BUFFER_DECLARATION(buf, 1, uint, restrict);
layout(std140) uniform shader_params
{
    TENSOR3D_PARAM_DECLARATION(buf);
    uint width;
    uint height;
    int  start_pos_x;
    int  start_pos_y;
};

void set_replicate(uint offset, int pos, uint replicate_value)
{
    uint packed_b;
    LOAD1(packed_b, buf, offset);

    vec2 b = unpackHalf2x16(packed_b);
    vec2 c = unpackHalf2x16(replicate_value);

    if(pos % 2 == 0)
    {
        b.x = c.y;
    }
    else
    {
        b.y = c.x;
    }

    packed_b = packHalf2x16(b);

    STORE1(buf, offset, packed_b);
}

/** Fill N pixel of the padding edge of a single channel image by replicating the closest valid pixel.
 *
 * @attention  The border size for top, bottom, left, right needs to be passed at the compile time.
 * e.g. BORDER_SIZE_TOP=0 BORDER_SIZE_BOTTOM=2 BORDER_SIZE_LEFT=0 BORDER_SIZE_RIGHT=2
 *
 * @param[in,out] buf_ptr                           Pointer to the source image. Supported data types: F16
 * @param[in]     buf_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]     buf_step_x                        buf_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     buf_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]     buf_step_y                        buf_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     buf_stride_z                      Stride between images if batching images (in bytes)
 * @param[in]     buf_step_z                        buf_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]     buf_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]     width                             Width of the valid region of the image
 * @param[in]     height                            Height of the valid region of the image
 * @param[in]     start_pos_x                       X coordinate indicating the start point of the valid region
 * @param[in]     start_pos_y                       Y coordinate indicating the start point of the valid region
 */
void main()
{
    Image buf = CONVERT_TENSOR3D_TO_IMAGE_STRUCT_NO_STEP_FP16(buf);

    // Update pointer to point to the starting point of the valid region
    buf.current_offset = uint(buf.current_offset + uint(start_pos_y) * buf_stride_y + uint(start_pos_x) * buf_stride_x);

    int total_width = BORDER_SIZE_LEFT + int(width) + BORDER_SIZE_RIGHT;
    int gid0        = int(gl_GlobalInvocationID.x);
    int gidH        = gid0 - total_width;
    int gidW        = gid0 - BORDER_SIZE_LEFT;

    if(gidH >= 0)
    {
        // Handle left border
        uint left_val;
        LOAD1(left_val, buf, offset_fp16(buf, 0, gidH) >> uint(2));
        for(int i = -BORDER_SIZE_LEFT; i < 0; ++i)
        {
            uint offset = offset_fp16(buf, i, gidH) >> 2;
            int  pos    = i + BORDER_SIZE_LEFT;
            if(i == -1)
            {
                if(pos % 2 == 0)
                {
                    set_replicate(offset, pos, left_val);
                }
            }
            else
            {
                if(pos % 2 == 0)
                {
                    vec2 a = unpackHalf2x16(left_val);
                    uint b = packHalf2x16(a.xx);
                    STORE1(buf, offset, b);
                }
            }
        }
        // Handle right border
        uint right_val;
        LOAD1(right_val, buf, offset_fp16(buf, int(width) - 1, gidH) >> uint(2));
        for(int i = 0; i < BORDER_SIZE_RIGHT; ++i)
        {
            uint offset = offset_fp16(buf, int(width) + i, gidH) >> 2;
            int  pos    = i + BORDER_SIZE_LEFT + int(width);

            if(i == 0)
            {
                if(pos % 2 == 0)
                {
                    vec2 a = unpackHalf2x16(right_val);
                    uint b = packHalf2x16(a.yy);
                    STORE1(buf, offset, b);
                }
                else
                {
                    set_replicate(offset, pos, right_val);
                }
            }
            else
            {
                if(pos % 2 == 0)
                {
                    vec2 a = unpackHalf2x16(right_val);
                    uint b = packHalf2x16(a.yy);
                    STORE1(buf, offset, b);
                }
            }
        }
    }
    else
    {
        // Get value for corners
        int val_idx = gidW;
        if(gidW < 0 || (gidW > (int(width) - 1)))
        {
            val_idx = gidW < 0 ? 0 : (int(width) - 1);
        }

        // Handle top border
        uint top_val;
        LOAD1(top_val, buf, offset_fp16(buf, val_idx, 0) >> uint(2));
        for(int i = -BORDER_SIZE_TOP; i < 0; ++i)
        {
            uint offset = offset_fp16(buf, gidW, i) >> 2;

            if(gid0 % 2 == 0)
            {
                if(gidW == (int(width) - 1))
                {
                    vec2 a = unpackHalf2x16(top_val);
                    uint b = packHalf2x16(a.xx);
                    STORE1(buf, offset, b);
                }
                else
                {
                    if(gidW < 0)
                    {
                        vec2 a = unpackHalf2x16(top_val);
                        uint b;
                        if(BORDER_SIZE_LEFT % 2 == 0)
                        {
                            b = packHalf2x16(a.xx);
                        }
                        else
                        {
                            b = packHalf2x16(a.yy);
                        }
                        STORE1(buf, offset, b);
                    }
                    else if(gidW >= int(width))
                    {
                        vec2 a = unpackHalf2x16(top_val);
                        uint b;
                        if((BORDER_SIZE_LEFT + int(width)) % 2 == 0)
                        {
                            b = packHalf2x16(a.yy);
                        }
                        STORE1(buf, offset, b);
                    }
                    else
                    {
                        STORE1(buf, offset, top_val);
                    }
                }
            }
        }
        // Handle bottom border
        uint bottom_val;
        LOAD1(bottom_val, buf, offset_fp16(buf, val_idx, int(height) - 1) >> uint(2));
        for(int i = 0; i < BORDER_SIZE_BOTTOM; ++i)
        {
            uint offset = offset_fp16(buf, gidW, int(height) + i) >> 2;

            if(gid0 % 2 == 0)
            {
                if(gidW == (int(width) - 1))
                {
                    vec2 a = unpackHalf2x16(bottom_val);
                    uint b = packHalf2x16(a.xx);
                    STORE1(buf, offset, b);
                }
                else
                {
                    if(gidW < 0)
                    {
                        vec2 a = unpackHalf2x16(bottom_val);
                        uint b;
                        if(BORDER_SIZE_LEFT % 2 == 0)
                        {
                            b = packHalf2x16(a.xx);
                        }
                        else
                        {
                            b = packHalf2x16(a.yy);
                        }
                        STORE1(buf, offset, b);
                    }
                    else if(gidW >= int(width))
                    {
                        vec2 a = unpackHalf2x16(bottom_val);
                        uint b;
                        if((BORDER_SIZE_LEFT + int(width)) % 2 == 0)
                        {
                            b = packHalf2x16(a.yy);
                        }
                        STORE1(buf, offset, b);
                    }
                    else
                    {
                        STORE1(buf, offset, bottom_val);
                    }
                }
            }
        }
    }
}
#endif /* FILL_IMAGE_BORDERS_REPLICATE */

#ifdef FILL_IMAGE_BORDERS_CONSTANT
BUFFER_DECLARATION(buf, 1, uint, restrict);

layout(std140) uniform shader_params
{
    TENSOR3D_PARAM_DECLARATION(buf);
    uint  width;
    uint  height;
    int   start_pos_x;
    int   start_pos_y;
    float constant_value;
};

void set_constant(uint offset, int pos)
{
    uint packed_b;
    LOAD1(packed_b, buf, offset);

    vec2 b = unpackHalf2x16(packed_b);

    if(pos % 2 == 0)
    {
        b.x = constant_value;
    }
    else
    {
        b.y = constant_value;
    }

    packed_b = packHalf2x16(b);

    STORE1(buf, offset, packed_b);
}

/** Fill N pixels of the padding edge of a single channel image with a constant value.
 *
 * @attention  The border size for top, bottom, left, right needs to be passed at the compile time.
 * e.g. BORDER_SIZE_TOP=0 BORDER_SIZE_BOTTOM=2 BORDER_SIZE_LEFT=0 BORDER_SIZE_RIGHT=2
 *
 * @param[out] buf_ptr                           Pointer to the source image. Supported data types: F16
 * @param[in]  buf_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  buf_step_x                        buf_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  buf_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  buf_step_y                        buf_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  buf_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]  width                             Width of the valid region of the image
 * @param[in]  height                            Height of the valid region of the image
 * @param[in]  start_pos_x                       X coordinate indicating the start point of the valid region
 * @param[in]  start_pos_y                       Y coordinate indicating the start point of the valid region
 * @param[in]  constant_value                    Constant value to use to fill the edges
 */
void main()
{
    Image buf = CONVERT_TENSOR3D_TO_IMAGE_STRUCT_NO_STEP_FP16(buf);

    int total_width = BORDER_SIZE_LEFT + int(width) + BORDER_SIZE_RIGHT;
    int gid0        = int(gl_GlobalInvocationID.x);
    int gidH        = gid0 - total_width;
    int gidW        = gid0 - BORDER_SIZE_LEFT;

    // Update pointer to point to the starting point of the valid region
    buf.current_offset = uint(int(buf.current_offset) + ((start_pos_y * int(buf_stride_y) + start_pos_x * int(buf_stride_x))));

    vec2 b = vec2(constant_value, constant_value);

    uint packed_b = packHalf2x16(b);

    if(gidH >= 0)
    {
        // Handle left border
        for(int i = -BORDER_SIZE_LEFT; i < 0; ++i)
        {
            uint offset = offset_fp16(buf, i, gidH) >> 2;
            int  pos    = i + BORDER_SIZE_LEFT;

            if(i == -1)
            {
                if(pos % 2 == 0)
                {
                    set_constant(offset, pos);
                }
            }
            else
            {
                if(pos % 2 == 0)
                {
                    STORE1(buf, offset, packed_b);
                }
            }
        }
        // Handle right border
        for(int i = 0; i < BORDER_SIZE_RIGHT; ++i)
        {
            uint offset = offset_fp16(buf, int(width) + i, gidH) >> 2;
            int  pos    = i + BORDER_SIZE_LEFT + int(width);

            if(i == 0)
            {
                if(pos % 2 == 0)
                {
                    STORE1(buf, offset, packed_b);
                }
                else
                {
                    set_constant(offset, pos);
                }
            }
            else
            {
                if(pos % 2 == 0)
                {
                    STORE1(buf, offset, packed_b);
                }
            }
        }
    }
    else
    {
        // Handle top border
        for(int i = -BORDER_SIZE_TOP; i < 0; ++i)
        {
            uint offset = offset_fp16(buf, gidW, i) >> 2;

            if(gid0 % 2 == 0)
            {
                STORE1(buf, offset, packed_b);
            }
        }
        // Handle bottom border
        for(int i = 0; i < BORDER_SIZE_BOTTOM; ++i)
        {
            uint offset = offset_fp16(buf, gidW, int(height) + i) >> 2;

            if(gid0 % 2 == 0)
            {
                STORE1(buf, offset, packed_b);
            }
        }
    }
}
#endif /* FILL_IMAGE_BORDERS_CONSTANT */
#endif /* DATA_TYPE_FP32 */
