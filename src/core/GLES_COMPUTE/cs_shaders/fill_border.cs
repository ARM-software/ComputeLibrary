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

#include "helpers_cs.h"

#if defined(DATA_TYPE_FP16)
precision mediump float;
#endif // DATA_TYPE_FP16

#ifdef FILL_IMAGE_BORDERS_REPLICATE

/** Fill N pixel of the padding edge of a single channel image by replicating the closest valid pixel.
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_NAME". e.g. "#define DATA_TYPE_FP32"
 * @attention  The border size for top, bottom, left, right needs to be passed at the compile time.
 * e.g. BORDER_SIZE_TOP=0 BORDER_SIZE_BOTTOM=2 BORDER_SIZE_LEFT=0 BORDER_SIZE_RIGHT=2
 *
 * @param[in,out] buf_ptr     Pointer to the source image. Supported data types: F16/F32
 * @param[in]     buf_attrs   The attributes of the source image
 * @param[in]     width       Width of the valid region of the image
 * @param[in]     height      Height of the valid region of the image
 * @param[in]     start_pos_x X coordinate indicating the start point of the valid region
 * @param[in]     start_pos_y Y coordinate indicating the start point of the valid region
 */
SHADER_PARAMS_DECLARATION
{
    Tensor3DAttributes buf_attrs;
    uint               width;
    uint               height;
    int                start_pos_x;
    int                start_pos_y;
};

#if defined(DATA_TYPE_FP32)

TENSOR_DECLARATION(1, bufBuffer, float, buf_ptr, buf_shift, 2, restrict);

void main()
{
    ImageIterator buf_iter = CONVERT_TENSOR3D_TO_IMAGE_ITERATOR_NO_STEP(buf_attrs, buf_shift);

    // Update pointer to point to the starting point of the valid region
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(buf_iter, start_pos_y * int(buf_attrs.stride_y) + start_pos_x * int(buf_attrs.stride_x));

    int total_width = BORDER_SIZE_LEFT + int(width) + BORDER_SIZE_RIGHT;
    int gid0        = int(gl_GlobalInvocationID.x);
    int gidH        = gid0 - total_width;
    int gidW        = gid0 - BORDER_SIZE_LEFT;

    if(gidH >= 0)
    {
        // Handle left border
        float left_val = LOAD(buf_ptr, IMAGE_OFFSET(buf_iter, 0, gidH));
        for(int i = 0; i < BORDER_SIZE_LEFT; ++i)
        {
            STORE(buf_ptr, IMAGE_OFFSET(buf_iter, -(i + 1), gidH), left_val);
        }
        // Handle right border
        float right_val = LOAD(buf_ptr, IMAGE_OFFSET(buf_iter, int(width) - 1, gidH));
        for(int i = 0; i < BORDER_SIZE_RIGHT; ++i)
        {
            STORE(buf_ptr, IMAGE_OFFSET(buf_iter, int(width) + i, gidH), right_val);
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
        float top_val = LOAD(buf_ptr, IMAGE_OFFSET(buf_iter, val_idx, 0));
        for(int i = 0; i < BORDER_SIZE_TOP; ++i)
        {
            STORE(buf_ptr, IMAGE_OFFSET(buf_iter, gidW, -(i + 1)), top_val);
        }
        // Handle bottom border
        float bottom_val = LOAD(buf_ptr, IMAGE_OFFSET(buf_iter, val_idx, int(height) - 1));
        for(int i = 0; i < BORDER_SIZE_BOTTOM; ++i)
        {
            STORE(buf_ptr, IMAGE_OFFSET(buf_iter, gidW, int(height) + i), bottom_val);
        }
    }
}
#elif defined(DATA_TYPE_FP16)

TENSOR_DECLARATION(1, bufBuffer, uint, buf_ptr, buf_shift, 2, restrict);

void set_replicate(uint offset, int pos, vec2 replicate_value)
{
    vec2 b = LOAD_UNPACK2_HALF(buf_ptr, offset);

    if(pos % 2 == 0)
    {
        b.x = replicate_value.y;
    }
    else
    {
        b.y = replicate_value.x;
    }

    STORE_PACK2_HALF(buf_ptr, offset, b);
}

void main()
{
    ImageIterator buf_iter = CONVERT_TENSOR3D_TO_IMAGE_ITERATOR_NO_STEP(buf_attrs, buf_shift);

    // Update pointer to point to the starting point of the valid region
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(buf_iter, start_pos_y * int(buf_attrs.stride_y) + start_pos_x * int(buf_attrs.stride_x));

    int total_width = BORDER_SIZE_LEFT + int(width) + BORDER_SIZE_RIGHT;
    int gid0        = int(gl_GlobalInvocationID.x);
    int gidH        = gid0 - total_width;
    int gidW        = gid0 - BORDER_SIZE_LEFT;

    if(gidH >= 0)
    {
        // Handle left border
        vec2 left_val = LOAD_UNPACK2_HALF(buf_ptr, IMAGE_OFFSET(buf_iter, 0, gidH));
        for(int i = 0; i < BORDER_SIZE_LEFT; ++i)
        {
            uint offset = IMAGE_OFFSET(buf_iter, -(i + 1), gidH);
            int  pos    = BORDER_SIZE_LEFT - i - 1;
            if(i == 0)
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
                    if(BORDER_SIZE_LEFT % 2 == 0)
                    {
                        STORE_PACK2_HALF(buf_ptr, offset, left_val.xx);
                    }
                    else
                    {
                        STORE_PACK2_HALF(buf_ptr, offset, left_val.yy);
                    }
                    i++;
                }
            }
        }
        // Handle right border
        vec2 right_val_origin = LOAD_UNPACK2_HALF(buf_ptr, IMAGE_OFFSET(buf_iter, int(width) - 1, gidH));
        vec2 right_val;
        if((((BORDER_SIZE_LEFT + int(width)) % 2)) == 1)
        {
            right_val = vec2(right_val_origin.x, right_val_origin.x);
        }
        else
        {
            right_val = vec2(right_val_origin.y, right_val_origin.y);
        }
        for(int i = 0; i < BORDER_SIZE_RIGHT; ++i)
        {
            uint offset = IMAGE_OFFSET(buf_iter, int(width) + i, gidH);
            int  pos    = i + BORDER_SIZE_LEFT + int(width);

            if(i == 0)
            {
                if(pos % 2 == 0)
                {
                    STORE_PACK2_HALF(buf_ptr, offset, right_val);
                    i++;
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
                    STORE_PACK2_HALF(buf_ptr, offset, right_val);
                    i++;
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
        vec2 top_val = LOAD_UNPACK2_HALF(buf_ptr, IMAGE_OFFSET(buf_iter, val_idx, 0));
        for(int i = 0; i < BORDER_SIZE_TOP; ++i)
        {
            uint offset = IMAGE_OFFSET(buf_iter, gidW, -(i + 1));

            if(gid0 % 2 == 0)
            {
                if(gidW == (int(width) - 1))
                {
                    if(((BORDER_SIZE_LEFT + int(width)) % 2 == 1))
                    {
                        STORE_PACK2_HALF(buf_ptr, offset, top_val.xx);
                    }
                    else
                    {
                        STORE_PACK2_HALF(buf_ptr, offset, top_val.yy);
                    }
                }
                else
                {
                    if(gidW < 0)
                    {
                        if(BORDER_SIZE_LEFT % 2 == 0)
                        {
                            STORE_PACK2_HALF(buf_ptr, offset, top_val.xx);
                        }
                        else
                        {
                            STORE_PACK2_HALF(buf_ptr, offset, top_val.yy);
                        }
                    }
                    else if(gidW >= int(width))
                    {
                        if((BORDER_SIZE_LEFT + int(width)) % 2 == 0)
                        {
                            STORE_PACK2_HALF(buf_ptr, offset, top_val.yy);
                        }
                        else
                        {
                            STORE_PACK2_HALF(buf_ptr, offset, top_val.xx);
                        }
                    }
                    else
                    {
                        STORE_PACK2_HALF(buf_ptr, offset, top_val);
                    }
                }
            }
        }
        // Handle bottom border
        vec2 bottom_val = LOAD_UNPACK2_HALF(buf_ptr, IMAGE_OFFSET(buf_iter, val_idx, int(height) - 1));
        for(int i = 0; i < BORDER_SIZE_BOTTOM; ++i)
        {
            uint offset = IMAGE_OFFSET(buf_iter, gidW, int(height) + i);

            if(gid0 % 2 == 0)
            {
                if(gidW == (int(width) - 1))
                {
                    STORE_PACK2_HALF(buf_ptr, offset, bottom_val.xx);
                }
                else
                {
                    if(gidW < 0)
                    {
                        if(BORDER_SIZE_LEFT % 2 == 0)
                        {
                            STORE_PACK2_HALF(buf_ptr, offset, bottom_val.xx);
                        }
                        else
                        {
                            STORE_PACK2_HALF(buf_ptr, offset, bottom_val.yy);
                        }
                    }
                    else if(gidW >= int(width))
                    {
                        if((BORDER_SIZE_LEFT + int(width)) % 2 == 0)
                        {
                            STORE_PACK2_HALF(buf_ptr, offset, bottom_val.yy);
                        }
                        else
                        {
                            STORE_PACK2_HALF(buf_ptr, offset, bottom_val.xx);
                        }
                    }
                    else
                    {
                        STORE_PACK2_HALF(buf_ptr, offset, bottom_val);
                    }
                }
            }
        }
    }
}

#endif /* DATA_TYPE_FP32 */

#endif /* FILL_IMAGE_BORDERS_REPLICATE */

#ifdef FILL_IMAGE_BORDERS_CONSTANT

/** Fill N pixels of the padding edge of a single channel image with a constant value.
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_NAME". e.g. "#define DATA_TYPE_FP32"
 * @attention  The border size for top, bottom, left, right needs to be passed at the compile time.
 * e.g. BORDER_SIZE_TOP=0 BORDER_SIZE_BOTTOM=2 BORDER_SIZE_LEFT=0 BORDER_SIZE_RIGHT=2
 *
 * @param[out] buf_ptr        Pointer to the source image. Supported data types: F16/F32
 * @param[in]  buf_attrs      The attributes of the source image
 * @param[in]  width          Width of the valid region of the image
 * @param[in]  height         Height of the valid region of the image
 * @param[in]  start_pos_x    X coordinate indicating the start point of the valid region
 * @param[in]  start_pos_y    Y coordinate indicating the start point of the valid region
 * @param[in]  constant_value Constant value to use to fill the edges
 */
SHADER_PARAMS_DECLARATION
{
    Tensor3DAttributes buf_attrs;
    uint               width;
    uint               height;
    int                start_pos_x;
    int                start_pos_y;
    float              constant_value;
};

#if defined(DATA_TYPE_FP32)
TENSOR_DECLARATION(1, bufBuffer, float, buf_ptr, buf_shift, 2, writeonly);

void main()
{
    ImageIterator buf_iter = CONVERT_TENSOR3D_TO_IMAGE_ITERATOR_NO_STEP(buf_attrs, buf_shift);

    // Update pointer to point to the starting point of the valid region
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(buf_iter, start_pos_y * int(buf_attrs.stride_y) + start_pos_x * int(buf_attrs.stride_x));

    int total_width = BORDER_SIZE_LEFT + int(width) + BORDER_SIZE_RIGHT;
    int gid0        = int(gl_GlobalInvocationID.x);
    int gidH        = gid0 - total_width;
    int gidW        = gid0 - BORDER_SIZE_LEFT;

    if(gidH >= 0)
    {
        // Handle left border
        for(int i = 0; i < BORDER_SIZE_LEFT; ++i)
        {
            STORE(buf_ptr, IMAGE_OFFSET(buf_iter, -(i + 1), gidH), constant_value);
        }
        // Handle right border
        for(int i = 0; i < BORDER_SIZE_RIGHT; ++i)
        {
            STORE(buf_ptr, IMAGE_OFFSET(buf_iter, int(width) + i, gidH), constant_value);
        }
    }
    else
    {
        // Handle top border
        for(int i = 0; i < BORDER_SIZE_TOP; ++i)
        {
            STORE(buf_ptr, IMAGE_OFFSET(buf_iter, gidW, -(i + 1)), constant_value);
        }
        // Handle bottom border
        for(int i = 0; i < BORDER_SIZE_BOTTOM; ++i)
        {
            STORE(buf_ptr, IMAGE_OFFSET(buf_iter, gidW, int(height) + i), constant_value);
        }
    }
}

#elif defined(DATA_TYPE_FP16)
TENSOR_DECLARATION(1, bufBuffer, uint, buf_ptr, buf_shift, 2, restrict);

void set_constant(uint offset, int pos)
{
    vec2 b = LOAD_UNPACK2_HALF(buf_ptr, offset);

    if(pos % 2 == 0)
    {
        b.x = constant_value;
    }
    else
    {
        b.y = constant_value;
    }

    STORE_PACK2_HALF(buf_ptr, offset, b);
}

void main()
{
    ImageIterator buf_iter = CONVERT_TENSOR3D_TO_IMAGE_ITERATOR_NO_STEP(buf_attrs, buf_shift);

    int total_width = BORDER_SIZE_LEFT + int(width) + BORDER_SIZE_RIGHT;
    int gid0        = int(gl_GlobalInvocationID.x);
    int gidH        = gid0 - total_width;
    int gidW        = gid0 - BORDER_SIZE_LEFT;

    // Update pointer to point to the starting point of the valid region
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(buf_iter, start_pos_y * int(buf_attrs.stride_y) + start_pos_x * int(buf_attrs.stride_x));

    vec2 b = vec2(constant_value, constant_value);

    if(gidH >= 0)
    {
        // Handle left border
        for(int i = 0; i < BORDER_SIZE_LEFT; ++i)
        {
            uint offset = IMAGE_OFFSET(buf_iter, -(i + 1), gidH);
            int  pos    = BORDER_SIZE_LEFT - i - 1;

            if(i == 0)
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
                    STORE_PACK2_HALF(buf_ptr, offset, b);
                }
            }
        }
        // Handle right border
        for(int i = 0; i < BORDER_SIZE_RIGHT; ++i)
        {
            uint offset = IMAGE_OFFSET(buf_iter, int(width) + i, gidH);
            int  pos    = i + BORDER_SIZE_LEFT + int(width);

            if(i == 0)
            {
                if(pos % 2 == 0)
                {
                    STORE_PACK2_HALF(buf_ptr, offset, b);
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
                    STORE_PACK2_HALF(buf_ptr, offset, b);
                }
            }
        }
    }
    else
    {
        // Handle top border
        for(int i = 0; i < BORDER_SIZE_TOP; ++i)
        {
            uint offset = IMAGE_OFFSET(buf_iter, gidW, -(i + 1));

            if(gid0 % 2 == 0)
            {
                STORE_PACK2_HALF(buf_ptr, offset, b);
            }
        }
        // Handle bottom border
        for(int i = 0; i < BORDER_SIZE_BOTTOM; ++i)
        {
            uint offset = IMAGE_OFFSET(buf_iter, gidW, int(height) + i);

            if(gid0 % 2 == 0)
            {
                STORE_PACK2_HALF(buf_ptr, offset, b);
            }
        }
    }
}

#endif /* DATA_TYPE_FP32 */

#endif /* FILL_IMAGE_BORDERS_CONSTANT */
