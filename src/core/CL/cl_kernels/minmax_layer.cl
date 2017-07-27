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

#if defined(WIDTH) && defined(HEIGHT) && defined(DEPTH)
/** This function identifies the min and maximum value of an input 3D tensor.
 *
 * @note The width, height and depth of the input tensor must be provided at compile time using -DWIDTH, -DHEIGHT and -DDEPTH (e.g. -DWIDTH=320, -DHEIGHT=240, -DDEPTH=3)
 *
 * @param[in] src_ptr                           Pointer to the source tensor. Supported data types: F32
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_stride_z                      Stride of the source image in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] dst_ptr                           Pointer to the min/max vector. Minimum value in position 0, maximum value in position 1. Supported data types: F32.
 * @param[in] dst_stride_x                      Stride of the min/max vector in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the min/max vector
 */
__kernel void minmax_layer(
    TENSOR3D_DECLARATION(src),
    VECTOR_DECLARATION(dst))
{
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
    Vector   dst = CONVERT_TO_VECTOR_STRUCT(dst);

    float4 min_value     = (float4)FLT_MAX;
    float4 max_value     = (float4) - FLT_MAX;
    float2 min_max_value = (float2)(FLT_MAX, -FLT_MAX);

    for(int z = 0; z < DEPTH; ++z)
    {
        for(int y = 0; y < HEIGHT; ++y)
        {
            int             x        = 0;
            __global float *src_addr = (__global float *)(src.ptr + y * src_stride_y + z * src_stride_z);

            for(; x <= (int)(WIDTH - 8); x += 8)
            {
                float8 value = *(src_addr + x);

                min_value = select(value.s0123, min_value, min_value < value.s0123);
                min_value = select(value.s4567, min_value, min_value < value.s4567);

                max_value = select(value.s0123, max_value, max_value > value.s0123);
                max_value = select(value.s4567, max_value, max_value > value.s4567);
            }

            for(; x < WIDTH; ++x)
            {
                float value = *(src_addr + x);

                min_max_value.s0 = min(min_max_value.s0, value);
                min_max_value.s1 = max(min_max_value.s1, value);
            }
        }
    }

    // Perform min/max reduction
    min_value.s01 = min(min_value.s01, min_value.s23);
    min_value.s0  = min(min_value.s0, min_value.s1);
    max_value.s01 = max(max_value.s01, max_value.s23);
    max_value.s0  = max(max_value.s0, max_value.s1);

    min_max_value.s0 = min(min_max_value.s0, min_value.s0);
    min_max_value.s1 = max(min_max_value.s1, max_value.s0);

    if(min_max_value.s0 == min_max_value.s1)
    {
        min_max_value.s0 = 0.0f;
        min_max_value.s1 = 1.0f;
    }

    // Store min and max
    vstore2(min_max_value, 0, (__global float *)dst.ptr);
}
#endif // defined(WIDTH) && defined(HEIGHT) && defined(DEPTH)