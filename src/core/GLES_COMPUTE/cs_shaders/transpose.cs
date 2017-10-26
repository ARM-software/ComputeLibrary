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

#ifdef DATA_TYPE_FP32
precision highp float;

BUFFER_DECLARATION(src, 1, float, readonly);
BUFFER_DECLARATION(dst, 2, float, writeonly);

layout(std140) uniform shader_params
{
    IMAGE_PARAM_DECLARATION(src);
    IMAGE_PARAM_DECLARATION(dst);
};

#define LOAD16(r, name, offset)          \
    r.x = LOAD4(name, offset);           \
    r.y = LOAD4(name, offset + uint(1)); \
    r.z = LOAD4(name, offset + uint(2)); \
    r.w = LOAD4(name, offset + uint(3))

#define STORE16(name, offset, r)         \
    STORE4(name, offset, r.x);           \
    STORE4(name, offset + uint(1), r.y); \
    STORE4(name, offset + uint(2), r.z); \
    STORE4(name, offset + uint(3), r.w)

/** This OpenGL ES kernel computes the matrix transposition of input matrix
 *
 * @param[in]  src_ptr                           Pointer to the source matrix. Supported data types: F32
 * @param[in]  src_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data type: same as src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination matrix
 */
void main(void)
{
    // Compute source address
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Load the NxN block at (x, y)
    vec4 u0;
    vec4 u1;
    vec4 u2;
    vec4 u3;
    LOAD16(u0, src, offset(src, 0, 0));
    LOAD16(u1, src, offset(src, 0, 1));
    LOAD16(u2, src, offset(src, 0, 2));
    LOAD16(u3, src, offset(src, 0, 3));

    // Transpose the block
    vec4 tmp;
    tmp.xyz = u0.yzw;
    u0.y    = u1.x;
    u0.z    = u2.x;
    u0.w    = u3.x;
    u1.x    = tmp.x;
    u2.x    = tmp.y;
    u3.x    = tmp.z;
    tmp.xy  = u1.zw;
    u1.z    = u2.y;
    u1.w    = u3.y;
    u2.y    = tmp.x;
    u3.y    = tmp.y;
    tmp.x   = u2.w;
    u2.w    = u3.z;
    u3.z    = tmp.x;

    // Store the block at (y, x)
    uint dst_offset_in_bytes = uint(16) * uint(gl_GlobalInvocationID.y) + uint(4) * uint(gl_GlobalInvocationID.x) * (dst.stride_y) + (dst.offset_first_element_in_bytes);

    STORE16(dst, uint((dst_offset_in_bytes + uint(0) * dst.stride_y) >> 2), u0);
    STORE16(dst, uint((dst_offset_in_bytes + uint(1) * dst.stride_y) >> 2), u1);
    STORE16(dst, uint((dst_offset_in_bytes + uint(2) * dst.stride_y) >> 2), u2);
    STORE16(dst, uint((dst_offset_in_bytes + uint(3) * dst.stride_y) >> 2), u3);
}

#elif defined(DATA_TYPE_FP16)
precision mediump float;

BUFFER_DECLARATION(src, 1, uvec2, readonly);
BUFFER_DECLARATION(dst, 2, uvec2, writeonly);

layout(std140) uniform shader_params
{
    IMAGE_PARAM_DECLARATION(src);
    IMAGE_PARAM_DECLARATION(dst);
};

/** This OpenGL ES kernel computes the matrix transposition of input matrix
 *
 * @param[in]  src_ptr                           Pointer to the source matrix. Supported data types: F16
 * @param[in]  src_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data type: same as src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination matrix
 */
void main(void)
{
    // Compute source address
    Image src = GC_CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = GC_CONVERT_TO_IMAGE_STRUCT(dst);

    // Load the NxN block at (x, y)
    vec4  u0;
    vec4  u1;
    vec4  u2;
    vec4  u3;
    uvec2 packed_s[4];
    GC_LOAD1_2D_OFFSET(packed_s[0], src, 0, 0);
    GC_LOAD1_2D_OFFSET(packed_s[1], src, 0, 1);
    GC_LOAD1_2D_OFFSET(packed_s[2], src, 0, 2);
    GC_LOAD1_2D_OFFSET(packed_s[3], src, 0, 3);
    u0 = vec4(unpackHalf2x16(packed_s[0].x), unpackHalf2x16(packed_s[0].y));
    u1 = vec4(unpackHalf2x16(packed_s[1].x), unpackHalf2x16(packed_s[1].y));
    u2 = vec4(unpackHalf2x16(packed_s[2].x), unpackHalf2x16(packed_s[2].y));
    u3 = vec4(unpackHalf2x16(packed_s[3].x), unpackHalf2x16(packed_s[3].y));

    // Transpose the block
    vec4 tmp;
    tmp.xyz = u0.yzw;
    u0.y    = u1.x;
    u0.z    = u2.x;
    u0.w    = u3.x;
    u1.x    = tmp.x;
    u2.x    = tmp.y;
    u3.x    = tmp.z;
    tmp.xy  = u1.zw;
    u1.z    = u2.y;
    u1.w    = u3.y;
    u2.y    = tmp.x;
    u3.y    = tmp.y;
    tmp.x   = u2.w;
    u2.w    = u3.z;
    u3.z    = tmp.x;

    // Store the block at (y, x)
    uint dst_offset_in_bytes = uint(8) * uint(gl_GlobalInvocationID.y) + uint(gl_GlobalInvocationID.x) * (dst_step_y) + (dst.offset_first_element_in_bytes);

    packed_s[0] = uvec2(packHalf2x16(u0.xy), packHalf2x16(u0.zw));
    packed_s[1] = uvec2(packHalf2x16(u1.xy), packHalf2x16(u1.zw));
    packed_s[2] = uvec2(packHalf2x16(u2.xy), packHalf2x16(u2.zw));
    packed_s[3] = uvec2(packHalf2x16(u3.xy), packHalf2x16(u3.zw));
    GC_STORE1(packed_s[0], dst, uint((dst_offset_in_bytes + uint(0) * dst_stride_y) >> 3));
    GC_STORE1(packed_s[1], dst, uint((dst_offset_in_bytes + uint(1) * dst_stride_y) >> 3));
    GC_STORE1(packed_s[2], dst, uint((dst_offset_in_bytes + uint(2) * dst_stride_y) >> 3));
    GC_STORE1(packed_s[3], dst, uint((dst_offset_in_bytes + uint(3) * dst_stride_y) >> 3));
}
#endif /*ARM_COMPUTE_ENABLE_FP16*/
