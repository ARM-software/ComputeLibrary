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

/** This kernel performs a direct convolution to convolve the low three dimensions.
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_NAME". e.g. "#define DATA_TYPE_FP32"
 * @note This kernel has multiple optimized direct convolution options for FP16.
 *       The direct convolution option must be passed at compile time using "#define PROCESS_nX_nY_nZ" e.g. "#define PROCESS_8X_1Y_1Z"
 * @note The convolution stride x must be passed at compile time using "#define STRIDE_X n" e.g. "#define STRIDE_X 1"
 *       This OpenGL ES shader works with stride_x = 1 and 2
 * @note In case biases will be added to the convolution "#define HAS_BIAS" has to be passed to append the final matrix with 1 in each row.
 *
 * @param[in]  src_ptr          Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  src_attrs        The attributes of the source tensor
 * @param[out] dst_ptr          Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_attrs        The attributes of the destination tensor
 * @param[in]  weights_ptr      Pointer to the weights tensor. Supported data types: same as @p src_ptr
 * @param[in]  weights_attrs    The attributes of the weights tensor
 * @param[in]  biases_ptr       Pointer to the biases tensor. Same as @p src_ptr
 * @param[in]  biases_attrs     The attributes of the weights tensor
 * @param[in]  weights_stride_w Stride of the weights tensor in the 4th dimension
 * @param[in]  weights_depth    The third dimensions of the weights tensors
 */
SHADER_PARAMS_DECLARATION
{
    Tensor3DAttributes src_attrs;
    Tensor3DAttributes dst_attrs;
    Tensor3DAttributes weights_attrs;
#ifdef BIAS
    VectorAttributes biases_attrs;
#endif /* BIAS */
    uint weights_stride_w;
    uint weights_depth;
};

#if defined(DATA_TYPE_FP32)
#if defined(PROCESS_1X_1Y_1Z)
TENSOR_DECLARATION(1, srcBuffer, float, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, dstBuffer, float, dst_ptr, dst_shift, 2, writeonly);
TENSOR_DECLARATION(3, weightsBuffer, float, weights_ptr, weights_shift, 2, readonly);
#ifdef BIAS
TENSOR_DECLARATION(4, biasesBuffer, float, biases_ptr, biases_shift, 2, readonly);
#endif /* BIAS */

void main()
{
    ImageIterator    src_iter     = CONVERT_TO_IMAGE_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator weights_iter = CONVERT_TO_TENSOR3D_ITERATOR_NO_STEP(weights_attrs, weights_shift);
    Tensor3DIterator dst_iter     = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

#ifdef BIAS
    VectorIterator biases_iter = CONVERT_TO_VECTOR_ITERATOR_NO_STEP(biases_attrs, biases_shift);
#endif /* BIAS */

    float pixels = 0.f;

    uint z_index = gl_GlobalInvocationID.z;

    TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, z_index * weights_stride_w);

    for(int d = 0; d < int(weights_depth); ++d)
    {
        vec3 temp;
        vec3 w;

        temp = VLOAD3(vec3, src_ptr, IMAGE_OFFSET(src_iter, 0, 0));
        w    = VLOAD3(vec3, weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 0, 0));

        pixels += temp.x * w[0] + temp.y * w[1] + temp.z * w[2];

        temp = VLOAD3(vec3, src_ptr, IMAGE_OFFSET(src_iter, 0, 1));
        w    = VLOAD3(vec3, weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 1, 0));

        pixels += temp.x * w[0] + temp.y * w[1] + temp.z * w[2];

        temp = VLOAD3(vec3, src_ptr, IMAGE_OFFSET(src_iter, 0, 2));
        w    = VLOAD3(vec3, weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 2, 0));

        pixels += temp.x * w[0] + temp.y * w[1] + temp.z * w[2];

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, weights_attrs.stride_z);
    }

#ifdef BIAS
    pixels += LOAD(biases_ptr, VECTOR_OFFSET(biases_iter, z_index));
#endif /* BIAS */

    STORE_CURRENT_ITEM(dst_ptr, dst_iter, pixels);
}

#elif defined(PROCESS_8X_1Y_1Z)

TENSOR_DECLARATION(1, srcBuffer, vec4, src_ptr, src_shift, 4, readonly);
TENSOR_DECLARATION(2, dstBuffer, vec4, dst_ptr, dst_shift, 4, writeonly);
TENSOR_DECLARATION(3, weightsBuffer, float, weights_ptr, weights_shift, 2, readonly);
#ifdef BIAS
TENSOR_DECLARATION(4, biasesBuffer, float, biases_ptr, biases_shift, 2, readonly);
#endif /* BIAS */

#if STRIDE_X == 2
#define CONVOLVE1x3(offset, w) convolve1x3_stride2(offset, w)
#elif STRIDE_X == 1 /* STRIDE_X == 1 */
#define CONVOLVE1x3(offset, w) convolve1x3_stride1(offset, w)
#else /* STRIDE_X not equals 1 or 2 */
#error STRIDE_X larger than 2 is not supported
#endif /* STRIDE_X == 2 */

vec4[2] convolve1x3_stride1(uint offset, vec3 w)
{
    vec4 middle;
    vec4 right;
    vec4 tmp[3];
    vec4 r[2];

    tmp = VLOAD3(vec4[3], src_ptr, offset);

    middle = vec4(tmp[0].yzw, tmp[1].x);
    right  = vec4(tmp[0].zw, tmp[1].xy);

    r[0] = tmp[0] * w[0] + middle * w[1] + right * w[2];

    middle = vec4(tmp[1].yzw, tmp[2].x);
    right  = vec4(tmp[1].zw, tmp[2].xy);

    r[1] = tmp[1] * w[0] + middle * w[1] + right * w[2];

    return r;
}

vec4[2] convolve1x3_stride2(uint offset, vec3 w)
{
    vec4 left;
    vec4 middle;
    vec4 right;
    vec4 tmp1[3];
    vec4 tmp2[2];
    vec4 r[2];

    tmp1 = VLOAD3(vec4[3], src_ptr, offset);

    left   = vec4(tmp1[0].xz, tmp1[1].xz);
    middle = vec4(tmp1[0].yw, tmp1[1].yw);
    right  = vec4(tmp1[0].z, tmp1[1].xz, tmp1[2].x);

    r[0] = left * w[0] + middle * w[1] + right * w[2];

    tmp2 = VLOAD2(vec4[2], src_ptr, offset + uint(3));

    left   = vec4(tmp1[2].xz, tmp2[0].xz);
    middle = vec4(tmp1[2].yw, tmp2[0].yw);
    right  = vec4(tmp1[2].z, tmp2[0].xz, tmp2[1].x);

    r[1] = left * w[0] + middle * w[1] + right * w[2];

    return r;
}

void main()
{
    ImageIterator    src_iter     = CONVERT_TO_IMAGE_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator weights_iter = CONVERT_TO_TENSOR3D_ITERATOR_NO_STEP(weights_attrs, weights_shift);
    Tensor3DIterator dst_iter     = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

#ifdef BIAS
    VectorIterator   biases_iter  = CONVERT_TO_VECTOR_ITERATOR_NO_STEP(biases_attrs, biases_shift);
#endif /* BIAS */

    vec4 pixels[2];
    pixels[0] = vec4(0);
    pixels[1] = vec4(0);

    uint z_index = gl_GlobalInvocationID.z;
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, z_index * weights_stride_w);

    for(int d = 0; d < int(weights_depth); ++d)
    {
        // load 3 weights once
        vec3 w;
        vec4 r[2];

        // first line
        w = VLOAD3(vec3, weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 0, 0));

        r = CONVOLVE1x3(CURRENT_ITEM_OFFSET(src_iter), w);
        pixels[0] += r[0];
        pixels[1] += r[1];

        // second line
        w = VLOAD3(vec3, weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 1, 0));

        r = CONVOLVE1x3(IMAGE_OFFSET(src_iter, 0, 1), w);
        pixels[0] += r[0];
        pixels[1] += r[1];

        // third line
        w = VLOAD3(vec3, weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 2, 0));

        r = CONVOLVE1x3(IMAGE_OFFSET(src_iter, 0, 2), w);
        pixels[0] += r[0];
        pixels[1] += r[1];

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, weights_attrs.stride_z);
    }

#ifdef BIAS
    float b = LOAD(biases_ptr, VECTOR_OFFSET(biases_iter, z_index));
    pixels[0] += vec4(b);
    pixels[1] += vec4(b);
#endif /* BIAS */

    VSTORE2_CURRENT_ITEM(dst_ptr, dst_iter, pixels);
}

#elif defined(PROCESS_4X_1Y_1Z)

TENSOR_DECLARATION(1, srcBuffer, vec4, src_ptr, src_shift, 4, readonly);
TENSOR_DECLARATION(2, dstBuffer, vec4, dst_ptr, dst_shift, 4, writeonly);
TENSOR_DECLARATION(3, weightsBuffer, float, weights_ptr, weights_shift, 2, readonly);
#ifdef BIAS
TENSOR_DECLARATION(4, biasesBuffer, float, biases_ptr, biases_shift, 2, readonly);
#endif /* BIAS */

#if STRIDE_X == 2
#define CONVOLVE1x3(offset, w) convolve1x3_stride2(offset, w)
#elif STRIDE_X == 1 /* STRIDE_X == 1 */
#define CONVOLVE1x3(offset, w) convolve1x3_stride1(offset, w)
#else /* STRIDE_X not equals 1 or 2 */
#error STRIDE_X larger than 2 is not supported
#endif /* STRIDE_X == 2 */

vec4 convolve1x3_stride1(uint offset, vec3 w)
{
    vec4 tmp[2];
    vec4 middle;
    vec4 right;

    tmp = VLOAD2(vec4[2], src_ptr, offset);

    middle = vec4(tmp[0].yzw, tmp[1].x);
    right  = vec4(tmp[0].zw, tmp[1].xy);

    tmp[1] = tmp[0] * w[0] + middle * w[1] + right * w[2];

    return tmp[1];
}

vec4 convolve1x3_stride2(uint offset, vec3 w)
{
    vec4 left;
    vec4 middle;
    vec4 right;

    vec4 tmp[3];

    tmp = VLOAD3(vec4[3], src_ptr, offset);

    left   = vec4(tmp[0].xz, tmp[1].xz);
    middle = vec4(tmp[0].yw, tmp[1].yw);
    right  = vec4(tmp[0].z, tmp[1].xz, tmp[2].x);

    tmp[0] = left * w[0] + middle * w[1] + right * w[2];

    return tmp[0];
}

void main()
{
    ImageIterator    src_iter     = CONVERT_TO_IMAGE_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator weights_iter = CONVERT_TO_TENSOR3D_ITERATOR_NO_STEP(weights_attrs, weights_shift);
    Tensor3DIterator dst_iter     = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

#ifdef BIAS
    VectorIterator   biases_iter  = CONVERT_TO_VECTOR_ITERATOR_NO_STEP(biases_attrs, biases_shift);
#endif /* BIAS */

    vec4 pixels;
    pixels = vec4(0.f);

    uint z_index = gl_GlobalInvocationID.z;
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, z_index * weights_stride_w);

    for(int d = 0; d < int(weights_depth); ++d)
    {
        // load 3 weights once
        vec3 w;

        // first line
        w = VLOAD3(vec3, weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 0, 0));
        pixels += CONVOLVE1x3(CURRENT_ITEM_OFFSET(src_iter), w);

        // second line
        w = VLOAD3(vec3, weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 1, 0));
        pixels += CONVOLVE1x3(IMAGE_OFFSET(src_iter, 0, 1), w);

        // third line
        w = VLOAD3(vec3, weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 2, 0));
        pixels += CONVOLVE1x3(IMAGE_OFFSET(src_iter, 0, 2), w);

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, weights_attrs.stride_z);
    }

#ifdef BIAS
    float b = LOAD(biases_ptr, VECTOR_OFFSET(biases_iter, z_index));
    pixels += b;
#endif /* BIAS */

    STORE_CURRENT_ITEM(dst_ptr, dst_iter, pixels);
}

#elif defined(PROCESS_4X_3Y_1Z)

TENSOR_DECLARATION(1, srcBuffer, vec4, src_ptr, src_shift, 4, readonly);
TENSOR_DECLARATION(2, dstBuffer, vec4, dst_ptr, dst_shift, 4, writeonly);
TENSOR_DECLARATION(3, weightsBuffer, float, weights_ptr, weights_shift, 2, readonly);
#ifdef BIAS
TENSOR_DECLARATION(4, biasesBuffer, float, biases_ptr, biases_shift, 2, readonly);
#endif /* BIAS */

#define CONVOLVE1x3(left, middle, right, w) convolve1x3_stride1(left, middle, right, w)

vec4 convolve1x3_stride1(vec4 left, vec4 middle, vec4 right, vec3 w)
{
    vec4 r;

    r = left * w[0] + middle * w[1] + right * w[2];

    return r;
}

void main()
{
    ImageIterator    src_iter     = CONVERT_TO_IMAGE_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator weights_iter = CONVERT_TO_TENSOR3D_ITERATOR_NO_STEP(weights_attrs, weights_shift);
    Tensor3DIterator dst_iter     = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

#ifdef BIAS
    VectorIterator   biases_iter  = CONVERT_TO_VECTOR_ITERATOR_NO_STEP(biases_attrs, biases_shift);
#endif /* BIAS */

    vec4 pixels[3];
    pixels[0] = vec4(0);
    pixels[1] = vec4(0);
    pixels[2] = vec4(0);

    uint z_index = gl_GlobalInvocationID.z;
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, z_index * weights_stride_w);

    for(int d = 0; d < int(weights_depth); ++d)
    {
        // load 3 weights once
        vec3 w[3];

        w[0] = VLOAD3(vec3, weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 0, 0));
        w[1] = VLOAD3(vec3, weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 1, 0));
        w[2] = VLOAD3(vec3, weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 2, 0));

        vec4 s[2];
        vec4 middle;
        vec4 right;
        // first line
        s      = VLOAD2_CURRENT_ITEM(vec4[2], src_ptr, src_iter);
        middle = vec4(s[0].yzw, s[1].x);
        right  = vec4(s[0].zw, s[1].xy);
        pixels[0] += CONVOLVE1x3(s[0], middle, right, w[0]);

        // second line
        s      = VLOAD2(vec4[2], src_ptr, IMAGE_OFFSET(src_iter, 0, 1));
        middle = vec4(s[0].yzw, s[1].x);
        right  = vec4(s[0].zw, s[1].xy);
        pixels[0] += CONVOLVE1x3(s[0], middle, right, w[1]);
        pixels[1] += CONVOLVE1x3(s[0], middle, right, w[0]);

        // third line
        s      = VLOAD2(vec4[2], src_ptr, IMAGE_OFFSET(src_iter, 0, 2));
        middle = vec4(s[0].yzw, s[1].x);
        right  = vec4(s[0].zw, s[1].xy);
        pixels[0] += CONVOLVE1x3(s[0], middle, right, w[2]);
        pixels[1] += CONVOLVE1x3(s[0], middle, right, w[1]);
        pixels[2] += CONVOLVE1x3(s[0], middle, right, w[0]);

        // forth line
        s      = VLOAD2(vec4[2], src_ptr, IMAGE_OFFSET(src_iter, 0, 3));
        middle = vec4(s[0].yzw, s[1].x);
        right  = vec4(s[0].zw, s[1].xy);
        pixels[1] += CONVOLVE1x3(s[0], middle, right, w[2]);
        pixels[2] += CONVOLVE1x3(s[0], middle, right, w[1]);

        // fifth line
        s      = VLOAD2(vec4[2], src_ptr, IMAGE_OFFSET(src_iter, 0, 4));
        middle = vec4(s[0].yzw, s[1].x);
        right  = vec4(s[0].zw, s[1].xy);
        pixels[2] += CONVOLVE1x3(s[0], middle, right, w[2]);

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, weights_attrs.stride_z);
    }

#ifdef BIAS
    float b = LOAD(biases_ptr, VECTOR_OFFSET(biases_iter, z_index));

    pixels[0] += vec4(b);
    pixels[1] += vec4(b);
    pixels[2] += vec4(b);
#endif /* BIAS */

    STORE_CURRENT_ITEM(dst_ptr, dst_iter, pixels[0]);
    STORE(dst_ptr, TENSOR3D_OFFSET(dst_iter, 0, 1, 0), pixels[1]);
    STORE(dst_ptr, TENSOR3D_OFFSET(dst_iter, 0, 2, 0), pixels[2]);
}

#endif // PROCESS_nX_nY

#elif defined(DATA_TYPE_FP16)

#if defined(PROCESS_8X_3Y_1Z)
TENSOR_DECLARATION(1, srcBuffer, uvec4, src_ptr, src_shift, 4, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec4, dst_ptr, dst_shift, 4, writeonly);
TENSOR_DECLARATION(3, weightsBuffer, uint, weights_ptr, weights_shift, 2, readonly);
#ifdef BIAS
TENSOR_DECLARATION(4, biasesBuffer, uint, biases_ptr, biases_shift, 2, readonly);
#endif /* BIAS */

#define CONVOLVE1x3(s, w) convolve1x3_stride1(s, w)

vec4[2] convolve1x3_stride1(vec4 tmp[3], vec3 w)
{
    vec4 middle;
    vec4 right;
    vec4 r[2];

    middle = vec4(tmp[0].yzw, tmp[1].x);
    right  = vec4(tmp[0].zw, tmp[1].xy);

    r[0] = tmp[0] * w[0] + middle * w[1] + right * w[2];

    middle = vec4(tmp[1].yzw, tmp[2].x);
    right  = vec4(tmp[1].zw, tmp[2].xy);

    r[1] = tmp[1] * w[0] + middle * w[1] + right * w[2];

    return r;
}

vec4[3] vload2_src_unpack12_half(uint offset)
{
    uvec4 packed_s[2];
    vec4  s[3];

    packed_s = VLOAD2(uvec4[2], src_ptr, offset);

    s[0] = vec4(unpackHalf2x16(packed_s[0].x), unpackHalf2x16(packed_s[0].y));
    s[1] = vec4(unpackHalf2x16(packed_s[0].z), unpackHalf2x16(packed_s[0].w));
    s[2] = vec4(unpackHalf2x16(packed_s[1].x), unpackHalf2x16(packed_s[1].y));

    return s;
}

void main()
{
    ImageIterator    src_iter     = CONVERT_TO_IMAGE_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator weights_iter = CONVERT_TO_TENSOR3D_ITERATOR_NO_STEP(weights_attrs, weights_shift);
    Tensor3DIterator dst_iter     = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

#ifdef BIAS
    VectorIterator   biases_iter  = CONVERT_TO_VECTOR_ITERATOR_NO_STEP(biases_attrs, biases_shift);
#endif /* BIAS */

    vec4 pixels[3][2];
    int  i, j;
    for(i = 0; i < 3; i++)
    {
        for(j = 0; j < 2; j++)
        {
            pixels[i][j] = vec4(0);
        }
    }

    uint z_index = gl_GlobalInvocationID.z;
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, z_index * weights_stride_w);

    for(int d = 0; d < int(weights_depth); ++d)
    {
        // load 3 weights once
        uvec2 packed_w[3];

        packed_w[0] = VLOAD2_CURRENT_ITEM(uvec2, weights_ptr, weights_iter);
        packed_w[1] = VLOAD2(uvec2, weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 1, 0));
        packed_w[2] = VLOAD2(uvec2, weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 2, 0));

        vec3 w[3];
        w[0] = vec3(unpackHalf2x16(packed_w[0].x), unpackHalf2x16(packed_w[0].y).x);
        w[1] = vec3(unpackHalf2x16(packed_w[1].x), unpackHalf2x16(packed_w[1].y).x);
        w[2] = vec3(unpackHalf2x16(packed_w[2].x), unpackHalf2x16(packed_w[2].y).x);

        uvec4 packed_s[2];
        vec4  s[3];
        vec4  r[2];

        // first line
        s = vload2_src_unpack12_half(CURRENT_ITEM_OFFSET(src_iter));

        r = CONVOLVE1x3(s, w[0]);
        pixels[0][0] += r[0];
        pixels[0][1] += r[1];

        // second line
        s = vload2_src_unpack12_half(IMAGE_OFFSET(src_iter, 0, 1));

        r = CONVOLVE1x3(s, w[1]);
        pixels[0][0] += r[0];
        pixels[0][1] += r[1];
        r = CONVOLVE1x3(s, w[0]);
        pixels[1][0] += r[0];
        pixels[1][1] += r[1];

        // third line
        s = vload2_src_unpack12_half(IMAGE_OFFSET(src_iter, 0, 2));

        r = CONVOLVE1x3(s, w[2]);
        pixels[0][0] += r[0];
        pixels[0][1] += r[1];
        r = CONVOLVE1x3(s, w[1]);
        pixels[1][0] += r[0];
        pixels[1][1] += r[1];
        r = CONVOLVE1x3(s, w[0]);
        pixels[2][0] += r[0];
        pixels[2][1] += r[1];

        // forth line
        s = vload2_src_unpack12_half(IMAGE_OFFSET(src_iter, 0, 3));

        r = CONVOLVE1x3(s, w[2]);
        pixels[1][0] += r[0];
        pixels[1][1] += r[1];
        r = CONVOLVE1x3(s, w[1]);
        pixels[2][0] += r[0];
        pixels[2][1] += r[1];

        // fifth line
        s = vload2_src_unpack12_half(IMAGE_OFFSET(src_iter, 0, 4));

        r = CONVOLVE1x3(s, w[2]);
        pixels[2][0] += r[0];
        pixels[2][1] += r[1];

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, weights_attrs.stride_z);
    }

#ifdef BIAS
    vec2  vec2_b;
    float b;
    vec2_b = LOAD_UNPACK2_HALF(biases_ptr, VECTOR_OFFSET(biases_iter, z_index));

    if(z_index % uint(2) == uint(0))
    {
        b = vec2_b.x;
    }
    else
    {
        b = vec2_b.y;
    }

    for(i = 0; i < 3; i++)
    {
        for(j = 0; j < 2; j++)
        {
            pixels[i][j] += vec4(b);
        }
    }
#endif /* BIAS */

    STORE_PACK8_CURRENT_ITEM_HALF(dst_ptr, dst_iter, pixels[0]);
    STORE_PACK8_HALF(dst_ptr, TENSOR3D_OFFSET(dst_iter, 0, 1, 0), pixels[1]);
    STORE_PACK8_HALF(dst_ptr, TENSOR3D_OFFSET(dst_iter, 0, 2, 0), pixels[2]);
}

#elif defined(PROCESS_4X_1Y_1Z)
TENSOR_DECLARATION(1, srcBuffer, uvec2, src_ptr, src_shift, 3, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec2, dst_ptr, dst_shift, 3, writeonly);
TENSOR_DECLARATION(3, weightsBuffer, uint, weights_ptr, weights_shift, 2, readonly);
#ifdef BIAS
TENSOR_DECLARATION(4, biasesBuffer, uint, biases_ptr, biases_shift, 2, readonly);
#endif /* BIAS */

#if STRIDE_X == 2
#define CONVOLVE1x3(s, w) convolve1x3_stride2(s, w)
#define LOAD_AND_UNPACK(offset) VLOAD3_UNPACK12_HALF(src_ptr, offset)
#elif STRIDE_X == 1 /* STRIDE_X == 1 */
#define CONVOLVE1x3(s, w) convolve1x3_stride1(s, w)
#define LOAD_AND_UNPACK(offset) VLOAD2_UNPACK8_HALF(src_ptr, offset)
#else /* STRIDE_X not equals 1 or 2 */
#error STRIDE_X larger than 2 is not supported
#endif /* STRIDE_X == 2 */

vec4 convolve1x3_stride1(vec4 tmp[2], vec3 w)
{
    vec4 middle;
    vec4 right;
    vec4 r;

    middle = vec4(tmp[0].yzw, tmp[1].x);
    right  = vec4(tmp[0].zw, tmp[1].xy);

    r = tmp[0] * w[0] + middle * w[1] + right * w[2];

    return r;
}

vec4 convolve1x3_stride2(vec4 tmp[3], vec3 w)
{
    vec4 left;
    vec4 middle;
    vec4 right;
    vec4 r;

    left   = vec4(tmp[0].xz, tmp[1].xz);
    middle = vec4(tmp[0].yw, tmp[1].yw);
    right  = vec4(tmp[0].z, tmp[1].xz, tmp[2].x);

    r = left * w[0] + middle * w[1] + right * w[2];

    return r;
}

void main()
{
    ImageIterator    src_iter     = CONVERT_TO_IMAGE_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator weights_iter = CONVERT_TO_TENSOR3D_ITERATOR_NO_STEP(weights_attrs, weights_shift);
    Tensor3DIterator dst_iter     = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

#ifdef BIAS
    VectorIterator   biases_iter  = CONVERT_TO_VECTOR_ITERATOR_NO_STEP(biases_attrs, biases_shift);
#endif /* BIAS */

    uvec2 packed_d;

    vec4 pixels = vec4(0);

    uint z_index = gl_GlobalInvocationID.z;
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, z_index * weights_stride_w);

    for(int d = 0; d < int(weights_depth); ++d)
    {
        // load 3 weights once
        uvec2 packed_w[3];

        packed_w[0] = VLOAD2_CURRENT_ITEM(uvec2, weights_ptr, weights_iter);
        packed_w[1] = VLOAD2(uvec2, weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 1, 0));
        packed_w[2] = VLOAD2(uvec2, weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 2, 0));

        vec3 w[3];
        w[0] = vec3(unpackHalf2x16(packed_w[0].x), unpackHalf2x16(packed_w[0].y).x);
        w[1] = vec3(unpackHalf2x16(packed_w[1].x), unpackHalf2x16(packed_w[1].y).x);
        w[2] = vec3(unpackHalf2x16(packed_w[2].x), unpackHalf2x16(packed_w[2].y).x);

#if STRIDE_X == 2
        vec4 s[3];
#elif STRIDE_X == 1 /* STRIDE_X == 1 */
        vec4 s[2];
#else               /* STRIDE_X not equals 1 or 2 */
#error STRIDE_X larger than 2 is not supported
#endif /* STRIDE_X == 2 */
        vec4 r;

        // first line
        s = LOAD_AND_UNPACK(CURRENT_ITEM_OFFSET(src_iter));
        pixels += CONVOLVE1x3(s, w[0]);

        // second line
        s = LOAD_AND_UNPACK(IMAGE_OFFSET(src_iter, 0, 1));
        pixels += CONVOLVE1x3(s, w[1]);

        // third line
        s = LOAD_AND_UNPACK(IMAGE_OFFSET(src_iter, 0, 2));
        pixels += CONVOLVE1x3(s, w[2]);

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, weights_attrs.stride_z);
    }

#ifdef BIAS
    vec2  vec2_b;
    float b;

    vec2_b = LOAD_UNPACK2_HALF(biases_ptr, VECTOR_OFFSET(biases_iter, z_index));

    if(z_index % uint(2) == uint(0))
    {
        b = vec2_b.x;
    }
    else
    {
        b = vec2_b.y;
    }

    pixels += vec4(b);
#endif /* BIAS */

    STORE_PACK4_CURRENT_ITEM_HALF(dst_ptr, dst_iter, pixels);
}

#elif defined(PROCESS_4X_3Y_1Z)
TENSOR_DECLARATION(1, srcBuffer, uvec2, src_ptr, src_shift, 3, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec2, dst_ptr, dst_shift, 3, writeonly);
TENSOR_DECLARATION(3, weightsBuffer, uint, weights_ptr, weights_shift, 2, readonly);
#ifdef BIAS
TENSOR_DECLARATION(4, biasesBuffer, uint, biases_ptr, biases_shift, 2, readonly);
#endif /* BIAS */

#define CONVOLVE1x3(s, w) convolve1x3_stride1(s, w)

vec4 convolve1x3_stride1(vec4 tmp[2], vec3 w)
{
    vec4 middle;
    vec4 right;
    vec4 r;

    middle = vec4(tmp[0].yzw, tmp[1].x);
    right  = vec4(tmp[0].zw, tmp[1].xy);

    r = tmp[0] * w[0] + middle * w[1] + right * w[2];

    return r;
}

void main()
{
    ImageIterator    src_iter     = CONVERT_TO_IMAGE_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator weights_iter = CONVERT_TO_TENSOR3D_ITERATOR_NO_STEP(weights_attrs, weights_shift);
    Tensor3DIterator dst_iter     = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

#ifdef BIAS
    VectorIterator   biases_iter  = CONVERT_TO_VECTOR_ITERATOR_NO_STEP(biases_attrs, biases_shift);
#endif /* BIAS */

    vec4 pixels[3];
    int  i;

    for(i = 0; i < 3; i++)
    {
        pixels[i] = vec4(0);
    }

    uint z_index = gl_GlobalInvocationID.z;
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, z_index * weights_stride_w);

    for(int d = 0; d < int(weights_depth); ++d)
    {
        // load 3 weights once
        uvec2 packed_w[3];

        packed_w[0] = VLOAD2_CURRENT_ITEM(uvec2, weights_ptr, weights_iter);
        packed_w[1] = VLOAD2(uvec2, weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 1, 0));
        packed_w[2] = VLOAD2(uvec2, weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 2, 0));

        vec3 w[3];
        w[0] = vec3(unpackHalf2x16(packed_w[0].x), unpackHalf2x16(packed_w[0].y).x);
        w[1] = vec3(unpackHalf2x16(packed_w[1].x), unpackHalf2x16(packed_w[1].y).x);
        w[2] = vec3(unpackHalf2x16(packed_w[2].x), unpackHalf2x16(packed_w[2].y).x);

        vec4 s[2];
        vec4 r;

        // first line
        s = VLOAD2_UNPACK8_CURRENT_ITEM_HALF(src_ptr, src_iter);
        pixels[0] += CONVOLVE1x3(s, w[0]);

        // second line
        s = VLOAD2_UNPACK8_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, 1));
        pixels[0] += CONVOLVE1x3(s, w[1]);
        pixels[1] += CONVOLVE1x3(s, w[0]);

        // third line
        s = VLOAD2_UNPACK8_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, 2));
        pixels[0] += CONVOLVE1x3(s, w[2]);
        pixels[1] += CONVOLVE1x3(s, w[1]);
        pixels[2] += CONVOLVE1x3(s, w[0]);

        // forth line
        s = VLOAD2_UNPACK8_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, 3));
        pixels[1] += CONVOLVE1x3(s, w[2]);
        pixels[2] += CONVOLVE1x3(s, w[1]);

        // fifth line
        s = VLOAD2_UNPACK8_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, 4));
        pixels[2] += CONVOLVE1x3(s, w[2]);

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, weights_attrs.stride_z);
    }

#ifdef BIAS
    vec2  vec2_b;
    float b;
    vec2_b = LOAD_UNPACK2_HALF(biases_ptr, VECTOR_OFFSET(biases_iter, z_index));

    if(z_index % uint(2) == uint(0))
    {
        b = vec2_b.x;
    }
    else
    {
        b = vec2_b.y;
    }

    for(i = 0; i < 3; i++)
    {
        pixels[i] += vec4(b);
    }
#endif /* BIAS */

    STORE_PACK4_CURRENT_ITEM_HALF(dst_ptr, dst_iter, pixels[0]);
    STORE_PACK4_HALF(dst_ptr, TENSOR3D_OFFSET(dst_iter, 0, 1, 0), pixels[1]);
    STORE_PACK4_HALF(dst_ptr, TENSOR3D_OFFSET(dst_iter, 0, 2, 0), pixels[2]);
}

#elif defined(PROCESS_4X_4Y_1Z)
TENSOR_DECLARATION(1, srcBuffer, uvec2, src_ptr, src_shift, 3, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec2, dst_ptr, dst_shift, 3, writeonly);
TENSOR_DECLARATION(3, weightsBuffer, uint, weights_ptr, weights_shift, 2, readonly);
#ifdef BIAS
TENSOR_DECLARATION(4, biasesBuffer, uint, biases_ptr, biases_shift, 2, readonly);
#endif /* BIAS */

#define CONVOLVE1x3(s, w) convolve1x3_stride1(s, w)

vec4 convolve1x3_stride1(vec4 tmp[2], vec3 w)
{
    vec4 middle;
    vec4 right;
    vec4 r;

    middle = vec4(tmp[0].yzw, tmp[1].x);
    right  = vec4(tmp[0].zw, tmp[1].xy);

    r = tmp[0] * w[0] + middle * w[1] + right * w[2];

    return r;
}

void main()
{
    ImageIterator    src_iter     = CONVERT_TO_IMAGE_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator weights_iter = CONVERT_TO_TENSOR3D_ITERATOR_NO_STEP(weights_attrs, weights_shift);
    Tensor3DIterator dst_iter     = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

#ifdef BIAS
    VectorIterator   biases_iter  = CONVERT_TO_VECTOR_ITERATOR_NO_STEP(biases_attrs, biases_shift);
#endif /* BIAS */

    vec4 pixels[4];
    int  i;

    for(i = 0; i < 4; i++)
    {
        pixels[i] = vec4(0);
    }

    uint z_index = gl_GlobalInvocationID.z;
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, z_index * weights_stride_w);

    for(int d = 0; d < int(weights_depth); ++d)
    {
        // load 3 weights once
        uvec2 packed_w[3];

        packed_w[0] = VLOAD2(uvec2, weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 0, 0));
        packed_w[1] = VLOAD2(uvec2, weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 1, 0));
        packed_w[2] = VLOAD2(uvec2, weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 2, 0));

        vec3 w[3];
        w[0] = vec3(unpackHalf2x16(packed_w[0].x), unpackHalf2x16(packed_w[0].y).x);
        w[1] = vec3(unpackHalf2x16(packed_w[1].x), unpackHalf2x16(packed_w[1].y).x);
        w[2] = vec3(unpackHalf2x16(packed_w[2].x), unpackHalf2x16(packed_w[2].y).x);

        vec4 s[2];
        vec4 r;

        // first line
        s = VLOAD2_UNPACK8_CURRENT_ITEM_HALF(src_ptr, src_iter);
        pixels[0] += CONVOLVE1x3(s, w[0]);

        // second line
        s = VLOAD2_UNPACK8_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, 1));
        pixels[0] += CONVOLVE1x3(s, w[1]);
        pixels[1] += CONVOLVE1x3(s, w[0]);

        // third line
        s = VLOAD2_UNPACK8_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, 2));
        pixels[0] += CONVOLVE1x3(s, w[2]);
        pixels[1] += CONVOLVE1x3(s, w[1]);
        pixels[2] += CONVOLVE1x3(s, w[0]);

        // forth line
        s = VLOAD2_UNPACK8_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, 3));
        pixels[1] += CONVOLVE1x3(s, w[2]);
        pixels[2] += CONVOLVE1x3(s, w[1]);
        pixels[3] += CONVOLVE1x3(s, w[0]);

        // fifth line
        s = VLOAD2_UNPACK8_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, 4));
        pixels[2] += CONVOLVE1x3(s, w[2]);
        pixels[3] += CONVOLVE1x3(s, w[1]);

        // sixth line
        s = VLOAD2_UNPACK8_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, 5));
        pixels[3] += CONVOLVE1x3(s, w[2]);

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, weights_attrs.stride_z);
    }

#ifdef BIAS
    vec2  vec2_b;
    float b;
    vec2_b = LOAD_UNPACK2_HALF(biases_ptr, VECTOR_OFFSET(biases_iter, z_index));

    if(z_index % uint(2) == uint(0))
    {
        b = vec2_b.x;
    }
    else
    {
        b = vec2_b.y;
    }

    for(i = 0; i < 4; i++)
    {
        pixels[i] += vec4(b);
    }
#endif /* BIAS */

    STORE_PACK4_CURRENT_ITEM_HALF(dst_ptr, dst_iter, pixels[0]);
    STORE_PACK4_HALF(dst_ptr, TENSOR3D_OFFSET(dst_iter, 0, 1, 0), pixels[1]);
    STORE_PACK4_HALF(dst_ptr, TENSOR3D_OFFSET(dst_iter, 0, 2, 0), pixels[2]);
    STORE_PACK4_HALF(dst_ptr, TENSOR3D_OFFSET(dst_iter, 0, 3, 0), pixels[3]);
}
#elif defined(PROCESS_4X_3Y_2Z)
TENSOR_DECLARATION(1, srcBuffer, uvec2, src_ptr, src_shift, 3, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec2, dst_ptr, dst_shift, 3, writeonly);
TENSOR_DECLARATION(3, weightsBuffer, uint, weights_ptr, weights_shift, 2, readonly);
#ifdef BIAS
TENSOR_DECLARATION(4, biasesBuffer, uint, biases_ptr, biases_shift, 2, readonly);
#endif /* BIAS */

#define CONVOLVE1x3(s, w) convolve1x3_stride1(s, w)

vec4 convolve1x3_stride1(vec4 tmp[2], vec3 w)
{
    vec4 middle;
    vec4 right;
    vec4 r;

    middle = vec4(tmp[0].yzw, tmp[1].x);
    right  = vec4(tmp[0].zw, tmp[1].xy);

    r = tmp[0] * w[0] + middle * w[1] + right * w[2];

    return r;
}

void main()
{
    ImageIterator    src_iter     = CONVERT_TO_IMAGE_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator weights_iter = CONVERT_TO_TENSOR3D_ITERATOR_NO_STEP(weights_attrs, weights_shift);
    Tensor3DIterator dst_iter     = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

#ifdef BIAS
    VectorIterator   biases_iter  = CONVERT_TO_VECTOR_ITERATOR_NO_STEP(biases_attrs, biases_shift);
#endif /* BIAS */

    vec4 pixels[3];
    int  i;

    uint z_base_index = gl_GlobalInvocationID.z << 1;

    // store orginal src current offset
    uint s_offset_in_bytes = CURRENT_ITEM_OFFSET_IN_BYTES(srcc_iter);

    TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, z_base_index * weights_stride_w);

    for(int z = 0; z < 2; ++z)
    {
        uint z_index = z_base_index + uint(z);

        SET_TENSOR_ITERATOR_OFFSET_IN_BYTES(src_iter, s_offset_in_bytes);

        for(i = 0; i < 3; i++)
        {
            pixels[i] = vec4(0);
        }

        for(int d = 0; d < int(weights_depth); ++d)
        {
            // load 3 weights once
            uvec2 packed_w[3];

            packed_w[0] = VLOAD2(uvec2, weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 0, 0));
            packed_w[1] = VLOAD2(uvec2, weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 1, 0));
            packed_w[2] = VLOAD2(uvec2, weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 2, 0));

            vec3 w[3];
            w[0] = vec3(unpackHalf2x16(packed_w[0].x), unpackHalf2x16(packed_w[0].y).x);
            w[1] = vec3(unpackHalf2x16(packed_w[1].x), unpackHalf2x16(packed_w[1].y).x);
            w[2] = vec3(unpackHalf2x16(packed_w[2].x), unpackHalf2x16(packed_w[2].y).x);

            vec4 s[2];
            vec4 r;

            // first line
            s = VLOAD2_UNPACK8_CURRENT_ITEM_HALF(src_ptr, src_iter);
            pixels[0] += CONVOLVE1x3(s, w[0]);

            // second line
            s = VLOAD2_UNPACK8_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, 1));
            pixels[0] += CONVOLVE1x3(s, w[1]);
            pixels[1] += CONVOLVE1x3(s, w[0]);

            // third line
            s = VLOAD2_UNPACK8_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, 2));
            pixels[0] += CONVOLVE1x3(s, w[2]);
            pixels[1] += CONVOLVE1x3(s, w[1]);
            pixels[2] += CONVOLVE1x3(s, w[0]);

            // forth line
            s = VLOAD2_UNPACK8_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, 3));
            pixels[1] += CONVOLVE1x3(s, w[2]);
            pixels[2] += CONVOLVE1x3(s, w[1]);

            // fifth line
            s = VLOAD2_UNPACK8_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, 4));
            pixels[2] += CONVOLVE1x3(s, w[2]);

            TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);
            TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, weights_attrs.stride_z);
        }

#ifdef BIAS
        vec2  vec2_b;
        float b;
        vec2_b = LOAD_UNPACK2_HALF(biases_ptr, VECTOR_OFFSET(biases_iter, z_index));

        if(z_index % uint(2) == uint(0))
        {
            b = vec2_b.x;
        }
        else
        {
            b = vec2_b.y;
        }

        for(i = 0; i < 3; i++)
        {
            pixels[i] += vec4(b);
        }
#endif /* BIAS */

        STORE_PACK4_CURRENT_ITEM_HALF(dst_ptr, dst_iter, pixels[0]);
        STORE_PACK4_HALF(dst_ptr, TENSOR3D_OFFSET(dst_iter, 0, 1, 0), pixels[1]);
        STORE_PACK4_HALF(dst_ptr, TENSOR3D_OFFSET(dst_iter, 0, 2, 0), pixels[2]);

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(dst_iter, dst_stride_z);
    }
}

#endif /* PROCESS_nX_nY_nZ */

#else /* DATA_TYPE_FP32 */
#error Data type not supported
#endif /* DATA_TYPE_FP32 */
