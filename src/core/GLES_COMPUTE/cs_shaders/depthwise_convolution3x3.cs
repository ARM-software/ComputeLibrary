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

/** This kernel performs a depthwise convolution.
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_NAME". e.g. "#define DATA_TYPE_FP32"
 * @note This kernel has multiple optimized depthwise convolution options for FP16.
 *       The depthwise convolution option must be passed at compile time using "#define PROCESS_nX_nY_nZ" e.g. "#define PROCESS_8X_1Y_1Z"
 * @note The convolution stride x must be passed at compile time using "#define STRIDE_X n" e.g. "#define STRIDE_X 1"
 * @note In case biases will be added to the convolution "#define HAS_BIAS" has to be passed to append the final matrix with 1 in each row.
 *
 * @param[in]  src_ptr       Pointer to the source tensor. Supported data types: F16
 * @param[in]  src_attrs     The attributes of the source tensor
 * @param[out] dst_ptr       Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_attrs     The attributes of the destination tensor
 * @param[in]  weights_ptr   Pointer to the weights tensor. Supported data types: same as @p src_ptr
 * @param[in]  weights_attrs The attributes of the weights tensor
 * @param[in]  biases_ptr    Pointer to the biases tensor. Same as @p src_ptr
 * @param[in]  biases_attrs  The attributes of the weights tensor
 */
SHADER_PARAMS_DECLARATION
{
    Tensor3DAttributes src_attrs;
    Tensor3DAttributes dst_attrs;
    Tensor3DAttributes weights_attrs;
#ifdef BIAS
    VectorAttributes biases_attrs;
#endif /* BIAS */
};

#if defined(DATA_TYPE_FP16)
#if defined(PROCESS_4X_3Y_1Z)
TENSOR_DECLARATION(1, srcBuffer, uvec2, src_ptr, src_shift, 3, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec2, dst_ptr, dst_shift, 3, writeonly);
TENSOR_DECLARATION(3, weightsBuffer, uvec2, weights_ptr, weights_shift, 3, readonly);
#ifdef BIAS
TENSOR_DECLARATION(4, biasesBuffer, uint, biases_ptr, biases_shift, 2, readonly);
#endif /* BIAS */

#define LOAD_UNPACK_SWIZZLE(offset) load_unpack_swizzle_stride1(offset)

vec4 convolve1x3(vec4 s[3], vec4 w)
{
    vec4 r;

    r = s[0] * w[0] + s[1] * w[1] + s[2] * w[2];

    return r;
}

vec4[3] load_unpack_swizzle_stride1(uint offset)
{
    vec4 s[2];
    s = VLOAD2_UNPACK8_HALF(src_ptr, offset);

    vec4 r[3];
    r[0] = s[0];
    r[1] = vec4(s[0].yzw, s[1].x);
    r[2] = vec4(s[0].zw, s[1].xy);

    return r;
}

void main()
{
    Tensor3DIterator src_iter     = CONVERT_TO_TENSOR3D_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator weights_iter = CONVERT_TO_TENSOR3D_ITERATOR_NO_STEP(weights_attrs, weights_shift);
    Tensor3DIterator dst_iter     = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

#ifdef BIAS
    VectorIterator biases_iter = CONVERT_TO_VECTOR_ITERATOR_NO_STEP(biases_attrs, biases_shift);
#endif /* BIAS */

    vec4 pixels[3];
    for(int i = 0; i < 3; i++)
    {
        pixels[i] = vec4(0);
    }

    uint z_index = gl_GlobalInvocationID.z;
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, z_index * weights_attrs.stride_z);

    vec4 w[3];
    w[0] = LOAD_UNPACK4_CURRENT_ITEM_HALF(weights_ptr, weights_iter);
    w[1] = LOAD_UNPACK4_HALF(weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 1, 0));
    w[2] = LOAD_UNPACK4_HALF(weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 2, 0));

    vec4 s[3];
    vec4 r;
    // first line
    s = LOAD_UNPACK_SWIZZLE(CURRENT_ITEM_OFFSET(src_iter));

    r = convolve1x3(s, w[0]);
    pixels[0] += r;

    // second line
    s = LOAD_UNPACK_SWIZZLE(TENSOR3D_OFFSET(src_iter, 0, 1, 0));

    r = convolve1x3(s, w[1]);
    pixels[0] += r;
    r = convolve1x3(s, w[0]);
    pixels[1] += r;

    // third line
    s = LOAD_UNPACK_SWIZZLE(TENSOR3D_OFFSET(src_iter, 0, 2, 0));

    r = convolve1x3(s, w[2]);
    pixels[0] += r;
    r = convolve1x3(s, w[1]);
    pixels[1] += r;
    r = convolve1x3(s, w[0]);
    pixels[2] += r;

    // forth line
    s = LOAD_UNPACK_SWIZZLE(TENSOR3D_OFFSET(src_iter, 0, 3, 0));

    r = convolve1x3(s, w[2]);
    pixels[1] += r;
    r = convolve1x3(s, w[1]);
    pixels[2] += r;

    // fifth line
    s = LOAD_UNPACK_SWIZZLE(TENSOR3D_OFFSET(src_iter, 0, 4, 0));

    r = convolve1x3(s, w[2]);
    pixels[2] += r;

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

    for(int i = 0; i < 3; i++)
    {
        pixels[i] += vec4(b);
    }
#endif /* BIAS */

    STORE_PACK4_CURRENT_ITEM_HALF(dst_ptr, dst_iter, pixels[0]);
    STORE_PACK4_HALF(dst_ptr, TENSOR3D_OFFSET(dst_iter, 0, 1, 0), pixels[1]);
    STORE_PACK4_HALF(dst_ptr, TENSOR3D_OFFSET(dst_iter, 0, 2, 0), pixels[2]);
}
#elif defined(PROCESS_4X_1Y_1Z)
TENSOR_DECLARATION(1, srcBuffer, uvec2, src_ptr, src_shift, 3, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec2, dst_ptr, dst_shift, 3, writeonly);
TENSOR_DECLARATION(3, weightsBuffer, uvec2, weights_ptr, weights_shift, 3, readonly);
#ifdef BIAS
TENSOR_DECLARATION(4, biasesBuffer, uint, biases_ptr, biases_shift, 2, readonly);
#endif /* BIAS */

#if STRIDE_X == 3
#define LOAD_UNPACK_SWIZZLE(offset) load_unpack_swizzle_stride3(offset)
#elif STRIDE_X == 2
#define LOAD_UNPACK_SWIZZLE(offset) load_unpack_swizzle_stride2(offset)
#elif STRIDE_X == 1 /* STRIDE_X == 1 */
#define LOAD_UNPACK_SWIZZLE(offset) load_unpack_swizzle_stride1(offset)
#else /* STRIDE_X not equals 1 or 2 */
#error STRIDE_X larger than 2 is not supported
#endif /* STRIDE_X == 2 */

vec4 convolve1x3(vec4 s[3], vec4 w)
{
    vec4 r;

    r = s[0] * w[0] + s[1] * w[1] + s[2] * w[2];

    return r;
}

vec4[3] load_unpack_swizzle_stride1(uint offset)
{
    vec4 s[2];
    s = VLOAD2_UNPACK8_HALF(src_ptr, offset);

    vec4 r[3];
    r[0] = s[0];
    r[1] = vec4(s[0].yzw, s[1].x);
    r[2] = vec4(s[0].zw, s[1].xy);

    return r;
}

vec4[3] load_unpack_swizzle_stride2(uint offset)
{
    vec4 s[3];
    s[0] = LOAD_UNPACK4_HALF(src_ptr, offset);
    s[1] = LOAD_UNPACK4_HALF(src_ptr, offset + uint(1));
    s[2] = LOAD_UNPACK4_HALF(src_ptr, offset + uint(2));

    vec4 r[3];
    r[0] = vec4(s[0].xz, s[1].xz);
    r[1] = vec4(s[0].yw, s[1].yw);
    r[2] = vec4(s[0].z, s[1].xz, s[2].x);

    return r;
}

vec4[3] load_unpack_swizzle_stride3(uint offset)
{
    vec4 s[3];
    s[0] = LOAD_UNPACK4_HALF(src_ptr, offset);
    s[1] = LOAD_UNPACK4_HALF(src_ptr, offset + uint(1));
    s[2] = LOAD_UNPACK4_HALF(src_ptr, offset + uint(2));

    vec4 r[3];
    r[0] = vec4(s[0].xw, s[1].z, s[2].y);
    r[1] = vec4(s[0].y, s[1].xw, s[2].z);
    r[2] = vec4(s[0].z, s[1].y, s[2].xw);

    return r;
}

void main()
{
    Tensor3DIterator src_iter     = CONVERT_TO_TENSOR3D_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator weights_iter = CONVERT_TO_TENSOR3D_ITERATOR_NO_STEP(weights_attrs, weights_shift);
    Tensor3DIterator dst_iter     = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

#ifdef BIAS
    VectorIterator   biases_iter  = CONVERT_TO_VECTOR_ITERATOR_NO_STEP(biases_attrs, biases_shift);
#endif /* BIAS */

    vec4 pixels = vec4(0.f);

    uint z_index = gl_GlobalInvocationID.z;
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, z_index * weights_attrs.stride_z);

    vec4 w[3];
    w[0] = LOAD_UNPACK4_CURRENT_ITEM_HALF(weights_ptr, weights_iter);
    w[1] = LOAD_UNPACK4_HALF(weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 1, 0));
    w[2] = LOAD_UNPACK4_HALF(weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 2, 0));

    vec4 s[3];
    vec4 r;
    // first line
    s = LOAD_UNPACK_SWIZZLE(CURRENT_ITEM_OFFSET(src_iter));

    r = convolve1x3(s, w[0]);
    pixels += r;

    // second line
    s = LOAD_UNPACK_SWIZZLE(TENSOR3D_OFFSET(src_iter, 0, 1, 0));

    r = convolve1x3(s, w[1]);
    pixels += r;

    // third line
    s = LOAD_UNPACK_SWIZZLE(TENSOR3D_OFFSET(src_iter, 0, 2, 0));

    r = convolve1x3(s, w[2]);
    pixels += r;

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
#endif /* PROCESS_4X_3Y_1Z */
#endif /* DATA_TYPE_FP16 */
