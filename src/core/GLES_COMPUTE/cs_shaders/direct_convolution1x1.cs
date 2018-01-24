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

    float pixels  = 0.f;
    uint  z_index = gl_GlobalInvocationID.z;
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, z_index * weights_stride_w);

    float temp;
    float temp_weight;
    for(int d = 0; d < int(weights_depth); ++d)
    {
        temp        = LOAD_CURRENT_ITEM(src_ptr, src_iter);
        temp_weight = LOAD_CURRENT_ITEM(weights_ptr, weights_iter);
        pixels += temp * temp_weight;

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, weights_attrs.stride_z);
    }

#ifdef BIAS
    pixels += LOAD(biases_ptr, VECTOR_OFFSET(biases_iter, z_index));
#endif /* BIAS */

    STORE_CURRENT_ITEM(dst_ptr, dst_iter, pixels);
}

#elif defined(DATA_TYPE_FP16)
#if defined(PROCESS_4X_1Y_1Z)
TENSOR_DECLARATION(1, srcBuffer, uvec2, src_ptr, src_shift, 3, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec2, dst_ptr, dst_shift, 3, writeonly);
TENSOR_DECLARATION(3, weightsBuffer, uint, weights_ptr, weights_shift, 2, readonly);
#ifdef BIAS
TENSOR_DECLARATION(4, biasesBuffer, uint, biases_ptr, biases_shift, 2, readonly);
#endif /* BIAS */

#if STRIDE_X == 2
#define CONVOLVE(s, w) convolve_stride2(s, w)
#elif STRIDE_X == 1 /* STRIDE_X == 1 */
#define CONVOLVE(s, w) convolve_stride1(s, w)
#else /* STRIDE_X not equals 1 or 2 */
#error STRIDE_X larger than 2 is not supported
#endif /* STRIDE_X == 2 */

vec4 convolve_stride1(ImageIterator src_iter, float w)
{
    vec4 s;
    s = LOAD_UNPACK4_CURRENT_ITEM_HALF(src_ptr, src_iter);

    s *= w;

    return s;
}

vec4 convolve_stride2(ImageIterator src_iter, float w)
{
    vec4 s[2];
    vec4 r;

    s[0] = LOAD_UNPACK4_CURRENT_ITEM_HALF(src_ptr, src_iter);
    s[1] = LOAD_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, 4, 0));
    r    = vec4(s[0].xz, s[1].xz);

    r *= w;

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

    vec4 pixels = vec4(0.f);

    uint z_index = gl_GlobalInvocationID.z;
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, z_index * weights_stride_w);

#ifdef WEIGHTS_OPTIMIZATION
    float w1, w2;
    int   nums = (int(weights_depth)) / 2;
    for(int d = 0; d < nums; ++d)
    {
        vec2 vec2_w = LOAD_UNPACK2_CURRENT_ITEM_HALF(weights_ptr, weights_iter);

        w1      = vec2_w.x;
        vec4 r1 = CONVOLVE(src_iter, w1);
        pixels += r1;

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);

        w2      = vec2_w.y;
        vec4 r2 = CONVOLVE(src_iter, w2);
        pixels += r2;

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, weights_attrs.stride_z * uint(2));
    }
#else  /* WEIGHTS_OPTIMIZATION */
    float w;
    for(int d = 0; d < int(weights_depth); ++d)
    {
        w = LOAD_UNPACK2_CURRENT_ITEM_HALF(weights_ptr, weights_iter).x;

        vec4 r = CONVOLVE(src_iter, w);
        pixels += r;

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, weights_attrs.stride_z);
    }
#endif /* WEIGHTS_OPTIMIZATION */

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

    pixels += b;
#endif /* BIAS */

    STORE_PACK4_CURRENT_ITEM_HALF(dst_ptr, dst_iter, pixels);
}
#elif defined(PROCESS_4X_2Y_1Z)
TENSOR_DECLARATION(1, srcBuffer, uvec2, src_ptr, src_shift, 3, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec2, dst_ptr, dst_shift, 3, writeonly);
TENSOR_DECLARATION(3, weightsBuffer, uint, weights_ptr, weights_shift, 2, readonly);
#ifdef BIAS
TENSOR_DECLARATION(4, biasesBuffer, uint, biases_ptr, biases_shift, 2, readonly);
#endif /* BIAS */

#if STRIDE_X == 2
#define CONVOLVE(s, w) convolve_stride2(s, w)
#elif STRIDE_X == 1 /* STRIDE_X == 1 */
#define CONVOLVE(s, w) convolve_stride1(s, w)
#else /* STRIDE_X not equals 1 or 2 */
#error STRIDE_X larger than 2 is not supported
#endif /* STRIDE_X == 2 */

vec4[2] convolve_stride1(ImageIterator src_iter, float w)
{
    vec4 s[2];
    s[0] = LOAD_UNPACK4_CURRENT_ITEM_HALF(src_ptr, src_iter);
    s[1] = LOAD_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, int(STRIDE_Y)));

    s[0] *= w;
    s[1] *= w;

    return s;
}

vec4[2] convolve_stride2(ImageIterator src_iter, float w)
{
    vec4 s1[2];
    vec4 s2[2];
    vec4 r[2];

    s1[0] = LOAD_UNPACK4_CURRENT_ITEM_HALF(src_ptr, src_iter);
    s1[1] = LOAD_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, 4, 0));
    r[0]  = vec4(s1[0].xz, s1[1].xz);

    s2[0] = LOAD_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, int(STRIDE_Y)));
    s2[1] = LOAD_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, 4, int(STRIDE_Y)));
    r[1]  = vec4(s2[0].xz, s2[1].xz);

    r[0] *= w;
    r[1] *= w;

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
    pixels[0] = vec4(0.f);
    pixels[1] = vec4(0.f);

    uint z_index = gl_GlobalInvocationID.z;
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, z_index * weights_stride_w);

#ifdef WEIGHTS_OPTIMIZATION
    float w1, w2;
    int   nums = (int(weights_depth)) / 2;
    for(int d = 0; d < nums; ++d)
    {
        vec2 vec2_w = LOAD_UNPACK2_CURRENT_ITEM_HALF(weights_ptr, weights_iter);

        w1         = vec2_w.x;
        vec4 r1[2] = CONVOLVE(src_iter, w1);
        pixels[0] += r1[0];
        pixels[1] += r1[1];

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);

        w2         = vec2_w.y;
        vec4 r2[2] = CONVOLVE(src_iter, w2);
        pixels[0] += r2[0];
        pixels[1] += r2[1];

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, weights_attrs.stride_z * uint(2));
    }
#else  /* WEIGHTS_OPTIMIZATION */
    float w;
    for(int d = 0; d < int(weights_depth); ++d)
    {
        w = LOAD_UNPACK2_CURRENT_ITEM_HALF(weights_ptr, weights_iter).x;

        vec4 r[2] = CONVOLVE(src_iter, w);
        pixels[0] += r[0];
        pixels[1] += r[1];

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, weights_attrs.stride_z);
    }
#endif /* WEIGHTS_OPTIMIZATION */

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

    pixels[0] += b;
    pixels[1] += b;
#endif /* BIAS */

    STORE_PACK4_CURRENT_ITEM_HALF(dst_ptr, dst_iter, pixels[0]);
    STORE_PACK4_HALF(dst_ptr, TENSOR3D_OFFSET(dst_iter, 0, 1, 0), pixels[1]);
}
#elif defined(PROCESS_4X_3Y_1Z)
TENSOR_DECLARATION(1, srcBuffer, uvec2, src_ptr, src_shift, 3, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec2, dst_ptr, dst_shift, 3, writeonly);
TENSOR_DECLARATION(3, weightsBuffer, uint, weights_ptr, weights_shift, 2, readonly);
#ifdef BIAS
TENSOR_DECLARATION(4, biasesBuffer, uint, biases_ptr, biases_shift, 2, readonly);
#endif /* BIAS */

#if STRIDE_X == 2
#define CONVOLVE(s, w) convolve_stride2(s, w)
#elif STRIDE_X == 1 /* STRIDE_X == 1 */
#define CONVOLVE(s, w) convolve_stride1(s, w)
#else /* STRIDE_X not equals 1 or 2 */
#error STRIDE_X larger than 2 is not supported
#endif /* STRIDE_X == 2 */

vec4[3] convolve_stride1(ImageIterator src_iter, float w)
{
    vec4 s[3];
    s[0] = LOAD_UNPACK4_CURRENT_ITEM_HALF(src_ptr, src_iter);
    s[1] = LOAD_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, int(STRIDE_Y)));
    s[2] = LOAD_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, (2 * int(STRIDE_Y))));

    s[0] *= w;
    s[1] *= w;
    s[2] *= w;

    return s;
}

vec4[3] convolve_stride2(ImageIterator src_iter, float w)
{
    vec4 s1[2];
    vec4 s2[2];
    vec4 s3[2];
    vec4 r[3];

    s1[0] = LOAD_UNPACK4_CURRENT_ITEM_HALF(src_ptr, src_iter);
    s1[1] = LOAD_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, 4, 0));
    r[0]  = vec4(s1[0].xz, s1[1].xz);

    s2[0] = LOAD_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, int(STRIDE_Y)));
    s2[1] = LOAD_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, 4, int(STRIDE_Y)));
    r[1]  = vec4(s2[0].xz, s2[1].xz);

    s3[0] = LOAD_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, (2 * int(STRIDE_Y))));
    s3[1] = LOAD_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, 4, (2 * int(STRIDE_Y))));
    r[2]  = vec4(s3[0].xz, s3[1].xz);

    r[0] *= w;
    r[1] *= w;
    r[2] *= w;

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
    pixels[0] = vec4(0.f);
    pixels[1] = vec4(0.f);
    pixels[2] = vec4(0.f);

    uint z_index = gl_GlobalInvocationID.z;
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, z_index * weights_stride_w);

#ifdef WEIGHTS_OPTIMIZATION
    float w1, w2;
    int   nums = (int(weights_depth)) / 2;
    for(int d = 0; d < nums; ++d)
    {
        vec2 vec2_w = LOAD_UNPACK2_CURRENT_ITEM_HALF(weights_ptr, weights_iter);

        w1         = vec2_w.x;
        vec4 r1[3] = CONVOLVE(src_iter, w1);
        pixels[0] += r1[0];
        pixels[1] += r1[1];
        pixels[2] += r1[2];

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);

        w2         = vec2_w.y;
        vec4 r2[3] = CONVOLVE(src_iter, w2);
        pixels[0] += r2[0];
        pixels[1] += r2[1];
        pixels[2] += r2[2];

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, weights_attrs.stride_z * uint(2));
    }
#else  /* WEIGHTS_OPTIMIZATION */
    float w;
    for(int d = 0; d < int(weights_depth); ++d)
    {
        w = LOAD_UNPACK2_CURRENT_ITEM_HALF(weights_ptr, weights_iter).x;

        vec4 r[3] = CONVOLVE(src_iter, w);
        pixels[0] += r[0];
        pixels[1] += r[1];
        pixels[2] += r[2];

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, weights_attrs.stride_z);
    }
#endif /* WEIGHTS_OPTIMIZATION */

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

    pixels[0] += b;
    pixels[1] += b;
    pixels[2] += b;
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

#if STRIDE_X == 2
#define CONVOLVE(s, w, x1, y1) convolve_stride2(s, w, x1, y1)
#elif STRIDE_X == 1 /* STRIDE_X == 1 */
#define CONVOLVE(s, w, x1, y1) convolve_stride1(s, w, x1, y1)
#else /* STRIDE_X not equals 1 or 2 */
#error STRIDE_X larger than 2 is not supported
#endif /* STRIDE_X == 2 */

vec4[2] convolve_stride1(ImageIterator src_iter, float w, int x1, int y1)
{
    vec4 s[2];
    s[0] = LOAD_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, x1, y1));
    s[1] = LOAD_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, x1, (y1 + int(STRIDE_Y))));

    s[0] *= w;
    s[1] *= w;

    return s;
}

vec4[2] convolve_stride2(ImageIterator src_iter, float w, int x1, int y1)
{
    vec4 s1[2];
    vec4 s2[2];
    vec4 r[2];

    s1[0] = LOAD_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, x1, y1));
    s1[1] = LOAD_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, (4 + x1), y1));
    r[0]  = vec4(s1[0].xz, s1[1].xz);

    s2[0] = LOAD_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, x1, (y1 + int(STRIDE_Y))));
    s2[1] = LOAD_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, (4 + x1), (y1 + int(STRIDE_Y))));
    r[1]  = vec4(s2[0].xz, s2[1].xz);

    r[0] *= w;
    r[1] *= w;

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
    vec4 pixels1[2];
    pixels[0]  = vec4(0.f);
    pixels[1]  = vec4(0.f);
    pixels1[0] = vec4(0.f);
    pixels1[1] = vec4(0.f);

    uint z_index = gl_GlobalInvocationID.z;
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, z_index * weights_stride_w);

#ifdef WEIGHTS_OPTIMIZATION
    float w1, w2;
    int   nums = (int(weights_depth)) / 2;
    for(int d = 0; d < nums; ++d)
    {
        vec2 vec2_w = LOAD_UNPACK2_CURRENT_ITEM_HALF(weights_ptr, weights_iter);

        w1         = vec2_w.x;
        vec4 r1[2] = CONVOLVE(src_iter, w1, 0, 0);
        vec4 r2[2] = CONVOLVE(src_iter, w1, 0, (2 * int(STRIDE_Y)));
        pixels[0] += r1[0];
        pixels[1] += r1[1];
        pixels1[0] += r2[0];
        pixels1[1] += r2[1];

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);

        w2         = vec2_w.y;
        vec4 r3[2] = CONVOLVE(src_iter, w2, 0, 0);
        vec4 r4[2] = CONVOLVE(src_iter, w2, 0, (2 * int(STRIDE_Y)));
        pixels[0] += r3[0];
        pixels[1] += r3[1];
        pixels1[0] += r4[0];
        pixels1[1] += r4[1];

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, weights_attrs.stride_z * uint(2));
    }
#else  /* WEIGHTS_OPTIMIZATION */
    float w;
    for(int d = 0; d < int(weights_depth); ++d)
    {
        w = LOAD_UNPACK2_CURRENT_ITEM_HALF(weights_ptr, weights_iter).x;

        vec4 r1[2] = CONVOLVE(src_iter, w, 0, 0);
        vec4 r2[2] = CONVOLVE(src_iter, w, 0, (2 * int(STRIDE_Y)));
        pixels[0] += r1[0];
        pixels[1] += r1[1];
        pixels1[0] += r2[0];
        pixels1[1] += r2[1];

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, weights_attrs.stride_z);
    }
#endif /* WEIGHTS_OPTIMIZATION */

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

    pixels[0] += b;
    pixels[1] += b;
    pixels1[0] += b;
    pixels1[1] += b;
#endif /* BIAS */

    STORE_PACK4_CURRENT_ITEM_HALF(dst_ptr, dst_iter, pixels[0]);
    STORE_PACK4_HALF(dst_ptr, TENSOR3D_OFFSET(dst_iter, 0, 1, 0), pixels[1]);
    STORE_PACK4_HALF(dst_ptr, TENSOR3D_OFFSET(dst_iter, 0, 2, 0), pixels1[0]);
    STORE_PACK4_HALF(dst_ptr, TENSOR3D_OFFSET(dst_iter, 0, 3, 0), pixels1[1]);
}
#elif defined(PROCESS_4X_2Y_2Z)
TENSOR_DECLARATION(1, srcBuffer, uvec2, src_ptr, src_shift, 3, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec2, dst_ptr, dst_shift, 3, writeonly);
TENSOR_DECLARATION(3, weightsBuffer, uint, weights_ptr, weights_shift, 2, readonly);
#ifdef BIAS
TENSOR_DECLARATION(4, biasesBuffer, uint, biases_ptr, biases_shift, 2, readonly);
#endif /* BIAS */

#if STRIDE_X == 2
#define CONVOLVE(s, w) convolve_stride2(s, w)
#elif STRIDE_X == 1 /* STRIDE_X == 1 */
#define CONVOLVE(s, w) convolve_stride1(s, w)
#else /* STRIDE_X not equals 1 or 2 */
#error STRIDE_X larger than 2 is not supported
#endif /* STRIDE_X == 2 */

vec4[2] convolve_stride1(ImageIterator src_iter, float w)
{
    vec4 s[2];
    s[0] = LOAD_UNPACK4_CURRENT_ITEM_HALF(src_ptr, src_iter);
    s[1] = LOAD_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, int(STRIDE_Y)));

    s[0] *= w;
    s[1] *= w;

    return s;
}

vec4[2] convolve_stride2(ImageIterator src_iter, float w)
{
    vec4 s1[2];
    vec4 s2[2];
    vec4 r[2];

    s1[0] = LOAD_UNPACK4_CURRENT_ITEM_HALF(src_ptr, src_iter);
    s1[1] = LOAD_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, 4, 0));
    r[0]  = vec4(s1[0].xz, s1[1].xz);

    s2[0] = LOAD_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, int(STRIDE_Y)));
    s2[1] = LOAD_UNPACK4_HALF(src_ptr, IMAGE_OFFSET(src_iter, 4, int(STRIDE_Y)));
    r[1]  = vec4(s2[0].xz, s2[1].xz);

    r[0] *= w;
    r[1] *= w;

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

    uint z_base_index = uint(gl_GlobalInvocationID.z) << uint(1);

    // store orginal src current offset
    int s_offset_in_bytes = src_iter.current_offset_in_bytes;

    TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, z_base_index * weights_stride_w);

    for(int z = 0; z < 2; ++z)
    {
        uint z_index = z_base_index + uint(z);

        src_iter.current_offset_in_bytes = s_offset_in_bytes;

        vec4 pixels[2];
        pixels[0] = vec4(0.f);
        pixels[1] = vec4(0.f);

#ifdef WEIGHTS_OPTIMIZATION
        float w1, w2;
        int   nums = (int(weights_depth)) / 2;
        for(int d = 0; d < nums; ++d)
        {
            vec2 vec2_w = LOAD_UNPACK2_CURRENT_ITEM_HALF(weights_ptr, weights_iter);

            w1         = vec2_w.x;
            vec4 r1[2] = CONVOLVE(src_iter, w1);
            pixels[0] += r1[0];
            pixels[1] += r1[1];

            TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);

            w2         = vec2_w.y;
            vec4 r2[2] = CONVOLVE(src_iter, w2);
            pixels[0] += r2[0];
            pixels[1] += r2[1];

            TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);
            TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, weights_attrs.stride_z * uint(2));
        }
#else  /* WEIGHTS_OPTIMIZATION */
        float w;
        for(int d = 0; d < int(weights_depth); ++d)
        {
            w = LOAD_UNPACK2_CURRENT_ITEM_HALF(weights_ptr, weights_iter).x;

            vec4 r[2] = CONVOLVE(src_iter, w);
            pixels[0] += r[0];
            pixels[1] += r[1];

            TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);
            TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, weights_attrs.stride_z);
        }
#endif /* WEIGHTS_OPTIMIZATION */

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

        pixels[0] += b;
        pixels[1] += b;
#endif /* BIAS */

        STORE_PACK4_CURRENT_ITEM_HALF(dst_ptr, dst_iter, pixels[0]);
        STORE_PACK4_HALF(dst_ptr, TENSOR3D_OFFSET(dst_iter, 0, 1, 0), pixels[1]);

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(dst_iter, dst_attrs.stride_z);
    }
}
#elif defined(PROCESS_8X_1Y_1Z)
TENSOR_DECLARATION(1, srcBuffer, uvec4, src_ptr, src_shift, 4, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec4, dst_ptr, dst_shift, 4, writeonly);
TENSOR_DECLARATION(3, weightsBuffer, uint, weights_ptr, weights_shift, 2, readonly);
#ifdef BIAS
TENSOR_DECLARATION(4, biasesBuffer, uint, biases_ptr, biases_shift, 2, readonly);
#endif /* BIAS */

#if STRIDE_X == 2
#define CONVOLVE(s, w) convolve_stride2(s, w)
#elif STRIDE_X == 1 /* STRIDE_X == 1 */
#define CONVOLVE(s, w) convolve_stride1(s, w)
#else /* STRIDE_X not equals 1 or 2 */
#error STRIDE_X larger than 2 is not supported
#endif /* STRIDE_X == 2 */

vec4[2] convolve_stride1(ImageIterator src_iter, float w)
{
    vec4 s[2];
    s = LOAD_UNPACK8_CURRENT_ITEM_HALF(src_ptr, src_iter);

    s[0] *= w;
    s[1] *= w;

    return s;
}

vec4[2] convolve_stride2(ImageIterator src_iter, float w)
{
    vec4 s1[2];
    vec4 s2[2];
    vec4 r[2];

    s1   = LOAD_UNPACK8_CURRENT_ITEM_HALF(src_ptr, src_iter);
    r[0] = vec4(s1[0].xz, s1[1].xz);
    s2   = LOAD_UNPACK8_HALF(src_ptr, IMAGE_OFFSET(src_iter, 8, 0));
    r[1] = vec4(s2[0].xz, s2[1].xz);

    r[0] *= w;
    r[1] *= w;

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
    pixels[0] = vec4(0.f);
    pixels[1] = vec4(0.f);

    uint z_index = gl_GlobalInvocationID.z;
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, z_index * weights_stride_w);

#ifdef WEIGHTS_OPTIMIZATION
    float w1, w2;
    int   nums = (int(weights_depth)) / 2;
    for(int d = 0; d < nums; ++d)
    {
        vec2 vec2_w = LOAD_UNPACK2_CURRENT_ITEM_HALF(weights_ptr, weights_iter);

        w1         = vec2_w.x;
        vec4 r1[2] = CONVOLVE(src_iter, w1);
        pixels[0] += r1[0];
        pixels[1] += r1[1];

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);

        w2         = vec2_w.y;
        vec4 r2[2] = CONVOLVE(src_iter, w2);
        pixels[0] += r2[0];
        pixels[1] += r2[1];

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, weights_attrs.stride_z * uint(2));
    }
#else  /* WEIGHTS_OPTIMIZATION */
    float w;
    for(int d = 0; d < int(weights_depth); ++d)
    {
        w = LOAD_UNPACK2_CURRENT_ITEM_HALF(weights_ptr, weights_iter).x;

        vec4 r[2] = CONVOLVE(src_iter, w);
        pixels[0] += r[0];
        pixels[1] += r[1];

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, weights_attrs.stride_z);
    }
#endif /* WEIGHTS_OPTIMIZATION */

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

    pixels[0] += b;
    pixels[1] += b;
#endif /* BIAS */

    STORE_PACK8_CURRENT_ITEM_HALF(dst_ptr, dst_iter, pixels);
}
#elif defined(PROCESS_8X_2Y_1Z)
TENSOR_DECLARATION(1, srcBuffer, uvec4, src_ptr, src_shift, 4, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec4, dst_ptr, dst_shift, 4, writeonly);
TENSOR_DECLARATION(3, weightsBuffer, uint, weights_ptr, weights_shift, 2, readonly);
#ifdef BIAS
TENSOR_DECLARATION(4, biasesBuffer, uint, biases_ptr, biases_shift, 2, readonly);
#endif /* BIAS */

#if STRIDE_X == 2
#define CONVOLVE(s, w, x1, y1) convolve_stride2(s, w, x1, y1)
#elif STRIDE_X == 1 /* STRIDE_X == 1 */
#define CONVOLVE(s, w, x1, y1) convolve_stride1(s, w, x1, y1)
#else /* STRIDE_X not equals 1 or 2 */
#error STRIDE_X larger than 2 is not supported
#endif /* STRIDE_X == 2 */

vec4[2] convolve_stride1(ImageIterator src_iter, float w, int x1, int y1)
{
    vec4 s[2];
    s = LOAD_UNPACK8_HALF(src_ptr, IMAGE_OFFSET(src_iter, x1, y1));

    s[0] *= w;
    s[1] *= w;

    return s;
}

vec4[2] convolve_stride2(ImageIterator src_iter, float w, int x1, int y1)
{
    vec4 s1[2];
    vec4 s2[2];
    vec4 r[2];

    s1   = LOAD_UNPACK8_HALF(src_ptr, IMAGE_OFFSET(src_iter, x1, y1));
    r[0] = vec4(s1[0].xz, s1[1].xz);
    s2   = LOAD_UNPACK8_HALF(src_ptr, IMAGE_OFFSET(src_iter, (8 + x1), y1));
    r[1] = vec4(s2[0].xz, s2[1].xz);

    r[0] *= w;
    r[1] *= w;

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
    vec4 pixels1[2];
    pixels[0]  = vec4(0.f);
    pixels[1]  = vec4(0.f);
    pixels1[0] = vec4(0.f);
    pixels1[1] = vec4(0.f);

    uint z_index = gl_GlobalInvocationID.z;
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, z_index * weights_stride_w);

#ifdef WEIGHTS_OPTIMIZATION
    float w1, w2;
    int   nums = (int(weights_depth)) / 2;
    for(int d = 0; d < nums; ++d)
    {
        vec2 vec2_w = LOAD_UNPACK2_CURRENT_ITEM_HALF(weights_ptr, weights_iter);

        w1         = vec2_w.x;
        vec4 r1[2] = CONVOLVE(src_iter, w1, 0, 0);
        vec4 r2[2] = CONVOLVE(src_iter, w1, 0, (int(STRIDE_Y)));
        pixels[0] += r1[0];
        pixels[1] += r1[1];
        pixels1[0] += r2[0];
        pixels1[1] += r2[1];

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);

        w2         = vec2_w.y;
        vec4 r3[2] = CONVOLVE(src_iter, w2, 0, 0);
        vec4 r4[2] = CONVOLVE(src_iter, w2, 0, (int(STRIDE_Y)));
        pixels[0] += r3[0];
        pixels[1] += r3[1];
        pixels1[0] += r4[0];
        pixels1[1] += r4[1];

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, weights_attrs.stride_z * uint(2));
    }
#else  /* WEIGHTS_OPTIMIZATION */
    float w;
    for(int d = 0; d < int(weights_depth); ++d)
    {
        w = LOAD_UNPACK2_CURRENT_ITEM_HALF(weights_ptr, weights_iter).x;

        vec4 r1[2] = CONVOLVE(src_iter, w, 0, 0);
        vec4 r2[2] = CONVOLVE(src_iter, w, 0, (int(STRIDE_Y)));
        pixels[0] += r1[0];
        pixels[1] += r1[1];
        pixels1[0] += r2[0];
        pixels1[1] += r2[1];

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, weights_attrs.stride_z);
    }
#endif /* WEIGHTS_OPTIMIZATION */

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

    pixels[0] += b;
    pixels[1] += b;
    pixels1[0] += b;
    pixels1[1] += b;
#endif /* BIAS */

    STORE_PACK8_CURRENT_ITEM_HALF(dst_ptr, dst_iter, pixels);
    STORE_PACK8_HALF(dst_ptr, TENSOR3D_OFFSET(dst_iter, 0, 1, 0), pixels1);
}
#endif /* PROCESS_4X_1Y_1Z */
#else  /* DATA_TYPE_FP32 */
#error Data type not supported
#endif /* DATA_TYPE_FP32 */
