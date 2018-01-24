/*
 * Copyright (c) 2017, 2018 ARM Limited.
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

/** This kernel performs a direct convolution to convolve the low three dimensions
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_NAME". e.g. "#define DATA_TYPE_FP32"
 * @note This kernel has multiple optimized direct convolution options for FP16.
 *       The direct convolution option must be passed at compile time using "#define PROCESS_nX_nY_nZ" e.g. "#define PROCESS_8X_1Y_1Z"
 * @note The convolution stride x must be passed at compile time using "#define STRIDE_X n" e.g. "#define STRIDE_X 1"
 *       This OpenGL ES shader works with stride_x = 1 and 2
 * @note If biases are used then "define HAS_BIAS" has to be passed at compile time
 *
 * @param[in]  src_ptr          Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  src_attrs        The attributes of the source tensor
 * @param[out] dst_ptr          Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_attrs        The attributes of the destination tensor
 * @param[out] weights_ptr      Pointer to the weights tensor. Supported data types: same as @p src_ptr
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

#ifdef DATA_TYPE_FP32
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

    float temp[5];
    float temp_weight[5];
    for(int d = 0; d < int(weights_depth); ++d)
    {
        temp        = VLOAD5(float[5], src_ptr, IMAGE_OFFSET(src_iter, 0, 0));
        temp_weight = VLOAD5(float[5], weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 0, 0));
        pixels += temp[0] * temp_weight[0] + temp[1] * temp_weight[1] + temp[2] * temp_weight[2] + temp[3] * temp_weight[3] + temp[4] * temp_weight[4];

        temp        = VLOAD5(float[5], src_ptr, IMAGE_OFFSET(src_iter, 0, 1));
        temp_weight = VLOAD5(float[5], weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 1, 0));
        pixels += temp[0] * temp_weight[0] + temp[1] * temp_weight[1] + temp[2] * temp_weight[2] + temp[3] * temp_weight[3] + temp[4] * temp_weight[4];

        temp        = VLOAD5(float[5], src_ptr, IMAGE_OFFSET(src_iter, 0, 2));
        temp_weight = VLOAD5(float[5], weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 2, 0));
        pixels += temp[0] * temp_weight[0] + temp[1] * temp_weight[1] + temp[2] * temp_weight[2] + temp[3] * temp_weight[3] + temp[4] * temp_weight[4];

        temp        = VLOAD5(float[5], src_ptr, IMAGE_OFFSET(src_iter, 0, 3));
        temp_weight = VLOAD5(float[5], weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 3, 0));
        pixels += temp[0] * temp_weight[0] + temp[1] * temp_weight[1] + temp[2] * temp_weight[2] + temp[3] * temp_weight[3] + temp[4] * temp_weight[4];

        temp        = VLOAD5(float[5], src_ptr, IMAGE_OFFSET(src_iter, 0, 4));
        temp_weight = VLOAD5(float[5], weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, 4, 0));
        pixels += temp[0] * temp_weight[0] + temp[1] * temp_weight[1] + temp[2] * temp_weight[2] + temp[3] * temp_weight[3] + temp[4] * temp_weight[4];

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, weights_attrs.stride_z);
    }

#ifdef BIAS
    pixels += LOAD(biases_ptr, VECTOR_OFFSET(biases_iter, z_index));
#endif /* BIAS */

    STORE_CURRENT_ITEM(dst_ptr, dst_iter, pixels);
}
#elif defined(DATA_TYPE_FP16)

// Common definitions for DATA_TYPE_FP16
#if STRIDE_X == 1
#define LOAD_SRC_AT_ROW(row) VLOAD2_UNPACK8_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, row))
#define CONVOLVE1x5(src, weight) convolve1x5_stride1(src, weight)
#elif STRIDE_X == 2 /* STRIDE_X == 1 */
#define LOAD_SRC_AT_ROW(row) VLOAD3_UNPACK12_HALF(src_ptr, IMAGE_OFFSET(src_iter, 0, row))
#define CONVOLVE1x5(src, weight) convolve1x5_stride2(src, weight)
#else /* STRDIDE_X == 1 */
#error STRIDE_X larger than 2 is not supported
#endif /* STRIDE_X == 1 */

#define LOAD_WEIGHT_AT_ROW(row) VLOAD3_UNPACK6_HALF(weights_ptr, TENSOR3D_OFFSET(weights_iter, 0, row, 0))

vec4 convolve1x5_stride1(vec4 tmp[2], vec2 w[3])
{
    vec4 src0 = tmp[0];
    vec4 src1 = vec4(tmp[0].yzw, tmp[1].x);
    vec4 src2 = vec4(tmp[0].zw, tmp[1].xy);
    vec4 src3 = vec4(tmp[0].w, tmp[1].xyz);
    vec4 src4 = tmp[1];
    vec4 ret  = src0 * w[0].x + src1 * w[0].y + src2 * w[1].x + src3 * w[1].y + src4 * w[2].x;

    return ret;
}

vec4 convolve1x5_stride2(vec4 tmp[3], vec2 w[3])
{
    vec4 src0 = vec4(tmp[0].xz, tmp[1].xz);
    vec4 src1 = vec4(tmp[0].yw, tmp[1].yw);
    vec4 src2 = vec4(tmp[0].z, tmp[1].xz, tmp[2].x);
    vec4 src3 = vec4(tmp[0].w, tmp[1].yw, tmp[2].y);
    vec4 src4 = vec4(tmp[1].x, tmp[1].z, tmp[2].xz);
    vec4 ret  = src0 * w[0].x + src1 * w[0].y + src2 * w[1].x + src3 * w[1].y + src4 * w[2].x;

    return ret;
}

#if defined(PROCESS_4X_1Y_1Z)
TENSOR_DECLARATION(1, srcBuffer, uvec2, src_ptr, src_shift, 3, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec2, dst_ptr, dst_shift, 3, writeonly);
TENSOR_DECLARATION(3, weightsBuffer, uint, weights_ptr, weights_shift, 2, readonly);
#ifdef BIAS
TENSOR_DECLARATION(4, biasesBuffer, uint, biases_ptr, biases_shift, 2, readonly);
#endif /* BIAS */

void main()
{
    ImageIterator    src_iter     = CONVERT_TO_IMAGE_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator weights_iter = CONVERT_TO_TENSOR3D_ITERATOR_NO_STEP(weights_attrs, weights_shift);
    Tensor3DIterator dst_iter     = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

#ifdef BIAS
    VectorIterator   biases_iter  = CONVERT_TO_VECTOR_ITERATOR_NO_STEP(biases_attrs, biases_shift);
#endif /* BIAS */

    vec4 res = vec4(0);
    vec2 w[3];
    vec4 s[STRIDE_X + 1];

    uint z_index = gl_GlobalInvocationID.z;
    TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, z_index * weights_stride_w);

    for(int d = 0; d < int(weights_depth); ++d)
    {
        for(int row = 0; row < 5; row++)
        {
            w = LOAD_WEIGHT_AT_ROW(row);
            s = LOAD_SRC_AT_ROW(row);
            res += CONVOLVE1x5(s, w);
        }

        TENSOR_ITERATOR_ADVANCE_IN_BYTES(src_iter, src_attrs.stride_z);
        TENSOR_ITERATOR_ADVANCE_IN_BYTES(weights_iter, weights_attrs.stride_z);
    }

#ifdef BIAS
    vec2  vec2_b;
    float b;

    vec2_b = LOAD_UNPACK2_HALF(biases_ptr, VECTOR_OFFSET(biases_iter, z_index));
    b      = (z_index % uint(2) == uint(0)) ? vec2_b.x : vec2_b.y;
    res += vec4(b);
#endif /* BIAS */

    STORE_PACK4_CURRENT_ITEM_HALF(dst_ptr, dst_iter, res);
}

#endif /* PROCESS_nX_nY_nZ */
#else  /* DATA_TYPE_FP32 */
#error Data type not supported
#endif /* DATA_TYPE_FP32 */
