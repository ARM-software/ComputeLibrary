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

layout(std140) uniform shader_params
{
    TENSOR3D_PARAM_DECLARATION(src);
    TENSOR3D_PARAM_DECLARATION(dst);
    TENSOR3D_PARAM_DECLARATION(weights);
#ifdef BIAS
    VECTOR_PARAM_DECLARATION(biases);
#endif /* BIAS */
    uint weights_stride_w;
    uint weights_depth;
};

#ifdef DATA_TYPE_FP32

precision highp float;

BUFFER_DECLARATION(src, 1, float, readonly);
BUFFER_DECLARATION(dst, 2, float, writeonly);
BUFFER_DECLARATION(weights, 3, float, readonly);
#ifdef BIAS
BUFFER_DECLARATION(biases, 4, float, readonly);
#endif /* BIAS */

#define LOAD20(r, name, offset)           \
    r[0] = LOAD4(name, offset);           \
    r[1] = LOAD4(name, offset + uint(1)); \
    r[2] = LOAD4(name, offset + uint(2)); \
    r[3] = LOAD4(name, offset + uint(3)); \
    r[4] = LOAD4(name, offset + uint(4))

/** This kernel performs a direct convolution to convolve the low three dimensions.
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_FP32"
 * @note If biases are used then "define HAS_BIAS" has to be passed at compile time
 *
 * @param[in]  src_ptr                               Pointer to the source tensor. Supported data types: F32
 * @param[in]  src_stride_x                          Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                          Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                            src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes     The offset of the first element in the source tensor
 * @param[out] dst_ptr                               Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                          Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                            dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                          Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                            dst_stride_y * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_z                          Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                            dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
 * @param[in]  weights_ptr                           Pointer to the weights tensor. Supported data types: same as @p src_ptr
 * @param[in]  weights_stride_x                      Stride of the weights tensor in X dimension (in bytes)
 * @param[in]  weights_step_x                        weights_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  weights_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
 * @param[in]  weights_step_y                        weights_stride_y * number of elements along y processed per workitem(in bytes)
 * @param[in]  weights_stride_z                      Stride of the weights tensor in Z dimension (in bytes)
 * @param[in]  weights_step_z                        weights_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  weights_offset_first_element_in_bytes The offset of the first element in the weights tensor
 * @param[in]  biases_ptr                            Pointer to the biases tensor. Same as @p src_ptr
 * @param[in]  biases_stride_x                       Stride of the biases tensor in X dimension (in bytes)
 * @param[in]  biases_step_x                         biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  biases_offset_first_element_in_bytes  The offset of the first element in the biases tensor
 * @param[in]  weights_stride_w                      Stride of the weights tensor in the 4th dimension
 * @param[in]  weights_depth                         The third dimensions of the weights tensors
 */
void main()
{
    Image    src     = CONVERT_TO_IMAGE_STRUCT(src);
    Tensor3D weights = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(weights);
    Tensor3D dst     = CONVERT_TO_TENSOR3D_STRUCT(dst);

#ifdef BIAS
    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);
#endif /* BIAS */

    float pixels  = CONVERT(0, float);
    uint  z_index = gl_GlobalInvocationID.z;
    weights.current_offset += z_index * weights_stride_w >> 2;
    float temp[5];
    float temp_weight[5];

    for(int d = 0; d < int(weights_depth); ++d)
    {
        LOAD20(temp, src, offset(src, 0, 0));
        LOAD20(temp_weight, weights, tensor3D_offset(weights, 0, 0, 0));
        pixels += temp[0] * temp_weight[0] + temp[1] * temp_weight[1] + temp[2] * temp_weight[2] + temp[3] * temp_weight[3] + temp[4] * temp_weight[4];

        LOAD20(temp, src, offset(src, 0, 1));
        LOAD20(temp_weight, weights, tensor3D_offset(weights, 0, 1, 0));
        pixels += temp[0] * temp_weight[0] + temp[1] * temp_weight[1] + temp[2] * temp_weight[2] + temp[3] * temp_weight[3] + temp[4] * temp_weight[4];

        LOAD20(temp, src, offset(src, 0, 2));
        LOAD20(temp_weight, weights, tensor3D_offset(weights, 0, 2, 0));
        pixels += temp[0] * temp_weight[0] + temp[1] * temp_weight[1] + temp[2] * temp_weight[2] + temp[3] * temp_weight[3] + temp[4] * temp_weight[4];

        LOAD20(temp, src, offset(src, 0, 3));
        LOAD20(temp_weight, weights, tensor3D_offset(weights, 0, 3, 0));
        pixels += temp[0] * temp_weight[0] + temp[1] * temp_weight[1] + temp[2] * temp_weight[2] + temp[3] * temp_weight[3] + temp[4] * temp_weight[4];

        LOAD20(temp, src, offset(src, 0, 4));
        LOAD20(temp_weight, weights, tensor3D_offset(weights, 0, 4, 0));
        pixels += temp[0] * temp_weight[0] + temp[1] * temp_weight[1] + temp[2] * temp_weight[2] + temp[3] * temp_weight[3] + temp[4] * temp_weight[4];

        src.current_offset += (src_stride_z >> 2);
        weights.current_offset += (weights_stride_z >> 2);
    }

#ifdef BIAS
    pixels += LOAD4(biases, vector_offset(biases, int(z_index)));
#endif /* BIAS */

    STORE4(dst, CURRENT_OFFSET(dst), pixels);
}

#elif defined(DATA_TYPE_FP16)

precision mediump float;

BUFFER_DECLARATION(src, 1, uvec2, readonly);
BUFFER_DECLARATION(dst, 2, uvec2, writeonly);
BUFFER_DECLARATION(weights, 3, uint, readonly);
#ifdef BIAS
BUFFER_DECLARATION(biases, 4, uint, readonly);
#endif /* BIAS */

#if STRIDE_X == 1
#define LOAD_SRC(src, row) load_src_stride1(src, row)
#define CONVOLVE1x5(src, weight) convolve1x5_stride1(src, weight)
#elif STRIDE_X == 2 /* STRIDE_X == 1 */
#define LOAD_SRC(src, row) load_src_stride2(src, row)
#define CONVOLVE1x5(src, weight) convolve1x5_stride2(src, weight)
#else /* STRDIDE_X == 1 */
#error STRIDE_X larger than 2 is not supported
#endif /* STRIDE_X == 1 */

vec4[2] load_src_stride1(Image src, int row)
{
    uvec2 packed[2];
    vec4  ret[2];

    GC_LOAD2_2D_OFFSET(packed, src, 0, row);

    ret[0] = vec4(unpackHalf2x16(packed[0].x), unpackHalf2x16(packed[0].y));
    ret[1] = vec4(unpackHalf2x16(packed[1].x), unpackHalf2x16(packed[1].y));

    return ret;
}

vec4[3] load_src_stride2(Image src, int row)
{
    uvec2 packed[3];
    vec4  ret[3];

    GC_LOAD3_2D_OFFSET(packed, src, 0, row);

    ret[0] = vec4(unpackHalf2x16(packed[0].x), unpackHalf2x16(packed[0].y));
    ret[1] = vec4(unpackHalf2x16(packed[1].x), unpackHalf2x16(packed[1].y));
    ret[2] = vec4(unpackHalf2x16(packed[2].x), unpackHalf2x16(packed[2].y));

    return ret;
}

vec2[3] load_weight(Tensor3D weights, int row)
{
    uvec3 packed_w;
    vec2  ret[3];

    GC_LOAD3_3D_OFFSET(packed_w, weights, 0, row, 0);

    ret[0] = vec2(unpackHalf2x16(packed_w[0]));
    ret[1] = vec2(unpackHalf2x16(packed_w[1]));
    ret[2] = vec2(unpackHalf2x16(packed_w[2]));

    return ret;
}

// output 4 element per thread
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

/** This kernel performs a direct convolution to convolve the low three dimensions.
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_FP16"
 * @note If biases are used then "define HAS_BIAS" has to be passed at compile time
 *
 * @param[in]  src_ptr                               Pointer to the source tensor. Supported data types: F16
 * @param[in]  src_stride_x                          Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                          Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                            src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes     The offset of the first element in the source tensor
 * @param[out] dst_ptr                               Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                          Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                            dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                          Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                            dst_stride_y * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_z                          Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                            dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
 * @param[in]  weights_ptr                           Pointer to the weights tensor. Supported data types: same as @p src_ptr
 * @param[in]  weights_stride_x                      Stride of the weights tensor in X dimension (in bytes)
 * @param[in]  weights_step_x                        weights_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  weights_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
 * @param[in]  weights_step_y                        weights_stride_y * number of elements along y processed per workitem(in bytes)
 * @param[in]  weights_stride_z                      Stride of the weights tensor in Z dimension (in bytes)
 * @param[in]  weights_step_z                        weights_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  weights_offset_first_element_in_bytes The offset of the first element in the weights tensor
 * @param[in]  biases_ptr                            Pointer to the biases tensor. Same as @p src_ptr
 * @param[in]  biases_stride_x                       Stride of the biases tensor in X dimension (in bytes)
 * @param[in]  biases_step_x                         biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  biases_offset_first_element_in_bytes  The offset of the first element in the biases tensor
 * @param[in]  weights_stride_w                      Stride of the weights tensor in the 4th dimension
 * @param[in]  weights_depth                         The third dimensions of the weights tensors
 */
void main()
{
    Image    src     = GC_CONVERT_TO_IMAGE_STRUCT(src);
    Tensor3D weights = GC_CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(weights);
    Tensor3D dst     = GC_CONVERT_TO_TENSOR3D_STRUCT(dst);

#ifdef BIAS
    Vector   biases  = GC_CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);
#endif /* BIAS */

    vec4  res = vec4(0);
    vec2  w[3];
    vec4  s[STRIDE_X + 1];
    uvec2 packed_d;
    uint  z_index = gl_GlobalInvocationID.z;

    weights.current_offset += z_index * weights_stride_w;

    for(int d = 0; d < int(weights_depth); ++d)
    {
        for(int row = 0; row < 5; row++)
        {
            w = load_weight(weights, row);
            s = LOAD_SRC(src, row);
            res += CONVOLVE1x5(s, w);
        }

        src.current_offset += src_stride_z;
        weights.current_offset += weights_stride_z;
    }

#ifdef BIAS
    uint  packed_b;
    float b;

    GC_LOAD1_1D_OFFSET(packed_b, biases, z_index);
    b = (z_index % uint(2) == uint(0)) ? unpackHalf2x16(packed_b).x : unpackHalf2x16(packed_b).y;
    res += vec4(b);
#endif /* BIAS */

    packed_d = uvec2(packHalf2x16(res.xy), packHalf2x16(res.zw));
    GC_STORE1_3D_OFFSET(packed_d, dst, 0, 0, 0);
}

#else /* DATA_TYPE_FP16 */
#error Data type not supported
#endif /* DATA_TYPE_FP16 */
