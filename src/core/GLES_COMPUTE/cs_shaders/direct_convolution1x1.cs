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

#if defined(DATA_TYPE_FP32)
precision highp float;

BUFFER_DECLARATION(src, 1, float, readonly);
BUFFER_DECLARATION(dst, 2, float, writeonly);
BUFFER_DECLARATION(weights, 3, float, readonly);
#ifdef BIAS
BUFFER_DECLARATION(biases, 4, float, readonly);
#endif /* BIAS */

/** This kernel performs a direct convolution to convolve the low three dimensions.
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_FP32"
 * @note The convolution stride x must be passed at compile time using "#define STRIDE_X" e.g. "#define STRIDE_X 1"
 * @note In case biases will be added to the convolution "#define HAS_BIAS" has to be passed to append the final matrix with 1 in each row.
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
 * @param[out] weights_ptr                           Pointer to the weights tensor. Supported data types: same as @p src_ptr
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
    float temp;
    float temp_weight;

    for(int d = 0; d < int(weights_depth); ++d)
    {
        temp        = LOAD4(src, CURRENT_OFFSET(src));
        temp_weight = LOAD4(weights, CURRENT_OFFSET(weights));
        pixels += temp * temp_weight;

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

BUFFER_DECLARATION(src, 1, uvec4, readonly);
BUFFER_DECLARATION(dst, 2, uvec4, writeonly);
BUFFER_DECLARATION(weights, 3, uint, readonly);
#ifdef BIAS
BUFFER_DECLARATION(biases, 4, uint, readonly);
#endif /* BIAS */

#if STRIDE_X == 2
#define CONVOLVE(s, w) convolve_stride2(s, w)
#elif STRIDE_X == 1 /* STRIDE_X == 1 */
#define CONVOLVE(s, w) convolve_stride1(s, w)
#else /* STRIDE_X not equals 1 or 2 */
#error STRIDE_X larger than 2 is not supported
#endif /* STRIDE_X == 2 */

vec4[2] convolve_stride1(Image src, float w)
{
    uvec4 packed_s;
    vec4  s[2];

    GC_LOAD1_2D_OFFSET(packed_s, src, 0, 0);

    s[0] = vec4(unpackHalf2x16(packed_s.x), unpackHalf2x16(packed_s.y));
    s[1] = vec4(unpackHalf2x16(packed_s.z), unpackHalf2x16(packed_s.w));

    s[0] *= w;
    s[1] *= w;

    return s;
}

vec4[2] convolve_stride2(Image src, float w)
{
    uvec4 packed_s;
    vec4  s[2];
    vec4  r[2];

    GC_LOAD1_2D_OFFSET(packed_s, src, 0, 0);
    s[0] = vec4(unpackHalf2x16(packed_s.x), unpackHalf2x16(packed_s.y));
    s[1] = vec4(unpackHalf2x16(packed_s.z), unpackHalf2x16(packed_s.w));

    r[0] = vec4(s[0].xz, s[1].xz);

    GC_LOAD1_2D_OFFSET(packed_s, src, 8, 0);
    s[0] = vec4(unpackHalf2x16(packed_s.x), unpackHalf2x16(packed_s.y));
    s[1] = vec4(unpackHalf2x16(packed_s.z), unpackHalf2x16(packed_s.w));

    r[1] = vec4(s[0].xz, s[1].xz);

    r[0] *= w;
    r[1] *= w;

    return r;
}

/** This kernel performs a direct convolution to convolve the low three dimensions.
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_FP16"
 * @note The convolution stride x must be passed at compile time using "#define STRIDE_X" e.g. "#define STRIDE_X 1"
 * @note In case biases will be added to the convolution "#define HAS_BIAS" has to be passed to append the final matrix with 1 in each row.
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
 * @param[out] weights_ptr                           Pointer to the weights tensor. Supported data types: same as @p src_ptr
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

    vec4 pixels[2];
    pixels[0] = vec4(0.f);
    pixels[1] = vec4(0.f);

    uint z_index = gl_GlobalInvocationID.z;

    weights.current_offset += z_index * weights_stride_w;

    uint  packed_w;
    float w;

    for(int d = 0; d < int(weights_depth); ++d)
    {
        GC_LOAD1_3D_OFFSET(packed_w, weights, 0, 0, 0);
        w = unpackHalf2x16(packed_w).x;

        vec4 r[2] = CONVOLVE(src, w);
        pixels[0] += r[0];
        pixels[1] += r[1];

        src.current_offset += src_stride_z;
        weights.current_offset += weights_stride_z;
    }

#ifdef BIAS
    uint  packed_b;
    float b;

    GC_LOAD1_1D_OFFSET(packed_b, biases, z_index);

    if(z_index % uint(2) == uint(0))
    {
        b = unpackHalf2x16(packed_b).x;
    }
    else
    {
        b = unpackHalf2x16(packed_b).y;
    }

    pixels[0] += vec4(b);
    pixels[1] += vec4(b);
#endif /* BIAS */

    uvec4 packed_d;
    packed_d = uvec4(packHalf2x16(pixels[0].xy), packHalf2x16(pixels[0].zw),
                     packHalf2x16(pixels[1].xy), packHalf2x16(pixels[1].zw));
    GC_STORE1_3D_OFFSET(packed_d, dst, 0, 0, 0);
}
#else  /* DATA_TYPE_FP32 */
#error Data type not supported
#endif /* DATA_TYPE_FP32 */
