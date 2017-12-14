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

#define LOAD12(r, name, offset)          \
    r.x = LOAD4(name, offset);           \
    r.y = LOAD4(name, offset + uint(1)); \
    r.z = LOAD4(name, offset + uint(2))

#define LOAD3X3(r, name)                                \
    r[0] = LOAD4(name, tensor3D_offset(name, 0, 0, 0)); \
    r[1] = LOAD4(name, tensor3D_offset(name, 1, 0, 0)); \
    r[2] = LOAD4(name, tensor3D_offset(name, 2, 0, 0)); \
    r[3] = LOAD4(name, tensor3D_offset(name, 0, 1, 0)); \
    r[4] = LOAD4(name, tensor3D_offset(name, 1, 1, 0)); \
    r[5] = LOAD4(name, tensor3D_offset(name, 2, 1, 0)); \
    r[6] = LOAD4(name, tensor3D_offset(name, 0, 2, 0)); \
    r[7] = LOAD4(name, tensor3D_offset(name, 1, 2, 0)); \
    r[8] = LOAD4(name, tensor3D_offset(name, 2, 2, 0))

#if defined(PROCESS_1_ELEMENT)
BUFFER_DECLARATION(src, 1, float, readonly);
BUFFER_DECLARATION(dst, 2, float, writeonly);
BUFFER_DECLARATION(weights, 3, float, readonly);
#ifdef BIAS
BUFFER_DECLARATION(biases, 4, float, readonly);
#endif /* BIAS */

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

    float pixels = CONVERT(0, float);

    uint z_index = gl_GlobalInvocationID.z;

    weights.current_offset += z_index * weights_stride_w >> 2;

    for(int d = 0; d < int(weights_depth); ++d)
    {
        vec3 temp;
        vec3 w;

        LOAD12(temp, src, offset(src, 0, 0));
        LOAD12(w, weights, tensor3D_offset(weights, 0, 0, 0));

        pixels += temp.x * w[0] + temp.y * w[1] + temp.z * w[2];

        LOAD12(temp, src, offset(src, 0, 1));
        LOAD12(w, weights, tensor3D_offset(weights, 0, 1, 0));

        pixels += temp.x * w[0] + temp.y * w[1] + temp.z * w[2];

        LOAD12(temp, src, offset(src, 0, 2));
        LOAD12(w, weights, tensor3D_offset(weights, 0, 2, 0));

        pixels += temp.x * w[0] + temp.y * w[1] + temp.z * w[2];

        src.current_offset += src_stride_z >> 2;
        weights.current_offset += weights_stride_z >> 2;
    }

#ifdef BIAS
    pixels += LOAD4(biases, vector_offset(biases, int(z_index)));
#endif /* BIAS */

    STORE4(dst, CURRENT_OFFSET(dst), pixels);
}
#elif defined(PROCESS_8_ELEMENT)
BUFFER_DECLARATION(src, 1, vec4, readonly);
BUFFER_DECLARATION(dst, 2, vec4, writeonly);
BUFFER_DECLARATION(weights, 3, float, readonly);
#ifdef BIAS
BUFFER_DECLARATION(biases, 4, float, readonly);
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

    LOAD3(tmp, src, offset);

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
    vec4 tmp[3];
    vec4 r[2];

    LOAD3(tmp, src, offset);

    left   = vec4(tmp[0].xz, tmp[1].xz);
    middle = vec4(tmp[0].yw, tmp[1].yw);
    right  = vec4(tmp[0].z, tmp[1].xz, tmp[2].x);

    r[0] = left * w[0] + middle * w[1] + right * w[2];

    LOAD2(tmp, src, offset + ((uint(3) * src_stride_x) >> 2));

    left   = vec4(tmp[2].xz, tmp[0].xz);
    middle = vec4(tmp[2].yw, tmp[0].yw);
    right  = vec4(tmp[2].z, tmp[0].xz, tmp[1].x);

    r[1] = left * w[0] + middle * w[1] + right * w[2];

    return r;
}

/** An optimized direct convolution 3x3 OpenGL ES compute shader for process 8 elements at once
 *
 * @note This OpenGL ES shader works with stride_x = 1 and 2
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
    Vector   biases  = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);
#endif /* BIAS */

    vec4 pixels[2];
    pixels[0] = vec4(0);
    pixels[1] = vec4(0);

    uint z_index = gl_GlobalInvocationID.z;

    weights.current_offset += z_index * weights_stride_w >> 2;

    for(int d = 0; d < int(weights_depth); ++d)
    {
        // load 3 weights once
        vec3 w;
        vec4 r[2];

        // first line
        LOAD3(w, weights, tensor3D_offset(weights, 0, 0, 0));

        r = CONVOLVE1x3(src.current_offset >> uint(2), w);
        pixels[0] += r[0];
        pixels[1] += r[1];

        // second line
        LOAD3(w, weights, tensor3D_offset(weights, 0, 1, 0));

        r = CONVOLVE1x3((src.current_offset + (src_stride_y >> 2)) >> uint(2), w);
        pixels[0] += r[0];
        pixels[1] += r[1];

        // third line
        LOAD3(w, weights, tensor3D_offset(weights, 0, 2, 0));

        r = CONVOLVE1x3((src.current_offset + (src_stride_y >> 1)) >> uint(2), w);
        pixels[0] += r[0];
        pixels[1] += r[1];

        src.current_offset += src_stride_z >> 2;
        weights.current_offset += weights_stride_z >> 2;
    }

#ifdef BIAS
    float b;
    LOAD1(b, biases, vector_offset(biases, int(z_index)));
    pixels[0] += vec4(b);
    pixels[1] += vec4(b);
#endif /* BIAS */

    STORE2(dst, dst.current_offset >> uint(2), pixels);
}
#elif defined(PROCESS_4_ELEMENT)
BUFFER_DECLARATION(src, 1, vec4, readonly);
BUFFER_DECLARATION(dst, 2, vec4, writeonly);
BUFFER_DECLARATION(weights, 3, float, readonly);
#ifdef BIAS
BUFFER_DECLARATION(biases, 4, float, readonly);
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

    LOAD2(tmp, src, offset);

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

    LOAD3(tmp, src, offset);

    left   = vec4(tmp[0].xz, tmp[1].xz);
    middle = vec4(tmp[0].yw, tmp[1].yw);
    right  = vec4(tmp[0].z, tmp[1].xz, tmp[2].x);

    tmp[0] = left * w[0] + middle * w[1] + right * w[2];

    return tmp[0];
}

/** An optimized direct convolution 3x3 OpenGL ES compute shader for process 4 elements at once
 *
 * @note This OpenGL ES shader works with stride_x = 1 and 2
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
    Vector   biases  = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);
#endif /* BIAS */

    vec4 pixels;
    pixels = vec4(0);

    uint z_index = gl_GlobalInvocationID.z;

    weights.current_offset += z_index * weights_stride_w >> 2;

    for(int d = 0; d < int(weights_depth); ++d)
    {
        // load 3 weights once
        vec3 w;

        // first line
        LOAD3(w, weights, tensor3D_offset(weights, 0, 0, 0));

        pixels += CONVOLVE1x3(src.current_offset >> uint(2), w);

        // second line
        LOAD3(w, weights, tensor3D_offset(weights, 0, 1, 0));

        pixels += CONVOLVE1x3((src.current_offset + (src_stride_y >> 2)) >> uint(2), w);

        // third line
        LOAD3(w, weights, tensor3D_offset(weights, 0, 2, 0));

        pixels += CONVOLVE1x3((src.current_offset + (src_stride_y >> 1)) >> uint(2), w);

        src.current_offset += src_stride_z >> 2;
        weights.current_offset += weights_stride_z >> 2;
    }

#ifdef BIAS
    float b;
    LOAD1(b, biases, vector_offset(biases, int(z_index)));
    pixels += vec4(b);
#endif /* BIAS */

    STORE1(dst, dst.current_offset >> uint(2), pixels);
}
#elif defined(PROCESS_X_4ELEMENTS_Y_3ELEMENTS)
BUFFER_DECLARATION(src, 1, vec4, readonly);
BUFFER_DECLARATION(dst, 2, vec4, writeonly);
BUFFER_DECLARATION(weights, 3, float, readonly);
#ifdef BIAS
BUFFER_DECLARATION(biases, 4, float, readonly);
#endif /* BIAS */

#define CONVOLVE1x3(left, middle, right, w) convolve1x3_stride1(left, middle, right, w)

vec4 convolve1x3_stride1(vec4 left, vec4 middle, vec4 right, vec3 w)
{
    vec4 r;

    r = left * w[0] + middle * w[1] + right * w[2];

    return r;
}

/** An optimized direct convolution 3x3 OpenGL ES compute shader for process 4x3 elements at once
 *
 * @note This OpenGL ES shader works with stride_x = 1 and 2
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
    Vector   biases  = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);
#endif /* BIAS */

    vec4 pixels[3];
    pixels[0] = vec4(0);
    pixels[1] = vec4(0);
    pixels[2] = vec4(0);

    uint z_index = gl_GlobalInvocationID.z;

    weights.current_offset += z_index * weights_stride_w >> 2;

    for(int d = 0; d < int(weights_depth); ++d)
    {
        // load 3 weights once
        vec3 w[3];

        LOAD3(w[0], weights, tensor3D_offset(weights, 0, 0, 0));
        LOAD3(w[1], weights, tensor3D_offset(weights, 0, 1, 0));
        LOAD3(w[2], weights, tensor3D_offset(weights, 0, 2, 0));

        vec4 s[2];
        vec4 middle;
        vec4 right;
        // first line
        LOAD2(s, src, src.current_offset >> uint(2));
        middle = vec4(s[0].yzw, s[1].x);
        right  = vec4(s[0].zw, s[1].xy);
        pixels[0] += CONVOLVE1x3(s[0], middle, right, w[0]);

        // second line
        LOAD2(s, src, (src.current_offset + (src_stride_y >> 2)) >> uint(2));
        middle = vec4(s[0].yzw, s[1].x);
        right  = vec4(s[0].zw, s[1].xy);
        pixels[0] += CONVOLVE1x3(s[0], middle, right, w[1]);
        pixels[1] += CONVOLVE1x3(s[0], middle, right, w[0]);

        // third line
        LOAD2(s, src, (src.current_offset + (src_stride_y >> 1)) >> uint(2));
        middle = vec4(s[0].yzw, s[1].x);
        right  = vec4(s[0].zw, s[1].xy);
        pixels[0] += CONVOLVE1x3(s[0], middle, right, w[2]);
        pixels[1] += CONVOLVE1x3(s[0], middle, right, w[1]);
        pixels[2] += CONVOLVE1x3(s[0], middle, right, w[0]);

        // forth line
        LOAD2(s, src, (src.current_offset + (uint(3) * (src_stride_y >> 2))) >> uint(2));
        middle = vec4(s[0].yzw, s[1].x);
        right  = vec4(s[0].zw, s[1].xy);
        pixels[1] += CONVOLVE1x3(s[0], middle, right, w[2]);
        pixels[2] += CONVOLVE1x3(s[0], middle, right, w[1]);

        // fifth line
        LOAD2(s, src, (src.current_offset + (src_stride_y)) >> uint(2));
        middle = vec4(s[0].yzw, s[1].x);
        right  = vec4(s[0].zw, s[1].xy);
        pixels[2] += CONVOLVE1x3(s[0], middle, right, w[2]);

        src.current_offset += src_stride_z >> 2;
        weights.current_offset += weights_stride_z >> 2;
    }

#ifdef BIAS
    float b;
    LOAD1(b, biases, vector_offset(biases, int(z_index)));

    pixels[0] += vec4(b);
    pixels[1] += vec4(b);
    pixels[2] += vec4(b);
#endif /* BIAS */

    STORE1(dst, dst.current_offset >> uint(2), pixels[0]);
    STORE1(dst, (dst.current_offset + (dst_stride_y >> 2)) >> uint(2), pixels[1]);
    STORE1(dst, (dst.current_offset + (dst_stride_y >> 1)) >> uint(2), pixels[2]);
}
#elif defined(PROCESS_X_8ELEMENTS_Y_3ELEMENTS_FP16)
precision mediump float;

BUFFER_DECLARATION(src, 1, uvec4, readonly);
BUFFER_DECLARATION(dst, 2, uvec4, writeonly);
BUFFER_DECLARATION(weights, 3, uint, readonly);
#ifdef BIAS
BUFFER_DECLARATION(biases, 4, uint, readonly);
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

vec4[3] load_and_unpack(uint offset)
{
    uvec4 packed_s[2];
    vec4  s[3];

    LOAD1(packed_s[0], src, offset);
    LOAD1(packed_s[1], src, offset + uint(1));
    ;

    s[0] = vec4(unpackHalf2x16(packed_s[0].x), unpackHalf2x16(packed_s[0].y));
    s[1] = vec4(unpackHalf2x16(packed_s[0].z), unpackHalf2x16(packed_s[0].w));
    s[2] = vec4(unpackHalf2x16(packed_s[1].x), unpackHalf2x16(packed_s[1].y));

    return s;
}

/** An optimized direct convolution 3x3 OpenGL ES compute shader for process 8x3 elements at once
 *
 * @note This OpenGL ES shader works with stride_x = 1 and 2
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
    Image    src     = CONVERT_TO_IMAGE_STRUCT_FP16(src);
    Tensor3D weights = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP_FP16(weights);
    Tensor3D dst     = CONVERT_TO_TENSOR3D_STRUCT_FP16(dst);

#ifdef BIAS
    Vector   biases  = CONVERT_TO_VECTOR_STRUCT_NO_STEP_FP16(biases);
#endif /* BIAS */

    uvec2 packed_d[2];
    uvec4 vd;

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

    weights.current_offset += z_index * weights_stride_w;

    for(int d = 0; d < int(weights_depth); ++d)
    {
        // load 3 weights once
        uvec2 packed_w[3];

        LOAD2(packed_w[0], weights, tensor3D_offset_fp16(weights, 0, 0, 0) >> 2);
        LOAD2(packed_w[1], weights, tensor3D_offset_fp16(weights, 0, 1, 0) >> 2);
        LOAD2(packed_w[2], weights, tensor3D_offset_fp16(weights, 0, 2, 0) >> 2);

        vec3 w[3];
        w[0] = vec3(unpackHalf2x16(packed_w[0].x), unpackHalf2x16(packed_w[0].y).x);
        w[1] = vec3(unpackHalf2x16(packed_w[1].x), unpackHalf2x16(packed_w[1].y).x);
        w[2] = vec3(unpackHalf2x16(packed_w[2].x), unpackHalf2x16(packed_w[2].y).x);

        uvec4 packed_s[2];
        vec4  s[3];
        vec4  r[2];
        uint  offset;
        // first line
        offset = src.current_offset >> uint(4);
        s      = load_and_unpack(offset);

        r = CONVOLVE1x3(s, w[0]);
        pixels[0][0] += r[0];
        pixels[0][1] += r[1];

        // second line
        offset = (src.current_offset + src_stride_y) >> uint(4);
        s      = load_and_unpack(offset);

        r = CONVOLVE1x3(s, w[1]);
        pixels[0][0] += r[0];
        pixels[0][1] += r[1];
        r = CONVOLVE1x3(s, w[0]);
        pixels[1][0] += r[0];
        pixels[1][1] += r[1];

        // third line
        offset = (src.current_offset + (src_stride_y << 1)) >> uint(4);
        s      = load_and_unpack(offset);

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
        offset = (src.current_offset + uint(3) * (src_stride_y)) >> uint(4);
        s      = load_and_unpack(offset);

        r = CONVOLVE1x3(s, w[2]);
        pixels[1][0] += r[0];
        pixels[1][1] += r[1];
        r = CONVOLVE1x3(s, w[1]);
        pixels[2][0] += r[0];
        pixels[2][1] += r[1];

        // fifth line
        offset = (src.current_offset + (src_stride_y << 2)) >> uint(4);
        s      = load_and_unpack(offset);

        r = CONVOLVE1x3(s, w[2]);
        pixels[2][0] += r[0];
        pixels[2][1] += r[1];

        src.current_offset += src_stride_z;
        weights.current_offset += weights_stride_z;
    }

#ifdef BIAS
    uint  packed_b;
    float b;
    LOAD1(packed_b, biases, vector_offset_fp16(biases, int(z_index)) >> 2);

    if(z_index % uint(2) == uint(0))
    {
        b = unpackHalf2x16(packed_b).x;
    }
    else
    {
        b = unpackHalf2x16(packed_b).y;
    }

    for(i = 0; i < 3; i++)
    {
        for(j = 0; j < 2; j++)
        {
            pixels[i][j] += vec4(b);
        }
    }
#endif /* BIAS */

    packed_d[0] = uvec2(packHalf2x16(pixels[0][0].xy), packHalf2x16(pixels[0][0].zw));
    packed_d[1] = uvec2(packHalf2x16(pixels[0][1].xy), packHalf2x16(pixels[0][1].zw));
    vd          = uvec4(packed_d[0], packed_d[1]);
    STORE1(dst, dst.current_offset >> uint(4), vd);

    packed_d[0] = uvec2(packHalf2x16(pixels[1][0].xy), packHalf2x16(pixels[1][0].zw));
    packed_d[1] = uvec2(packHalf2x16(pixels[1][1].xy), packHalf2x16(pixels[1][1].zw));
    vd          = uvec4(packed_d[0], packed_d[1]);
    STORE1(dst, (dst.current_offset + dst_stride_y) >> uint(4), vd);

    packed_d[0] = uvec2(packHalf2x16(pixels[2][0].xy), packHalf2x16(pixels[2][0].zw));
    packed_d[1] = uvec2(packHalf2x16(pixels[2][1].xy), packHalf2x16(pixels[2][1].zw));
    vd          = uvec4(packed_d[0], packed_d[1]);
    STORE1(dst, (dst.current_offset + (dst_stride_y << 1)) >> uint(4), vd);
}
#elif defined(PROCESS_X_4ELEMENTS_FP16)
precision mediump float;

BUFFER_DECLARATION(src, 1, uvec2, readonly);
BUFFER_DECLARATION(dst, 2, uvec2, writeonly);
BUFFER_DECLARATION(weights, 3, uint, readonly);
#ifdef BIAS
BUFFER_DECLARATION(biases, 4, uint, readonly);
#endif /* BIAS */

#if STRIDE_X == 2
#define CONVOLVE1x3(s, w) convolve1x3_stride2(s, w)
#define LOAD_AND_UNPACK(offset) load_and_unpack_stride2(offset)
#elif STRIDE_X == 1 /* STRIDE_X == 1 */
#define CONVOLVE1x3(s, w) convolve1x3_stride1(s, w)
#define LOAD_AND_UNPACK(offset) load_and_unpack_stride1(offset)
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

vec4[2] load_and_unpack_stride1(uint offset)
{
    uvec2 packed_s[2];
    vec4  s[2];

    LOAD1(packed_s[0], src, offset);
    LOAD1(packed_s[1], src, offset + uint(1));

    s[0] = vec4(unpackHalf2x16(packed_s[0].x), unpackHalf2x16(packed_s[0].y));
    s[1] = vec4(unpackHalf2x16(packed_s[1].x), unpackHalf2x16(packed_s[1].y));

    return s;
}

vec4[3] load_and_unpack_stride2(uint offset)
{
    uvec2 packed_s[3];
    vec4  s[3];

    LOAD1(packed_s[0], src, offset);
    LOAD1(packed_s[1], src, offset + uint(1));
    LOAD1(packed_s[2], src, offset + uint(2));

    s[0] = vec4(unpackHalf2x16(packed_s[0].x), unpackHalf2x16(packed_s[0].y));
    s[1] = vec4(unpackHalf2x16(packed_s[1].x), unpackHalf2x16(packed_s[1].y));
    s[2] = vec4(unpackHalf2x16(packed_s[2].x), unpackHalf2x16(packed_s[2].y));

    return s;
}

/** An optimized direct convolution 3x3 OpenGL ES compute shader for process 4 elements at once
 *
 * @note This OpenGL ES shader works with stride_x = 1 and 2
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
    Image    src     = CONVERT_TO_IMAGE_STRUCT_FP16(src);
    Tensor3D weights = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP_FP16(weights);
    Tensor3D dst     = CONVERT_TO_TENSOR3D_STRUCT_FP16(dst);

#ifdef BIAS
    Vector   biases  = CONVERT_TO_VECTOR_STRUCT_NO_STEP_FP16(biases);
#endif /* BIAS */

    uvec2 packed_d;

    vec4 pixels = vec4(0);

    uint z_index = gl_GlobalInvocationID.z;

    weights.current_offset += z_index * weights_stride_w;

    for(int d = 0; d < int(weights_depth); ++d)
    {
        // load 3 weights once
        uvec2 packed_w[3];

        LOAD2(packed_w[0], weights, tensor3D_offset_fp16(weights, 0, 0, 0) >> 2);
        LOAD2(packed_w[1], weights, tensor3D_offset_fp16(weights, 0, 1, 0) >> 2);
        LOAD2(packed_w[2], weights, tensor3D_offset_fp16(weights, 0, 2, 0) >> 2);

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
        uint offset;
        // first line
        offset = src.current_offset >> uint(3);
        s      = LOAD_AND_UNPACK(offset);

        pixels += CONVOLVE1x3(s, w[0]);

        // second line
        offset = (src.current_offset + src_stride_y) >> uint(3);
        s      = LOAD_AND_UNPACK(offset);

        pixels += CONVOLVE1x3(s, w[1]);

        // third line
        offset = (src.current_offset + (src_stride_y << 1)) >> uint(3);
        s      = LOAD_AND_UNPACK(offset);

        pixels += CONVOLVE1x3(s, w[2]);

        src.current_offset += src_stride_z;
        weights.current_offset += weights_stride_z;
    }

#ifdef BIAS
    uint  packed_b;
    float b;
    LOAD1(packed_b, biases, vector_offset_fp16(biases, int(z_index)) >> 2);

    if(z_index % uint(2) == uint(0))
    {
        b = unpackHalf2x16(packed_b).x;
    }
    else
    {
        b = unpackHalf2x16(packed_b).y;
    }

    pixels += vec4(b);
#endif /* BIAS */

    packed_d = uvec2(packHalf2x16(pixels.xy), packHalf2x16(pixels.zw));
    STORE1(dst, dst.current_offset >> uint(3), packed_d);
}
#elif defined(PROCESS_X_4ELEMENTS_Y_3ELEMENTS_FP16)
precision mediump float;

BUFFER_DECLARATION(src, 1, uvec2, readonly);
BUFFER_DECLARATION(dst, 2, uvec2, writeonly);
BUFFER_DECLARATION(weights, 3, uint, readonly);
#ifdef BIAS
BUFFER_DECLARATION(biases, 4, uint, readonly);
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

vec4[2] load_and_unpack(uint offset)
{
    uvec2 packed_s[2];
    vec4  s[2];

    LOAD1(packed_s[0], src, offset);
    LOAD1(packed_s[1], src, offset + uint(1));

    s[0] = vec4(unpackHalf2x16(packed_s[0].x), unpackHalf2x16(packed_s[0].y));
    s[1] = vec4(unpackHalf2x16(packed_s[1].x), unpackHalf2x16(packed_s[1].y));

    return s;
}

/** An optimized direct convolution 3x3 OpenGL ES compute shader for process 4x3 elements at once
 *
 * @note This OpenGL ES shader works with stride_x = 1 and 2
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
    Image    src     = CONVERT_TO_IMAGE_STRUCT_FP16(src);
    Tensor3D weights = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP_FP16(weights);
    Tensor3D dst     = CONVERT_TO_TENSOR3D_STRUCT_FP16(dst);

#ifdef BIAS
    Vector   biases  = CONVERT_TO_VECTOR_STRUCT_NO_STEP_FP16(biases);
#endif /* BIAS */

    uvec2 packed_d;

    vec4 pixels[3];
    int  i;

    for(i = 0; i < 3; i++)
    {
        pixels[i] = vec4(0);
    }

    uint z_index = gl_GlobalInvocationID.z;

    weights.current_offset += z_index * weights_stride_w;

    for(int d = 0; d < int(weights_depth); ++d)
    {
        // load 3 weights once
        uvec2 packed_w[3];

        LOAD2(packed_w[0], weights, tensor3D_offset_fp16(weights, 0, 0, 0) >> 2);
        LOAD2(packed_w[1], weights, tensor3D_offset_fp16(weights, 0, 1, 0) >> 2);
        LOAD2(packed_w[2], weights, tensor3D_offset_fp16(weights, 0, 2, 0) >> 2);

        vec3 w[3];
        w[0] = vec3(unpackHalf2x16(packed_w[0].x), unpackHalf2x16(packed_w[0].y).x);
        w[1] = vec3(unpackHalf2x16(packed_w[1].x), unpackHalf2x16(packed_w[1].y).x);
        w[2] = vec3(unpackHalf2x16(packed_w[2].x), unpackHalf2x16(packed_w[2].y).x);

        vec4 s[2];
        vec4 r;
        uint offset;
        // first line
        offset = src.current_offset >> uint(3);
        s      = load_and_unpack(offset);

        pixels[0] += CONVOLVE1x3(s, w[0]);

        // second line
        offset = (src.current_offset + src_stride_y) >> uint(3);
        s      = load_and_unpack(offset);

        pixels[0] += CONVOLVE1x3(s, w[1]);
        pixels[1] += CONVOLVE1x3(s, w[0]);

        // third line
        offset = (src.current_offset + (src_stride_y << 1)) >> uint(3);
        s      = load_and_unpack(offset);

        pixels[0] += CONVOLVE1x3(s, w[2]);
        pixels[1] += CONVOLVE1x3(s, w[1]);
        pixels[2] += CONVOLVE1x3(s, w[0]);

        // forth line
        offset = (src.current_offset + uint(3) * (src_stride_y)) >> uint(3);
        s      = load_and_unpack(offset);

        pixels[1] += CONVOLVE1x3(s, w[2]);
        pixels[2] += CONVOLVE1x3(s, w[1]);

        // fifth line
        offset = (src.current_offset + (src_stride_y << 2)) >> uint(3);
        s      = load_and_unpack(offset);

        pixels[2] += CONVOLVE1x3(s, w[2]);

        src.current_offset += src_stride_z;
        weights.current_offset += weights_stride_z;
    }

#ifdef BIAS
    uint  packed_b;
    float b;
    LOAD1(packed_b, biases, vector_offset_fp16(biases, int(z_index)) >> 2);

    if(z_index % uint(2) == uint(0))
    {
        b = unpackHalf2x16(packed_b).x;
    }
    else
    {
        b = unpackHalf2x16(packed_b).y;
    }

    for(i = 0; i < 3; i++)
    {
        pixels[i] += vec4(b);
    }
#endif /* BIAS */

    packed_d = uvec2(packHalf2x16(pixels[0].xy), packHalf2x16(pixels[0].zw));
    STORE1(dst, dst.current_offset >> uint(3), packed_d);

    packed_d = uvec2(packHalf2x16(pixels[1].xy), packHalf2x16(pixels[1].zw));
    STORE1(dst, (dst.current_offset + dst_stride_y) >> uint(3), packed_d);

    packed_d = uvec2(packHalf2x16(pixels[2].xy), packHalf2x16(pixels[2].zw));
    STORE1(dst, (dst.current_offset + (dst_stride_y << 1)) >> uint(3), packed_d);
}
#elif defined(PROCESS_X_4ELEMENTS_Y_4ELEMENTS_FP16)
precision mediump float;

BUFFER_DECLARATION(src, 1, uvec2, readonly);
BUFFER_DECLARATION(dst, 2, uvec2, writeonly);
BUFFER_DECLARATION(weights, 3, uint, readonly);
#ifdef BIAS
BUFFER_DECLARATION(biases, 4, uint, readonly);
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

vec4[2] load_and_unpack(uint offset)
{
    uvec2 packed_s[2];
    vec4  s[2];

    LOAD1(packed_s[0], src, offset);
    LOAD1(packed_s[1], src, offset + uint(1));

    s[0] = vec4(unpackHalf2x16(packed_s[0].x), unpackHalf2x16(packed_s[0].y));
    s[1] = vec4(unpackHalf2x16(packed_s[1].x), unpackHalf2x16(packed_s[1].y));

    return s;
}

/** An optimized direct convolution 3x3 OpenGL ES compute shader for process 4x4 elements at once
 *
 * @note This OpenGL ES shader works with stride_x = 1 and 2
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
    Image    src     = CONVERT_TO_IMAGE_STRUCT_FP16(src);
    Tensor3D weights = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP_FP16(weights);
    Tensor3D dst     = CONVERT_TO_TENSOR3D_STRUCT_FP16(dst);

#ifdef BIAS
    Vector   biases  = CONVERT_TO_VECTOR_STRUCT_NO_STEP_FP16(biases);
#endif /* BIAS */

    uvec2 packed_d;

    vec4 pixels[4];
    int  i;

    for(i = 0; i < 4; i++)
    {
        pixels[i] = vec4(0);
    }

    uint z_index = gl_GlobalInvocationID.z;

    weights.current_offset += z_index * weights_stride_w;

    for(int d = 0; d < int(weights_depth); ++d)
    {
        // load 3 weights once
        uvec2 packed_w[3];

        LOAD2(packed_w[0], weights, tensor3D_offset_fp16(weights, 0, 0, 0) >> 2);
        LOAD2(packed_w[1], weights, tensor3D_offset_fp16(weights, 0, 1, 0) >> 2);
        LOAD2(packed_w[2], weights, tensor3D_offset_fp16(weights, 0, 2, 0) >> 2);

        vec3 w[3];
        w[0] = vec3(unpackHalf2x16(packed_w[0].x), unpackHalf2x16(packed_w[0].y).x);
        w[1] = vec3(unpackHalf2x16(packed_w[1].x), unpackHalf2x16(packed_w[1].y).x);
        w[2] = vec3(unpackHalf2x16(packed_w[2].x), unpackHalf2x16(packed_w[2].y).x);

        vec4 s[2];
        vec4 r;
        uint offset;
        // first line
        offset = src.current_offset >> uint(3);
        s      = load_and_unpack(offset);

        pixels[0] += CONVOLVE1x3(s, w[0]);

        // second line
        offset = (src.current_offset + src_stride_y) >> uint(3);
        s      = load_and_unpack(offset);

        pixels[0] += CONVOLVE1x3(s, w[1]);
        pixels[1] += CONVOLVE1x3(s, w[0]);

        // third line
        offset = (src.current_offset + (src_stride_y << 1)) >> uint(3);
        s      = load_and_unpack(offset);

        pixels[0] += CONVOLVE1x3(s, w[2]);
        pixels[1] += CONVOLVE1x3(s, w[1]);
        pixels[2] += CONVOLVE1x3(s, w[0]);

        // forth line
        offset = (src.current_offset + uint(3) * (src_stride_y)) >> uint(3);
        s      = load_and_unpack(offset);

        pixels[1] += CONVOLVE1x3(s, w[2]);
        pixels[2] += CONVOLVE1x3(s, w[1]);
        pixels[3] += CONVOLVE1x3(s, w[0]);

        // fifth line
        offset = (src.current_offset + (src_stride_y << 2)) >> uint(3);
        s      = load_and_unpack(offset);

        pixels[2] += CONVOLVE1x3(s, w[2]);
        pixels[3] += CONVOLVE1x3(s, w[1]);

        // sixth line
        offset = (src.current_offset + uint(5) * (src_stride_y)) >> uint(3);
        s      = load_and_unpack(offset);

        pixels[3] += CONVOLVE1x3(s, w[2]);

        src.current_offset += src_stride_z;
        weights.current_offset += weights_stride_z;
    }

#ifdef BIAS
    uint  packed_b;
    float b;
    LOAD1(packed_b, biases, vector_offset_fp16(biases, int(z_index)) >> 2);

    if(z_index % uint(2) == uint(0))
    {
        b = unpackHalf2x16(packed_b).x;
    }
    else
    {
        b = unpackHalf2x16(packed_b).y;
    }

    for(i = 0; i < 4; i++)
    {
        pixels[i] += vec4(b);
    }
#endif /* BIAS */

    packed_d = uvec2(packHalf2x16(pixels[0].xy), packHalf2x16(pixels[0].zw));
    STORE1(dst, dst.current_offset >> uint(3), packed_d);

    packed_d = uvec2(packHalf2x16(pixels[1].xy), packHalf2x16(pixels[1].zw));
    STORE1(dst, (dst.current_offset + dst_stride_y) >> uint(3), packed_d);

    packed_d = uvec2(packHalf2x16(pixels[2].xy), packHalf2x16(pixels[2].zw));
    STORE1(dst, (dst.current_offset + (dst_stride_y << 1)) >> uint(3), packed_d);

    packed_d = uvec2(packHalf2x16(pixels[3].xy), packHalf2x16(pixels[3].zw));
    STORE1(dst, (dst.current_offset + uint(3) * (dst_stride_y)) >> uint(3), packed_d);
}
#elif defined(PROCESS_X_4ELEMENTS_Y_3ELEMENTS_Z_2ELEMENTS_FP16)
precision mediump float;

BUFFER_DECLARATION(src, 1, uvec2, readonly);
BUFFER_DECLARATION(dst, 2, uvec2, writeonly);
BUFFER_DECLARATION(weights, 3, uint, readonly);
#ifdef BIAS
BUFFER_DECLARATION(biases, 4, uint, readonly);
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

vec4[2] load_and_unpack(uint offset)
{
    uvec2 packed_s[2];
    vec4  s[2];

    LOAD1(packed_s[0], src, offset);
    LOAD1(packed_s[1], src, offset + uint(1));

    s[0] = vec4(unpackHalf2x16(packed_s[0].x), unpackHalf2x16(packed_s[0].y));
    s[1] = vec4(unpackHalf2x16(packed_s[1].x), unpackHalf2x16(packed_s[1].y));

    return s;
}

/** An optimized direct convolution 3x3 OpenGL ES compute shader for process 4x3x2 elements at once
 *
 * @note This OpenGL ES shader works with stride_x = 1 and 2
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
    Image    src     = CONVERT_TO_IMAGE_STRUCT_FP16(src);
    Tensor3D weights = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP_FP16(weights);
    Tensor3D dst     = CONVERT_TO_TENSOR3D_STRUCT_FP16(dst);

#ifdef BIAS
    Vector   biases  = CONVERT_TO_VECTOR_STRUCT_NO_STEP_FP16(biases);
#endif /* BIAS */

    uvec2 packed_d;

    vec4 pixels[3];
    int  i;

    uint z_base_index = gl_GlobalInvocationID.z << 1;

    // store orginal src current offset
    uint s_offset = src.current_offset;

    weights.current_offset += z_base_index * weights_stride_w;

    for(int z = 0; z < 2; ++z)
    {
        uint z_index = z_base_index + uint(z);

        src.current_offset = s_offset;
        //weights.current_offset = z_index * weights_stride_w;

        for(i = 0; i < 3; i++)
        {
            pixels[i] = vec4(0);
        }

        for(int d = 0; d < int(weights_depth); ++d)
        {
            // load 3 weights once
            uvec2 packed_w[3];

            LOAD2(packed_w[0], weights, tensor3D_offset_fp16(weights, 0, 0, 0) >> 2);
            LOAD2(packed_w[1], weights, tensor3D_offset_fp16(weights, 0, 1, 0) >> 2);
            LOAD2(packed_w[2], weights, tensor3D_offset_fp16(weights, 0, 2, 0) >> 2);

            vec3 w[3];
            w[0] = vec3(unpackHalf2x16(packed_w[0].x), unpackHalf2x16(packed_w[0].y).x);
            w[1] = vec3(unpackHalf2x16(packed_w[1].x), unpackHalf2x16(packed_w[1].y).x);
            w[2] = vec3(unpackHalf2x16(packed_w[2].x), unpackHalf2x16(packed_w[2].y).x);

            vec4 s[2];
            vec4 r;
            uint offset;
            // first line
            offset = src.current_offset >> uint(3);
            s      = load_and_unpack(offset);

            pixels[0] += CONVOLVE1x3(s, w[0]);

            // second line
            offset = (src.current_offset + src_stride_y) >> uint(3);
            s      = load_and_unpack(offset);

            pixels[0] += CONVOLVE1x3(s, w[1]);
            pixels[1] += CONVOLVE1x3(s, w[0]);

            // third line
            offset = (src.current_offset + (src_stride_y << 1)) >> uint(3);
            s      = load_and_unpack(offset);

            pixels[0] += CONVOLVE1x3(s, w[2]);
            pixels[1] += CONVOLVE1x3(s, w[1]);
            pixels[2] += CONVOLVE1x3(s, w[0]);

            // forth line
            offset = (src.current_offset + uint(3) * (src_stride_y)) >> uint(3);
            s      = load_and_unpack(offset);

            pixels[1] += CONVOLVE1x3(s, w[2]);
            pixels[2] += CONVOLVE1x3(s, w[1]);

            // fifth line
            offset = (src.current_offset + (src_stride_y << 2)) >> uint(3);
            s      = load_and_unpack(offset);

            pixels[2] += CONVOLVE1x3(s, w[2]);

            src.current_offset += src_stride_z;
            weights.current_offset += weights_stride_z;
        }

#ifdef BIAS
        uint  packed_b;
        float b;
        LOAD1(packed_b, biases, vector_offset_fp16(biases, int(z_index)) >> 2);

        if(z_index % uint(2) == uint(0))
        {
            b = unpackHalf2x16(packed_b).x;
        }
        else
        {
            b = unpackHalf2x16(packed_b).y;
        }

        for(i = 0; i < 3; i++)
        {
            pixels[i] += vec4(b);
        }
#endif /* BIAS */

        packed_d = uvec2(packHalf2x16(pixels[0].xy), packHalf2x16(pixels[0].zw));
        STORE1(dst, dst.current_offset >> uint(3), packed_d);

        packed_d = uvec2(packHalf2x16(pixels[1].xy), packHalf2x16(pixels[1].zw));
        STORE1(dst, (dst.current_offset + dst_stride_y) >> uint(3), packed_d);

        packed_d = uvec2(packHalf2x16(pixels[2].xy), packHalf2x16(pixels[2].zw));
        STORE1(dst, (dst.current_offset + (dst_stride_y << 1)) >> uint(3), packed_d);

        dst.current_offset += dst_stride_z;
    }
}
#endif /* PROCESS_1_ELEMENT */
