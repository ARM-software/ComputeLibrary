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
    TENSOR3D_PARAM_DECLARATION(mask);
    TENSOR3D_PARAM_DECLARATION(dst);
};

uint hash(uint x)
{
    x += (x << 10u);
    x ^= (x >> 6u);
    x += (x << 3u);
    x ^= (x >> 11u);
    x += (x << 15u);
    return x;
}

uint hash(uvec3 v)
{
    return hash(v.x ^ hash(v.y) ^ hash(v.z));
}

float float_construct(uint m)
{
    const uint ieee_mantissa = 0x007FFFFFu;
    const uint ieee_one      = 0x3F800000u;

    m &= ieee_mantissa;
    m |= ieee_one;

    float f = uintBitsToFloat(m);
    return f - 1.0;
}

float rand(vec3 v, float seed)
{
    return float_construct(hash(floatBitsToUint(v + seed)));
}

#ifdef DATA_TYPE_FP32

precision highp float;

BUFFER_DECLARATION(src, 1, float, readonly);
BUFFER_DECLARATION(mask, 2, float, );
BUFFER_DECLARATION(dst, 3, float, writeonly);

/** Dropout is used to improve over-fit on neural networks.
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_FP32"
 *
 * @param[in]  src_ptr                            Pointer to the source tensor. Supported data types: F32
 * @param[in]  src_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                         src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                         src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out] mask_ptr                           Pointer to the mask tensor. Supported data types: same as @p src_ptr
 * @param[in]  mask_stride_x                      Stride of the mask tensor in X dimension (in bytes)
 * @param[in]  mask_step_x                        mask_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  mask_stride_y                      Stride of the mask tensor in Y dimension (in bytes)
 * @param[in]  mask_step_y                        mask_stride_y * number of elements along y processed per workitem(in bytes)
 * @param[in]  mask_stride_z                      Stride of the mask tensor in Z dimension (in bytes)
 * @param[in]  mask_step_z                        mask_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  mask_offset_first_element_in_bytes The offset of the first element in the mask tensor
 * @param[out] dst_ptr                            Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                       Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                         dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination tensor
 */
void main(void)
{
    Tensor3D src  = GC_CONVERT_TO_TENSOR3D_STRUCT(src);
    Tensor3D mask = GC_CONVERT_TO_TENSOR3D_STRUCT(mask);
    Tensor3D dst  = GC_CONVERT_TO_TENSOR3D_STRUCT(dst);

    float random  = 0.f;
    float inputv  = 0.f;
    float maskv   = 0.f;
    float outputv = 0.f;

#ifdef FORWARD
    random = rand(vec3(gl_GlobalInvocationID.xyz), SEED);
    maskv  = (random > RATIO) ? 1.f : 0.f;
    GC_STORE1_3D_OFFSET(maskv, mask, 0, 0, 0);
#else  /* FORWARD */
    GC_LOAD1_3D_OFFSET(maskv, mask, 0, 0, 0);
#endif /* FORWARD */

    GC_LOAD1_3D_OFFSET(inputv, src, 0, 0, 0);
    outputv = maskv * inputv * float(SCALE);
    GC_STORE1_3D_OFFSET(outputv, dst, 0, 0, 0);
}

#elif defined(DATA_TYPE_FP16)

precision mediump float;

BUFFER_DECLARATION(src, 1, uint, readonly);
BUFFER_DECLARATION(mask, 2, uint, );
BUFFER_DECLARATION(dst, 3, uint, writeonly);

/** Dropout is used to improve over-fit on neural networks.
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_FP16"
 *
 * @param[in]  src_ptr                            Pointer to the source tensor. Supported data types: F16
 * @param[in]  src_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                         src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                         src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out] mask_ptr                           Pointer to the mask tensor. Supported data types: same as @p src_ptr
 * @param[in]  mask_stride_x                      Stride of the mask tensor in X dimension (in bytes)
 * @param[in]  mask_step_x                        mask_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  mask_stride_y                      Stride of the mask tensor in Y dimension (in bytes)
 * @param[in]  mask_step_y                        mask_stride_y * number of elements along y processed per workitem(in bytes)
 * @param[in]  mask_stride_z                      Stride of the mask tensor in Z dimension (in bytes)
 * @param[in]  mask_step_z                        mask_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  mask_offset_first_element_in_bytes The offset of the first element in the mask tensor
 * @param[out] dst_ptr                            Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                       Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                         dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination tensor
 */
void main(void)
{
    Tensor3D src  = GC_CONVERT_TO_TENSOR3D_STRUCT(src);
    Tensor3D mask = GC_CONVERT_TO_TENSOR3D_STRUCT(mask);
    Tensor3D dst  = GC_CONVERT_TO_TENSOR3D_STRUCT(dst);

    float random1    = 0.f;
    float random2    = 0.f;
    uint  inputv     = uint(0);
    uint  outputv    = uint(0);
    uint  maskv      = uint(0);
    vec2  input_vec  = vec2(0, 0);
    vec2  output_vec = vec2(0, 0);
    vec2  mask_vec   = vec2(0, 0);

#ifdef FORWARD
    random1          = rand(vec3(gl_GlobalInvocationID.xyz), SEED);
    random2          = rand(vec3(float(gl_GlobalInvocationID.x) + 0.5f, gl_GlobalInvocationID.yz), SEED);
    mask_vec.x       = (random1 > RATIO) ? 1.f : 0.f;
    mask_vec.y       = (random2 > RATIO) ? 1.f : 0.f;
    maskv            = packHalf2x16(mask_vec);
    GC_STORE1_3D_OFFSET(maskv, mask, 0, 0, 0);
#else  /* FORWARD */
    GC_LOAD1_3D_OFFSET(maskv, mask, 0, 0, 0);
    mask_vec = unpackHalf2x16(maskv);
#endif /* FORWARD */

    GC_LOAD1_3D_OFFSET(inputv, src, 0, 0, 0);

    input_vec  = unpackHalf2x16(inputv);
    output_vec = mask_vec * input_vec * float(SCALE);
    outputv    = packHalf2x16(output_vec);

    GC_STORE1_3D_OFFSET(outputv, dst, 0, 0, 0);
}

#else /* DATA_TYPE_FP32 */

#endif /* DATA_TYPE_FP32 */
