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
#endif /*DATA_TYPE_FP16*/

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

/** Dropout is used to improve over-fit on neural networks.
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_NAME". e.g. "#define DATA_TYPE_FP32"
 *
 * @param[in]  src_ptr    Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  src_attrs  The attributes of the source tensor
 * @param[out] mask_ptr   Pointer to the mask tensor. Supported data types: same as @p src_ptr
 * @param[in]  mask_attrs The attributes of the mask tensor
 * @param[out] dst_ptr    Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_attrs  The attributes of the destination tensor
 */
SHADER_PARAMS_DECLARATION
{
    Tensor3DAttributes src_attrs;
    Tensor3DAttributes mask_attrs;
    Tensor3DAttributes dst_attrs;
};

#ifdef DATA_TYPE_FP32
TENSOR_DECLARATION(1, srcBuffer, float, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, maskBuffer, float, mask_ptr, mask_shift, 2, );
TENSOR_DECLARATION(3, dstBuffer, float, dst_ptr, dst_shift, 2, writeonly);

void main(void)
{
    Tensor3DIterator src_iter  = CONVERT_TO_TENSOR3D_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator mask_iter = CONVERT_TO_TENSOR3D_ITERATOR(mask_attrs, mask_shift);
    Tensor3DIterator dst_iter  = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

    float random  = 0.f;
    float inputv  = 0.f;
    float maskv   = 0.f;
    float outputv = 0.f;

#ifdef FORWARD
    random = rand(vec3(gl_GlobalInvocationID.xyz), SEED);
    maskv  = (random > RATIO) ? 1.f : 0.f;
    STORE_CURRENT_ITEM(mask_ptr, mask_iter, maskv);
#else  /* FORWARD */
    maskv = LOAD_CURRENT_ITEM(mask_ptr, mask_iter);
#endif /* FORWARD */

    inputv  = LOAD_CURRENT_ITEM(src_ptr, src_iter);
    outputv = maskv * inputv * float(SCALE);
    STORE_CURRENT_ITEM(dst_ptr, dst_iter, outputv);
}

#elif defined(DATA_TYPE_FP16)
TENSOR_DECLARATION(1, srcBuffer, uint, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, maskBuffer, uint, mask_ptr, mask_shift, 2, );
TENSOR_DECLARATION(3, dstBuffer, uint, dst_ptr, dst_shift, 2, writeonly);

void main(void)
{
    Tensor3DIterator src_iter  = CONVERT_TO_TENSOR3D_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator mask_iter = CONVERT_TO_TENSOR3D_ITERATOR(mask_attrs, mask_shift);
    Tensor3DIterator dst_iter  = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);

    float random1    = 0.f;
    float random2    = 0.f;
    vec2  input_vec  = vec2(0, 0);
    vec2  output_vec = vec2(0, 0);
    vec2  mask_vec   = vec2(0, 0);

#ifdef FORWARD
    random1          = rand(vec3(gl_GlobalInvocationID.xyz), SEED);
    random2          = rand(vec3(float(gl_GlobalInvocationID.x) + 0.5f, gl_GlobalInvocationID.yz), SEED);
    mask_vec.x       = (random1 > RATIO) ? 1.f : 0.f;
    mask_vec.y       = (random2 > RATIO) ? 1.f : 0.f;

    STORE_PACK2_CURRENT_ITEM_HALF(mask_ptr, mask_iter, mask_vec);
#else  /* FORWARD */
    mask_vec = LOAD_UNPACK2_CURRENT_ITEM_HALF(mask_ptr, mask_iter);
#endif /* FORWARD */

    input_vec  = LOAD_UNPACK2_CURRENT_ITEM_HALF(src_ptr, src_iter);
    output_vec = mask_vec * input_vec * float(SCALE);

    STORE_PACK2_CURRENT_ITEM_HALF(dst_ptr, dst_iter, output_vec);
}

#else /* DATA_TYPE_FP32 */

#endif /* DATA_TYPE_FP32 */
