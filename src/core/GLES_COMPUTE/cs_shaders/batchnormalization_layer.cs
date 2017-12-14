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
#endif /*DATA_TYPE_FP32*/

#define ADD_OP(a, b) ((a) + (b))
#define SUB_OP(a, b) ((a) - (b))
#define MUL_OP(a, b) ((a) * (b))
#define INVSQRT_OP(a) inversesqrt((a))
#define SQCVT_SAT(a) (a)

/** Apply batch normalization.
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_NAME". e.g. "#define DATA_TYPE_FP32"
 * @note Epsilon parameter in the batch normalization equation should be given as a preprocessor argument using "#define EPSILON". e.g. "#define EPSILON 0.1"
 *
 * @param[in]  src_ptr     Pointer to the first source tensor. Supported data types: F16/F32
 * @param[in]  src_attrs   The attributes of the source tensor
 * @param[out] dst_ptr     Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_attrs   The attributes of the destination tensor
 * @param[in]  mean_ptr    Pointer to the mean source tensor. Supported data types: same as @p src_ptr
 * @param[in]  mean_attrs  The attributes of the mean tensor
 * @param[in]  var_ptr     Pointer to the var tensor. Supported data types: same as @p src_ptr
 * @param[in]  var_attrs   The attributes of the var tensor
 * @param[in]  beta_ptr    Pointer to the beta source tensor. Supported data types: same as @p src_ptr
 * @param[in]  beta_attrs  The attributes of the beta tensor
 * @param[in]  gamma_ptr   Pointer to the gamma source tensor. Supported data types: same as @p src_ptr
 * @param[in]  gamma_attrs The attributes of the gamma tensor
 */
SHADER_PARAMS_DECLARATION
{
    Tensor3DAttributes src_attrs;
    Tensor3DAttributes dst_attrs;
    VectorAttributes   mean_attrs;
    VectorAttributes   var_attrs;
    VectorAttributes   beta_attrs;
    VectorAttributes   gamma_attrs;
};

#ifdef DATA_TYPE_FP32
TENSOR_DECLARATION(1, srcBuffer, float, src_ptr, src_shift, 2, readonly);
TENSOR_DECLARATION(2, dstBuffer, float, dst_ptr, dst_shift, 2, writeonly);
TENSOR_DECLARATION(3, meanBuffer, float, mean_ptr, mean_shift, 2, readonly);
TENSOR_DECLARATION(4, varBuffer, float, var_ptr, var_shift, 2, readonly);
TENSOR_DECLARATION(5, betaBuffer, float, beta_ptr, beta_shift, 2, readonly);
TENSOR_DECLARATION(6, gammaBuffer, float, gamma_ptr, gamma_shift, 2, readonly);

void main(void)
{
    Tensor3DIterator src_iter   = CONVERT_TO_TENSOR3D_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator dst_iter   = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);
    VectorIterator   mean_iter  = CONVERT_TO_VECTOR_ITERATOR(mean_attrs, mean_shift);
    VectorIterator   var_iter   = CONVERT_TO_VECTOR_ITERATOR(var_attrs, var_shift);
    VectorIterator   beta_iter  = CONVERT_TO_VECTOR_ITERATOR(beta_attrs, beta_shift);
    VectorIterator   gamma_iter = CONVERT_TO_VECTOR_ITERATOR(gamma_attrs, gamma_shift);

    float input_value = 0.f;
    float denominator = 0.f;
    float numerator   = 0.f;
    float x_bar       = 0.f;
    float gamma_param = 0.f;
    float beta_param  = 0.f;

    uint current_slice = gl_GlobalInvocationID.z;

    input_value = LOAD_CURRENT_ITEM(src_ptr, src_iter);
    denominator = LOAD(var_ptr, TENSOR_OFFSET_ADVANCE_IN_BYTES(var_iter, current_slice * var_attrs.stride_x));
    denominator = INVSQRT_OP(ADD_OP(denominator, SQCVT_SAT(float(ESPILON))));

    // Calculate x bar and store results
    numerator = LOAD(mean_ptr, TENSOR_OFFSET_ADVANCE_IN_BYTES(mean_iter, current_slice * mean_attrs.stride_x));
    numerator = SUB_OP(input_value, numerator);
    x_bar     = MUL_OP(numerator, denominator);

    gamma_param = LOAD(gamma_ptr, TENSOR_OFFSET_ADVANCE_IN_BYTES(gamma_iter, current_slice * beta_attrs.stride_x));
    beta_param  = LOAD(beta_ptr, TENSOR_OFFSET_ADVANCE_IN_BYTES(beta_iter, current_slice * beta_attrs.stride_x));

    STORE_CURRENT_ITEM(dst_ptr, dst_iter, ADD_OP(MUL_OP(gamma_param, x_bar), beta_param));
}

#elif defined(DATA_TYPE_FP16)
TENSOR_DECLARATION(1, srcBuffer, uvec2, src_ptr, src_shift, 3, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec2, dst_ptr, dst_shift, 3, writeonly);
TENSOR_DECLARATION(3, meanBuffer, uvec2, mean_ptr, mean_shift, 3, readonly);
TENSOR_DECLARATION(4, varBuffer, uvec2, var_ptr, var_shift, 3, readonly);
TENSOR_DECLARATION(5, betaBuffer, uvec2, beta_ptr, beta_shift, 3, readonly);
TENSOR_DECLARATION(6, gammaBuffer, uvec2, gamma_ptr, gamma_shift, 3, readonly);

void main(void)
{
    Tensor3DIterator src_iter   = CONVERT_TO_TENSOR3D_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator dst_iter   = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);
    VectorIterator   mean_iter  = CONVERT_TO_VECTOR_ITERATOR(mean_attrs, mean_shift);
    VectorIterator   var_iter   = CONVERT_TO_VECTOR_ITERATOR(var_attrs, var_shift);
    VectorIterator   beta_iter  = CONVERT_TO_VECTOR_ITERATOR(beta_attrs, beta_shift);
    VectorIterator   gamma_iter = CONVERT_TO_VECTOR_ITERATOR(gamma_attrs, gamma_shift);

    vec4  unpacked_s[5];
    float denominator;
    float numerator;
    float gamma_param;
    float beta_param;
    vec4  x_bar;
    vec4  result;

    uint current_slice = gl_GlobalInvocationID.z;
    unpacked_s[0]      = LOAD_UNPACK4_CURRENT_ITEM_HALF(src_ptr, src_iter);
    unpacked_s[1]      = LOAD_UNPACK4_HALF(var_ptr, TENSOR_OFFSET_ADVANCE_IN_BYTES(var_iter, current_slice * var_attrs.stride_x));
    unpacked_s[2]      = LOAD_UNPACK4_HALF(mean_ptr, TENSOR_OFFSET_ADVANCE_IN_BYTES(mean_iter, current_slice * mean_attrs.stride_x));
    unpacked_s[3]      = LOAD_UNPACK4_HALF(gamma_ptr, TENSOR_OFFSET_ADVANCE_IN_BYTES(gamma_iter, current_slice * beta_attrs.stride_x));
    unpacked_s[4]      = LOAD_UNPACK4_HALF(beta_ptr, TENSOR_OFFSET_ADVANCE_IN_BYTES(beta_iter, current_slice * beta_attrs.stride_x));

    if((current_slice % uint(4)) == uint(0))
    {
        denominator = unpacked_s[1].x;
        denominator = INVSQRT_OP(ADD_OP(denominator, SQCVT_SAT(float(ESPILON))));

        //Calculate x bar and store results
        numerator = unpacked_s[2].x;
        x_bar     = MUL_OP(SUB_OP(unpacked_s[0], numerator), denominator);

        gamma_param = unpacked_s[3].x;
        beta_param  = unpacked_s[4].x;
        result      = ADD_OP(MUL_OP(gamma_param, x_bar), beta_param);

        STORE_PACK4_CURRENT_ITEM_HALF(dst_ptr, dst_iter, result);
    }
    else if((current_slice % uint(4)) == uint(1))
    {
        denominator = unpacked_s[1].y;
        denominator = INVSQRT_OP(ADD_OP(denominator, SQCVT_SAT(float(ESPILON))));

        //Calculate x bar and store results
        numerator = unpacked_s[2].y;
        x_bar     = MUL_OP(SUB_OP(unpacked_s[0], numerator), denominator);

        gamma_param = unpacked_s[3].y;
        beta_param  = unpacked_s[4].y;
        result      = ADD_OP(MUL_OP(gamma_param, x_bar), beta_param);

        STORE_PACK4_CURRENT_ITEM_HALF(dst_ptr, dst_iter, result);
    }
    else if((current_slice % uint(4)) == uint(2))
    {
        denominator = unpacked_s[1].z;
        denominator = INVSQRT_OP(ADD_OP(denominator, SQCVT_SAT(float(ESPILON))));

        //Calculate x bar and store results
        numerator = unpacked_s[2].z;
        x_bar     = MUL_OP(SUB_OP(unpacked_s[0], numerator), denominator);

        gamma_param = unpacked_s[3].z;
        beta_param  = unpacked_s[4].z;
        result      = ADD_OP(MUL_OP(gamma_param, x_bar), beta_param);

        STORE_PACK4_CURRENT_ITEM_HALF(dst_ptr, dst_iter, result);
    }
    else
    {
        denominator = unpacked_s[1].w;
        denominator = INVSQRT_OP(ADD_OP(denominator, SQCVT_SAT(float(ESPILON))));

        //Calculate x bar and store results
        numerator = unpacked_s[2].w;
        x_bar     = MUL_OP(SUB_OP(unpacked_s[0], numerator), denominator);

        gamma_param = unpacked_s[3].w;
        beta_param  = unpacked_s[4].w;
        result      = ADD_OP(MUL_OP(gamma_param, x_bar), beta_param);

        STORE_PACK4_CURRENT_ITEM_HALF(dst_ptr, dst_iter, result);
    }
}
#endif /*DATA_TYPE_FP16*/
