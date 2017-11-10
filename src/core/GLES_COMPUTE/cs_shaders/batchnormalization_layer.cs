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

#ifdef DATA_TYPE_FP32
precision highp float;
#elif defined(DATA_TYPE_FP16)
precision mediump float;
#endif /*DATA_TYPE_FP32*/

#define ADD_OP(a, b) ((a) + (b))
#define SUB_OP(a, b) ((a) - (b))
#define MUL_OP(a, b) ((a) * (b))
#define INVSQRT_OP(a) inversesqrt((a))
#define SQCVT_SAT(a) (a)

layout(std140) uniform shader_params
{
    TENSOR3D_PARAM_DECLARATION(src);
    TENSOR3D_PARAM_DECLARATION(dst);
    VECTOR_PARAM_DECLARATION(mean);
    VECTOR_PARAM_DECLARATION(var);
    VECTOR_PARAM_DECLARATION(beta);
    VECTOR_PARAM_DECLARATION(gamma);
};

#ifdef DATA_TYPE_FP32
BUFFER_DECLARATION(src, 1, float, readonly);
BUFFER_DECLARATION(dst, 2, float, writeonly);
BUFFER_DECLARATION(mean, 3, float, readonly);
BUFFER_DECLARATION(var, 4, float, readonly);
BUFFER_DECLARATION(beta, 5, float, readonly);
BUFFER_DECLARATION(gamma, 6, float, readonly);

/** Apply batch normalization.
 *
 * @note Epsilon parameter in the batch normalization equation should be given as a preprocessor argument using "#define EPSILON". e.g. "#define EPSILON 0.1"
 *
 * @param[in]  src_ptr                             Pointer to the first source tensor. Supported data types: F32
 * @param[in]  src_stride_x                        Stride of the first source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                          src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                        Stride of the first source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                          src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                        Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                          src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes   The offset of the first element in the first source tensor
 * @param[out] dst_ptr                             Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                        Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                          dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                        Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                          dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                        Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                          dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes   The offset of the first element in the destination tensor
 * @param[in]  mean_ptr                            Pointer to the mean source tensor. Supported data types: same as @p src_ptr
 * @param[in]  mean_stride_x                       Stride of the mean source tensor in X dimension (in bytes)
 * @param[in]  mean_step_x                         mean_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  mean_offset_first_element_in_bytes  The offset of the first element in the mean source tensor
 * @param[in]  var_ptr                             Pointer to the var tensor. Supported data types: same as @p src_ptr
 * @param[in]  var_stride_x                        Stride of the var tensor in X dimension (in bytes)
 * @param[in]  var_step_x                          var_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  var_offset_first_element_in_bytes   The offset of the first element in the var source tensor
 * @param[in]  beta_ptr                            Pointer to the beta source tensor. Supported data types: same as @p src_ptr
 * @param[in]  beta_stride_x                       Stride of the beta source tensor in X dimension (in bytes)
 * @param[in]  beta_step_x                         beta_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  beta_offset_first_element_in_bytes  The offset of the first element in the beta source tensor
 * @param[in]  gamma_ptr                           Pointer to the gamma source tensor. Supported data types: same as @p src_ptr
 * @param[in]  gamma_stride_x                      Stride of the gamma source tensor in X dimension (in bytes)
 * @param[in]  gamma_step_x                        gamma_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  gamma_offset_first_element_in_bytes The offset of the first element in the gamma source tensor
 */
void main(void)
{
    Tensor3D src   = CONVERT_TO_TENSOR3D_STRUCT(src);
    Tensor3D dst   = CONVERT_TO_TENSOR3D_STRUCT(dst);
    Vector   mean  = CONVERT_TO_VECTOR_STRUCT(mean);
    Vector   var   = CONVERT_TO_VECTOR_STRUCT(var);
    Vector   beta  = CONVERT_TO_VECTOR_STRUCT(beta);
    Vector   gamma = CONVERT_TO_VECTOR_STRUCT(gamma);

    float input_value = 0.f;
    float denominator = 0.f;
    float numerator   = 0.f;
    float x_bar       = 0.f;
    float gamma_param = 0.f;
    float beta_param  = 0.f;

    uint current_slice = gl_GlobalInvocationID.z;

    input_value = src_ptr[src.current_offset];
    denominator = var_ptr[var.current_offset + (current_slice * var.stride_x) >> 2];
    denominator = INVSQRT_OP(ADD_OP(denominator, SQCVT_SAT(float(ESPILON))));

    // Calculate x bar and store results
    numerator = mean_ptr[mean.current_offset + (current_slice * mean.stride_x) >> 2];
    numerator = SUB_OP(input_value, numerator);
    x_bar     = MUL_OP(numerator, denominator);

    gamma_param = gamma_ptr[gamma.current_offset + (current_slice * beta.stride_x) >> 2];
    beta_param  = beta_ptr[beta.current_offset + (current_slice * beta.stride_x) >> 2];

    dst_ptr[dst.current_offset] = ADD_OP(MUL_OP(gamma_param, x_bar), beta_param);
}

#elif defined(DATA_TYPE_FP16)
BUFFER_DECLARATION(src, 1, uint, );
BUFFER_DECLARATION(dst, 2, uint, writeonly);
BUFFER_DECLARATION(mean, 3, uint, );
BUFFER_DECLARATION(var, 4, uint, );
BUFFER_DECLARATION(beta, 5, uint, );
BUFFER_DECLARATION(gamma, 6, uint, );

/** Apply batch normalization.
 *
 * @note Epsilon parameter in the batch normalization equation should be given as a preprocessor argument using "#define EPSILON". e.g. "#define EPSILON 0.1"
 *
 * @param[in]  src_ptr                             Pointer to the first source tensor. Supported data types: F16
 * @param[in]  src_stride_x                        Stride of the first source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                          src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                        Stride of the first source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                          src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                        Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                          src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes   The offset of the first element in the first source tensor
 * @param[out] dst_ptr                             Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                        Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                          dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                        Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                          dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                        Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                          dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes   The offset of the first element in the destination tensor
 * @param[in]  mean_ptr                            Pointer to the mean source tensor. Supported data types: same as @p src_ptr
 * @param[in]  mean_stride_x                       Stride of the mean source tensor in X dimension (in bytes)
 * @param[in]  mean_step_x                         mean_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  mean_offset_first_element_in_bytes  The offset of the first element in the mean source tensor
 * @param[in]  var_ptr                             Pointer to the var tensor. Supported data types: same as @p src_ptr
 * @param[in]  var_stride_x                        Stride of the var tensor in X dimension (in bytes)
 * @param[in]  var_step_x                          var_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  var_offset_first_element_in_bytes   The offset of the first element in the var source tensor
 * @param[in]  beta_ptr                            Pointer to the beta source tensor. Supported data types: same as @p src_ptr
 * @param[in]  beta_stride_x                       Stride of the beta source tensor in X dimension (in bytes)
 * @param[in]  beta_step_x                         beta_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  beta_offset_first_element_in_bytes  The offset of the first element in the beta source tensor
 * @param[in]  gamma_ptr                           Pointer to the gamma source tensor. Supported data types: same as @p src_ptr
 * @param[in]  gamma_stride_x                      Stride of the gamma source tensor in X dimension (in bytes)
 * @param[in]  gamma_step_x                        gamma_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  gamma_offset_first_element_in_bytes The offset of the first element in the gamma source tensor
 */
void main(void)
{
    Tensor3D src   = CONVERT_TO_TENSOR3D_STRUCT_FP16(src);
    Tensor3D dst   = CONVERT_TO_TENSOR3D_STRUCT_FP16(dst);
    Vector   mean  = CONVERT_TO_VECTOR_STRUCT_FP16(mean);
    Vector   var   = CONVERT_TO_VECTOR_STRUCT_FP16(var);
    Vector   beta  = CONVERT_TO_VECTOR_STRUCT_FP16(beta);
    Vector   gamma = CONVERT_TO_VECTOR_STRUCT_FP16(gamma);

    vec2  input_value;
    float denominator;
    float numerator;
    vec2  x_bar;
    float gamma_param;
    float beta_param;

    uint current_slice = gl_GlobalInvocationID.z;
    if((current_slice % uint(2)) == uint(0))
    {
        input_value = unpackHalf2x16(src_ptr[src.current_offset >> 2]);
        denominator = unpackHalf2x16(var_ptr[(var.current_offset + current_slice * var.stride_x) >> 2]).x;
        denominator = INVSQRT_OP(ADD_OP(denominator, SQCVT_SAT(float(ESPILON))));

        //Calculate x bar and store results
        numerator = unpackHalf2x16(mean_ptr[(mean.current_offset + current_slice * mean.stride_x) >> 2]).x;
        x_bar     = MUL_OP(SUB_OP(input_value, numerator), denominator);

        gamma_param = unpackHalf2x16(gamma_ptr[(gamma.current_offset + current_slice * beta.stride_x) >> 2]).x;
        beta_param  = unpackHalf2x16(beta_ptr[(beta.current_offset + current_slice * beta.stride_x) >> 2]).x;

        dst_ptr[dst.current_offset >> 2] = packHalf2x16(ADD_OP(MUL_OP(gamma_param, x_bar), beta_param));
    }
    else
    {
        input_value = unpackHalf2x16(src_ptr[src.current_offset >> 2]);
        denominator = unpackHalf2x16(var_ptr[(var.current_offset + current_slice * var.stride_x) >> 2]).y;
        denominator = INVSQRT_OP(ADD_OP(denominator, SQCVT_SAT(float(ESPILON))));

        //Calculate x bar and store results
        numerator = unpackHalf2x16(mean_ptr[(mean.current_offset + current_slice * mean.stride_x) >> 2]).y;
        x_bar     = MUL_OP(SUB_OP(input_value, numerator), denominator);

        gamma_param = unpackHalf2x16(gamma_ptr[(gamma.current_offset + current_slice * beta.stride_x) >> 2]).y;
        beta_param  = unpackHalf2x16(beta_ptr[(beta.current_offset + current_slice * beta.stride_x) >> 2]).y;

        dst_ptr[dst.current_offset >> 2] = packHalf2x16(ADD_OP(MUL_OP(gamma_param, x_bar), beta_param));
    }
}
#endif /*DATA_TYPE_FP32*/
