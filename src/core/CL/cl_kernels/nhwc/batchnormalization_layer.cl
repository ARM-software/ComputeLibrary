/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "helpers.h"

#define ADD_OP(a, b) ((a) + (b))
#define SUB_OP(a, b) ((a) - (b))
#define MUL_OP(a, b) ((a) * (b))
#define INVSQRT_OP(a) rsqrt((a))
#define SQCVT_SAT(a) (a)

#if defined(VEC_SIZE) && defined(DATA_TYPE) && defined(ACTIVATION_TYPE)
#include "activation_float_helpers.h"

/** Apply batch normalization on tensors with NHWC format.
 *
 * @note It is possible to select the activation function to apply using -DACTIVATION_TYPE e.g. -DACTIVATION_TYPE=relu
 * @note A, B variables required by some activation functions are set using -DA_VAL= and -DB_VAL= respectively
 *
 * @param[in]  input_ptr                            Pointer to the first source tensor. Supported data types: F16/F32
 * @param[in]  input_stride_x                       Stride of the first source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the first source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the first source tensor
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  mean_ptr                             Pointer to the mean source tensor. Supported data types: same as @p input_ptr
 * @param[in]  mean_stride_x                        Stride of the mean source tensor in X dimension (in bytes)
 * @param[in]  mean_step_x                          mean_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  mean_offset_first_element_in_bytes   The offset of the first element in the mean source tensor
 * @param[in]  var_ptr                              Pointer to the var tensor. Supported data types: same as @p input_ptr
 * @param[in]  var_stride_x                         Stride of the var tensor in X dimension (in bytes)
 * @param[in]  var_step_x                           var_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  var_offset_first_element_in_bytes    The offset of the first element in the var source tensor
 * @param[in]  beta_ptr                             Pointer to the beta source tensor. Supported data types: same as @p input_ptr
 * @param[in]  beta_stride_x                        Stride of the beta source tensor in X dimension (in bytes)
 * @param[in]  beta_step_x                          beta_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  beta_offset_first_element_in_bytes   The offset of the first element in the beta source tensor
 * @param[in]  gamma_ptr                            Pointer to the gamma source tensor. Supported data types: same as @p input_ptr
 * @param[in]  gamma_stride_x                       Stride of the gamma source tensor in X dimension (in bytes)
 * @param[in]  gamma_step_x                         gamma_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  gamma_offset_first_element_in_bytes  The offset of the first element in the gamma source tensor
 * @param[in]  epsilon                              Epsilon parameter in the batch normalization equation
 */
__kernel void batchnormalization_layer_nhwc(TENSOR3D_DECLARATION(input),
#ifndef IN_PLACE
                                            TENSOR3D_DECLARATION(output),
#endif /* not IN_PLACE */
                                            VECTOR_DECLARATION(mean),
                                            VECTOR_DECLARATION(var),
#ifndef USE_DEFAULT_BETA
                                            VECTOR_DECLARATION(beta),
#endif /* USE_DEFAULT_BETA */
#ifndef USE_DEFAULT_GAMMA
                                            VECTOR_DECLARATION(gamma),
#endif /* USE_DEFAULT_GAMMA */
                                            float epsilon)
{
    uint x_offs = max((int)(get_global_id(0) * VEC_SIZE * sizeof(DATA_TYPE) - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE * sizeof(DATA_TYPE)), 0);

    __global uchar *input_addr = input_ptr + input_offset_first_element_in_bytes + x_offs + get_global_id(1) * input_stride_y + get_global_id(2) * input_stride_z;
#ifdef IN_PLACE
    __global uchar *output_addr = input_ptr;
#else  /* IN_PLACE */
    __global uchar *output_addr = output_ptr + output_offset_first_element_in_bytes + x_offs + get_global_id(1) * output_stride_y + get_global_id(2) * output_stride_z;
#endif /* IN_PLACE */
    __global uchar *mean_addr = mean_ptr + mean_offset_first_element_in_bytes + x_offs;
    __global uchar *var_addr  = var_ptr + var_offset_first_element_in_bytes + x_offs;
#ifndef USE_DEFAULT_BETA
    __global uchar *beta_addr = beta_ptr + beta_offset_first_element_in_bytes + x_offs;
#endif /* USE_DEFAULT_BETA */
#ifndef USE_DEFAULT_GAMMA
    __global uchar *gamma_addr = gamma_ptr + gamma_offset_first_element_in_bytes + x_offs;
#endif /* USE_DEFAULT_GAMMA */

    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    data = 0;
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    denominator = 0;
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    numerator = 0;
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    x_bar = 0;
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    res0 = 0;

    data        = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)input_addr);
    denominator = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)var_addr);
    denominator = INVSQRT_OP(ADD_OP(denominator, ((VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))SQCVT_SAT(epsilon))));

    // Calculate x bar and store results
    numerator = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)mean_addr);
    numerator = SUB_OP(data, numerator);
    x_bar     = MUL_OP(numerator, denominator);

#ifndef USE_DEFAULT_GAMMA
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    gamma_vec = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)gamma_addr);

    res0 = MUL_OP(gamma_vec, x_bar);
#else  /* USE_DEFAULT_GAMMA */
    // gamma is equal to 1, no need to perform multiplications
    res0 = x_bar;
#endif /* USE_DEFAULT_GAMMA */

#ifndef USE_DEFAULT_BETA
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    beta_vec = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)beta_addr);
    // beta is not zero, hence we need to perform the addition
    res0 = ADD_OP(res0, beta_vec);
#endif /* USE_DEFAULT_BETA */

    res0 = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, res0, A_VAL, B_VAL);

    STORE_VECTOR_SELECT(res, DATA_TYPE, output_addr, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0)
}
#endif /* defined(VEC_SIZE) && defined(DATA_TYPE) && defined(DATA_TYPE)*/