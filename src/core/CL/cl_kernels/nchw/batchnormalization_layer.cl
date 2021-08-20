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

/** Apply batch normalization.
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
__kernel void batchnormalization_layer_nchw(TENSOR3D_DECLARATION(input),
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
    Tensor3D in = CONVERT_TO_TENSOR3D_STRUCT(input);
#ifdef IN_PLACE
    Tensor3D out = in;
#else  /* IN_PLACE */
    Tensor3D out = CONVERT_TO_TENSOR3D_STRUCT(output);
#endif /* IN_PLACE */
    Vector mean = CONVERT_TO_VECTOR_STRUCT(mean);
    Vector var  = CONVERT_TO_VECTOR_STRUCT(var);
#ifndef USE_DEFAULT_BETA
    Vector beta = CONVERT_TO_VECTOR_STRUCT(beta);
#endif /* USE_DEFAULT_BETA */
#ifndef USE_DEFAULT_GAMMA
    Vector gamma = CONVERT_TO_VECTOR_STRUCT(gamma);
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
    res = 0;

    const int current_slice = get_global_id(2);

    data        = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)in.ptr);
    denominator = *((__global DATA_TYPE *)(var.ptr + current_slice * var.stride_x));
    denominator = INVSQRT_OP(ADD_OP(denominator, ((VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))SQCVT_SAT(epsilon))));

    // Calculate x bar and store results
    numerator = *((__global DATA_TYPE *)(mean.ptr + current_slice * mean.stride_x));
    numerator = SUB_OP(data, numerator);
    x_bar     = MUL_OP(numerator, denominator);

#ifndef USE_DEFAULT_GAMMA
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    gamma_vec = *((__global DATA_TYPE *)(gamma.ptr + current_slice * gamma.stride_x));

    res = MUL_OP(gamma_vec, x_bar);
#else  /* USE_DEFAULT_GAMMA */
    // gamma is equal to 1, no need to perform multiplications
    res = x_bar;
#endif /* USE_DEFAULT_GAMMA */

#ifndef USE_DEFAULT_BETA
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    beta_vec = *((__global DATA_TYPE *)(beta.ptr + current_slice * beta.stride_x));
    // beta is not zero, hence we need to perform the addition
    res = ADD_OP(res, beta_vec);
#endif /* USE_DEFAULT_BETA */

    res = ACTIVATION(ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, res, A_VAL, B_VAL);

    VSTORE(VEC_SIZE)
    (res, 0, (__global DATA_TYPE *)out.ptr);
}
#endif /* defined(VEC_SIZE) && defined(DATA_TYPE) && defined(DATA_TYPE)*/