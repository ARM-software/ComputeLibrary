/*
 * Copyright (c) 2017-2019 ARM Limited.
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

#if defined(VEC_SIZE) && defined(DATA_TYPE)

#if defined(FUSED_ACTIVATION)
#include "activation_layer.cl"
#define ACTIVATION_FUNC(x) ACTIVATION_OP(FUSED_ACTIVATION, x)
#else /* defined(FUSED_ACTIVATION) */
#define ACTIVATION_FUNC(x) (x)
#endif /* defined(FUSED_ACTIVATION) */

/** Apply batch normalization.
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
    res          = x_bar;
#endif /* USE_DEFAULT_GAMMA */

#ifndef USE_DEFAULT_BETA
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    beta_vec = *((__global DATA_TYPE *)(beta.ptr + current_slice * beta.stride_x));
    // beta is not zero, hence we need to perform the addition
    res = ADD_OP(res, beta_vec);
#endif /* USE_DEFAULT_BETA */

    res = ACTIVATION_FUNC(res);

    VSTORE(VEC_SIZE)
    (res, 0, (__global DATA_TYPE *)out.ptr);
}

/** Apply batch normalization on tensors with NHWC format.
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

    const int current_slice = get_global_id(0);

    data        = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)in.ptr);
    denominator = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(var.ptr + current_slice * VEC_SIZE * var.stride_x));
    denominator = INVSQRT_OP(ADD_OP(denominator, ((VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))SQCVT_SAT(epsilon))));

    // Calculate x bar and store results
    numerator = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(mean.ptr + current_slice * VEC_SIZE * mean.stride_x));
    numerator = SUB_OP(data, numerator);
    x_bar     = MUL_OP(numerator, denominator);

#ifndef USE_DEFAULT_GAMMA
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    gamma_vec = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(gamma.ptr + current_slice * VEC_SIZE * gamma.stride_x));

    res = MUL_OP(gamma_vec, x_bar);
#else  /* USE_DEFAULT_GAMMA */
    // gamma is equal to 1, no need to perform multiplications
    res = x_bar;
#endif /* USE_DEFAULT_GAMMA */

#ifndef USE_DEFAULT_BETA
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    beta_vec = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(beta.ptr + current_slice * VEC_SIZE * beta.stride_x));
    // beta is not zero, hence we need to perform the addition
    res = ADD_OP(res, beta_vec);
#endif /* USE_DEFAULT_BETA */

    res = ACTIVATION_FUNC(res);

    VSTORE(VEC_SIZE)
    (res, 0, (__global DATA_TYPE *)out.ptr);
}
#endif /* defined(VEC_SIZE) && defined(DATA_TYPE) */

#if defined(NUM_CHANNELS) && defined(DATA_TYPE) && defined(EPSILON)
/** Fuse batchnorm parameters to convolution layer parameters
 *
 * @attention Data type should be passed using the -DDATA_TYPE compile flag, e.g. -DDATA_TYPE=float
 * @attention Input tensor depth should be given as a preprocessor argument using -DNUM_CHANNELS=size. e.g. -DNUM_CHANNELS=16
 * @attention Batch normalization epsilon parameter should be given as a preprocessor argument with -DEPSILON=value. e.g. -DEPSILON=0.001f
 *
 * @param[in]  conv_w_ptr                             Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  conv_w_stride_x                        Stride of the source tensor in X dimension (in bytes)
 * @param[in]  conv_w_step_x                          input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  conv_w_stride_y                        Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  conv_w_step_y                          input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  conv_w_stride_z                        Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  conv_w_step_z                          input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  conv_w_stride_w                        Stride of the source tensor in W dimension (in bytes)
 * @param[in]  conv_w_step_w                          input_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  conv_w_offset_first_element_in_bytes   The offset of the first element in the source tensor
 * @param[in]  bn_mean_ptr                            Pointer to the mean source tensor. Supported data types: same as @p input_ptr
 * @param[in]  bn_mean_stride_x                       Stride of the mean source tensor in X dimension (in bytes)
 * @param[in]  bn_mean_step_x                         bn_mean_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  bn_mean_offset_first_element_in_bytes  The offset of the first element in the mean source tensor
 * @param[in]  bn_var_ptr                             Pointer to the var tensor. Supported data types: same as @p input_ptr
 * @param[in]  bn_var_stride_x                        Stride of the var tensor in X dimension (in bytes)
 * @param[in]  bn_var_step_x                          bn_var_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  bn_var_offset_first_element_in_bytes   The offset of the first element in the var source tensor
 * @param[out] fused_w_ptr                            Pointer to the destination weights tensors. Supported data types: same as @p input_ptr
 * @param[in]  fused_w_stride_x                       Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  fused_w_step_x                         fused_w_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  fused_w_stride_y                       Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  fused_w_step_y                         fused_w_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  fused_w_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  fused_w_step_z                         fused_w_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  fused_w_stride_w                       Stride of the destination tensor in W dimension (in bytes)
 * @param[in]  fused_w_step_w                         fused_w_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  fused_w_offset_first_element_in_bytes  The offset of the first element in the destination tensor
 * @param[in]  fused_b_ptr                            Pointer to the destination bias tensor. Supported data types: same as @p input_ptr
 * @param[in]  fused_b_stride_x                       Stride of the bias source tensor in X dimension (in bytes)
 * @param[in]  fused_b_step_x                         fused_b_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  fused_b_offset_first_element_in_bytes  The offset of the first element in the destination tensor
 * @param[in]  conv_b_ptr                             Pointer to the source bias tensor. Supported data types: same as @p input_ptr
 * @param[in]  conv_b_stride_x                        Stride of the beta source tensor in X dimension (in bytes)
 * @param[in]  conv_b_step_x                          conv_b_beta_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  conv_b_offset_first_element_in_bytes   The offset of the first element in the source bias tensor
 * @param[in]  bn_beta_ptr                            Pointer to the beta source tensor. Supported data types: same as @p input_ptr
 * @param[in]  bn_beta_stride_x                       Stride of the beta source tensor in X dimension (in bytes)
 * @param[in]  bn_beta_step_x                         bn_beta_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  bn_beta_offset_first_element_in_bytes  The offset of the first element in the beta source tensor
 * @param[in]  bn_gamma_ptr                           Pointer to the gamma source tensor. Supported data types: same as @p input_ptr
 * @param[in]  bn_gamma_stride_x                      Stride of the gamma source tensor in X dimension (in bytes)
 * @param[in]  bn_gamma_step_x                        bn_gamma_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  bn_gamma_offset_first_element_in_bytes The offset of the first element in the gamma source tensor
 * @param[in]  epsilon                                Epsilon parameter in the batch normalization equation
 */
__kernel void fuse_batchnormalization_layer(TENSOR4D_DECLARATION(conv_w),
                                            VECTOR_DECLARATION(bn_mean),
                                            VECTOR_DECLARATION(bn_var)
#ifndef IN_PLACE_W
                                            ,
                                            TENSOR4D_DECLARATION(fused_w)
#endif /* not IN_PLACE_W */
#ifndef IN_PLACE_B
                                            ,
                                            VECTOR_DECLARATION(fused_b)
#endif /* not IN_PLACE_B */
#ifdef HAS_BIAS
                                            ,
                                            VECTOR_DECLARATION(conv_b)
#endif /* HAS_BIAS */
#ifndef USE_DEFAULT_BETA
                                            ,
                                            VECTOR_DECLARATION(bn_beta)
#endif /* USE_DEFAULT_BETA */
#ifndef USE_DEFAULT_GAMMA
                                            ,
                                            VECTOR_DECLARATION(bn_gamma)
#endif /* USE_DEFAULT_GAMMA */
                                           )
{
    Tensor4D conv_w  = CONVERT_TO_TENSOR4D_STRUCT(conv_w, NUM_CHANNELS);
    Vector   bn_mean = CONVERT_TO_VECTOR_STRUCT_NO_STEP(bn_mean);
    Vector   bn_var  = CONVERT_TO_VECTOR_STRUCT_NO_STEP(bn_var);

    // Conditional ops
#ifdef HAS_BIAS
    Vector conv_b = CONVERT_TO_VECTOR_STRUCT_NO_STEP(conv_b);
#endif /* HAS_BIAS */
#ifndef USE_DEFAULT_BETA
    Vector bn_beta = CONVERT_TO_VECTOR_STRUCT_NO_STEP(bn_beta);
#endif /* USE_DEFAULT_BETA */
#ifndef USE_DEFAULT_GAMMA
    Vector bn_gamma = CONVERT_TO_VECTOR_STRUCT_NO_STEP(bn_gamma);
#endif /* USE_DEFAULT_GAMMA */

    // In-place ops
#ifdef IN_PLACE_W
    Tensor4D fused_w          = conv_w;
    uint     fused_w_stride_x = conv_w_stride_x;
#else  /* IN_PLACE_W */
    Tensor4D  fused_w                      = CONVERT_TO_TENSOR4D_STRUCT(fused_w, NUM_CHANNELS);
#endif /* IN_PLACE_W */
#ifdef IN_PLACE_B
    Vector fused_b = conv_b;
#else  /* IN_PLACE_B */
    Vector    fused_b                      = CONVERT_TO_VECTOR_STRUCT_NO_STEP(fused_b);
#endif /* IN_PLACE_B */

    const int current_slice = get_global_id(2) / NUM_CHANNELS;

#if defined(VEC_SIZE) && defined(LAST_ACCESSED_X)
    // Check if access on width gets out of bounds
    // If it does shift access vector to access elements within bounds
    const int xi = (int)(get_global_id(0) * VEC_SIZE);
    conv_w.ptr -= max(xi - (int)LAST_ACCESSED_X, 0) * conv_w_stride_x;
    fused_w.ptr -= max(xi - (int)LAST_ACCESSED_X, 0) * fused_w_stride_x;

    // Load W
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    wn = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)conv_w.ptr);
#else  // !defined(VEC_SIZE) || !defined(LAST_ACCESSED_X)
    DATA_TYPE wn                           = *((__global DATA_TYPE *)(conv_w.ptr));
#endif // defined(VEC_SIZE) && defined(LAST_ACCESSED_X)

    // rvar = 1 / sqrt(var + epsilon)
    const DATA_TYPE var  = *((__global DATA_TYPE *)(bn_var.ptr + current_slice * bn_var.stride_x));
    const DATA_TYPE rvar = INVSQRT_OP(ADD_OP(var, SQCVT_SAT((float)EPSILON)));
    wn *= rvar;

    // Load b
    const DATA_TYPE mean = *((__global DATA_TYPE *)(bn_mean.ptr + current_slice * bn_mean.stride_x));
    DATA_TYPE bn         = 0;
#ifdef HAS_BIAS
    bn = *((__global DATA_TYPE *)(conv_b.ptr + current_slice * conv_b.stride_x));
#endif /* HAS_BIAS */
    bn = (bn - mean) * rvar;

#ifndef USE_DEFAULT_GAMMA
    const DATA_TYPE gamma_scalar = *((__global DATA_TYPE *)(bn_gamma.ptr + current_slice * bn_gamma.stride_x));
    wn *= gamma_scalar;
    bn *= gamma_scalar;
#endif /* USE_DEFAULT_GAMMA */

#ifndef USE_DEFAULT_BETA
    const DATA_TYPE beta_scalar = *((__global DATA_TYPE *)(bn_beta.ptr + current_slice * bn_beta.stride_x));
    bn += beta_scalar;
#endif /* USE_DEFAULT_BETA */

#if defined(VEC_SIZE) && defined(LAST_ACCESSED_X)
    // Store updated weights
    VSTORE(VEC_SIZE)
    (wn, 0, (__global DATA_TYPE *)fused_w.ptr);
#else  // !defined(VEC_SIZE) || !defined(LAST_ACCESSED_X)
    *((__global DATA_TYPE *)(fused_w.ptr)) = wn;
#endif // defined(VEC_SIZE) && defined(LAST_ACCESSED_X)

    // Store updated bias
    *((__global DATA_TYPE *)(fused_b.ptr + current_slice * fused_b.stride_x)) = bn;
}
#endif /* defined(NUM_CHANNELS) && defined(DATA_TYPE) && defined(EPSILON) */
