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

#if defined(DATA_TYPE) && defined(EPSILON)
/** OpenCL kernel to fuse the weights of convolution or depthwise convolution layer with batch normalization when the data layout is either NCHW or NHWC
 *
 * @note The input weights tensor is assumed 4D with the OFMs in the fourth dimension
 * @note Data type should be passed at compile time using the -DDATA_TYPE, e.g. -DDATA_TYPE=float
 * @note The third dimension of the input tensor should be passed at compile time when weights belong to a convolution layer using -DDIM2=size. e.g. -DDIM2=16.
 *       For depthwise convolution weight do not pass DIM2
 * @note Data layout NHWC should be passed at compile time with -DNHWC. For data layout NCHW it is not required to pass any parameter
 * @note Batch normalization epsilon parameter should be passed at compile time using -DEPSILON=value. e.g. -DEPSILON=0.001f
 *
 * @param[in]  w_ptr                                 Pointer to the weights tensor. Supported data types: F16/F32
 * @param[in]  w_stride_x                            Stride of the weights tensor in X dimension (in bytes)
 * @param[in]  w_step_x                              w_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  w_stride_y                            Stride of the weights tensor in Y dimension (in bytes)
 * @param[in]  w_step_y                              w_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  w_stride_z                            Stride of the weights tensor in Z dimension (in bytes)
 * @param[in]  w_step_z                              w_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  w_offset_first_element_in_bytes       The offset of the first element in the weights tensor
 * @param[in]  b_ptr                                 (Optional) Pointer to the bias tensor. Supported data types: same as @p w_ptr
 * @param[in]  b_stride_x                            (Optional) Stride of the bias tensor in X dimension (in bytes)
 * @param[in]  b_step_x                              (Optional) b_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  b_stride_y                            (Optional) Stride of the bias tensor in Y dimension (in bytes)
 * @param[in]  b_step_y                              (Optional) b_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  b_stride_z                            (Optional) Stride of the bias tensor in Z dimension (in bytes)
 * @param[in]  b_step_z                              (Optional) b_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  b_offset_first_element_in_bytes       (Optional) The offset of the first element in the bias tensor
 * @param[in]  mean_ptr                              Pointer to the mean source tensor. Supported data types: same as @p w_ptr
 * @param[in]  mean_stride_x                         Stride of the mean source tensor in X dimension (in bytes)
 * @param[in]  mean_step_x                           mean_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  mean_offset_first_element_in_bytes    The offset of the first element in the mean source tensor
 * @param[in]  var_ptr                               Pointer to the var tensor. Supported data types: same as @p w_ptr
 * @param[in]  var_stride_x                          Stride of the var tensor in X dimension (in bytes)
 * @param[in]  var_step_x                            var_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  var_offset_first_element_in_bytes     The offset of the first element in the var source tensor
 * @param[out] w_fused_ptr                           (Optional) Pointer to the destination weights tensors. Supported data types: same as @p w_ptr
 * @param[in]  w_fused_stride_x                      (Optional) Stride of the destination weights tensor in X dimension (in bytes)
 * @param[in]  w_fused_step_x                        (Optional) w_fused_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  w_fused_stride_y                      (Optional) Stride of the destination weights tensor in Y dimension (in bytes)
 * @param[in]  w_fused_step_y                        (Optional) w_fused_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  w_fused_stride_z                      (Optional) Stride of the destination weights tensor in Z dimension (in bytes)
 * @param[in]  w_fused_step_z                        (Optional) w_fused_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  w_fused_offset_first_element_in_bytes (Optional) The offset of the first element in the destination weights tensor
 * @param[in]  b_fused_ptr                           (Optional) Pointer to the destination bias tensor. Supported data types: same as @p w_ptr
 * @param[in]  b_fused_stride_x                      (Optional) Stride of the destination bias tensor in X dimension (in bytes)
 * @param[in]  b_fused_step_x                        (Optional) b_fused_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  b_fused_offset_first_element_in_bytes (Optional) The offset of the first element in the destination bias tensor
 * @param[in]  beta_ptr                              (Optional) Pointer to the beta source tensor. Supported data types: same as @p w_ptr
 * @param[in]  beta_stride_x                         (Optional) Stride of the beta source tensor in X dimension (in bytes)
 * @param[in]  beta_step_x                           (Optional) beta_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  beta_offset_first_element_in_bytes    (Optional) The offset of the first element in the beta source tensor
 * @param[in]  gamma_ptr                             (Optional) Pointer to the gamma source tensor. Supported data types: same as @p w_ptr
 * @param[in]  gamma_stride_x                        (Optional) Stride of the gamma source tensor in X dimension (in bytes)
 * @param[in]  gamma_step_x                          (Optional) gamma_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  gamma_offset_first_element_in_bytes   (Optional) The offset of the first element in the gamma source tensor
 */
__kernel void fuse_batchnormalization_layer(TENSOR3D_DECLARATION(w),
#if defined(BIAS)
                                            VECTOR_DECLARATION(b),
#endif // defined(BIAS)
                                            VECTOR_DECLARATION(mean),
                                            VECTOR_DECLARATION(var)
#ifndef IN_PLACE_W
                                            ,
                                            TENSOR3D_DECLARATION(w_fused)
#endif // ifndef IN_PLACE_W
#ifndef IN_PLACE_B
                                            ,
                                            VECTOR_DECLARATION(b_fused)
#endif // ifndef IN_PLACE_B
#if defined(BETA)
                                            ,
                                            VECTOR_DECLARATION(beta)
#endif // defined(BETA)
#if defined(GAMMA)
                                            ,
                                            VECTOR_DECLARATION(gamma)
#endif // defined(GAMMA)
                                           )
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

#if defined(DIM2)
    int c0 = z % DIM2;
    int c1 = z / DIM2;
#else // ! defined(DIM2)
    int c0                                                                                    = 0;
#if defined(NHWC)
    int c1                                                                                    = x;
#else  // defined(NHWC)
    int c1 = z;
#endif // defined(NHWC)
#endif // defined(DIM2)

    int w_offset = x * sizeof(DATA_TYPE) + y * w_stride_y + z * w_stride_z;
    int v_offset = c1 * sizeof(DATA_TYPE);

    DATA_TYPE w_old = 0.0f;
    DATA_TYPE b_old = 0.0f;
    DATA_TYPE w_new = 0.0f;
    DATA_TYPE b_new = 0.0f;
    DATA_TYPE gamma = 1.0f;
    DATA_TYPE mean  = 0.0f;
    DATA_TYPE var   = 1.0f;
    DATA_TYPE beta  = 0.0f;

    w_old = *((__global DATA_TYPE *)(w_ptr + w_offset + w_offset_first_element_in_bytes));
    var   = *((__global DATA_TYPE *)(var_ptr + v_offset + var_offset_first_element_in_bytes));
    mean  = *((__global DATA_TYPE *)(mean_ptr + v_offset + mean_offset_first_element_in_bytes));

#if defined(GAMMA)
    gamma = *((__global DATA_TYPE *)(gamma_ptr + v_offset + gamma_offset_first_element_in_bytes));
#endif // defined(GAMMA)

    // Compute new weight
    w_new = (gamma * w_old) / (sqrt(var + EPSILON));

#if defined(IN_PLACE_W)
    *((__global DATA_TYPE *)(w_ptr + w_offset + w_offset_first_element_in_bytes)) = w_new;
#else  // defined(IN_PLACE_W)
    *((__global DATA_TYPE *)(w_fused_ptr + w_offset + w_fused_offset_first_element_in_bytes)) = w_new;
#endif // defined(IN_PLACE_W)

    // Compute bias
#if !defined(DIM2) && defined(NHWC)
    if(z == 0 && y == 0)
#else  // !defined(DIM2) && defined(NHWC)
    if(x == 0 && y == 0 && c0 == 0)
#endif // !defined(DIM2) && defined(NHWC)
    {
#if defined(BIAS)
        b_old = *((__global DATA_TYPE *)(b_ptr + v_offset + b_offset_first_element_in_bytes));
#endif // defined(BIAS)
#if defined(BETA)
        beta = *((__global DATA_TYPE *)(beta_ptr + v_offset + beta_offset_first_element_in_bytes));
#endif // defined(BETA)

        b_new = ((gamma * (b_old - mean)) / (sqrt(var + EPSILON))) + beta;

#if defined(BIAS)

#if defined(IN_PLACE_B)
        *((__global DATA_TYPE *)(b_ptr + v_offset + b_offset_first_element_in_bytes)) = b_new;
#else  // defined(IN_PLACE_B)
        *((__global DATA_TYPE *)(b_fused_ptr + v_offset + b_fused_offset_first_element_in_bytes)) = b_new;
#endif // defined(IN_PLACE_B)

#else // defined(BIAS)

#ifndef IN_PLACE_B
        *((__global DATA_TYPE *)(b_fused_ptr + v_offset + b_fused_offset_first_element_in_bytes)) = b_new;
#endif // ifndef IN_PLACE_B

#endif // defined(BIAS)
    }
}
#endif // defined(DATA_TYPE) && defined(EPSILON)