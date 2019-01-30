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

#include "helpers_asymm.h"

#if defined(WEIGHTS_OFFSET) && defined(INPUT_OFFSET) && defined(K_OFFSET) && ((defined(OUTPUT_OFFSET) && defined(OUTPUT_MULTIPLIER) && defined(OUTPUT_SHIFT)) || defined(REAL_MULTIPLIER))

#if defined(FUSED_ACTIVATION)
#define DATA_TYPE uchar
#ifndef VEC_SIZE
#define VEC_SIZE 8
#endif /* VEC_SIZE */
#include "activation_layer_qa8.cl"
#define ACTIVATION_FUNC(x) PERFORM_ACTIVATION_QA8(FUSED_ACTIVATION, x)
#else /* defined(FUSED_ACTIVATION) */
#define ACTIVATION_FUNC(x) (x)
#endif /* defined(FUSED_ACTIVATION) */

#if defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)
#if defined(ARM_COMPUTE_OPENCL_DOT8_ACC_ENABLED) && defined(cl_arm_integer_dot_product_accumulate_int8)
#define ARM_DOT(x, y, val) val = arm_dot_acc((x), (y), val);
#else // defined(ARM_COMPUTE_OPENCL_DOT8_ACC_ENABLED) && defined(cl_arm_integer_dot_product_accumulate_int8)
#define ARM_DOT(x, y, val) val += arm_dot((x), (y));
#endif // defined(ARM_COMPUTE_OPENCL_DOT8_ACC_ENABLED) && defined(cl_arm_integer_dot_product_accumulate_int8)
#endif // defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)

#if defined(CONV_STRIDE_Y) && defined(CONV_STRIDE_X) && defined(DEPTH_MULTIPLIER) && defined(DST_CHANNELS)

#if CONV_STRIDE_X > 3
#error "Stride X not supported"
#endif /* CONV_STRIDE_X > 3 */

#if !(defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8))

#if CONV_STRIDE_X == 1
#define GET_VALUES(first_value, left, middle, right)                              \
    ({                                                                            \
        int8 temp0 = CONVERT(vload8(0, first_value), int8);                       \
        int2 temp1 = CONVERT(vload2(0, (first_value + 8 * sizeof(uchar))), int2); \
        \
        left   = CONVERT(temp0.s01234567, int8);                                  \
        middle = CONVERT((int8)(temp0.s1234, temp0.s567, temp1.s0), int8);        \
        right  = CONVERT((int8)(temp0.s2345, temp0.s67, temp1.s01), int8);        \
    })
#elif CONV_STRIDE_X == 2
#define GET_VALUES(first_value, left, middle, right)                     \
    ({                                                                   \
        int16 temp0 = CONVERT(vload16(0, first_value), int16);           \
        int   temp1 = CONVERT(*(first_value + 16 * sizeof(uchar)), int); \
        \
        left   = CONVERT(temp0.s02468ace, int8);                         \
        middle = CONVERT(temp0.s13579bdf, int8);                         \
        right  = CONVERT((int8)(temp0.s2468, temp0.sace, temp1), int8);  \
    })
#else /* CONV_STRIDE_X */
#define GET_VALUES(first_value, left, middle, right)                                \
    ({                                                                              \
        int16 temp0 = CONVERT(vload16(0, first_value), int16);                      \
        int8  temp1 = CONVERT(vload8(0, (first_value + 16 * sizeof(uchar))), int8); \
        \
        left   = CONVERT((int8)(temp0.s0369, temp0.scf, temp1.s25), int8);          \
        middle = CONVERT((int8)(temp0.s147a, temp0.sd, temp1.s036), int8);          \
        right  = CONVERT((int8)(temp0.s258b, temp0.se, temp1.s147), int8);          \
    })
#endif /* CONV_STRIDE_X */

/** This function computes the depthwise convolution quantized.
 *
 * @param[in] src_ptr                               Pointer to the source tensor. Supported data types: QASYMM8
 * @param[in] src_stride_x                          Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                          Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                            src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source tensor
 * @param[in] dst_ptr                               Pointer to the destination tensor. Supported data types: QASYMM8
 * @param[in] dst_stride_x                          Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                            dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                          Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                            dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                          Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                            dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
 * @param[in] weights_ptr                           Pointer to the weights tensor. Supported data types: QASYMM8
 * @param[in] weights_stride_x                      Stride of the weights tensor in X dimension (in bytes)
 * @param[in] weights_step_x                        weights_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] weights_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
 * @param[in] weights_step_y                        weights_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_stride_z                      Stride of the weights tensor in Z dimension (in bytes)
 * @param[in] weights_step_z                        weights_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_offset_first_element_in_bytes The offset of the first element in the weights tensor
 * @param[in] biases_ptr                            (Optional) Pointer to the biases vector. Supported data types: QASYMM8
 * @param[in] biases_stride_x                       (Optional) Stride of the biases vector in X dimension (in bytes)
 * @param[in] biases_step_x                         (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] biases_offset_first_element_in_bytes  (Optional) The offset of the first element in the biases vector
 */

__kernel void depthwise_convolution_3x3_quantized_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(weights)
#if defined(HAS_BIAS)
    ,
    VECTOR_DECLARATION(biases)
#endif //defined(HAS_BIAS)
)
{
    Image    src     = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(src);
    Image    dst     = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);
    Tensor3D weights = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(weights);

    // Extract channel and linearized batch indices
    const int channel = get_global_id(2) % DST_CHANNELS;
    const int batch   = get_global_id(2) / DST_CHANNELS;

#if defined(HAS_BIAS)
    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);

    int bias_value = *((__global int *)(vector_offset(&biases, channel)));
#endif //defined(HAS_BIAS)

    // Load relevant input and weights data (Accounts depth multiplier when indexing input, OFM = IFM * DEPTH_MULTIPLIER)
    src.ptr -= batch * (DST_CHANNELS / DEPTH_MULTIPLIER) * (DEPTH_MULTIPLIER - 1) * src_step_z + (channel - (channel / DEPTH_MULTIPLIER)) * src_step_z;
    __global uchar *weights_addr = weights.ptr + get_global_id(0) * weights_step_x + get_global_id(1) * weights_step_y + channel * weights_step_z;

    uchar3 w0 = vload3(0, weights_addr + 0 * weights_stride_y);
    uchar3 w1 = vload3(0, weights_addr + 1 * weights_stride_y);
    uchar3 w2 = vload3(0, weights_addr + 2 * weights_stride_y);

    int8 values0 = 0;
    int8 sum0    = 0;
#if CONV_STRIDE_Y == 1
    int8 values1 = 0;
    int8 sum1    = 0;
#endif /* CONV_STRIDE_Y */

    // Row0
    int8 left, middle, right;
    GET_VALUES(src.ptr + 0 * src_stride_y, left, middle, right);
    values0 += left * (int8)(w0.s0);
    values0 += middle * (int8)(w0.s1);
    values0 += right * (int8)(w0.s2);

#if WEIGHTS_OFFSET != 0
    sum0 += left + middle + right;
#endif /* WEIGHTS_OFFSET != 0 */

    // Row1
    GET_VALUES(src.ptr + 1 * src_stride_y, left, middle, right);
    values0 += left * (int8)(w1.s0);
    values0 += middle * (int8)(w1.s1);
    values0 += right * (int8)(w1.s2);
#if CONV_STRIDE_Y == 1
    values1 += left * (int8)(w0.s0);
    values1 += middle * (int8)(w0.s1);
    values1 += right * (int8)(w0.s2);
#endif /* CONV_STRIDE_Y == 1 */

#if WEIGHTS_OFFSET != 0
    int8 tmp = left + middle + right;
    sum0 += tmp;
#if CONV_STRIDE_Y == 1
    sum1 += tmp;
#endif /* CONV_STRIDE_Y == 1 */
#endif /* WEIGHTS_OFFSET != 0 */

    // Row2
    GET_VALUES(src.ptr + 2 * src_stride_y, left, middle, right);
    values0 += left * (int8)(w2.s0);
    values0 += middle * (int8)(w2.s1);
    values0 += right * (int8)(w2.s2);
#if CONV_STRIDE_Y == 1
    values1 += left * (int8)(w1.s0);
    values1 += middle * (int8)(w1.s1);
    values1 += right * (int8)(w1.s2);
#endif /* CONV_STRIDE_Y == 1 */

#if WEIGHTS_OFFSET != 0
    tmp = left + middle + right;
    sum0 += tmp;
#if CONV_STRIDE_Y == 1
    sum1 += tmp;
#endif /* CONV_STRIDE_Y == 1 */
#endif /* WEIGHTS_OFFSET != 0 */

#if CONV_STRIDE_Y == 1
    // Row3
    GET_VALUES(src.ptr + 3 * src_stride_y, left, middle, right);
    values1 += left * (int8)(w2.s0);
    values1 += middle * (int8)(w2.s1);
    values1 += right * (int8)(w2.s2);

#if WEIGHTS_OFFSET != 0
    sum1 += left + middle + right;
#endif /* WEIGHTS_OFFSET != 0 */
#endif /* CONV_STRIDE_Y == 1 */

#if defined(HAS_BIAS)
    values0 += (int8)(bias_value);
#if CONV_STRIDE_Y == 1
    values1 += (int8)(bias_value);
#endif /* CONV_STRIDE_Y == 1 */
#endif //defined(HAS_BIAS)

#if WEIGHTS_OFFSET != 0
    values0 += sum0 * (int8)(WEIGHTS_OFFSET);
#if CONV_STRIDE_Y == 1
    values1 += sum1 * (int8)(WEIGHTS_OFFSET);
#endif /* CONV_STRIDE_Y == 1 */
#endif /* WEIGHTS_OFFSET != 0 */

#if INPUT_OFFSET != 0
    ushort  sum_weights = 0;
    ushort3 tmp_we      = convert_ushort3(w0) + convert_ushort3(w1) + convert_ushort3(w2);
    sum_weights += tmp_we.s0 + tmp_we.s1 + tmp_we.s2;
    values0 += sum_weights * (int8)(INPUT_OFFSET);
#if CONV_STRIDE_Y == 1
    values1 += sum_weights * (int8)(INPUT_OFFSET);
#endif /* CONV_STRIDE_Y == 1 */
#endif /* INPUT_OFFSET != 0 */

#if K_OFFSET != 0
    values0 += (int8)(K_OFFSET);
#if CONV_STRIDE_Y == 1
    values1 += (int8)(K_OFFSET);
#endif /* CONV_STRIDE_Y == 1 */
#endif /* K_OFFSET != 0 */

#if defined(REAL_MULTIPLIER)

    values0 = CONVERT(round(CONVERT(values0, float8) * (float8)REAL_MULTIPLIER), int8);

#else // defined(REAL_MULTIPLIER)

    values0 = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(values0, OUTPUT_MULTIPLIER, OUTPUT_SHIFT, 8);

#endif // defined(REAL_MULTIPLIER)

    values0 += (int8)OUTPUT_OFFSET;
    uchar8 res0 = convert_uchar8_sat(values0);
    res0        = max(res0, (uchar8)0);
    res0        = min(res0, (uchar8)255);

    vstore8(ACTIVATION_FUNC(res0), 0, dst.ptr);
#if CONV_STRIDE_Y == 1
#if defined(REAL_MULTIPLIER)

    values1 = CONVERT(round(CONVERT(values1, float8) * (float8)REAL_MULTIPLIER), int8);

#else // defined(REAL_MULTIPLIER)

    values1 = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(values1, OUTPUT_MULTIPLIER, OUTPUT_SHIFT, 8);

#endif // defined(REAL_MULTIPLIER)

    values1 += (int8)OUTPUT_OFFSET;
    uchar8 res1 = convert_uchar8_sat(values1);
    res1        = max(res1, (uchar8)0);
    res1        = min(res1, (uchar8)255);

    vstore8(ACTIVATION_FUNC(res1), 0, dst.ptr + dst_stride_y);
#endif /* CONV_STRIDE_Y == 1 */
}

#else // !(defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8))

#if CONV_STRIDE_X == 1
#define GET_VALUES(first_value, left, middle, right)                 \
    ({                                                               \
        uchar8 temp0 = vload8(0, first_value);                       \
        uchar2 temp1 = vload2(0, (first_value + 8 * sizeof(uchar))); \
        \
        left   = temp0.s01234567;                                    \
        middle = (uchar8)(temp0.s1234, temp0.s567, temp1.s0);        \
        right  = (uchar8)(temp0.s2345, temp0.s67, temp1.s01);        \
    })
#elif CONV_STRIDE_X == 2
#define GET_VALUES(first_value, left, middle, right)         \
    ({                                                       \
        uchar16 temp0 = vload16(0, first_value);             \
        uchar   temp1 = *(first_value + 16 * sizeof(uchar)); \
        \
        left   = temp0.s02468ace;                            \
        middle = temp0.s13579bdf;                            \
        right  = (uchar8)(temp0.s2468, temp0.sace, temp1);   \
    })
#else /* CONV_STRIDE_X */
#define GET_VALUES(first_value, left, middle, right)                   \
    ({                                                                 \
        uchar16 temp0 = vload16(0, first_value);                       \
        uchar8  temp1 = vload8(0, (first_value + 16 * sizeof(uchar))); \
        \
        left   = (uchar8)(temp0.s0369, temp0.scf, temp1.s25);          \
        middle = (uchar8)(temp0.s147a, temp0.sd, temp1.s036);          \
        right  = (uchar8)(temp0.s258b, temp0.se, temp1.s147);          \
    })
#endif /* CONV_STRIDE_X */
/** This function computes the depthwise convolution quantized using dot product when the data layout is NCHW.
 *
 * @param[in] src_ptr                               Pointer to the source tensor. Supported data types: QASYMM8
 * @param[in] src_stride_x                          Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                          Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                            src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source tensor
 * @param[in] dst_ptr                               Pointer to the destination tensor. Supported data types: QASYMM8
 * @param[in] dst_stride_x                          Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                            dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                          Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                            dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                          Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                            dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
 * @param[in] weights_ptr                           Pointer to the weights tensor. Supported data types: QASYMM8
 * @param[in] weights_stride_x                      Stride of the weights tensor in X dimension (in bytes)
 * @param[in] weights_step_x                        weights_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] weights_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
 * @param[in] weights_step_y                        weights_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_stride_z                      Stride of the weights tensor in Z dimension (in bytes)
 * @param[in] weights_step_z                        weights_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_offset_first_element_in_bytes The offset of the first element in the weights tensor
 * @param[in] biases_ptr                            (Optional) Pointer to the biases vector. Supported data types: QASYMM8
 * @param[in] biases_stride_x                       (Optional) Stride of the biases vector in X dimension (in bytes)
 * @param[in] biases_step_x                         (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] biases_offset_first_element_in_bytes  (Optional) The offset of the first element in the biases vector
 */

__kernel void depthwise_convolution_3x3_quantized_dot8_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(weights)
#if defined(HAS_BIAS)
    ,
    VECTOR_DECLARATION(biases)
#endif //defined(HAS_BIAS)
)
{
    Image    src     = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(src);
    Image    dst     = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);
    Tensor3D weights = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(weights);

    // Extract channel and linearized batch indices
    const int channel = get_global_id(2) % DST_CHANNELS;
    const int batch   = get_global_id(2) / DST_CHANNELS;

#if defined(HAS_BIAS)
    Vector    biases  = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);

    const int bias_value = *((__global int *)(vector_offset(&biases, channel)));
#endif //defined(HAS_BIAS)

    // Load relevant input and weights data (Accounts depth multiplier when indexing input, OFM = IFM * DEPTH_MULTIPLIER)
    src.ptr -= batch * (DST_CHANNELS / DEPTH_MULTIPLIER) * (DEPTH_MULTIPLIER - 1) * src_step_z + (channel - (channel / DEPTH_MULTIPLIER)) * src_step_z;
    __global uchar *weights_addr = weights.ptr + get_global_id(0) * weights_step_x + get_global_id(1) * weights_step_y + channel * weights_step_z;

    uchar3 w0 = vload3(0, weights_addr + 0 * weights_stride_y);
    uchar3 w1 = vload3(0, weights_addr + 1 * weights_stride_y);
    uchar3 w2 = vload3(0, weights_addr + 2 * weights_stride_y);

    uchar8 left0, middle0, right0;
    uchar8 left1, middle1, right1;
    uchar8 left2, middle2, right2;

    int8 values0 = 0;
    int8 sum0    = 0;

    GET_VALUES(src.ptr + 0 * src_stride_y, left0, middle0, right0);
    GET_VALUES(src.ptr + 1 * src_stride_y, left1, middle1, right1);
    GET_VALUES(src.ptr + 2 * src_stride_y, left2, middle2, right2);

#if WEIGHTS_OFFSET != 0
    sum0 += convert_int8(left0) + convert_int8(middle0) + convert_int8(right0);
    sum0 += convert_int8(left1) + convert_int8(middle1) + convert_int8(right1);
    sum0 += convert_int8(left2) + convert_int8(middle2) + convert_int8(right2);
#endif /* WEIGHTS_OFFSET != 0 */

#if CONV_STRIDE_Y == 1
    // If conv_stride_y is equals to 1, we compute two output rows

    uchar8 left3, middle3, right3;
    int8   values1 = 0;
    int8   sum1    = 0;

    GET_VALUES(src.ptr + 3 * src_stride_y, left3, middle3, right3);

#if WEIGHTS_OFFSET != 0
    sum1 += convert_int8(left1) + convert_int8(middle1) + convert_int8(right1);
    sum1 += convert_int8(left2) + convert_int8(middle2) + convert_int8(right2);
    sum1 += convert_int8(left3) + convert_int8(middle3) + convert_int8(right3);
#endif /* WEIGHTS_OFFSET != 0 */
#endif // CONV_STRIDE_Y == 1

    ARM_DOT((uchar4)(left0.s0, middle0.s0, right0.s0, left1.s0), (uchar4)(w0.s0, w0.s1, w0.s2, w1.s0), values0.s0);
    ARM_DOT((uchar4)(middle1.s0, right1.s0, left2.s0, middle2.s0), (uchar4)(w1.s1, w1.s2, w2.s0, w2.s1), values0.s0);
    values0.s0 += right2.s0 * w2.s2;

    ARM_DOT((uchar4)(left0.s1, middle0.s1, right0.s1, left1.s1), (uchar4)(w0.s0, w0.s1, w0.s2, w1.s0), values0.s1);
    ARM_DOT((uchar4)(middle1.s1, right1.s1, left2.s1, middle2.s1), (uchar4)(w1.s1, w1.s2, w2.s0, w2.s1), values0.s1);
    values0.s1 += right2.s1 * w2.s2;

    ARM_DOT((uchar4)(left0.s2, middle0.s2, right0.s2, left1.s2), (uchar4)(w0.s0, w0.s1, w0.s2, w1.s0), values0.s2);
    ARM_DOT((uchar4)(middle1.s2, right1.s2, left2.s2, middle2.s2), (uchar4)(w1.s1, w1.s2, w2.s0, w2.s1), values0.s2);
    values0.s2 += right2.s2 * w2.s2;

    ARM_DOT((uchar4)(left0.s3, middle0.s3, right0.s3, left1.s3), (uchar4)(w0.s0, w0.s1, w0.s2, w1.s0), values0.s3);
    ARM_DOT((uchar4)(middle1.s3, right1.s3, left2.s3, middle2.s3), (uchar4)(w1.s1, w1.s2, w2.s0, w2.s1), values0.s3);
    values0.s3 += right2.s3 * w2.s2;

    ARM_DOT((uchar4)(left0.s4, middle0.s4, right0.s4, left1.s4), (uchar4)(w0.s0, w0.s1, w0.s2, w1.s0), values0.s4);
    ARM_DOT((uchar4)(middle1.s4, right1.s4, left2.s4, middle2.s4), (uchar4)(w1.s1, w1.s2, w2.s0, w2.s1), values0.s4);
    values0.s4 += right2.s4 * w2.s2;

    ARM_DOT((uchar4)(left0.s5, middle0.s5, right0.s5, left1.s5), (uchar4)(w0.s0, w0.s1, w0.s2, w1.s0), values0.s5);
    ARM_DOT((uchar4)(middle1.s5, right1.s5, left2.s5, middle2.s5), (uchar4)(w1.s1, w1.s2, w2.s0, w2.s1), values0.s5);
    values0.s5 += right2.s5 * w2.s2;

    ARM_DOT((uchar4)(left0.s6, middle0.s6, right0.s6, left1.s6), (uchar4)(w0.s0, w0.s1, w0.s2, w1.s0), values0.s6);
    ARM_DOT((uchar4)(middle1.s6, right1.s6, left2.s6, middle2.s6), (uchar4)(w1.s1, w1.s2, w2.s0, w2.s1), values0.s6);
    values0.s6 += right2.s6 * w2.s2;

    ARM_DOT((uchar4)(left0.s7, middle0.s7, right0.s7, left1.s7), (uchar4)(w0.s0, w0.s1, w0.s2, w1.s0), values0.s7);
    ARM_DOT((uchar4)(middle1.s7, right1.s7, left2.s7, middle2.s7), (uchar4)(w1.s1, w1.s2, w2.s0, w2.s1), values0.s7);
    values0.s7 += right2.s7 * w2.s2;

#if CONV_STRIDE_Y == 1
    ARM_DOT((uchar4)(left1.s0, middle1.s0, right1.s0, left2.s0), (uchar4)(w0.s0, w0.s1, w0.s2, w1.s0), values1.s0);
    ARM_DOT((uchar4)(middle2.s0, right2.s0, left3.s0, middle3.s0), (uchar4)(w1.s1, w1.s2, w2.s0, w2.s1), values1.s0);
    values1.s0 += right3.s0 * w2.s2;

    ARM_DOT((uchar4)(left1.s1, middle1.s1, right1.s1, left2.s1), (uchar4)(w0.s0, w0.s1, w0.s2, w1.s0), values1.s1);
    ARM_DOT((uchar4)(middle2.s1, right2.s1, left3.s1, middle3.s1), (uchar4)(w1.s1, w1.s2, w2.s0, w2.s1), values1.s1);
    values1.s1 += right3.s1 * w2.s2;

    ARM_DOT((uchar4)(left1.s2, middle1.s2, right1.s2, left2.s2), (uchar4)(w0.s0, w0.s1, w0.s2, w1.s0), values1.s2);
    ARM_DOT((uchar4)(middle2.s2, right2.s2, left3.s2, middle3.s2), (uchar4)(w1.s1, w1.s2, w2.s0, w2.s1), values1.s2);
    values1.s2 += right3.s2 * w2.s2;

    ARM_DOT((uchar4)(left1.s3, middle1.s3, right1.s3, left2.s3), (uchar4)(w0.s0, w0.s1, w0.s2, w1.s0), values1.s3);
    ARM_DOT((uchar4)(middle2.s3, right2.s3, left3.s3, middle3.s3), (uchar4)(w1.s1, w1.s2, w2.s0, w2.s1), values1.s3);
    values1.s3 += right3.s3 * w2.s2;

    ARM_DOT((uchar4)(left1.s4, middle1.s4, right1.s4, left2.s4), (uchar4)(w0.s0, w0.s1, w0.s2, w1.s0), values1.s4);
    ARM_DOT((uchar4)(middle2.s4, right2.s4, left3.s4, middle3.s4), (uchar4)(w1.s1, w1.s2, w2.s0, w2.s1), values1.s4);
    values1.s4 += right3.s4 * w2.s2;

    ARM_DOT((uchar4)(left1.s5, middle1.s5, right1.s5, left2.s5), (uchar4)(w0.s0, w0.s1, w0.s2, w1.s0), values1.s5);
    ARM_DOT((uchar4)(middle2.s5, right2.s5, left3.s5, middle3.s5), (uchar4)(w1.s1, w1.s2, w2.s0, w2.s1), values1.s5);
    values1.s5 += right3.s5 * w2.s2;

    ARM_DOT((uchar4)(left1.s6, middle1.s6, right1.s6, left2.s6), (uchar4)(w0.s0, w0.s1, w0.s2, w1.s0), values1.s6);
    ARM_DOT((uchar4)(middle2.s6, right2.s6, left3.s6, middle3.s6), (uchar4)(w1.s1, w1.s2, w2.s0, w2.s1), values1.s6);
    values1.s6 += right3.s6 * w2.s2;

    ARM_DOT((uchar4)(left1.s7, middle1.s7, right1.s7, left2.s7), (uchar4)(w0.s0, w0.s1, w0.s2, w1.s0), values1.s7);
    ARM_DOT((uchar4)(middle2.s7, right2.s7, left3.s7, middle3.s7), (uchar4)(w1.s1, w1.s2, w2.s0, w2.s1), values1.s7);
    values1.s7 += right3.s7 * w2.s2;
#endif // CONV_STRIDE_Y == 1

#if defined(HAS_BIAS)
    values0 += (int8)(bias_value);
#if CONV_STRIDE_Y == 1
    values1 += (int8)(bias_value);
#endif /* CONV_STRIDE_Y == 1 */
#endif //defined(HAS_BIAS)

#if WEIGHTS_OFFSET != 0
    values0 += sum0 * (int8)(WEIGHTS_OFFSET);
#if CONV_STRIDE_Y == 1
    values1 += sum1 * (int8)(WEIGHTS_OFFSET);
#endif /* CONV_STRIDE_Y == 1 */
#endif /* WEIGHTS_OFFSET != 0 */

#if INPUT_OFFSET != 0
    ushort  sum_weights = 0;
    ushort3 tmp_we      = convert_ushort3(w0) + convert_ushort3(w1) + convert_ushort3(w2);
    sum_weights += tmp_we.s0 + tmp_we.s1 + tmp_we.s2;
    values0 += sum_weights * (int8)(INPUT_OFFSET);
#if CONV_STRIDE_Y == 1
    values1 += sum_weights * (int8)(INPUT_OFFSET);
#endif /* CONV_STRIDE_Y == 1 */
#endif /* INPUT_OFFSET != 0 */

#if K_OFFSET != 0
    values0 += (int8)(K_OFFSET);
#if CONV_STRIDE_Y == 1
    values1 += (int8)(K_OFFSET);
#endif /* CONV_STRIDE_Y == 1 */
#endif /* K_OFFSET != 0 */

#if defined(REAL_MULTIPLIER)

    values0 = CONVERT(round(CONVERT(values0, float8) * (float8)REAL_MULTIPLIER), int8);

#else // defined(REAL_MULTIPLIER)

    values0 = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(values0, OUTPUT_MULTIPLIER, OUTPUT_SHIFT, 8);

#endif // defined(REAL_MULTIPLIER)

    values0 += (int8)OUTPUT_OFFSET;
    uchar8 res0 = convert_uchar8_sat(values0);
    res0        = max(res0, (uchar8)0);
    res0        = min(res0, (uchar8)255);

    vstore8(ACTIVATION_FUNC(res0), 0, dst.ptr);
#if CONV_STRIDE_Y == 1

#if defined(REAL_MULTIPLIER)

    values1 = CONVERT(round(CONVERT(values1, float8) * (float8)REAL_MULTIPLIER), int8);

#else // defined(REAL_MULTIPLIER)

    values1 = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(values1, OUTPUT_MULTIPLIER, OUTPUT_SHIFT, 8);

#endif // defined(REAL_MULTIPLIER)

    values1 += (int8)OUTPUT_OFFSET;
    uchar8 res1 = convert_uchar8_sat(values1);
    res1        = max(res1, (uchar8)0);
    res1        = min(res1, (uchar8)255);

    vstore8(ACTIVATION_FUNC(res1), 0, dst.ptr + dst_stride_y);
#endif /* CONV_STRIDE_Y == 1 */
}

#endif // defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)

#endif /* defined(CONV_STRIDE_Y) && defined(CONV_STRIDE_X) && defined(DEPTH_MULTIPLIER) && defined(DST_CHANNELS) */

#if defined(VEC_SIZE) && defined(SRC_DIM_1) && defined(SRC_DIM_2) && defined(CONV_PAD_TOP) && defined(CONV_PAD_LEFT)

#define asymm_mult_by_quant_multiplier_less_than_one(x, y, z) ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(x, y, z, VEC_SIZE)

#define VEC_INT VEC_DATA_TYPE(int, VEC_SIZE)
#define VEC_FLOAT VEC_DATA_TYPE(float, VEC_SIZE)
#define VEC_UCHAR VEC_DATA_TYPE(uchar, VEC_SIZE)
#define VEC_USHORT VEC_DATA_TYPE(ushort, VEC_SIZE)

#define MULTIPLY_ADD(x, y, acc) acc += CONVERT(CONVERT(x, VEC_USHORT) * CONVERT(y, VEC_USHORT), VEC_INT)

#if WEIGHTS_OFFSET != 0
#define MULTIPLY_ADD_ACCUMULATE(x, y, acc, sum) \
    ({                                          \
        sum += CONVERT(x, VEC_INT);             \
        MULTIPLY_ADD(x, y, acc);                \
    })
#else /* WEIGHTS_OFFSET != 0 */
#define MULTIPLY_ADD_ACCUMULATE(x, y, acc, sum) MULTIPLY_ADD(x, y, acc)
#endif /* WEIGHTS_OFFSET != 0 */

#if defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)
#define DOT_PRODUCT(acc, val0, val1, val2, val3, val4, val5, val6, val7, val8, w0, w1) \
    ({                                                                                 \
        ARM_DOT((uchar4)(val0, val1, val2, val3), w0.s0123, acc);                      \
        ARM_DOT((uchar4)(val4, val5, val6, val7), w0.s4567, acc);                      \
        acc += val8 * w1;                                                              \
    })

#define DOT_PRODUCT_REDUCTION(sum, val0, val1, val2, val3, val4, val5, val6, val7, val8) \
    ({                                                                                   \
        sum = val0;                                                                      \
        ARM_DOT((uchar4)(val1, val2, val3, val4), (uchar4)1, sum);                       \
        ARM_DOT((uchar4)(val5, val6, val7, val8), (uchar4)1, sum);                       \
    })

#define DOT_PRODUCT_REDUCTION_WEIGHTS(sum, w0, w1) \
    ({                                             \
        sum = w1;                                  \
        ARM_DOT(w0.s0123, (uchar4)1, sum);         \
        ARM_DOT(w0.s4567, (uchar4)1, sum);         \
    })

#endif // defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)

#if defined(CONV_STRIDE_X) && defined(CONV_STRIDE_Y)
/** This function computes the depthwise convolution quantized for NHWC data layout when the stride along the width or height is not 1.
 *
 * @note The weights tensor is expected to be reshaped using @ref CLDepthwiseConvolutionLayerReshapeWeightsKernel.
 * @note The number of elements read per thread must be passed at compile time using -DVEC_SIZE (e.g. -DVEC_SIZE=2)
 * @note Dimension two of the input tensor (height for NHWC data layout) must be passed at compile time using -DSRC_DIM2 (e.g. -DSRC_DIM_2=112)
 * @note The convolution pad top must be passed at compile time using -DCONV_PAD_TOP (e.g. -DCONV_PAD_TOP=1)
 * @note The convolution pad top must be passed at compile time using -DCONV_PAD_LEFT (e.g. -DCONV_PAD_LEFT=1)
 * @note The convolution stride along the width must be passed at compile time using -DCONV_STRIDE_X (e.g. -DCONV_STRIDE_Y=X)
 * @note The convolution stride along the height must be passed at compile time using -DCONV_STRIDE_Y (e.g. -DCONV_STRIDE_Y=1)
 *
 * @param[in] src_ptr                               Pointer to the source tensor. Supported data types: QASYMM8
 * @param[in] src_stride_x                          Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                          Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                            src_stride_y * number of elements along Z processed per workitem(in bytes)
 * @param[in] src_stride_w                          Stride of the source tensor in W dimension (in bytes)
 * @param[in] src_step_w                            src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source tensor
 * @param[in] dst_ptr                               Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in] dst_stride_x                          Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                            dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                          Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                            dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                          Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                            dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_w                          Stride of the destination tensor in W dimension (in bytes)
 * @param[in] dst_step_w                            dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
 * @param[in] weights_ptr                           Pointer to the weights tensor reshaped. Supported data types: same as @p src_ptr
 * @param[in] weights_stride_x                      Stride of the weights tensor in X dimension (in bytes)
 * @param[in] weights_step_x                        weights_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] weights_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
 * @param[in] weights_step_y                        weights_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_offset_first_element_in_bytes The offset of the first element in the weights tensor
 * @param[in] biases_ptr                            (Optional) Pointer to the biases vector. Supported data types: same as @p src_ptr
 * @param[in] biases_stride_x                       (Optional) Stride of the biases vector in X dimension (in bytes)
 * @param[in] biases_step_x                         (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] biases_offset_first_element_in_bytes  (Optional) The offset of the first element in the biases vector
 * @param[in] max_offset                            Max offset for the input tensor
 */
__kernel void depthwise_convolution_3x3_quantized_nhwc(
    TENSOR4D_DECLARATION(src),
    TENSOR4D_DECLARATION(dst),
    IMAGE_DECLARATION(weights),
#if defined(HAS_BIAS)
    VECTOR_DECLARATION(biases),
#endif /* defined(HAS_BIAS) */
    int max_offset)
{
    const int x = get_global_id(0); // channels
    const int y = get_global_id(1); // spatial coordinate x
#if defined(DST_DEPTH)
    int z = get_global_id(2) % (int)DST_DEPTH; // spatial coordinate y
    int b = get_global_id(2) / (int)DST_DEPTH; // batch
#else // defined(DST_DEPTH)
    int z = get_global_id(2); // spatial coordinate y
#endif // defined(DST_DEPTH)

    __global uchar *weights_addr = weights_ptr + weights_offset_first_element_in_bytes + x * weights_stride_y;

#if defined(DST_DEPTH)
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * VEC_SIZE + b * src_stride_w;
#else  /* defined(DST_DEPTH) */
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * VEC_SIZE;
#endif /* defined(DST_DEPTH) */

    int  z_coord = 0;
    int4 offset  = 0;
    int4 y_coord = ((int4)(y * CONV_STRIDE_X) + (int4)(0, 1, 2, 3)) - (int)CONV_PAD_LEFT;

    // Only for y = 0 we can have a negative coordinate. If so, we convert it to SRC_DIM_1
    y_coord.s0 = min((uint)y_coord.s0, (uint)SRC_DIM_1);
    y_coord.s1 = min((uint)y_coord.s1, (uint)SRC_DIM_1);
    y_coord.s2 = min((uint)y_coord.s2, (uint)SRC_DIM_1);
    y_coord.s3 = min((uint)y_coord.s3, (uint)SRC_DIM_1);

    int4 y_offset = convert_int4(y_coord * (int)src_stride_y);

    // We compute VEC_SIZEx1x1 [C,W,H] elements
    VEC_INT acc = 0, sum = 0;

    // Load weights
    VEC_UCHAR w0 = VLOAD(VEC_SIZE)(0, weights_addr + 0);
    VEC_UCHAR w1 = VLOAD(VEC_SIZE)(0, weights_addr + VEC_SIZE);
    VEC_UCHAR w2 = VLOAD(VEC_SIZE)(0, weights_addr + 2 * VEC_SIZE);
    VEC_UCHAR w3 = VLOAD(VEC_SIZE)(0, weights_addr + 3 * VEC_SIZE);
    VEC_UCHAR w4 = VLOAD(VEC_SIZE)(0, weights_addr + 4 * VEC_SIZE);
    VEC_UCHAR w5 = VLOAD(VEC_SIZE)(0, weights_addr + 5 * VEC_SIZE);
    VEC_UCHAR w6 = VLOAD(VEC_SIZE)(0, weights_addr + 6 * VEC_SIZE);
    VEC_UCHAR w7 = VLOAD(VEC_SIZE)(0, weights_addr + 7 * VEC_SIZE);
    VEC_UCHAR w8 = VLOAD(VEC_SIZE)(0, weights_addr + 8 * VEC_SIZE);

#if INPUT_OFFSET != 0
    VEC_INT sum_we = CONVERT(w0, VEC_INT) + CONVERT(w1, VEC_INT) + CONVERT(w2, VEC_INT)
                     + CONVERT(w3, VEC_INT) + CONVERT(w4, VEC_INT) + CONVERT(w5, VEC_INT)
                     + CONVERT(w6, VEC_INT) + CONVERT(w7, VEC_INT) + CONVERT(w8, VEC_INT);
#endif /* INPUT_OFFSET != 0 */

    // Load input values
    // z == 0
    // Clamp z_coord as for z = 0, it can be negative
    // z_coord is casted to unsigned int in order to use just a min() operation
    // A "-1" 32 bit signed variable converted to unsigned gives 4294967295
    z_coord = z * (int)CONV_STRIDE_Y - (int)CONV_PAD_TOP;
    z_coord = min((uint)z_coord, (uint)SRC_DIM_2);
    offset  = y_offset + (int4)(z_coord * src_stride_z);
    offset  = min(offset, (int4)max_offset);

    VEC_UCHAR values0 = VLOAD(VEC_SIZE)(0, src_addr + offset.s0);
    VEC_UCHAR values1 = VLOAD(VEC_SIZE)(0, src_addr + offset.s1);
    VEC_UCHAR values2 = VLOAD(VEC_SIZE)(0, src_addr + offset.s2);

    // z == 1
    // z_coord can be only negative for z = 0 so we do not need to clamp it
    // Moreover z_coord cannot be out-of-bound for z = 1 so we do not need to clamp the offset
    z_coord           = z * (int)CONV_STRIDE_Y - (int)CONV_PAD_TOP + 1;
    offset            = y_offset + (int4)(z_coord * src_stride_z);
    VEC_UCHAR values3 = VLOAD(VEC_SIZE)(0, src_addr + offset.s0);
    VEC_UCHAR values4 = VLOAD(VEC_SIZE)(0, src_addr + offset.s1);
    VEC_UCHAR values5 = VLOAD(VEC_SIZE)(0, src_addr + offset.s2);

    // z == 2
    // After z = 1 we can simply add src_stride_z to offset without updating z_coord
    // However offset can be out-of-bound so we need to check if it is greater than max_offset
    offset += (int4)src_stride_z;
    offset            = min(offset, (int4)max_offset);
    VEC_UCHAR values6 = VLOAD(VEC_SIZE)(0, src_addr + offset.s0);
    VEC_UCHAR values7 = VLOAD(VEC_SIZE)(0, src_addr + offset.s1);
    VEC_UCHAR values8 = VLOAD(VEC_SIZE)(0, src_addr + offset.s2);

    MULTIPLY_ADD_ACCUMULATE(values0, w0, acc, sum);
    MULTIPLY_ADD_ACCUMULATE(values1, w1, acc, sum);
    MULTIPLY_ADD_ACCUMULATE(values2, w2, acc, sum);

    MULTIPLY_ADD_ACCUMULATE(values3, w3, acc, sum);
    MULTIPLY_ADD_ACCUMULATE(values4, w4, acc, sum);
    MULTIPLY_ADD_ACCUMULATE(values5, w5, acc, sum);

    MULTIPLY_ADD_ACCUMULATE(values6, w6, acc, sum);
    MULTIPLY_ADD_ACCUMULATE(values7, w7, acc, sum);
    MULTIPLY_ADD_ACCUMULATE(values8, w8, acc, sum);

#if defined(HAS_BIAS)
    Vector  biases      = CONVERT_TO_VECTOR_STRUCT(biases);
    VEC_INT bias_values = VLOAD(VEC_SIZE)(0, (__global int *)biases.ptr);
    acc += bias_values;
#endif // defined(HAS_BIAS)

#if WEIGHTS_OFFSET != 0
    acc += WEIGHTS_OFFSET * sum;
#endif /* WEIGHTS_OFFSET != 0 */

#if INPUT_OFFSET != 0
    acc += INPUT_OFFSET * sum_we;
#endif /* INPUT_OFFSET != 0 */

#if K_OFFSET != 0
    acc += (VEC_INT)K_OFFSET;
#endif /* K_OFFSET != 0 */

#if defined(REAL_MULTIPLIER)

    acc = CONVERT(round(CONVERT(acc, VEC_FLOAT) * (VEC_FLOAT)REAL_MULTIPLIER), VEC_INT);

#else  // defined(REAL_MULTIPLIER)

    acc = asymm_mult_by_quant_multiplier_less_than_one(acc, OUTPUT_MULTIPLIER, OUTPUT_SHIFT);
#endif // defined(REAL_MULTIPLIER)

    acc += (VEC_INT)OUTPUT_OFFSET;

    VEC_UCHAR res = CONVERT_SAT(acc, VEC_UCHAR);
    res           = CLAMP(res, (VEC_UCHAR)0, (VEC_UCHAR)255);

#if defined(DST_DEPTH)
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x * dst_step_x + y * dst_step_y + z * dst_step_z + b * dst_stride_w;
#else  /* defined(DST_DEPTH) */
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x * dst_step_x + y * dst_step_y + z * dst_step_z;
#endif /* defined(DST_DEPTH) */

    VSTORE(VEC_SIZE)
    (ACTIVATION_FUNC(res), 0, dst_addr);
}
#endif // defined(CONV_STRIDE_X) && defined(CONV_STRIDE_Y)

#if defined(NUM_ROWS_PROCESSED) && defined(NUM_PLANES_PROCESSED)
/** This function computes the depthwise convolution quantized for NHWC data layout when the stride along the width and height is 1.
 *
 * @note The weights tensor is expected to be reshaped using @ref CLDepthwiseConvolutionLayerReshapeWeightsKernel.
 * @note The number of elements read per thread must be passed at compile time using -DVEC_SIZE (e.g. -DVEC_SIZE=2)
 * @note Dimension two of the input tensor (height for NHWC data layout) must be passed at compile time using -DSRC_DIM2 (e.g. -DSRC_DIM_2=112)
 * @note The number of rows processed per thread must be passed at compile time using -DNUM_ROWS_PROCESSED (i.e. -DNUM_ROWS_PROCESSED=2)
 * @note The number of planes processed per thread must be passed at compile time using -DNUM_PLANES_PROCESSED (i.e. -DNUM_PLANES_PROCESSED=2)
 * @note The convolution pad top must be passed at compile time using -DCONV_PAD_TOP (e.g. -DCONV_PAD_TOP=1)
 * @note The convolution pad top must be passed at compile time using -DCONV_PAD_LEFT (e.g. -DCONV_PAD_LEFT=1).
 *
 * @param[in] src_ptr                               Pointer to the source tensor. Supported data types: QASYMM8
 * @param[in] src_stride_x                          Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                          Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                            src_stride_y * number of elements along Z processed per workitem(in bytes)
 * @param[in] src_stride_w                          Stride of the source tensor in W dimension (in bytes)
 * @param[in] src_step_w                            src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source tensor
 * @param[in] dst_ptr                               Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in] dst_stride_x                          Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                            dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                          Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                            dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                          Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                            dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_w                          Stride of the destination tensor in W dimension (in bytes)
 * @param[in] dst_step_w                            dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
 * @param[in] weights_ptr                           Pointer to the weights tensor. Supported data types: same as @p src_ptr
 * @param[in] weights_stride_x                      Stride of the weights tensor in X dimension (in bytes)
 * @param[in] weights_step_x                        weights_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] weights_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
 * @param[in] weights_step_y                        weights_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_offset_first_element_in_bytes The offset of the first element in the weights tensor
 * @param[in] biases_ptr                            (Optional) Pointer to the biases vector. Supported data types: same as @p src_ptr
 * @param[in] biases_stride_x                       (Optional) Stride of the biases vector in X dimension (in bytes)
 * @param[in] biases_step_x                         (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] biases_offset_first_element_in_bytes  (Optional) The offset of the first element in the biases vector
 * @param[in] max_offset                            Max offset for the input tensor
 */

__kernel void depthwise_convolution_3x3_quantized_nhwc_stride1(
    TENSOR4D_DECLARATION(src),
    TENSOR4D_DECLARATION(dst),
    IMAGE_DECLARATION(weights),
#if defined(HAS_BIAS)
    VECTOR_DECLARATION(biases),
#endif /* defined(HAS_BIAS) */
    int max_offset)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
#if defined(DST_DEPTH)
    int z = get_global_id(2) % (int)DST_DEPTH; // spatial coordinate y
    int b = get_global_id(2) / (int)DST_DEPTH; // batch
#else // defined(DST_DEPTH)
    int z = get_global_id(2); // spatial coordinate y
#endif // defined(DST_DEPTH)

    __global uchar *weights_addr = weights_ptr + weights_offset_first_element_in_bytes + x * weights_stride_y;

#if defined(DST_DEPTH)
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * VEC_SIZE + b * src_stride_w;
#else  /* defined(DST_DEPTH) */
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * VEC_SIZE;
#endif /* defined(DST_DEPTH) */

    int  z_coord = 0;
    int4 offset  = 0;
    int4 y_coord = ((int4)(y * NUM_ROWS_PROCESSED) + (int4)(0, 1, 2, 3)) - (int)CONV_PAD_LEFT;

    // Only for y = 0 we can have a negative coordinate. If so, we convert it to SRC_DIM_1
    y_coord.s0 = min((uint)y_coord.s0, (uint)SRC_DIM_1);
    y_coord.s1 = min((uint)y_coord.s1, (uint)SRC_DIM_1);
    y_coord.s2 = min((uint)y_coord.s2, (uint)SRC_DIM_1);
    y_coord.s3 = min((uint)y_coord.s3, (uint)SRC_DIM_1);

    int4 y_offset = convert_int4(y_coord * (int)src_stride_y);

    // We compute 4x2x2 [C,W,H] elements
    VEC_INT acc0 = 0, sum0 = 0;
    VEC_INT acc1 = 0, sum1 = 0;
    VEC_INT acc2 = 0, sum2 = 0;
    VEC_INT acc3 = 0, sum3 = 0;

    // Load weights
    VEC_UCHAR w0 = VLOAD(VEC_SIZE)(0, weights_addr + 0);
    VEC_UCHAR w1 = VLOAD(VEC_SIZE)(0, weights_addr + VEC_SIZE);
    VEC_UCHAR w2 = VLOAD(VEC_SIZE)(0, weights_addr + 2 * VEC_SIZE);
    VEC_UCHAR w3 = VLOAD(VEC_SIZE)(0, weights_addr + 3 * VEC_SIZE);
    VEC_UCHAR w4 = VLOAD(VEC_SIZE)(0, weights_addr + 4 * VEC_SIZE);
    VEC_UCHAR w5 = VLOAD(VEC_SIZE)(0, weights_addr + 5 * VEC_SIZE);
    VEC_UCHAR w6 = VLOAD(VEC_SIZE)(0, weights_addr + 6 * VEC_SIZE);
    VEC_UCHAR w7 = VLOAD(VEC_SIZE)(0, weights_addr + 7 * VEC_SIZE);
    VEC_UCHAR w8 = VLOAD(VEC_SIZE)(0, weights_addr + 8 * VEC_SIZE);

#if INPUT_OFFSET != 0
    VEC_INT sum_we = CONVERT(w0, VEC_INT) + CONVERT(w1, VEC_INT) + CONVERT(w2, VEC_INT)
                     + CONVERT(w3, VEC_INT) + CONVERT(w4, VEC_INT) + CONVERT(w5, VEC_INT)
                     + CONVERT(w6, VEC_INT) + CONVERT(w7, VEC_INT) + CONVERT(w8, VEC_INT);
#endif /* INPUT_OFFSET != 0 */

    // Load input values
    // z == 0
    // Clamp z_coord as for z = 0, it can be negative
    // z_coord is casted to unsigned int in order to use just a min() operation
    // A "-1" 32 bit signed variable converted to unsigned gives 4294967295
    z_coord = z * (int)NUM_PLANES_PROCESSED - (int)CONV_PAD_TOP;
    z_coord = min((uint)z_coord, (uint)SRC_DIM_2);
    offset  = y_offset + (int4)(z_coord * src_stride_z);
    offset  = min(offset, (int4)max_offset);

    VEC_UCHAR values0 = VLOAD(VEC_SIZE)(0, src_addr + offset.s0);
    VEC_UCHAR values1 = VLOAD(VEC_SIZE)(0, src_addr + offset.s1);
    VEC_UCHAR values2 = VLOAD(VEC_SIZE)(0, src_addr + offset.s2);
    VEC_UCHAR values3 = VLOAD(VEC_SIZE)(0, src_addr + offset.s3);

    // z == 1
    // z_coord can be only negative for z = 0 so we do not need to clamp it
    // Moreover z_coord cannot be out-of-bound for z = 1 so we do not need to clamp the offset
    z_coord           = z * (int)NUM_PLANES_PROCESSED - (int)CONV_PAD_TOP + 1;
    offset            = y_offset + (int4)(z_coord * src_stride_z);
    VEC_UCHAR values4 = VLOAD(VEC_SIZE)(0, src_addr + offset.s0);
    VEC_UCHAR values5 = VLOAD(VEC_SIZE)(0, src_addr + offset.s1);
    VEC_UCHAR values6 = VLOAD(VEC_SIZE)(0, src_addr + offset.s2);
    VEC_UCHAR values7 = VLOAD(VEC_SIZE)(0, src_addr + offset.s3);

    // z == 2
    // After z = 1 we can simply add src_stride_z to offset without updating z_coord
    // However offset can be out-of-bound so we need to check if it is greater than max_offset
    offset += (int4)src_stride_z;
    offset             = min(offset, (int4)max_offset);
    VEC_UCHAR values8  = VLOAD(VEC_SIZE)(0, src_addr + offset.s0);
    VEC_UCHAR values9  = VLOAD(VEC_SIZE)(0, src_addr + offset.s1);
    VEC_UCHAR values10 = VLOAD(VEC_SIZE)(0, src_addr + offset.s2);
    VEC_UCHAR values11 = VLOAD(VEC_SIZE)(0, src_addr + offset.s3);

    // z == 3
    // After z = 1 we can simply add src_stride_z to offset without updating z_coord
    // However offset can be out-of-bound so we need to check if it is greater than max_offset
    offset += (int4)(src_stride_z);
    offset             = min(offset, (int4)max_offset);
    VEC_UCHAR values12 = VLOAD(VEC_SIZE)(0, src_addr + offset.s0);
    VEC_UCHAR values13 = VLOAD(VEC_SIZE)(0, src_addr + offset.s1);
    VEC_UCHAR values14 = VLOAD(VEC_SIZE)(0, src_addr + offset.s2);
    VEC_UCHAR values15 = VLOAD(VEC_SIZE)(0, src_addr + offset.s3);

    MULTIPLY_ADD_ACCUMULATE(values0, w0, acc0, sum0);
    MULTIPLY_ADD_ACCUMULATE(values1, w1, acc0, sum0);
    MULTIPLY_ADD_ACCUMULATE(values2, w2, acc0, sum0);
    MULTIPLY_ADD_ACCUMULATE(values1, w0, acc1, sum1);
    MULTIPLY_ADD_ACCUMULATE(values2, w1, acc1, sum1);
    MULTIPLY_ADD_ACCUMULATE(values3, w2, acc1, sum1);

    MULTIPLY_ADD_ACCUMULATE(values4, w3, acc0, sum0);
    MULTIPLY_ADD_ACCUMULATE(values5, w4, acc0, sum0);
    MULTIPLY_ADD_ACCUMULATE(values6, w5, acc0, sum0);
    MULTIPLY_ADD_ACCUMULATE(values5, w3, acc1, sum1);
    MULTIPLY_ADD_ACCUMULATE(values6, w4, acc1, sum1);
    MULTIPLY_ADD_ACCUMULATE(values7, w5, acc1, sum1);

    MULTIPLY_ADD_ACCUMULATE(values8, w6, acc0, sum0);
    MULTIPLY_ADD_ACCUMULATE(values9, w7, acc0, sum0);
    MULTIPLY_ADD_ACCUMULATE(values10, w8, acc0, sum0);
    MULTIPLY_ADD_ACCUMULATE(values9, w6, acc1, sum1);
    MULTIPLY_ADD_ACCUMULATE(values10, w7, acc1, sum1);
    MULTIPLY_ADD_ACCUMULATE(values11, w8, acc1, sum1);

    MULTIPLY_ADD_ACCUMULATE(values4, w0, acc2, sum2);
    MULTIPLY_ADD_ACCUMULATE(values5, w1, acc2, sum2);
    MULTIPLY_ADD_ACCUMULATE(values6, w2, acc2, sum2);
    MULTIPLY_ADD_ACCUMULATE(values5, w0, acc3, sum3);
    MULTIPLY_ADD_ACCUMULATE(values6, w1, acc3, sum3);
    MULTIPLY_ADD_ACCUMULATE(values7, w2, acc3, sum3);

    MULTIPLY_ADD_ACCUMULATE(values8, w3, acc2, sum2);
    MULTIPLY_ADD_ACCUMULATE(values9, w4, acc2, sum2);
    MULTIPLY_ADD_ACCUMULATE(values10, w5, acc2, sum2);
    MULTIPLY_ADD_ACCUMULATE(values9, w3, acc3, sum3);
    MULTIPLY_ADD_ACCUMULATE(values10, w4, acc3, sum3);
    MULTIPLY_ADD_ACCUMULATE(values11, w5, acc3, sum3);

    MULTIPLY_ADD_ACCUMULATE(values12, w6, acc2, sum2);
    MULTIPLY_ADD_ACCUMULATE(values13, w7, acc2, sum2);
    MULTIPLY_ADD_ACCUMULATE(values14, w8, acc2, sum2);
    MULTIPLY_ADD_ACCUMULATE(values13, w6, acc3, sum3);
    MULTIPLY_ADD_ACCUMULATE(values14, w7, acc3, sum3);
    MULTIPLY_ADD_ACCUMULATE(values15, w8, acc3, sum3);

#if defined(HAS_BIAS)
    Vector biases = CONVERT_TO_VECTOR_STRUCT(biases);

    VEC_INT bias_values = VLOAD(VEC_SIZE)(0, (__global int *)biases.ptr);

    acc0 += bias_values;
    acc1 += bias_values;
    acc2 += bias_values;
    acc3 += bias_values;
#endif /* defined(HAS_BIAS) */

#if WEIGHTS_OFFSET != 0
    acc0 += WEIGHTS_OFFSET * sum0;
    acc1 += WEIGHTS_OFFSET * sum1;
    acc2 += WEIGHTS_OFFSET * sum2;
    acc3 += WEIGHTS_OFFSET * sum3;
#endif /* WEIGHTS_OFFSET != 0 */

#if INPUT_OFFSET != 0
    VEC_INT offs = INPUT_OFFSET * sum_we;

    acc0 += offs;
    acc1 += offs;
    acc2 += offs;
    acc3 += offs;
#endif /* INPUT_OFFSET != 0 */

#if K_OFFSET != 0
    acc0 += (VEC_INT)K_OFFSET;
    acc1 += (VEC_INT)K_OFFSET;
    acc2 += (VEC_INT)K_OFFSET;
    acc3 += (VEC_INT)K_OFFSET;
#endif /* K_OFFSET != 0 */

#if defined(REAL_MULTIPLIER)

    acc0 = CONVERT(round(CONVERT(acc0, VEC_FLOAT) * (VEC_FLOAT)REAL_MULTIPLIER), VEC_INT);
    acc1 = CONVERT(round(CONVERT(acc1, VEC_FLOAT) * (VEC_FLOAT)REAL_MULTIPLIER), VEC_INT);
    acc2 = CONVERT(round(CONVERT(acc2, VEC_FLOAT) * (VEC_FLOAT)REAL_MULTIPLIER), VEC_INT);
    acc3 = CONVERT(round(CONVERT(acc3, VEC_FLOAT) * (VEC_FLOAT)REAL_MULTIPLIER), VEC_INT);

#else // defined(REAL_MULTIPLIER)

    acc0 = asymm_mult_by_quant_multiplier_less_than_one(acc0, OUTPUT_MULTIPLIER, OUTPUT_SHIFT);
    acc1 = asymm_mult_by_quant_multiplier_less_than_one(acc1, OUTPUT_MULTIPLIER, OUTPUT_SHIFT);
    acc2 = asymm_mult_by_quant_multiplier_less_than_one(acc2, OUTPUT_MULTIPLIER, OUTPUT_SHIFT);
    acc3 = asymm_mult_by_quant_multiplier_less_than_one(acc3, OUTPUT_MULTIPLIER, OUTPUT_SHIFT);

#endif // defined(REAL_MULTIPLIER)

    acc0 += (VEC_INT)OUTPUT_OFFSET;
    acc1 += (VEC_INT)OUTPUT_OFFSET;
    acc2 += (VEC_INT)OUTPUT_OFFSET;
    acc3 += (VEC_INT)OUTPUT_OFFSET;

    VEC_UCHAR res0 = CONVERT_SAT(acc0, VEC_UCHAR);
    VEC_UCHAR res1 = CONVERT_SAT(acc1, VEC_UCHAR);
    VEC_UCHAR res2 = CONVERT_SAT(acc2, VEC_UCHAR);
    VEC_UCHAR res3 = CONVERT_SAT(acc3, VEC_UCHAR);

    res0 = CLAMP(res0, (VEC_UCHAR)0, (VEC_UCHAR)255);
    res1 = CLAMP(res1, (VEC_UCHAR)0, (VEC_UCHAR)255);
    res2 = CLAMP(res2, (VEC_UCHAR)0, (VEC_UCHAR)255);
    res3 = CLAMP(res3, (VEC_UCHAR)0, (VEC_UCHAR)255);

#if defined(DST_DEPTH)
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x * dst_step_x + y * dst_step_y + (z * NUM_PLANES_PROCESSED) * dst_step_z + b * dst_stride_w;
#else  /* defined(DST_DEPTH) */
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x * dst_step_x + y * dst_step_y + (z * NUM_PLANES_PROCESSED) * dst_step_z;
#endif /* defined(DST_DEPTH) */

    VSTORE(VEC_SIZE)
    (ACTIVATION_FUNC(res0), 0, dst_addr + 0 * dst_stride_y);
    VSTORE(VEC_SIZE)
    (ACTIVATION_FUNC(res1), 0, dst_addr + 1 * dst_stride_y);

#if((DST_DIM_2 % NUM_PLANES_PROCESSED) != 0)
    if((z * NUM_PLANES_PROCESSED + 1) < DST_DIM_2)
#endif // ((DST_DIM_2 % NUM_PLANES_PROCESSED) != 0)
    {
        VSTORE(VEC_SIZE)
        (ACTIVATION_FUNC(res2), 0, dst_addr + 0 * dst_stride_y + 1 * dst_stride_z);
        VSTORE(VEC_SIZE)
        (ACTIVATION_FUNC(res3), 0, dst_addr + 1 * dst_stride_y + 1 * dst_stride_z);
    }
}

#if defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8) && VEC_SIZE == 4
/** This function computes the depthwise convolution quantized for NHWC data layout when the stride along the width and height is 1 using dot product.
 *
 * @note This kernel assumes VEC_SIZE is 4.
 * @note The weights tensor is expected to be reshaped using @ref CLDepthwiseConvolutionLayerReshapeWeightsKernel.
 * @note The number of elements read per thread must be passed at compile time using -DVEC_SIZE (e.g. -DVEC_SIZE=2)
 * @note Dimension two of the input tensor (height for NHWC data layout) must be passed at compile time using -DSRC_DIM2 (e.g. -DSRC_DIM_2=112)
 * @note The number of rows processed per thread must be passed at compile time using -DNUM_ROWS_PROCESSED (i.e. -DNUM_ROWS_PROCESSED=2)
 * @note The number of planes processed per thread must be passed at compile time using -DNUM_PLANES_PROCESSED (i.e. -DNUM_PLANES_PROCESSED=2)
 * @note The convolution pad top must be passed at compile time using -DCONV_PAD_TOP (e.g. -DCONV_PAD_TOP=1)
 * @note The convolution pad top must be passed at compile time using -DCONV_PAD_LEFT (e.g. -DCONV_PAD_LEFT=1).
 * @note If REAL_MULTIPLIER is passed at compile time (i.e. -DREAL_MULTIPLIER=1.355f), the final quantization is performed using a floating point multiplication.
 *       If not, the quantization will be performed using a fixed point multiplication
 *
 * @param[in] src_ptr                               Pointer to the source tensor. Supported data types: QASYMM8
 * @param[in] src_stride_x                          Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                          Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                            src_stride_y * number of elements along Z processed per workitem(in bytes)
 * @param[in] src_stride_w                          Stride of the source tensor in W dimension (in bytes)
 * @param[in] src_step_w                            src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source tensor
 * @param[in] dst_ptr                               Pointer to the destination tensor. Supported data types: QASYMM8
 * @param[in] dst_stride_x                          Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                            dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                          Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                            dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                          Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                            dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_w                          Stride of the destination tensor in W dimension (in bytes)
 * @param[in] dst_step_w                            dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
 * @param[in] weights_ptr                           Pointer to the weights tensor. Supported data types: QASYMM8
 * @param[in] weights_stride_x                      Stride of the weights tensor in X dimension (in bytes)
 * @param[in] weights_step_x                        weights_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] weights_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
 * @param[in] weights_step_y                        weights_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_offset_first_element_in_bytes The offset of the first element in the weights tensor
 * @param[in] biases_ptr                            (Optional) Pointer to the biases vector. Supported data types: QASYMM8
 * @param[in] biases_stride_x                       (Optional) Stride of the biases vector in X dimension (in bytes)
 * @param[in] biases_step_x                         (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] biases_offset_first_element_in_bytes  (Optional) The offset of the first element in the biases vector
 * @param[in] max_offset                            The maximum allowed offset for the input tensor
 */
__kernel void depthwise_convolution_3x3_quantized_dot8_nhwc_stride1(
    TENSOR4D_DECLARATION(src),
    TENSOR4D_DECLARATION(dst),
    IMAGE_DECLARATION(weights),
#if defined(HAS_BIAS)
    VECTOR_DECLARATION(biases),
#endif // defined(HAS_BIAS)
    int max_offset)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
#if defined(DST_DEPTH)
    int z = get_global_id(2) % (int)DST_DEPTH; // spatial coordinate y
    int b = get_global_id(2) / (int)DST_DEPTH; // batch
#else // defined(DST_DEPTH)
    int z = get_global_id(2); // spatial coordinate y
#endif // defined(DST_DEPTH)

    __global uchar *weights_addr = weights_ptr + weights_offset_first_element_in_bytes + x * weights_stride_y;

#if defined(DST_DEPTH)
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * VEC_SIZE + b * src_stride_w;
#else  /* defined(DST_DEPTH) */
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * VEC_SIZE;
#endif /* defined(DST_DEPTH) */

    int  z_coord = 0;
    int4 offset  = 0;
    int4 y_coord = ((int4)(y * NUM_ROWS_PROCESSED) + (int4)(0, 1, 2, 3)) - (int)CONV_PAD_LEFT;

    // Only for y = 0 we can have a negative coordinate. If so, we convert it to SRC_DIM_1
    y_coord.s0 = min((uint)y_coord.s0, (uint)SRC_DIM_1);
    y_coord.s1 = min((uint)y_coord.s1, (uint)SRC_DIM_1);
    y_coord.s2 = min((uint)y_coord.s2, (uint)SRC_DIM_1);
    y_coord.s3 = min((uint)y_coord.s3, (uint)SRC_DIM_1);

    int4 y_offset = convert_int4(y_coord * (int)src_stride_y);

    // We compute 4x2x1 [C,W,H] elements
    VEC_INT acc0 = 0;
    VEC_INT acc1 = 0;
    VEC_INT sum0 = 0;
    VEC_INT sum1 = 0;

    // Load weights
    uchar16 w0 = VLOAD(16)(0, weights_addr);
    uchar16 w1 = VLOAD(16)(0, weights_addr + 16);
    uchar4  w2 = VLOAD(4)(0, weights_addr + 32);

#if INPUT_OFFSET != 0
    // Initilize the final result with the weights reduction multiplied by INPUT_OFFSET
    DOT_PRODUCT_REDUCTION_WEIGHTS(acc0.s0, w0.s01234567, w0.s8);
    DOT_PRODUCT_REDUCTION_WEIGHTS(acc0.s1, (uchar8)((w0.s9ABC), (w0.sDEF), w1.s0), w1.s1);
    DOT_PRODUCT_REDUCTION_WEIGHTS(acc0.s2, w1.s23456789, w1.sA);
    DOT_PRODUCT_REDUCTION_WEIGHTS(acc0.s3, (uchar8)((w1.sBCD), (w1.sEF), (w2.s012)), w2.s3);

    // Multiply the weights reduction with INPUT_OFFSET
    acc0 = INPUT_OFFSET * acc0;

    acc1 = acc0;
#endif // INPUT_OFFSET != 0

    // Load input values
    // z == 0
    // Clamp z_coord as for z = 0, it can be negative
    // z_coord is casted to unsigned int in order to use just a min() operation
    // A "-1" 32 bit signed variable converted to unsigned gives 4294967295
    z_coord = z - (int)CONV_PAD_TOP;
    z_coord = min((uint)z_coord, (uint)SRC_DIM_2);
    offset  = y_offset + (int4)(z_coord * src_stride_z);
    offset  = min(offset, (int4)max_offset);

    VEC_UCHAR values0 = VLOAD(VEC_SIZE)(0, src_addr + offset.s0);
    VEC_UCHAR values1 = VLOAD(VEC_SIZE)(0, src_addr + offset.s1);
    VEC_UCHAR values2 = VLOAD(VEC_SIZE)(0, src_addr + offset.s2);
    VEC_UCHAR values3 = VLOAD(VEC_SIZE)(0, src_addr + offset.s3);

    // z == 1
    // z_coord can be only negative for z = 0 so we do not need to clamp it
    // Moreover z_coord cannot be out-of-bound for z = 1 so we do not need to clamp the offset
    z_coord           = z - (int)CONV_PAD_TOP + 1;
    offset            = y_offset + (int4)(z_coord * src_stride_z);
    VEC_UCHAR values4 = VLOAD(VEC_SIZE)(0, src_addr + offset.s0);
    VEC_UCHAR values5 = VLOAD(VEC_SIZE)(0, src_addr + offset.s1);
    VEC_UCHAR values6 = VLOAD(VEC_SIZE)(0, src_addr + offset.s2);
    VEC_UCHAR values7 = VLOAD(VEC_SIZE)(0, src_addr + offset.s3);

    // z == 2
    // After z = 1 we can simply add src_stride_z to offset without updating z_coord
    // However offset can be out-of-bound so we need to check if it is greater than max_offset
    offset += (int4)src_stride_z;
    offset             = min(offset, (int4)max_offset);
    VEC_UCHAR values8  = VLOAD(VEC_SIZE)(0, src_addr + offset.s0);
    VEC_UCHAR values9  = VLOAD(VEC_SIZE)(0, src_addr + offset.s1);
    VEC_UCHAR values10 = VLOAD(VEC_SIZE)(0, src_addr + offset.s2);
    VEC_UCHAR values11 = VLOAD(VEC_SIZE)(0, src_addr + offset.s3);

    DOT_PRODUCT_REDUCTION(sum0.s0, values0.s0, values1.s0, values2.s0, values4.s0, values5.s0, values6.s0, values8.s0, values9.s0, values10.s0);
    DOT_PRODUCT_REDUCTION(sum1.s0, values1.s0, values2.s0, values3.s0, values5.s0, values6.s0, values7.s0, values9.s0, values10.s0, values11.s0);
    DOT_PRODUCT(acc0.s0, values0.s0, values1.s0, values2.s0, values4.s0, values5.s0, values6.s0, values8.s0, values9.s0, values10.s0, w0.s01234567, w0.s8);
    DOT_PRODUCT(acc1.s0, values1.s0, values2.s0, values3.s0, values5.s0, values6.s0, values7.s0, values9.s0, values10.s0, values11.s0, w0.s01234567, w0.s8);

    DOT_PRODUCT_REDUCTION(sum0.s1, values0.s1, values1.s1, values2.s1, values4.s1, values5.s1, values6.s1, values8.s1, values9.s1, values10.s1);
    DOT_PRODUCT_REDUCTION(sum1.s1, values1.s1, values2.s1, values3.s1, values5.s1, values6.s1, values7.s1, values9.s1, values10.s1, values11.s1);
    DOT_PRODUCT(acc0.s1, values0.s1, values1.s1, values2.s1, values4.s1, values5.s1, values6.s1, values8.s1, values9.s1, values10.s1, (uchar8)((w0.s9ABC), (w0.sDEF), w1.s0), w1.s1);
    DOT_PRODUCT(acc1.s1, values1.s1, values2.s1, values3.s1, values5.s1, values6.s1, values7.s1, values9.s1, values10.s1, values11.s1, (uchar8)((w0.s9ABC), (w0.sDEF), w1.s0), w1.s1);

    DOT_PRODUCT_REDUCTION(sum0.s2, values0.s2, values1.s2, values2.s2, values4.s2, values5.s2, values6.s2, values8.s2, values9.s2, values10.s2);
    DOT_PRODUCT_REDUCTION(sum1.s2, values1.s2, values2.s2, values3.s2, values5.s2, values6.s2, values7.s2, values9.s2, values10.s2, values11.s2);
    DOT_PRODUCT(acc0.s2, values0.s2, values1.s2, values2.s2, values4.s2, values5.s2, values6.s2, values8.s2, values9.s2, values10.s2, w1.s23456789, w1.sA);
    DOT_PRODUCT(acc1.s2, values1.s2, values2.s2, values3.s2, values5.s2, values6.s2, values7.s2, values9.s2, values10.s2, values11.s2, w1.s23456789, w1.sA);

    DOT_PRODUCT_REDUCTION(sum0.s3, values0.s3, values1.s3, values2.s3, values4.s3, values5.s3, values6.s3, values8.s3, values9.s3, values10.s3);
    DOT_PRODUCT_REDUCTION(sum1.s3, values1.s3, values2.s3, values3.s3, values5.s3, values6.s3, values7.s3, values9.s3, values10.s3, values11.s3);
    DOT_PRODUCT(acc0.s3, values0.s3, values1.s3, values2.s3, values4.s3, values5.s3, values6.s3, values8.s3, values9.s3, values10.s3, (uchar8)((w1.sBCD), (w1.sEF), (w2.s012)), w2.s3);
    DOT_PRODUCT(acc1.s3, values1.s3, values2.s3, values3.s3, values5.s3, values6.s3, values7.s3, values9.s3, values10.s3, values11.s3, (uchar8)((w1.sBCD), (w1.sEF), (w2.s012)), w2.s3);

#if defined(HAS_BIAS)
    Vector biases = CONVERT_TO_VECTOR_STRUCT(biases);

    VEC_INT bias_values = VLOAD(VEC_SIZE)(0, (__global int *)biases.ptr);

    acc0 += bias_values;
    acc1 += bias_values;

#endif // defined(HAS_BIAS)

#if WEIGHTS_OFFSET != 0
    acc0 += WEIGHTS_OFFSET * sum0;
    acc1 += WEIGHTS_OFFSET * sum1;
#endif // WEIGHTS_OFFSET != 0

#if K_OFFSET != 0
    acc0 += (VEC_INT)K_OFFSET;
    acc1 += (VEC_INT)K_OFFSET;

#endif // K_OFFSET != 0

#if defined(REAL_MULTIPLIER)

    acc0 = CONVERT(round(CONVERT(acc0, VEC_FLOAT) * (VEC_FLOAT)REAL_MULTIPLIER), VEC_INT);
    acc1 = CONVERT(round(CONVERT(acc1, VEC_FLOAT) * (VEC_FLOAT)REAL_MULTIPLIER), VEC_INT);

#else // defined(REAL_MULTIPLIER)

    acc0 = asymm_mult_by_quant_multiplier_less_than_one(acc0, OUTPUT_MULTIPLIER, OUTPUT_SHIFT);
    acc1 = asymm_mult_by_quant_multiplier_less_than_one(acc1, OUTPUT_MULTIPLIER, OUTPUT_SHIFT);

#endif // defined(REAL_MULTIPLIER)
    acc0 += (VEC_INT)OUTPUT_OFFSET;
    acc1 += (VEC_INT)OUTPUT_OFFSET;

    VEC_UCHAR res0 = CONVERT_SAT(acc0, VEC_UCHAR);
    VEC_UCHAR res1 = CONVERT_SAT(acc1, VEC_UCHAR);

    res0 = CLAMP(res0, (VEC_UCHAR)0, (VEC_UCHAR)255);
    res1 = CLAMP(res1, (VEC_UCHAR)0, (VEC_UCHAR)255);

#if defined(DST_DEPTH)
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x * dst_step_x + y * dst_step_y + z * dst_step_z + b * dst_stride_w;
#else  /* defined(DST_DEPTH) */
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x * dst_step_x + y * dst_step_y + z * dst_step_z;
#endif /* defined(DST_DEPTH) */

    VSTORE(VEC_SIZE)
    (ACTIVATION_FUNC(res0), 0, dst_addr + 0 * dst_stride_y);
    VSTORE(VEC_SIZE)
    (ACTIVATION_FUNC(res1), 0, dst_addr + 1 * dst_stride_y);
}
#endif // defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8) && VEC_SIZE==4

#endif // defined(NUM_ROWS_PROCESSED) && defined(NUM_PLANES_PROCESSED)

#endif // defined(VEC_SIZE) && defined(SRC_DIM_1) && defined(SRC_DIM_2) && defined(CONV_PAD_TOP) && defined(CONV_PAD_LEFT)

#endif // defined(WEIGHTS_OFFSET) && defined(INPUT_OFFSET) && defined(K_OFFSET) && ((defined(OUTPUT_OFFSET) && defined(OUTPUT_MULTIPLIER) && defined(OUTPUT_SHIFT)) || defined(REAL_MULTIPLIER))
