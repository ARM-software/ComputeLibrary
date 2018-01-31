/*
 * Copyright (c) 2017-2018 ARM Limited.
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

#if defined(WEIGHTS_OFFSET) && defined(INPUT_OFFSET) && defined(K_OFFSET) && defined(OUTPUT_OFFSET) && defined(OUTPUT_MULTIPLIER) && defined(OUTPUT_SHIFT)

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

#if defined(CONV_STRIDE_Y) && defined(CONV_STRIDE_X)

#if CONV_STRIDE_X > 3
#error "Stride X not supported"
#endif /* CONV_STRIDE_X > 3 */

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
 * @param[in] src_ptr                               Pointer to the source image. Supported data types: QASYMM8
 * @param[in] src_stride_x                          Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                          Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source image
 * @param[in] src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                            src_stride_z * number of elements along Y processed per workitem(in bytes)
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
    Tensor3D weights = CONVERT_TO_TENSOR3D_STRUCT(weights);
#if defined(HAS_BIAS)
    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);

    int bias_value = *((__global int *)(vector_offset(&biases, get_global_id(2))));
#endif //defined(HAS_BIAS)

    uchar3 w0 = vload3(0, weights.ptr + 0 * weights_stride_y);
    uchar3 w1 = vload3(0, weights.ptr + 1 * weights_stride_y);
    uchar3 w2 = vload3(0, weights.ptr + 2 * weights_stride_y);

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

    values0 = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(values0, OUTPUT_MULTIPLIER, OUTPUT_SHIFT, 8);
    values0 += (int8)OUTPUT_OFFSET;
    uchar8 res0 = convert_uchar8_sat(values0);
    res0        = max(res0, (uchar8)0);
    res0        = min(res0, (uchar8)255);

    vstore8(ACTIVATION_FUNC(res0), 0, dst.ptr);
#if CONV_STRIDE_Y == 1

    values1 = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(values1, OUTPUT_MULTIPLIER, OUTPUT_SHIFT, 8);
    values1 += (int8)OUTPUT_OFFSET;
    uchar8 res1 = convert_uchar8_sat(values1);
    res1        = max(res1, (uchar8)0);
    res1        = min(res1, (uchar8)255);

    vstore8(ACTIVATION_FUNC(res1), 0, dst.ptr + dst_stride_y);
#endif /* CONV_STRIDE_Y == 1 */
}

#endif /* defined(CONV_STRIDE_Y) && defined(CONV_STRIDE_X) */

#if defined(VEC_SIZE) && defined(SRC_DEPTH) && defined(CONV_PAD_TOP) && defined(ROWS_READ)

#define asymm_mult_by_quant_multiplier_less_than_one(x, y, z) ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(x, y, z, VEC_SIZE)

#define VEC_INT VEC_DATA_TYPE(int, VEC_SIZE)
#define VEC_UCHAR VEC_DATA_TYPE(uchar, VEC_SIZE)

#define BIFROST_MAD_4(acc, x, y)               \
    ({                                         \
        acc.s0 += (ushort)x.s0 * (ushort)y.s0; \
        acc.s1 += (ushort)x.s1 * (ushort)y.s1; \
        acc.s2 += (ushort)x.s2 * (ushort)y.s2; \
        acc.s3 += (ushort)x.s3 * (ushort)y.s3; \
    })

#if WEIGHTS_OFFSET != 0
#define BIFROST_MAD_ACC_4(acc, sum, x, y) \
    ({                                    \
        sum += CONVERT(x, VEC_INT);       \
        BIFROST_MAD_4(acc, x, y);         \
    })
#else /* WEIGHTS_OFFSET != 0 */
#define BIFROST_MAD_ACC_4(acc, sum, x, y) BIFROST_MAD_4(acc, x, y)
#endif /* WEIGHTS_OFFSET != 0 */

/** This function computes the depthwise convolution quantized.
 *
 * @param[in] src_ptr                               Pointer to the source image. Supported data types: QASYMM8
 * @param[in] src_stride_x                          Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                          Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source image
 * @param[in] src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                            src_stride_z * number of elements along Y processed per workitem(in bytes)
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

__kernel void depthwise_convolution_3x3_quantized_nhwc_stride1(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(weights),
#if defined(HAS_BIAS)
    VECTOR_DECLARATION(biases)
#endif /* defined(HAS_BIAS) */
)
{
    Image  dst     = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);
    Vector weights = CONVERT_TO_VECTOR_STRUCT(weights);
#if defined(HAS_BIAS)
    Vector biases = CONVERT_TO_VECTOR_STRUCT(biases);

    VEC_INT bias_values = VLOAD(VEC_SIZE)(0, (__global int *)biases.ptr);
#endif /* defined(HAS_BIAS) */

    __global uchar *first_elem = src_ptr + src_offset_first_element_in_bytes;

    const int z         = get_global_id(2);
    const int pad_offs  = -ROWS_READ * src_stride_y;
    const int src_offs0 = get_global_id(0) * src_step_x + get_global_id(1) * src_step_y + z * src_step_z - CONV_PAD_TOP * src_stride_z;
    const int src_offs1 = src_offs0 + src_stride_z;
    const int src_offs2 = src_offs1 + src_stride_z;

    const int cond_top    = z - CONV_PAD_TOP < 0;
    const int cond_bottom = z * (src_step_z / src_stride_z) + 2 >= SRC_DEPTH;

    __global uchar *src_addr0 = first_elem + select(src_offs0, pad_offs, cond_top);
    __global uchar *src_addr1 = first_elem + src_offs1;
    __global uchar *src_addr2 = first_elem + select(src_offs2, pad_offs, cond_bottom);

    VEC_INT sum_we = 0;
    VEC_INT acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    VEC_INT sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;

    // z == 0
    VEC_UCHAR w0, w1, w2;
    w0 = VLOAD(VEC_SIZE)(0, weights.ptr + 0 * weights_stride_y);
    w1 = VLOAD(VEC_SIZE)(0, weights.ptr + 1 * weights_stride_y);
    w2 = VLOAD(VEC_SIZE)(0, weights.ptr + 2 * weights_stride_y);

#if INPUT_OFFSET != 0
    sum_we += CONVERT(w0, VEC_INT) + CONVERT(w1, VEC_INT) + CONVERT(w2, VEC_INT);
#endif /* INPUT_OFFSET != 0 */

    VEC_UCHAR values = VLOAD(VEC_SIZE)(0, src_addr0);
    BIFROST_MAD_ACC_4(acc0, sum0, values, w0);

    src_addr0 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr0);
    BIFROST_MAD_ACC_4(acc0, sum0, values, w1);
    BIFROST_MAD_ACC_4(acc1, sum1, values, w0);

    src_addr0 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr0);
    BIFROST_MAD_ACC_4(acc0, sum0, values, w2);
    BIFROST_MAD_ACC_4(acc1, sum1, values, w1);
    BIFROST_MAD_ACC_4(acc2, sum2, values, w0);

    src_addr0 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr0);
    BIFROST_MAD_ACC_4(acc1, sum1, values, w2);
    BIFROST_MAD_ACC_4(acc2, sum2, values, w1);
    BIFROST_MAD_ACC_4(acc3, sum3, values, w0);

    src_addr0 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr0);
    BIFROST_MAD_ACC_4(acc2, sum2, values, w2);
    BIFROST_MAD_ACC_4(acc3, sum3, values, w1);

    src_addr0 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr0);
    BIFROST_MAD_ACC_4(acc3, sum3, values, w2);

    weights.ptr += weights_stride_z;

    // z == 1
    w0 = VLOAD(VEC_SIZE)(0, weights.ptr + 0 * weights_stride_y);
    w1 = VLOAD(VEC_SIZE)(0, weights.ptr + 1 * weights_stride_y);
    w2 = VLOAD(VEC_SIZE)(0, weights.ptr + 2 * weights_stride_y);

#if INPUT_OFFSET != 0
    sum_we += CONVERT(w0, VEC_INT) + CONVERT(w1, VEC_INT) + CONVERT(w2, VEC_INT);
#endif /* INPUT_OFFSET != 0 */

    values = VLOAD(VEC_SIZE)(0, src_addr1);
    BIFROST_MAD_ACC_4(acc0, sum0, values, w0);

    src_addr1 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr1);
    BIFROST_MAD_ACC_4(acc0, sum0, values, w1);
    BIFROST_MAD_ACC_4(acc1, sum1, values, w0);

    src_addr1 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr1);
    BIFROST_MAD_ACC_4(acc0, sum0, values, w2);
    BIFROST_MAD_ACC_4(acc1, sum1, values, w1);
    BIFROST_MAD_ACC_4(acc2, sum2, values, w0);

    src_addr1 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr1);
    BIFROST_MAD_ACC_4(acc1, sum1, values, w2);
    BIFROST_MAD_ACC_4(acc2, sum2, values, w1);
    BIFROST_MAD_ACC_4(acc3, sum3, values, w0);

    src_addr1 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr1);
    BIFROST_MAD_ACC_4(acc2, sum2, values, w2);
    BIFROST_MAD_ACC_4(acc3, sum3, values, w1);

    src_addr1 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr1);
    BIFROST_MAD_ACC_4(acc3, sum3, values, w2);

    weights.ptr += weights_stride_z;

    // z == 2
    w0 = VLOAD(VEC_SIZE)(0, weights.ptr + 0 * weights_stride_y);
    w1 = VLOAD(VEC_SIZE)(0, weights.ptr + 1 * weights_stride_y);
    w2 = VLOAD(VEC_SIZE)(0, weights.ptr + 2 * weights_stride_y);

#if INPUT_OFFSET != 0
    sum_we += CONVERT(w0, VEC_INT) + CONVERT(w1, VEC_INT) + CONVERT(w2, VEC_INT);
#endif /* INPUT_OFFSET != 0 */

    values = VLOAD(VEC_SIZE)(0, src_addr2);
    BIFROST_MAD_ACC_4(acc0, sum0, values, w0);

    src_addr2 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr2);
    BIFROST_MAD_ACC_4(acc0, sum0, values, w1);
    BIFROST_MAD_ACC_4(acc1, sum1, values, w0);

    src_addr2 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr2);
    BIFROST_MAD_ACC_4(acc0, sum0, values, w2);
    BIFROST_MAD_ACC_4(acc1, sum1, values, w1);
    BIFROST_MAD_ACC_4(acc2, sum2, values, w0);

    src_addr2 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr2);
    BIFROST_MAD_ACC_4(acc1, sum1, values, w2);
    BIFROST_MAD_ACC_4(acc2, sum2, values, w1);
    BIFROST_MAD_ACC_4(acc3, sum3, values, w0);

    src_addr2 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr2);
    BIFROST_MAD_ACC_4(acc2, sum2, values, w2);
    BIFROST_MAD_ACC_4(acc3, sum3, values, w1);

    src_addr2 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr2);
    BIFROST_MAD_ACC_4(acc3, sum3, values, w2);

#if defined(HAS_BIAS)
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

    acc0 = asymm_mult_by_quant_multiplier_less_than_one(acc0, OUTPUT_MULTIPLIER, OUTPUT_SHIFT);
    acc1 = asymm_mult_by_quant_multiplier_less_than_one(acc1, OUTPUT_MULTIPLIER, OUTPUT_SHIFT);
    acc2 = asymm_mult_by_quant_multiplier_less_than_one(acc2, OUTPUT_MULTIPLIER, OUTPUT_SHIFT);
    acc3 = asymm_mult_by_quant_multiplier_less_than_one(acc3, OUTPUT_MULTIPLIER, OUTPUT_SHIFT);

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

    VSTORE(VEC_SIZE)
    (res0, 0, dst.ptr + 0 * dst_stride_y);
    VSTORE(VEC_SIZE)
    (res1, 0, dst.ptr + 1 * dst_stride_y);
    VSTORE(VEC_SIZE)
    (res2, 0, dst.ptr + 2 * dst_stride_y);
    VSTORE(VEC_SIZE)
    (res3, 0, dst.ptr + 3 * dst_stride_y);
}

/** This function computes the depthwise convolution quantized.
 *
 * @param[in] src_ptr                               Pointer to the source image. Supported data types: QASYMM8
 * @param[in] src_stride_x                          Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                          Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes     The offset of the first element in the source image
 * @param[in] src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                            src_stride_z * number of elements along Y processed per workitem(in bytes)
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

__kernel void depthwise_convolution_3x3_quantized_nhwc_stride2(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(weights),
#if defined(HAS_BIAS)
    VECTOR_DECLARATION(biases)
#endif /* defined(HAS_BIAS) */
)
{
    Image  dst     = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);
    Vector weights = CONVERT_TO_VECTOR_STRUCT(weights);
#if defined(HAS_BIAS)
    Vector biases = CONVERT_TO_VECTOR_STRUCT(biases);

    VEC_INT bias_values = VLOAD(VEC_SIZE)(0, (__global int *)biases.ptr);
#endif /* defined(HAS_BIAS) */

    __global uchar *first_elem = src_ptr + src_offset_first_element_in_bytes;

    const int z         = get_global_id(2);
    const int pad_offs  = -ROWS_READ * src_stride_y;
    const int src_offs0 = get_global_id(0) * src_step_x + get_global_id(1) * src_step_y + z * src_step_z - CONV_PAD_TOP * src_stride_z;
    const int src_offs1 = src_offs0 + src_stride_z;
    const int src_offs2 = src_offs1 + src_stride_z;

    const int cond_top    = z - CONV_PAD_TOP < 0;
    const int cond_bottom = z * (src_step_z / src_stride_z) + 2 >= SRC_DEPTH;
    ;

    __global uchar *src_addr0 = first_elem + select(src_offs0, pad_offs, cond_top);
    __global uchar *src_addr1 = first_elem + src_offs1;
    __global uchar *src_addr2 = first_elem + select(src_offs2, pad_offs, cond_bottom);

    VEC_INT sum_we = 0;
    VEC_INT acc0 = 0, acc2 = 0;
    VEC_INT sum0 = 0, sum2 = 0;

    // z == 0
    VEC_UCHAR w0, w1, w2;
    w0 = VLOAD(VEC_SIZE)(0, weights.ptr + 0 * weights_stride_y);
    w1 = VLOAD(VEC_SIZE)(0, weights.ptr + 1 * weights_stride_y);
    w2 = VLOAD(VEC_SIZE)(0, weights.ptr + 2 * weights_stride_y);

#if INPUT_OFFSET != 0
    sum_we += CONVERT(w0, VEC_INT) + CONVERT(w1, VEC_INT) + CONVERT(w2, VEC_INT);
#endif /* INPUT_OFFSET != 0 */

    VEC_UCHAR values = VLOAD(VEC_SIZE)(0, src_addr0);
    BIFROST_MAD_ACC_4(acc0, sum0, values, w0);

    src_addr0 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr0);
    BIFROST_MAD_ACC_4(acc0, sum0, values, w1);

    src_addr0 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr0);
    BIFROST_MAD_ACC_4(acc0, sum0, values, w2);
    BIFROST_MAD_ACC_4(acc2, sum2, values, w0);

    src_addr0 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr0);
    BIFROST_MAD_ACC_4(acc2, sum2, values, w1);

    src_addr0 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr0);
    BIFROST_MAD_ACC_4(acc2, sum2, values, w2);

    weights.ptr += weights_stride_z;

    // z == 1
    w0 = VLOAD(VEC_SIZE)(0, weights.ptr + 0 * weights_stride_y);
    w1 = VLOAD(VEC_SIZE)(0, weights.ptr + 1 * weights_stride_y);
    w2 = VLOAD(VEC_SIZE)(0, weights.ptr + 2 * weights_stride_y);

#if INPUT_OFFSET != 0
    sum_we += CONVERT(w0, VEC_INT) + CONVERT(w1, VEC_INT) + CONVERT(w2, VEC_INT);
#endif /* INPUT_OFFSET != 0 */

    values = VLOAD(VEC_SIZE)(0, src_addr1);
    BIFROST_MAD_ACC_4(acc0, sum0, values, w0);

    src_addr1 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr1);
    BIFROST_MAD_ACC_4(acc0, sum0, values, w1);

    src_addr1 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr1);
    BIFROST_MAD_ACC_4(acc0, sum0, values, w2);
    BIFROST_MAD_ACC_4(acc2, sum2, values, w0);

    src_addr1 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr1);
    BIFROST_MAD_ACC_4(acc2, sum2, values, w1);

    src_addr1 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr1);
    BIFROST_MAD_ACC_4(acc2, sum2, values, w2);

    weights.ptr += weights_stride_z;

    // z == 2
    w0 = VLOAD(VEC_SIZE)(0, weights.ptr + 0 * weights_stride_y);
    w1 = VLOAD(VEC_SIZE)(0, weights.ptr + 1 * weights_stride_y);
    w2 = VLOAD(VEC_SIZE)(0, weights.ptr + 2 * weights_stride_y);

#if INPUT_OFFSET != 0
    sum_we += CONVERT(w0, VEC_INT) + CONVERT(w1, VEC_INT) + CONVERT(w2, VEC_INT);
#endif /* INPUT_OFFSET != 0 */

    values = VLOAD(VEC_SIZE)(0, src_addr2);
    BIFROST_MAD_ACC_4(acc0, sum0, values, w0);

    src_addr2 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr2);
    BIFROST_MAD_ACC_4(acc0, sum0, values, w1);

    src_addr2 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr2);
    BIFROST_MAD_ACC_4(acc0, sum0, values, w2);
    BIFROST_MAD_ACC_4(acc2, sum2, values, w0);

    src_addr2 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr2);
    BIFROST_MAD_ACC_4(acc2, sum2, values, w1);

    src_addr2 += src_stride_y;
    values = VLOAD(VEC_SIZE)(0, src_addr2);
    BIFROST_MAD_ACC_4(acc2, sum2, values, w2);

#if defined(HAS_BIAS)
    acc0 += bias_values;
    acc2 += bias_values;
#endif /* defined(HAS_BIAS) */

#if WEIGHTS_OFFSET != 0
    acc0 += WEIGHTS_OFFSET * sum0;
    acc2 += WEIGHTS_OFFSET * sum2;
#endif /* WEIGHTS_OFFSET != 0 */

#if INPUT_OFFSET != 0
    VEC_INT offs = INPUT_OFFSET * sum_we;

    acc0 += offs;
    acc2 += offs;
#endif /* INPUT_OFFSET != 0 */

#if K_OFFSET != 0
    acc0 += (VEC_INT)K_OFFSET;
    acc2 += (VEC_INT)K_OFFSET;
#endif /* K_OFFSET != 0 */

    acc0 = asymm_mult_by_quant_multiplier_less_than_one(acc0, OUTPUT_MULTIPLIER, OUTPUT_SHIFT);
    acc2 = asymm_mult_by_quant_multiplier_less_than_one(acc2, OUTPUT_MULTIPLIER, OUTPUT_SHIFT);
    acc0 += (VEC_INT)OUTPUT_OFFSET;
    acc2 += (VEC_INT)OUTPUT_OFFSET;
    VEC_UCHAR res0 = CONVERT_SAT(acc0, VEC_UCHAR);
    VEC_UCHAR res2 = CONVERT_SAT(acc2, VEC_UCHAR);
    res0           = CLAMP(res0, (VEC_UCHAR)0, (VEC_UCHAR)255);
    res2           = CLAMP(res2, (VEC_UCHAR)0, (VEC_UCHAR)255);

    VSTORE(VEC_SIZE)
    (res0, 0, dst.ptr + 0 * dst_stride_y);
    VSTORE(VEC_SIZE)
    (res2, 0, dst.ptr + 1 * dst_stride_y);
}

#endif /* defined(VEC_SIZE) && defined(SRC_DEPTH) && defined(CONV_PAD_TOP) && defined(ROWS_READ) */

#endif /* defined(WEIGHTS_OFFSET) && defined(INPUT_OFFSET) && defined(K_OFFSET) && defined(OUTPUT_OFFSET) && defined(OUTPUT_MULTIPLIER) && defined(OUTPUT_SHIFT) */
