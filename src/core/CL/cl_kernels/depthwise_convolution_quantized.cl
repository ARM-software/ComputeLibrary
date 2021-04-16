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

#include "helpers_asymm.h"

#ifndef VEC_SIZE
#if defined(N0)
#define VEC_SIZE N0
#else /* defined(N0) */
#define VEC_SIZE 8
#endif /* defined(N0) */
#endif /* VEC_SIZE */

#if defined(ACTIVATION_TYPE) && defined(CONST_0)
#include "activation_layer_quant.cl"
#define ACTIVATION_FUNC(x) PERFORM_ACTIVATION_QUANT(ACTIVATION_TYPE, x)
#else /* defined(ACTIVATION_TYPE) && defined(CONST_0) */
#define ACTIVATION_FUNC(x) (x)
#endif /* defined(ACTIVATION_TYPE) && defined(CONST_0) */

#define VEC_INT VEC_DATA_TYPE(int, VEC_SIZE)
#define VEC_FLOAT VEC_DATA_TYPE(float, VEC_SIZE)
#define VEC_SHORT VEC_DATA_TYPE(short, VEC_SIZE)

#if defined(DATA_TYPE) && defined(WEIGHTS_TYPE)

#define VEC_TYPE(size) VEC_DATA_TYPE(DATA_TYPE, size)

#if defined(WEIGHTS_OFFSET) && defined(INPUT_OFFSET) && defined(K_OFFSET) && ((defined(OUTPUT_OFFSET) && defined(OUTPUT_MULTIPLIER) && defined(OUTPUT_SHIFT)) || defined(REAL_MULTIPLIER))

#if defined(WEIGHTS_PROMOTED_TYPE)
#define VEC_WEIGHTS_PROMOTED_TYPE(size) VEC_DATA_TYPE(WEIGHTS_PROMOTED_TYPE, size)

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

#if !defined(IS_DOT8)

#if DILATION_X == 1

#if CONV_STRIDE_X == 1
#define GET_VALUES(first_value, left, middle, right)                                                        \
    ({                                                                                                      \
        int8 temp0 = CONVERT(vload8(0, (__global DATA_TYPE *)(first_value)), int8);                         \
        int2 temp1 = CONVERT(vload2(0, (__global DATA_TYPE *)(first_value + 8 * sizeof(DATA_TYPE))), int2); \
        \
        left   = CONVERT(temp0.s01234567, int8);                                                            \
        middle = CONVERT((int8)(temp0.s1234, temp0.s567, temp1.s0), int8);                                  \
        right  = CONVERT((int8)(temp0.s2345, temp0.s67, temp1.s01), int8);                                  \
    })
#elif CONV_STRIDE_X == 2
#define GET_VALUES(first_value, left, middle, right)                                                 \
    ({                                                                                               \
        int16 temp0 = CONVERT(vload16(0, (__global DATA_TYPE *)(first_value)), int16);               \
        int temp1   = CONVERT(*((__global DATA_TYPE *)(first_value + 16 * sizeof(DATA_TYPE))), int); \
        \
        left   = CONVERT(temp0.s02468ace, int8);                                                     \
        middle = CONVERT(temp0.s13579bdf, int8);                                                     \
        right  = CONVERT((int8)(temp0.s2468, temp0.sace, temp1), int8);                              \
    })
#else /* CONV_STRIDE_X */
#define GET_VALUES(first_value, left, middle, right)                                                          \
    ({                                                                                                        \
        int16 temp0 = CONVERT(vload16(0, (__global DATA_TYPE *)(first_value)), int16);                        \
        int8 temp1  = CONVERT(vload8(0, (__global DATA_TYPE *)(first_value + 16 * sizeof(DATA_TYPE))), int8); \
        \
        left   = CONVERT((int8)(temp0.s0369, temp0.scf, temp1.s25), int8);                                    \
        middle = CONVERT((int8)(temp0.s147a, temp0.sd, temp1.s036), int8);                                    \
        right  = CONVERT((int8)(temp0.s258b, temp0.se, temp1.s147), int8);                                    \
    })
#endif /* CONV_STRIDE_X */

#else /* DILATION_X == 1 */

#if CONV_STRIDE_X == 1
#define GET_VALUES(first_value, left, middle, right)                                                                 \
    ({                                                                                                               \
        left   = CONVERT(vload8(0, (__global DATA_TYPE *)(first_value)), int8);                                      \
        middle = CONVERT(vload8(0, (__global DATA_TYPE *)(first_value + DILATION_X * sizeof(DATA_TYPE))), int8);     \
        right  = CONVERT(vload8(0, (__global DATA_TYPE *)(first_value + 2 * DILATION_X * sizeof(DATA_TYPE))), int8); \
    })
#elif CONV_STRIDE_X == 2
#define GET_VALUES(first_value, left, middle, right)                                                                  \
    ({                                                                                                                \
        int16 temp0 = CONVERT(vload16(0, (__global DATA_TYPE *)(first_value)), int16);                                \
        left        = CONVERT(temp0.s02468ace, int8);                                                                 \
        \
        temp0  = CONVERT(vload16(0, (__global DATA_TYPE *)(first_value + DILATION_X * sizeof(DATA_TYPE))), int16);    \
        middle = CONVERT(temp0.s02468ace, int8);                                                                      \
        \
        temp0 = CONVERT(vload16(0, (__global DATA_TYPE *)(first_value + 2 * DILATION_X * sizeof(DATA_TYPE))), int16); \
        right = CONVERT(temp0.s02468ace, int8);                                                                       \
    })
#else /* CONV_STRIDE_X */
#define GET_VALUES(first_value, left, middle, right)                                                                       \
    ({                                                                                                                     \
        int16 temp0 = CONVERT(vload16(0, (__global DATA_TYPE *)(first_value)), int16);                                     \
        int8 temp1  = CONVERT(vload8(0, (__global DATA_TYPE *)(first_value + 16 * sizeof(DATA_TYPE))), int8);              \
        left        = CONVERT((int8)(temp0.s0369, temp0.scf, temp1.s25), int8);                                            \
        \
        temp0  = CONVERT(vload16(0, (__global DATA_TYPE *)(first_value + DILATION_X * sizeof(DATA_TYPE))), int16);         \
        temp1  = CONVERT(vload8(0, (__global DATA_TYPE *)(first_value + (16 + DILATION_X) * sizeof(DATA_TYPE))), int8);    \
        middle = CONVERT((int8)(temp0.s0369, temp0.scf, temp1.s25), int8);                                                 \
        \
        temp0 = CONVERT(vload16(0, (__global DATA_TYPE *)(first_value + 2 * DILATION_X * sizeof(DATA_TYPE))), int16);      \
        temp1 = CONVERT(vload8(0, (__global DATA_TYPE *)(first_value + (16 + 2 * DILATION_X) * sizeof(DATA_TYPE))), int8); \
        right = CONVERT((int8)(temp0.s0369, temp0.scf, temp1.s25), int8);                                                  \
    })

#endif /* CONV_STRIDE_X */
#endif /* DILATION_X==1 */

/** This function computes the depthwise convolution quantized.
 *
 * @param[in] src_ptr                                          Pointer to the source tensor. Supported data types: QASYMM8/QASYMM8_SIGNED
 * @param[in] src_stride_x                                     Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_step_x                                       src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                                     Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_step_y                                       src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_stride_z                                     Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                                       src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes                The offset of the first element in the source tensor
 * @param[in] dst_ptr                                          Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in] dst_stride_x                                     Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                                       dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                                     Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                                       dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                                     Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                                       dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes                The offset of the first element in the destination tensor
 * @param[in] weights_ptr                                      Pointer to the weights tensor. Supported data types: QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL
 * @param[in] weights_stride_x                                 Stride of the weights tensor in X dimension (in bytes)
 * @param[in] weights_step_x                                   weights_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] weights_stride_y                                 Stride of the weights tensor in Y dimension (in bytes)
 * @param[in] weights_step_y                                   weights_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_stride_z                                 Stride of the weights tensor in Z dimension (in bytes)
 * @param[in] weights_step_z                                   weights_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_offset_first_element_in_bytes            The offset of the first element in the weights tensor
 * @param[in] output_multipliers_ptr                           Pointer to the output multipliers vector. Supported data types: S32
 * @param[in] output_multipliers_stride_x                      Stride of the output multipliers vector in X dimension (in bytes)
 * @param[in] output_multipliers_step_x                        output_multipliers_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] output_multipliers_offset_first_element_in_bytes The offset of the first element in the output multipliers vector
 * @param[in] output_shifts_ptr                                Pointer to the output shifts vector. Supported data types: S32
 * @param[in] output_shifts_stride_x                           Stride of the output shifts vector in X dimension (in bytes)
 * @param[in] output_shifts_step_x                             output_shifts_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] output_shifts_offset_first_element_in_bytes      The offset of the first element in the output shifts vector
 * @param[in] biases_ptr                                       (Optional) Pointer to the biases vector. Supported data types: S32
 * @param[in] biases_stride_x                                  (Optional) Stride of the biases vector in X dimension (in bytes)
 * @param[in] biases_step_x                                    (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] biases_offset_first_element_in_bytes             (Optional) The offset of the first element in the biases vector
 */

__kernel void dwc_3x3_native_quantized8_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(weights),
    VECTOR_DECLARATION(output_multipliers),
    VECTOR_DECLARATION(output_shifts)
#if defined(HAS_BIAS)
    ,
    VECTOR_DECLARATION(biases)
#endif //defined(HAS_BIAS)
)
{
    __global uchar *src_addr           = src_ptr + get_global_id(0) * src_step_x + get_global_id(1) * src_step_y + get_global_id(2) * src_step_z;
    Image           dst                = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);
    Tensor3D        weights            = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(weights);
    Vector          output_multipliers = CONVERT_TO_VECTOR_STRUCT_NO_STEP(output_multipliers);
    Vector          output_shifts      = CONVERT_TO_VECTOR_STRUCT_NO_STEP(output_shifts);

    // Extract channel and linearized batch indices
    const int channel = get_global_id(2) % DST_CHANNELS;
    const int batch   = get_global_id(2) / DST_CHANNELS;

#if defined(HAS_BIAS)
    Vector biases = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);

    int bias_value = *((__global int *)(vector_offset(&biases, channel)));
#endif //defined(HAS_BIAS)

    // Load relevant input and weights data (Accounts depth multiplier when indexing input, OFM = IFM * DEPTH_MULTIPLIER)
    src_addr -= batch * (DST_CHANNELS / DEPTH_MULTIPLIER) * (DEPTH_MULTIPLIER - 1) * src_step_z + (channel - (channel / DEPTH_MULTIPLIER)) * src_step_z;
    __global uchar *weights_addr = weights.ptr + get_global_id(0) * weights_step_x + get_global_id(1) * weights_step_y + channel * weights_step_z;

    VEC_DATA_TYPE(WEIGHTS_TYPE, 3)
    w0 = vload3(0, (__global WEIGHTS_TYPE *)(weights_addr + 0 * weights_stride_y));
    VEC_DATA_TYPE(WEIGHTS_TYPE, 3)
    w1 = vload3(0, (__global WEIGHTS_TYPE *)(weights_addr + 1 * weights_stride_y));
    VEC_DATA_TYPE(WEIGHTS_TYPE, 3)
    w2 = vload3(0, (__global WEIGHTS_TYPE *)(weights_addr + 2 * weights_stride_y));

#if defined(PER_CHANNEL_QUANTIZATION)
    const int output_multiplier = *((__global int *)vector_offset(&output_multipliers, channel));
    const int output_shift      = *((__global int *)vector_offset(&output_shifts, channel));
#endif // defined(PER_CHANNEL_QUANTIZATION)

    int8 values0 = 0;
    int8 sum0    = 0;
#if CONV_STRIDE_Y == 1 && DILATION_Y == 1
    int8 values1 = 0;
    int8 sum1    = 0;
#endif /* CONV_STRIDE_Y &&DILATION_Y==1 */

    // Row0
    int8 left, middle, right;
    GET_VALUES(src_addr + 0 * src_stride_y, left, middle, right);
    values0 += left * (int8)(w0.s0);
    values0 += middle * (int8)(w0.s1);
    values0 += right * (int8)(w0.s2);

#if WEIGHTS_OFFSET != 0
    sum0 += left + middle + right;
#endif /* WEIGHTS_OFFSET != 0 */

    // Row1
    GET_VALUES(src_addr + DILATION_Y * src_stride_y, left, middle, right);
    values0 += left * (int8)(w1.s0);
    values0 += middle * (int8)(w1.s1);
    values0 += right * (int8)(w1.s2);

#if CONV_STRIDE_Y == 1 && DILATION_Y == 1
    values1 += left * (int8)(w0.s0);
    values1 += middle * (int8)(w0.s1);
    values1 += right * (int8)(w0.s2);
#endif /* CONV_STRIDE_Y && DILATION_Y== 1 */

#if WEIGHTS_OFFSET != 0
    int8 tmp = left + middle + right;
    sum0 += tmp;
#if CONV_STRIDE_Y == 1 && DILATION_Y == 1
    sum1 += tmp;
#endif /* CONV_STRIDE_Y &&DILATION_Y== 1 */
#endif /* WEIGHTS_OFFSET != 0 */

    // Row2
    GET_VALUES(src_addr + 2 * DILATION_Y * src_stride_y, left, middle, right);
    values0 += left * (int8)(w2.s0);
    values0 += middle * (int8)(w2.s1);
    values0 += right * (int8)(w2.s2);
#if CONV_STRIDE_Y == 1 && DILATION_Y == 1
    values1 += left * (int8)(w1.s0);
    values1 += middle * (int8)(w1.s1);
    values1 += right * (int8)(w1.s2);
#endif /* CONV_STRIDE_Y &&DILATION_Y == 1 */

#if WEIGHTS_OFFSET != 0
    tmp = left + middle + right;
    sum0 += tmp;
#if CONV_STRIDE_Y == 1 && DILATION_Y == 1
    sum1 += tmp;
#endif /* CONV_STRIDE_Y == 1 && DILATION_Y==1 */
#endif /* WEIGHTS_OFFSET != 0 */

#if CONV_STRIDE_Y == 1 && DILATION_Y == 1
    // Row3
    GET_VALUES(src_addr + 3 * src_stride_y, left, middle, right);
    values1 += left * (int8)(w2.s0);
    values1 += middle * (int8)(w2.s1);
    values1 += right * (int8)(w2.s2);

#if WEIGHTS_OFFSET != 0
    sum1 += left + middle + right;
#endif /* WEIGHTS_OFFSET != 0 */
#endif /* CONV_STRIDE_Y && DILATION_Y == 1 */

#if defined(HAS_BIAS)
    values0 += (int8)(bias_value);
#if CONV_STRIDE_Y == 1 && DILATION_Y == 1
    values1 += (int8)(bias_value);
#endif /* CONV_STRIDE_Y & &DILATION_Y == 1 */
#endif //defined(HAS_BIAS)

#if WEIGHTS_OFFSET != 0
    values0 += sum0 * (int8)(WEIGHTS_OFFSET);
#if CONV_STRIDE_Y == 1 && DILATION_Y == 1
    values1 += sum1 * (int8)(WEIGHTS_OFFSET);
#endif /* CONV_STRIDE_Y == 1 && DILATION_Y==1 */
#endif /* WEIGHTS_OFFSET != 0 */

#if INPUT_OFFSET != 0
    VEC_WEIGHTS_PROMOTED_TYPE(3)
    tmp_we = CONVERT(w0, VEC_WEIGHTS_PROMOTED_TYPE(3)) + CONVERT(w1, VEC_WEIGHTS_PROMOTED_TYPE(3)) + CONVERT(w2, VEC_WEIGHTS_PROMOTED_TYPE(3));

    WEIGHTS_PROMOTED_TYPE sum_weights = tmp_we.s0 + tmp_we.s1 + tmp_we.s2;
    values0 += sum_weights * (int8)(INPUT_OFFSET);
#if CONV_STRIDE_Y == 1 && DILATION_Y == 1
    values1 += sum_weights * (int8)(INPUT_OFFSET);
#endif /* CONV_STRIDE_Y == 1 && DILATION_Y==1 */
#endif /* INPUT_OFFSET != 0 */

#if K_OFFSET != 0
    values0 += (int8)(K_OFFSET);
#if CONV_STRIDE_Y == 1 && DILATION_Y == 1
    values1 += (int8)(K_OFFSET);
#endif /* CONV_STRIDE_Y == 1 && DILATION_Y==1*/
#endif /* K_OFFSET != 0 */

#if defined(REAL_MULTIPLIER)

    values0 = CONVERT(round(CONVERT(values0, float8) * (float8)REAL_MULTIPLIER), int8);

#else // defined(REAL_MULTIPLIER)

#if defined(PER_CHANNEL_QUANTIZATION)
    int8 res0_shift_lt0 = ASYMM_MULT_BY_QUANT_MULTIPLIER_GREATER_THAN_ONE(values0, output_multiplier, output_shift, 8);
    int8 res0_shift_gt0 = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(values0, output_multiplier, output_shift, 8);
    values0             = select(res0_shift_lt0, res0_shift_gt0, (int8)(output_shift) >= 0);
#else // defined(PER_CHANNEL_QUANTIZATION)
#if OUTPUT_SHIFT < 0
    values0 = ASYMM_MULT_BY_QUANT_MULTIPLIER_GREATER_THAN_ONE(values0, OUTPUT_MULTIPLIER, OUTPUT_SHIFT, 8);
#else  // OUTPUT_SHIFT < 0
    values0 = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(values0, OUTPUT_MULTIPLIER, OUTPUT_SHIFT, 8);
#endif // OUTPUT_OFFSET < 0
#endif // defined(PER_CHANNEL_QUANTIZATION)

#endif // defined(REAL_MULTIPLIER)

    values0 += (int8)OUTPUT_OFFSET;
    VEC_TYPE(8)
    res0 = CONVERT_SAT(values0, VEC_TYPE(8));

    vstore8(ACTIVATION_FUNC(res0), 0, dst.ptr);
#if CONV_STRIDE_Y == 1 && DILATION_Y == 1
#if defined(REAL_MULTIPLIER)

    values1 = CONVERT(round(CONVERT(values1, float8) * (float8)REAL_MULTIPLIER), int8);

#else // defined(REAL_MULTIPLIER)

#if defined(PER_CHANNEL_QUANTIZATION)
    int8 res1_shift_lt0 = ASYMM_MULT_BY_QUANT_MULTIPLIER_GREATER_THAN_ONE(values1, output_multiplier, output_shift, 8);
    int8 res1_shift_gt0 = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(values1, output_multiplier, output_shift, 8);
    values1             = select(res1_shift_lt0, res1_shift_gt0, (int8)(output_shift) >= 0);
#else // defined(PER_CHANNEL_QUANTIZATION)
#if OUTPUT_SHIFT < 0
    values1 = ASYMM_MULT_BY_QUANT_MULTIPLIER_GREATER_THAN_ONE(values1, OUTPUT_MULTIPLIER, OUTPUT_SHIFT, 8);
#else  // OUTPUT_SHIFT < 0
    values1 = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(values1, OUTPUT_MULTIPLIER, OUTPUT_SHIFT, 8);
#endif // OUTPUT_OFFSET < 0
#endif // defined(PER_CHANNEL_QUANTIZATION)

#endif // defined(REAL_MULTIPLIER)

    values1 += (int8)OUTPUT_OFFSET;
    VEC_TYPE(8)
    res1 = CONVERT_SAT(values1, VEC_TYPE(8));

    vstore8(ACTIVATION_FUNC(res1), 0, dst.ptr + dst_stride_y);
#endif /* CONV_STRIDE_Y == 1 && DILATION_Y==1*/
}

#else // !defined(IS_DOT8)

#if DILATION_X == 1
#if CONV_STRIDE_X == 1
#define GET_VALUES(first_value, left, middle, right)                                    \
    ({                                                                                  \
        VEC_TYPE(8)                                                                     \
        temp0 = vload8(0, (__global DATA_TYPE *)(first_value));                         \
        VEC_TYPE(2)                                                                     \
        temp1 = vload2(0, (__global DATA_TYPE *)(first_value + 8 * sizeof(DATA_TYPE))); \
        \
        left   = temp0.s01234567;                                                       \
        middle = (VEC_TYPE(8))(temp0.s1234, temp0.s567, temp1.s0);                      \
        right  = (VEC_TYPE(8))(temp0.s2345, temp0.s67, temp1.s01);                      \
    })
#elif CONV_STRIDE_X == 2
#define GET_VALUES(first_value, left, middle, right)                                       \
    ({                                                                                     \
        VEC_TYPE(16)                                                                       \
        temp0           = vload16(0, (__global DATA_TYPE *)(first_value));                 \
        DATA_TYPE temp1 = *((__global DATA_TYPE *)(first_value + 16 * sizeof(DATA_TYPE))); \
        \
        left   = temp0.s02468ace;                                                          \
        middle = temp0.s13579bdf;                                                          \
        right  = (VEC_TYPE(8))(temp0.s2468, temp0.sace, temp1);                            \
    })
#else /* CONV_STRIDE_X */
#define GET_VALUES(first_value, left, middle, right)                                     \
    ({                                                                                   \
        VEC_TYPE(16)                                                                     \
        temp0 = vload16(0, (__global DATA_TYPE *)(first_value));                         \
        VEC_TYPE(8)                                                                      \
        temp1 = vload8(0, (__global DATA_TYPE *)(first_value + 16 * sizeof(DATA_TYPE))); \
        \
        left   = (VEC_TYPE(8))(temp0.s0369, temp0.scf, temp1.s25);                       \
        middle = (VEC_TYPE(8))(temp0.s147a, temp0.sd, temp1.s036);                       \
        right  = (VEC_TYPE(8))(temp0.s258b, temp0.se, temp1.s147);                       \
    })
#endif /* CONV_STRIDE_X */
#else  /*DILATION_X==1*/

#if CONV_STRIDE_X == 1
#define GET_VALUES(first_value, left, middle, right)                                                  \
    ({                                                                                                \
        left   = vload8(0, (__global DATA_TYPE *)(first_value));                                      \
        middle = vload8(0, (__global DATA_TYPE *)(first_value + DILATION_X * sizeof(DATA_TYPE)));     \
        right  = vload8(0, (__global DATA_TYPE *)(first_value + 2 * DILATION_X * sizeof(DATA_TYPE))); \
    })
#elif CONV_STRIDE_X == 2
#define GET_VALUES(first_value, left, middle, right)                                                   \
    ({                                                                                                 \
        VEC_TYPE(16)                                                                                   \
        temp0  = vload16(0, (__global DATA_TYPE *)(first_value));                                      \
        left   = temp0.s02468ace;                                                                      \
        temp0  = vload16(0, (__global DATA_TYPE *)(first_value + DILATION_X * sizeof(DATA_TYPE)));     \
        middle = temp0.s02468ace;                                                                      \
        temp0  = vload16(0, (__global DATA_TYPE *)(first_value + 2 * DILATION_X * sizeof(DATA_TYPE))); \
        right  = temp0.s02468ace;                                                                      \
    })
#else /* CONV_STRIDE_X */
#define GET_VALUES(first_value, left, middle, right)                                                        \
    ({                                                                                                      \
        VEC_TYPE(16)                                                                                        \
        temp0 = vload16(0, (__global DATA_TYPE *)(first_value));                                            \
        VEC_TYPE(8)                                                                                         \
        temp1 = vload8(0, (__global DATA_TYPE *)(first_value + 16 * sizeof(DATA_TYPE)));                    \
        left  = (VEC_TYPE(8))(temp0.s0369, temp0.scf, temp1.s25);                                           \
        \
        temp0  = vload16(0, (__global DATA_TYPE *)(first_value + DILATION_X * sizeof(DATA_TYPE)));          \
        temp1  = vload8(0, (__global DATA_TYPE *)(first_value + (16 + DILATION_X) * sizeof(DATA_TYPE)));    \
        middle = (VEC_TYPE(8))(temp0.s0369, temp0.scf, temp1.s25);                                          \
        \
        temp0 = vload16(0, (__global DATA_TYPE *)(first_value + 2 * DILATION_X * sizeof(DATA_TYPE)));       \
        temp1 = vload8(0, (__global DATA_TYPE *)(first_value + (16 + 2 * DILATION_X) * sizeof(DATA_TYPE))); \
        right = (VEC_TYPE(8))(temp0.s0369, temp0.scf, temp1.s25);                                           \
    })

#endif /* CONV_STRIDE_X */
#endif /*DILATION_X==1*/
/** This function computes the depthwise convolution quantized using dot product when the data layout is NCHW.
 *
 * @param[in] src_ptr                                          Pointer to the source tensor. Supported data types: QASYMM8/QASYMM8_SIGNED
 * @param[in] src_stride_x                                     Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_step_x                                       src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                                     Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_step_y                                       src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_stride_z                                     Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                                       src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes                The offset of the first element in the source tensor
 * @param[in] dst_ptr                                          Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in] dst_stride_x                                     Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                                       dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                                     Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                                       dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                                     Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                                       dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes                The offset of the first element in the destination tensor
 * @param[in] weights_ptr                                      Pointer to the weights tensor. Supported data types: QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL
 * @param[in] weights_stride_x                                 Stride of the weights tensor in X dimension (in bytes)
 * @param[in] weights_step_x                                   weights_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] weights_stride_y                                 Stride of the weights tensor in Y dimension (in bytes)
 * @param[in] weights_step_y                                   weights_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_stride_z                                 Stride of the weights tensor in Z dimension (in bytes)
 * @param[in] weights_step_z                                   weights_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_offset_first_element_in_bytes            The offset of the first element in the weights tensor
 * @param[in] output_multipliers_ptr                           Pointer to the output multipliers vector. Supported data types: S32
 * @param[in] output_multipliers_stride_x                      Stride of the output multipliers vector in X dimension (in bytes)
 * @param[in] output_multipliers_step_x                        output_multipliers_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] output_multipliers_offset_first_element_in_bytes The offset of the first element in the output multipliers vector
 * @param[in] output_shifts_ptr                                Pointer to the output shifts vector. Supported data types: S32
 * @param[in] output_shifts_stride_x                           Stride of the output shifts vector in X dimension (in bytes)
 * @param[in] output_shifts_step_x                             output_shifts_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] output_shifts_offset_first_element_in_bytes      The offset of the first element in the output shifts vector
 * @param[in] biases_ptr                                       (Optional) Pointer to the biases vector. Supported data types: S32
 * @param[in] biases_stride_x                                  (Optional) Stride of the biases vector in X dimension (in bytes)
 * @param[in] biases_step_x                                    (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] biases_offset_first_element_in_bytes             (Optional) The offset of the first element in the biases vector
 */

__kernel void dwc_3x3_native_quantized8_dot8_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(weights),
    VECTOR_DECLARATION(output_multipliers),
    VECTOR_DECLARATION(output_shifts)
#if defined(HAS_BIAS)
    ,
    VECTOR_DECLARATION(biases)
#endif //defined(HAS_BIAS)
)
{
    __global uchar *src_addr           = src_ptr + get_global_id(0) * src_step_x + get_global_id(1) * src_step_y + get_global_id(2) * src_step_z;
    Image           dst                = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);
    Tensor3D        weights            = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(weights);
    Vector          output_multipliers = CONVERT_TO_VECTOR_STRUCT_NO_STEP(output_multipliers);
    Vector          output_shifts      = CONVERT_TO_VECTOR_STRUCT_NO_STEP(output_shifts);

    // Extract channel and linearized batch indices
    const int channel = get_global_id(2) % DST_CHANNELS;
    const int batch   = get_global_id(2) / DST_CHANNELS;

#if defined(HAS_BIAS)
    Vector    biases  = CONVERT_TO_VECTOR_STRUCT_NO_STEP(biases);

    const int bias_value = *((__global int *)(vector_offset(&biases, channel)));
#endif //defined(HAS_BIAS)

    // Load relevant input and weights data (Accounts depth multiplier when indexing input, OFM = IFM * DEPTH_MULTIPLIER)
    src_addr -= batch * (DST_CHANNELS / DEPTH_MULTIPLIER) * (DEPTH_MULTIPLIER - 1) * src_step_z + (channel - (channel / DEPTH_MULTIPLIER)) * src_step_z;
    __global uchar *weights_addr = weights.ptr + get_global_id(0) * weights_step_x + get_global_id(1) * weights_step_y + channel * weights_step_z;

    VEC_TYPE(3)
    w0 = vload3(0, (__global WEIGHTS_TYPE *)(weights_addr + 0 * weights_stride_y));
    VEC_TYPE(3)
    w1 = vload3(0, (__global WEIGHTS_TYPE *)(weights_addr + 1 * weights_stride_y));
    VEC_TYPE(3)
    w2 = vload3(0, (__global WEIGHTS_TYPE *)(weights_addr + 2 * weights_stride_y));

    const int output_multiplier = *((__global int *)vector_offset(&output_multipliers, 0));
    const int output_shift      = *((__global int *)vector_offset(&output_shifts, 0));

    VEC_TYPE(8)
    left0, middle0, right0;
    VEC_TYPE(8)
    left1, middle1, right1;
    VEC_TYPE(8)
    left2, middle2, right2;

    int8 values0 = 0;
    int8 sum0    = 0;

    GET_VALUES(src_addr + 0 * src_stride_y, left0, middle0, right0);
    GET_VALUES(src_addr + DILATION_Y * src_stride_y, left1, middle1, right1);
    GET_VALUES(src_addr + 2 * DILATION_Y * src_stride_y, left2, middle2, right2);

#if WEIGHTS_OFFSET != 0
    sum0 += convert_int8(left0) + convert_int8(middle0) + convert_int8(right0);
    sum0 += convert_int8(left1) + convert_int8(middle1) + convert_int8(right1);
    sum0 += convert_int8(left2) + convert_int8(middle2) + convert_int8(right2);
#endif /* WEIGHTS_OFFSET != 0 */

#if CONV_STRIDE_Y == 1 && DILATION_Y == 1
    // If conv_stride_y is equals to 1, we compute two output rows

    VEC_TYPE(8)
    left3, middle3, right3;
    int8 values1 = 0;
    int8 sum1    = 0;

    GET_VALUES(src_addr + 3 * src_stride_y, left3, middle3, right3);

#if WEIGHTS_OFFSET != 0
    sum1 += convert_int8(left1) + convert_int8(middle1) + convert_int8(right1);
    sum1 += convert_int8(left2) + convert_int8(middle2) + convert_int8(right2);
    sum1 += convert_int8(left3) + convert_int8(middle3) + convert_int8(right3);
#endif /* WEIGHTS_OFFSET != 0 */
#endif // CONV_STRIDE_Y == 1 && DILATION_Y==1

    ARM_DOT((VEC_TYPE(4))(left0.s0, middle0.s0, right0.s0, left1.s0), (VEC_TYPE(4))(w0.s0, w0.s1, w0.s2, w1.s0), values0.s0);
    ARM_DOT((VEC_TYPE(4))(middle1.s0, right1.s0, left2.s0, middle2.s0), (VEC_TYPE(4))(w1.s1, w1.s2, w2.s0, w2.s1), values0.s0);
    values0.s0 += right2.s0 * w2.s2;

    ARM_DOT((VEC_TYPE(4))(left0.s1, middle0.s1, right0.s1, left1.s1), (VEC_TYPE(4))(w0.s0, w0.s1, w0.s2, w1.s0), values0.s1);
    ARM_DOT((VEC_TYPE(4))(middle1.s1, right1.s1, left2.s1, middle2.s1), (VEC_TYPE(4))(w1.s1, w1.s2, w2.s0, w2.s1), values0.s1);
    values0.s1 += right2.s1 * w2.s2;

    ARM_DOT((VEC_TYPE(4))(left0.s2, middle0.s2, right0.s2, left1.s2), (VEC_TYPE(4))(w0.s0, w0.s1, w0.s2, w1.s0), values0.s2);
    ARM_DOT((VEC_TYPE(4))(middle1.s2, right1.s2, left2.s2, middle2.s2), (VEC_TYPE(4))(w1.s1, w1.s2, w2.s0, w2.s1), values0.s2);
    values0.s2 += right2.s2 * w2.s2;

    ARM_DOT((VEC_TYPE(4))(left0.s3, middle0.s3, right0.s3, left1.s3), (VEC_TYPE(4))(w0.s0, w0.s1, w0.s2, w1.s0), values0.s3);
    ARM_DOT((VEC_TYPE(4))(middle1.s3, right1.s3, left2.s3, middle2.s3), (VEC_TYPE(4))(w1.s1, w1.s2, w2.s0, w2.s1), values0.s3);
    values0.s3 += right2.s3 * w2.s2;

    ARM_DOT((VEC_TYPE(4))(left0.s4, middle0.s4, right0.s4, left1.s4), (VEC_TYPE(4))(w0.s0, w0.s1, w0.s2, w1.s0), values0.s4);
    ARM_DOT((VEC_TYPE(4))(middle1.s4, right1.s4, left2.s4, middle2.s4), (VEC_TYPE(4))(w1.s1, w1.s2, w2.s0, w2.s1), values0.s4);
    values0.s4 += right2.s4 * w2.s2;

    ARM_DOT((VEC_TYPE(4))(left0.s5, middle0.s5, right0.s5, left1.s5), (VEC_TYPE(4))(w0.s0, w0.s1, w0.s2, w1.s0), values0.s5);
    ARM_DOT((VEC_TYPE(4))(middle1.s5, right1.s5, left2.s5, middle2.s5), (VEC_TYPE(4))(w1.s1, w1.s2, w2.s0, w2.s1), values0.s5);
    values0.s5 += right2.s5 * w2.s2;

    ARM_DOT((VEC_TYPE(4))(left0.s6, middle0.s6, right0.s6, left1.s6), (VEC_TYPE(4))(w0.s0, w0.s1, w0.s2, w1.s0), values0.s6);
    ARM_DOT((VEC_TYPE(4))(middle1.s6, right1.s6, left2.s6, middle2.s6), (VEC_TYPE(4))(w1.s1, w1.s2, w2.s0, w2.s1), values0.s6);
    values0.s6 += right2.s6 * w2.s2;

    ARM_DOT((VEC_TYPE(4))(left0.s7, middle0.s7, right0.s7, left1.s7), (VEC_TYPE(4))(w0.s0, w0.s1, w0.s2, w1.s0), values0.s7);
    ARM_DOT((VEC_TYPE(4))(middle1.s7, right1.s7, left2.s7, middle2.s7), (VEC_TYPE(4))(w1.s1, w1.s2, w2.s0, w2.s1), values0.s7);
    values0.s7 += right2.s7 * w2.s2;

#if CONV_STRIDE_Y == 1 && DILATION_Y == 1
    ARM_DOT((VEC_TYPE(4))(left1.s0, middle1.s0, right1.s0, left2.s0), (VEC_TYPE(4))(w0.s0, w0.s1, w0.s2, w1.s0), values1.s0);
    ARM_DOT((VEC_TYPE(4))(middle2.s0, right2.s0, left3.s0, middle3.s0), (VEC_TYPE(4))(w1.s1, w1.s2, w2.s0, w2.s1), values1.s0);
    values1.s0 += right3.s0 * w2.s2;

    ARM_DOT((VEC_TYPE(4))(left1.s1, middle1.s1, right1.s1, left2.s1), (VEC_TYPE(4))(w0.s0, w0.s1, w0.s2, w1.s0), values1.s1);
    ARM_DOT((VEC_TYPE(4))(middle2.s1, right2.s1, left3.s1, middle3.s1), (VEC_TYPE(4))(w1.s1, w1.s2, w2.s0, w2.s1), values1.s1);
    values1.s1 += right3.s1 * w2.s2;

    ARM_DOT((VEC_TYPE(4))(left1.s2, middle1.s2, right1.s2, left2.s2), (VEC_TYPE(4))(w0.s0, w0.s1, w0.s2, w1.s0), values1.s2);
    ARM_DOT((VEC_TYPE(4))(middle2.s2, right2.s2, left3.s2, middle3.s2), (VEC_TYPE(4))(w1.s1, w1.s2, w2.s0, w2.s1), values1.s2);
    values1.s2 += right3.s2 * w2.s2;

    ARM_DOT((VEC_TYPE(4))(left1.s3, middle1.s3, right1.s3, left2.s3), (VEC_TYPE(4))(w0.s0, w0.s1, w0.s2, w1.s0), values1.s3);
    ARM_DOT((VEC_TYPE(4))(middle2.s3, right2.s3, left3.s3, middle3.s3), (VEC_TYPE(4))(w1.s1, w1.s2, w2.s0, w2.s1), values1.s3);
    values1.s3 += right3.s3 * w2.s2;

    ARM_DOT((VEC_TYPE(4))(left1.s4, middle1.s4, right1.s4, left2.s4), (VEC_TYPE(4))(w0.s0, w0.s1, w0.s2, w1.s0), values1.s4);
    ARM_DOT((VEC_TYPE(4))(middle2.s4, right2.s4, left3.s4, middle3.s4), (VEC_TYPE(4))(w1.s1, w1.s2, w2.s0, w2.s1), values1.s4);
    values1.s4 += right3.s4 * w2.s2;

    ARM_DOT((VEC_TYPE(4))(left1.s5, middle1.s5, right1.s5, left2.s5), (VEC_TYPE(4))(w0.s0, w0.s1, w0.s2, w1.s0), values1.s5);
    ARM_DOT((VEC_TYPE(4))(middle2.s5, right2.s5, left3.s5, middle3.s5), (VEC_TYPE(4))(w1.s1, w1.s2, w2.s0, w2.s1), values1.s5);
    values1.s5 += right3.s5 * w2.s2;

    ARM_DOT((VEC_TYPE(4))(left1.s6, middle1.s6, right1.s6, left2.s6), (VEC_TYPE(4))(w0.s0, w0.s1, w0.s2, w1.s0), values1.s6);
    ARM_DOT((VEC_TYPE(4))(middle2.s6, right2.s6, left3.s6, middle3.s6), (VEC_TYPE(4))(w1.s1, w1.s2, w2.s0, w2.s1), values1.s6);
    values1.s6 += right3.s6 * w2.s2;

    ARM_DOT((VEC_TYPE(4))(left1.s7, middle1.s7, right1.s7, left2.s7), (VEC_TYPE(4))(w0.s0, w0.s1, w0.s2, w1.s0), values1.s7);
    ARM_DOT((VEC_TYPE(4))(middle2.s7, right2.s7, left3.s7, middle3.s7), (VEC_TYPE(4))(w1.s1, w1.s2, w2.s0, w2.s1), values1.s7);
    values1.s7 += right3.s7 * w2.s2;
#endif // CONV_STRIDE_Y == 1 && DILATION_Y==1

#if defined(HAS_BIAS)
    values0 += (int8)(bias_value);
#if CONV_STRIDE_Y == 1 && DILATION_Y == 1
    values1 += (int8)(bias_value);
#endif /* CONV_STRIDE_Y == 1 && DILATION_Y==1 */
#endif //defined(HAS_BIAS)

#if WEIGHTS_OFFSET != 0
    values0 += sum0 * (int8)(WEIGHTS_OFFSET);
#if CONV_STRIDE_Y == 1 && DILATION_Y == 1
    values1 += sum1 * (int8)(WEIGHTS_OFFSET);
#endif /* CONV_STRIDE_Y == 1 && DILATION_Y==1 */
#endif /* WEIGHTS_OFFSET != 0 */

#if INPUT_OFFSET != 0
    WEIGHTS_PROMOTED_TYPE sum_weights = 0;
    VEC_WEIGHTS_PROMOTED_TYPE(3)
    tmp_we = CONVERT(w0, VEC_WEIGHTS_PROMOTED_TYPE(3)) + CONVERT(w1, VEC_WEIGHTS_PROMOTED_TYPE(3)) + CONVERT(w2, VEC_WEIGHTS_PROMOTED_TYPE(3));
    sum_weights += tmp_we.s0 + tmp_we.s1 + tmp_we.s2;
    values0 += sum_weights * (int8)(INPUT_OFFSET);
#if CONV_STRIDE_Y == 1 && DILATION_Y == 1
    values1 += sum_weights * (int8)(INPUT_OFFSET);
#endif /* CONV_STRIDE_Y == 1 && DILATION_Y==1*/
#endif /* INPUT_OFFSET != 0 */

#if K_OFFSET != 0
    values0 += (int8)(K_OFFSET);
#if CONV_STRIDE_Y == 1 && DILATION_Y == 1
    values1 += (int8)(K_OFFSET);
#endif /* CONV_STRIDE_Y == 1 && DILATION_Y==1*/
#endif /* K_OFFSET != 0 */

#if defined(REAL_MULTIPLIER)

    values0 = CONVERT(round(CONVERT(values0, float8) * (float8)REAL_MULTIPLIER), int8);

#else // defined(REAL_MULTIPLIER)

#if defined(PER_CHANNEL_QUANTIZATION)
    int8 res0_shift_lt0 = ASYMM_MULT_BY_QUANT_MULTIPLIER_GREATER_THAN_ONE(values0, output_multiplier, output_shift, 8);
    int8 res0_shift_gt0 = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(values0, output_multiplier, output_shift, 8);
    values0             = select(res0_shift_lt0, res0_shift_gt0, (int8)(output_shift) >= 0);
#else // defined(PER_CHANNEL_QUANTIZATION)
#if OUTPUT_SHIFT < 0
    values0 = ASYMM_MULT_BY_QUANT_MULTIPLIER_GREATER_THAN_ONE(values0, OUTPUT_MULTIPLIER, OUTPUT_SHIFT, 8);
#else  // OUTPUT_SHIFT < 0
    values0 = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(values0, OUTPUT_MULTIPLIER, OUTPUT_SHIFT, 8);
#endif // OUTPUT_OFFSET < 0
#endif // defined(PER_CHANNEL_QUANTIZATION)

#endif // defined(REAL_MULTIPLIER)

    values0 += (int8)OUTPUT_OFFSET;
    VEC_TYPE(8)
    res0 = CONVERT_SAT(values0, VEC_TYPE(8));

    vstore8(ACTIVATION_FUNC(res0), 0, dst.ptr);
#if CONV_STRIDE_Y == 1 && DILATION_Y == 1

#if defined(REAL_MULTIPLIER)

    values1 = CONVERT(round(CONVERT(values1, float8) * (float8)REAL_MULTIPLIER), int8);

#else // defined(REAL_MULTIPLIER)

#if defined(PER_CHANNEL_QUANTIZATION)
    int8 res1_shift_lt0 = ASYMM_MULT_BY_QUANT_MULTIPLIER_GREATER_THAN_ONE(values1, output_multiplier, output_shift, 8);
    int8 res1_shift_gt0 = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(values1, output_multiplier, output_shift, 8);
    values1             = select(res1_shift_lt0, res1_shift_gt0, (int8)(output_shift) >= 0);
#else // defined(PER_CHANNEL_QUANTIZATION)
#if OUTPUT_SHIFT < 0
    values1 = ASYMM_MULT_BY_QUANT_MULTIPLIER_GREATER_THAN_ONE(values1, OUTPUT_MULTIPLIER, OUTPUT_SHIFT, 8);
#else  // OUTPUT_SHIFT < 0
    values1 = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(values1, OUTPUT_MULTIPLIER, OUTPUT_SHIFT, 8);
#endif // OUTPUT_OFFSET < 0
#endif // defined(PER_CHANNEL_QUANTIZATION)

#endif // defined(REAL_MULTIPLIER)

    values1 += (int8)OUTPUT_OFFSET;
    VEC_TYPE(8)
    res1 = CONVERT_SAT(values1, VEC_TYPE(8));

    vstore8(ACTIVATION_FUNC(res1), 0, dst.ptr + dst_stride_y);
#endif /* CONV_STRIDE_Y == 1 && DILATION_Y==1*/
}

#endif // !defined(IS_DOT8)

#endif /* defined(CONV_STRIDE_Y) && defined(CONV_STRIDE_X) && defined(DEPTH_MULTIPLIER) && defined(DST_CHANNELS) */

#if defined(VEC_SIZE) && defined(SRC_DIM_1) && defined(SRC_DIM_2) && defined(CONV_PAD_TOP) && defined(CONV_PAD_LEFT)

#define asymm_mult_by_quant_multiplier_less_than_one(x, y, z) ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(x, y, z, VEC_SIZE)

#define MULTIPLY_ADD(x, y, acc) acc += CONVERT(CONVERT(x, VEC_WEIGHTS_PROMOTED_TYPE(VEC_SIZE)) * CONVERT(y, VEC_WEIGHTS_PROMOTED_TYPE(VEC_SIZE)), VEC_INT)

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
        ARM_DOT((VEC_TYPE(4))(val0, val1, val2, val3), w0.s0123, acc);                 \
        ARM_DOT((VEC_TYPE(4))(val4, val5, val6, val7), w0.s4567, acc);                 \
        acc += val8 * w1;                                                              \
    })

#define DOT_PRODUCT_REDUCTION(sum, val0, val1, val2, val3, val4, val5, val6, val7, val8) \
    ({                                                                                   \
        sum = val0;                                                                      \
        ARM_DOT((VEC_TYPE(4))(val1, val2, val3, val4), (VEC_TYPE(4))1, sum);             \
        ARM_DOT((VEC_TYPE(4))(val5, val6, val7, val8), (VEC_TYPE(4))1, sum);             \
    })

#define DOT_PRODUCT_REDUCTION_WEIGHTS(sum, w0, w1) \
    ({                                             \
        sum = w1;                                  \
        ARM_DOT(w0.s0123, (VEC_TYPE(4))1, sum);    \
        ARM_DOT(w0.s4567, (VEC_TYPE(4))1, sum);    \
    })

#endif // defined(ARM_COMPUTE_OPENCL_DOT8_ENABLED) && defined(cl_arm_integer_dot_product_int8)

#endif // defined(VEC_SIZE) && defined(SRC_DIM_1) && defined(SRC_DIM_2) && defined(CONV_PAD_TOP) && defined(CONV_PAD_LEFT)

#endif // defined(WEIGHTS_PROMOTED_TYPE)

#endif // defined(WEIGHTS_OFFSET) && defined(INPUT_OFFSET) && defined(K_OFFSET) && ((defined(OUTPUT_OFFSET) && defined(OUTPUT_MULTIPLIER) && defined(OUTPUT_SHIFT)) || defined(REAL_MULTIPLIER))

#if defined(SRC_DIM1) && defined(SRC_DIM2) && defined(KERNEL_WIDTH) && defined(KERNEL_HEIGHT) && defined(N0) && defined(DILATION_X) && defined(DILATION_Y) && defined(CONV_STRIDE_X) && defined(CONV_STRIDE_Y) && defined(CONV_PAD_LEFT) && defined(CONV_PAD_TOP) && defined(INPUT_OFFSET) && defined(WEIGHTS_OFFSET) && defined(OUTPUT_OFFSET) && defined(OUTPUT_SHIFT) && defined(OUTPUT_MULTIPLIER) && defined(VEC_SIZE_LEFTOVER)
/** This function computes the depthwise convolution for NHWC data layout.
 *
 * @note The number of elements processed must be passed at compile time using -DN0 (e.g. -DN0=2)
 * @note The depth multiplier must be passed at compile time using -DDEPTH_MULTIPLIER (e.g. -DDEPTH_MULTIPLIER=1)
 * @note The first dimension of the input tensor must be passed at compile time using -DSRC_DIM1 (e.g. -DSRC_DIM1=112)
 * @note The second dimension of the input tensor must be passed at compile time using -DSRC_DIM2 (e.g. -DSRC_DIM2=80)
 * @note The kernel width must be passed at compile time using -DKERNEL_WIDTH (e.g. -DKERNEL_WIDTH=5)
 * @note The kernel height must be passed at compile time using -DKERNEL_HEIGHT (e.g. -DKERNEL_HEIGHT=5)
 * @note The convolution pad top must be passed at compile time using -DCONV_PAD_TOP (e.g. -DCONV_PAD_TOP=1)
 * @note The convolution pad top must be passed at compile time using -DCONV_PAD_LEFT (e.g. -DCONV_PAD_LEFT=1)
 * @note The convolution stride along the width must be passed at compile time using -DCONV_STRIDE_X (e.g. -DCONV_STRIDE_Y=X)
 * @note The convolution stride along the height must be passed at compile time using -DCONV_STRIDE_Y (e.g. -DCONV_STRIDE_Y=1)
 * @note Leftover vector size has to be passed at compile time using -DVEC_SIZE_LEFTOVER. e.g. -DVEC_SIZE=3. It is defined as the remainder between the input's first dimension and VEC_SIZE
 * @note It is possible to select the activation function to apply using -DACTIVATION_TYPE e.g. -DACTIVATION_TYPE=relu
 * @note A, B variables required by some activation functions are set using -DA_VAL= and -DB_VAL= respectively
 *
 * @param[in] src_ptr                                          Pointer to the source tensor. Supported data types: QASYMM8/QASYMM8_SIGNED
 * @param[in] src_stride_x                                     Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_step_x                                       src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                                     Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_step_y                                       src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_stride_z                                     Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                                       src_stride_y * number of elements along Z processed per workitem(in bytes)
 * @param[in] src_stride_w                                     Stride of the source tensor in W dimension (in bytes)
 * @param[in] src_step_w                                       src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes                The offset of the first element in the source tensor
 * @param[in] dst_ptr                                          Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in] dst_stride_x                                     Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                                       dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                                     Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                                       dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                                     Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                                       dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_w                                     Stride of the destination tensor in W dimension (in bytes)
 * @param[in] dst_step_w                                       dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes                The offset of the first element in the destination tensor
 * @param[in] weights_ptr                                      Pointer to the weights tensor. Supported data types: QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL
 * @param[in] weights_stride_x                                 Stride of the weights tensor in X dimension (in bytes)
 * @param[in] weights_step_x                                   weights_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] weights_stride_y                                 Stride of the weights tensor in Y dimension (in bytes)
 * @param[in] weights_step_y                                   weights_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_stride_z                                 Stride of the weights tensor in Z dimension (in bytes)
 * @param[in] weights_step_z                                   weights_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] weights_offset_first_element_in_bytes            The offset of the first element in the weights tensor
 * @param[in] output_multipliers_ptr                           Pointer to the output multipliers vector. Supported data types: S32
 * @param[in] output_multipliers_stride_x                      Stride of the output multipliers vector in X dimension (in bytes)
 * @param[in] output_multipliers_step_x                        output_multipliers_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] output_multipliers_offset_first_element_in_bytes The offset of the first element in the output multipliers vector
 * @param[in] output_shifts_ptr                                Pointer to the output shifts vector. Supported data types: S32
 * @param[in] output_shifts_stride_x                           Stride of the output shifts vector in X dimension (in bytes)
 * @param[in] output_shifts_step_x                             output_shifts_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] output_shifts_offset_first_element_in_bytes      The offset of the first element in the output shifts vector
 * @param[in] biases_ptr                                       (Optional) Pointer to the biases vector. Supported data types: S32
 * @param[in] biases_stride_x                                  (Optional) Stride of the biases vector in X dimension (in bytes)
 * @param[in] biases_step_x                                    (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] biases_offset_first_element_in_bytes             (Optional) The offset of the first element in the biases vector
 */
__kernel void dwc_MxN_native_quantized8_nhwc(
    TENSOR4D_DECLARATION(src),
    TENSOR4D_DECLARATION(dst),
    TENSOR3D_DECLARATION(weights),
    VECTOR_DECLARATION(output_multipliers),
    VECTOR_DECLARATION(output_shifts)
#if defined(HAS_BIAS)
    ,
    VECTOR_DECLARATION(biases)
#endif // defined(HAS_BIAS)
)
{
    int x_offs = max((int)(get_global_id(0) * N0 - (N0 - VEC_SIZE_LEFTOVER) % N0), 0);
    int y      = get_global_id(1); // spatial coordinate x
#if defined(DST_DEPTH)
    int z = get_global_id(2) % (int)DST_DEPTH; // spatial coordinate y
    int b = get_global_id(2) / (int)DST_DEPTH; // batch
#else                                          // defined(DST_DEPTH)
    int z = get_global_id(2); // spatial coordinate y
#endif                                         // defined(DST_DEPTH)

    __global uchar *s_addr = src_ptr + src_offset_first_element_in_bytes + x_offs * sizeof(DATA_TYPE);

    __global uchar *d_addr = dst_ptr + dst_offset_first_element_in_bytes + x_offs * sizeof(DATA_TYPE) * (int)DEPTH_MULTIPLIER + y * dst_stride_y + z * dst_stride_z;

    __global uchar *w_addr = weights_ptr + weights_offset_first_element_in_bytes + x_offs * sizeof(WEIGHTS_TYPE) * (int)DEPTH_MULTIPLIER;

#if defined(HAS_BIAS)
    __global uchar *b_addr = biases_ptr + biases_offset_first_element_in_bytes + x_offs * sizeof(int) * (int)DEPTH_MULTIPLIER;
#endif // defined(HAS_BIAS)

#if defined(PER_CHANNEL_QUANTIZATION)
    __global uchar *out_mul_addr   = output_multipliers_ptr + output_multipliers_offset_first_element_in_bytes + x_offs * sizeof(int) * (int)DEPTH_MULTIPLIER;
    __global uchar *out_shift_addr = output_shifts_ptr + output_shifts_offset_first_element_in_bytes + x_offs * sizeof(int) * (int)DEPTH_MULTIPLIER;
#endif // defined(PER_CHANNEL_QUANTIZATION)

#if defined(DST_DEPTH)
    s_addr += b * src_stride_w;
    d_addr += b * dst_stride_w;
#endif // defined(DST_DEPTH)

#if DEPTH_MULTIPLIER > 1
    for(int d = 0; d < (int)DEPTH_MULTIPLIER; ++d)
    {
#endif // DEPTH_MULTIPLIER > 1
        // Each work-item computes N0x1x1 elements
        VEC_INT res = 0;

        int x_coord = y * CONV_STRIDE_X - (int)CONV_PAD_LEFT;
        int y_coord = z * CONV_STRIDE_Y - (int)CONV_PAD_TOP;

        for(int yk = 0; yk < KERNEL_HEIGHT; ++yk)
        {
            if(y_coord >= 0 && y_coord < SRC_DIM2)
            {
                int x_coord_tmp = x_coord;

                for(int xk = 0; xk < KERNEL_WIDTH; ++xk)
                {
                    if(x_coord_tmp >= 0 && x_coord_tmp < SRC_DIM1)
                    {
                        int s_offset = x_coord_tmp * (int)src_stride_y + y_coord * (int)src_stride_z;
                        int w_offset = xk * weights_stride_y + yk * weights_stride_z;

                        // Load input and weights values
                        VEC_INT i = CONVERT(VLOAD(N0)(0, (__global DATA_TYPE *)(s_addr + s_offset)), VEC_INT);
                        VEC_INT w = CONVERT(VLOAD(N0)(0, (__global WEIGHTS_TYPE *)(w_addr + w_offset)), VEC_INT);

                        res += (i + (VEC_INT)INPUT_OFFSET) * (w + (VEC_INT)WEIGHTS_OFFSET);
                    }
                    x_coord_tmp += DILATION_X;
                }
            }
            y_coord += DILATION_Y;
        }

#if defined(HAS_BIAS)
        VEC_INT bias = VLOAD(N0)(0, (__global int *)(b_addr));
        res += bias;
#endif // defined(HAS_BIAS)

#if defined(PER_CHANNEL_QUANTIZATION)
        VEC_INT output_multiplier = VLOAD(N0)(0, (__global int *)(out_mul_addr));
        VEC_INT output_shift      = VLOAD(N0)(0, (__global int *)(out_shift_addr));

        VEC_INT res_shift_lt0 = ASYMM_MULT_BY_QUANT_MULTIPLIER_GREATER_THAN_ONE(res, output_multiplier, output_shift, N0);
        VEC_INT res_shift_gt0 = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(res, output_multiplier, output_shift, N0);
        res                   = select(res_shift_lt0, res_shift_gt0, (VEC_INT)(output_shift) >= 0);
#else // defined(PER_CHANNEL_QUANTIZATION)
#if OUTPUT_SHIFT < 0
        res   = ASYMM_MULT_BY_QUANT_MULTIPLIER_GREATER_THAN_ONE(res, OUTPUT_MULTIPLIER, OUTPUT_SHIFT, N0);
#else  // OUTPUT_SHIFT < 0
        res = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(res, OUTPUT_MULTIPLIER, OUTPUT_SHIFT, N0);
#endif // OUTPUT_OFFSET < 0
#endif // defined(PER_CHANNEL_QUANTIZATION)

        res += (VEC_INT)OUTPUT_OFFSET;

        VEC_TYPE(VEC_SIZE)
        res0 = CONVERT_SAT(res, VEC_TYPE(VEC_SIZE));
        res0 = ACTIVATION_FUNC(res0);

        STORE_VECTOR_SELECT(res, DATA_TYPE, d_addr, N0, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0)

#if DEPTH_MULTIPLIER > 1
        w_addr += sizeof(WEIGHTS_TYPE);
        d_addr += sizeof(DATA_TYPE);
#if defined(PER_CHANNEL_QUANTIZATION)
        out_mul_addr += sizeof(int);
        out_shift_addr += sizeof(int);
#endif // defined(PER_CHANNEL_QUANTIZATION)
#if defined(HAS_BIAS)
        b_addr += sizeof(int);
#endif // defined(HAS_BIAS)
    }
#endif // DEPTH_MULTIPLIER > 1
}
#endif // defined(SRC_DIM1) && defined(SRC_DIM2) && defined(KERNEL_WIDTH) && defined(KERNEL_HEIGHT) && defiend(N0) && defined(DILATION_X) && defined(DILATION_Y) && defined(CONV_STRIDE_X) && defined(CONV_STRIDE_Y) && defined(CONV_PAD_LEFT) && defined(CONV_PAD_TOP) && defined(INPUT_OFFSET) && defined(WEIGHTS_OFFSET) && defined(OUTPUT_OFFSET) && defined(OUTPUT_SHIFT) && defined(OUTPUT_MULTIPLIER) && defined(VEC_SIZE_LEFTOVER)
#endif // defined(DATA_TYPE) && defined(WEIGHTS_TYPE)
