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
#include "asymm_helper.h"
#include "helpers.h"

#define MAX_OP(x, y, type, size) max((x), (y))
#define ADD_OP(x, y, type, size) ((x) + (y))

__constant uchar16 type_min = 0;
__constant uint16 idx16     = (uint16)(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

/** Identifies the maximum value across the 1st dimension.
 *
 * @note In case the input is not multiple of 16 -DNON_MULTIPLE_OF_16 must be passed.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor slice. Supported data types: QASYMM8
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  width                             Input image width
 */
__kernel void softmax_layer_max_quantized(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    uint width)
{
    Image src = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);

    // Initialize local maximum
    uchar16 max_val = 0;

    // Calculate max of row
    const uint width4 = width >> 4;
    for(uint i = 0; i < width4; i++)
    {
        uchar16 data = vload16(0, (__global uchar *)offset(&src, i << 4, 0));
        max_val      = MAX_OP(data, max_val, uchar, 16);
    }

#ifdef NON_MULTIPLE_OF_16
    // Handle non multiple of 16
    uchar16 data = vload16(0, (__global uchar *)offset(&src, width4 << 4, 0));
    uchar16 widx = convert_uchar16(((uint16)(width4 << 4) + idx16) < width);
    max_val      = MAX_OP(max_val, select(type_min, data, widx), uchar, 16);
#endif /* NON_MULTIPLE_OF_16 */

    // Perform max reduction
    max_val.s01234567 = MAX_OP(max_val.s01234567, max_val.s89ABCDEF, uchar, 8);
    max_val.s0123     = MAX_OP(max_val.s0123, max_val.s4567, uchar, 4);
    max_val.s01       = MAX_OP(max_val.s01, max_val.s23, uchar, 2);
    max_val.s0        = MAX_OP(max_val.s0, max_val.s1, uchar, 1);

    // Store result
    *((__global uchar *)dst.ptr) = max_val.s0;
}

#if defined(DIFF_MIN)

int16 mult_by_quantized_multiplier(int16 data)
{
#if defined(INPUT_BETA_MULTIPLIER) && defined(INPUT_BETA_LEFT_SHIFT)
    if(INPUT_BETA_MULTIPLIER > 1)
    {
        return asymm_mult(data * (1 << INPUT_BETA_LEFT_SHIFT), INPUT_BETA_MULTIPLIER);
    }
#endif /* defined(INPUT_BETA_MULTIPLIER) && defined(INPUT_BETA_LEFT_SHIFT) */
    return data;
}

/** Shifts the values of the input tensor by the max calculated in softmax_layer_max kernel,
 * then gets the exponent of each element as sums all elements across each row.
 *
 * @note In case the input is not multiple of 16 -DNON_MULTIPLE_OF_16 must be passed.
 * @note Quantized beta can be optionally passed at compile time using -DINPUT_BETA_MULTIPLIER and -DINPUT_BETA_LEFT_SHIFT (if undefined, assume beta equals 1.0)
 * @note -DDIFF_MIN must be passed at compile time. It is threshold difference between maximum value of input data and current processed value, it defines whether the value will be taken into account or not.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor slice. Supported data types: QASYMM8
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in]  max_ptr                           Pointer to the max values tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  max_stride_x                      Stride of the max values tensor in X dimension (in bytes)
 * @param[in]  max_step_x                        max_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  max_stride_y                      Stride of the max values tensor in Y dimension (in bytes)
 * @param[in]  max_step_y                        max_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  max_stride_z                      Stride of the max values tensor in Z dimension (in bytes)
 * @param[in]  max_step_z                        max_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  max_offset_first_element_in_bytes The offset of the first element in the max values tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor slice. Supported data types: S32
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[out] sum_ptr                           Pointer to the sum values tensor slice. Supported data types: same as @p dst_ptr
 * @param[in]  sum_stride_x                      Stride of the sum values tensor in X dimension (in bytes)
 * @param[in]  sum_step_x                        sum_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_stride_y                      Stride of the sum values tensor in Y dimension (in bytes)
 * @param[in]  sum_step_y                        sum_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  sum_stride_z                      Stride of the sum values tensor in Z dimension (in bytes)
 * @param[in]  sum_step_z                        sum_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  sum_offset_first_element_in_bytes The offset of the first element in the sum values tensor
 * @param[in]  width                             Input image width
 */
__kernel void softmax_layer_shift_exp_sum_quantized(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(max),
    TENSOR3D_DECLARATION(dst),
    TENSOR3D_DECLARATION(sum),
    uint width)
{
    Image src = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);
    Image max = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(max);
    Image sum = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(sum);

    // Load max value of 1D logits vector (row)
    int max_val = convert_int(*((__global uchar *)offset(&max, 0, 0)));

    // Set sum vector, Q(EXP_ACCUMULATION_INT_BITS)
    int16 sum1D = 0;

    // Shift values, exp and sum
    const uint width4 = width >> 4;
    for(uint i = 0; i < width4; i++)
    {
        uchar16 data         = vload16(0, (__global uchar *)offset(&src, i << 4, 0));
        int16 data_fp        = convert_int16(data);
        int16 data_diff      = data_fp - max_val;
        int16 data_diff_mult = mult_by_quantized_multiplier(data_diff);
        data_fp              = asymm_exp_on_negative_values(data_diff_mult, SCALED_DIFF_INT_BITS);
        data_fp              = asymm_rescale(data_fp, 0, EXP_ACCUMULATION_INT_BITS);
        vstore16(data_diff, 0, (__global int *)offset(&dst, i << 4, 0));
        sum1D = sum1D + select(0, data_fp, data_diff >= (int16)(DIFF_MIN));
    }

#ifdef NON_MULTIPLE_OF_16
    // Handle non multiple of 16
    uchar16 data         = vload16(0, (__global uchar *)offset(&src, width4 << 4, 0));
    int16 data_fp        = convert_int16(data);
    int16 data_diff      = data_fp - max_val;
    int16 data_diff_mult = mult_by_quantized_multiplier(data_diff);
    data_fp              = asymm_exp_on_negative_values(data_diff_mult, SCALED_DIFF_INT_BITS);
    data_fp              = asymm_rescale(data_fp, 0, EXP_ACCUMULATION_INT_BITS);
    int16 widx           = convert_int16(((uint16)(width4 << 4) + idx16) < width);
    vstore16(data_diff, 0, (__global int *)offset(&dst, width4 << 4, 0));
    data_fp = select(0, data_fp, data_diff >= (int16)(DIFF_MIN));
    sum1D   = sum1D + select(0, data_fp, widx);
#endif /* NON_MULTIPLE_OF_16 */

    // Perform min/max reduction
    sum1D.s01234567 = ADD_OP(sum1D.s01234567, sum1D.s89ABCDEF, qs16, 8);
    sum1D.s0123     = ADD_OP(sum1D.s0123, sum1D.s4567, qs16, 4);
    sum1D.s01       = ADD_OP(sum1D.s01, sum1D.s23, qs16, 2);
    sum1D.s0        = ADD_OP(sum1D.s0, sum1D.s1, qs16, 1);

    // Calculate and store result
    *((__global int *)sum.ptr) = sum1D.s0;
}

/** Divides all the values of the input tensor by the sum calculated from softmax_layer_shift_exp_sum kernel.
 *
 * @note Fixed point position must be given as a preprocessor argument using -DFIXED_POINT_POSITION=pos. e.g. DFIXED_POINT_POSITION=4
 * @note Quantized beta can be optionally passed at compile time using -DINPUT_BETA_MULTIPLIER and -DINPUT_BETA_LEFT_SHIFT (if undefined, assume beta equals 1.0)
 * @note -DDIFF_MIN must be passed at compile time. It is threshold difference between maximum value of input data and current processed value, it defines whether the value will be taken into account or not.
 *
 * @param[in]  src_ptr                           Pointer to the source tensor slice. Supported data types: S32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in]  sum_ptr                           Pointer to the sum values tensor slice. Supported data types: same as @p src_ptr
 * @param[in]  sum_stride_x                      Stride of the sum values tensor in X dimension (in bytes)
 * @param[in]  sum_step_x                        sum_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_stride_y                      Stride of the sum values tensor in Y dimension (in bytes)
 * @param[in]  sum_step_y                        sum_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  sum_stride_z                      Stride of the sum values tensor in Z dimension (in bytes)
 * @param[in]  sum_step_z                        sum_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  sum_offset_first_element_in_bytes The offset of the first element in the sum values tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor slice. Supported data types: QASYMM8
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void softmax_layer_norm_quantized(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(sum),
    TENSOR3D_DECLARATION(dst))
{
    Image src = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TENSOR3D_TO_IMAGE_STRUCT(dst);
    Image sum = CONVERT_TENSOR3D_TO_IMAGE_STRUCT_NO_STEP(sum);

    // Load max value of 1D logits vector (row)
    int sum_val = *((__global int *)offset(&sum, 0, get_global_id(1)));

    // It will be better to calculate this in prev layer and pass here as parameter
    uint  sum_val_u               = convert_uint(sum_val);
    int   headroom_plus_one       = clz(sum_val_u);
    int   num_bits_over_unit      = EXP_ACCUMULATION_INT_BITS - headroom_plus_one;
    int   shifted_sum_minus_one_1 = convert_int((sum_val_u << headroom_plus_one) - (1u << 31));
    int16 shifted_sum_minus_one   = shifted_sum_minus_one_1;
    int16 shifted_scale           = asymm_one_over_one_plus_x_for_x_in_0_1(shifted_sum_minus_one);

    // It was already calculated in prev layer, should be stored into tmp output and reused
    int16 data_diff      = vload16(0, (__global int *)offset(&src, 0, 0));
    int16 data_diff_mult = mult_by_quantized_multiplier(data_diff);
    int16 data           = asymm_exp_on_negative_values(data_diff_mult, SCALED_DIFF_INT_BITS);

    data = asymm_mult(shifted_scale, data);
    data = asymm_rounding_divide_by_pow2(data, num_bits_over_unit + 31 - 8);
    data = select(0, data, data_diff >= (int16)(DIFF_MIN));
    vstore16(convert_uchar16_sat(data), 0, (__global uchar *)offset(&dst, 0, 0));
}

#endif /* defined(DIFF_MIN) */
