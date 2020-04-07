/*
 * Copyright (c) 2020 ARM Limited.
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

#if VEC_SIZE == 2
#define multiply_by_quantized_multiplier(input, qmul, shift) MULTIPLY_BY_QUANTIZED_MULTIPLIER(input, qmul, shift, 2)
#define PERFORM_REDUCTION_IMPL(type)                                                   \
    inline VEC_DATA_TYPE(type, 1) perform_reduction_##type(VEC_DATA_TYPE(type, 2) sum) \
    {                                                                                  \
        sum.s0 += sum.s1;                                                              \
        return sum.s0;                                                                 \
    }
#elif VEC_SIZE == 4
#define multiply_by_quantized_multiplier(input, qmul, shift) MULTIPLY_BY_QUANTIZED_MULTIPLIER(input, qmul, shift, 4)
#define PERFORM_REDUCTION_IMPL(type)                                                   \
    inline VEC_DATA_TYPE(type, 1) perform_reduction_##type(VEC_DATA_TYPE(type, 4) sum) \
    {                                                                                  \
        sum.s01 += sum.s23;                                                            \
        sum.s0 += sum.s1;                                                              \
        return sum.s0;                                                                 \
    }
#elif VEC_SIZE == 8
#define multiply_by_quantized_multiplier(input, qmul, shift) MULTIPLY_BY_QUANTIZED_MULTIPLIER(input, qmul, shift, 8)
#define PERFORM_REDUCTION_IMPL(type)                                                   \
    inline VEC_DATA_TYPE(type, 1) perform_reduction_##type(VEC_DATA_TYPE(type, 8) sum) \
    {                                                                                  \
        sum.s0123 += sum.s4567;                                                        \
        sum.s01 += sum.s23;                                                            \
        sum.s0 += sum.s1;                                                              \
        return sum.s0;                                                                 \
    }
#else /* VEC_SIZE DEFAULT */
#define VEC_SIZE 16
#define multiply_by_quantized_multiplier(input, qmul, shift) MULTIPLY_BY_QUANTIZED_MULTIPLIER(input, qmul, shift, 16)
#define PERFORM_REDUCTION_IMPL(type)                                                    \
    inline VEC_DATA_TYPE(type, 1) perform_reduction_##type(VEC_DATA_TYPE(type, 16) sum) \
    {                                                                                   \
        sum.s01234567 += sum.s89abcdef;                                                 \
        sum.s0123 += sum.s4567;                                                         \
        sum.s01 += sum.s23;                                                             \
        sum.s0 += sum.s1;                                                               \
        return sum.s0;                                                                  \
    }
#endif /* VEC_SIZE END */

#define PERFORM_REDUCTION_STR(input, type) perform_reduction_##type(input)
#define PERFORM_REDUCTION(input, type) PERFORM_REDUCTION_STR(input, type)

PERFORM_REDUCTION_IMPL(int)
PERFORM_REDUCTION_IMPL(long)

/** Compute quantized multiplier and shift for the inverse square root of input.
 *  Using 3-bit fixed point and 5 iteration of Newton-Raphson method.
 *
 * @param[in] in            Input to use
 * @param[in] reverse_shift -1 to reverse the shift direction
 *
 * @return:
 *             .s0  Quantized multiplier for inverse square root
 *             .s1  Shift for inverse square root
 *
 */
inline int2 get_invsqrt_quantized_multiplier_exp(int in, int reverse_shift)
{
    int2 stddev_inv;
    int  stddev_inv_multiplier = INT_MAX;
    int  stddev_inv_shift      = 0;
    int  input                 = in;
    if(input <= 1)
    {
        stddev_inv.s0 = stddev_inv_multiplier;
        stddev_inv.s1 = stddev_inv_shift;
        return stddev_inv;
    }

    stddev_inv_shift = 11;
    while(input >= (1 << 29))
    {
        input /= 4;
        ++stddev_inv_shift;
    }

    const unsigned int max_left_shift_bits       = clz(input) - 1;
    const unsigned int max_left_shift_bits_pairs = max_left_shift_bits / 2;
    const unsigned int left_shift_bit_pairs      = max_left_shift_bits_pairs - 1;
    stddev_inv_shift -= left_shift_bit_pairs;
    input <<= 2 * left_shift_bit_pairs;

    typedef int               FixedPointRawType;
    const unsigned int        fixedpoint_position     = 3;
    const unsigned int        fixedpoint_int_position = sizeof(FixedPointRawType) * 8 - 1 - fixedpoint_position;
    typedef FixedPointRawType FixedPoint3;
    typedef FixedPointRawType FixedPoint0;

    const FixedPoint3 fixedpoint_input      = (input >> 1);
    const FixedPoint3 fixedpoint_half_input = ASYMM_ROUNDING_DIVIDE_BY_POW2(fixedpoint_input, 1, 1);
    const FixedPoint3 fixedpoint_half_three = (0x1 << fixedpoint_int_position) + (0x1 << (fixedpoint_int_position - 1));
    FixedPoint3       x                     = 0x1 << fixedpoint_int_position;

    const int num_iteration = 5;
    for(int i = 0; i < num_iteration; i++)
    {
        int x3 = ASYMM_RESCALE(ASYMM_MULT(ASYMM_MULT(x, x, 1), x, 1), 9, fixedpoint_position, 1);
        x      = ASYMM_RESCALE(ASYMM_MULT(fixedpoint_half_three, x, 1) - ASYMM_MULT(fixedpoint_half_input, x3, 1), 6, fixedpoint_position, 1);
    }
    const FixedPoint0 fixedpoint_half_sqrt_2 = 1518500250;
    x                                        = ASYMM_MULT(fixedpoint_half_sqrt_2, x, 1);
    stddev_inv_multiplier                    = x;
    if(stddev_inv_shift < 0)
    {
        stddev_inv_multiplier <<= -stddev_inv_shift;
        stddev_inv_shift = 0;
    }
    stddev_inv_shift *= reverse_shift;

    stddev_inv.s0 = stddev_inv_multiplier;
    stddev_inv.s1 = stddev_inv_shift;
    return stddev_inv;
}

#if defined(VEC_SIZE) && defined(DATA_TYPE) && defined(WIDTH) && defined(OUTPUT_MULTIPLIER) && defined(OUTPUT_SHIFT)
/** This function implements QLSTM layer normalization.
 *
 * @attention Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 * @attention Data type should be passed using the -DDATA_TYPE compile flag, e.g. -DDATA_TYPE=float
 * @attention Width of the input tensor should be passed using the -DWIDTH compile flag, e.g. -DWIDTH=16
 *
 * @param[in]  input_ptr                            Pointer to the first source tensor. Supported data types: QSYMM16
 * @param[in]  input_stride_x                       Stride of the first source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the first source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the first source tensor
 * @param[in]  weight_ptr                           Pointer to the weight tensor. Supported data type: same as @p input_ptr
 * @param[in]  weight_stride_x                      Stride of the weight tensor in X dimension (in bytes)
 * @param[in]  weight_step_x                        weight_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  weight_offset_first_element_in_bytes The offset of the first element in the weight tensor
 * @param[in]  bias_ptr                             Pointer to the bias tensor. Supported data type: S32
 * @param[in]  bias_stride_x                        Stride of the bias tensor in X dimension (in bytes)
 * @param[in]  bias_step_x                          bias_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  bias_offset_first_element_in_bytes   The offset of the first element in the biases tensor
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void qlstm_layer_normalization(
    IMAGE_DECLARATION(input),
    VECTOR_DECLARATION(weight),
    VECTOR_DECLARATION(bias),
    IMAGE_DECLARATION(output))
{
    // Get pixels pointer
    Image  input  = CONVERT_TO_IMAGE_STRUCT(input);
    Vector weight = CONVERT_TO_VECTOR_STRUCT(weight);
    Vector bias   = CONVERT_TO_VECTOR_STRUCT(bias);
    Image  output = CONVERT_TO_IMAGE_STRUCT(output);

    VEC_DATA_TYPE(int, VEC_SIZE)
    sum = 0;
    VEC_DATA_TYPE(long, VEC_SIZE)
    sum_sq = 0;
    // Calculate partial sum
    int i = 0;
    for(; i <= (WIDTH - VEC_SIZE); i += VEC_SIZE)
    {
        // Load data
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
        data = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)offset(&input, i, 0));

        sum += CONVERT(data, VEC_DATA_TYPE(int, VEC_SIZE));
        sum_sq += CONVERT(data, VEC_DATA_TYPE(long, VEC_SIZE)) * CONVERT(data, VEC_DATA_TYPE(long, VEC_SIZE));
    }
    // Perform reduction
    sum.s0    = PERFORM_REDUCTION(sum, int);
    sum_sq.s0 = PERFORM_REDUCTION(sum_sq, long);

    // Left-overs loop
    for(; i < WIDTH; ++i)
    {
        DATA_TYPE data = *((__global DATA_TYPE *)offset(&input, i, 0));

        sum.s0 += CONVERT(data, int);
        sum_sq.s0 += CONVERT(data, long) * CONVERT(data, long);
    }

    int  temp       = 0x100000 / WIDTH;
    int  mean       = (int)(sum.s0 * 1024 / WIDTH);
    int  var2       = ((sum_sq.s0 * (long)temp) - ((long)mean * (long)mean)) / 0x100000;
    int2 stddev_inv = get_invsqrt_quantized_multiplier_exp(var2, -1);

    i = 0;
    for(; i <= (WIDTH - VEC_SIZE); i += VEC_SIZE)
    {
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
        data = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)offset(&input, i, 0));
        VEC_DATA_TYPE(int, VEC_SIZE)
        res = CONVERT(data, VEC_DATA_TYPE(int, VEC_SIZE)) * 1024 - mean;
        res = multiply_by_quantized_multiplier(res, stddev_inv.s0, stddev_inv.s1);
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
        w   = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)vector_offset(&weight, i));
        res = res * CONVERT(w, VEC_DATA_TYPE(int, VEC_SIZE));
        res = res + VLOAD(VEC_SIZE)(0, (__global int *)vector_offset(&bias, i));
        // Due to different rounding scheme, we might need to revisit in the future: res = select(res - 512, res + 512, res > 0) / 1024;
        res = (res + 512) >> 10;
        res = multiply_by_quantized_multiplier(res, OUTPUT_MULTIPLIER, OUTPUT_SHIFT + 12);
#if defined(MIN_BOUND)
        res = max(res, (VEC_DATA_TYPE(int, VEC_SIZE))MIN_BOUND);
#endif // defined(MIN_BOUND)
#if defined(MAX_BOUND)
        res = min(res, (VEC_DATA_TYPE(int, VEC_SIZE))MAX_BOUND);
#endif // defined(MAX_BOUND)
        VSTORE(VEC_SIZE)
        (CONVERT(res, VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)), 0, (__global DATA_TYPE *)offset(&output, i, 0));
    }
    for(; i < WIDTH; ++i)
    {
        DATA_TYPE data = *((__global DATA_TYPE *)offset(&input, i, 0));
        int res        = (int)data * 1024 - mean;
        res            = MULTIPLY_BY_QUANTIZED_MULTIPLIER(res, stddev_inv.s0, stddev_inv.s1, 1);
        DATA_TYPE w    = *((__global DATA_TYPE *)vector_offset(&weight, i));
        res            = res * (int)w;
        int b          = *((__global int *)vector_offset(&bias, i));
        res            = res + b;
        // Due to different rounding scheme, we might need to revisit in the future: res = select(res - 512, res + 512, res > 0) / 1024;
        res = (res + 512) >> 10;
        res = MULTIPLY_BY_QUANTIZED_MULTIPLIER(res, OUTPUT_MULTIPLIER, OUTPUT_SHIFT + 12, 1);
#if defined(MIN_BOUND)
        res = max(res, MIN_BOUND);
#endif // defined(MIN_BOUND)
#if defined(MAX_BOUND)
        res = min(res, MAX_BOUND);
#endif // defined(MAX_BOUND)
        *((__global DATA_TYPE *)offset(&output, i, 0)) = (DATA_TYPE)res;
    }
}
#endif /* defined(VEC_SIZE) && defined(DATA_TYPE) && defined(WIDTH) && defined(OUTPUT_MULTIPLIER) && defined(OUTPUT_SHIFT) */