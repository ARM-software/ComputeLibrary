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
#ifndef ARM_COMPUTE_HELPERS_ASYMM_H
#define ARM_COMPUTE_HELPERS_ASYMM_H

#include "helpers.h"

/** Convert the given vector with round to nearest even rounding mode
 *
 * @param[in] x    The target to be converted
 * @param[in] type The target type
 *
 * @return The converted vector
 */
#define CONVERT_DOWN_RTE_STR(x, type) (convert_##type##_rte((x)))
#define CONVERT_DOWN_RTE(x, type) CONVERT_DOWN_RTE_STR(x, type)

/** Quantize a floating-point scalar value to 8-bit asymmetric
 *
 * @param[in] input  Input value to quantize
 * @param[in] offset Quantization offset
 * @param[in] scale  Quantization scale
 *
 * @return quantized value
 */
inline uchar quantize_qasymm8(float input, float offset, float scale)
{
    float out_f32 = input / scale + offset;
    uchar res_u8  = CONVERT_SAT(CONVERT_DOWN_RTE(out_f32, int), uchar);
    return res_u8;
}

/** Dequantize a scalar value from 8-bit asymmetric to floating-point
 *
 * @param[in] input  Input value to quantize
 * @param[in] offset Quantization offset
 * @param[in] scale  Quantization scale
 *
 * @return quantized value
 */
inline float dequantize_qasymm8(uchar input, float offset, float scale)
{
    return ((float)input - offset) * scale;
}

/** Dequantize a scalar value from signed 8-bit asymmetric to floating-point
 *
 * @param[in] input  Input value to quantize
 * @param[in] offset Quantization offset
 * @param[in] scale  Quantization scale
 *
 * @return quantized value
 */
inline float dequantize_qasymm8_signed(char input, float offset, float scale)
{
    return ((float)input - offset) * scale;
}

/** Quantize a vector of values from floating-point
 *
 * @param[in] type Output data type.
 * @param[in] size Size of vector.
 *
 * @return quantized values
 */
#define QUANTIZE_IMPL(type, size)                                                                                       \
    inline VEC_DATA_TYPE(type, size) quantize_##type##size(VEC_DATA_TYPE(float, size) input, float offset, float scale) \
    {                                                                                                                   \
        VEC_DATA_TYPE(float, size)                                                                                      \
        out_f32 = input / (VEC_DATA_TYPE(float, size))(scale) + (VEC_DATA_TYPE(float, size))(offset);                   \
        VEC_DATA_TYPE(type, size)                                                                                       \
        res = CONVERT_SAT(CONVERT_DOWN_RTE(out_f32, VEC_DATA_TYPE(int, size)), VEC_DATA_TYPE(type, size));              \
        return res;                                                                                                     \
    }

/** Dequantize a vector of values to floating-point
 *
 * @param[in] type Input data type.
 * @param[in] size Size of vector.
 *
 * @return dequantized values in floating point
 */
#define DEQUANTIZE_IMPL(type, size)                                                                                       \
    inline VEC_DATA_TYPE(float, size) dequantize_##type##size(VEC_DATA_TYPE(type, size) input, float offset, float scale) \
    {                                                                                                                     \
        return (CONVERT(input, VEC_DATA_TYPE(float, size)) - offset) * scale;                                             \
    }

/** Correctly-rounded-to-nearest division by a power-of-two.
 *
 * @param[in] size Size of vector.
 *
 * @return Correctly-rounded-to-nearest division by a power-of-two.
 */
#define ASYMM_ROUNDING_DIVIDE_BY_POW2_IMPL(size)                                                                                        \
    inline VEC_DATA_TYPE(int, size) asymm_rounding_divide_by_POW2_##size(VEC_DATA_TYPE(int, size) x, VEC_DATA_TYPE(int, size) exponent) \
    {                                                                                                                                   \
        const VEC_DATA_TYPE(int, size)                                                                                                  \
        zero = (VEC_DATA_TYPE(int, size))0;                                                                                         \
        const VEC_DATA_TYPE(int, size)                                                                                                  \
        one = (VEC_DATA_TYPE(int, size))1;                                                                                          \
        VEC_DATA_TYPE(int, size)                                                                                                        \
        mask = (one << exponent) - one;                                                                                                 \
        VEC_DATA_TYPE(int, size)                                                                                                        \
        threshold = (mask >> 1) + select(zero, one, (SELECT_VEC_DATA_TYPE(int, size))(x < 0));                                          \
        return (x >> exponent) + select(zero, one, (SELECT_VEC_DATA_TYPE(int, size))((x & mask) > threshold));                          \
    }

/** Product of two numbers, interpreting them as fixed-point values in the interval [-1, 1),
 * rounding to the nearest value, and saturating -1 * -1 to the maximum value.
 *
 * @param[in] size Size of vector.
 *
 * @return Product of two fixed-point numbers.
 */
#define ASYMM_MULT_IMPL(size)                                                                                \
    inline VEC_DATA_TYPE(int, size) asymm_mult##size(VEC_DATA_TYPE(int, size) a, VEC_DATA_TYPE(int, size) b) \
    {                                                                                                        \
        VEC_DATA_TYPE(int, size)                                                                             \
        overflow = a == b && a == INT_MIN;                                                                   \
        VEC_DATA_TYPE(long, size)                                                                            \
        a_64 = convert_long##size(a);                                                                        \
        VEC_DATA_TYPE(long, size)                                                                            \
        b_64 = convert_long##size(b);                                                                        \
        VEC_DATA_TYPE(long, size)                                                                            \
        ab_64 = a_64 * b_64;                                                                                 \
        /* Revert COMPMID-907 */                                                                             \
        VEC_DATA_TYPE(long, size)                                                                            \
        mask1 = 1 << 30;                                                                                     \
        VEC_DATA_TYPE(long, size)                                                                            \
        mask2 = 1 - (1 << 30);                                                                               \
        VEC_DATA_TYPE(long, size)                                                                            \
        is_positive_or_zero = ab_64 >= 0;                                                                    \
        VEC_DATA_TYPE(long, size)                                                                            \
        nudge = select(mask2, mask1, (SELECT_VEC_DATA_TYPE(long, size))(is_positive_or_zero));               \
        VEC_DATA_TYPE(long, size)                                                                            \
        mask = 1ll << 31;                                                                                    \
        VEC_DATA_TYPE(int, size)                                                                             \
        ab_x2_high32 = convert_int##size((ab_64 + nudge) / mask);                                            \
        return select(ab_x2_high32, INT_MAX, (SELECT_VEC_DATA_TYPE(int, size))(overflow));                   \
    }

/** Calculates \f$ exp(x) \f$ for x in [-1/4, 0).
 *
 * @param[in] size Size of vector.
 *
 * @return Result in fixed-point format Q0.
 */
#define ASYMM_EXP_ON_INTERVAL_BETWEEN_NEGATIVE_ONE_QUARTER_AND_0_EXCL_IMPL(size)                                                    \
    inline VEC_DATA_TYPE(int, size) asymm_exp_on_interval_between_negative_one_quarter_and_0_excl##size(VEC_DATA_TYPE(int, size) a) \
    {                                                                                                                               \
        const VEC_DATA_TYPE(int, size) constant_term     = 1895147668;                                                              \
        const VEC_DATA_TYPE(int, size) constant_1_over_3 = 715827883;                                                               \
        const int k_fractional_bits = 31;                                                                                           \
        VEC_DATA_TYPE(int, size)                                                                                                    \
        x = a + (1 << (k_fractional_bits - 3));                                                                                     \
        VEC_DATA_TYPE(int, size)                                                                                                    \
        x2 = ASYMM_MULT(x, x, size);                                                                                                \
        VEC_DATA_TYPE(int, size)                                                                                                    \
        x3 = ASYMM_MULT(x2, x, size);                                                                                               \
        VEC_DATA_TYPE(int, size)                                                                                                    \
        x4 = ASYMM_MULT(x2, x2, size);                                                                                              \
        VEC_DATA_TYPE(int, size)                                                                                                    \
        x4_over_4 = ASYMM_ROUNDING_DIVIDE_BY_POW2(x4, 2, size);                                                                     \
        VEC_DATA_TYPE(int, size)                                                                                                    \
        x4_over_24_plus_x3_over_6_plus_x2 = ASYMM_MULT((x4_over_4 + x3), constant_1_over_3, size) + x2;                             \
        VEC_DATA_TYPE(int, size)                                                                                                    \
        x4_over_24_plus_x3_over_6_plus_x2_over_2 = ASYMM_ROUNDING_DIVIDE_BY_POW2(x4_over_24_plus_x3_over_6_plus_x2, 1, size);       \
        return constant_term + ASYMM_MULT(constant_term, x + x4_over_24_plus_x3_over_6_plus_x2_over_2, size);                       \
    }

/** Each bit of the result is set to the corresponding bit of either then_val or
 * else_val depending on whether the corresponding bit of if_mask is set.
 * Equivalent to the VBSL instruction in Arm® Neon™.
 *
 * @param[in] size Size of vector.
 *
 * @returns Result contaning bits from @p then_val or from @p else_val depending on corresponding bit in @p if_mask is set or not.
 */
#define ASYMM_SELECT_USING_MASK_IMPL(size)                                                                                                                                \
    inline VEC_DATA_TYPE(int, size) asymm_select_using_mask##size(VEC_DATA_TYPE(int, size) if_mask, VEC_DATA_TYPE(int, size) then_val, VEC_DATA_TYPE(int, size) else_val) \
    {                                                                                                                                                                     \
        return (if_mask & then_val) ^ (~if_mask & else_val);                                                                                                              \
    }

/** For each element of input vector, the corresponding bits of the result item are set
 * if the input item is zero.
 *
 * @param[in] size Size of vector.
 *
 * @returns Output vector with bits set when corresponding bit in @p a is zero.
 */
#define ASYMM_MASK_IF_ZERO_IMPL(size)                                                    \
    inline VEC_DATA_TYPE(int, size) asymm_mask_if_zero##size(VEC_DATA_TYPE(int, size) a) \
    {                                                                                    \
        const VEC_DATA_TYPE(int, size) all_zeros = 0;                                    \
        const VEC_DATA_TYPE(int, size) all_ones  = ~0;                                   \
        return select(all_zeros, all_ones, (SELECT_VEC_DATA_TYPE(int, size))(a == 0));   \
    }

/** For each element of input vector, the corresponding bits of the result item are set
 * if the input item is non-zero.
 *
 * @param[in] size Size of vector.
 *
 * @returns Output vector with bits set when corresponding bit in @p a is non zero.
 */
#define ASYMM_MASK_IF_NON_ZERO_IMPL(size)                                                    \
    inline VEC_DATA_TYPE(int, size) asymm_mask_if_non_zero##size(VEC_DATA_TYPE(int, size) a) \
    {                                                                                        \
        const VEC_DATA_TYPE(int, size) all_zeros = 0;                                        \
        const VEC_DATA_TYPE(int, size) all_ones  = ~0;                                       \
        return select(all_zeros, all_ones, (SELECT_VEC_DATA_TYPE(int, size))(a != 0));       \
    }

#define EXP_BARREL_SHIFTER_IMPL(size)                                                                                                                                                                         \
    inline VEC_DATA_TYPE(int, size) exp_barrel_shifter##size(VEC_DATA_TYPE(int, size) result, int exponent, int fp_multiplier, int k_integer_bits, int k_fractional_bits, VEC_DATA_TYPE(int, size) remainder) \
    {                                                                                                                                                                                                         \
        if(k_integer_bits > exponent)                                                                                                                                                                         \
        {                                                                                                                                                                                                     \
            const int k_shift_amount = k_integer_bits > exponent ? k_fractional_bits + exponent : 0;                                                                                                          \
            return ASYMM_SELECT_USING_MASK(                                                                                                                                                                   \
                    ASYMM_MASK_IF_NON_ZERO(remainder & (1 << k_shift_amount), size),                                                                                                                              \
                    ASYMM_MULT(result, fp_multiplier, size), result, size);                                                                                                                                       \
        }                                                                                                                                                                                                     \
        \
        return result;                                                                                                                                                                                        \
    }

/** Calculates \f$ exp(x) \f$ for x < 0.
 *
 * @param[in] size Size of vector.
 *
 * @return Result in fixed-point format Q0.
 */
#define ASYMM_EXP_ON_NEGATIVE_VALUES_IMPL(size)                                                                               \
    inline VEC_DATA_TYPE(int, size) asymm_exp_on_negative_values##size(VEC_DATA_TYPE(int, size) a, int k_integer_bits)        \
    {                                                                                                                         \
        const int k_fractional_bits = 31 - k_integer_bits;                                                                    \
        VEC_DATA_TYPE(int, size)                                                                                              \
        k_one_quarter = 1 << (k_fractional_bits - 2);                                                                         \
        VEC_DATA_TYPE(int, size)                                                                                              \
        mask = k_one_quarter - 1;                                                                                             \
        VEC_DATA_TYPE(int, size)                                                                                              \
        a_mod_quarter_minus_one_quarter = (a & mask) - k_one_quarter;                                                         \
        VEC_DATA_TYPE(int, size)                                                                                              \
        a_mod_quarter_minus_one_quarter_scaled = a_mod_quarter_minus_one_quarter << k_integer_bits;                           \
        VEC_DATA_TYPE(int, size)                                                                                              \
        result = ASYMM_EXP_ON_INTERVAL_BETWEEN_NEGATIVE_ONE_QUARTER_AND_0_EXCL(a_mod_quarter_minus_one_quarter_scaled, size); \
        VEC_DATA_TYPE(int, size)                                                                                              \
        remainder = a_mod_quarter_minus_one_quarter - a;                                                                      \
        \
        result = EXP_BARREL_SHIFTER(result, -2, 1672461947, k_integer_bits, k_fractional_bits, remainder, size);              \
        result = EXP_BARREL_SHIFTER(result, -1, 1302514674, k_integer_bits, k_fractional_bits, remainder, size);              \
        result = EXP_BARREL_SHIFTER(result, +0, 790015084, k_integer_bits, k_fractional_bits, remainder, size);               \
        result = EXP_BARREL_SHIFTER(result, +1, 290630308, k_integer_bits, k_fractional_bits, remainder, size);               \
        result = EXP_BARREL_SHIFTER(result, +2, 39332535, k_integer_bits, k_fractional_bits, remainder, size);                \
        result = EXP_BARREL_SHIFTER(result, +3, 720401, k_integer_bits, k_fractional_bits, remainder, size);                  \
        result = EXP_BARREL_SHIFTER(result, +4, 242, k_integer_bits, k_fractional_bits, remainder, size);                     \
        \
        if(k_integer_bits > 5)                                                                                                \
        {                                                                                                                     \
            const VEC_DATA_TYPE(int, size) clamp = -(1 << (k_fractional_bits + 5));                                           \
            result = ASYMM_SELECT_USING_MASK(ASYMM_MASK_IF_NON_ZERO(a < clamp, size), 0, result, size);                       \
        }                                                                                                                     \
        \
        const VEC_DATA_TYPE(int, size) Q0_one = INT_MAX;                                                                      \
        return ASYMM_SELECT_USING_MASK(ASYMM_MASK_IF_ZERO(a, size), Q0_one, result, size);                                    \
    }

/** Calculates the product of a integer value by a power of two, with either a positive exponent
 * (equivalent to an arithmetic left shift, saturating) or a negative exponent
 * (equivalent to an arithmetic right shift, rounding to nearest).
 *
 * @param[in] size Size of vector.
 *
 * @return Arithmetic left or right shift.
 */
#define ASYMM_SATURATING_ROUNDING_MULT_BY_POW2_IMPL(size)                                                                  \
    inline VEC_DATA_TYPE(int, size) asymm_saturating_rounding_mult_by_pow2##size(VEC_DATA_TYPE(int, size) x, int exponent) \
    {                                                                                                                      \
        if(exponent < 0)                                                                                                   \
        {                                                                                                                  \
            return ASYMM_ROUNDING_DIVIDE_BY_POW2(x, -exponent, size);                                                      \
        }                                                                                                                  \
        \
        const VEC_DATA_TYPE(int, size) min = INT_MIN;                                                                      \
        const VEC_DATA_TYPE(int, size) max = INT_MAX;                                                                      \
        int threshold = ((1 << (31 - exponent)) - 1);                                                                      \
        VEC_DATA_TYPE(int, size)                                                                                           \
        positive_mask = ASYMM_MASK_IF_NON_ZERO(x > threshold, size);                                                       \
        VEC_DATA_TYPE(int, size)                                                                                           \
        negative_mask = ASYMM_MASK_IF_NON_ZERO(x < -threshold, size);                                                      \
        VEC_DATA_TYPE(int, size)                                                                                           \
        result = x << exponent;                                                                                            \
        result = ASYMM_SELECT_USING_MASK(positive_mask, max, result, size);                                                \
        result = ASYMM_SELECT_USING_MASK(negative_mask, min, result, size);                                                \
        return result;                                                                                                     \
    }

/** Calculates (a+b)/2, rounded to the nearest integer.
 * Equivalent to VRHADD in the Arm Arm® Neon™ instruction set.
 *
 * @param[in] size Size of vector.
 *
 * @return (a+b)/2, rounded to the nearest integer.
 */
#define ASYMM_ROUNDING_HALF_SUM_IMPL(size)                                                                                \
    inline VEC_DATA_TYPE(int, size) asymm_rounding_half_sum##size(VEC_DATA_TYPE(int, size) a, VEC_DATA_TYPE(int, size) b) \
    {                                                                                                                     \
        VEC_DATA_TYPE(long, size)                                                                                         \
        a64 = convert_long##size(a);                                                                                      \
        VEC_DATA_TYPE(long, size)                                                                                         \
        b64 = convert_long##size(b);                                                                                      \
        VEC_DATA_TYPE(long, size)                                                                                         \
        sum = a64 + b64;                                                                                                  \
        const VEC_DATA_TYPE(long, size) one       = 1;                                                                    \
        const VEC_DATA_TYPE(long, size) minus_one = -1;                                                                   \
        VEC_DATA_TYPE(long, size)                                                                                         \
        sign = select(minus_one, one, (SELECT_VEC_DATA_TYPE(long, size))(sum >= 0));                                      \
        return convert_int##size((sum + sign) / 2);                                                                       \
    }

/** Calculates \f$ 1 / (1 + x) \f$ for x in (0, 1).
 *
 * @param[in] size Size of vector.
 *
 * @return Result in fixed-point format Q0.
 */
#define ASYMM_ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_IMPL(size)                                                    \
    inline VEC_DATA_TYPE(int, size) asymm_one_over_one_plus_x_for_x_in_0_1##size(VEC_DATA_TYPE(int, size) a) \
    {                                                                                                        \
        const VEC_DATA_TYPE(int, size) Q0_one = INT_MAX;                                                     \
        const VEC_DATA_TYPE(int, size) Q2_one = 1 << (31 - 2);                                               \
        VEC_DATA_TYPE(int, size)                                                                             \
        half_denominator = ASYMM_ROUNDING_HALF_SUM(a, Q0_one, size);                                         \
        const VEC_DATA_TYPE(int, size) Q2_48_over_17     = 1515870810;                                       \
        const VEC_DATA_TYPE(int, size) Q2_neg_32_over_17 = -1010580540;                                      \
        VEC_DATA_TYPE(int, size)                                                                             \
        x = Q2_48_over_17 + ASYMM_MULT(half_denominator, Q2_neg_32_over_17, size);                           \
        for(int i = 0; i < 3; i++)                                                                           \
        {                                                                                                    \
            VEC_DATA_TYPE(int, size)                                                                         \
            half_denominator_times_x = ASYMM_MULT(half_denominator, x, size);                                \
            VEC_DATA_TYPE(int, size)                                                                         \
            one_minus_half_denominator_times_x = Q2_one - half_denominator_times_x;                          \
            VEC_DATA_TYPE(int, size)                                                                         \
            tmp = ASYMM_MULT(x, one_minus_half_denominator_times_x, size);                                   \
            x   = x + ASYMM_SATURATING_ROUNDING_MULT_BY_POW2(tmp, 2, size);                                  \
        }                                                                                                    \
        return ASYMM_SATURATING_ROUNDING_MULT_BY_POW2(x, 1, size);                                           \
    }

/** Considering the integer value as fixed-point, change the number of integer bits and update value accordingly.
 *
 * @param[in] size Size of vector.
 *
 * @return Rescaled value.
 */
#define ASYMM_RESCALE_IMPL(size)                                                                                                    \
    inline VEC_DATA_TYPE(int, size) asymm_rescale##size(VEC_DATA_TYPE(int, size) value, int src_integer_bits, int dst_integer_bits) \
    {                                                                                                                               \
        int exponent = src_integer_bits - dst_integer_bits;                                                                         \
        return ASYMM_SATURATING_ROUNDING_MULT_BY_POW2(value, exponent, size);                                                       \
    }

#define QUANTIZE_STR(input, offset, scale, type, size) quantize_##type##size(input, offset, scale)
#define QUANTIZE(input, offset, scale, type, size) QUANTIZE_STR(input, offset, scale, type, size)
#define DEQUANTIZE_STR(input, offset, scale, type, size) dequantize_##type##size(input, offset, scale)
#define DEQUANTIZE(input, offset, scale, type, size) DEQUANTIZE_STR(input, offset, scale, type, size)

#define ASYMM_ROUNDING_DIVIDE_BY_POW2_STR(x, exponent, size) asymm_rounding_divide_by_POW2_##size(x, exponent)
#define ASYMM_ROUNDING_DIVIDE_BY_POW2(x, exponent, size) ASYMM_ROUNDING_DIVIDE_BY_POW2_STR(x, exponent, size)
#define ASYMM_MULT_STR(a, b, size) asymm_mult##size(a, b)
#define ASYMM_MULT(a, b, size) ASYMM_MULT_STR(a, b, size)
#define ASYMM_MULT_BY_QUANT_MULTIPLIER_GREATER_THAN_ONE(x, quantized_multiplier, left_shift, size) \
    ASYMM_MULT(x *((VEC_DATA_TYPE(int, size))(1) << (-left_shift)), quantized_multiplier, size)
#define ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(x, quantized_multiplier, right_shift, size) \
    ASYMM_ROUNDING_DIVIDE_BY_POW2(ASYMM_MULT(x, quantized_multiplier, size), right_shift, size)
#define ASYMM_EXP_ON_INTERVAL_BETWEEN_NEGATIVE_ONE_QUARTER_AND_0_EXCL(a, size) asymm_exp_on_interval_between_negative_one_quarter_and_0_excl##size(a)
#define ASYMM_SELECT_USING_MASK(if_mask, then_val, else_val, size) asymm_select_using_mask##size(if_mask, then_val, else_val)
#define ASYMM_MASK_IF_ZERO(a, size) asymm_mask_if_zero##size(a)
#define ASYMM_MASK_IF_NON_ZERO(a, size) asymm_mask_if_non_zero##size(a)
#define EXP_BARREL_SHIFTER(result, exponent, fp_multiplier, k_integer_bits, k_fractional_bits, remainder, size) exp_barrel_shifter##size(result, exponent, fp_multiplier, k_integer_bits, k_fractional_bits, remainder)
#define ASYMM_EXP_ON_NEGATIVE_VALUES_STR(a, k_integer_bits, size) asymm_exp_on_negative_values##size(a, k_integer_bits)
#define ASYMM_EXP_ON_NEGATIVE_VALUES(a, k_integer_bits, size) ASYMM_EXP_ON_NEGATIVE_VALUES_STR(a, k_integer_bits, size)
#define ASYMM_ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_STR(a, size) asymm_one_over_one_plus_x_for_x_in_0_1##size(a)
#define ASYMM_ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1(a, size) ASYMM_ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_STR(a, size)
#define ASYMM_SATURATING_ROUNDING_MULT_BY_POW2(x, exponent, size) asymm_saturating_rounding_mult_by_pow2##size(x, exponent)
#define ASYMM_ROUNDING_HALF_SUM(a, b, size) asymm_rounding_half_sum##size(a, b)
#define ASYMM_RESCALE_STR(value, src_integer_bits, dst_integer_bits, size) asymm_rescale##size(value, src_integer_bits, dst_integer_bits)
#define ASYMM_RESCALE(value, src_integer_bits, dst_integer_bits, size) ASYMM_RESCALE_STR(value, src_integer_bits, dst_integer_bits, size)

#define MULTIPLY_BY_QUANTIZED_MULTIPLIER_IMPL(size)                                                                             \
    inline VEC_DATA_TYPE(int, size) multiply_by_quantized_multiplier##size(VEC_DATA_TYPE(int, size) input, int qmul, int shift) \
    {                                                                                                                           \
        const int left_shift  = shift > 0 ? shift : 0;                                                                          \
        const int right_shift = shift > 0 ? 0 : -shift;                                                                         \
        return ASYMM_ROUNDING_DIVIDE_BY_POW2(ASYMM_MULT(input * (1 << left_shift), qmul, size), right_shift, size);             \
    }
#define MULTIPLY_BY_QUANTIZED_MULTIPLIER(input, qmul, shift, size) multiply_by_quantized_multiplier##size(input, qmul, shift)

QUANTIZE_IMPL(uchar, 1)
QUANTIZE_IMPL(char, 1)
QUANTIZE_IMPL(uint, 1)
QUANTIZE_IMPL(int, 1)
QUANTIZE_IMPL(uchar, 2)
QUANTIZE_IMPL(char, 2)
QUANTIZE_IMPL(uint, 2)
QUANTIZE_IMPL(int, 2)
QUANTIZE_IMPL(uchar, 3)
QUANTIZE_IMPL(char, 3)
QUANTIZE_IMPL(uint, 3)
QUANTIZE_IMPL(int, 3)
QUANTIZE_IMPL(uchar, 4)
QUANTIZE_IMPL(ushort, 4)
QUANTIZE_IMPL(short, 4)
QUANTIZE_IMPL(int, 4)
QUANTIZE_IMPL(uchar, 8)
QUANTIZE_IMPL(char, 8)
QUANTIZE_IMPL(uint, 8)
QUANTIZE_IMPL(int, 8)
QUANTIZE_IMPL(uchar, 16)
QUANTIZE_IMPL(char, 16)
QUANTIZE_IMPL(ushort, 16)
QUANTIZE_IMPL(short, 16)
QUANTIZE_IMPL(uint, 16)
QUANTIZE_IMPL(int, 16)

DEQUANTIZE_IMPL(uchar, 1)
DEQUANTIZE_IMPL(char, 1)
DEQUANTIZE_IMPL(uint, 1)
DEQUANTIZE_IMPL(int, 1)
DEQUANTIZE_IMPL(uchar, 2)
DEQUANTIZE_IMPL(char, 2)
DEQUANTIZE_IMPL(uint, 2)
DEQUANTIZE_IMPL(int, 2)
DEQUANTIZE_IMPL(uchar, 3)
DEQUANTIZE_IMPL(char, 3)
DEQUANTIZE_IMPL(uint, 3)
DEQUANTIZE_IMPL(int, 3)
DEQUANTIZE_IMPL(uchar, 4)
DEQUANTIZE_IMPL(ushort, 4)
DEQUANTIZE_IMPL(short, 4)
DEQUANTIZE_IMPL(int, 4)
DEQUANTIZE_IMPL(uchar, 8)
DEQUANTIZE_IMPL(char, 8)
DEQUANTIZE_IMPL(uint, 8)
DEQUANTIZE_IMPL(int, 8)
DEQUANTIZE_IMPL(uchar, 16)
DEQUANTIZE_IMPL(char, 16)
DEQUANTIZE_IMPL(ushort, 16)
DEQUANTIZE_IMPL(short, 16)
DEQUANTIZE_IMPL(uint, 16)
DEQUANTIZE_IMPL(int, 16)

ASYMM_ROUNDING_DIVIDE_BY_POW2_IMPL(1)
ASYMM_ROUNDING_DIVIDE_BY_POW2_IMPL(2)
ASYMM_ROUNDING_DIVIDE_BY_POW2_IMPL(3)
ASYMM_ROUNDING_DIVIDE_BY_POW2_IMPL(4)
ASYMM_ROUNDING_DIVIDE_BY_POW2_IMPL(8)
ASYMM_ROUNDING_DIVIDE_BY_POW2_IMPL(16)

ASYMM_MULT_IMPL(1)
ASYMM_MULT_IMPL(2)
ASYMM_MULT_IMPL(3)
ASYMM_MULT_IMPL(4)
ASYMM_MULT_IMPL(8)
ASYMM_MULT_IMPL(16)

ASYMM_EXP_ON_INTERVAL_BETWEEN_NEGATIVE_ONE_QUARTER_AND_0_EXCL_IMPL(1)
ASYMM_EXP_ON_INTERVAL_BETWEEN_NEGATIVE_ONE_QUARTER_AND_0_EXCL_IMPL(2)
ASYMM_EXP_ON_INTERVAL_BETWEEN_NEGATIVE_ONE_QUARTER_AND_0_EXCL_IMPL(3)
ASYMM_EXP_ON_INTERVAL_BETWEEN_NEGATIVE_ONE_QUARTER_AND_0_EXCL_IMPL(4)
ASYMM_EXP_ON_INTERVAL_BETWEEN_NEGATIVE_ONE_QUARTER_AND_0_EXCL_IMPL(8)
ASYMM_EXP_ON_INTERVAL_BETWEEN_NEGATIVE_ONE_QUARTER_AND_0_EXCL_IMPL(16)

ASYMM_SELECT_USING_MASK_IMPL(1)
ASYMM_SELECT_USING_MASK_IMPL(2)
ASYMM_SELECT_USING_MASK_IMPL(3)
ASYMM_SELECT_USING_MASK_IMPL(4)
ASYMM_SELECT_USING_MASK_IMPL(8)
ASYMM_SELECT_USING_MASK_IMPL(16)

ASYMM_MASK_IF_ZERO_IMPL(1)
ASYMM_MASK_IF_ZERO_IMPL(2)
ASYMM_MASK_IF_ZERO_IMPL(3)
ASYMM_MASK_IF_ZERO_IMPL(4)
ASYMM_MASK_IF_ZERO_IMPL(8)
ASYMM_MASK_IF_ZERO_IMPL(16)

ASYMM_MASK_IF_NON_ZERO_IMPL(1)
ASYMM_MASK_IF_NON_ZERO_IMPL(2)
ASYMM_MASK_IF_NON_ZERO_IMPL(3)
ASYMM_MASK_IF_NON_ZERO_IMPL(4)
ASYMM_MASK_IF_NON_ZERO_IMPL(8)
ASYMM_MASK_IF_NON_ZERO_IMPL(16)

EXP_BARREL_SHIFTER_IMPL(1)
EXP_BARREL_SHIFTER_IMPL(2)
EXP_BARREL_SHIFTER_IMPL(3)
EXP_BARREL_SHIFTER_IMPL(4)
EXP_BARREL_SHIFTER_IMPL(8)
EXP_BARREL_SHIFTER_IMPL(16)

ASYMM_EXP_ON_NEGATIVE_VALUES_IMPL(1)
ASYMM_EXP_ON_NEGATIVE_VALUES_IMPL(2)
ASYMM_EXP_ON_NEGATIVE_VALUES_IMPL(3)
ASYMM_EXP_ON_NEGATIVE_VALUES_IMPL(4)
ASYMM_EXP_ON_NEGATIVE_VALUES_IMPL(8)
ASYMM_EXP_ON_NEGATIVE_VALUES_IMPL(16)

ASYMM_SATURATING_ROUNDING_MULT_BY_POW2_IMPL(1)
ASYMM_SATURATING_ROUNDING_MULT_BY_POW2_IMPL(2)
ASYMM_SATURATING_ROUNDING_MULT_BY_POW2_IMPL(3)
ASYMM_SATURATING_ROUNDING_MULT_BY_POW2_IMPL(4)
ASYMM_SATURATING_ROUNDING_MULT_BY_POW2_IMPL(8)
ASYMM_SATURATING_ROUNDING_MULT_BY_POW2_IMPL(16)

ASYMM_ROUNDING_HALF_SUM_IMPL(1)
ASYMM_ROUNDING_HALF_SUM_IMPL(2)
ASYMM_ROUNDING_HALF_SUM_IMPL(3)
ASYMM_ROUNDING_HALF_SUM_IMPL(4)
ASYMM_ROUNDING_HALF_SUM_IMPL(8)
ASYMM_ROUNDING_HALF_SUM_IMPL(16)

ASYMM_ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_IMPL(1)
ASYMM_ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_IMPL(2)
ASYMM_ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_IMPL(3)
ASYMM_ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_IMPL(4)
ASYMM_ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_IMPL(8)
ASYMM_ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_IMPL(16)

ASYMM_RESCALE_IMPL(1)
ASYMM_RESCALE_IMPL(2)
ASYMM_RESCALE_IMPL(3)
ASYMM_RESCALE_IMPL(4)
ASYMM_RESCALE_IMPL(8)
ASYMM_RESCALE_IMPL(16)

MULTIPLY_BY_QUANTIZED_MULTIPLIER_IMPL(1)
MULTIPLY_BY_QUANTIZED_MULTIPLIER_IMPL(2)
MULTIPLY_BY_QUANTIZED_MULTIPLIER_IMPL(3)
MULTIPLY_BY_QUANTIZED_MULTIPLIER_IMPL(4)
MULTIPLY_BY_QUANTIZED_MULTIPLIER_IMPL(8)
MULTIPLY_BY_QUANTIZED_MULTIPLIER_IMPL(16)

#endif // ARM_COMPUTE_HELPERS_ASYMM_H
