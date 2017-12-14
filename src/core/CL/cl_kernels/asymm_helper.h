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
#ifndef ARM_COMPUTE_ASYMM_HELPER_H
#define ARM_COMPUTE_ASYMM_HELPER_H

// Algoriths for these functions were taken from
// https://github.com/google/gemmlowp/blob/master/fixedpoint/fixedpoint.h
// and adapted to operate on integer vectors.

/** For each element of input vector, the corresponding bits of the result item are set
 * if the input item is zero.
 *
 * @param[in] a Input vector whose zero bits define which corresponding bits in result will be set.
 *
 * @returns Output vector with bits set when corresponding bit in @p a is zero.
 */
inline int16 asymm_mask_if_zero(int16 a)
{
    const int16 all_zeros = 0;
    const int16 all_ones  = ~0;
    return select(all_zeros, all_ones, a == 0);
}

/** For each element of input vector, the corresponding bits of the result item are set
 * if the input item is non-zero.
 *
 * @param[in] a Input vector whose non-zero bits define which corresponding bits in result will be set.
 *
 * @returns Output vector with bits set when corresponding bit in @p a is non zero.
 */
inline int16 asymm_mask_if_non_zero(int16 a)
{
    const int16 all_zeros = 0;
    const int16 all_ones  = ~0;
    return select(all_zeros, all_ones, a != 0);
}

/** Each bit of the result is set to the corresponding bit of either then_val or
 * else_val depending on whether the corresponding bit of if_mask is set.
 * Equivalent to the VBSL instruction in ARM NEON.
 *
 * @param[in] if_mask  Mask defines will bit be taken from @p then_val or @p else_val depending on corresponding bit in mask is set or not.
 * @param[in] then_val Value whose bit will be used for result when corresponding bit in @p if_mask is set.
 * @param[in] else_val Value whose bit will be used for result when corresponding bit in @p if_mask is not set.
 *
 * @returns Result contaning bits from @p then_val or from @p else_val depending on corresponding bit in @p if_mask is set or not.
 */
inline int16 asymm_select_using_mask(int16 if_mask, int16 then_val, int16 else_val)
{
    return (if_mask & then_val) ^ (~if_mask & else_val);
}

/** Correctly rounded to nearest division by a power of two.
 * Also known as a rounding arithmetic right shift.
 *
 * @param[in] x        Value needed to be divided by power of two.
 * @param[in] exponent Power of two, must be positive number.
 *
 * @return Arithmetic right shift.
 */
inline int16 asymm_rounding_divide_by_pow2(int16 x, int exponent)
{
    int16       mask      = (1 << exponent) - 1;
    const int16 zero      = 0;
    const int16 one       = 1;
    int16       threshold = (mask >> 1) + select(zero, one, x < 0);
    return (x >> exponent) + select(zero, one, (x & mask) > threshold);
}

/** Calculates the product of a integer value by a power of two, with either a positive exponent
 * (equivalent to an arithmetic left shift, saturating) or a negative exponent
 * (equivalent to an arithmetic right shift, rounding to nearest).
 *
 * @param[in] x        Value needed to be multiplied or divided by power of two depending on sign of @p exponent.
 * @param[in] exponent Power of two, can be positive or negative number.
 *
 * @return Arithmetic left or right shift.
 */
inline int16 asymm_saturating_rounding_mult_by_pow2(int16 x, int exponent)
{
    if(exponent < 0)
    {
        return asymm_rounding_divide_by_pow2(x, -exponent);
    }

    const int16 min           = INT_MIN;
    const int16 max           = INT_MAX;
    int         threshold     = ((1 << (31 - exponent)) - 1);
    int16       positive_mask = asymm_mask_if_non_zero(x > threshold);
    int16       negative_mask = asymm_mask_if_non_zero(x < -threshold);
    int16       result        = x << exponent;
    result                    = asymm_select_using_mask(positive_mask, max, result);
    result                    = asymm_select_using_mask(negative_mask, min, result);
    return result;
}

/** Calculates (a+b)/2, rounded to the nearest integer.
 * Equivalent to VRHADD in the ARM NEON instruction set.
 *
 * @param[in] a First term of half-sum.
 * @param[in] b Second term of half-sum.
 *
 * @return (a+b)/2, rounded to the nearest integer.
 */
inline int16 asymm_rounding_half_sum(int16 a, int16 b)
{
    long16       a64       = convert_long16(a);
    long16       b64       = convert_long16(b);
    long16       sum       = a64 + b64;
    const long16 one       = 1;
    const long16 minus_one = -1;
    long16       sign      = select(minus_one, one, sum >= 0);
    return convert_int16((sum + sign) / 2);
}

/** Product of two numbers, interpreting them as fixed-point values in the interval [-1, 1),
 * rounding to the nearest value, and saturating -1 * -1 to the maximum value.
 * This is equivalent to the VQRDMULH instruction in ARM NEON.
 *
 * @param[in] a First term of product.
 * @param[in] b Second term of product.
 *
 * @return Product of two numbers.
 */
inline int16 asymm_saturating_rounding_doubling_high_mul(int16 a, int16 b)
{
    int16  overflow     = (a == b) && (a == INT_MIN);
    long16 a_64         = convert_long16(a);
    long16 b_64         = convert_long16(b);
    long16 ab_64        = a_64 * b_64;
    long16 mask1        = 1 << 30;
    long16 mask2        = 1 - (1 << 30);
    long16 nudge        = select(mask2, mask1, ab_64 >= 0);
    long16 mask         = 1ll << 31;
    int16  ab_x2_high32 = convert_int16((ab_64 + nudge) / mask);
    return select(ab_x2_high32, INT_MAX, overflow);
}

/** Fixed-point multiplication.
 *
 * @param[in] a Argument 1 in fixed-point format Q(a).
 * @param[in] b Argument 2 in fixed-point format Q(b).
 *
 * @return Result in fixed-point format Q(a+b).
 */
inline int16 asymm_mult(int16 a, int16 b)
{
    return asymm_saturating_rounding_doubling_high_mul(a, b);
}

/** Calculates \f$ exp(x) \f$ for x in [-1/4, 0).
 *
 * @param[in] a Argument in fixed-point format Q0.
 *
 * @return Result in fixed-point format Q0.
 */
inline int16 asymm_exp_on_interval_between_negative_one_quarter_and_0_excl(int16 a)
{
    const int16 constant_term                            = 1895147668;
    const int16 constant_1_over_3                        = 715827883;
    const int   k_fractional_bits                        = 31;
    int16       x                                        = a + (1 << (k_fractional_bits - 3));
    int16       x2                                       = asymm_mult(x, x);
    int16       x3                                       = asymm_mult(x2, x);
    int16       x4                                       = asymm_mult(x2, x2);
    int16       x4_over_4                                = asymm_rounding_divide_by_pow2(x4, 2);
    int16       x4_over_24_plus_x3_over_6_plus_x2        = asymm_mult((x4_over_4 + x3), constant_1_over_3) + x2;
    int16       x4_over_24_plus_x3_over_6_plus_x2_over_2 = asymm_rounding_divide_by_pow2(x4_over_24_plus_x3_over_6_plus_x2, 1);
    return constant_term + asymm_mult(constant_term, x + x4_over_24_plus_x3_over_6_plus_x2_over_2);
}

/** Calculates \f$ exp(x) \f$ for x < 0.
 *
 * @param[in] a              Argument in fixed-point format Q(k_integer_bits).
 * @param[in] k_integer_bits Number of integer bit in argument.
 *
 * @return Result in fixed-point format Q0.
 */
inline int16 asymm_exp_on_negative_values(int16 a, int k_integer_bits)
{
    const int k_fractional_bits                      = 31 - k_integer_bits;
    int16     k_one_quarter                          = 1 << (k_fractional_bits - 2);
    int16     mask                                   = k_one_quarter - 1;
    int16     a_mod_quarter_minus_one_quarter        = (a & mask) - k_one_quarter;
    int16     a_mod_quarter_minus_one_quarter_scaled = a_mod_quarter_minus_one_quarter << k_integer_bits;
    int16     result                                 = asymm_exp_on_interval_between_negative_one_quarter_and_0_excl(a_mod_quarter_minus_one_quarter_scaled);
    int16     remainder                              = a_mod_quarter_minus_one_quarter - a;

#define EXP_BARREL_SHIFTER(Exponent, FixedPointMultiplier)                                       \
    if(k_integer_bits > Exponent)                                                                \
    {                                                                                            \
        const int k_shift_amount = k_integer_bits > Exponent ? k_fractional_bits + Exponent : 0; \
        result                   = asymm_select_using_mask(                                      \
                                                                                                 asymm_mask_if_non_zero(remainder & (1 << k_shift_amount)),                           \
                                                                                                 asymm_mult(result, FixedPointMultiplier), result);                                   \
    }
    EXP_BARREL_SHIFTER(-2, 1672461947);
    EXP_BARREL_SHIFTER(-1, 1302514674);
    EXP_BARREL_SHIFTER(+0, 790015084);
    EXP_BARREL_SHIFTER(+1, 290630308);
    EXP_BARREL_SHIFTER(+2, 39332535);
    EXP_BARREL_SHIFTER(+3, 720401);
    EXP_BARREL_SHIFTER(+4, 242);
#undef EXP_BARREL_SHIFTER

    if(k_integer_bits > 5)
    {
        const int16 clamp = -(1 << (k_fractional_bits + 5));
        result            = asymm_select_using_mask(asymm_mask_if_non_zero(a < clamp), 0, result);
    }

    const int16 Q0_one = INT_MAX;
    return asymm_select_using_mask(asymm_mask_if_zero(a), Q0_one, result);
}

/** Calculates \f$ 1 / (1 + x) \f$ for x in (0, 1).
 *
 * @param[in] a Argument in fixed-point format Q0.
 *
 * @return Result in fixed-point format Q0.
 */
inline int16 asymm_one_over_one_plus_x_for_x_in_0_1(int16 a)
{
    const int16 Q0_one            = INT_MAX;
    const int16 Q2_one            = 1 << (31 - 2);
    int16       half_denominator  = asymm_rounding_half_sum(a, Q0_one);
    const int16 Q2_48_over_17     = 1515870810;
    const int16 Q2_neg_32_over_17 = -1010580540;
    int16       x                 = Q2_48_over_17 + asymm_mult(half_denominator, Q2_neg_32_over_17);
    for(int i = 0; i < 3; i++)
    {
        int16 half_denominator_times_x           = asymm_mult(half_denominator, x);
        int16 one_minus_half_denominator_times_x = Q2_one - half_denominator_times_x;
        int16 tmp                                = asymm_mult(x, one_minus_half_denominator_times_x);
        x                                        = x + asymm_saturating_rounding_mult_by_pow2(tmp, 2);
    }
    return asymm_saturating_rounding_mult_by_pow2(x, 1);
}

/** Considering the integer value as fixed-point, change the number of integer bits and update value accordingly.
 *
 * @param[in] value            Value to be rescaled.
 * @param[in] src_integer_bits Old number of integer bits.
 * @param[in] dst_integer_bits New number of integer bits.
 *
 * @return Rescaled value.
 */
inline int16 asymm_rescale(int16 value, int src_integer_bits, int dst_integer_bits)
{
    int exponent = src_integer_bits - dst_integer_bits;
    return asymm_saturating_rounding_mult_by_pow2(value, exponent);
}

#endif // ARM_COMPUTE_ASYMM_HELPER_H
