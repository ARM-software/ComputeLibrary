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
#ifndef ARM_COMPUTE_FIXED_POINT_H
#define ARM_COMPUTE_FIXED_POINT_H

#define TYPE_ALIAS(type, alias)  \
    typedef type alias;          \
    typedef type alias##x##1;    \
    typedef type##2 alias##x##2; \
    typedef type##3 alias##x##3; \
    typedef type##4 alias##x##4; \
    typedef type##8 alias##x##8; \
    typedef type##16 alias##x##16;

TYPE_ALIAS(char, qs8)
TYPE_ALIAS(short, qs16)

#define qs8_MIN ((char)CHAR_MIN)
#define qs8_MAX ((char)CHAR_MAX)
#define qs16_MIN ((short)SHRT_MIN)
#define qs16_MAX ((short)SHRT_MAX)

#define qu8_MIN ((uchar)0)
#define qu8_MAX ((uchar)UCHAR_MAX)
#define qu16_MIN ((ushort)0)
#define qu16_MAX ((ushort)USHRT_MAX)

#define qs8_TYPE char
#define qs8x1_TYPE char
#define qs8x2_TYPE char2
#define qs8x4_TYPE char4
#define qs8x8_TYPE char8
#define qs8x16_TYPE char16

#define qs16_TYPE short
#define qs16x1_TYPE short
#define qs16x2_TYPE short2
#define qs16x4_TYPE short4
#define qs16x8_TYPE short8
#define qs16x16_TYPE short16

#undef VEC_DATA_TYPE_STR
#undef VEC_DATA_TYPE
#undef CONVERT_STR
#undef CONVERT
#undef CONVERT_SAT_STR
#undef CONVERT_SAT

#define VEC_DATA_TYPE_STR(type, size) type##x##size
#define VEC_DATA_TYPE(type, size) VEC_DATA_TYPE_STR(type, size)

#define CONVERT_STR3(x, type, rtype) (convert_##rtype((x)))
#define CONVERT_STR2(x, type, rtype) CONVERT_STR3(x, type, rtype)
#define CONVERT_STR(x, type) CONVERT_STR2(x, type, type##_TYPE)
#define CONVERT(x, type) CONVERT_STR(x, type)

#define CONVERT_SAT_STR3(x, type, rtype) (convert_##rtype##_sat((x)))
#define CONVERT_SAT_STR2(x, type, rtype) CONVERT_SAT_STR3(x, type, rtype)
#define CONVERT_SAT_STR(x, type) CONVERT_SAT_STR2(x, type, type##_TYPE)
#define CONVERT_SAT(x, type) CONVERT_SAT_STR(x, type)

/* Computes max of fixed point types.
 *
 * @param[in] type is the actual data type.
 *
 * @return The result of the fixed point vector maximum.
 */
#define MAXQ_IMPL(type)                          \
    inline type max_##type(type VopA, type VopB) \
    {                                            \
        return max(VopA, VopB);                  \
    }

MAXQ_IMPL(qs8x1)
MAXQ_IMPL(qs8x2)
MAXQ_IMPL(qs8x4)
MAXQ_IMPL(qs8x8)
MAXQ_IMPL(qs8x16)

#define MAX_OP_EXPAND_STR(a, b, type, size) max_##type##x##size((a), (b))
#define MAX_OP_EXPAND(a, b, type, size) MAX_OP_EXPAND_STR(a, b, type, size)

/* Computes saturated addition of fixed point types.
 *
 * @param[in] type is the actual data type.
 *
 * @return The result of the fixed point vector addition. The result is saturated in case of overflow
 */
#define ADDQ_SAT_IMPL(type)                          \
    inline type add_sat_##type(type VopA, type VopB) \
    {                                                \
        return add_sat(VopA, VopB);                  \
    }

ADDQ_SAT_IMPL(qs8x1)
ADDQ_SAT_IMPL(qs8x2)
ADDQ_SAT_IMPL(qs8x4)
ADDQ_SAT_IMPL(qs8x8)
ADDQ_SAT_IMPL(qs8x16)

#define ADD_SAT_OP_EXPAND_STR(a, b, type, size) add_sat_##type##x##size((a), (b))
#define ADD_SAT_OP_EXPAND(a, b, type, size) ADD_SAT_OP_EXPAND_STR(a, b, type, size)

/* Computes saturated subtraction of fixed point types.
 *
 * @param[in] type is the actual data type.
 *
 * @return The result of the fixed point vector subtraction. The result is saturated in case of overflow
 */
#define SUBQ_SAT_IMPL(type)                          \
    inline type sub_sat_##type(type VopA, type VopB) \
    {                                                \
        return sub_sat(VopA, VopB);                  \
    }

SUBQ_SAT_IMPL(qs8x1)
SUBQ_SAT_IMPL(qs8x2)
SUBQ_SAT_IMPL(qs8x4)
SUBQ_SAT_IMPL(qs8x8)
SUBQ_SAT_IMPL(qs8x16)

#define SUB_SAT_OP_EXPAND_STR(a, b, type, size) sub_sat_##type##x##size((a), (b))
#define SUB_SAT_OP_EXPAND(a, b, type, size) SUB_SAT_OP_EXPAND_STR(a, b, type, size)

/* Saturate multiply of two fixed point vectors
 *
 * @param[in] type  is the actual data type.
 * @param[in] itype is the intermediate data type.
 *
 * @return The result of the fixed point vector subtraction. The result is saturated in case of overflow
 */
#define MULQ_SAT_IMPL(type, itype)                                                            \
    inline type mul_sat_##type(type VopA, type VopB, int fixed_point_position)                \
    {                                                                                         \
        itype round_val = (itype)(1 << (fixed_point_position - 1));                           \
        itype res       = mad_sat(CONVERT((VopA), itype), CONVERT((VopB), itype), round_val); \
        return CONVERT_SAT((res >> (itype)fixed_point_position), type);                       \
    }

MULQ_SAT_IMPL(qs8x16, qs16x16)

#define MUL_SAT_OP_EXPAND_STR(a, b, type, size, position) mul_sat_##type##x##size((a), (b), (position))
#define MUL_SAT_OP_EXPAND(a, b, type, size, position) MUL_SAT_OP_EXPAND_STR(a, b, type, size, position)

/** Saturate division of two fixed point vectors
  *
  * @param[in] stype is the actual scalar data type.
  * @param[in] type  is the actual data type.
  * @param[in] itype is the intermediate data type.
  *
  * @return The result of the fixed point division. The result is saturated in case of overflow
  */
#define DIVQ_SAT_IMPL(stype, type, itype)                                                                                                                \
    inline type div_sat_##type(type VopA, type VopB, int fixed_point_position)                                                                           \
    {                                                                                                                                                    \
        itype conv_a      = CONVERT((VopA), itype);                                                                                                      \
        itype denominator = CONVERT((VopB), itype);                                                                                                      \
        itype numerator   = conv_a << (itype)(fixed_point_position);                                                                                     \
        itype res         = select(numerator / denominator, select((itype)stype##_MAX, (itype)stype##_MIN, conv_a < (itype)0), denominator == (itype)0); \
        return CONVERT_SAT((res), type);                                                                                                                 \
    }

DIVQ_SAT_IMPL(qs8, qs8x16, qs16x16)

#define DIV_SAT_OP_EXPAND_STR(a, b, type, size, position) div_sat_##type##x##size((a), (b), (position))
#define DIV_SAT_OP_EXPAND(a, b, type, size, position) DIV_SAT_OP_EXPAND_STR(a, b, type, size, position)

/** Saturate exponential fixed point 8 bit (16 elements)
  *
  * @param[in] a                    8 bit fixed point input vector
  * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8 bit fixed point exponential. The result is saturated in case of overflow
 */
qs8x16 inline exp_qs8x16(qs8x16 a, int fixed_point_position)
{
    // Constants (literal constants are calculated by converting the respective float to the fixed point with the highest supported fixed point position)
    char16 const_one = (char16)(1 << (fixed_point_position));
    char16 ln2       = (char16)(((0x58 >> (6 - fixed_point_position)) + 1) >> 1);                 // 0.693147
    char16 inv_ln2   = ((char16)(((0x38 >> (6 - (fixed_point_position))) + 1) >> 1)) | const_one; // 1.442695
    char16 A         = (char16)(((0x7F >> (6 - (fixed_point_position))) + 1) >> 1);               // 0.9978546
    char16 B         = (char16)(((0x3F >> (6 - (fixed_point_position))) + 1) >> 1);               // 0.4994721
    char16 C         = (char16)(((0x16 >> (6 - (fixed_point_position))) + 1) >> 1);               // 0.1763723
    char16 D         = (char16)(((0x05 >> (6 - (fixed_point_position))) + 1) >> 1);               // 0.0435108

    // Perform range reduction [-log(2),log(2)]
    char16 m = mul_sat_qs8x16(a, inv_ln2, fixed_point_position);

    // get decimal part of m
    char16 dec_m = m >> (char16)fixed_point_position;

    char16 alpha = mul_sat_qs8x16(dec_m << (char16)fixed_point_position, ln2, fixed_point_position);
    alpha        = convert_char16(abs_diff(a, alpha));

    // Polynomial expansion
    char16 sum = add_sat_qs8x16(mul_sat_qs8x16(alpha, D, fixed_point_position), C);
    sum        = add_sat_qs8x16(mul_sat_qs8x16(alpha, sum, fixed_point_position), B);
    sum        = add_sat_qs8x16(mul_sat_qs8x16(alpha, sum, fixed_point_position), A);
    sum        = add_sat_qs8x16(mul_sat_qs8x16(alpha, sum, fixed_point_position), const_one);

    // Reconstruct and saturate result
    return select(select(sum << dec_m, sum >> -dec_m, dec_m < (char16)0), (char16)0x7F, clz(sum) <= dec_m);
}

#define EXP_OP_EXPAND_STR(a, type, size, position) exp_##type##x##size((a), (position))
#define EXP_OP_EXPAND(a, type, size, position) EXP_OP_EXPAND_STR(a, type, size, position)

#endif // ARM_COMPUTE_FIXED_POINT_H
