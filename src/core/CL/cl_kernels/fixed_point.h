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
TYPE_ALIAS(int, qs32)

#define qs8_MIN ((char)CHAR_MIN)
#define qs8_MAX ((char)CHAR_MAX)
#define qs16_MIN ((short)SHRT_MIN)
#define qs16_MAX ((short)SHRT_MAX)
#define qs32_MIN ((int)INT_MIN)
#define qs32_MAX ((int)INT_MAX)

#define qu8_MIN ((uchar)0)
#define qu8_MAX ((uchar)UCHAR_MAX)
#define qu16_MIN ((ushort)0)
#define qu16_MAX ((ushort)USHRT_MAX)
#define qu32_MIN ((uint)0)
#define qu32_MAX ((uint)UINT_MAX)

#define qs8_TYPE char
#define qs8x1_TYPE char
#define qs8x2_TYPE char2
#define qs8x3_TYPE char3
#define qs8x4_TYPE char4
#define qs8x8_TYPE char8
#define qs8x16_TYPE char16

#define qs16_TYPE short
#define qs16x1_TYPE short
#define qs16x2_TYPE short2
#define qs16x3_TYPE short3
#define qs16x4_TYPE short4
#define qs16x8_TYPE short8
#define qs16x16_TYPE short16

#define qs32_TYPE int
#define qs32x1_TYPE int
#define qs32x2_TYPE int2
#define qs32x3_TYPE int3
#define qs32x4_TYPE int4
#define qs32x8_TYPE int8
#define qs32x16_TYPE int16

/* All internal constants are represented in the maximum supported fixed point format (QS16),
 * thus we define an additional shift parameter required to convert the constant
 * from the maximum supported format to the require one.
 */
#define qs8_SHIFT 8
#define qs16_SHIFT 0

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

/** Computes saturating absolute value of fixed point vector.
 *
 * @param[in] type the actual data type.
 *
 * @return The result of the fixed point absolute value.
 */
#define ABSQ_SAT_IMPL(type)                  \
    inline type abs_##type##_sat(type VopA)  \
    {                                        \
        return CONVERT_SAT(abs(VopA), type); \
    }

ABSQ_SAT_IMPL(qs8x16)
ABSQ_SAT_IMPL(qs16x8)

#define ABS_SAT_OP_EXPAND_STR(a, type, size) abs_##type##x##size##_sat((a))
#define ABS_SAT_OP_EXPAND(a, type, size) ABS_SAT_OP_EXPAND_STR(a, type, size)

/** Computes max of fixed point types.
 *
 * @param[in] type the actual data type.
 *
 * @return The result of the fixed point maximum.
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
MAXQ_IMPL(qs16x1)
MAXQ_IMPL(qs16x2)
MAXQ_IMPL(qs16x4)
MAXQ_IMPL(qs16x8)
MAXQ_IMPL(qs16x16)

#define MAX_OP_EXPAND_STR(a, b, type, size) max_##type##x##size((a), (b))
#define MAX_OP_EXPAND(a, b, type, size) MAX_OP_EXPAND_STR(a, b, type, size)

/** Computes saturated addition of fixed point types.
 *
 * @param[in] type the actual data type.
 *
 * @return The result of the fixed point addition. The result is saturated in case of overflow
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
ADDQ_SAT_IMPL(qs16x1)
ADDQ_SAT_IMPL(qs16x2)
ADDQ_SAT_IMPL(qs16x4)
ADDQ_SAT_IMPL(qs16x8)
ADDQ_SAT_IMPL(qs16x16)
ADDQ_SAT_IMPL(qs32x1)
ADDQ_SAT_IMPL(qs32x2)
ADDQ_SAT_IMPL(qs32x4)
ADDQ_SAT_IMPL(qs32x8)
ADDQ_SAT_IMPL(qs32x16)

#define ADD_SAT_OP_EXPAND_STR(a, b, type, size) add_sat_##type##x##size((a), (b))
#define ADD_SAT_OP_EXPAND(a, b, type, size) ADD_SAT_OP_EXPAND_STR(a, b, type, size)

/** Computes saturated subtraction of fixed point types.
 *
 * @param[in] type the actual data type.
 *
 * @return The result of the fixed point subtraction. The result is saturated in case of overflow
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
SUBQ_SAT_IMPL(qs16x1)
SUBQ_SAT_IMPL(qs16x2)
SUBQ_SAT_IMPL(qs16x4)
SUBQ_SAT_IMPL(qs16x8)
SUBQ_SAT_IMPL(qs16x16)

#define SUB_SAT_OP_EXPAND_STR(a, b, type, size) sub_sat_##type##x##size((a), (b))
#define SUB_SAT_OP_EXPAND(a, b, type, size) SUB_SAT_OP_EXPAND_STR(a, b, type, size)

/* Multiply of two fixed point numbers
 *
 * @param[in] type  the actual data type.
 * @param[in] itype the intermediate data type.
 *
 * @return The result of the fixed point multiplication.
 */
#define MULQ_IMPL(type, itype)                                                         \
    inline type mul_##type(type VopA, type VopB, int fixed_point_position)             \
    {                                                                                  \
        itype round_val = (itype)(1 << (fixed_point_position - 1));                    \
        itype res       = CONVERT((VopA), itype) * CONVERT((VopB), itype) + round_val; \
        return CONVERT((res >> (itype)fixed_point_position), type);                    \
    }

MULQ_IMPL(qs8x8, qs16x8)
MULQ_IMPL(qs16x8, qs32x8)
MULQ_IMPL(qs8x16, qs16x16)
MULQ_IMPL(qs16x16, qs32x16)

#define MUL_OP_EXPAND_STR(a, b, type, size, position) mul_##type##x##size((a), (b), (position))
#define MUL_OP_EXPAND(a, b, type, size, position) MUL_OP_EXPAND_STR(a, b, type, size, position)

/* Saturate multiply of two fixed point numbers
 *
 * @param[in] type  the actual data type.
 * @param[in] itype the intermediate data type.
 *
 * @return The result of the fixed point multiplication. The result is saturated in case of overflow
 */
#define MULQ_SAT_IMPL(type, itype)                                                            \
    inline type mul_sat_##type(type VopA, type VopB, int fixed_point_position)                \
    {                                                                                         \
        itype round_val = (itype)(1 << (fixed_point_position - 1));                           \
        itype res       = mad_sat(CONVERT((VopA), itype), CONVERT((VopB), itype), round_val); \
        return CONVERT_SAT((res >> (itype)fixed_point_position), type);                       \
    }

MULQ_SAT_IMPL(qs8x1, qs16x1)
MULQ_SAT_IMPL(qs8x2, qs16x2)
MULQ_SAT_IMPL(qs8x3, qs16x3)
MULQ_SAT_IMPL(qs8x4, qs16x4)
MULQ_SAT_IMPL(qs8x8, qs16x8)
MULQ_SAT_IMPL(qs8x16, qs16x16)
MULQ_SAT_IMPL(qs16x1, qs32x1)
MULQ_SAT_IMPL(qs16x2, qs32x2)
MULQ_SAT_IMPL(qs16x3, qs32x3)
MULQ_SAT_IMPL(qs16x4, qs32x4)
MULQ_SAT_IMPL(qs16x8, qs32x8)
MULQ_SAT_IMPL(qs16x16, qs32x16)

#define MUL_SAT_OP_EXPAND_STR(a, b, type, size, position) mul_sat_##type##x##size((a), (b), (position))
#define MUL_SAT_OP_EXPAND(a, b, type, size, position) MUL_SAT_OP_EXPAND_STR(a, b, type, size, position)

/** Saturate multiply-accumulate
 *
 * @param[in] type  the actual data type.
 * @param[in] itype the intermediate data type.
 *
 * @return The result of the fixed point multiply-accumulate. The result is saturated in case of overflow
 */
#define MLAQ_SAT_IMPL(type, itype)                                                                                 \
    type mla_sat_##type(type VopA, type VopB, type VopC, int fixed_point_position)                                 \
    {                                                                                                              \
        itype res = mad_sat(CONVERT(VopB, itype), CONVERT(VopC, itype), (itype)(1 << (fixed_point_position - 1))); \
        return add_sat(VopA, CONVERT_SAT(res >> (itype)fixed_point_position, type));                               \
    }

MLAQ_SAT_IMPL(qs8x8, qs16x8)
MLAQ_SAT_IMPL(qs8x16, qs16x16)
MLAQ_SAT_IMPL(qs16x8, qs32x8)

#define MLA_SAT_OP_EXPAND_STR(a, b, c, type, size, position) mla_sat_##type##x##size((a), (b), (c), (position))
#define MLA_SAT_OP_EXPAND(a, b, c, type, size, position) MLA_SAT_OP_EXPAND_STR(a, b, c, type, size, position)

/** Saturate multiply-accumulate long
 *
 * @param[in] type  the actual data type.
 * @param[in] itype the intermediate data type.
 *
 * @return The result of the fixed point multiply-accumulate long. The result is saturated in case of overflow
 */
#define MLALQ_SAT_IMPL(type, itype)                                                                                \
    itype mlal_sat_##type(itype VopA, type VopB, type VopC, int fixed_point_position)                              \
    {                                                                                                              \
        itype res = mad_sat(CONVERT(VopB, itype), CONVERT(VopC, itype), (itype)(1 << (fixed_point_position - 1))); \
        return add_sat(VopA, res >> (itype)fixed_point_position);                                                  \
    }

MLALQ_SAT_IMPL(qs8x8, qs16x8)
MLALQ_SAT_IMPL(qs16x8, qs32x8)

#define MLAL_SAT_OP_EXPAND_STR(a, b, c, type, size, position) mlal_sat_##type##x##size((a), (b), (c), (position))
#define MLAL_SAT_OP_EXPAND(a, b, c, type, size, position) MLAL_SAT_OP_EXPAND_STR(a, b, c, type, size, position)

/** Saturate division of two fixed point vectors
 *
 * @param[in] stype the actual scalar data type.
 * @param[in] type  the actual data type.
 * @param[in] itype the intermediate data type.
 *
 * @return The result of the fixed point division. The result is saturated in case of overflow
 */
#define DIVQ_SAT_IMPL(stype, type, itype)                                                                                                                                           \
    inline type div_sat_##type(type VopA, type VopB, int fixed_point_position)                                                                                                      \
    {                                                                                                                                                                               \
        itype conv_a      = CONVERT((VopA), itype);                                                                                                                                 \
        itype denominator = CONVERT((VopB), itype);                                                                                                                                 \
        itype numerator   = conv_a << (itype)(fixed_point_position);                                                                                                                \
        itype res         = select((itype)(numerator / denominator), select((itype)stype##_MAX, (itype)stype##_MIN, (itype)(conv_a < (itype)0)), (itype)(denominator == (itype)0)); \
        return CONVERT_SAT((res), type);                                                                                                                                            \
    }

DIVQ_SAT_IMPL(qs8, qs8x16, qs16x16)
DIVQ_SAT_IMPL(qs16, qs16x8, qs32x8)
DIVQ_SAT_IMPL(qs16, qs16x16, qs32x16)
DIVQ_SAT_IMPL(qs8, qs8, qs16)
DIVQ_SAT_IMPL(qs16, qs16, qs32)

#define DIV_SAT_OP_EXPAND_STR(a, b, type, position) div_sat_##type((a), (b), (position))
#define DIV_SAT_OP_EXPAND(a, b, type, position) DIV_SAT_OP_EXPAND_STR(a, b, type, position)

#define DIV_SAT_OP_VEC_EXPAND_STR(a, b, type, size, position) div_sat_##type##x##size((a), (b), (position))
#define DIV_SAT_OP_VEC_EXPAND(a, b, type, size, position) DIV_SAT_OP_VEC_EXPAND_STR(a, b, type, size, position)

/** Saturate exponential of a fixed point vector
 *
 * @note Implemented approach uses taylor polynomial to approximate the exponential function.
 *
 * @param[in] stype the actual scalar data type.
 * @param[in] type  the actual data type.
 * @param[in] size  the number of the calculated elements.
 *
 * @return The result of the fixed point exponential. The result is saturated in case of overflow
 */
#define EXPQ_IMPL(stype, type, size)                                                                                                              \
    inline type exp_sat_##type(type VopA, int fixed_point_position)                                                                               \
    {                                                                                                                                             \
        type const_one = (type)(1 << (fixed_point_position));                                                                                     \
        type ln2       = (type)((((0x58B9 >> (14 - fixed_point_position))) + 1) >> 1);                                                            \
        type inv_ln2   = (type)((((0x38AA >> (14 - fixed_point_position)) + 1) >> 1)) | const_one;                                                \
        type A         = (type)(((0x7FBA >> (14 - fixed_point_position)) + 1) >> 1);                                                              \
        type B         = (type)(((0x3FE9 >> (14 - fixed_point_position)) + 1) >> 1);                                                              \
        type C         = (type)(((0x1693 >> (14 - fixed_point_position)) + 1) >> 1);                                                              \
        type D         = (type)(((0x0592 >> (14 - fixed_point_position)) + 1) >> 1);                                                              \
        type m         = MUL_SAT_OP_EXPAND(VopA, inv_ln2, stype, size, fixed_point_position);                                                     \
        type dec_m     = m >> (type)fixed_point_position;                                                                                         \
        type alpha     = MUL_SAT_OP_EXPAND(dec_m << (type)fixed_point_position, ln2, stype, size, fixed_point_position);                          \
        alpha          = CONVERT(abs_diff(VopA, alpha), type);                                                                                    \
        type sum       = add_sat(MUL_SAT_OP_EXPAND(alpha, D, stype, size, fixed_point_position), C);                                              \
        sum            = add_sat(MUL_SAT_OP_EXPAND(alpha, sum, stype, size, fixed_point_position), B);                                            \
        sum            = add_sat(MUL_SAT_OP_EXPAND(alpha, sum, stype, size, fixed_point_position), A);                                            \
        sum            = add_sat(MUL_SAT_OP_EXPAND(alpha, sum, stype, size, fixed_point_position), const_one);                                    \
        return select((type)stype##_MAX, select(sum << dec_m, sum >> -dec_m, dec_m < (type)0), clz(sum) > dec_m); /* Saturate result if needed */ \
    }

EXPQ_IMPL(qs8, qs8x2, 2)
EXPQ_IMPL(qs8, qs8x4, 4)
EXPQ_IMPL(qs8, qs8x8, 8)
EXPQ_IMPL(qs8, qs8x16, 16)
EXPQ_IMPL(qs16, qs16x2, 2)
EXPQ_IMPL(qs16, qs16x4, 4)
EXPQ_IMPL(qs16, qs16x8, 8)
EXPQ_IMPL(qs16, qs16x16, 16)

#define EXP_OP_EXPAND_STR(a, type, size, position) exp_sat_##type##x##size((a), (position))
#define EXP_OP_EXPAND(a, type, size, position) EXP_OP_EXPAND_STR(a, type, size, position)

/** Saturate logarithm of a fixed point vector
 *
 * @note Implemented approach uses taylor polynomial to approximate the logarithm function.
 *
 * @param[in] stype the actual scalar data type.
 * @param[in] type  the actual data type.
 * @param[in] size  the number of the calculated elements.
 *
 * @return The result of the fixed point logarithm. The result is saturated in case of overflow
 */
#define LOGQ_IMPL(stype, type, size)                                                                                                       \
    inline type log_sat_##type(type VopA, int fixed_point_position)                                                                        \
    {                                                                                                                                      \
        type const_one = (type)(1 << (fixed_point_position));                                                                              \
        type ln2       = (type)(0x58B9 >> (15 - fixed_point_position));  /* 1.4384189 */                                                   \
        type A         = (type)(0x5C0F >> (14 - fixed_point_position));  /* 1.4384189 */                                                   \
        type B         = -(type)(0x56AE >> (15 - fixed_point_position)); /* -0.6771900 */                                                  \
        type C         = (type)(0x2933 >> (15 - fixed_point_position));  /* 0.3218538 */                                                   \
        type D         = -(type)(0x0AA7 >> (15 - fixed_point_position)); /* -0.0832229 */                                                  \
        type inter_a   = select(VopA, DIV_SAT_OP_VEC_EXPAND(const_one, VopA, stype, size, fixed_point_position), VopA < const_one);        \
        type shift_val = (type)(15 - stype##_SHIFT) - clz(inter_a >> (type)fixed_point_position);                                          \
        inter_a        = inter_a >> shift_val;                                                                                             \
        inter_a        = sub_sat(inter_a, const_one);                                                                                      \
        type sum       = add_sat(MUL_SAT_OP_EXPAND(inter_a, D, stype, size, fixed_point_position), C);                                     \
        sum            = add_sat(MUL_SAT_OP_EXPAND(inter_a, sum, stype, size, fixed_point_position), B);                                   \
        sum            = add_sat(MUL_SAT_OP_EXPAND(inter_a, sum, stype, size, fixed_point_position), A);                                   \
        sum            = MUL_SAT_OP_EXPAND(inter_a, sum, stype, size, fixed_point_position);                                               \
        sum            = MUL_SAT_OP_EXPAND(add_sat(sum, shift_val << (type)fixed_point_position), ln2, stype, size, fixed_point_position); \
        return select(select(sum, -sum, VopA < const_one), (type)0, VopA < (type)0); /* Saturate result if needed */                       \
    }

LOGQ_IMPL(qs8, qs8x16, 16)
LOGQ_IMPL(qs16, qs16x8, 8)
LOGQ_IMPL(qs16, qs16x16, 16)

#define LOG_OP_EXPAND_STR(a, type, size, position) log_sat_##type##x##size((a), (position))
#define LOG_OP_EXPAND(a, type, size, position) LOG_OP_EXPAND_STR(a, type, size, position)

/** Saturate inverse square root of a fixed point vector
 *
 * @note Implemented approach uses Newton's method to approximate the inverse square root function.
 *
 * @param[in] stype the actual scalar data type.
 * @param[in] type  the actual data type.
 * @param[in] size  the number of the calculated elements.
 *
 * @return The result of the fixed point inverse square root. The result is saturated in case of overflow
 */
#define INVSQRTQ_IMPL(stype, type, size)                                                                                                                                                                                               \
    inline type invsqrt_sat_##type(type VopA, int fixed_point_position)                                                                                                                                                                \
    {                                                                                                                                                                                                                                  \
        type const_three = (type)(3 << (fixed_point_position));                                                                                                                                                                        \
        type shift_value = (type)(16 - stype##_SHIFT) - (clz(VopA) + (type)fixed_point_position);                                                                                                                                      \
        type temp        = select((type)(VopA >> shift_value), select((type)stype##_MAX, (type)(VopA << (-shift_value)), (type)(clz(VopA) > (-shift_value))), (type)(shift_value < (type)0));                                          \
        type x           = temp;                                                                                                                                                                                                       \
        x                = MUL_SAT_OP_EXPAND(x, sub_sat(const_three, MUL_SAT_OP_EXPAND(MUL_SAT_OP_EXPAND(x, x, stype, size, fixed_point_position), temp, stype, size, fixed_point_position)), stype, size, fixed_point_position) >> 1; \
        x                = MUL_SAT_OP_EXPAND(x, sub_sat(const_three, MUL_SAT_OP_EXPAND(MUL_SAT_OP_EXPAND(x, x, stype, size, fixed_point_position), temp, stype, size, fixed_point_position)), stype, size, fixed_point_position) >> 1; \
        x                = MUL_SAT_OP_EXPAND(x, sub_sat(const_three, MUL_SAT_OP_EXPAND(MUL_SAT_OP_EXPAND(x, x, stype, size, fixed_point_position), temp, stype, size, fixed_point_position)), stype, size, fixed_point_position) >> 1; \
        if(sizeof((stype)(1)) > 1) /* Perform more iterations if datatype is QS16 */                                                                                                                                                   \
        {                                                                                                                                                                                                                              \
            x = MUL_SAT_OP_EXPAND(x, sub_sat(const_three, MUL_SAT_OP_EXPAND(MUL_SAT_OP_EXPAND(x, x, stype, size, fixed_point_position), temp, stype, size, fixed_point_position)), stype, size, fixed_point_position) >> 1;            \
            x = MUL_SAT_OP_EXPAND(x, sub_sat(const_three, MUL_SAT_OP_EXPAND(MUL_SAT_OP_EXPAND(x, x, stype, size, fixed_point_position), temp, stype, size, fixed_point_position)), stype, size, fixed_point_position) >> 1;            \
        }                                                                                                                                                                                                                              \
        type shift_value2 = select(shift_value >> 1, (-shift_value) >> 1, shift_value < (type)0);                                                                                                                                      \
        return select((type)(x >> shift_value2), select((type)stype##_MAX, (type)(x << shift_value2), (type)(clz(x) > shift_value2)), (type)(shift_value < (type)0)); /* Saturate result if needed */                                  \
    }

INVSQRTQ_IMPL(qs8, qs8x1, 1)
INVSQRTQ_IMPL(qs16, qs16x1, 1)
INVSQRTQ_IMPL(qs8, qs8x16, 16)
INVSQRTQ_IMPL(qs16, qs16x8, 8)

#define INVSQRT_OP_EXPAND_STR(a, type, size, position) invsqrt_sat_##type##x##size((a), (position))
#define INVSQRT_OP_EXPAND(a, type, size, position) INVSQRT_OP_EXPAND_STR(a, type, size, position)

/** Saturate hyperbolic tangent of a fixed point vector
 *
 * tanh(x) = (e^2x - 1)/(e^2x + 1)
 *
 * @param[in] stype the actual scalar data type.
 * @param[in] type  the actual data type.
 * @param[in] size  the number of the calculated elements.
 *
 * @return The result of the fixed point hyperbolic tangent. The result is saturated in case of overflow
 */
#define TANHQ_IMPL(stype, type, size)                                                                                                             \
    inline type tanh_sat_##type(type VopA, int fixed_point_position)                                                                              \
    {                                                                                                                                             \
        type const_one = (type)(1 << (fixed_point_position));                                                                                     \
        type const_two = (type)(2 << (fixed_point_position));                                                                                     \
        type exp2x     = EXP_OP_EXPAND(MUL_SAT_OP_EXPAND(const_two, VopA, stype, size, fixed_point_position), stype, size, fixed_point_position); \
        type num       = SUB_SAT_OP_EXPAND(exp2x, const_one, stype, size);                                                                        \
        type den       = ADD_SAT_OP_EXPAND(exp2x, const_one, stype, size);                                                                        \
        return DIV_SAT_OP_VEC_EXPAND(num, den, stype, size, fixed_point_position);                                                                \
    }

TANHQ_IMPL(qs8, qs8x16, 16)
TANHQ_IMPL(qs16, qs16x8, 8)

#define TANH_OP_EXPAND_STR(a, type, size, position) tanh_sat_##type##x##size((a), (position))
#define TANH_OP_EXPAND(a, type, size, position) TANH_OP_EXPAND_STR(a, type, size, position)

#define floatx16 float16
#define float16_TYPE float16

#define CONVERTQ_DOWN_IMPL(in_type, out_type)                                                                                      \
    inline out_type convert_##out_type##_##in_type(in_type a, int fixed_point_position)                                            \
    {                                                                                                                              \
        return CONVERT(a * (1 << fixed_point_position) + select((in_type)-0.5, (in_type)0.5, isgreater(a, (in_type)0)), out_type); \
    }

CONVERTQ_DOWN_IMPL(float16, qs8x16)
CONVERTQ_DOWN_IMPL(float16, qs16x16)

#define CONVERTQ_DOWN_SAT_IMPL(in_type, out_type)                                                                                      \
    inline out_type convert_##out_type##_##in_type##_sat(in_type a, int fixed_point_position)                                          \
    {                                                                                                                                  \
        return CONVERT_SAT(a * (1 << fixed_point_position) + select((in_type)-0.5, (in_type)0.5, isgreater(a, (in_type)0)), out_type); \
    }

CONVERTQ_DOWN_SAT_IMPL(float16, qs8x16)
CONVERTQ_DOWN_SAT_IMPL(float16, qs16x16)

#define CONVERTQ_UP_IMPL(in_type, out_type)                                             \
    inline out_type convert_##out_type##_##in_type(in_type a, int fixed_point_position) \
    {                                                                                   \
        return CONVERT(a, out_type) / (1 << fixed_point_position);                      \
    }

CONVERTQ_UP_IMPL(qs8x16, float16)
CONVERTQ_UP_IMPL(qs16x16, float16)

#define SQCVT_SAT_IMPL(type)                                                                    \
    inline type sqcvt_##type##_sat(float a, int fixed_point_position)                           \
    {                                                                                           \
        return CONVERT_SAT((a * (1 << fixed_point_position) + ((a < 0) ? -0.5f : 0.5f)), type); \
    }

SQCVT_SAT_IMPL(qs8)
SQCVT_SAT_IMPL(qs16)

#define SQCVT_SAT_OP_EXPAND_STR(a, type, position) sqcvt_##type##_sat((a), (position))
#define SQCVT_SAT_OP_EXPAND(a, type, position) SQCVT_SAT_OP_EXPAND_STR((a), type, position)

#endif // ARM_COMPUTE_FIXED_POINT_H
