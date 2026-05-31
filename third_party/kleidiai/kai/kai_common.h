//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif  // defined(__ARM_NEON)

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN(cppcoreguidelines-avoid-do-while,cppcoreguidelines-pro-type-vararg,cert-err33-c)
//
//   * cppcoreguidelines-avoid-do-while: do-while is necessary for macros.
//   * cppcoreguidelines-pro-type-vararg: use of variadic arguments in fprintf is expected.
//   * cert-err33-c: checking the output of fflush and fprintf is not necessary for error reporting.

#ifndef KLEIDIAI_ERROR_TRAP
#define KLEIDIAI_ERROR_TRAP 0
#endif

#ifndef KLEIDIAI_HAS_BUILTIN_UNREACHABLE
#define KLEIDIAI_HAS_BUILTIN_UNREACHABLE 0
#endif

#ifndef KLEIDIAI_HAS_BUILTIN_ASSUME0
#define KLEIDIAI_HAS_BUILTIN_ASSUME0 0
#endif

#if KLEIDIAI_ERROR_TRAP
#define KAI_ABORT() __builtin_trap()
#else
#define KAI_ABORT() abort()
#endif

#if KLEIDIAI_HAS_BUILTIN_UNREACHABLE
#define KAI_UNREACHABLE() __builtin_unreachable()
#elif KLEIDIAI_HAS_BUILTIN_ASSUME0
#define KAI_UNREACHABLE() __assume(0);
#else
#define KAI_UNREACHABLE()
#endif

#ifdef NDEBUG
#define KAI_ERROR(msg)   \
    do {                 \
        KAI_UNUSED(msg); \
        KAI_ABORT();     \
    } while (0)

#define KAI_ASSERT_MSG(cond, msg) \
    do {                          \
        KAI_UNUSED(msg);          \
        if (!(cond)) {            \
            KAI_UNREACHABLE();    \
        }                         \
    } while (0)
#else
#define KAI_ERROR(msg)                                        \
    do {                                                      \
        fflush(stdout);                                       \
        fprintf(stderr, "%s:%d %s", __FILE__, __LINE__, msg); \
        KAI_ABORT();                                          \
    } while (0)

#define KAI_ASSERT_MSG(cond, msg) KAI_ASSERT_ALWAYS_MSG(cond, msg)
#endif  // NDEBUG

#define KAI_ASSERT_ALWAYS_MSG(cond, msg) \
    do {                                 \
        if (!(cond)) {                   \
            KAI_ERROR(msg);              \
        }                                \
    } while (0)

// NOLINTEND(cppcoreguidelines-avoid-do-while,cppcoreguidelines-pro-type-vararg,cert-err33-c)

/// KAI_ASSERT* is used for logic sanity checking in the program
/// flow. Checks are optimized away in release builds same as
/// `assert`
#define KAI_ASSERT(cond) KAI_ASSERT_MSG(cond, #cond)
#define KAI_ASSERT_IF_MSG(precond, cond, msg) KAI_ASSERT_MSG(!(precond) || (cond), msg)
#define KAI_ASSERT_IF(precond, cond) KAI_ASSERT_IF_MSG(precond, cond, #precond " |-> " #cond)

/// `KAI_ASSERT_ALWAYS*` is same as `KAI_ASSERT*`, but doesn't get removed by `NDEBUG`
#define KAI_ASSERT_ALWAYS(cond) KAI_ASSERT_ALWAYS_MSG(cond, #cond)
#define KAI_ASSERT_ALWAYS_IF_MSG(precond, cond, msg) KAI_ASSERT_ALWAYS_MSG(!(precond) || (cond), msg)
#define KAI_ASSERT_ALWAYS_IF(precond, cond) KAI_ASSERT_ALWAYS_IF_MSG(precond, cond, #precond " |-> " #cond)

/// KAI_ASSUME* is used for function pre-condition checking, similar to `[[assume]]` in C++23.
/// So KAI_ASSUME should be used directly on the function parameters, rather than inside
/// function logic.
#define KAI_ASSUME_MSG KAI_ASSERT_MSG
#define KAI_ASSUME KAI_ASSERT
#define KAI_ASSUME_IF_MSG KAI_ASSERT_IF_MSG
#define KAI_ASSUME_IF KAI_ASSERT_IF

/// `KAI_ASSUME_ALWAYS*` is same as `KAI_ASSUME*`, but doesn't get removed by `NDEBUG`
#define KAI_ASSUME_ALWAYS_MSG KAI_ASSERT_ALWAYS_MSG
#define KAI_ASSUME_ALWAYS KAI_ASSERT_ALWAYS
#define KAI_ASSUME_ALWAYS_IF_MSG KAI_ASSERT_ALWAYS_IF_MSG
#define KAI_ASSUME_ALWAYS_IF KAI_ASSERT_ALWAYS_IF

/// Indicate that result of `x` is unused
#define KAI_UNUSED(x) (void)(x)

/// Return minimum or maximum of `a` and `b`
#define KAI_MIN(a, b) (((a) < (b)) ? (a) : (b))
#define KAI_MAX(a, b) (((a) > (b)) ? (a) : (b))

/// Largest supported SME vector length in bytes
#define KAI_SME_VEC_LENGTH_MAX_BYTES 256  // NOLINT(cppcoreguidelines-macro-to-enum,modernize-macro-to-enum)

/// Gets the version of the project in the Major.Minor.Patch semantic versioning format.
///
/// @return Project version as a string literal.
inline const char* kai_get_version(void) {
    return "1.19.0";
}

/// KleidiAI data types
/// Format: <byte 3>(reserved)|<byte 2>(num-bytes)|<byte 1>(type)|<byte 0>(variant-type)
enum kai_datatype {
    kai_dt_unknown = 0x0000,
    kai_dt_f32 = 0x0411,
    kai_dt_f16 = 0x0212,
    kai_dt_bf16 = 0x0213,
    kai_dt_int32 = 0x0421,
    kai_dt_int16 = 0x0222,
    kai_dt_int8 = 0x0124,
    kai_dt_uint32 = 0x0431,
    kai_dt_uint16 = 0x0232,
    kai_dt_uint8 = 0x0134,
    kai_dt_bool = 0x0441
};

/// Gets number of bytes for a given data type
/// @param[in] dt KleidiAI data type
///
/// @return the numbers of bytes for the data type
inline static size_t kai_get_datatype_size_in_bytes(enum kai_datatype dt) {
    return (size_t)(dt >> 8);
}

/// Converts a scalar f16 value to f32
/// @param[in] f16 The f16 value
///
/// @return the f32 value
#if defined(__ARM_NEON)
inline static float kai_cast_f32_f16(uint16_t f16) {
    float16_t f32 = 0;
    memcpy(&f32, &f16, sizeof(uint16_t));
    return (float)f32;
}
#endif

/// Converts a scalar bf16 value to f32
/// @param[in] bf16 The f16 value
///
/// @return the f32 value
inline static float kai_cast_f32_bf16(uint16_t bf16) {
    const uint32_t i32 = (bf16 << 16);
    float f32 = 0;
    memcpy(&f32, &i32, sizeof(i32));
    return f32;
}

/// Converts a f32 value to bf16
/// @param[in] f32 The f32 value
///
/// @return the bf16 value
inline static uint16_t kai_cast_bf16_f32(float f32) {
    uint16_t bf16 = 0;
#ifdef __ARM_FEATURE_BF16
    __asm__ __volatile__("bfcvt %h[output], %s[input]" : [output] "=w"(bf16) : [input] "w"(f32));
#else
    const uint32_t* i32 = (uint32_t*)(&f32);
    bf16 = (*i32 >> 16);
#endif
    return bf16;
}

/// Converts a scalar f32 value to f16
/// @param[in] f32 The f32 value
///
/// @return the f16 value
#if defined(__ARM_NEON)
inline static uint16_t kai_cast_f16_f32(float f32) {
    uint16_t f16 = 0;
    float16_t tmp = (float16_t)f32;
    memcpy(&f16, &tmp, sizeof(uint16_t));
    return f16;
}
#endif

inline static size_t kai_roundup(size_t a, size_t b) {
    return ((a + b - 1) / b) * b;
}

#if defined(__ARM_FEATURE_SVE2) || defined(_M_ARM64)
/// Gets the SME vector length for 8-bit elements.
uint64_t kai_get_sme_vector_length_u8(void);

/// Gets the SME vector length for 16-bit elements.
inline static uint64_t kai_get_sme_vector_length_u16(void) {
    return kai_get_sme_vector_length_u8() / 2;
}

/// Gets the SME vector length for 32-bit elements.
inline static uint64_t kai_get_sme_vector_length_u32(void) {
    return kai_get_sme_vector_length_u8() / 4;
}

/// Commit ZA to lazy save buffer
void kai_commit_za(void);
#endif  // defined(__ARM_FEATURE_SVE2) || defined(_M_ARM64)

/// Gets the SVE vector length for 8-bit elements.
uint64_t kai_get_sve_vector_length_u8(void);

/// Gets the SVE vector length for 16-bit elements.
inline static uint64_t kai_get_sve_vector_length_u16(void) {
    return kai_get_sve_vector_length_u8() / 2;
}

/// Gets the SVE vector length for 32-bit elements.
inline static uint64_t kai_get_sve_vector_length_u32(void) {
    return kai_get_sve_vector_length_u8() / 4;
}

/// Extends the sign bit of int 4-bit value (stored in int8_t variable)
/// @param[in] value The 4-bit int value
///
/// @return the int8_t value with sign extended
inline static int8_t kai_ext_sign_i8_i4(int8_t value) {
    // Make sure value holds correct int4 value
    KAI_ASSERT(value <= 0xF);

    return (value ^ 0x8) - 8;  // NOLINT(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
}

/// Parameter struct for RHS matrix packing (Quantized Symmetric Integer 8-bit with per-channel quantization)
struct kai_rhs_pack_qsi8cx_params {
    int32_t lhs_zero_point;  ///< LHS Matrix quantization zero-point
    float scale_multiplier;  ///< Product of input (refers to lhs and rhs) and output quantization scales.
};

/// Parameter struct for RHS matrix packing (Quantized Symmetric Integer 4-bit with per-block quantizatio and s1s0
/// nibble ordering)
struct kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params {
    int8_t lhs_zero_point;
    uint8_t rhs_zero_point;
    enum kai_datatype scale_dt;
};

/// Parameter struct for RHS matrix packing (KxN variant for int4 qsi4c32p_qsu4c32s1s0)
struct kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0_params {
    int8_t lhs_zero_point;
    uint8_t rhs_zero_point;
    enum kai_datatype scale_dt;
};

/// Parameter struct for RHS matrix packing
struct kai_rhs_pack_qs4cxs1s0_param {
    int8_t lhs_zero_point;   ///< LHS Matrix quantization zero-point
    uint8_t rhs_zero_point;  ///< RHS Matrix quantization zero-point
};

/// Requantization and clamp parameters for GEMM/GEMV output stage.
struct kai_matmul_requantize32_params {
    int32_t min_value;          ///< Minimum output value.
    int32_t max_value;          ///< Maximum output value.
    int32_t output_zero_point;  ///< Output quantization zero point.
};

#ifdef __cplusplus
}
#endif
