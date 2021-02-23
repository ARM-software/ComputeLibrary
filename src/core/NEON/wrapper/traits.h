/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_WRAPPER_TRAITS_H
#define ARM_COMPUTE_WRAPPER_TRAITS_H

#include <arm_neon.h>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif /* defined(__ARM_FEATURE_SVE) */

namespace arm_compute
{
namespace wrapper
{
namespace traits
{
// *INDENT-OFF*
// clang-format off

/** 64-bit vector tag */
struct vector_64_tag {};
/** 128-bit vector tag */
struct vector_128_tag {};

/** Create the appropriate Neon vector given its type and size in terms of elements */
template <typename T, int S> struct neon_vector;

// Specializations
#ifndef DOXYGEN_SKIP_THIS
template <> struct neon_vector<uint8_t, 8>{ using scalar_type = uint8_t; using type = uint8x8_t; using tag_type = vector_64_tag; };
template <> struct neon_vector<int8_t, 8>{ using scalar_type = int8_t; using type = int8x8_t; using tag_type = vector_64_tag; };
template <> struct neon_vector<uint8_t, 16>{ using scalar_type = uint8_t; using type = uint8x16_t; using tag_type = vector_128_tag; };
template <> struct neon_vector<int8_t, 16>{ using scalar_type = int8_t; using type = int8x16_t; using tag_type = vector_128_tag; };
template <> struct neon_vector<uint16_t, 4>{ using scalar_type = uint16_t; using type = uint16x4_t; using tag_type = vector_64_tag; };
template <> struct neon_vector<int16_t, 4>{ using scalar_type = int16_t; using type = int16x4_t; using tag_type = vector_64_tag; };
template <> struct neon_vector<uint16_t, 8>{ using scalar_type = uint16_t; using type = uint16x8_t; using tag_type = vector_128_tag; };
template <> struct neon_vector<uint16_t, 16>{ using scalar_type = uint16_t; using type = uint16x8x2_t; };
template <> struct neon_vector<int16_t, 8>{ using scalar_type = int16_t; using type = int16x8_t; using tag_type = vector_128_tag; };
template <> struct neon_vector<int16_t, 16>{ using scalar_type = int16_t; using type = int16x8x2_t; };
template <> struct neon_vector<uint32_t, 2>{ using scalar_type = uint32_t; using type = uint32x2_t; using tag_type = vector_64_tag; };
template <> struct neon_vector<int32_t, 2>{ using scalar_type = int32_t; using type = int32x2_t; using tag_type = vector_64_tag; };
template <> struct neon_vector<uint32_t, 4>{ using scalar_type = uint32_t; using type = uint32x4_t; using tag_type = vector_128_tag; };
template <> struct neon_vector<int32_t, 4>{ using scalar_type = int32_t; using type = int32x4_t; using tag_type = vector_128_tag; };
template <> struct neon_vector<uint64_t, 1>{ using scalar_type = uint64_t;using type = uint64x1_t; using tag_type = vector_64_tag; };
template <> struct neon_vector<int64_t, 1>{ using scalar_type = int64_t; using type = int64x1_t; using tag_type = vector_64_tag; };
template <> struct neon_vector<uint64_t, 2>{ using scalar_type = uint64_t; using type = uint64x2_t; using tag_type = vector_128_tag; };
template <> struct neon_vector<int64_t, 2>{ using scalar_type = int64_t; using type = int64x2_t; using tag_type = vector_128_tag; };
template <> struct neon_vector<float_t, 2>{ using scalar_type = float_t; using type = float32x2_t; using tag_type = vector_64_tag; };
template <> struct neon_vector<float_t, 4>{ using scalar_type = float_t; using type = float32x4_t; using tag_type = vector_128_tag; };

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <> struct neon_vector<float16_t, 4>{ using scalar_type = float16_t; using type = float16x4_t; using tag_type = vector_64_tag; };
template <> struct neon_vector<float16_t, 8>{ using scalar_type = float16_t; using type = float16x8_t; using tag_type = vector_128_tag; };
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#endif /* DOXYGEN_SKIP_THIS */

/**  Helper type template to get the type of a neon vector */
template <typename T, int S> using neon_vector_t = typename neon_vector<T, S>::type;
/**  Helper type template to get the tag type of a neon vector */
template <typename T, int S> using neon_vector_tag_t = typename neon_vector<T, S>::tag_type;

/** Vector bit-width enum class */
enum class BitWidth
{
    W64,  /**< 64-bit width */
    W128, /**< 128-bit width */
};

/** Create the appropriate Neon vector given its type and size in terms of bits */
template <typename T, BitWidth BW> struct neon_bitvector;
// Specializations
#ifndef DOXYGEN_SKIP_THIS
template <> struct neon_bitvector<uint8_t, BitWidth::W64>{ using type = uint8x8_t; using tag_type = vector_64_tag; };
template <> struct neon_bitvector<int8_t, BitWidth::W64>{ using type = int8x8_t; using tag_type = vector_64_tag; };
template <> struct neon_bitvector<uint8_t, BitWidth::W128>{ using type = uint8x16_t; using tag_type = vector_128_tag; };
template <> struct neon_bitvector<int8_t, BitWidth::W128>{ using type = int8x16_t; using tag_type = vector_128_tag; };
template <> struct neon_bitvector<uint16_t, BitWidth::W64>{ using type = uint16x4_t; using tag_type = vector_64_tag; };
template <> struct neon_bitvector<int16_t, BitWidth::W64>{ using type = int16x4_t; using tag_type = vector_64_tag; };
template <> struct neon_bitvector<uint16_t, BitWidth::W128>{ using type = uint16x8_t; using tag_type = vector_128_tag; };
template <> struct neon_bitvector<int16_t, BitWidth::W128>{ using type = int16x8_t; using tag_type = vector_128_tag; };
template <> struct neon_bitvector<uint32_t, BitWidth::W64>{ using type = uint32x2_t; using tag_type = vector_64_tag; };
template <> struct neon_bitvector<int32_t, BitWidth::W64>{ using type = int32x2_t; using tag_type = vector_64_tag; };
template <> struct neon_bitvector<uint32_t, BitWidth::W128>{ using type = uint32x4_t; using tag_type = vector_128_tag; };
template <> struct neon_bitvector<int32_t, BitWidth::W128>{ using type = int32x4_t; using tag_type = vector_128_tag; };
template <> struct neon_bitvector<uint64_t, BitWidth::W64>{ using type = uint64x1_t; using tag_type = vector_64_tag; };
template <> struct neon_bitvector<int64_t, BitWidth::W64>{ using type = int64x1_t; using tag_type = vector_64_tag; };
template <> struct neon_bitvector<uint64_t, BitWidth::W128>{ using type = uint64x2_t; using tag_type = vector_128_tag; };
template <> struct neon_bitvector<int64_t, BitWidth::W128>{ using type = int64x2_t; using tag_type = vector_128_tag; };
template <> struct neon_bitvector<float_t, BitWidth::W64>{ using type = float32x2_t; using tag_type = vector_64_tag; };
template <> struct neon_bitvector<float_t, BitWidth::W128>{ using type = float32x4_t; using tag_type = vector_128_tag; };
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <> struct neon_bitvector<float16_t, BitWidth::W64>{ using type = float16x4_t; using tag_type = vector_64_tag; };
template <> struct neon_bitvector<float16_t, BitWidth::W128>{ using type = float16x8_t; using tag_type = vector_128_tag; };
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC


#if defined(__ARM_FEATURE_SVE)
/** Create the appropriate SVE vector given its type */
template <typename T> struct sve_vector;

template <> struct sve_vector<uint8_t>{ using scalar_type = uint8_t; using type = svuint8_t; };
template <> struct sve_vector<int8_t>{ using scalar_type = int8_t; using type = svint8_t; };
#endif /* defined(__ARM_FEATURE_SVE) */

#endif /* DOXYGEN_SKIP_THIS */

/**  Helper type template to get the type of a neon vector */
template <typename T, BitWidth BW> using neon_bitvector_t = typename neon_bitvector<T, BW>::type;
/**  Helper type template to get the tag type of a neon vector */
template <typename T, BitWidth BW> using neon_bitvector_tag_t = typename neon_bitvector<T, BW>::tag_type;

/** Promote a type */
template <typename T> struct promote { };
template <> struct promote<uint8_t> { using type = uint16_t; };
template <> struct promote<int8_t> { using type = int16_t; };
template <> struct promote<uint16_t> { using type = uint32_t; };
template <> struct promote<int16_t> { using type = int32_t; };
template <> struct promote<uint32_t> { using type = uint64_t; };
template <> struct promote<int32_t> { using type = int64_t; };
template <> struct promote<float> { using type = float; };
template <> struct promote<half> { using type = half; };

/** Get promoted type */
template <typename T>
using promote_t = typename promote<T>::type;

// clang-format on
// *INDENT-ON*
} // namespace traits
} // namespace wrapper
} // namespace arm_compute
#endif /* ARM_COMPUTE_WRAPPER_TRAITS_H */
