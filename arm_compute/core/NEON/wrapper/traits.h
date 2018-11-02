/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_WRAPPER_TRAITS_H__
#define __ARM_COMPUTE_WRAPPER_TRAITS_H__

#include <arm_neon.h>

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

/** Create the appropriate NEON vector given its type and size */
template <typename T, int S> struct neon_vector;
// Specializations
#ifndef DOXYGEN_SKIP_THIS
template <> struct neon_vector<uint8_t, 8>{ using type = uint8x8_t; using tag_type = vector_64_tag; };
template <> struct neon_vector<int8_t, 8>{ using type = int8x8_t; using tag_type = vector_64_tag; };
template <> struct neon_vector<uint8_t, 16>{ using type = uint8x16_t; using tag_type = vector_128_tag; };
template <> struct neon_vector<int8_t, 16>{ using type = int8x16_t; using tag_type = vector_128_tag; };
template <> struct neon_vector<uint16_t, 4>{ using type = uint16x4_t; using tag_type = vector_64_tag; };
template <> struct neon_vector<int16_t, 4>{ using type = int16x4_t; using tag_type = vector_64_tag; };
template <> struct neon_vector<uint16_t, 8>{ using type = uint16x8_t; using tag_type = vector_128_tag; };
template <> struct neon_vector<int16_t, 8>{ using type = int16x8_t; using tag_type = vector_128_tag; };
template <> struct neon_vector<uint32_t, 2>{ using type = uint32x2_t; using tag_type = vector_64_tag; };
template <> struct neon_vector<int32_t, 2>{ using type = int32x2_t; using tag_type = vector_64_tag; };
template <> struct neon_vector<uint32_t, 4>{ using type = uint32x4_t; using tag_type = vector_128_tag; };
template <> struct neon_vector<int32_t, 4>{ using type = int32x4_t; using tag_type = vector_128_tag; };
template <> struct neon_vector<uint64_t, 1>{ using type = uint64x1_t; using tag_type = vector_64_tag; };
template <> struct neon_vector<int64_t, 1>{ using type = int64x1_t; using tag_type = vector_64_tag; };
template <> struct neon_vector<uint64_t, 2>{ using type = uint64x2_t; using tag_type = vector_128_tag; };
template <> struct neon_vector<int64_t, 2>{ using type = int64x2_t; using tag_type = vector_128_tag; };
template <> struct neon_vector<float_t, 2>{ using type = float32x2_t; using tag_type = vector_64_tag; };
template <> struct neon_vector<float_t, 4>{ using type = float32x4_t; using tag_type = vector_128_tag; };
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <> struct neon_vector<float16_t, 4>{ using type = float16x4_t; using tag_type = vector_64_tag; };
template <> struct neon_vector<float16_t, 8>{ using type = float16x8_t; using tag_type = vector_128_tag; };
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#endif /* DOXYGEN_SKIP_THIS */

/**  Helper type template to get the type of a neon vector */
template <typename T, int S> using neon_vector_t = typename neon_vector<T, S>::type;
/**  Helper type template to get the tag type of a neon vector */
template <typename T, int S> using neon_vector_tag_t = typename neon_vector<T, S>::tag_type;
// clang-format on
// *INDENT-ON*
} // namespace traits
} // namespace wrapper
} // namespace arm_compute
#endif /* __ARM_COMPUTE_WRAPPER_TRAITS_H__ */
