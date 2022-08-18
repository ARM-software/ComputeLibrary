/*
 * Copyright (c) 2021-2022 Arm Limited.
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
#ifndef SRC_CORE_NEON_WRAPPER_SVTRAITS_H
#define SRC_CORE_NEON_WRAPPER_SVTRAITS_H
#if defined(ARM_COMPUTE_ENABLE_SVE)
#include "src/core/NEON/SVEMath.h"
#include <arm_sve.h>

namespace arm_compute
{
namespace wrapper
{
template <typename T>
struct sve_scalar;
template <typename T>
struct sve_vector;

#define DEFINE_TYPES(stype)      \
    template <>                  \
    struct sve_scalar<sv##stype> \
    {                            \
        using type = stype;      \
    };                           \
    template <>                  \
    struct sve_vector<stype>     \
    {                            \
        using type = sv##stype;  \
    };

DEFINE_TYPES(int8_t)
DEFINE_TYPES(uint8_t)
DEFINE_TYPES(int16_t)
DEFINE_TYPES(uint16_t)
DEFINE_TYPES(int32_t)
DEFINE_TYPES(uint32_t)
DEFINE_TYPES(int64_t)
DEFINE_TYPES(uint64_t)
DEFINE_TYPES(float16_t)
DEFINE_TYPES(float32_t)
DEFINE_TYPES(float64_t)

#if __ARM_FEATURE_SVE_BF16
DEFINE_TYPES(bfloat16_t)
#endif // #if __ARM_FEATURE_SVE_BF16

#undef DEFINE_TYPES

} // namespace wrapper
} // namespace arm_compute

#endif /* defined(ARM_COMPUTE_ENABLE_SVE) */
#endif /* #ifndef SRC_CORE_NEON_WRAPPER_SVTRAITS_H */
