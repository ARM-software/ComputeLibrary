/*
 * Copyright (c) 2020, 2022-2023 Arm Limited.
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
#ifndef ARM_COMPUTE_WRAPPER_CVT_H
#define ARM_COMPUTE_WRAPPER_CVT_H

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VCVT_TO_F32_IMPL(ptype, vtype, prefix, postfix1, postfix2)                   \
    template <typename T>                                                            \
    inline typename std::enable_if<std::is_same<T, float>::value, float32x4_t>::type \
    vcvt(const vtype &a)                                                             \
    {                                                                                \
        return prefix##_##postfix1##_##postfix2(a);                                  \
    }

VCVT_TO_F32_IMPL(float32x4_t, uint32x4_t, vcvtq, f32, u32)
VCVT_TO_F32_IMPL(float32x4_t, int32x4_t, vcvtq, f32, s32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VCVT_TO_F32_IMPL(float32x4_t, float16x4_t, vcvt, f32, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#undef VCVT_TO_F32_IMPL

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#define VCVT_TO_F16_IMPL(ptype, vtype, prefix, postfix1, postfix2)                       \
    template <typename T>                                                                \
    inline typename std::enable_if<std::is_same<T, float16_t>::value, float16x4_t>::type \
    vcvt(const vtype &a)                                                                 \
    {                                                                                    \
        return prefix##_##postfix1##_##postfix2(a);                                      \
    }

VCVT_TO_F16_IMPL(float16x4_t, float32x4_t, vcvt, f16, f32)
#undef VCVT_TO_F16_IMPL
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

template <typename T>
inline typename std::enable_if < std::is_same<T, uint8_t>::value || std::is_same<T, uint32_t>::value, uint32x4_t >::type
vcvt(const float32x4_t &a)
{
    return vcvtq_u32_f32(a);
}

template <typename T>
inline typename std::enable_if < std::is_same<T, int8_t>::value || std::is_same<T, int32_t>::value, int32x4_t >::type
vcvt(const float32x4_t &a)
{
    return vcvtq_s32_f32(a);
}

#ifdef __aarch64__
template <typename T>
inline typename std::enable_if<std::is_same<T, uint32_t>::value, uint32x4_t>::type
vcvta(const float32x4_t &a)
{
    return vcvtaq_u32_f32(a);
}

template <typename T>
inline typename std::enable_if<std::is_same<T, int32_t>::value, int32x4_t>::type
vcvta(const float32x4_t &a)
{
    return vcvtaq_s32_f32(a);
}
#endif //__aarch64__

#if defined(ARM_COMPUTE_ENABLE_BF16)
/** Convert 2x128-bit floating point vectors into 1x128-bit bfloat16 vector
 *
 * @param[in]     inptr  Pointer to the input memory to load values from
 * @param[in,out] outptr Pointer to the output memory to store values to
 */
inline void vcvt_bf16_f32(const float *inptr, uint16_t *outptr)
{
    __asm __volatile(
        "ldp    q0, q1, [%[inptr]]\n"
        ".inst  0xea16800\n"  // BFCVTN v0, v0
        ".inst  0x4ea16820\n" // BFCVTN2 v0, v1
        "str    q0, [%[outptr]]\n"
        : [inptr] "+r"(inptr)
        : [outptr] "r"(outptr)
        : "v0", "v1", "memory");
}
#endif /* defined(ARM_COMPUTE_ENABLE_BF16) */

} // namespace wrapper
} // namespace arm_compute
#endif /* ARM_COMPUTE_WRAPPER_CVT_H */
