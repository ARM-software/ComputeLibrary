/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef SRC_CORE_NEON_KERNELS_CROP_CROP_HELPER_H
#define SRC_CORE_NEON_KERNELS_CROP_CROP_HELPER_H

#include "src/core/NEON/wrapper/wrapper.h"

namespace arm_compute
{
namespace cpu
{
template <typename T>
inline float32x4_t load_as_f32(T *ptr)
{
    ARM_COMPUTE_UNUSED(ptr);
    ARM_COMPUTE_ERROR("Type not supported.");
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)
template <>
inline float32x4_t load_as_f32(float16_t *ptr)
{
    return vcvt_f32_f16(wrapper::vload(ptr));
}
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS) */

template <>
inline float32x4_t load_as_f32(float *ptr)
{
    return wrapper::vloadq(ptr);
}

template <>
inline float32x4_t load_as_f32(int32_t *ptr)
{
    return vcvtq_f32_s32(wrapper::vloadq(ptr));
}

template <>
inline float32x4_t load_as_f32(uint32_t *ptr)
{
    return vcvtq_f32_u32(wrapper::vloadq(ptr));
}

template <>
inline float32x4_t load_as_f32(int16_t *ptr)
{
    return vcvtq_f32_s32(vmovl_s16(wrapper::vload(ptr)));
}

template <>
inline float32x4_t load_as_f32(uint16_t *ptr)
{
    return vcvtq_f32_u32(vmovl_u16(wrapper::vload(ptr)));
}

template <>
inline float32x4_t load_as_f32(uint8_t *ptr)
{
    return vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(wrapper::vload(ptr)))));
}
}
} // namespace arm_compute

#endif //SRC_CORE_NEON_KERNELS_CROP_CROP_HELPER_H