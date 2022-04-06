/*
* Copyright (c) 2022 Arm Limited.
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
#ifndef SRC_CORE_HELPERS_POOLINGHELPERS_H
#define SRC_CORE_HELPERS_POOLINGHELPERS_H

#include "src/core/NEON/NEAsymm.h"

namespace arm_compute
{
namespace cpu
{
namespace
{

inline float calculate_avg_scale_pool3d(bool exclude_padding, const Coordinates &id, const int pool_size_x, const int pool_size_y, const int pool_size_z, const int upper_bound_w,
                                 const int upper_bound_h, const int upper_bound_d, const int pad_x, const int pad_y, const int pad_z, const int stride_x, const int stride_y, const int stride_z)
{
    // Based on NDHWC
    int start_x = id[1] * stride_x - pad_x;
    int start_y = id[2] * stride_y - pad_y;
    int start_z = id[3] * stride_z - pad_z;

    const int end_x = std::min(start_x + pool_size_x, upper_bound_w);
    const int end_y = std::min(start_y + pool_size_y, upper_bound_h);
    const int end_z = std::min(start_z + pool_size_z, upper_bound_d);
    if(exclude_padding)
    {
        start_x = std::max(0, start_x);
        start_y = std::max(0, start_y);
        start_z = std::max(0, start_z);
    }
    return 1.f / ((end_y - start_y) * (end_x - start_x) * (end_z - start_z));
}

inline float calculate_avg_scale_pool2d(bool exclude_padding, DataLayout data_layout, const Coordinates &id, const int pool_size_x, const int pool_size_y, const int upper_bound_w, const int upper_bound_h,
                                 const int pad_x, const int pad_y, const int stride_x, const int stride_y)
{
    const unsigned int idx_width  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int idx_height = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    int start_x = id[idx_width] * stride_x - pad_x;
    int start_y = id[idx_height] * stride_y - pad_y;

    const int end_x = std::min(start_x + pool_size_x, upper_bound_w);
    const int end_y = std::min(start_y + pool_size_y, upper_bound_h);
    if(exclude_padding)
    {
        start_x = std::max(0, start_x);
        start_y = std::max(0, start_y);
    }
    return 1.f / ((end_y - start_y) * (end_x - start_x));
}

template <typename T>
inline typename std::enable_if<std::is_same<T, int8_t>::value, int8_t>::type
quantize(float val, const UniformQuantizationInfo &info)
{
    return quantize_qasymm8_signed(val, info);
}

template <typename T>
inline typename std::enable_if<std::is_same<T, uint8_t>::value, uint8_t>::type
quantize(float val, const UniformQuantizationInfo &info)
{
    return quantize_qasymm8(val, info);
}

template <typename T>
inline T vcvtq_q32_f32(float32x4_t values);

template <>
inline uint32x4_t vcvtq_q32_f32(float32x4_t values)
{
    return vcvtq_u32_f32(values);
}

template <>
inline int32x4_t vcvtq_q32_f32(float32x4_t values)
{
    return vcvtq_s32_f32(values);
}

template <typename T>
inline float32x4_t vcvtq_f32_q32(T values);

template <>
inline float32x4_t vcvtq_f32_q32(uint32x4_t values)
{
    return vcvtq_f32_u32(values);
}

template <>
inline float32x4_t vcvtq_f32_q32(int32x4_t values)
{
    return vcvtq_f32_s32(values);
}

template <typename Tout>
inline Tout vrequantize_pooling_with_scale(const float32x4x4_t &acc, const float quant_rescale, const float scale_pooling, const int32_t new_offset);

template <>
inline uint8x16_t vrequantize_pooling_with_scale(const float32x4x4_t &acc, const float quant_rescale, const float scale_pooling, const int32_t new_offset)
{
    const float new_scale = quant_rescale / scale_pooling;
    return vquantize(acc, UniformQuantizationInfo(new_scale, new_offset));
}

template <>
inline int8x16_t vrequantize_pooling_with_scale(const float32x4x4_t &acc, const float quant_rescale, const float scale_pooling, const int32_t new_offset)
{
    const float new_scale = quant_rescale / scale_pooling;
    return vquantize_signed(acc, UniformQuantizationInfo(new_scale, new_offset));
}

template <typename Tin, typename Tout>
inline Tout vrequantize_pooling(Tin vec1, Tin vec2, const UniformQuantizationInfo &requant_qinfo);

template <>
inline uint8x16_t vrequantize_pooling(uint8x8_t vec1, uint8x8_t vec2, const UniformQuantizationInfo &requant_qinfo)
{
    const float32x4x4_t acc =
    {
        {
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8((vec1))))),
            vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8((vec1))))),
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8((vec2))))),
            vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8((vec2))))),
        }
    };
    return vquantize(acc, requant_qinfo);
}

template <>
inline int8x16_t vrequantize_pooling(int8x8_t vec1, int8x8_t vec2, const UniformQuantizationInfo &requant_qinfo)
{
    const float32x4x4_t acc =
    {
        {
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8((vec1))))),
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8((vec1))))),
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8((vec2))))),
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8((vec2))))),
        }
    };
    return vquantize_signed(acc, requant_qinfo);
}

template <typename T>
inline T vrequantize_pooling(T &vec, const UniformQuantizationInfo &requant_qinfo);

template <>
inline uint8x8_t vrequantize_pooling(uint8x8_t &vec, const UniformQuantizationInfo &requant_qinfo)
{
    const float32x4x2_t acc =
    {
        {
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8((vec))))),
            vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8((vec))))),
        }
    };
    return vquantize(acc, requant_qinfo);
}

template <>
inline int8x8_t vrequantize_pooling(int8x8_t &vec, const UniformQuantizationInfo &requant_qinfo)
{
    const float32x4x2_t acc =
    {
        {
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8((vec))))),
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8((vec))))),
        }
    };
    return vquantize_signed(acc, requant_qinfo);
}

} // namespace
} // namespace cpu
} // namespace arm_compute
#endif /* SRC_CORE_HELPERS_POOLINGHELPERS_H */

