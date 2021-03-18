/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_PIXELVALUE_H
#define ARM_COMPUTE_PIXELVALUE_H

#include "arm_compute/core/Types.h"

#include <cstdint>

namespace arm_compute
{
/** Class describing the value of a pixel for any image format. */
class PixelValue
{
public:
    /** Default constructor: value initialized to 0 */
    PixelValue() noexcept
        : value
    {
        int64_t(0)
    }
    {
    }
    /** Initialize the union with a pixel value of chosen datatype
     *
     * @param[in] v        value.
     * @param[in] datatype DataType that @p v have to be stored
     * @param[in] qinfo    (Optional) QuantizationInfo to apply in case of quantized data types to @p v
     */
    PixelValue(double v, DataType datatype, QuantizationInfo qinfo = QuantizationInfo())
        : PixelValue()
    {
        switch(datatype)
        {
            case DataType::U8:
                value.u8 = static_cast<uint8_t>(v);
                break;
            case DataType::S8:
                value.s8 = static_cast<int8_t>(v);
                break;
            case DataType::QASYMM8:
                value.u8 = quantize_qasymm8(static_cast<float>(v), qinfo);
                break;
            case DataType::QASYMM8_SIGNED:
                value.s8 = quantize_qasymm8_signed(static_cast<float>(v), qinfo);
                break;
            case DataType::QSYMM8:
                value.s8 = quantize_qsymm8(static_cast<float>(v), qinfo);
                break;
            case DataType::U16:
                value.u16 = static_cast<uint16_t>(v);
                break;
            case DataType::S16:
                value.s16 = static_cast<int16_t>(v);
                break;
            case DataType::QASYMM16:
                value.u16 = quantize_qasymm16(static_cast<float>(v), qinfo);
                break;
            case DataType::QSYMM16:
                value.s16 = quantize_qsymm16(static_cast<float>(v), qinfo);
                break;
            case DataType::U32:
                value.u32 = static_cast<uint32_t>(v);
                break;
            case DataType::S32:
                value.s32 = static_cast<int32_t>(v);
                break;
            case DataType::U64:
                value.u64 = static_cast<uint64_t>(v);
                break;
            case DataType::S64:
                value.s64 = static_cast<int64_t>(v);
                break;
            case DataType::BFLOAT16:
                value.bf16 = static_cast<bfloat16>(v);
                break;
            case DataType::F16:
                value.f16 = static_cast<half>(v);
                break;
            case DataType::F32:
                value.f32 = static_cast<float>(v);
                break;
            case DataType::F64:
            default:
                value.f64 = v;
                break;
        }
    }
    /** Initialize the union with a S8 pixel value
     *
     * @param[in] v S8 value.
     */
    PixelValue(int8_t v)
        : PixelValue()
    {
        value.s8 = v;
    }
    /** Initialize the union with a U8 pixel value
     *
     * @param[in] v U8 value.
     */
    PixelValue(uint8_t v)
        : PixelValue()
    {
        value.u8 = v;
    }
    /** Initialize the union with a U16 pixel value
     *
     * @param[in] v U16 value.
     */
    PixelValue(uint16_t v)
        : PixelValue()
    {
        value.u16 = v;
    }
    /** Initialize the union with a S16 pixel value
     *
     * @param[in] v S16 value.
     */
    PixelValue(int16_t v)
        : PixelValue()
    {
        value.s16 = v;
    }
    /** Initialize the union with a U32 pixel value
     *
     * @param[in] v U32 value.
     */
    PixelValue(uint32_t v)
        : PixelValue()
    {
        value.u32 = v;
    }
    /** Initialize the union with a S32 pixel value
     *
     * @param[in] v S32 value.
     */
    PixelValue(int32_t v)
        : PixelValue()
    {
        value.s32 = v;
    }

    /** Initialize the union with a U64 pixel value
     *
     * @param[in] v U64 value.
     */
    PixelValue(uint64_t v)
        : PixelValue()
    {
        value.u64 = v;
    }
    /** Initialize the union with a S64 pixel value
     *
     * @param[in] v S64 value.
     */
    PixelValue(int64_t v)
        : PixelValue()
    {
        value.s64 = v;
    }
    /** Initialize the union with a BFLOAT16 pixel value
     *
     * @param[in] v F16 value.
     */
    PixelValue(bfloat16 v)
        : PixelValue()
    {
        value.bf16 = v;
    }
    /** Initialize the union with a F16 pixel value
     *
     * @param[in] v F16 value.
     */
    PixelValue(half v)
        : PixelValue()
    {
        value.f16 = v;
    }
    /** Initialize the union with a F32 pixel value
     *
     * @param[in] v F32 value.
     */
    PixelValue(float v)
        : PixelValue()
    {
        value.f32 = v;
    }
    /** Initialize the union with a F64 pixel value
     *
     * @param[in] v F64 value.
     */
    PixelValue(double v)
        : PixelValue()
    {
        value.f64 = v;
    }
    /** Union which describes the value of a pixel for any image format.
     * Use the field corresponding to the image format
     */
    union
        {
            uint64_t u64;     /**< Single channel U64 */
            int64_t  s64;     /**< Single channel S64 */
            uint8_t  rgb[3];  /**< 3 channels: RGB888 */
            uint8_t  yuv[3];  /**< 3 channels: Any YUV format */
            uint8_t  rgbx[4]; /**< 4 channels: RGBX8888 */
            double   f64;     /**< Single channel double */
            float    f32;     /**< Single channel float 32 */
            half     f16;     /**< Single channel F16 */
            bfloat16 bf16;    /**< Single channel brain floating-point number */
            uint8_t  u8;      /**< Single channel U8 */
            int8_t   s8;      /**< Single channel S8 */
            uint16_t u16;     /**< Single channel U16 */
            int16_t  s16;     /**< Single channel S16 */
            uint32_t u32;     /**< Single channel U32 */
            int32_t  s32;     /**< Single channel S32 */
        } value;
    /** Interpret the pixel value as a U8
     *
     * @param[out] v Returned value
     */
    void get(uint8_t &v) const
    {
        v = value.u8;
    }
    /** Interpret the pixel value as a S8
     *
     * @param[out] v Returned value
     */
    void get(int8_t &v) const
    {
        v = value.s8;
    }
    /** Interpret the pixel value as a U16
     *
     * @param[out] v Returned value
     */
    void get(uint16_t &v) const
    {
        v = value.u16;
    }
    /** Interpret the pixel value as a S16
     *
     * @param[out] v Returned value
     */
    void get(int16_t &v) const
    {
        v = value.s16;
    }
    /** Interpret the pixel value as a U32
     *
     * @param[out] v Returned value
     */
    void get(uint32_t &v) const
    {
        v = value.u32;
    }
    /** Interpret the pixel value as a S32
     *
     * @param[out] v Returned value
     */
    void get(int32_t &v) const
    {
        v = value.s32;
    }
    /** Interpret the pixel value as a U64
     *
     * @param[out] v Returned value
     */
    void get(uint64_t &v) const
    {
        v = value.u64;
    }
    /** Interpret the pixel value as a S64
     *
     * @param[out] v Returned value
     */
    void get(int64_t &v) const
    {
        v = value.s64;
    }
    /** Interpret the pixel value as a BFLOAT16
     *
     * @param[out] v Returned value
     */
    void get(bfloat16 &v) const
    {
        v = value.bf16;
    }
    /** Interpret the pixel value as a F16
     *
     * @param[out] v Returned value
     */
    void get(half &v) const
    {
        v = value.f16;
    }
    /** Interpret the pixel value as a F32
     *
     * @param[out] v Returned value
     */
    void get(float &v) const
    {
        v = value.f32;
    }
    /** Interpret the pixel value as a double
     *
     * @param[out] v Returned value
     */
    void get(double &v) const
    {
        v = value.f64;
    }
    /** Get the pixel value
     *
     * @return Pixel value
     */
    template <typename T>
    T get() const
    {
        T val;
        get(val);
        return val;
    }
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_PIXELVALUE_H */
