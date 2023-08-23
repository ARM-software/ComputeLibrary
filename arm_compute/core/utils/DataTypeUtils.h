/*
 * Copyright (c) 2016-2023 Arm Limited.
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
#ifndef ARM_COMPUTE_CORE_UTILS_DATATYPEUTILS_H
#define ARM_COMPUTE_CORE_UTILS_DATATYPEUTILS_H

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
/** The size in bytes of the data type
 *
 * @param[in] data_type Input data type
 *
 * @return The size in bytes of the data type
 */
inline size_t data_size_from_type(DataType data_type)
{
    switch(data_type)
    {
        case DataType::U8:
        case DataType::S8:
        case DataType::QSYMM8:
        case DataType::QASYMM8:
        case DataType::QASYMM8_SIGNED:
        case DataType::QSYMM8_PER_CHANNEL:
            return 1;
        case DataType::U16:
        case DataType::S16:
        case DataType::QSYMM16:
        case DataType::QASYMM16:
        case DataType::BFLOAT16:
        case DataType::F16:
            return 2;
        case DataType::F32:
        case DataType::U32:
        case DataType::S32:
            return 4;
        case DataType::F64:
        case DataType::U64:
        case DataType::S64:
            return 8;
        case DataType::SIZET:
            return sizeof(size_t);
        default:
            ARM_COMPUTE_ERROR("Invalid data type");
            return 0;
    }
}

/** The size in bytes of the data type
 *
 * @param[in] dt Input data type
 *
 * @return The size in bytes of the data type
 */
inline size_t element_size_from_data_type(DataType dt)
{
    switch(dt)
    {
        case DataType::S8:
        case DataType::U8:
        case DataType::QSYMM8:
        case DataType::QASYMM8:
        case DataType::QASYMM8_SIGNED:
        case DataType::QSYMM8_PER_CHANNEL:
            return 1;
        case DataType::U16:
        case DataType::S16:
        case DataType::QSYMM16:
        case DataType::QASYMM16:
        case DataType::BFLOAT16:
        case DataType::F16:
            return 2;
        case DataType::U32:
        case DataType::S32:
        case DataType::F32:
            return 4;
        case DataType::U64:
        case DataType::S64:
            return 8;
        default:
            ARM_COMPUTE_ERROR("Undefined element size for given data type");
            return 0;
    }
}

/** Return the data type used by a given single-planar pixel format
 *
 * @param[in] format Input format
 *
 * @return The size in bytes of the pixel format
 */
inline DataType data_type_from_format(Format format)
{
    switch(format)
    {
        case Format::U8:
        case Format::UV88:
        case Format::RGB888:
        case Format::RGBA8888:
        case Format::YUYV422:
        case Format::UYVY422:
            return DataType::U8;
        case Format::U16:
            return DataType::U16;
        case Format::S16:
            return DataType::S16;
        case Format::U32:
            return DataType::U32;
        case Format::S32:
            return DataType::S32;
        case Format::BFLOAT16:
            return DataType::BFLOAT16;
        case Format::F16:
            return DataType::F16;
        case Format::F32:
            return DataType::F32;
        //Doesn't make sense for planar formats:
        case Format::NV12:
        case Format::NV21:
        case Format::IYUV:
        case Format::YUV444:
        default:
            ARM_COMPUTE_ERROR("Not supported data_type for given format");
            return DataType::UNKNOWN;
    }
}

/** Return the promoted data type of a given data type.
 *
 * @note If promoted data type is not supported an error will be thrown
 *
 * @param[in] dt Data type to get the promoted type of.
 *
 * @return Promoted data type
 */
inline DataType get_promoted_data_type(DataType dt)
{
    switch(dt)
    {
        case DataType::U8:
            return DataType::U16;
        case DataType::S8:
            return DataType::S16;
        case DataType::U16:
            return DataType::U32;
        case DataType::S16:
            return DataType::S32;
        case DataType::QSYMM8:
        case DataType::QASYMM8:
        case DataType::QASYMM8_SIGNED:
        case DataType::QSYMM8_PER_CHANNEL:
        case DataType::QSYMM16:
        case DataType::QASYMM16:
        case DataType::BFLOAT16:
        case DataType::F16:
        case DataType::U32:
        case DataType::S32:
        case DataType::F32:
            ARM_COMPUTE_ERROR("Unsupported data type promotions!");
        default:
            ARM_COMPUTE_ERROR("Undefined data type!");
    }
    return DataType::UNKNOWN;
}

/** Compute the mininum and maximum values a data type can take
 *
 * @param[in] dt Data type to get the min/max bounds of
 *
 * @return A tuple (min,max) with the minimum and maximum values respectively wrapped in PixelValue.
 */
inline std::tuple<PixelValue, PixelValue> get_min_max(DataType dt)
{
    PixelValue min{};
    PixelValue max{};
    switch(dt)
    {
        case DataType::U8:
        case DataType::QASYMM8:
        {
            min = PixelValue(static_cast<int32_t>(std::numeric_limits<uint8_t>::lowest()));
            max = PixelValue(static_cast<int32_t>(std::numeric_limits<uint8_t>::max()));
            break;
        }
        case DataType::S8:
        case DataType::QSYMM8:
        case DataType::QASYMM8_SIGNED:
        case DataType::QSYMM8_PER_CHANNEL:
        {
            min = PixelValue(static_cast<int32_t>(std::numeric_limits<int8_t>::lowest()));
            max = PixelValue(static_cast<int32_t>(std::numeric_limits<int8_t>::max()));
            break;
        }
        case DataType::U16:
        case DataType::QASYMM16:
        {
            min = PixelValue(static_cast<int32_t>(std::numeric_limits<uint16_t>::lowest()));
            max = PixelValue(static_cast<int32_t>(std::numeric_limits<uint16_t>::max()));
            break;
        }
        case DataType::S16:
        case DataType::QSYMM16:
        {
            min = PixelValue(static_cast<int32_t>(std::numeric_limits<int16_t>::lowest()));
            max = PixelValue(static_cast<int32_t>(std::numeric_limits<int16_t>::max()));
            break;
        }
        case DataType::U32:
        {
            min = PixelValue(std::numeric_limits<uint32_t>::lowest());
            max = PixelValue(std::numeric_limits<uint32_t>::max());
            break;
        }
        case DataType::S32:
        {
            min = PixelValue(std::numeric_limits<int32_t>::lowest());
            max = PixelValue(std::numeric_limits<int32_t>::max());
            break;
        }
        case DataType::BFLOAT16:
        {
            min = PixelValue(bfloat16::lowest());
            max = PixelValue(bfloat16::max());
            break;
        }
        case DataType::F16:
        {
            min = PixelValue(std::numeric_limits<half>::lowest());
            max = PixelValue(std::numeric_limits<half>::max());
            break;
        }
        case DataType::F32:
        {
            min = PixelValue(std::numeric_limits<float>::lowest());
            max = PixelValue(std::numeric_limits<float>::max());
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Undefined data type!");
    }
    return std::make_tuple(min, max);
}

/** Convert a data type identity into a string.
 *
 * @param[in] dt @ref DataType to be translated to string.
 *
 * @return The string describing the data type.
 */
const std::string &string_from_data_type(DataType dt);

/** Convert a string to DataType
 *
 * @param[in] name The name of the data type
 *
 * @return DataType
 */
DataType data_type_from_name(const std::string &name);

/** Input Stream operator for @ref DataType
 *
 * @param[in]  stream    Stream to parse
 * @param[out] data_type Output data type
 *
 * @return Updated stream
 */
inline ::std::istream &operator>>(::std::istream &stream, DataType &data_type)
{
    std::string value;
    stream >> value;
    data_type = data_type_from_name(value);
    return stream;
}

/** Check if a given data type is of floating point type
 *
 * @param[in] dt Input data type.
 *
 * @return True if data type is of floating point type, else false.
 */
inline bool is_data_type_float(DataType dt)
{
    switch(dt)
    {
        case DataType::F16:
        case DataType::F32:
            return true;
        default:
            return false;
    }
}

/** Check if a given data type is of quantized type
 *
 * @note Quantized is considered a super-set of fixed-point and asymmetric data types.
 *
 * @param[in] dt Input data type.
 *
 * @return True if data type is of quantized type, else false.
 */
inline bool is_data_type_quantized(DataType dt)
{
    switch(dt)
    {
        case DataType::QSYMM8:
        case DataType::QASYMM8:
        case DataType::QASYMM8_SIGNED:
        case DataType::QSYMM8_PER_CHANNEL:
        case DataType::QSYMM16:
        case DataType::QASYMM16:
            return true;
        default:
            return false;
    }
}

/** Check if a given data type is of asymmetric quantized type
 *
 * @param[in] dt Input data type.
 *
 * @return True if data type is of asymmetric quantized type, else false.
 */
inline bool is_data_type_quantized_asymmetric(DataType dt)
{
    switch(dt)
    {
        case DataType::QASYMM8:
        case DataType::QASYMM8_SIGNED:
        case DataType::QASYMM16:
            return true;
        default:
            return false;
    }
}

/** Check if a given data type is of asymmetric quantized signed type
 *
 * @param[in] dt Input data type.
 *
 * @return True if data type is of asymmetric quantized signed type, else false.
 */
inline bool is_data_type_quantized_asymmetric_signed(DataType dt)
{
    switch(dt)
    {
        case DataType::QASYMM8_SIGNED:
            return true;
        default:
            return false;
    }
}

/** Check if a given data type is of symmetric quantized type
 *
 * @param[in] dt Input data type.
 *
 * @return True if data type is of symmetric quantized type, else false.
 */
inline bool is_data_type_quantized_symmetric(DataType dt)
{
    switch(dt)
    {
        case DataType::QSYMM8:
        case DataType::QSYMM8_PER_CHANNEL:
        case DataType::QSYMM16:
            return true;
        default:
            return false;
    }
}

/** Check if a given data type is of per channel type
 *
 * @param[in] dt Input data type.
 *
 * @return True if data type is of per channel type, else false.
 */
inline bool is_data_type_quantized_per_channel(DataType dt)
{
    switch(dt)
    {
        case DataType::QSYMM8_PER_CHANNEL:
            return true;
        default:
            return false;
    }
}

/** Returns true if the value can be represented by the given data type
 *
 * @param[in] val   value to be checked
 * @param[in] dt    data type that is checked
 * @param[in] qinfo (Optional) quantization info if the data type is QASYMM8
 *
 * @return true if the data type can hold the value.
 */
template <typename T>
bool check_value_range(T val, DataType dt, QuantizationInfo qinfo = QuantizationInfo())
{
    switch(dt)
    {
        case DataType::U8:
        {
            const auto val_u8 = static_cast<uint8_t>(val);
            return ((val_u8 == val) && val >= std::numeric_limits<uint8_t>::lowest() && val <= std::numeric_limits<uint8_t>::max());
        }
        case DataType::QASYMM8:
        {
            double min = static_cast<double>(dequantize_qasymm8(0, qinfo));
            double max = static_cast<double>(dequantize_qasymm8(std::numeric_limits<uint8_t>::max(), qinfo));
            return ((double)val >= min && (double)val <= max);
        }
        case DataType::S8:
        {
            const auto val_s8 = static_cast<int8_t>(val);
            return ((val_s8 == val) && val >= std::numeric_limits<int8_t>::lowest() && val <= std::numeric_limits<int8_t>::max());
        }
        case DataType::U16:
        {
            const auto val_u16 = static_cast<uint16_t>(val);
            return ((val_u16 == val) && val >= std::numeric_limits<uint16_t>::lowest() && val <= std::numeric_limits<uint16_t>::max());
        }
        case DataType::S16:
        {
            const auto val_s16 = static_cast<int16_t>(val);
            return ((val_s16 == val) && val >= std::numeric_limits<int16_t>::lowest() && val <= std::numeric_limits<int16_t>::max());
        }
        case DataType::U32:
        {
            const auto val_d64 = static_cast<double>(val);
            const auto val_u32 = static_cast<uint32_t>(val);
            return ((val_u32 == val_d64) && val_d64 >= std::numeric_limits<uint32_t>::lowest() && val_d64 <= std::numeric_limits<uint32_t>::max());
        }
        case DataType::S32:
        {
            const auto val_d64 = static_cast<double>(val);
            const auto val_s32 = static_cast<int32_t>(val);
            return ((val_s32 == val_d64) && val_d64 >= std::numeric_limits<int32_t>::lowest() && val_d64 <= std::numeric_limits<int32_t>::max());
        }
        case DataType::BFLOAT16:
            return (val >= bfloat16::lowest() && val <= bfloat16::max());
        case DataType::F16:
            return (val >= std::numeric_limits<half>::lowest() && val <= std::numeric_limits<half>::max());
        case DataType::F32:
            return (val >= std::numeric_limits<float>::lowest() && val <= std::numeric_limits<float>::max());
        default:
            ARM_COMPUTE_ERROR("Data type not supported");
            return false;
    }
}

/** Returns the suffix string of CPU kernel implementation names based on the given data type
 *
 * @param[in] data_type The data type the CPU kernel implemetation uses
 *
 * @return the suffix string of CPU kernel implementations
 */
inline std::string cpu_impl_dt(const DataType &data_type)
{
    std::string ret = "";

    switch(data_type)
    {
        case DataType::F32:
            ret = "fp32";
            break;
        case DataType::F16:
            ret = "fp16";
            break;
        case DataType::U8:
            ret = "u8";
            break;
        case DataType::S16:
            ret = "s16";
            break;
        case DataType::S32:
            ret = "s32";
            break;
        case DataType::QASYMM8:
            ret = "qu8";
            break;
        case DataType::QASYMM8_SIGNED:
            ret = "qs8";
            break;
        case DataType::QSYMM16:
            ret = "qs16";
            break;
        case DataType::QSYMM8_PER_CHANNEL:
            ret = "qp8";
            break;
        case DataType::BFLOAT16:
            ret = "bf16";
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported.");
    }

    return ret;
}

}
#endif /*ARM_COMPUTE_CORE_UTILS_DATATYPEUTILS_H */
