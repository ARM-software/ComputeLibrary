/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_QUANTIZATION_INFO_H__
#define __ARM_COMPUTE_QUANTIZATION_INFO_H__

#include "arm_compute/core/Rounding.h"

#include <cstddef>
#include <vector>

namespace arm_compute
{
using qasymm8_t = uint8_t; /**< 8 bit quantized asymmetric scalar value */
using qsymm8_t  = int8_t;  /**< 8 bit quantized symmetric scalar value */

/** Quantization info when assuming per layer quantization */
struct UniformQuantizationInfo
{
    /** Default constructor */
    UniformQuantizationInfo()
        : scale(0.f), offset(0)
    {
    }
    /** Constructor
     *
     * @param[in] scale  Quantization scale
     * @param[in] offset Quantization offset
     */
    UniformQuantizationInfo(float scale, int32_t offset)
        : scale(scale), offset(offset)
    {
    }
    /** Checks if the scale and offset are both zero */
    bool empty() const
    {
        return (scale == 0) && (offset == 0);
    }

    float   scale;
    int32_t offset;
};

/** Quantization information */
struct QuantizationInfo
{
    /** Default constructor */
    QuantizationInfo() noexcept
        : scale(),
          offset()
    {
    }
    /** Construct quantization info.
     *
     * @note Used for symmetric quantization
     *
     * @param[in] scale Scale.
     */
    QuantizationInfo(float scale)
        : scale(1, scale), offset()
    {
    }
    /** Construct quantization info.
     *
     * @note Used for asymmetric quantization
     *
     * @param[in] scale  Scale.
     * @param[in] offset Offset.
     */
    QuantizationInfo(float scale, int offset)
        : scale(1, scale), offset(1, offset)
    {
    }
    /** Construct quantization info.
     *
     * @note Used for symmetric per channel quantization
     *
     * @param[in] scale Scale.
     */
    QuantizationInfo(std::vector<float> scale)
        : scale(scale), offset()
    {
    }
    /** Indicates whether this QuantizationInfo has valid settings or not
     *
     * @return True if the this has invalid settings.
     */
    bool empty() const
    {
        return scale.empty() && offset.empty();
    }
    /** Return per layer quantization info
     *
     * @return Uniform quantization information in case of empty information zero is returned in the respective fields
     */
    UniformQuantizationInfo uniform() const
    {
        UniformQuantizationInfo uqinfo;
        uqinfo.scale  = scale.empty() ? 0 : scale[0];
        uqinfo.offset = offset.empty() ? 0 : offset[0];

        return uqinfo;
    }

    std::vector<float>   scale;  /**< Vector containing scaling factors */
    std::vector<int32_t> offset; /**< Vector containing zero offsets */
};

/** Check whether two quantization info are equal.
 *
 * @param[in] lhs RHS quantization info.
 * @param[in] rhs LHS quantization info.
 *
 * @return True if the given quantization info is the same.
 */
inline bool operator==(const QuantizationInfo &lhs, const QuantizationInfo &rhs)
{
    return (lhs.scale == rhs.scale) && (lhs.offset == rhs.offset);
}

/** Check whether two quantization info are not equal.
 *
 * @param[in] lhs RHS quantization info.
 * @param[in] rhs LHS quantization info.
 *
 * @return True if the given quantization info is the same.
 */
inline bool operator!=(const QuantizationInfo &lhs, const QuantizationInfo &rhs)
{
    return !(operator==(lhs, rhs));
}

/** Check whether two quantization info are equal.
 *
 * @param[in] lhs RHS quantization info.
 * @param[in] rhs LHS quantization info.
 *
 * @return True if the given quantization info is the same.
 */
inline bool operator==(const UniformQuantizationInfo &lhs, const UniformQuantizationInfo &rhs)
{
    return (lhs.scale == rhs.scale) && (lhs.offset == rhs.offset);
}

/** Check whether two quantization info are not equal.
 *
 * @param[in] lhs RHS quantization info.
 * @param[in] rhs LHS quantization info.
 *
 * @return True if the given quantization info is the same.
 */
inline bool operator!=(const UniformQuantizationInfo &lhs, const UniformQuantizationInfo &rhs)
{
    return !(operator==(lhs, rhs));
}

/** Quantize a value given a asymmetric quantization scheme
 *
 * @param[in] value           Value to quantize
 * @param[in] qinfo           Quantization information to use for quantizing
 * @param[in] rounding_policy (Optional) Rounding policy to use. Default: nearest up
 *
 * @return Quantized value
 */
inline uint8_t quantize_qasymm8(float value, const UniformQuantizationInfo &qinfo, RoundingPolicy rounding_policy = RoundingPolicy::TO_NEAREST_UP)
{
    int quantized = arm_compute::round(value / qinfo.scale, rounding_policy) + qinfo.offset;
    quantized     = std::max(0, std::min(quantized, 255));
    return quantized;
}

/** Quantize a value given a asymmetric quantization scheme
 *
 * @param[in] value           Value to quantize
 * @param[in] qinfo           Quantization information to use for quantizing
 * @param[in] rounding_policy (Optional) Rounding policy to use. Default: nearest up
 *
 * @return Quantized value
 */
inline uint8_t quantize_qasymm8(float value, const QuantizationInfo &qinfo, RoundingPolicy rounding_policy = RoundingPolicy::TO_NEAREST_UP)
{
    UniformQuantizationInfo uqinfo    = qinfo.uniform();
    int                     quantized = arm_compute::round(value / uqinfo.scale, rounding_policy) + uqinfo.offset;
    quantized                         = std::max(0, std::min(quantized, 255));
    return quantized;
}

/** Quantize a value given a symmetric quantization scheme
 *
 * @param[in] value Value to quantize
 * @param[in] qinfo Quantization information to use for quantizing
 *
 * @return Quantized value
 */
inline int8_t quantize_qsymm8(float value, const QuantizationInfo &qinfo)
{
    int quantized = arm_compute::round(value / qinfo.uniform().scale, RoundingPolicy::TO_NEAREST_UP);
    quantized     = std::max(-128, std::min(quantized, 127));
    return quantized;
}

/** Dequantize a value given a asymmetric quantization scheme
 *
 * @param[in] value Value to dequantize
 * @param[in] qinfo Quantization information to use for dequantizing
 *
 * @return Dequantized value
 */
inline float dequantize_qasymm8(uint8_t value, const UniformQuantizationInfo &qinfo)
{
    return (static_cast<int>(value) - qinfo.offset) * qinfo.scale;
}

/** Dequantize a value given a asymmetric quantization scheme
 *
 * @param[in] value Value to dequantize
 * @param[in] qinfo Quantization information to use for dequantizing
 *
 * @return Dequantized value
 */
inline float dequantize_qasymm8(uint8_t value, const QuantizationInfo &qinfo)
{
    UniformQuantizationInfo uqinfo = qinfo.uniform();
    return (static_cast<int>(value) - uqinfo.offset) * uqinfo.scale;
}

/** Dequantize a value given a symmetric quantization scheme
 *
 * @param[in] value Value to dequantize
 * @param[in] qinfo Quantization information to use for dequantizing
 *
 * @return Dequantized value
 */
inline float dequantize_qsymm8(int8_t value, const QuantizationInfo &qinfo)
{
    return value * qinfo.uniform().scale;
}
} // namespace arm_compute
#endif /*__ARM_COMPUTE_QUANTIZATION_INFO_H__ */