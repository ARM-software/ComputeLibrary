/*
 * Copyright (c) 2019-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_QUANTIZATION_INFO_H
#define ARM_COMPUTE_QUANTIZATION_INFO_H

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Rounding.h"
#include "support/ToolchainSupport.h"
#include "utils/misc/Utility.h"

#include <cstddef>
#include <type_traits>
#include <vector>

namespace arm_compute
{
using qasymm8_signed_t = int8_t;   /**< 8 bit signed quantized asymmetric scalar value */
using qasymm8_t        = uint8_t;  /**< 8 bit quantized asymmetric scalar value */
using qsymm16_t        = int16_t;  /**< 16 bit quantized symmetric scalar value */
using qasymm16_t       = uint16_t; /**< 16 bit quantized asymmetric scalar value */

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
class QuantizationInfo
{
public:
    /** Default constructor */
    QuantizationInfo() noexcept
        : _scale(),
          _offset()
    {
    }
    /** Construct quantization info.
     *
     * @note Used for symmetric quantization
     *
     * @param[in] scale Scale.
     */
    QuantizationInfo(float scale)
        : _scale(1, scale), _offset()
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
        : _scale(1, scale), _offset(1, offset)
    {
    }
    /** Construct quantization info.
     *
     * @note Used for symmetric per channel quantization
     *
     * @param[in] scale Scale.
     */
    QuantizationInfo(std::vector<float> scale)
        : _scale(scale), _offset()
    {
    }
    /** Construct quantization info.
     *
     * @note Used for asymmetric per channel quantization
     *
     * @param[in] scale  Scale.
     * @param[in] offset Offset.
     */
    QuantizationInfo(std::vector<float> scale, std::vector<int32_t> offset)
        : _scale(scale), _offset(offset)
    {
    }
    /** Scale vector accessor
     *
     * @return A reference to quantization scale metadata
     */
    const std::vector<float> &scale() const
    {
        return _scale;
    }
    /** Offset vector accessor
     *
     * @return A reference to quantization offset metadata
     */
    const std::vector<int32_t> &offset() const
    {
        return _offset;
    }
    /** Indicates whether this QuantizationInfo has valid settings or not
     *
     * @return True if the this has invalid settings.
     */
    bool empty() const
    {
        return _scale.empty() && _offset.empty();
    }
    /** Return per layer quantization info
     *
     * @return Uniform quantization information in case of empty information zero is returned in the respective fields
     */
    UniformQuantizationInfo uniform() const
    {
        UniformQuantizationInfo uqinfo;
        uqinfo.scale  = _scale.empty() ? 0 : _scale[0];
        uqinfo.offset = _offset.empty() ? 0 : _offset[0];

        return uqinfo;
    }

private:
    std::vector<float>   _scale;  /**< Vector containing scaling factors */
    std::vector<int32_t> _offset; /**< Vector containing zero offsets */
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
    return (lhs.scale() == rhs.scale()) && (lhs.offset() == rhs.offset());
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
template <typename QUANTIZED_TYPE = uint8_t>
struct Qasymm8QuantizationHelper
{
    static_assert(std::is_same<QUANTIZED_TYPE, uint8_t>::value
                  || std::is_same<QUANTIZED_TYPE, int8_t>::value,
                  "quantized type should be either uint8_t or int8_t.");

    /** Quantize a value given a 8-bit asymmetric quantization scheme
     *
     * @param[in] value Value to quantize
     * @param[in] qinfo Quantization information to use for quantizing
     *
     * @return Quantized value
     */
    static inline QUANTIZED_TYPE quantize(float value, const UniformQuantizationInfo &qinfo)
    {
        ARM_COMPUTE_ERROR_ON(qinfo.scale == 0);
        const int quantized = support::cpp11::lround(value / qinfo.scale) + qinfo.offset;
        return static_cast<QUANTIZED_TYPE>(arm_compute::utility::clamp<decltype(quantized), QUANTIZED_TYPE>(quantized));
    }

    /** Quantize a value given a 8-bit asymmetric quantization scheme using a specific rounding policy
     *
     * @param[in] value           Value to quantize
     * @param[in] qinfo           Quantization information to use for quantizing
     * @param[in] rounding_policy Rounding policy to use
     *
     * @return Quantized value
     */
    static inline QUANTIZED_TYPE quantize(float value, const UniformQuantizationInfo &qinfo, RoundingPolicy rounding_policy)
    {
        if(rounding_policy == RoundingPolicy::TO_NEAREST_UP)
        {
            return quantize(value, qinfo);
        }

        ARM_COMPUTE_ERROR_ON(qinfo.scale == 0);
        const int quantized = arm_compute::round(value / qinfo.scale, rounding_policy) + qinfo.offset;
        return static_cast<QUANTIZED_TYPE>(arm_compute::utility::clamp<decltype(quantized), QUANTIZED_TYPE>(quantized));
    }

    /** Quantize a value given a 8-bit asymmetric quantization scheme
     *
     * @param[in] value           Value to quantize
     * @param[in] qinfo           Quantization information to use for quantizing
     * @param[in] rounding_policy (Optional) Rounding policy to use. Default: nearest up
     *
     * @return Quantized value
     */
    static inline QUANTIZED_TYPE quantize(float value, const QuantizationInfo &qinfo, RoundingPolicy rounding_policy = RoundingPolicy::TO_NEAREST_UP)
    {
        const UniformQuantizationInfo uqinfo = qinfo.uniform();
        ARM_COMPUTE_ERROR_ON(uqinfo.scale == 0);
        const int quantized = arm_compute::round(value / uqinfo.scale, rounding_policy) + uqinfo.offset;
        return static_cast<QUANTIZED_TYPE>(arm_compute::utility::clamp<decltype(quantized), QUANTIZED_TYPE>(quantized));
    }

    /** Dequantize a value given a 8-bit asymmetric quantization scheme
     *
     * @param[in] value Value to dequantize
     * @param[in] qinfo Quantization information to use for dequantizing
     *
     * @return Dequantized value
     */
    static inline float dequantize(QUANTIZED_TYPE value, const UniformQuantizationInfo &qinfo)
    {
        return (static_cast<int>(value) - qinfo.offset) * qinfo.scale;
    }

    /** Dequantize a value given a 8-bit asymmetric quantization scheme
     *
     * @param[in] value Value to dequantize
     * @param[in] qinfo Quantization information to use for dequantizing
     *
     * @return Dequantized value
     */
    static inline float dequantize(QUANTIZED_TYPE value, const QuantizationInfo &qinfo)
    {
        const UniformQuantizationInfo uqinfo = qinfo.uniform();
        return (static_cast<int>(value) - uqinfo.offset) * uqinfo.scale;
    }
};

/** Quantize a value given an unsigned 8-bit asymmetric quantization scheme
 *
 * @param[in] value           Value to quantize
 * @param[in] qinfo           Quantization information to use for quantizing
 * @param[in] rounding_policy (Optional) Rounding policy to use. Default: nearest up
 *
 * @return Quantized value
 */
template <typename INFO_TYPE>
inline uint8_t quantize_qasymm8(float value, const INFO_TYPE &qinfo, RoundingPolicy rounding_policy = RoundingPolicy::TO_NEAREST_UP)
{
    return Qasymm8QuantizationHelper<uint8_t>::quantize(value, qinfo, rounding_policy);
}

/** Quantize a value given a signed 8-bit asymmetric quantization scheme
 *
 * @param[in] value           Value to quantize
 * @param[in] qinfo           Quantization information to use for quantizing
 * @param[in] rounding_policy (Optional) Rounding policy to use. Default: nearest up
 *
 * @return Quantized value
 */
template <typename INFO_TYPE>
inline int8_t quantize_qasymm8_signed(float value, const INFO_TYPE &qinfo, RoundingPolicy rounding_policy = RoundingPolicy::TO_NEAREST_UP)
{
    return Qasymm8QuantizationHelper<int8_t>::quantize(value, qinfo, rounding_policy);
}

/** Quantize a value given a 8-bit symmetric quantization scheme
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

/** Quantize a value given a 8-bit symmetric per channel quantization scheme
 *
 * @param[in] value      Value to quantize
 * @param[in] qinfo      Quantization information to use for quantizing
 * @param[in] channel_id channel index into the scale vector of quantization info
 *
 * @return Quantized value
 */
inline int8_t quantize_qsymm8_per_channel(float value, const QuantizationInfo &qinfo, size_t channel_id = 0)
{
    int quantized = arm_compute::round(value / qinfo.scale()[channel_id], RoundingPolicy::TO_NEAREST_UP);
    quantized     = std::max(-128, std::min(quantized, 127));
    return quantized;
}

/** Dequantize a value given an unsigned 8-bit asymmetric quantization scheme
 *
 * @param[in] value Value to dequantize
 * @param[in] qinfo Quantization information to use for dequantizing
 *
 * @return Dequantized value
 */
template <typename INFO_TYPE>
inline float dequantize_qasymm8(uint8_t value, const INFO_TYPE &qinfo)
{
    return Qasymm8QuantizationHelper<uint8_t>::dequantize(value, qinfo);
}

/** Dequantize a value given a signed 8-bit asymmetric quantization scheme
 *
 * @param[in] value Value to dequantize
 * @param[in] qinfo Quantization information to use for dequantizing
 *
 * @return Dequantized value
 */
template <typename INFO_TYPE>
inline float dequantize_qasymm8_signed(int8_t value, const INFO_TYPE &qinfo)
{
    return Qasymm8QuantizationHelper<int8_t>::dequantize(value, qinfo);
}

/** Dequantize a value given an 8-bit asymmetric quantization scheme
 *
 * @param[in] value  Value to dequantize
 * @param[in] scale  Scale to use for dequantization
 * @param[in] offset Zero-offset to use for dequantization
 *
 * @return Dequantized value
 */
inline float dequantize(uint8_t value, float scale, int32_t offset)
{
    return (static_cast<int>(value) - offset) * scale;
}

/** Dequantize a value given a 8-bit symmetric quantization scheme
 *
 * @param[in] value Value to dequantize
 * @param[in] qinfo Quantization information to use for dequantizing
 *
 * @return Dequantized value
 */
inline float dequantize_qsymm8(int8_t value, const UniformQuantizationInfo &qinfo)
{
    return value * qinfo.scale;
}

inline qasymm8_t qasymm8_hard_swish(qasymm8_t                      in,
                                    const UniformQuantizationInfo &qi_in,
                                    const UniformQuantizationInfo &qi_out)
{
    float tmp_f         = dequantize_qasymm8(in, qi_in);
    tmp_f               = tmp_f * ((std::min(std::max((tmp_f + 3), 0.0f), 6.0f)) * 0.166666667f);
    const qasymm8_t tmp = quantize_qasymm8(tmp_f, qi_out);
    return tmp;
}

inline qasymm8_signed_t qasymm8_signed_hard_swish(qasymm8_signed_t               in,
                                                  const UniformQuantizationInfo &qi_in,
                                                  const UniformQuantizationInfo &qi_out)
{
    float tmp_f         = dequantize_qasymm8_signed(in, qi_in);
    tmp_f               = tmp_f * ((std::min(std::max((tmp_f + 3), 0.0f), 6.0f)) * 0.166666667f);
    const qasymm8_t tmp = quantize_qasymm8_signed(tmp_f, qi_out);
    return tmp;
}

inline qasymm8_t qasymm8_leaky_relu(qasymm8_t                      in,
                                    const UniformQuantizationInfo &qi_in,
                                    const UniformQuantizationInfo &qi_out,
                                    float                          alpha)
{
    float tmp_f         = dequantize_qasymm8(in, qi_in);
    tmp_f               = tmp_f > 0 ? tmp_f : tmp_f * alpha;
    const qasymm8_t tmp = quantize_qasymm8(tmp_f, qi_out);
    return tmp;
}

inline qasymm8_t qasymm8_logistic(qasymm8_t                      in,
                                  const UniformQuantizationInfo &qi_in,
                                  const UniformQuantizationInfo &qi_out)
{
    float tmp_f         = dequantize_qasymm8(in, qi_in);
    tmp_f               = 1.f / (1.f + std::exp(-tmp_f));
    const qasymm8_t tmp = quantize_qasymm8(tmp_f, qi_out);
    return tmp;
}

inline qasymm8_signed_t qasymm8_signed_logistic(qasymm8_signed_t               in,
                                                const UniformQuantizationInfo &qi_in,
                                                const UniformQuantizationInfo &qi_out)
{
    float tmp_f                = dequantize_qasymm8_signed(in, qi_in);
    tmp_f                      = 1.f / (1.f + std::exp(-tmp_f));
    const qasymm8_signed_t tmp = quantize_qasymm8_signed(tmp_f, qi_out);
    return tmp;
}

/** Dequantize a value given a 8-bit symmetric quantization scheme
 *
 * @param[in] value Value to dequantize
 * @param[in] scale Scale to use for dequantization
 *
 * @return Dequantized value
 */
inline float dequantize(int8_t value, float scale)
{
    return value * scale;
}

/** Dequantize a value given a 16-bit symmetric quantization scheme
 *
 * @param[in] value Value to dequantize
 * @param[in] scale Scale to use for dequantization
 *
 * @return Dequantized value
 */
inline float dequantize(int16_t value, float scale)
{
    return value * scale;
}

/** Dequantize a value given a 16-bit asymmetric quantization scheme
 *
 * @param[in] value  Value to dequantize
 * @param[in] scale  Scale to use for dequantization
 * @param[in] offset Zero-offset to use for dequantization
 *
 * @return Dequantized value
 */
inline float dequantize(uint16_t value, float scale, int32_t offset)
{
    return (static_cast<int>(value) - offset) * scale;
}

/** Quantize a value given a 16-bit symmetric quantization scheme
 *
 * @param[in] value           Value to quantize
 * @param[in] qinfo           Quantization information to use for quantizing
 * @param[in] rounding_policy (Optional) Rounding policy to use. Default: nearest up
 *
 * @return Quantized value
 */
inline int16_t quantize_qsymm16(float value, const UniformQuantizationInfo &qinfo, RoundingPolicy rounding_policy = RoundingPolicy::TO_NEAREST_UP)
{
    int quantized = arm_compute::round(value / qinfo.scale, rounding_policy);
    quantized     = arm_compute::utility::clamp<int, int16_t>(quantized);
    return quantized;
}

/** Dequantize a value given a 16-bit symmetric quantization scheme
 *
 * @param[in] value Value to dequantize
 * @param[in] qinfo Quantization information to use for dequantizing
 *
 * @return Dequantized value
 */
inline float dequantize_qsymm16(int16_t value, const UniformQuantizationInfo &qinfo)
{
    return value * qinfo.scale;
}

/** Quantize a value given a 16-bit symmetric quantization scheme
 *
 * @param[in] value Value to quantize
 * @param[in] qinfo Quantization information to use for quantizing
 *
 * @return Quantized value
 */
inline int16_t quantize_qsymm16(float value, const QuantizationInfo &qinfo)
{
    return quantize_qsymm16(value, qinfo.uniform());
}

/** Dequantize a value given a 16-bit symmetric quantization scheme
 *
 * @param[in] value Value to dequantize
 * @param[in] qinfo Quantization information to use for dequantizing
 *
 * @return Dequantized value
 */
inline float dequantize_qsymm16(int16_t value, const QuantizationInfo &qinfo)
{
    return dequantize_qsymm16(value, qinfo.uniform());
}

/** Quantize a value given a 16-bit asymmetric quantization scheme
 *
 * @param[in] value           Value to quantize
 * @param[in] qinfo           Quantization information to use for quantizing
 * @param[in] rounding_policy (Optional) Rounding policy to use. Default: nearest up
 *
 * @return Quantized value
 */
inline uint16_t quantize_qasymm16(float value, const UniformQuantizationInfo &qinfo, RoundingPolicy rounding_policy = RoundingPolicy::TO_NEAREST_UP)
{
    int quantized = arm_compute::round(value / qinfo.scale, rounding_policy) + qinfo.offset;
    quantized     = arm_compute::utility::clamp<int, uint16_t>(quantized);
    return quantized;
}

/** Dequantize a value given a 16-bit asymmetric quantization scheme
 *
 * @param[in] value Value to dequantize
 * @param[in] qinfo Quantization information to use for dequantizing
 *
 * @return Dequantized value
 */
inline float dequantize_qasymm16(uint16_t value, const UniformQuantizationInfo &qinfo)
{
    return (static_cast<int>(value) - qinfo.offset) * qinfo.scale;
}

/** Quantize a value given a 16-bit asymmetric quantization scheme
 *
 * @param[in] value Value to quantize
 * @param[in] qinfo Quantization information to use for quantizing
 *
 * @return Quantized value
 */
inline uint16_t quantize_qasymm16(float value, const QuantizationInfo &qinfo)
{
    return quantize_qasymm16(value, qinfo.uniform());
}

/** Dequantize a value given a 16-bit asymmetric quantization scheme
 *
 * @param[in] value Value to dequantize
 * @param[in] qinfo Quantization information to use for dequantizing
 *
 * @return Dequantized value
 */
inline float dequantize_qasymm16(uint16_t value, const QuantizationInfo &qinfo)
{
    return dequantize_qasymm16(value, qinfo.uniform());
}

/*
 * In case of requantization of a quantized input tensor to an output tensor with another quantization
 * instead of applying dequantization and then a quantization functions, we just compute new scale and
 * offset.
 *
 * Assuming:
 *   - q_i as input quantized value
 *   - q_o as output quantized value
 *   - z_i as input quantization offset value
 *   - z_o as output quantization offset value
 *   - s_i as input quantization scale value
 *   - s_o as output quantization scale value
 *   - z_n as new quantization offset value
 *   - s_n as new quantization scale value
 *
 * q_o = ( q_i - z_i ) * s_i / s_o + z_o
 *
 * We can rewrite the formula as:
 *
 * q_o = ( q_i * s_i / s_o ) - z_i * s_i / s_o + z_o
 *
 * q_o = q_i / s_n + z_n
 *
 * Where:
 *
 * s_n = s_o / s_i
 *
 * z_n = - z_i * s_i / s_o + z_o
 *
 */
inline UniformQuantizationInfo compute_requantization_scale_offset(const UniformQuantizationInfo &uqinfo_in, const UniformQuantizationInfo &uqinfo_out)
{
    float   scale_to_apply  = uqinfo_out.scale;
    int32_t offset_to_apply = uqinfo_out.offset;

    scale_to_apply /= uqinfo_in.scale;
    // In order to minimize flooring we convert the offset to a float,
    // then compute the new offset in the float domain,
    // finally we convert it back as int32_t
    offset_to_apply -= static_cast<int32_t>(static_cast<float>(uqinfo_in.offset) * uqinfo_in.scale / uqinfo_out.scale);
    return UniformQuantizationInfo(scale_to_apply, offset_to_apply);
}

} // namespace arm_compute
#endif /* ARM_COMPUTE_QUANTIZATION_INFO_H */
