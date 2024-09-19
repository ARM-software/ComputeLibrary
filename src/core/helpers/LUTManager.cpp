/*
 * Copyright (c) 2024 Arm Limited.
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

#include "src/core/helpers/LUTManager.h"

#include "src/common/utils/Validate.h"
#include "support/Bfloat16.h"

namespace arm_compute
{
#ifdef __aarch64__
namespace
{

union Element
{
    uint16_t  i = 0;
    float16_t fp;
};

inline float16_t activation(float16_t x, const LUTInfo &info)
{
    float16_t out = 0.f;
    switch (info.act)
    {
        case ActivationLayerInfo::ActivationFunction::LOGISTIC:
            out = 1.f / (1.f + std::exp(-x));
            break;
        case ActivationLayerInfo::ActivationFunction::TANH:
        {
            out = static_cast<float16_t>(info.alpha * std::tanh(info.beta * x));
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Unsupported Activation for 16-bit LUT table");
            break;
    }
    return out;
}

inline float exponential(float fp, const LUTInfo &info)
{
    return std::exp(fp * info.beta);
}

// Read bf16 value as u16, convert to fp32.
// Calculate exp in fp32, return as bf16
inline uint16_t exponential_bf16(uint16_t x, const LUTInfo &info)
{
    float fp = bf16_to_float(x);
    fp       = exponential(fp, info);
    return float_to_bf16(fp);
}

void init_lut(LookupTable256 &lut, const LUTInfo &info)
{
    // assert lut is valid config.
    ARM_COMPUTE_ASSERT((info.type == LUTType::Exponential && info.dt == DataType::QASYMM8) ||
                       (info.type == LUTType::Exponential && info.dt == DataType::QASYMM8_SIGNED));

    for (int i = 0; i < 256; ++i)
    {
        const float deq = info.dt == DataType::QASYMM8 ? dequantize_qasymm8(i, info.qinfo)
                                                       : dequantize_qasymm8_signed(i - 128, info.qinfo);
        lut[i]          = exponential(deq, info);
    }
}

void init_lut(LookupTable65536 &lut, const LUTInfo &info)
{
    // assert lut is valid config.
    ARM_COMPUTE_ASSERT((info.type == LUTType::Activation && info.dt == DataType::F16) ||
                       (info.type == LUTType::Exponential && info.dt == DataType::BFLOAT16));

    Element item = {0}; // Fill lut by iterating over all 16 bit values using the union.
    Element bf16 = {0}; // Temporary object used to store bf16 values as fp16 in lut
    while (true)
    {
        switch (info.type)
        {
            case LUTType::Activation:
            {
                lut[item.i] = activation(item.fp, info);
                break;
            }
            case LUTType::Exponential:
            {
                bf16.i      = exponential_bf16(item.i, info);
                lut[item.i] = bf16.fp;
                break;
            }
            default:
                ARM_COMPUTE_ERROR("Unsupported Activation for 16-bit LUT table");
                break;
        }
        if (item.i == 65535)
            break;
        item.i++;
    }
}

} // namespace

template <>
inline std::map<LUTInfo, std::weak_ptr<LookupTable256>> &LUTManager::get_map<LookupTable256>()
{
    return map_fp32;
}

template <>
inline std::map<LUTInfo, std::weak_ptr<LookupTable65536>> &LUTManager::get_map<LookupTable65536>()
{
    return map_fp16;
}

template <typename T>
std::shared_ptr<T> LUTManager::get_lut_table(LUTInfo info)
{
    auto      &map   = get_map<T>();
    const auto itr   = map.find(info);
    auto       s_ptr = (itr != map.end()) ? itr->second.lock() : nullptr; // nullptr if invalid or not found.
    if (s_ptr != nullptr)
    {
        // Found and valid
        return s_ptr; // Return weak ptr as shared ptr
    }
    else
    {
        // Not found, or pointer not valid
        // We do not use make_shared to prevent the weak_ptr keeping the control block alive
        std::shared_ptr<T> ptr(new T);
        init_lut(*ptr, info);
        map[info] = ptr;
        return ptr;
    }
}

template std::shared_ptr<LookupTable256>   LUTManager::get_lut_table<LookupTable256>(LUTInfo info);
template std::shared_ptr<LookupTable65536> LUTManager::get_lut_table<LookupTable65536>(LUTInfo info);
#endif // __aarch64__

// Static function to get LutManager instance
LUTManager &LUTManager::get_instance()
{
    static auto inst_ = std::make_unique<LUTManager>(); // The one, single instance.
    return *inst_;
}

} // namespace arm_compute
