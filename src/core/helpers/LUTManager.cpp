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

namespace arm_compute
{
#ifdef __aarch64__
namespace
{
void init_lut_fp16(ActivationLayerInfo::LookupTable65536 *lut)
{
    for (uint16_t i = 0; i < lut->size() - 1; ++i)
    {
        const float16_t *v = reinterpret_cast<float16_t *>(&i);
        (*lut)[i]          = 1.f / (1.f + std::exp(-*v));
    }
    // Final value should be filled outside of loop to avoid overflows.
    const uint16_t   i = lut->size() - 1;
    const float16_t *v = reinterpret_cast<const float16_t *>(&i);
    (*lut)[i]          = 1.f / (1.f + std::exp(-*v));
}
} // namespace

std::shared_ptr<ActivationLayerInfo::LookupTable65536> LUTManager::get_lut_table(LUTInfo info)
{
    const auto itr = map_fp16.find(info);
    if (itr != map_fp16.end() && !itr->second.expired())
    {
        // Found and valid
        return itr->second.lock(); // Return weak ptr as shared ptr
    }
    else
    {
        // Not found, or pointer not valid
        const auto ptr = std::make_shared<ActivationLayerInfo::LookupTable65536>();
        init_lut_fp16(ptr.get());
        map_fp16[info] = ptr;
        return ptr;
    }
}
#endif // __aarch64__

// Static function to get LutManager instance
LUTManager &LUTManager::get_instance()
{
    static auto inst_ = std::make_unique<LUTManager>(); // The one, single instance.
    return *inst_;
}

} // namespace arm_compute
