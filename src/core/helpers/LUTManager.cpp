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
    union Element
    {
        uint16_t  i = 0;
        float16_t fp;
    } item;
    // Fill lut by iterating over all 16 bit values using the union.
    while (true)
    {
        (*lut)[item.i] = 1.f / (1.f + std::exp(-item.fp));
        if (item.i == 65535)
            break;
        item.i++;
    }
}
} // namespace

std::shared_ptr<ActivationLayerInfo::LookupTable65536> LUTManager::get_lut_table(LUTInfo info)
{
    const auto itr   = map_fp16.find(info);
    auto       s_ptr = (itr != map_fp16.end()) ? itr->second.lock() : nullptr; // nullptr if invalid or not found.
    if (s_ptr != nullptr)
    {
        // Found and valid
        return s_ptr; // Return weak ptr as shared ptr
    }
    else
    {
        // Not found, or pointer not valid
        // We do not use make_shared to prevent the weak_ptr keeping the control block alive
        std::shared_ptr<ActivationLayerInfo::LookupTable65536> ptr(new ActivationLayerInfo::LookupTable65536);
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
