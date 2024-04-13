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

#ifndef ACL_SRC_CORE_HELPERS_LUTMANAGER_H
#define ACL_SRC_CORE_HELPERS_LUTMANAGER_H

#include "arm_compute/core/CoreTypes.h"
#include "arm_compute/core/QuantizationInfo.h"
#include "arm_compute/function_info/ActivationLayerInfo.h"

#include <map>
#include <memory>

namespace arm_compute
{

struct LUTInfo
{
    ActivationLayerInfo::ActivationFunction act;
    DataType                                dt;
    QuantizationInfo                        qinfo;
    // Operators enable use of map with Lutinfo as key
    friend bool operator<(const LUTInfo &l, const LUTInfo &r)
    {
        return (l.act < r.act) || ((l.act == r.act) && (l.dt < r.dt)) ||
               ((l.act == r.act) && (l.dt == r.dt) && (l.qinfo.scale() < r.qinfo.scale())) ||
               ((l.act == r.act) && (l.dt == r.dt) && (l.qinfo.scale() == r.qinfo.scale()) &&
                (l.qinfo.offset() < l.qinfo.offset()));
    }
    bool operator==(const LUTInfo &l)
    {
        return this->act == l.act && this->dt == l.dt && this->qinfo == l.qinfo;
    }
};

/* Class to handle getting look up table */
class LUTManager
{
public:
    LUTManager() = default;

    static LUTManager &get_instance();
#ifdef __aarch64__
    std::shared_ptr<ActivationLayerInfo::LookupTable65536> get_lut_table(LUTInfo info);

private:
    std::map<LUTInfo, std::weak_ptr<ActivationLayerInfo::LookupTable65536>> map_fp16{};
#endif // __aarch64__
};

} // namespace arm_compute
#endif // ACL_SRC_CORE_HELPERS_LUTMANAGER_H
