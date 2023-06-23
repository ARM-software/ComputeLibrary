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

#include "arm_compute/core/utils/FormatUtils.h"

#include <map>

namespace arm_compute
{
const std::string &string_from_format(Format format)
{
    static std::map<Format, const std::string> formats_map =
    {
        { Format::UNKNOWN, "UNKNOWN" },
        { Format::U8, "U8" },
        { Format::S16, "S16" },
        { Format::U16, "U16" },
        { Format::S32, "S32" },
        { Format::U32, "U32" },
        { Format::F16, "F16" },
        { Format::F32, "F32" },
        { Format::UV88, "UV88" },
        { Format::RGB888, "RGB888" },
        { Format::RGBA8888, "RGBA8888" },
        { Format::YUV444, "YUV444" },
        { Format::YUYV422, "YUYV422" },
        { Format::NV12, "NV12" },
        { Format::NV21, "NV21" },
        { Format::IYUV, "IYUV" },
        { Format::UYVY422, "UYVY422" }
    };

    return formats_map[format];
}
} // namespace arm_compute
