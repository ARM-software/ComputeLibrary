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
#include "arm_compute/core/utils/StringUtils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>

namespace arm_compute
{
std::string lower_string(const std::string &val)
{
    std::string res = val;
    std::transform(res.begin(), res.end(), res.begin(), ::tolower);
    return res;
}

std::string upper_string(const std::string &val)
{
    std::string res = val;
    std::transform(res.begin(), res.end(), res.begin(), ::toupper);
    return res;
}

std::string float_to_string_with_full_precision(float val)
{
    std::stringstream ss;
    ss.precision(std::numeric_limits<float>::max_digits10);
    ss << val;

    if(val != static_cast<int>(val))
    {
        ss << "f";
    }

    return ss.str();
}

std::string join(const std::vector<std::string> strings, const std::string &sep)
{
    if(strings.empty())
    {
        return "";
    }
    return std::accumulate(
               std::next(strings.begin()),
               strings.end(),
               strings.at(0),
               [&sep](const std::string & a, const std::string & b)
    {
        return a + sep + b;
    });
}
}
