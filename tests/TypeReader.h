/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_TEST_TYPE_READER_H__
#define __ARM_COMPUTE_TEST_TYPE_READER_H__

#include "arm_compute/core/Types.h"

#include <algorithm>
#include <cctype>
#include <istream>

namespace arm_compute
{
/** Formatted input of the BorderMode type. */
inline ::std::istream &operator>>(::std::istream &is, BorderMode &mode)
{
    std::string value;

    is >> value;

    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c)
    {
        return std::toupper(c);
    });

    if(value == "UNDEFINED")
    {
        mode = BorderMode::UNDEFINED;
    }
    else if(value == "CONSTANT")
    {
        mode = BorderMode::CONSTANT;
    }
    else if(value == "REPLICATE")
    {
        mode = BorderMode::REPLICATE;
    }
    else
    {
        throw std::invalid_argument("Unsupported value '" + value + "' for border mode");
    }

    return is;
}
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_TYPE_READER_H__ */
