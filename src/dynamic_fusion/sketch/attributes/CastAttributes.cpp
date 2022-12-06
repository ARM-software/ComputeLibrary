/*
 * Copyright (c) 2022 Arm Limited.
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

#include "arm_compute/dynamic_fusion/sketch/attributes/CastAttributes.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
CastAttributes &CastAttributes::data_type(const DataType &data_type)
{
    _data_type = data_type;
    return *this;
}

DataType CastAttributes::data_type() const
{
    return _data_type;
}

CastAttributes &CastAttributes::convert_policy(const ConvertPolicy &convert_policy)
{
    _convert_policy = convert_policy;
    return *this;
}

ConvertPolicy CastAttributes::convert_policy() const
{
    return _convert_policy;
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
