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
#include "ToggleOption.h"

#include <utility>

namespace arm_compute
{
namespace test
{
namespace framework
{
ToggleOption::ToggleOption(std::string name, bool default_value)
    : SimpleOption<bool>
{
    std::move(name), default_value
}
{
}

bool ToggleOption::parse(std::string value)
{
    if(value == "true")
    {
        _value  = true;
        _is_set = true;
    }
    else if(value == "false")
    {
        _value  = false;
        _is_set = true;
    }

    return _is_set;
}

std::string ToggleOption::help() const
{
    return "--" + name() + ", --no-" + name() + " - " + _help;
}
} // namespace framework
} // namespace test
} // namespace arm_compute
