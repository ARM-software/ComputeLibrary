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
#include "Option.h"

namespace arm_compute
{
namespace test
{
namespace framework
{
Option::Option(std::string name)
    : _name{ std::move(name) }
{
}

Option::Option(std::string name, bool is_required, bool is_set)
    : _name{ std::move(name) }, _is_required{ is_required }, _is_set{ is_set }
{
}

std::string Option::name() const
{
    return _name;
}

void Option::set_required(bool is_required)
{
    _is_required = is_required;
}

void Option::set_help(std::string help)
{
    _help = std::move(help);
}

bool Option::is_required() const
{
    return _is_required;
}

bool Option::is_set() const
{
    return _is_set;
}
} // namespace framework
} // namespace test
} // namespace arm_compute
