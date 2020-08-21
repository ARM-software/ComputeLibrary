/*
 * Copyright (c) 2017-2018 Arm Limited.
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
#ifndef ARM_COMPUTE_UTILS_TOGGLEOPTION
#define ARM_COMPUTE_UTILS_TOGGLEOPTION

#include "SimpleOption.h"

#include <string>

namespace arm_compute
{
namespace utils
{
/** Implementation of an option that can be either true or false. */
class ToggleOption : public SimpleOption<bool>
{
public:
    using SimpleOption::SimpleOption;

    /** Construct the option with the given default value.
     *
     * @param[in] name          Name of the option.
     * @param[in] default_value Default value.
     */
    ToggleOption(std::string name, bool default_value);

    bool parse(std::string value) override;
    std::string help() const override;
};

inline ToggleOption::ToggleOption(std::string name, bool default_value)
    : SimpleOption<bool>
{
    std::move(name), default_value
}
{
}

inline bool ToggleOption::parse(std::string value)
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

inline std::string ToggleOption::help() const
{
    return "--" + name() + ", --no-" + name() + " - " + _help;
}
} // namespace utils
} // namespace arm_compute
#endif /* ARM_COMPUTE_UTILS_TOGGLEOPTION */
