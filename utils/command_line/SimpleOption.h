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
#ifndef ARM_COMPUTE_UTILS_SIMPLEOPTION
#define ARM_COMPUTE_UTILS_SIMPLEOPTION

#include "Option.h"

#include <sstream>
#include <stdexcept>
#include <string>

namespace arm_compute
{
namespace utils
{
/** Implementation of an option that accepts a single value. */
template <typename T>
class SimpleOption : public Option
{
public:
    using Option::Option;

    /** Construct the option with the given default value.
     *
     * @param[in] name          Name of the option.
     * @param[in] default_value Default value.
     */
    SimpleOption(std::string name, T default_value);

    /** Parses the given string.
     *
     * @param[in] value String representation as passed on the command line.
     *
     * @return True if the value could be parsed by the specific subclass.
     */
    bool parse(std::string value) override;

    /** Help message for the option.
     *
     * @return String representing the help message for the specific subclass.
     */
    std::string help() const override;

    /** Get the option value.
     *
     * @return the option value.
     */
    const T &value() const;

protected:
    T _value{};
};

template <typename T>
inline SimpleOption<T>::SimpleOption(std::string name, T default_value)
    : Option{ std::move(name), false, true }, _value{ std::move(default_value) }
{
}

template <typename T>
bool SimpleOption<T>::parse(std::string value)
{
    try
    {
        std::stringstream stream{ std::move(value) };
        stream >> _value;
        _is_set = !stream.fail();
        return _is_set;
    }
    catch(const std::invalid_argument &)
    {
        return false;
    }
}

template <>
inline bool SimpleOption<std::string>::parse(std::string value)
{
    _value  = std::move(value);
    _is_set = true;
    return true;
}

template <typename T>
inline std::string SimpleOption<T>::help() const
{
    return "--" + name() + "=VALUE - " + _help;
}

template <typename T>
inline const T &SimpleOption<T>::value() const
{
    return _value;
}
} // namespace utils
} // namespace arm_compute
#endif /* ARM_COMPUTE_UTILS_SIMPLEOPTION */
