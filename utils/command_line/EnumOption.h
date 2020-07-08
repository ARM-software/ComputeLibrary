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
#ifndef ARM_COMPUTE_UTILS_ENUMOPTION
#define ARM_COMPUTE_UTILS_ENUMOPTION

#include "SimpleOption.h"

#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace arm_compute
{
namespace utils
{
/** Implementation of a simple option that accepts a value from a fixed set. */
template <typename T>
class EnumOption : public SimpleOption<T>
{
public:
    /** Construct option with allowed values.
     *
     * @param[in] name           Name of the option.
     * @param[in] allowed_values Set of allowed values for the option.
     */
    EnumOption(std::string name, std::set<T> allowed_values);

    /** Construct option with allowed values, a fixed number of accepted values and default values for the option.
     *
     * @param[in] name           Name of the option.
     * @param[in] allowed_values Set of allowed values for the option.
     * @param[in] default_value  Default value.
     */
    EnumOption(std::string name, std::set<T> allowed_values, T default_value);

    bool parse(std::string value) override;
    std::string help() const override;

    /** Get the selected value.
     *
     * @return get the selected enum value.
     */
    const T &value() const;

private:
    std::set<T> _allowed_values{};
};

template <typename T>
inline EnumOption<T>::EnumOption(std::string name, std::set<T> allowed_values)
    : SimpleOption<T>{ std::move(name) }, _allowed_values{ std::move(allowed_values) }
{
}

template <typename T>
inline EnumOption<T>::EnumOption(std::string name, std::set<T> allowed_values, T default_value)
    : SimpleOption<T>{ std::move(name), std::move(default_value) }, _allowed_values{ std::move(allowed_values) }
{
}

template <typename T>
bool EnumOption<T>::parse(std::string value)
{
    try
    {
        std::stringstream stream{ value };
        T                 typed_value{};

        stream >> typed_value;

        if(!stream.fail())
        {
            if(_allowed_values.count(typed_value) == 0)
            {
                return false;
            }

            this->_value  = std::move(typed_value);
            this->_is_set = true;
            return true;
        }

        return false;
    }
    catch(const std::invalid_argument &)
    {
        return false;
    }
}

template <typename T>
std::string EnumOption<T>::help() const
{
    std::stringstream msg;
    msg << "--" + this->name() + "={";

    for(const auto &value : _allowed_values)
    {
        msg << value << ",";
    }

    msg << "} - " << this->_help;

    return msg.str();
}

template <typename T>
inline const T &EnumOption<T>::value() const
{
    return this->_value;
}
} // namespace utils
} // namespace arm_compute
#endif /* ARM_COMPUTE_UTILS_ENUMOPTION */
