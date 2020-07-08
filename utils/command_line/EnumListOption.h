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
#ifndef ARM_COMPUTE_UTILS_ENUMLISTOPTION
#define ARM_COMPUTE_UTILS_ENUMLISTOPTION

#include "Option.h"

#include <initializer_list>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace arm_compute
{
namespace utils
{
/** Implementation of an option that accepts any number of values from a fixed set. */
template <typename T>
class EnumListOption : public Option
{
public:
    /** Construct option with allowed values.
     *
     * @param[in] name           Name of the option.
     * @param[in] allowed_values Set of allowed values for the option.
     */
    EnumListOption(std::string name, std::set<T> allowed_values);

    /** Construct option with allowed values, a fixed number of accepted values and default values for the option.
     *
     * @param[in] name           Name of the option.
     * @param[in] allowed_values Set of allowed values for the option.
     * @param[in] default_values Default values.
     */
    EnumListOption(std::string name, std::set<T> allowed_values, std::initializer_list<T> &&default_values);

    bool parse(std::string value) override;
    std::string help() const override;

    /** Get the values of the option.
     *
     * @return a list of the selected option values.
     */
    const std::vector<T> &value() const;

private:
    std::vector<T> _values{};
    std::set<T>    _allowed_values{};
};

template <typename T>
inline EnumListOption<T>::EnumListOption(std::string name, std::set<T> allowed_values)
    : Option{ std::move(name) }, _allowed_values{ std::move(allowed_values) }
{
}

template <typename T>
inline EnumListOption<T>::EnumListOption(std::string name, std::set<T> allowed_values, std::initializer_list<T> &&default_values)
    : Option{ std::move(name), false, true }, _values{ std::forward<std::initializer_list<T>>(default_values) }, _allowed_values{ std::move(allowed_values) }
{
}

template <typename T>
bool EnumListOption<T>::parse(std::string value)
{
    // Remove default values
    _values.clear();
    _is_set = true;

    std::stringstream stream{ value };
    std::string       item;

    while(!std::getline(stream, item, ',').fail())
    {
        try
        {
            std::stringstream item_stream(item);
            T                 typed_value{};

            item_stream >> typed_value;

            if(!item_stream.fail())
            {
                if(_allowed_values.count(typed_value) == 0)
                {
                    _is_set = false;
                    continue;
                }

                _values.emplace_back(typed_value);
            }

            _is_set = _is_set && !item_stream.fail();
        }
        catch(const std::invalid_argument &)
        {
            _is_set = false;
        }
    }

    return _is_set;
}

template <typename T>
std::string EnumListOption<T>::help() const
{
    std::stringstream msg;
    msg << "--" + name() + "={";

    for(const auto &value : _allowed_values)
    {
        msg << value << ",";
    }

    msg << "}[,{...}[,...]] - " << _help;

    return msg.str();
}

template <typename T>
inline const std::vector<T> &EnumListOption<T>::value() const
{
    return _values;
}
} // namespace utils
} // namespace arm_compute
#endif /* ARM_COMPUTE_UTILS_ENUMLISTOPTION */
