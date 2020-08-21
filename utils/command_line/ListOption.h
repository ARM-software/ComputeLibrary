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
#ifndef ARM_COMPUTE_UTILS_LISTOPTION
#define ARM_COMPUTE_UTILS_LISTOPTION

#include "Option.h"

#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace arm_compute
{
namespace utils
{
/** Implementation of an option that accepts any number of values. */
template <typename T>
class ListOption : public Option
{
public:
    using Option::Option;

    /** Construct the option with the given default values.
     *
     * @param[in] name           Name of the option.
     * @param[in] default_values Default values.
     */
    ListOption(std::string name, std::initializer_list<T> &&default_values);

    bool parse(std::string value) override;
    std::string help() const override;

    /** Get the list of option values.
     *
     * @return get the list of option values.
     */
    const std::vector<T> &value() const;

private:
    std::vector<T> _values{};
};

template <typename T>
inline ListOption<T>::ListOption(std::string name, std::initializer_list<T> &&default_values)
    : Option{ std::move(name), false, true }, _values{ std::forward<std::initializer_list<T>>(default_values) }
{
}

template <typename T>
bool ListOption<T>::parse(std::string value)
{
    _is_set = true;

    try
    {
        std::stringstream stream{ value };
        std::string       item;

        while(!std::getline(stream, item, ',').fail())
        {
            std::stringstream item_stream(item);
            T                 typed_value{};

            item_stream >> typed_value;

            if(!item_stream.fail())
            {
                _values.emplace_back(typed_value);
            }

            _is_set = _is_set && !item_stream.fail();
        }

        return _is_set;
    }
    catch(const std::invalid_argument &)
    {
        return false;
    }
}

template <typename T>
inline std::string ListOption<T>::help() const
{
    return "--" + name() + "=VALUE[,VALUE[,...]] - " + _help;
}

template <typename T>
inline const std::vector<T> &ListOption<T>::value() const
{
    return _values;
}
} // namespace utils
} // namespace arm_compute
#endif /* ARM_COMPUTE_UTILS_LISTOPTION */
