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
#ifndef __ARM_COMPUTE_TEST_USER_CONFIGURATION_H__
#define __ARM_COMPUTE_TEST_USER_CONFIGURATION_H__

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Types.h"

#include <random>
#include <string>

namespace arm_compute
{
namespace test
{
class ProgramOptions;

/** Container providing easy access to runtime options provided by the user. */
struct UserConfiguration
{
protected:
    /** Wrapper around options to store if an option has been set. */
    template <typename T>
    class Option
    {
    public:
        /** Initialise the option to its default (C++) value and mark it as 'not set'. */
        Option();

        /** Initialise the option to the given @p value and mark it as 'set'. */
        Option(const T &value);

        /** Assign the given @p value and mark it as 'set'. */
        Option<T> &operator=(const T &value);

        /** Query if the option has been set. */
        constexpr bool is_set() const;

        /** Return the underlying value as constant. */
        T get() const;

        /** Return the underlying value. */
        T &get();

        /** Implicitly return the underlying value. */
        operator T() const;

    private:
        T    _value;
        bool _is_set;
    };

public:
    UserConfiguration() = default;

    /** Initialise the configuration according to the program options.
     *
     * @param[in] options Parsed command line options.
     */
    UserConfiguration(const ProgramOptions &options);

    Option<std::string>                     path{};
    Option<std::random_device::result_type> seed{};
    Option<unsigned int>                    threads{};
};

template <typename T>
UserConfiguration::Option<T>::Option()
    : _value{}, _is_set{ false }
{
}

template <typename T>
UserConfiguration::Option<T>::Option(const T &value)
    : _value{ value }, _is_set{ true }
{
}

template <typename T>
UserConfiguration::Option<T> &UserConfiguration::Option<T>::operator=(const T &value)
{
    _value  = value;
    _is_set = true;

    return *this;
}

template <typename T>
constexpr bool UserConfiguration::Option<T>::is_set() const
{
    return _is_set;
}

template <typename T>
T UserConfiguration::Option<T>::get() const
{
    ARM_COMPUTE_ERROR_ON(!is_set());
    return _value;
}

template <typename T>
T &UserConfiguration::Option<T>::get()
{
    return _value;
}

template <typename T>
UserConfiguration::Option<T>::operator T() const
{
    ARM_COMPUTE_ERROR_ON(!is_set());
    return _value;
}
} // namespace test
} // namespace arm_compute
#endif
