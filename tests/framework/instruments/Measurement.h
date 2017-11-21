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
#ifndef ARM_COMPUTE_TEST_MEASUREMENT
#define ARM_COMPUTE_TEST_MEASUREMENT

#include "../Utils.h"
#include "arm_compute/core/Error.h"

#include <list>
#include <ostream>
#include <string>

namespace arm_compute
{
namespace test
{
namespace framework
{
/** Generic measurement that stores values as either double or long long int. */
struct Measurement
{
    struct Value
    {
        /** Constructor
         *
         * @param[in] is_floating Will the value stored be floating point ?
         */
        Value(bool is_floating)
            : v{ 0 }, is_floating_point(is_floating)
        {
        }

        /** Add the value stored to the stream as a string
         */
        friend std::ostream &operator<<(std::ostream &os, const Value &value)
        {
            if(value.is_floating_point)
            {
                os << arithmetic_to_string(value.v.floating_point, 4);
            }
            else
            {
                os << arithmetic_to_string(value.v.integer);
            }
            return os;
        }
        /** Convert the value stored to string
         */
        std::string to_string() const
        {
            std::stringstream ss;
            ss << *this;
            return ss.str();
        }
        /** Add with another value and return the sum
         *
         * @param[in] b Other value
         *
         * @return Sum of the stored value + b
         */
        Value operator+(Value b) const
        {
            if(is_floating_point)
            {
                b.v.floating_point += v.floating_point;
            }
            else
            {
                b.v.integer += v.integer;
            }
            return b;
        }

        /** Subtract with another value and return the result
         *
         * @param[in] b Other value
         *
         * @return Result of the stored value - b
         */
        Value operator-(Value b) const
        {
            if(is_floating_point)
            {
                b.v.floating_point -= v.floating_point;
            }
            else
            {
                b.v.integer -= v.integer;
            }
            return b;
        }

        /** Multiple with another value and return the result
         *
         * @param[in] b Other value
         *
         * @return Result of the stored value * b
         */
        Value operator*(Value b) const
        {
            if(is_floating_point)
            {
                b.v.floating_point *= v.floating_point;
            }
            else
            {
                b.v.integer *= v.integer;
            }
            return b;
        }

        /** Return the stored value divided by an integer.
         *
         * @param[in] b Integer to divide the value by.
         *
         * @return Stored value / b
         */
        Value operator/(int b) const
        {
            Value res(is_floating_point);
            if(is_floating_point)
            {
                res.v.floating_point = v.floating_point / b;
            }
            else
            {
                res.v.integer = v.integer / b;
            }
            return res;
        }

        /** Subtract another value and return the updated stored value.
         *
         * @param[in] b Other value
         *
         * @return The updated stored value
         */
        Value &operator-=(const Value &b)
        {
            if(is_floating_point)
            {
                v.floating_point -= b.v.floating_point;
            }
            else
            {
                v.integer -= b.v.integer;
            }
            return *this;
        }

        /** Compare the stored value with another value
         *
         * @param[in] b Value to compare against
         *
         * @return The result of stored value < b
         */
        bool operator<(const Value &b) const
        {
            if(is_floating_point)
            {
                return v.floating_point < b.v.floating_point;
            }
            else
            {
                return v.integer < b.v.integer;
            }
        }

        static double relative_standard_deviation(const Value &variance, const Value &mean)
        {
            if(variance.is_floating_point)
            {
                return 100.0 * sqrt(variance.v.floating_point) / mean.v.floating_point;
            }
            else
            {
                return 100.0 * sqrt(static_cast<double>(variance.v.integer)) / mean.v.integer;
            }
        }

        /** Stored value */
        union
            {
                double        floating_point;
                long long int integer;
            } v;
        bool is_floating_point; /**< Is the stored value floating point or integer ? */
    };

    /** Stream output operator to print the measurement.
     *
     * Prints value and unit.
     */
    friend inline std::ostream &operator<<(std::ostream &os, const Measurement &measurement)
    {
        os << measurement._value << " " << measurement._unit;
        return os;
    }

    /** Constructor to store a floating point value
     *
     * @param[in] v    Value to store
     * @param[in] unit Unit of @p v
     * @param[in] raw  (Optional) The raw value(s) @p was generated from.
     */
    template < typename Floating, typename std::enable_if < !std::is_integral<Floating>::value, int >::type = 0 >
    Measurement(Floating v, std::string unit, std::list<std::string> raw = {})
        : _unit(unit), _raw_data(std::move(raw)), _value(true)
    {
        _value.v.floating_point = static_cast<double>(v);
        if(_raw_data.empty())
        {
            _raw_data = { _value.to_string() };
        }
    }

    /** Constructor to store an integer value
     *
     * @param[in] v    Value to store
     * @param[in] unit Unit of @p v
     * @param[in] raw  (Optional) The raw value(s) @p was generated from.
     */
    template <typename Integer, typename std::enable_if<std::is_integral<Integer>::value, int>::type = 0>
    Measurement(Integer v, std::string unit, std::list<std::string> raw = {})
        : _unit(unit), _raw_data(std::move(raw)), _value(false)
    {
        _value.v.integer = static_cast<long long int>(v);
        if(_raw_data.empty())
        {
            _raw_data = { _value.to_string() };
        }
    }

    /** Accessor for the unit of the measurement
     *
     * @return Unit of the measurement
     */
    const std::string &unit() const
    {
        return _unit;
    }

    /** Accessor for the raw data
     *
     * @return The raw data
     */
    const std::list<std::string> &raw_data() const
    {
        return _raw_data;
    }

    /** Accessor for the stored value
     *
     * @return The stored value
     */
    const Value &value() const
    {
        return _value;
    }

private:
    std::string            _unit;
    std::list<std::string> _raw_data;
    Value                  _value;
};

} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_MEASUREMENT */
