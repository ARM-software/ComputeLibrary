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

#include <ostream>
#include <string>

namespace arm_compute
{
namespace test
{
namespace framework
{
/** Abstract measurement.
 *
 * Every measurement needs to define it's unit.
 */
struct IMeasurement
{
    /** Constructor.
     *
     * @param[in] unit Unit of the measurement.
     */
    explicit IMeasurement(std::string unit)
        : unit{ std::move(unit) }
    {
    }

    std::string unit;
};

/** Measurement of a specific type. */
template <typename T>
struct TypedMeasurement : public IMeasurement
{
    /** Constructor.
     *
     * @param[in] value Measured value.
     * @param[in] unit  Unit of the Measurement.
     */
    TypedMeasurement(T value, std::string unit)
        : IMeasurement(std::move(unit)), value{ value }
    {
    }

    T value;
};

/** Stream output operator to print the measurement.
 *
 * Prints value and unit.
 */
template <typename T>
inline std::ostream &operator<<(std::ostream &os, const TypedMeasurement<T> &measurement)
{
    os << measurement.value << measurement.unit;
    return os;
}

/** Generic measurement that stores values as double. */
struct Measurement : public TypedMeasurement<double>
{
    using TypedMeasurement::TypedMeasurement;

    /** Conversion constructor.
     *
     * @param[in] measurement Typed measurement.
     */
    template <typename T>
    Measurement(TypedMeasurement<T> measurement)
        : Measurement(measurement.value, measurement.unit)
    {
    }
};
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_MEASUREMENT */
