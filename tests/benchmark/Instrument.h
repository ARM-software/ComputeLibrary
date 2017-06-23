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
#ifndef __ARM_COMPUTE_TEST_BENCHMARK_INSTRUMENT_H__
#define __ARM_COMPUTE_TEST_BENCHMARK_INSTRUMENT_H__

#include "Utils.h"

#include <memory>
#include <string>

namespace arm_compute
{
namespace test
{
namespace benchmark
{
/** Interface for classes that can be used to measure performance. */
class Instrument
{
public:
    /** Interface defining a measurement, e.g. time, cycles, ... */
    class IMeasurement
    {
    public:
        IMeasurement()                     = default;
        IMeasurement(const IMeasurement &) = default;
        IMeasurement(IMeasurement &&)      = default;
        IMeasurement &operator=(const IMeasurement &) = default;
        IMeasurement &operator=(IMeasurement &&) = default;
        virtual ~IMeasurement()                  = default;

        virtual operator double() const = 0;
    };

    /** Implementation of a Measurement class for arihtmetic types. */
    template <typename T>
    class Measurement : public IMeasurement
    {
    public:
        /** Store the given value as measurement.
         *
         * @param[in] value Measured value.
         */
        Measurement(T value);

        operator double() const override;

    private:
        T _value;
    };

    Instrument()                   = default;
    Instrument(const Instrument &) = default;
    Instrument(Instrument &&)      = default;
    Instrument &operator=(const Instrument &) = default;
    Instrument &operator=(Instrument &&) = default;
    virtual ~Instrument()                = default;

    /** Identifier for the instrument */
    virtual std::string id() const = 0;

    /** Start measuring. */
    virtual void start() = 0;

    /** Stop measuring. */
    virtual void stop() = 0;

    /** Return the latest measurement. */
    virtual std::unique_ptr<IMeasurement> get_measurement() const = 0;
};

template <typename T>
Instrument::Measurement<T>::Measurement(T value)
    : _value{ value }
{
}

template <typename T>
Instrument::Measurement<T>::operator double() const
{
    return _value;
}
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif
