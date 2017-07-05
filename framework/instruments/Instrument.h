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
#ifndef ARM_COMPUTE_TEST_INSTRUMENT
#define ARM_COMPUTE_TEST_INSTRUMENT

#include "../Utils.h"

#include <memory>
#include <ostream>
#include <string>

namespace arm_compute
{
namespace test
{
namespace framework
{
/** Interface for classes that can be used to measure performance. */
class Instrument
{
public:
    /** Helper function to create an instrument of the given type.
     *
     * @return Instance of an instrument of the given type.
     */
    template <typename T>
    static std::unique_ptr<Instrument> make_instrument();

    /** Struct representing measurement consisting of value and unit. */
    struct Measurement final
    {
        Measurement(double value, std::string unit)
            : value{ value }, unit{ std::move(unit) }
        {
        }

        friend std::ostream &operator<<(std::ostream &os, const Measurement &measurement);

        double      value;
        std::string unit;
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
    virtual Measurement measurement() const = 0;
};

inline std::ostream &operator<<(std::ostream &os, const Instrument::Measurement &measurement)
{
    os << measurement.value << measurement.unit;
    return os;
}

template <typename T>
inline std::unique_ptr<Instrument> Instrument::make_instrument()
{
    return support::cpp14::make_unique<T>();
}
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_INSTRUMENT */
