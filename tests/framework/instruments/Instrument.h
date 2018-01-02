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
#include "Measurement.h"

#include <map>
#include <memory>
#include <string>

namespace arm_compute
{
namespace test
{
namespace framework
{
enum class ScaleFactor : unsigned int
{
    NONE,     /* Default scale */
    SCALE_1K, /* 1000          */
    SCALE_1M, /* 1 000 000     */
    TIME_US,  /* Microseconds  */
    TIME_MS,  /* Milliseconds  */
    TIME_S,   /* Seconds       */
};
/** Interface for classes that can be used to measure performance. */
class Instrument
{
public:
    /** Helper function to create an instrument of the given type.
     *
     * @return Instance of an instrument of the given type.
     */
    template <typename T, ScaleFactor scale>
    static std::unique_ptr<Instrument> make_instrument();

    Instrument() = default;

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

    using MeasurementsMap = std::map<std::string, Measurement>;

    /** Return the latest measurement. */
    virtual MeasurementsMap measurements() const = 0;

protected:
    std::string _unit{};
};

template <typename T, ScaleFactor scale>
inline std::unique_ptr<Instrument> Instrument::make_instrument()
{
    return support::cpp14::make_unique<T>(scale);
}
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_INSTRUMENT */
