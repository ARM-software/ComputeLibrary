/*
 * Copyright (c) 2017-2020 ARM Limited.
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

#include "support/MemorySupport.h"

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

    /** Default constructor. */
    Instrument() = default;

    /** Allow instances of this class to be copy constructed */
    Instrument(const Instrument &) = default;
    /** Allow instances of this class to be move constructed */
    Instrument(Instrument &&) = default;
    /** Allow instances of this class to be copied */
    Instrument &operator=(const Instrument &) = default;
    /** Allow instances of this class to be moved */
    Instrument &operator=(Instrument &&) = default;
    /** Default destructor. */
    virtual ~Instrument() = default;

    /** Identifier for the instrument */
    virtual std::string id() const = 0;

    /** Start of the test
     *
     * Called before the test set up starts
     */
    virtual void test_start()
    {
    }

    /** Start measuring.
     *
     * Called just before the run of the test starts
     */
    virtual void start()
    {
    }

    /** Stop measuring.
     *
     * Called just after the run of the test ends
    */
    virtual void stop()
    {
    }

    /** End of the test
     *
     * Called after the test teardown ended
     */
    virtual void test_stop()
    {
    }
    /** Map of measurements */
    using MeasurementsMap = std::map<std::string, Measurement>;

    /** Return the latest measurements.
     *
     * @return the latest measurements.
     */
    virtual MeasurementsMap measurements() const
    {
        return MeasurementsMap();
    }

    /** Return the latest test measurements.
     *
     * @return the latest test measurements.
     */
    virtual MeasurementsMap test_measurements() const
    {
        return MeasurementsMap();
    }

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
