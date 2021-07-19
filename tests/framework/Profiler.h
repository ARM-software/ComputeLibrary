/*
 * Copyright (c) 2017-2018,2021 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_PROFILER
#define ARM_COMPUTE_TEST_PROFILER

#include "instruments/Instrument.h"
#include "instruments/Measurement.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace arm_compute
{
namespace test
{
namespace framework
{
/** Profiler class to collect benchmark numbers.
 *
 * A profiler manages multiple instruments that can collect different types of benchmarking numbers.
 */
class Profiler
{
public:
    /** Mapping from instrument ids to their measurements. */
    using MeasurementsMap = std::map<std::string, std::vector<Measurement>>;

    /** Add @p instrument to the performance monitor.
     *
     * All added instruments will be used when @ref start or @ref stop are
     * called to make measurements.
     *
     * @param[in] instrument Instrument to be used to measure performance.
     */
    void add(std::unique_ptr<Instrument> instrument);

    /** Call test_start() on all the added instruments
     *
     * Called before the test set up starts
     */
    void test_start();

    /** Call start() on all the added instruments
     *
     * Called just before the run of the test starts
     */
    void start();

    /** Call stop() on all the added instruments
     *
     * Called just after the run of the test ends
    */
    void stop();

    /** Call test_stop() on all the added instruments
     *
     * Called after the test teardown ended
     */
    void test_stop();

    /** Return measurements for all instruments.
     *
     * @return measurements for all instruments.
     */
    const MeasurementsMap &measurements() const;

    /** Return JSON formatted header data.
     *
     * @returns JSON formmated string
     */
    const std::string &header() const;

private:
    std::vector<std::unique_ptr<Instrument>> _instruments{};
    MeasurementsMap                          _measurements{};
    std::string                              _header_data{};
};
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_PROFILER */
