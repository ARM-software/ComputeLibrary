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
#ifndef __ARM_COMPUTE_TEST_BENCHMARK_PROFILER_H__
#define __ARM_COMPUTE_TEST_BENCHMARK_PROFILER_H__

#include "Instrument.h"

#include "benchmark/benchmark_api.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace arm_compute
{
namespace test
{
namespace benchmark
{
class Profiler
{
public:
    /** Mapping from instrument ids to their measurements. */
    using MeasurementsMap = std::map<std::string, std::vector<double>>;

    /** Add @p instrument to the performance montior.
     *
     * All added instruments will be used when @ref start or @ref stop are
     * called to make measurements.
     *
     * @param[in] instrument Instrument to be used to measure performance.
     */
    void add(const std::shared_ptr<Instrument> &instrument);

    /** Start all added instruments to measure performance. */
    void start();

    /** Stop all added instruments. */
    void stop();

    /** Commit all measured values to the current active test. */
    void submit(::benchmark::State &state);

    /** Return measurements for all instruments. */
    const MeasurementsMap &measurements() const;

private:
    std::vector<std::shared_ptr<Instrument>> _instruments{};
    MeasurementsMap                          _measurements{};
};
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif
