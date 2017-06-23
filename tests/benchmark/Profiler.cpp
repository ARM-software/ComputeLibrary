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
#include "Profiler.h"

#include <iostream>
#include <utility>

namespace arm_compute
{
namespace test
{
namespace benchmark
{
void Profiler::add(const std::shared_ptr<Instrument> &instrument)
{
    _instruments.push_back(instrument);
}

void Profiler::start()
{
    for(auto &instrument : _instruments)
    {
        instrument->start();
    }
}

void Profiler::stop()
{
    for(auto &instrument : _instruments)
    {
        instrument->stop();
    }

    for(const auto &instrument : _instruments)
    {
        _measurements[instrument->id()].push_back(*instrument->get_measurement());
    }
}

void Profiler::submit(::benchmark::State &state)
{
    for(auto &instrument : _measurements)
    {
        double sum_values = std::accumulate(instrument.second.begin(), instrument.second.end(), 0.);
        size_t num_values = instrument.second.size();

        if(num_values > 2)
        {
            auto minmax_values                        = std::minmax_element(instrument.second.begin(), instrument.second.end());
            state.counters[instrument.first + "_min"] = *minmax_values.first;
            state.counters[instrument.first + "_max"] = *minmax_values.second;
            sum_values -= *minmax_values.first + *minmax_values.second;
            num_values -= 2;
        }
        state.counters[instrument.first] = sum_values / num_values;
        instrument.second.clear();
    }
}

const Profiler::MeasurementsMap &Profiler::measurements() const
{
    return _measurements;
}
} // namespace benchmark
} // namespace test
} // namespace arm_compute
