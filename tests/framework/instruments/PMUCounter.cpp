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
#include "PMUCounter.h"

namespace arm_compute
{
namespace test
{
namespace framework
{
std::string PMUCounter::id() const
{
    return "PMU Counter";
}

void PMUCounter::start()
{
    _pmu_cycles.reset();
    _pmu_instructions.reset();
}

void PMUCounter::stop()
{
    try
    {
        _cycles = _pmu_cycles.get_value<long long>();
    }
    catch(const std::runtime_error &)
    {
        _cycles = 0;
    }

    try
    {
        _instructions = _pmu_instructions.get_value<long long>();
    }
    catch(const std::runtime_error &)
    {
        _instructions = 0;
    }
}

Instrument::MeasurementsMap PMUCounter::measurements() const
{
    return MeasurementsMap
    {
        { "CPU cycles", Measurement(_cycles / _scale_factor, _unit + "cycles") },
        { "CPU instructions", Measurement(_instructions / _scale_factor, _unit + "instructions") },
    };
}
} // namespace framework
} // namespace test
} // namespace arm_compute
