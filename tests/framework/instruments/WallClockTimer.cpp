/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "WallClockTimer.h"

#include "../Framework.h"
#include "../Utils.h"

namespace arm_compute
{
namespace test
{
namespace framework
{
template <bool output_timestamps>
std::string    WallClock<output_timestamps>::id() const
{
    if(output_timestamps)
    {
        return "Wall clock timestamps";
    }
    else
    {
        return "Wall clock";
    }
}

template <bool output_timestamps>
void           WallClock<output_timestamps>::start()
{
    _start = std::chrono::system_clock::now();
}

template <bool output_timestamps>
void           WallClock<output_timestamps>::stop()
{
    _stop = std::chrono::system_clock::now();
}

template <bool              output_timestamps>
Instrument::MeasurementsMap WallClock<output_timestamps>::measurements() const
{
    MeasurementsMap measurements;
    if(output_timestamps)
    {
        // _start / _stop are in ns, so divide by an extra 1000:
        measurements.emplace("[start]Wall clock time", Measurement(_start.time_since_epoch().count() / static_cast<uint64_t>(1000 * _scale_factor), _unit));
        measurements.emplace("[end]Wall clock time", Measurement(_stop.time_since_epoch().count() / static_cast<uint64_t>(1000 * _scale_factor), _unit));
    }
    else
    {
        const auto delta = std::chrono::duration_cast<std::chrono::microseconds>(_stop - _start);
        measurements.emplace("Wall clock time", Measurement(delta.count() / _scale_factor, _unit));
    }
    return measurements;
}

} // namespace framework
} // namespace test
} // namespace arm_compute

template class arm_compute::test::framework::WallClock<true>;
template class arm_compute::test::framework::WallClock<false>;
