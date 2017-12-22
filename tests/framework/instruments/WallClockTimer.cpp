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
#include "WallClockTimer.h"

#include "../Framework.h"
#include "../Utils.h"

namespace arm_compute
{
namespace test
{
namespace framework
{
std::string WallClockTimer::id() const
{
    return "Wall clock";
}

void WallClockTimer::start()
{
    _start = std::chrono::high_resolution_clock::now();
}

void WallClockTimer::stop()
{
    _stop = std::chrono::high_resolution_clock::now();
}

Instrument::MeasurementsMap WallClockTimer::measurements() const
{
    const auto delta = std::chrono::duration_cast<std::chrono::microseconds>(_stop - _start);
    return MeasurementsMap{ { "Wall clock time", Measurement(delta.count() / _scale_factor, _unit) } };
}
} // namespace framework
} // namespace test
} // namespace arm_compute
