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
#ifndef ARM_COMPUTE_TEST_WALL_CLOCK_TIMER
#define ARM_COMPUTE_TEST_WALL_CLOCK_TIMER

#include "Instrument.h"

#include <chrono>

namespace arm_compute
{
namespace test
{
namespace framework
{
/** Implementation of an instrument to measure elapsed wall-clock time in milliseconds. */
template <bool output_timestamps>
class WallClock : public Instrument
{
public:
    /** Construct a Wall clock timer.
     *
     * @param[in] scale_factor Measurement scale factor.
     */
    WallClock(ScaleFactor scale_factor)
    {
        switch(scale_factor)
        {
            case ScaleFactor::NONE:
                _scale_factor = 1.f;
                _unit         = "us";
                break;
            case ScaleFactor::TIME_MS:
                _scale_factor = 1000.f;
                _unit         = "ms";
                break;
            case ScaleFactor::TIME_S:
                _scale_factor = 1000000.f;
                _unit         = "s";
                break;
            default:
                ARM_COMPUTE_ERROR("Invalid scale");
        }
    };

    std::string     id() const override;
    void            start() override;
    void            stop() override;
    MeasurementsMap measurements() const override;

private:
    std::chrono::system_clock::time_point _start{};
    std::chrono::system_clock::time_point _stop{};
    float                                 _scale_factor{};
};

using WallClockTimer      = WallClock<false>;
using WallClockTimestamps = WallClock<true>;
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_WALL_CLOCK_TIMER */
