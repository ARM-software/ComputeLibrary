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
#ifndef ARM_COMPUTE_TEST_PMU_COUNTER
#define ARM_COMPUTE_TEST_PMU_COUNTER

#include "Instrument.h"
#include "PMU.h"

namespace arm_compute
{
namespace test
{
namespace framework
{
/** Implementation of an instrument to count CPU cycles. */
class PMUCounter : public Instrument
{
public:
    PMUCounter(ScaleFactor scale_factor)
    {
        switch(scale_factor)
        {
            case ScaleFactor::NONE:
                _scale_factor = 1;
                _unit         = "";
                break;
            case ScaleFactor::SCALE_1K:
                _scale_factor = 1000;
                _unit         = "K ";
                break;
            case ScaleFactor::SCALE_1M:
                _scale_factor = 1000000;
                _unit         = "M ";
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
    PMU       _pmu_cycles{ PERF_COUNT_HW_CPU_CYCLES };
    PMU       _pmu_instructions{ PERF_COUNT_HW_INSTRUCTIONS };
    long long _cycles{ 0 };
    long long _instructions{ 0 };
    int       _scale_factor{};
};
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_PMU_COUNTER */
