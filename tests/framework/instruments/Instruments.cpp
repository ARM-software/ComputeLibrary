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
#include "Instruments.h"

#include "../Utils.h"

#include <map>
#include <stdexcept>

namespace arm_compute
{
namespace test
{
namespace framework
{
InstrumentsDescription instrument_type_from_name(const std::string &name)
{
    static const std::map<std::string, framework::InstrumentsDescription> types =
    {
        { "all", std::pair<InstrumentType, ScaleFactor>(InstrumentType::ALL, ScaleFactor::NONE) },
        { "none", std::pair<InstrumentType, ScaleFactor>(InstrumentType::NONE, ScaleFactor::NONE) },
        { "wall_clock", std::pair<InstrumentType, ScaleFactor>(InstrumentType::WALL_CLOCK_TIMER, ScaleFactor::NONE) },
        { "wall_clock_timer", std::pair<InstrumentType, ScaleFactor>(InstrumentType::WALL_CLOCK_TIMER, ScaleFactor::NONE) },
        { "wall_clock_timer_ms", std::pair<InstrumentType, ScaleFactor>(InstrumentType::WALL_CLOCK_TIMER, ScaleFactor::TIME_MS) },
        { "wall_clock_timer_s", std::pair<InstrumentType, ScaleFactor>(InstrumentType::WALL_CLOCK_TIMER, ScaleFactor::TIME_S) },
        { "pmu", std::pair<InstrumentType, ScaleFactor>(InstrumentType::PMU, ScaleFactor::NONE) },
        { "pmu_k", std::pair<InstrumentType, ScaleFactor>(InstrumentType::PMU, ScaleFactor::SCALE_1K) },
        { "pmu_m", std::pair<InstrumentType, ScaleFactor>(InstrumentType::PMU, ScaleFactor::SCALE_1M) },
        { "pmu_cycles", std::pair<InstrumentType, ScaleFactor>(InstrumentType::PMU_CYCLE_COUNTER, ScaleFactor::NONE) },
        { "pmu_instructions", std::pair<InstrumentType, ScaleFactor>(InstrumentType::PMU_INSTRUCTION_COUNTER, ScaleFactor::NONE) },
        { "mali", std::pair<InstrumentType, ScaleFactor>(InstrumentType::MALI, ScaleFactor::NONE) },
        { "mali_k", std::pair<InstrumentType, ScaleFactor>(InstrumentType::MALI, ScaleFactor::SCALE_1K) },
        { "mali_m", std::pair<InstrumentType, ScaleFactor>(InstrumentType::MALI, ScaleFactor::SCALE_1M) },
        { "opencl_timer", std::pair<InstrumentType, ScaleFactor>(InstrumentType::OPENCL_TIMER, ScaleFactor::NONE) },
        { "opencl_timer_us", std::pair<InstrumentType, ScaleFactor>(InstrumentType::OPENCL_TIMER, ScaleFactor::TIME_US) },
        { "opencl_timer_ms", std::pair<InstrumentType, ScaleFactor>(InstrumentType::OPENCL_TIMER, ScaleFactor::TIME_MS) },
        { "opencl_timer_s", std::pair<InstrumentType, ScaleFactor>(InstrumentType::OPENCL_TIMER, ScaleFactor::TIME_S) },
    };

    try
    {
        return types.at(tolower(name));
    }
    catch(const std::out_of_range &)
    {
        throw std::invalid_argument(name);
    }
}
} // namespace framework
} // namespace test
} // namespace arm_compute
