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
InstrumentType instrument_type_from_name(const std::string &name)
{
    static const std::map<std::string, InstrumentType> types =
    {
        { "all", InstrumentType::ALL },
        { "none", InstrumentType::NONE },
        { "wall_clock", InstrumentType::WALL_CLOCK_TIMER },
        { "pmu", InstrumentType::PMU },
        { "pmu_cycles", InstrumentType::PMU_CYCLE_COUNTER },
        { "pmu_instructions", InstrumentType::PMU_INSTRUCTION_COUNTER },
        { "mali", InstrumentType::MALI },
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
