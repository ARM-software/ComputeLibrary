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
#ifndef ARM_COMPUTE_TEST_INSTRUMENTS
#define ARM_COMPUTE_TEST_INSTRUMENTS

#include "MaliCounter.h"
#include "OpenCLTimer.h"
#include "PMUCounter.h"
#include "WallClockTimer.h"

#include <sstream>
#include <string>

namespace arm_compute
{
namespace test
{
namespace framework
{
enum class InstrumentType : unsigned int
{
    ALL                     = ~0U,
    NONE                    = 0,
    WALL_CLOCK_TIMER        = 0x0100,
    PMU                     = 0x0200,
    PMU_CYCLE_COUNTER       = 0x0201,
    PMU_INSTRUCTION_COUNTER = 0x0202,
    MALI                    = 0x0300,
    OPENCL_TIMER            = 0x0400,
};

using InstrumentsDescription = std::pair<InstrumentType, ScaleFactor>;

InstrumentsDescription instrument_type_from_name(const std::string &name);

inline ::std::stringstream &operator>>(::std::stringstream &stream, InstrumentsDescription &instrument)
{
    std::string value;
    stream >> value;
    instrument = instrument_type_from_name(value);
    return stream;
}

inline ::std::stringstream &operator<<(::std::stringstream &stream, InstrumentsDescription instrument)
{
    switch(instrument.first)
    {
        case InstrumentType::WALL_CLOCK_TIMER:
            switch(instrument.second)
            {
                case ScaleFactor::NONE:
                    stream << "WALL_CLOCK_TIMER";
                    break;
                case ScaleFactor::TIME_MS:
                    stream << "WALL_CLOCK_TIMER_MS";
                    break;
                case ScaleFactor::TIME_S:
                    stream << "WALL_CLOCK_TIMER_S";
                    break;
                default:
                    throw std::invalid_argument("Unsupported instrument scale");
            }
            break;
        case InstrumentType::PMU:
            switch(instrument.second)
            {
                case ScaleFactor::NONE:
                    stream << "PMU";
                    break;
                case ScaleFactor::SCALE_1K:
                    stream << "PMU_K";
                    break;
                case ScaleFactor::SCALE_1M:
                    stream << "PMU_M";
                    break;
                default:
                    throw std::invalid_argument("Unsupported instrument scale");
            }
            break;
        case InstrumentType::PMU_CYCLE_COUNTER:
            stream << "PMU_CYCLE_COUNTER";
            break;
        case InstrumentType::PMU_INSTRUCTION_COUNTER:
            stream << "PMU_INSTRUCTION_COUNTER";
            break;
        case InstrumentType::MALI:
            switch(instrument.second)
            {
                case ScaleFactor::NONE:
                    stream << "MALI";
                    break;
                case ScaleFactor::SCALE_1K:
                    stream << "MALI_K";
                    break;
                case ScaleFactor::SCALE_1M:
                    stream << "MALI_M";
                    break;
                default:
                    throw std::invalid_argument("Unsupported instrument scale");
            }
            break;
        case InstrumentType::OPENCL_TIMER:
            switch(instrument.second)
            {
                case ScaleFactor::NONE:
                    stream << "OPENCL_TIMER";
                    break;
                case ScaleFactor::TIME_US:
                    stream << "OPENCL_TIMER_US";
                    break;
                case ScaleFactor::TIME_MS:
                    stream << "OPENCL_TIMER_MS";
                    break;
                case ScaleFactor::TIME_S:
                    stream << "OPENCL_TIMER_S";
                    break;
                default:
                    throw std::invalid_argument("Unsupported instrument scale");
            }
            break;
        case InstrumentType::ALL:
            stream << "ALL";
            break;
        case InstrumentType::NONE:
            stream << "NONE";
            break;
        default:
            throw std::invalid_argument("Unsupported instrument type");
    }

    return stream;
}
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_INSTRUMENTS */
