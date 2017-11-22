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
#include "Phase.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<uint8_t> phase(const SimpleTensor<T> &gx, const SimpleTensor<T> &gy, PhaseType phase_type)
{
    const float           PI = std::atan(1) * 4;
    SimpleTensor<uint8_t> phase(gx.shape(), DataType::U8);

    if(phase_type == PhaseType::UNSIGNED) // unsigned: map to [0-255)
    {
        for(int i = 0; i < gx.num_elements(); ++i)
        {
            float angle_deg = (std::atan2(float(gy[i]), float(gx[i])) / PI) * 180.0f;
            phase[i]        = (angle_deg < 0.0f) ? 180.f + angle_deg : angle_deg;
        }
    }
    else // signed: map to [0-180) degrees
    {
        for(int i = 0; i < gx.num_elements(); ++i)
        {
            float angle_pi = std::atan2(gy[i], gx[i]) / PI;
            angle_pi       = (angle_pi < 0.0f) ? 2 + angle_pi : angle_pi;
            phase[i]       = lround(angle_pi * 128) & 0xFFu;
        }
    }

    return phase;
}

template SimpleTensor<uint8_t> phase(const SimpleTensor<int16_t> &gx, const SimpleTensor<int16_t> &gy, PhaseType phase_type);
template SimpleTensor<uint8_t> phase(const SimpleTensor<int32_t> &gx, const SimpleTensor<int32_t> &gy, PhaseType phase_type);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
