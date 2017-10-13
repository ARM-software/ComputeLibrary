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
    const float pi           = std::atan(1) * 4;
    const float rad_to_deg   = 180.0f / pi;
    const float scale_factor = 128.f / 180.f;
    const float epsilon      = 1e-9f; // used to avoid division by zero

    const float ninety      = scale_factor * 90.f;
    const float one_eighty  = scale_factor * 180.f;
    const float two_seventy = scale_factor * 270.f;

    // unsigned: map to [0-255)
    // signed:   map to [0-180) degrees
    const float scale = (phase_type == PhaseType::UNSIGNED) ? rad_to_deg : rad_to_deg * scale_factor;

    SimpleTensor<uint8_t> phase(gx.shape(), DataType::U8);

    for(int i = 0; i < gx.num_elements(); ++i)
    {
        bool quad_two   = std::signbit(gx[i]) && !std::signbit(gy[i]);
        bool quad_three = std::signbit(gx[i]) && std::signbit(gy[i]);
        bool quad_four  = !std::signbit(gx[i]) && std::signbit(gy[i]);

        float  x      = gy[i] / (gx[i] + epsilon);
        double arctan = std::atan(x);

        const bool is_negative = std::signbit(arctan);

        // Radians to degrees conversion with applied scale factor
        arctan = arctan * scale;

        if(phase_type == PhaseType::UNSIGNED)
        {
            arctan = is_negative ? arctan + 180.f : arctan;
        }
        else
        {
            arctan = is_negative ? arctan + ninety : arctan;

            // Choose correct quandrant
            arctan = quad_two ? ninety + arctan : arctan;
            arctan = quad_three ? one_eighty + arctan : arctan;
            arctan = quad_four ? two_seventy + arctan : arctan;
        }

        phase[i] = saturate_cast<uint8_t>(arctan + 0.5f);
    }

    return phase;
}

template SimpleTensor<uint8_t> phase(const SimpleTensor<int16_t> &gx, const SimpleTensor<int16_t> &gy, PhaseType phase_type);
template SimpleTensor<uint8_t> phase(const SimpleTensor<int32_t> &gx, const SimpleTensor<int32_t> &gy, PhaseType phase_type);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
