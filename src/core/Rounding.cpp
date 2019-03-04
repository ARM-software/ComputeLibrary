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

#include "arm_compute/core/Rounding.h"

#include "arm_compute/core/Error.h"
#include "support/ToolchainSupport.h"

#include <cmath>

using namespace arm_compute;
using namespace std;

int arm_compute::round(float x, RoundingPolicy rounding_policy)
{
    using namespace std;
    int rounded = 0;
    switch(rounding_policy)
    {
        case RoundingPolicy::TO_ZERO:
        {
            rounded = static_cast<int>(x);
            break;
        }
        case RoundingPolicy::TO_NEAREST_UP:
        {
            rounded = static_cast<int>(support::cpp11::round(x));
            break;
        }
        case RoundingPolicy::TO_NEAREST_EVEN:
        {
#ifdef __aarch64__
            asm("fcvtns %x[res], %s[value]"
                : [res] "=r"(rounded)
                : [value] "w"(x));
#else  // __aarch64__
            ARM_COMPUTE_ERROR("TO_NEAREST_EVEN rounding policy is not supported.");
#endif // __aarch64__
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Unsupported rounding policy.");
            break;
        }
    }

    return rounded;
}
