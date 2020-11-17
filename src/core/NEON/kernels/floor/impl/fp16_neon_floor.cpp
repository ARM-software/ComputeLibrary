/*
 * Copyright (c) 2020 Arm Limited.
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
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)

#include "src/core/NEON/NEMath.h"
#include "src/core/common/StdTypes.h"
#include "src/core/common/Validate.h"

#include <arm_neon.h>
#include <cmath>
#include <cstddef>

namespace arm_compute
{
namespace cpu
{
constexpr int step = 8;

void fp16_neon_floor(const void *src, void *dst, int len)
{
    ARM_COMPUTE_ASSERT_NOT_NULLPTR(src);
    ARM_COMPUTE_ASSERT_NOT_NULLPTR(dst);
    ARM_COMPUTE_ASSERT(len >= 0);

    auto psrc = static_cast<const f16 *>(src);
    auto pdst = static_cast<f16 *>(dst);

    for(; len >= step; len -= step)
    {
        vst1q_f16(pdst, vfloorq_f16(vld1q_f16(psrc)));
        psrc += step;
        pdst += step;
    }

    for(; len > 0; --len)
    {
        *pdst = std::floor(*psrc);
        ++psrc;
        ++pdst;
    }
}
} // namespace cpu
} // namespace arm_compute
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS) */
