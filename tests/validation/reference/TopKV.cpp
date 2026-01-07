/*
 * Copyright (c) 2026 Arm Limited.
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
#include "tests/validation/reference/TopKV.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/CoreTypes.h"
#include "arm_compute/core/TensorShape.h"

#include "tests/SimpleTensor.h"

#include <cstdint>
#include <limits>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{

template <typename T>
SimpleTensor<uint8_t> topkv(SimpleTensor<T> &predictions, SimpleTensor<uint32_t> &targets, uint32_t k)
{
    const TensorShape &ps = predictions.shape();
    const int          C  = ps[0]; // classes
    const int          N  = ps[1]; // batch

    SimpleTensor<uint8_t> expected(TensorShape(N), DataType::U8);

    const float eps = std::numeric_limits<float>::epsilon();

    for (int i = 0; i < N; ++i)
    {
        // targets[i] (U32)
        const uint32_t target_class = targets[i];

        // Read predictions[target_class, i] as T, then promote to float
        const T     target_t   = *reinterpret_cast<const T *>(predictions(Coordinates{target_class, i}));
        const float target_val = static_cast<float>(target_t);

        unsigned int rank = 0;
        for (int c = 0; c < C; ++c)
        {
            const T     vt = *reinterpret_cast<const T *>(predictions(Coordinates{c, i}));
            const float v  = static_cast<float>(vt);

            if ((v - target_val) > eps)
            {
                ++rank;
            }
        }

        expected[i] = static_cast<uint8_t>(rank < k);
    }

    return expected;
}

template SimpleTensor<uint8_t> topkv<float>(SimpleTensor<float> &, SimpleTensor<uint32_t> &, uint32_t);
template SimpleTensor<uint8_t> topkv<half>(SimpleTensor<half> &, SimpleTensor<uint32_t> &, uint32_t);
template SimpleTensor<uint8_t> topkv<uint8_t>(SimpleTensor<uint8_t> &, SimpleTensor<uint32_t> &, uint32_t);
template SimpleTensor<uint8_t> topkv<int8_t>(SimpleTensor<int8_t> &, SimpleTensor<uint32_t> &, uint32_t);
template SimpleTensor<uint8_t> topkv<int32_t>(SimpleTensor<int32_t> &, SimpleTensor<uint32_t> &, uint32_t);

} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
