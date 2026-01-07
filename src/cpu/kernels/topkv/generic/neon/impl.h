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
#ifndef ACL_SRC_CPU_KERNELS_TOPKV_GENERIC_NEON_IMPL_H
#define ACL_SRC_CPU_KERNELS_TOPKV_GENERIC_NEON_IMPL_H

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Window.h"

#include <limits>
#include <type_traits>

namespace arm_compute
{
namespace cpu
{
namespace detail
{
template <typename ScalarType>
uint32_t count_gt_block(const ScalarType *ptr, ScalarType threshold);

template <typename ScalarType>
void topkv_neon_wrapper(
    const ITensor *predictions, const ITensor *targets, ITensor *out, uint32_t k, const Window &window)
{
    const auto        &pred_info = *predictions->info();
    const unsigned int C         = pred_info.tensor_shape()[0];

    ARM_COMPUTE_ERROR_ON(pred_info.strides_in_bytes()[0] != sizeof(ScalarType));
    ARM_COMPUTE_ERROR_ON(k == 0);

    constexpr unsigned int vec_elems = 16 / sizeof(ScalarType);

    Window win = window;

    Iterator tgt_it(targets, win);
    Iterator out_it(out, win);

    execute_window_loop(
        win,
        [&](const Coordinates &id)
        {
            const int      n = id.x();
            const uint32_t t = *reinterpret_cast<const uint32_t *>(tgt_it.ptr());

            const ScalarType *base =
                reinterpret_cast<const ScalarType *>(predictions->ptr_to_element(Coordinates{0, n}));

            const ScalarType thr = base[t];

            uint32_t     rank = 0;
            unsigned int c    = 0;
            bool         lost = false;

            // Vector loop with early-exit
            for (; c + vec_elems <= C; c += vec_elems)
            {
                rank += count_gt_block<ScalarType>(base + c, thr);
                if (rank >= k)
                {
                    // For large C and small K (e.g. QASYMM8, C=32000, K=3), the probability that the
                    // target index is in the top-K is very low. In most cases, the running rank
                    // reaches >= K after only a small prefix of the tensor (often within the first
                    // few hundred elements). Early-exit stops the scan as soon as rank >= K, avoiding
                    // thousands of unnecessary loads, compares, and reductions. This significantly
                    // reduces runtime (≈10% in validation benchmarks) and matches the
                    // common classification workload where K is small.
                    lost = true;
                    break;
                }
            }

            // Scalar tail with early-exit (only if we haven't already lost)
            if (!lost)
            {
                for (; c < C; ++c)
                {
                    rank += (base[c] > thr) ? 1u : 0u;
                    if (rank >= k)
                    {
                        lost = true;
                        break;
                    }
                }
            }

            *reinterpret_cast<uint8_t *>(out_it.ptr()) = static_cast<uint8_t>(!lost);
        },
        tgt_it, out_it);
}
} // namespace detail
} // namespace cpu
} // namespace arm_compute

#endif // ACL_SRC_CPU_KERNELS_TOPKV_GENERIC_NEON_IMPL_H
