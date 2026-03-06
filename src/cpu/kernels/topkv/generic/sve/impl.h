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
#ifndef ACL_SRC_CPU_KERNELS_TOPKV_GENERIC_SVE_IMPL_H
#define ACL_SRC_CPU_KERNELS_TOPKV_GENERIC_SVE_IMPL_H

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"

#include <cstdint>
#include <cstring>

namespace arm_compute
{
namespace cpu
{
namespace detail
{

/*
 * Type-specific hooks (declared here, defined in each cpp).
 *
 * - vector_length<Scalar>()
 *     Return the SVE vector length in elements for Scalar (no clamping).
 *
 * - count_gt_block<Scalar>(ptr, thr, block_elems)
 *     Count how many elements in [ptr, ptr + block_elems) are > thr.
 *     Tail-safe via predicate. block_elems is always <= vector_length<Scalar>().
 *
 t contains the SVE intrinsics
 * (e.g., qasymm8.cpp, qasymm8_signed.cpp, fp16.cpp, fp32.cpp, integer.cpp).
 */

template <typename Scalar>
uint32_t vector_length();

template <typename Scalar>
uint32_t count_gt_block(const Scalar *ptr, Scalar thr, uint32_t block_elems);

// ----------------------------------------------------------------------------
// Generic wrapper (type-agnostic) - uses the above hooks.
// Semantics (matching TopKV tests you showed):
//   - predictions is N x C
//   - window iterates across output elements (classes) => id.x() == class index c
//   - for each class c, targets[c] gives the sample index t
//   - scan across N samples and compute rank (#samples with value > predictions[t])
//   - output is U8 boolean: (rank < k)
// ----------------------------------------------------------------------------
template <typename Scalar>
inline void
topkv_sve_wrapper(const ITensor *predictions, const ITensor *targets, ITensor *out, uint32_t k, const Window &window)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(predictions, targets, out);
    ARM_COMPUTE_ERROR_ON(k == 0);

    const ITensorInfo *pred_info = predictions->info();
    const uint32_t     N         = pred_info->dimension(0); // samples
    const uint32_t     C         = pred_info->dimension(1); // classes

    const uint32_t vl = vector_length<Scalar>(); // cache once per kernel invocation

    Iterator tgt_it(targets, window);
    Iterator out_it(out, window);

    execute_window_loop(
        window,
        [&](const Coordinates &id)
        {
            const uint32_t c = static_cast<uint32_t>(id.x()); // class index
            ARM_COMPUTE_ERROR_ON(c >= C);

            uint32_t t = {*reinterpret_cast<uint32_t *>(tgt_it.ptr())};
            ARM_COMPUTE_ERROR_ON(t >= N);

            const Scalar *col_ptr = reinterpret_cast<const Scalar *>(predictions->ptr_to_element(Coordinates(0, c)));
            ARM_COMPUTE_ERROR_ON(col_ptr == nullptr);

            const Scalar thr = col_ptr[t];

            uint32_t rank = 0;
            uint32_t idx  = 0;

            while (idx < N)
            {
                const uint32_t remaining = N - idx;
                const uint32_t bw        = (remaining < vl) ? remaining : vl;

                rank += count_gt_block<Scalar>(col_ptr + idx, thr, bw);

                if (rank >= k)
                {
                    break;
                }

                idx += bw;
            }

            *reinterpret_cast<uint8_t *>(out_it.ptr()) = static_cast<uint8_t>(rank < k);
        },
        tgt_it, out_it);
}

} // namespace detail
} // namespace cpu
} // namespace arm_compute

#endif // ACL_SRC_CPU_KERNELS_TOPKV_GENERIC_SVE_IMPL_H
