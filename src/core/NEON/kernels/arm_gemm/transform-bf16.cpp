/*
 * Copyright (c) 2021-2026 Arm Limited.
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

#include "arm_common/internal/utils.hpp"
#include "src/core/NEON/wrapper/intrinsics/intrinsics.h"
#include "arm_common/bfloat.hpp"

#if !defined(_WIN64) && !defined(__OpenBSD__)
#include <alloca.h>
#endif /* !defined(_WIN64) && !defined(__OpenBSD__) */

namespace arm_gemm {

#ifdef ARM_COMPUTE_ENABLE_BF16
template <unsigned int tIntBy, unsigned int BlockBy, bool Transposed, size_t TOutSize, size_t TInSize, VLType vlt>
struct TransformImpl {
    /*
    * Generic BF16 transform.
    *
    * Assuming the untransposed case, this works by first reading <BlockBy>
    * consecutive values from the first input row.  This same number of values
    * are then read from the next <IntBy-1> rows.  Now return to the first
    * input row and repeat.
    *
    * Need to cope with the work requested in either dimension not actually
    * being a multiple of the block sizes.
    */
    static void Transform(bfloat16* out, const float* const in, const int stride,
        const int y0, const int ymax, const int x0, const int xmax) {
        // NOTE: This code is disabled to avoid the call to get_vector_length(), so templated transforms will not be
        // correct for SVE.  This is not an issue as we have specializations for all SVE cases.
        // For SVE cases we multiply the interleave factor by the vector length.
        // const unsigned int IntBy = tIntBy * (vlt == VLType::SVE ? get_vector_length<bfloat16>() / BlockBy : 1);
        const unsigned int IntBy = tIntBy;
        const int n_whole_y_blocks = (ymax - y0) / IntBy;
        const int y_remainders = (ymax - y0) % IntBy;
        const int n_y_blocks = n_whole_y_blocks + (y_remainders ? 1 : 0);

        const int n_whole_x_blocks = (xmax - x0) / BlockBy;
        const int x_remainders = (xmax - x0) % BlockBy;
        const int n_x_blocks = n_whole_x_blocks + (x_remainders ? 1 : 0);
        // "Y" loop: advance down the rows of the source IntBy rows at a time.
        // Set up fill_rows to show the number rows to copy from, and blank_rows
        // for the number of blank rows to add.
        for (int y_block=0 ; y_block < n_y_blocks; y_block++) {
            int fill_rows = (y_block < n_whole_y_blocks) ? IntBy : y_remainders;
            int blank_rows = IntBy - fill_rows;

            int y_base = y0 + (y_block * IntBy);

            // So now advance along this block of rows, BlockBy columns at a time.
            for (int x_block=0 ; x_block < n_x_blocks; x_block++) {
                int fill_cols = (x_block < n_whole_x_blocks) ? BlockBy : x_remainders;
                int blank_cols = BlockBy - fill_cols;

                int x_base = x0 + (x_block * BlockBy);

                for (int row = 0; row < fill_rows; row++) {
                    int full_vecs = fill_cols / 8;
                    int tail = fill_cols % 8;
                    int col = 0;
                    // Use neon vectors for f32->bf16 conversion
                    for (int i = 0; i < full_vecs; i++) {
                        if (Transposed) {
                            const float v[8] = {
                                in[(x_base + col) * stride + y_base + row],
                                in[(x_base + col + 1) * stride + y_base + row],
                                in[(x_base + col + 2) * stride + y_base + row],
                                in[(x_base + col + 3) * stride + y_base + row],
                                in[(x_base + col + 4) * stride + y_base + row],
                                in[(x_base + col + 5) * stride + y_base + row],
                                in[(x_base + col + 6) * stride + y_base + row],
                                in[(x_base + col + 7) * stride + y_base + row]
                            };
                            arm_compute::wrapper::vcvt_bf16_f32(v, reinterpret_cast<uint16_t *>(out));
                        } else {
                            const float * v = &in[(y_base + row) * stride + x_base + col];
                            arm_compute::wrapper::vcvt_bf16_f32(v, reinterpret_cast<uint16_t *>(out));
                        }
                        out += 8;
                        col += 8;
                    }
                    // Tail loop for vectorized load
                    for (int i = 0; i < tail; i++) {
                        if (Transposed) {
                            *out++ = static_cast<bfloat16>(in[(x_base + col) * stride + y_base + row]);
                        } else {
                            *out++ = static_cast<bfloat16>(in[(y_base + row) * stride + x_base + col]);
                        }
                        col++;
                    }
                    // "col" tail - row is in range but column is out of range.
                    for (int col=0; col < blank_cols; col++) {
                        *out++ = static_cast<bfloat16>(0);
                    }
                }
                // "row" tail - row is out of range so fill with zeros always.
                bfloat16 zeroval = static_cast<bfloat16>(0);
                int pads = blank_rows * (fill_cols + blank_cols);

                for (int i=0; i<pads; i++) {
                    out[i] = zeroval;
                }

                out += pads;
            }
        }
    }
};

/*****************************************************************************/
template <unsigned int IntBy, unsigned int BlockBy, bool Transposed, VLType vlt=VLType::None, typename TOut, typename TIn>
void Transform(
  TOut* out, const TIn* const in, const int stride,
  const int k0, const int kmax, const int x0, const int xmax
) {
  // Redirect to a specialised implementation predicated on argument size.
  TransformImpl<IntBy, BlockBy, Transposed, sizeof(TOut), sizeof(TIn), vlt>::Transform(
    out, in, stride, k0, kmax, x0, xmax
  );
}
/*****************************************************************************/

template void Transform<4, 4, false, VLType::None>(bfloat16 *, const float *, int, int, int, int, int);
template void Transform<8, 4, false, VLType::None>(bfloat16 *, const float *, int, int, int, int, int);
template void Transform<8, 4, true, VLType::None>(bfloat16 *, const float *, int, int, int, int, int);
#endif // ARM_COMPUTE_ENABLE_BF16

} // namespace arm_gemm
