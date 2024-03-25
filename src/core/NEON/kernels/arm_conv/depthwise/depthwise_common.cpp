/*
 * Copyright (c) 2023 Arm Limited.
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

#include "depthwise_common.hpp"

#include "utils.hpp"

using arm_gemm::iceildiv;

namespace arm_conv {
namespace depthwise {

std::tuple<size_t, size_t, size_t, size_t, size_t>
get_reduced_view_for_dilation(size_t out_size, size_t in_size, const size_t d,
                              const size_t dilation_factor,
                              const size_t kernel_size, const size_t stride,
                              const size_t orig_pad_before) {
    // Get the valid output range
    out_size = iceildiv(out_size - d, dilation_factor);

    // Compute the start offset and the amount of padding which applies to this
    // portion of the work.
    size_t start_pos = d * stride, pad_before = 0;
    if (start_pos < orig_pad_before) {
        pad_before = iceildiv(orig_pad_before - start_pos, dilation_factor);
    }
    start_pos += pad_before * dilation_factor - orig_pad_before;

    // Hence compute the valid input range
    in_size = start_pos < in_size
                  ? iceildiv(in_size - start_pos, dilation_factor)
                  : 0;

    // Finally, compute the "after" padding
    const size_t reqd_input = (out_size - 1) * stride + kernel_size;
    size_t pad_after = 0;
    if (reqd_input > (pad_before + in_size)) {
        pad_after = reqd_input - (pad_before + in_size);
    }

    return std::make_tuple(out_size, in_size, start_pos, pad_before, pad_after);
}

}  // namespace depthwise
}  // namespace arm_conv
