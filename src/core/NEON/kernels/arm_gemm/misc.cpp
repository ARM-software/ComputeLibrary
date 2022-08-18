/*
 * Copyright (c) 2017-2018, 2022 Arm Limited.
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

#ifndef NO_MULTI_THREADING
#include <mutex>
#endif
#include <cstdint>

#include "arm_gemm.hpp"
#include "kernel_weight_format.hpp"
#include "utils.hpp"

namespace arm_gemm {

#ifndef NO_MULTI_THREADING
std::mutex report_mutex;
#endif

WeightFormat get_weight_format(const KernelWeightFormat kwf, size_t element_size) {
    if (kwf==KernelWeightFormat::NON_FIXED) {
        return WeightFormat::UNSPECIFIED;
    }

    uint32_t kwf_i = static_cast<uint32_t>(kwf);
    uint32_t wf_i = 0;

    const auto block_bytes = (kwf_i >> 8) & 0xf;
    const auto vector_count = (kwf_i >> 12) & 0xf;

    uint32_t vector_bytes;

    // For fast mode BF16 kernels set the appropriate bit and override element size to 2.
    if (kwf_i & 0x10) {
        element_size = 2;
        wf_i |= 0x10;
    }

    // Get total bytes in vector output
    if (kwf_i & 0x1) {
        vector_bytes = vector_count * get_vector_length<uint8_t>();
    } else {
        vector_bytes = vector_count * 16;
    }

    auto input_blocking = block_bytes / element_size;
    auto output_blocking = vector_bytes / block_bytes;

    wf_i |= (input_blocking << 20);
    wf_i |= (output_blocking << 8);

    return static_cast<WeightFormat>(wf_i);
}

} // namespace arm_gemm
