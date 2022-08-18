/*
 * Copyright (c) 2022 Arm Limited.
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
#pragma once

#include "arm_gemm.hpp"

namespace arm_gemm {

/* Internal enum to define the weight format a kernel is expecting.
 *
 * This is distinct from the "external" WeightFormat defined in arm_gemm.hpp primarily to allow for SVE, where
 * internally kernels are defined in terms of multiples of the SVE vector length, but externally they are converted
 * to a fixed format (based on the VL of the machine we are running on).
 *
 * Encoded as a bitfield:
 *  bit     0 : SVE flag
 *  bit     4 : BF16 convert flag (fast mode)
 *  bits 11-8 : block length (bytes)
 *  bits 15-12: vector count
 */
enum class KernelWeightFormat {
    NON_FIXED        = 0,
    VL128_BL16       = 0x1200,
    VL128_BL32       = 0x1400,
    VL128_BL32_BF16  = 0x1410,
    VL128_BL64       = 0x1800,
    VL256_BL64       = 0x2800,
    VL256_BL64_BF16  = 0x2810,
    VL1VL_BL16       = 0x1201,
    VL1VL_BL32       = 0x1401,
    VL1VL_BL32_BF16  = 0x1411,
    VL1VL_BL64       = 0x1801,
    VL2VL_BL64       = 0x2801,
    VL2VL_BL64_BF16  = 0x2811
};

WeightFormat get_weight_format(const KernelWeightFormat, size_t);

} // namespace arm_gemm
