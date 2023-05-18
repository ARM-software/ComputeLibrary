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

#ifndef SRC_CORE_NEON_KERNELS_LUT_LIST_H
#define SRC_CORE_NEON_KERNELS_LUT_LIST_H

#include <cstddef>
#include <cstdint>

namespace arm_compute
{
namespace cpu
{

#ifdef __aarch64__
#define DECLARE_LUT_KERNEL(func_name) \
    void func_name( \
        const uint8_t        *table, \
        size_t                num_strings, \
        size_t                string_length, \
        const uint8_t *const *input, \
        uint8_t *const       *output)

DECLARE_LUT_KERNEL(lut_u8_neon);
DECLARE_LUT_KERNEL(lut_u8_sve2);

#undef DECLARE_LUT_KERNEL
#endif // __aarch64__

} // namespace cpu
} // namespace arm_compute

#endif // SRC_CORE_NEON_KERNELS_LUT_LIST_H
