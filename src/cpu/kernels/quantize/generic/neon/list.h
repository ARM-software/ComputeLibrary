/*
 * Copyright (c) 2024 Arm Limited.
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
#ifndef ACL_SRC_CPU_KERNELS_QUANTIZE_GENERIC_NEON_LIST_H
#define ACL_SRC_CPU_KERNELS_QUANTIZE_GENERIC_NEON_LIST_H

#include "arm_compute/core/Helpers.h"

namespace arm_compute
{
namespace cpu
{

#define DECLARE_QUANTIZE_KERNEL(func_name) void func_name(const ITensor *src, ITensor *dst, const Window &window)

DECLARE_QUANTIZE_KERNEL(u8_u8_run_quantize_qasymm8);
DECLARE_QUANTIZE_KERNEL(u8_i8_run_quantize_qasymm8);
DECLARE_QUANTIZE_KERNEL(i8_u8_run_quantize_qasymm8);
DECLARE_QUANTIZE_KERNEL(i8_i8_run_quantize_qasymm8);

DECLARE_QUANTIZE_KERNEL(u8_u8_run_requantize_offset_only);
DECLARE_QUANTIZE_KERNEL(u8_i8_run_requantize_offset_only);
DECLARE_QUANTIZE_KERNEL(i8_u8_run_requantize_offset_only);
DECLARE_QUANTIZE_KERNEL(i8_i8_run_requantize_offset_only);

DECLARE_QUANTIZE_KERNEL(i8_u8_run_requantize_offset_only_convert);
DECLARE_QUANTIZE_KERNEL(u8_i8_run_requantize_offset_only_convert);

DECLARE_QUANTIZE_KERNEL(u8_run_quantize_qasymm16);
DECLARE_QUANTIZE_KERNEL(i8_run_quantize_qasymm16);

DECLARE_QUANTIZE_KERNEL(fp32_u8_run_quantize_qasymm8);
DECLARE_QUANTIZE_KERNEL(fp32_i8_run_quantize_qasymm8);
DECLARE_QUANTIZE_KERNEL(fp32_run_quantize_qasymm16);

DECLARE_QUANTIZE_KERNEL(fp32_i8_run_quantize_qsymm8);

DECLARE_QUANTIZE_KERNEL(fp16_u8_run_quantize_qasymm8);
DECLARE_QUANTIZE_KERNEL(fp16_i8_run_quantize_qasymm8);
DECLARE_QUANTIZE_KERNEL(fp16_run_quantize_qasymm16);

#undef DECLARE_QUANTIZE_KERNEL

} // namespace cpu
} // namespace arm_compute
#endif // ACL_SRC_CPU_KERNELS_QUANTIZE_GENERIC_NEON_LIST_H
