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
#ifndef SRC_CPU_KERNELS_ADDMULADD_LIST
#define SRC_CPU_KERNELS_ADDMULADD_LIST

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"

namespace arm_compute
{
namespace cpu
{
#define DECLARE_ADD_MUL_ADD_KERNEL(func_name)                                                                          \
    void func_name(const ITensor *input1, const ITensor *input2, const ITensor *bn_mul, const ITensor *bn_add, \
                   ITensor *add_output, ITensor *final_output, ConvertPolicy policy, const ActivationLayerInfo &act_info, const Window &window)

DECLARE_ADD_MUL_ADD_KERNEL(add_mul_add_fp32_neon);
DECLARE_ADD_MUL_ADD_KERNEL(add_mul_add_fp16_neon);
DECLARE_ADD_MUL_ADD_KERNEL(add_mul_add_u8_neon);
DECLARE_ADD_MUL_ADD_KERNEL(add_mul_add_s8_neon);

#undef DECLARE_ADD_MUL_ADD_KERNEL

} // namespace cpu
} // namespace arm_compute
#endif /* SRC_CPU_KERNELS_ADDMULADD_LIST */
