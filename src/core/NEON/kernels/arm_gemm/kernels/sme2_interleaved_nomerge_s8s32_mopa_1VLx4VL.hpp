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

#ifdef __aarch64__

#include <cstdint>
#include "../std_transforms_sme.hpp"

namespace arm_gemm
{

// Implementations
void sme2_interleaved_nomerge_s8s32_mopa_1VLx4VL(const int8_t *const A, const int8_t *const B, int32_t *const C, int ldc, const int M, const int N, const int K, const int32_t *const bias, const Activation act, bool accumulate, int32_t *const accumulator_buffer);

class cls_sme2_interleaved_nomerge_s8s32_mopa_1VLx4VL
{
public:
  typedef int8_t operand_type;
  typedef int32_t result_type;

  typedef void (*kern_type)(const int8_t *const A, const int8_t *const B, int32_t *const C, int ldc, const int M, const int N, const int K, const int32_t *const bias, const Activation act, bool accumulate, int32_t *const accumulator_buffer);

  /* Kernel blocking parameters */
  static unsigned int out_height()
  {
    return sme::get_vector_length<int32_t>() * 1;
  }

  static unsigned int out_width()
  {
    return sme::get_vector_length<int32_t>() * 4;
  }

  static constexpr unsigned int k_unroll()
  {
    return 4;
  }

  static constexpr bool supports_accumulate()
  {
    return true;
  }

  static constexpr bool supports_bias()
  {
    return true;
  }

  static constexpr bool supports_activation()
  {
    return false;
  }

  static constexpr bool is_sme()
  {
    return true;
  }

  // Default to the generic kernel
  kern_type kernel = sme2_interleaved_nomerge_s8s32_mopa_1VLx4VL;

  StdTransformsSME<operand_type, result_type, 1, 4, 4> transforms = {};

  cls_sme2_interleaved_nomerge_s8s32_mopa_1VLx4VL(const CPUInfo *ci)
  {
    ARM_COMPUTE_UNUSED(ci);
  }
};

} // namespace arm_gemm

#endif // __aarch64__
