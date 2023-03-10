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
#pragma once

#ifdef ARM_COMPUTE_ENABLE_SVE


#include "../std_transforms_sme.hpp"

namespace arm_gemm
{

// Implementations
void sme2_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL(const __fp16 *const A, const __fp16 *const B, __fp16 *const C, int ldc, const int M, const int N, const int K, const __fp16 *const bias, const Activation act, bool accumulate, float *const accumulator_buffer);

class cls_sme2_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL
{
public:
  typedef __fp16 operand_type;
  typedef __fp16 result_type;

  typedef void (*kern_type)(const __fp16 *const A, const __fp16 *const B, __fp16 *const C, int ldc, const int M, const int N, const int K, const __fp16 *const bias, const Activation act, bool accumulate, float *const accumulator_buffer);

  /* Kernel blocking parameters */
  static unsigned int out_height()
  {
    return sme::get_vector_length<float>() * 2;
  }

  static unsigned int out_width()
  {
    return sme::get_vector_length<float>() * 2;
  }

  static constexpr unsigned int k_unroll()
  {
    return 2;
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
    return true;
  }

  static constexpr bool is_sme()
  {
    return true;
  }

  // Default to the generic kernel
  kern_type kernel = sme2_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL;

  StdTransformsSME<operand_type, result_type, 2, 2, 2> transforms = {};

  cls_sme2_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL(const CPUInfo *)
  {
  }
};

} // namespace arm_gemm

#endif // ARM_COMPUTE_ENABLE_SVE
