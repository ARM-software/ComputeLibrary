/*
 * Copyright (c) 2021-2022 Arm Limited.
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

namespace arm_conv {
namespace depthwise {

#if defined(ARM_COMPUTE_ENABLE_SVE)

struct interleave_sve_u8q_3x3_dot
{
  static void pack_parameters(unsigned int, void *, const int32_t *, const uint8_t *, const arm_gemm::Requantize32 &, size_t, size_t);
  static size_t get_packed_size(const DepthwiseArgs &);
};

struct interleave_sve_s8q_3x3_dot
{
  static void pack_parameters(unsigned int, void *, const int32_t *, const int8_t *, const arm_gemm::Requantize32 &, size_t, size_t);
  static size_t get_packed_size(const DepthwiseArgs &);
};

#endif  // defined(ARM_COMPUTE_ENABLE_SVE)

struct interleave_a64_u8q_3x3_dot
{
  static void pack_parameters(unsigned int, void *, const int32_t *, const uint8_t *, const arm_gemm::Requantize32 &, size_t, size_t);
  static size_t get_packed_size(const DepthwiseArgs &);
};

struct interleave_a64_s8q_3x3_dot
{
  static void pack_parameters(unsigned int, void *, const int32_t *, const int8_t *, const arm_gemm::Requantize32 &, size_t, size_t);
  static size_t get_packed_size(const DepthwiseArgs &);
};

}  // namespace depthwise
}  // namespace arm_conv
