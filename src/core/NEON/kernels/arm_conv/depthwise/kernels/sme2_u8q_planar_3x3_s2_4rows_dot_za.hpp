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

namespace arm_conv {
namespace depthwise {

void sme2_u8q_planar_3x3_s2_4rows_dot_za_impl(
  const uint8_t *inptr,
  size_t ld_in_row,
  size_t ld_in_col,
  size_t ld_in_vl,
  unsigned int pad_top,
  unsigned int valid_input_rows,
  unsigned int pad_left,
  unsigned int valid_input_cols,
  const uint8_t *weights,
  uint8_t **outptrs,
  const size_t *outlds,
  const size_t *outvllds,
  unsigned int output_cols,
  unsigned int start_channel,
  unsigned int valid_channels,
  const arm_gemm::Requantize32 &qp
);

class sme2_u8q_planar_3x3_s2_4rows_dot_za : public PlanarStrategy<uint8_t, uint8_t>
{
  using Parent = PlanarStrategy<uint8_t, uint8_t>;

  public:
  using return_type = uint8_t;
  constexpr static auto output_rows = 4u;
  constexpr static auto kernel_rows = 3u, kernel_cols = 3u;
  constexpr static auto stride_rows = 2u, stride_cols = 2u;
  constexpr static auto vl_type = arm_gemm::VLType::SME;

  sme2_u8q_planar_3x3_s2_4rows_dot_za(const CPUInfo *)
  : Parent(kernel_rows, kernel_cols, stride_rows, stride_cols, output_rows, vl_type)
  {
  }

  typename Parent::KernelType get_kernel(void) const override
  {
    return sme2_u8q_planar_3x3_s2_4rows_dot_za_impl;
  }
};

}  // namespace depthwise
}  // namespace arm_conv
