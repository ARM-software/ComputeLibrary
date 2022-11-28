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

void sme2_fp32bf16fp32_planar_5x5_s1_4rows_dot_za_impl(
  const float *inptr,
  size_t ld_in_row,
  size_t ld_in_col,
  size_t ld_in_vl,
  unsigned int pad_top,
  unsigned int valid_input_rows,
  unsigned int pad_left,
  unsigned int valid_input_cols,
  const float *weights,
  const float *bias,
  float **outptrs,
  const size_t *outlds,
  const size_t *outvllds,
  unsigned int output_cols,
  unsigned int start_channel,
  unsigned int valid_channels,
  float act_min,
  float act_max
);

class sme2_fp32bf16fp32_planar_5x5_s1_4rows_dot_za : public PlanarStrategy<float, float>
{
  using Parent = PlanarStrategy<float, float>;

  public:
  using return_type = float;
  constexpr static auto output_rows = 4u;
  constexpr static auto kernel_rows = 5u, kernel_cols = 5u;
  constexpr static auto stride_rows = 1u, stride_cols = 1u;
  constexpr static auto vl_type = arm_gemm::VLType::SME;

  sme2_fp32bf16fp32_planar_5x5_s1_4rows_dot_za(const CPUInfo *)
  : Parent(kernel_rows, kernel_cols, stride_rows, stride_cols, output_rows, vl_type)
  {
  }

  typename Parent::KernelType get_kernel(void) const override
  {
    return sme2_fp32bf16fp32_planar_5x5_s1_4rows_dot_za_impl;
  }
};

}  // namespace depthwise
}  // namespace arm_conv
