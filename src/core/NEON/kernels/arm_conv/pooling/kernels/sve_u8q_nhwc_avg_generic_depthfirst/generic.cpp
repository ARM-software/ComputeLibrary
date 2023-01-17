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

#include "pooling.hpp"
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>


#if defined(ARM_COMPUTE_ENABLE_SVE)

namespace arm_conv {
namespace pooling {

namespace {
  struct RescaleParams
  {
    int32_t multiplier, shift;
  };

  constexpr RescaleParams rescale_params[8] = {
    {0x40000000, -0},  // 1/2
    {0x55555556, -1},  // 1/3
    {0x40000000, -1},  // 1/4
    {0x66666666, -2},  // 1/5
    {0x55555556, -2},  // 1/6
    {0x49249249, -2},  // 1/7
    {0x40000000, -2},  // 1/8
    {0x71c71c72, -3},  // 1/9
  };
}

void sve_u8q_nhwc_avg_generic_depthfirst_impl(
  const uint64_t window_cells,
  const uint64_t n_valid_cells,
  uint64_t n_channels,
  const uint8_t *const *const inptrs,
  uint8_t *outptr,
  const Requantize32 &qp
)
{
  if (n_valid_cells == 1 && window_cells == 1)
  {
    // In this case, simply copy from the input to the output
    std::memcpy(outptr, *inptrs, n_channels);
    return;
  }

  // Compute (or look up) the rescale values
  int32_t shift_value = 0, rescale_value = 0;
  if (2 <= window_cells && window_cells <= 9)
  {
    auto &params = rescale_params[window_cells - 2];
    rescale_value = params.multiplier;
    shift_value = params.shift;
  }
  else
  {
    auto f_rescale_value = 1.0f / static_cast<float>(window_cells);

    shift_value = 0;
    while (f_rescale_value < 0.5f)
    {
      shift_value--;
      f_rescale_value *= 2.0f;
    }

    int64_t long_rescale_value = round(f_rescale_value * static_cast<float>(1ll << 31));
    if (long_rescale_value == (1ll << 31))
    {
      shift_value++;
      long_rescale_value >>= 1;
    }
    rescale_value = static_cast<int32_t>(long_rescale_value);
  }


  // Initialise the accumulators such that the offsets are subtracted for all
  // valid inputs.
  const int32_t accumulator_init = -qp.input_offset * n_valid_cells;

  // Combine together the rescale value for the requantization and the scaling
  // factor for the average pool.
  const int32_t shift = qp.per_layer_left_shift - qp.per_layer_right_shift + shift_value;
  const int32_t left_shift = shift > 0 ? shift : 0;
  const int32_t right_shift = shift <= 0 ? shift : 0;

  int32_t combined_rescale_value = 0;
  __asm__ __volatile__ (
      "mov v16.s[0], %w[per_layer_mul]\n"
      "mov v17.s[0], %w[rescale_value]\n"
      "sqrdmulh s18, s16, s17\n"
      "mov %w[combined_rescale_value], v18.s[0]\n"
    : [combined_rescale_value] "=r" (combined_rescale_value)
    : [per_layer_mul] "r" (qp.per_layer_mul), [rescale_value] "r" (rescale_value)
    : "v16", "v17", "v18"
  );

  __asm__ __volatile__(
    "mov x26, #0x0\n"
    "cntb x25\n"
    "cntb x24, ALL, MUL #2\n"
    "cntb x23, ALL, MUL #3\n"
    "whilelt p4.b, x26, %x[n_channels]\n"
    "whilelt p3.b, x25, %x[n_channels]\n"
    "whilelt p2.b, x24, %x[n_channels]\n"
    "whilelt p1.b, x23, %x[n_channels]\n"
    "ptrue p0.b\n"
    "b.none 7f\n"
    "1:"  // 4-vectors of channels
    "ld1rw { z15.s }, p0/Z, [%x[accumulator_init]]\n"
    "lsr x22, %x[n_valid_cells], #0x1\n"
    "mov z14.d, z15.d\n"
    "mov z13.d, z15.d\n"
    "mov z12.d, z15.d\n"
    "mov z11.d, z15.d\n"
    "mov x19, %x[inptrs]\n"
    "mov z10.d, z15.d\n"
    "mov z9.d, z15.d\n"
    "mov z8.d, z15.d\n"
    "mov z7.d, z15.d\n"
    "mov z6.d, z15.d\n"
    "mov z5.d, z15.d\n"
    "mov z4.d, z15.d\n"
    "mov z3.d, z15.d\n"
    "mov z2.d, z15.d\n"
    "mov z1.d, z15.d\n"
    "mov z0.d, z15.d\n"
    "cbz x22, 4f\n"
    "ldp x21, x20, [x19, #0x0]\n"
    "subs x22, x22, #0x1\n"
    "add x19, x19, #0x10\n"
    "ld1b { z31.b }, p4/Z, [x21, x26]\n"
    "ld1b { z30.b }, p4/Z, [x20, x26]\n"
    "ld1b { z29.b }, p3/Z, [x21, x25]\n"
    "ld1b { z28.b }, p3/Z, [x20, x25]\n"
    "ld1b { z27.b }, p2/Z, [x21, x24]\n"
    "ld1b { z26.b }, p2/Z, [x20, x24]\n"
    "ld1b { z25.b }, p1/Z, [x21, x23]\n"
    "ld1b { z24.b }, p1/Z, [x20, x23]\n"
    "beq 3f\n"
    "2:"  // 4-vectors of channels: 2 inputs loop
    ".inst 0x455e0bf7  // uaddlb z23.h, z31.b, z30.b\n"
    ".inst 0x455e0ff6  // uaddlt z22.h, z31.b, z30.b\n"
    "ldp x21, x20, [x19, #0x0]\n"
    "subs x22, x22, #0x1\n"
    ".inst 0x455c0bb5  // uaddlb z21.h, z29.b, z28.b\n"
    ".inst 0x455c0fb4  // uaddlt z20.h, z29.b, z28.b\n"
    "add x19, x19, #0x10\n"
    "ld1b { z31.b }, p4/Z, [x21, x26]\n"
    ".inst 0x455a0b73  // uaddlb z19.h, z27.b, z26.b\n"
    ".inst 0x455a0f72  // uaddlt z18.h, z27.b, z26.b\n"
    "ld1b { z30.b }, p4/Z, [x20, x26]\n"
    "ld1b { z29.b }, p3/Z, [x21, x25]\n"
    ".inst 0x45580b31  // uaddlb z17.h, z25.b, z24.b\n"
    ".inst 0x45580f30  // uaddlt z16.h, z25.b, z24.b\n"
    "ld1b { z28.b }, p3/Z, [x20, x25]\n"
    "ld1b { z27.b }, p2/Z, [x21, x24]\n"
    ".inst 0x459749ef  // uaddwb z15.s, z15.s, z23.h\n"
    ".inst 0x45974dce  // uaddwt z14.s, z14.s, z23.h\n"
    "ld1b { z26.b }, p2/Z, [x20, x24]\n"
    "ld1b { z25.b }, p1/Z, [x21, x23]\n"
    ".inst 0x459649ad  // uaddwb z13.s, z13.s, z22.h\n"
    ".inst 0x45964d8c  // uaddwt z12.s, z12.s, z22.h\n"
    "ld1b { z24.b }, p1/Z, [x20, x23]\n"
    ".inst 0x4595496b  // uaddwb z11.s, z11.s, z21.h\n"
    ".inst 0x45954d4a  // uaddwt z10.s, z10.s, z21.h\n"
    ".inst 0x45944929  // uaddwb z9.s, z9.s, z20.h\n"
    ".inst 0x45944d08  // uaddwt z8.s, z8.s, z20.h\n"
    ".inst 0x459348e7  // uaddwb z7.s, z7.s, z19.h\n"
    ".inst 0x45934cc6  // uaddwt z6.s, z6.s, z19.h\n"
    ".inst 0x459248a5  // uaddwb z5.s, z5.s, z18.h\n"
    ".inst 0x45924c84  // uaddwt z4.s, z4.s, z18.h\n"
    ".inst 0x45914863  // uaddwb z3.s, z3.s, z17.h\n"
    ".inst 0x45914c42  // uaddwt z2.s, z2.s, z17.h\n"
    ".inst 0x45904821  // uaddwb z1.s, z1.s, z16.h\n"
    ".inst 0x45904c00  // uaddwt z0.s, z0.s, z16.h\n"
    "bgt 2b\n"
    "3:"  // 4-vectors of channels: 2 inputs tail
    ".inst 0x455e0bf7  // uaddlb z23.h, z31.b, z30.b\n"
    ".inst 0x455e0ff6  // uaddlt z22.h, z31.b, z30.b\n"
    ".inst 0x455c0bb5  // uaddlb z21.h, z29.b, z28.b\n"
    ".inst 0x455c0fb4  // uaddlt z20.h, z29.b, z28.b\n"
    ".inst 0x455a0b73  // uaddlb z19.h, z27.b, z26.b\n"
    ".inst 0x455a0f72  // uaddlt z18.h, z27.b, z26.b\n"
    ".inst 0x45580b31  // uaddlb z17.h, z25.b, z24.b\n"
    ".inst 0x45580f30  // uaddlt z16.h, z25.b, z24.b\n"
    ".inst 0x459749ef  // uaddwb z15.s, z15.s, z23.h\n"
    ".inst 0x45974dce  // uaddwt z14.s, z14.s, z23.h\n"
    ".inst 0x459649ad  // uaddwb z13.s, z13.s, z22.h\n"
    ".inst 0x45964d8c  // uaddwt z12.s, z12.s, z22.h\n"
    ".inst 0x4595496b  // uaddwb z11.s, z11.s, z21.h\n"
    ".inst 0x45954d4a  // uaddwt z10.s, z10.s, z21.h\n"
    ".inst 0x45944929  // uaddwb z9.s, z9.s, z20.h\n"
    ".inst 0x45944d08  // uaddwt z8.s, z8.s, z20.h\n"
    ".inst 0x459348e7  // uaddwb z7.s, z7.s, z19.h\n"
    ".inst 0x45934cc6  // uaddwt z6.s, z6.s, z19.h\n"
    ".inst 0x459248a5  // uaddwb z5.s, z5.s, z18.h\n"
    ".inst 0x45924c84  // uaddwt z4.s, z4.s, z18.h\n"
    ".inst 0x45914863  // uaddwb z3.s, z3.s, z17.h\n"
    ".inst 0x45914c42  // uaddwt z2.s, z2.s, z17.h\n"
    ".inst 0x45904821  // uaddwb z1.s, z1.s, z16.h\n"
    ".inst 0x45904c00  // uaddwt z0.s, z0.s, z16.h\n"
    "4:"  // 4-vectors of channels: After loop
    "ands x20, %x[n_valid_cells], #0x1\n"
    "beq 6f\n"
    "5:"  // 4-vectors of channels: Single input loop
    "ldr x21, [x19], #0x8\n"
    "ld1b { z31.b }, p4/Z, [x21, x26]\n"
    ".inst 0x4508abf7  // ushllb z23.h, z31.b, #0x0\n"
    ".inst 0x4508aff6  // ushllt z22.h, z31.b, #0x0\n"
    "ld1b { z29.b }, p3/Z, [x21, x25]\n"
    "ld1b { z27.b }, p2/Z, [x21, x24]\n"
    ".inst 0x4508abb5  // ushllb z21.h, z29.b, #0x0\n"
    ".inst 0x4508afb4  // ushllt z20.h, z29.b, #0x0\n"
    "ld1b { z25.b }, p1/Z, [x21, x23]\n"
    ".inst 0x4508ab73  // ushllb z19.h, z27.b, #0x0\n"
    ".inst 0x4508af72  // ushllt z18.h, z27.b, #0x0\n"
    "subs x20, x20, #0x1\n"
    ".inst 0x4508ab31  // ushllb z17.h, z25.b, #0x0\n"
    ".inst 0x4508af30  // ushllt z16.h, z25.b, #0x0\n"
    ".inst 0x459749ef  // uaddwb z15.s, z15.s, z23.h\n"
    ".inst 0x45974dce  // uaddwt z14.s, z14.s, z23.h\n"
    ".inst 0x459649ad  // uaddwb z13.s, z13.s, z22.h\n"
    ".inst 0x45964d8c  // uaddwt z12.s, z12.s, z22.h\n"
    ".inst 0x4595496b  // uaddwb z11.s, z11.s, z21.h\n"
    ".inst 0x45954d4a  // uaddwt z10.s, z10.s, z21.h\n"
    ".inst 0x45944929  // uaddwb z9.s, z9.s, z20.h\n"
    ".inst 0x45944d08  // uaddwt z8.s, z8.s, z20.h\n"
    ".inst 0x459348e7  // uaddwb z7.s, z7.s, z19.h\n"
    ".inst 0x45934cc6  // uaddwt z6.s, z6.s, z19.h\n"
    ".inst 0x459248a5  // uaddwb z5.s, z5.s, z18.h\n"
    ".inst 0x45924c84  // uaddwt z4.s, z4.s, z18.h\n"
    ".inst 0x45914863  // uaddwb z3.s, z3.s, z17.h\n"
    ".inst 0x45914c42  // uaddwt z2.s, z2.s, z17.h\n"
    ".inst 0x45904821  // uaddwb z1.s, z1.s, z16.h\n"
    ".inst 0x45904c00  // uaddwt z0.s, z0.s, z16.h\n"
    "bgt 5b\n"
    "6:"  // 4-vectors of channels: Single input loop: End
    "ld1rw { z18.s }, p0/Z, [%x[left_shift]]\n"
    "ld1rw { z16.s }, p0/Z, [%x[combined_rescale_value]]\n"
    ".inst 0x4482824f  // srshl z15.s, p0/M, z15.s, z18.s\n"
    ".inst 0x4482824e  // srshl z14.s, p0/M, z14.s, z18.s\n"
    ".inst 0x4482824d  // srshl z13.s, p0/M, z13.s, z18.s\n"
    ".inst 0x4482824c  // srshl z12.s, p0/M, z12.s, z18.s\n"
    "ld1rw { z17.s }, p0/Z, [%x[right_shift]]\n"
    ".inst 0x04b075ef  // sqrdmulh z15.s, z15.s, z16.s\n"
    ".inst 0x4482824b  // srshl z11.s, p0/M, z11.s, z18.s\n"
    ".inst 0x4482824a  // srshl z10.s, p0/M, z10.s, z18.s\n"
    ".inst 0x04b075ce  // sqrdmulh z14.s, z14.s, z16.s\n"
    ".inst 0x04b075ad  // sqrdmulh z13.s, z13.s, z16.s\n"
    ".inst 0x44828249  // srshl z9.s, p0/M, z9.s, z18.s\n"
    ".inst 0x44828248  // srshl z8.s, p0/M, z8.s, z18.s\n"
    ".inst 0x04b0758c  // sqrdmulh z12.s, z12.s, z16.s\n"
    ".inst 0x04b0756b  // sqrdmulh z11.s, z11.s, z16.s\n"
    ".inst 0x44828247  // srshl z7.s, p0/M, z7.s, z18.s\n"
    ".inst 0x44828246  // srshl z6.s, p0/M, z6.s, z18.s\n"
    ".inst 0x04b0754a  // sqrdmulh z10.s, z10.s, z16.s\n"
    ".inst 0x04b07529  // sqrdmulh z9.s, z9.s, z16.s\n"
    ".inst 0x44828245  // srshl z5.s, p0/M, z5.s, z18.s\n"
    ".inst 0x44828244  // srshl z4.s, p0/M, z4.s, z18.s\n"
    ".inst 0x04b07508  // sqrdmulh z8.s, z8.s, z16.s\n"
    ".inst 0x04b074e7  // sqrdmulh z7.s, z7.s, z16.s\n"
    ".inst 0x44828243  // srshl z3.s, p0/M, z3.s, z18.s\n"
    ".inst 0x44828242  // srshl z2.s, p0/M, z2.s, z18.s\n"
    ".inst 0x04b074c6  // sqrdmulh z6.s, z6.s, z16.s\n"
    ".inst 0x04b074a5  // sqrdmulh z5.s, z5.s, z16.s\n"
    ".inst 0x44828241  // srshl z1.s, p0/M, z1.s, z18.s\n"
    ".inst 0x44828240  // srshl z0.s, p0/M, z0.s, z18.s\n"
    ".inst 0x04b07484  // sqrdmulh z4.s, z4.s, z16.s\n"
    ".inst 0x04b07463  // sqrdmulh z3.s, z3.s, z16.s\n"
    ".inst 0x04b07442  // sqrdmulh z2.s, z2.s, z16.s\n"
    ".inst 0x04b07421  // sqrdmulh z1.s, z1.s, z16.s\n"
    "add x19, %x[quant_params], %[offsetof_qp_output_offset]\n"
    ".inst 0x4482822f  // srshl z15.s, p0/M, z15.s, z17.s\n"
    ".inst 0x04b07400  // sqrdmulh z0.s, z0.s, z16.s\n"
    ".inst 0x4482822e  // srshl z14.s, p0/M, z14.s, z17.s\n"
    ".inst 0x4482822d  // srshl z13.s, p0/M, z13.s, z17.s\n"
    "ld1rw { z16.s }, p0/Z, [x19]\n"
    ".inst 0x4482822c  // srshl z12.s, p0/M, z12.s, z17.s\n"
    ".inst 0x4482822b  // srshl z11.s, p0/M, z11.s, z17.s\n"
    "add z15.s, z15.s, z16.s\n"
    "add z14.s, z14.s, z16.s\n"
    ".inst 0x4482822a  // srshl z10.s, p0/M, z10.s, z17.s\n"
    ".inst 0x44828229  // srshl z9.s, p0/M, z9.s, z17.s\n"
    "add z13.s, z13.s, z16.s\n"
    "add z12.s, z12.s, z16.s\n"
    ".inst 0x44828228  // srshl z8.s, p0/M, z8.s, z17.s\n"
    ".inst 0x44828227  // srshl z7.s, p0/M, z7.s, z17.s\n"
    "add z11.s, z11.s, z16.s\n"
    "add z10.s, z10.s, z16.s\n"
    ".inst 0x44828226  // srshl z6.s, p0/M, z6.s, z17.s\n"
    ".inst 0x44828225  // srshl z5.s, p0/M, z5.s, z17.s\n"
    "add z9.s, z9.s, z16.s\n"
    "add z8.s, z8.s, z16.s\n"
    ".inst 0x44828224  // srshl z4.s, p0/M, z4.s, z17.s\n"
    ".inst 0x44828223  // srshl z3.s, p0/M, z3.s, z17.s\n"
    "add z7.s, z7.s, z16.s\n"
    "add z6.s, z6.s, z16.s\n"
    ".inst 0x44828222  // srshl z2.s, p0/M, z2.s, z17.s\n"
    ".inst 0x44828221  // srshl z1.s, p0/M, z1.s, z17.s\n"
    "add z5.s, z5.s, z16.s\n"
    "add z4.s, z4.s, z16.s\n"
    ".inst 0x44828220  // srshl z0.s, p0/M, z0.s, z17.s\n"
    "add z3.s, z3.s, z16.s\n"
    "add z2.s, z2.s, z16.s\n"
    "add z1.s, z1.s, z16.s\n"
    "add z0.s, z0.s, z16.s\n"
    "mov z16.s, #0x0\n"
    "smax z15.s, p0/M, z15.s, z16.s\n"
    "smax z14.s, p0/M, z14.s, z16.s\n"
    "mov z18.s, #0xff\n"
    "smax z13.s, p0/M, z13.s, z16.s\n"
    "smax z12.s, p0/M, z12.s, z16.s\n"
    "smax z11.s, p0/M, z11.s, z16.s\n"
    "smax z10.s, p0/M, z10.s, z16.s\n"
    "smax z9.s, p0/M, z9.s, z16.s\n"
    "smax z8.s, p0/M, z8.s, z16.s\n"
    "smax z7.s, p0/M, z7.s, z16.s\n"
    "smax z6.s, p0/M, z6.s, z16.s\n"
    "smax z5.s, p0/M, z5.s, z16.s\n"
    "smax z4.s, p0/M, z4.s, z16.s\n"
    "smax z3.s, p0/M, z3.s, z16.s\n"
    "smax z2.s, p0/M, z2.s, z16.s\n"
    "smax z1.s, p0/M, z1.s, z16.s\n"
    "smax z0.s, p0/M, z0.s, z16.s\n"
    "smin z15.s, p0/M, z15.s, z18.s\n"
    "smin z14.s, p0/M, z14.s, z18.s\n"
    "trn1 z17.h, z15.h, z14.h\n"
    "smin z13.s, p0/M, z13.s, z18.s\n"
    "smin z12.s, p0/M, z12.s, z18.s\n"
    "trn1 z16.h, z13.h, z12.h\n"
    "trn1 z16.b, z17.b, z16.b\n"
    "smin z11.s, p0/M, z11.s, z18.s\n"
    "smin z10.s, p0/M, z10.s, z18.s\n"
    "trn1 z17.h, z11.h, z10.h\n"
    "st1b { z16.b }, p4, [%x[outptr], x26]\n"
    "smin z9.s, p0/M, z9.s, z18.s\n"
    "smin z8.s, p0/M, z8.s, z18.s\n"
    "trn1 z16.h, z9.h, z8.h\n"
    "trn1 z16.b, z17.b, z16.b\n"
    "smin z7.s, p0/M, z7.s, z18.s\n"
    "smin z6.s, p0/M, z6.s, z18.s\n"
    "trn1 z17.h, z7.h, z6.h\n"
    "st1b { z16.b }, p3, [%x[outptr], x25]\n"
    "smin z5.s, p0/M, z5.s, z18.s\n"
    "smin z4.s, p0/M, z4.s, z18.s\n"
    "trn1 z16.h, z5.h, z4.h\n"
    "trn1 z16.b, z17.b, z16.b\n"
    "smin z3.s, p0/M, z3.s, z18.s\n"
    "smin z2.s, p0/M, z2.s, z18.s\n"
    "trn1 z17.h, z3.h, z2.h\n"
    "st1b { z16.b }, p2, [%x[outptr], x24]\n"
    "smin z1.s, p0/M, z1.s, z18.s\n"
    "smin z0.s, p0/M, z0.s, z18.s\n"
    "trn1 z16.h, z1.h, z0.h\n"
    "trn1 z16.b, z17.b, z16.b\n"
    "st1b { z16.b }, p1, [%x[outptr], x23]\n"
    "incb x23, ALL, MUL #4\n"
    "whilelt p1.b, x23, %x[n_channels]\n"
    "incb x26, ALL, MUL #4\n"
    "incb x25, ALL, MUL #4\n"
    "incb x24, ALL, MUL #4\n"
    "b.any 1b\n"
    "7:"  // Single vector of channels
    "whilelt p4.b, x26, %x[n_channels]\n"
    "b.none 14f\n"
    "8:"  // Single vector of channels: Loop
    "ld1rw { z15.s }, p0/Z, [%x[accumulator_init]]\n"
    "lsr x22, %x[n_valid_cells], #0x1\n"
    "mov z14.d, z15.d\n"
    "mov z13.d, z15.d\n"
    "mov z12.d, z15.d\n"
    "mov x19, %x[inptrs]\n"
    "cbz x22, 11f\n"
    "ldp x21, x20, [x19, #0x0]\n"
    "subs x22, x22, #0x1\n"
    "add x19, x19, #0x10\n"
    "ld1b { z31.b }, p4/Z, [x21, x26]\n"
    "ld1b { z30.b }, p4/Z, [x20, x26]\n"
    "beq 10f\n"
    "9:"  // Single vector of channels: Loop: 2 inputs loop
    ".inst 0x455e0bf7  // uaddlb z23.h, z31.b, z30.b\n"
    ".inst 0x455e0ff6  // uaddlt z22.h, z31.b, z30.b\n"
    "ldp x21, x20, [x19, #0x0]\n"
    "subs x22, x22, #0x1\n"
    ".inst 0x459749ef  // uaddwb z15.s, z15.s, z23.h\n"
    ".inst 0x45974dce  // uaddwt z14.s, z14.s, z23.h\n"
    "add x19, x19, #0x10\n"
    "ld1b { z31.b }, p4/Z, [x21, x26]\n"
    ".inst 0x459649ad  // uaddwb z13.s, z13.s, z22.h\n"
    ".inst 0x45964d8c  // uaddwt z12.s, z12.s, z22.h\n"
    "ld1b { z30.b }, p4/Z, [x20, x26]\n"
    "bgt 9b\n"
    "10:"  // Single vector of channels: Loop: 2 inputs tail
    ".inst 0x455e0bf7  // uaddlb z23.h, z31.b, z30.b\n"
    ".inst 0x455e0ff6  // uaddlt z22.h, z31.b, z30.b\n"
    ".inst 0x459749ef  // uaddwb z15.s, z15.s, z23.h\n"
    ".inst 0x45974dce  // uaddwt z14.s, z14.s, z23.h\n"
    ".inst 0x459649ad  // uaddwb z13.s, z13.s, z22.h\n"
    ".inst 0x45964d8c  // uaddwt z12.s, z12.s, z22.h\n"
    "11:"  // Single vector of channels: Loop: After loop
    "ands x20, %x[n_valid_cells], #0x1\n"
    "beq 13f\n"
    "12:"  // Single vector of channels: Loop: Single input loop
    "ldr x21, [x19], #0x8\n"
    "ld1b { z31.b }, p4/Z, [x21, x26]\n"
    ".inst 0x4508abf7  // ushllb z23.h, z31.b, #0x0\n"
    ".inst 0x4508aff6  // ushllt z22.h, z31.b, #0x0\n"
    "subs x20, x20, #0x1\n"
    ".inst 0x459749ef  // uaddwb z15.s, z15.s, z23.h\n"
    ".inst 0x45974dce  // uaddwt z14.s, z14.s, z23.h\n"
    ".inst 0x459649ad  // uaddwb z13.s, z13.s, z22.h\n"
    ".inst 0x45964d8c  // uaddwt z12.s, z12.s, z22.h\n"
    "bgt 12b\n"
    "13:"  // Single vector of channels: Loop: Single input loop: End
    "ld1rw { z18.s }, p0/Z, [%x[left_shift]]\n"
    "ld1rw { z16.s }, p0/Z, [%x[combined_rescale_value]]\n"
    ".inst 0x4482824f  // srshl z15.s, p0/M, z15.s, z18.s\n"
    ".inst 0x4482824e  // srshl z14.s, p0/M, z14.s, z18.s\n"
    ".inst 0x4482824d  // srshl z13.s, p0/M, z13.s, z18.s\n"
    ".inst 0x4482824c  // srshl z12.s, p0/M, z12.s, z18.s\n"
    "ld1rw { z17.s }, p0/Z, [%x[right_shift]]\n"
    ".inst 0x04b075ef  // sqrdmulh z15.s, z15.s, z16.s\n"
    ".inst 0x04b075ce  // sqrdmulh z14.s, z14.s, z16.s\n"
    ".inst 0x04b075ad  // sqrdmulh z13.s, z13.s, z16.s\n"
    "add x19, %x[quant_params], %[offsetof_qp_output_offset]\n"
    ".inst 0x4482822f  // srshl z15.s, p0/M, z15.s, z17.s\n"
    ".inst 0x04b0758c  // sqrdmulh z12.s, z12.s, z16.s\n"
    ".inst 0x4482822e  // srshl z14.s, p0/M, z14.s, z17.s\n"
    ".inst 0x4482822d  // srshl z13.s, p0/M, z13.s, z17.s\n"
    "ld1rw { z16.s }, p0/Z, [x19]\n"
    ".inst 0x4482822c  // srshl z12.s, p0/M, z12.s, z17.s\n"
    "add z15.s, z15.s, z16.s\n"
    "add z14.s, z14.s, z16.s\n"
    "add z13.s, z13.s, z16.s\n"
    "add z12.s, z12.s, z16.s\n"
    "mov z16.s, #0x0\n"
    "smax z15.s, p0/M, z15.s, z16.s\n"
    "smax z14.s, p0/M, z14.s, z16.s\n"
    "mov z18.s, #0xff\n"
    "smax z13.s, p0/M, z13.s, z16.s\n"
    "smax z12.s, p0/M, z12.s, z16.s\n"
    "smin z15.s, p0/M, z15.s, z18.s\n"
    "smin z14.s, p0/M, z14.s, z18.s\n"
    "trn1 z17.h, z15.h, z14.h\n"
    "smin z13.s, p0/M, z13.s, z18.s\n"
    "smin z12.s, p0/M, z12.s, z18.s\n"
    "trn1 z16.h, z13.h, z12.h\n"
    "trn1 z16.b, z17.b, z16.b\n"
    "st1b { z16.b }, p4, [%x[outptr], x26]\n"
    "incb x26\n"
    "whilelt p4.b, x26, %x[n_channels]\n"
    "b.any 8b\n"
    "14:"  // End
    :
    : [accumulator_init] "r" (&accumulator_init), [combined_rescale_value] "r" (&combined_rescale_value), [inptrs] "r" (inptrs), [left_shift] "r" (&left_shift), [n_channels] "r" (n_channels), [n_valid_cells] "r" (n_valid_cells), [offsetof_qp_output_offset] "I" (offsetof(Requantize32, output_offset)), [outptr] "r" (outptr), [quant_params] "r" (&qp), [right_shift] "r" (&right_shift)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace pooling
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
