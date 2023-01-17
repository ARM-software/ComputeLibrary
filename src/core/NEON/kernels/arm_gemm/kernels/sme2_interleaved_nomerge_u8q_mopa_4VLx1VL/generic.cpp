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
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#ifdef __ARM_FEATURE_SVE
#ifdef ARM_COMPUTE_ENABLE_SME2

#include "arm_gemm.hpp"

#include <cstdint>
#include "../../asmlib.hpp"
#include "../../utils.hpp"

namespace arm_gemm {

void sme2_interleaved_nomerge_u8q_mopa_4VLx1VL(const uint8_t *const A, const uint8_t *const B, uint8_t *const C, int ldc, const int M, const int N, const int K, const int32_t *const bias, const Requantize32 &rq, const int n_0, bool accumulate, int32_t *const accumulator_buffer)
{
  struct KernelArgs
  {
    KernelArgs(
      const uint8_t *const A,
      const uint8_t *const B,
      uint8_t *const C, const int ldc,
      const int M, const int N, const int K,
      const int32_t *const bias,
      const Requantize32 &rq,
      const int n_0,
      bool accumulate,
      int32_t *const accumulator_buffer
    ) : A(A),
        B(B), kstride_bytes(roundup(K, 4) * sizeof(uint8_t)),
        C(C), ldcb(ldc * sizeof(uint8_t)),
        M(M), N(N), K(K),
        n_loops(((K / 4) - 1) / 2), n_tail_iters(((K / 4) - 1) % 2),

        bias(bias), n_0(n_0),
        accumulator_buffer(accumulator_buffer),
        flags(0x0)
    {
      if (accumulate)
      {
        flags |= 1 << 0;  // FILL_ACCUMULATORS_FROM_BUFFER
      }
      if (C == nullptr)
      {
        flags |= 1 << 1;  // STORE_ACCUMULATORS_TO_BUFFER
      }
      if (rq.per_channel_requant)
      {
        flags |= 1 << 2;  // PER_CHANNEL_QUANTISATION
      }
      }

    const uint8_t *const A;
    const uint8_t *const B;
    const long kstride_bytes;
    uint8_t *const C;
    const long ldcb;
    const long M, N, K, n_loops, n_tail_iters;
    int32_t min = std::numeric_limits<uint8_t>::min();
    int32_t max = std::numeric_limits<uint8_t>::max();

    const int32_t *const bias;
    const int n_0;

    int32_t *const accumulator_buffer;
    uint64_t flags;
  };

  // Construct arguments for this kernel
  KernelArgs args(A, B, C, ldc, M, N, K, bias, rq, n_0, accumulate, accumulator_buffer);

  __asm__ __volatile__(
      "ldr x15, [%x[args], %[offsetof_flags]]\n"
      ".inst 0xd503477f  // SMSTART ZA\n"
      "ptrue p1.b\n"
      ".inst 0x25207810  // ptrue pn8.b\n"
      "ldr x14, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "ldr x13, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "tbz x15, #0, 2f\n"
      "mov x12, #0x0\n"
      "cntw x19\n"
      "1:"  // Initial accumulator load from buffer: Loop
      ".inst 0xa040c1dc  // ld1w { z28.s-z31.s }, pn8.b/Z, [x14]\n"
      ".inst 0xc0840780  // mova za0h.s[x12], { z28.s-z31.s }\n"
      ".inst 0xa041c1cc  // ld1w { z12.s-z15.s }, pn8.b/Z, [x14, #0x4, MUL VL]\n"
      ".inst 0xc0840581  // mova za1h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xa042c1d4  // ld1w { z20.s-z23.s }, pn8.b/Z, [x14, #0x8, MUL VL]\n"
      ".inst 0xc0840682  // mova za2h.s[x12], { z20.s-z23.s }\n"
      ".inst 0xa043c1d8  // ld1w { z24.s-z27.s }, pn8.b/Z, [x14, #0xc, MUL VL]\n"
      ".inst 0xc0840703  // mova za3h.s[x12], { z24.s-z27.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x19\n"
      "addvl x14, x14, #16\n"
      "blt 1b\n"
      "2:"  // Initial accumulator load from buffer: End
      "ldr w11, [%x[args], %[offsetof_M]]\n"
      "mov x10, #0x0\n"
      "mov x9, #0x0\n"
      "ldr w28, [%x[args], %[offsetof_N]]\n"
      "ldr x27, [%x[args], %[offsetof_A]]\n"
      "3:"  // M and N loop
      "mov x26, x27\n"
      "whilelt p0.s, x9, x28\n"
      "tbnz x15, #0, 4f\n"
      "ldr x19, [%x[args], %[offsetof_bias]]\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
      "cbz x19, 5f\n"
      "ldnt1w { z15.s }, p0/Z, [x19, x9, LSL #2]\n"
      ".inst 0xc09025e0  // addha za0.s, p1/M, p1/M, z15.s\n"
      ".inst 0xc09025e1  // addha za1.s, p1/M, p1/M, z15.s\n"
      ".inst 0xc09025e2  // addha za2.s, p1/M, p1/M, z15.s\n"
      ".inst 0xc09025e3  // addha za3.s, p1/M, p1/M, z15.s\n"
      "4:"  // Prepare accumulators: Test for last block
      "mov x19, x9\n"
      "mov x20, x10\n"
      "incw x19\n"
      "incw x20, ALL, MUL #4\n"
      "cmp x19, x28\n"
      "csel x20, x10, x20, LT\n"
      "mov x19, x15\n"
      "bfm x15, XZR, #0x0, #0x0  // bfc x15, #0x0, #0x1\n"
      "cmp x20, x11\n"
      "csel x15, x19, x15, LT\n"
      "5:"  // Prepare accumulators: End
      "ldr x19, [%x[args], %[offsetof_K]]\n"
      "add x19, x19, #0x3\n"
      "lsr x19, x19, #0x2\n"
      "ldr x22, [%x[args], %[offsetof_B]]\n"
      "lsr x21, x19, #0x2\n"
      "and x20, x19, #0x3\n"
      "ldr x19, [%x[args], %[offsetof_kstride_bytes]]\n"
      "madd x22, x9, x19, x22\n"  // bptr = B + n * kstride_bytes
      "cbz x21, 8f\n"
      "subs x21, x21, #0x1\n"
      ".inst 0xa1408352  // ld1b { z18.b, z22.b, z26.b, z30.b }, pn8.b/Z, [x26]\n"
      "ldnt1b { z0.b }, p1/Z, [x22]\n"
      ".inst 0xa1418353  // ld1b { z19.b, z23.b, z27.b, z31.b }, pn8.b/Z, [x26, #0x4, MUL VL]\n"
      "ldnt1b { z9.b }, p1/Z, [x22, #1, MUL VL]\n"
      ".inst 0xa1428350  // ld1b { z16.b, z20.b, z24.b, z28.b }, pn8.b/Z, [x26, #0x8, MUL VL]\n"
      "ldnt1b { z21.b }, p1/Z, [x22, #2, MUL VL]\n"
      ".inst 0xa1438342  // ld1b { z2.b, z6.b, z10.b, z14.b }, pn8.b/Z, [x26, #0xc, MUL VL]\n"
      "addvl x26, x26, #16\n"
      "ldnt1b { z12.b }, p1/Z, [x22, #3, MUL VL]\n"
      "addvl x22, x22, #4\n"
      "ble 7f\n"
      "6:"  // K loop
      ".inst 0xa1a02640  // umopa za0.s, p1/M, p1/M, z18.b, z0.b\n"
      "subs x21, x21, #0x1\n"
      ".inst 0xa1a026c1  // umopa za1.s, p1/M, p1/M, z22.b, z0.b\n"
      ".inst 0xa1a02742  // umopa za2.s, p1/M, p1/M, z26.b, z0.b\n"
      ".inst 0xa1a027c3  // umopa za3.s, p1/M, p1/M, z30.b, z0.b\n"
      ".inst 0xa1408352  // ld1b { z18.b, z22.b, z26.b, z30.b }, pn8.b/Z, [x26]\n"
      ".inst 0xa1a92660  // umopa za0.s, p1/M, p1/M, z19.b, z9.b\n"
      "ldnt1b { z0.b }, p1/Z, [x22]\n"
      ".inst 0xa1a926e1  // umopa za1.s, p1/M, p1/M, z23.b, z9.b\n"
      ".inst 0xa1a92762  // umopa za2.s, p1/M, p1/M, z27.b, z9.b\n"
      ".inst 0xa1a927e3  // umopa za3.s, p1/M, p1/M, z31.b, z9.b\n"
      ".inst 0xa1418353  // ld1b { z19.b, z23.b, z27.b, z31.b }, pn8.b/Z, [x26, #0x4, MUL VL]\n"
      ".inst 0xa1b52600  // umopa za0.s, p1/M, p1/M, z16.b, z21.b\n"
      "ldnt1b { z9.b }, p1/Z, [x22, #1, MUL VL]\n"
      ".inst 0xa1b52681  // umopa za1.s, p1/M, p1/M, z20.b, z21.b\n"
      ".inst 0xa1b52702  // umopa za2.s, p1/M, p1/M, z24.b, z21.b\n"
      ".inst 0xa1b52783  // umopa za3.s, p1/M, p1/M, z28.b, z21.b\n"
      ".inst 0xa1428350  // ld1b { z16.b, z20.b, z24.b, z28.b }, pn8.b/Z, [x26, #0x8, MUL VL]\n"
      "ldnt1b { z21.b }, p1/Z, [x22, #2, MUL VL]\n"
      ".inst 0xa1ac2440  // umopa za0.s, p1/M, p1/M, z2.b, z12.b\n"
      ".inst 0xa1ac24c1  // umopa za1.s, p1/M, p1/M, z6.b, z12.b\n"
      ".inst 0xa1ac2542  // umopa za2.s, p1/M, p1/M, z10.b, z12.b\n"
      ".inst 0xa1ac25c3  // umopa za3.s, p1/M, p1/M, z14.b, z12.b\n"
      ".inst 0xa1438342  // ld1b { z2.b, z6.b, z10.b, z14.b }, pn8.b/Z, [x26, #0xc, MUL VL]\n"
      "addvl x26, x26, #16\n"
      "ldnt1b { z12.b }, p1/Z, [x22, #3, MUL VL]\n"
      "addvl x22, x22, #4\n"
      "bgt 6b\n"
      "7:"  // K loop tail
      ".inst 0xa1a02640  // umopa za0.s, p1/M, p1/M, z18.b, z0.b\n"
      ".inst 0xa1a026c1  // umopa za1.s, p1/M, p1/M, z22.b, z0.b\n"
      ".inst 0xa1a02742  // umopa za2.s, p1/M, p1/M, z26.b, z0.b\n"
      ".inst 0xa1a027c3  // umopa za3.s, p1/M, p1/M, z30.b, z0.b\n"
      ".inst 0xa1a92660  // umopa za0.s, p1/M, p1/M, z19.b, z9.b\n"
      ".inst 0xa1a926e1  // umopa za1.s, p1/M, p1/M, z23.b, z9.b\n"
      ".inst 0xa1a92762  // umopa za2.s, p1/M, p1/M, z27.b, z9.b\n"
      ".inst 0xa1a927e3  // umopa za3.s, p1/M, p1/M, z31.b, z9.b\n"
      ".inst 0xa1b52600  // umopa za0.s, p1/M, p1/M, z16.b, z21.b\n"
      ".inst 0xa1b52681  // umopa za1.s, p1/M, p1/M, z20.b, z21.b\n"
      ".inst 0xa1b52702  // umopa za2.s, p1/M, p1/M, z24.b, z21.b\n"
      ".inst 0xa1b52783  // umopa za3.s, p1/M, p1/M, z28.b, z21.b\n"
      ".inst 0xa1ac2440  // umopa za0.s, p1/M, p1/M, z2.b, z12.b\n"
      ".inst 0xa1ac24c1  // umopa za1.s, p1/M, p1/M, z6.b, z12.b\n"
      ".inst 0xa1ac2542  // umopa za2.s, p1/M, p1/M, z10.b, z12.b\n"
      ".inst 0xa1ac25c3  // umopa za3.s, p1/M, p1/M, z14.b, z12.b\n"
      "8:"  // K oddments
      "cbz x20, 10f\n"
      "9:"  // K oddments: Loop
      ".inst 0xa1408352  // ld1b { z18.b, z22.b, z26.b, z30.b }, pn8.b/Z, [x26]\n"
      "subs x20, x20, #0x1\n"
      "addvl x26, x26, #4\n"
      "ld1b { z0.b }, p1/Z, [x22]\n"
      "addvl x22, x22, #1\n"
      ".inst 0xa1a02640  // umopa za0.s, p1/M, p1/M, z18.b, z0.b\n"
      ".inst 0xa1a026c1  // umopa za1.s, p1/M, p1/M, z22.b, z0.b\n"
      ".inst 0xa1a02742  // umopa za2.s, p1/M, p1/M, z26.b, z0.b\n"
      ".inst 0xa1a027c3  // umopa za3.s, p1/M, p1/M, z30.b, z0.b\n"
      "bgt 9b\n"
      "10:"  // K oddments: End
      ".inst 0xa040c340  // ld1w { z0.s-z3.s }, pn8.b/Z, [x26]\n"
      "addvl x26, x26, #4\n"
      ".inst 0xc0912400  // addva za0.s, p1/M, p1/M, z0.s\n"
      ".inst 0xc0912421  // addva za1.s, p1/M, p1/M, z1.s\n"
      ".inst 0xc0912442  // addva za2.s, p1/M, p1/M, z2.s\n"
      ".inst 0xc0912463  // addva za3.s, p1/M, p1/M, z3.s\n"
      "tbz x15, #1, 14f\n"
      "tbz x15, #0, 12f\n"
      "mov x12, #0x0\n"
      "cntw x19\n"
      "11:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xa040c1d4  // ld1w { z20.s-z23.s }, pn8.b/Z, [x14]\n"
      ".inst 0xc0860410  // mova { z16.s-z19.s }, za0h.s[x12]\n"
      ".inst 0xc0840680  // mova za0h.s[x12], { z20.s-z23.s }\n"
      ".inst 0xc0860428  // mova { z8.s-z11.s }, za1h.s[x12]\n"
      ".inst 0xa041c1c4  // ld1w { z4.s-z7.s }, pn8.b/Z, [x14, #0x4, MUL VL]\n"
      ".inst 0xc0840481  // mova za1h.s[x12], { z4.s-z7.s }\n"
      ".inst 0xc086044c  // mova { z12.s-z15.s }, za2h.s[x12]\n"
      ".inst 0xc086047c  // mova { z28.s-z31.s }, za3h.s[x12]\n"
      ".inst 0xa042c1c4  // ld1w { z4.s-z7.s }, pn8.b/Z, [x14, #0x8, MUL VL]\n"
      ".inst 0xc0840482  // mova za2h.s[x12], { z4.s-z7.s }\n"
      ".inst 0xa043c1d4  // ld1w { z20.s-z23.s }, pn8.b/Z, [x14, #0xc, MUL VL]\n"
      ".inst 0xc0840683  // mova za3h.s[x12], { z20.s-z23.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x19\n"
      ".inst 0xa060c1b0  // st1w { z16.s-z19.s }, pn8.b, [x13]\n"
      "addvl x14, x14, #16\n"
      ".inst 0xa061c1a8  // st1w { z8.s-z11.s }, pn8.b, [x13, #0x4, MUL VL]\n"
      ".inst 0xa062c1ac  // st1w { z12.s-z15.s }, pn8.b, [x13, #0x8, MUL VL]\n"
      ".inst 0xa063c1bc  // st1w { z28.s-z31.s }, pn8.b, [x13, #0xc, MUL VL]\n"
      "addvl x13, x13, #16\n"
      "blt 11b\n"
      "b 30f\n"
      "12:"  // Store to partial result buffer: Store only
      "mov x12, #0x0\n"
      "cntw x19\n"
      "13:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xc0860410  // mova { z16.s-z19.s }, za0h.s[x12]\n"
      ".inst 0xc086042c  // mova { z12.s-z15.s }, za1h.s[x12]\n"
      ".inst 0xa060c1b0  // st1w { z16.s-z19.s }, pn8.b, [x13]\n"
      ".inst 0xc0860454  // mova { z20.s-z23.s }, za2h.s[x12]\n"
      ".inst 0xc0860478  // mova { z24.s-z27.s }, za3h.s[x12]\n"
      ".inst 0xa061c1ac  // st1w { z12.s-z15.s }, pn8.b, [x13, #0x4, MUL VL]\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x19\n"
      ".inst 0xa062c1b4  // st1w { z20.s-z23.s }, pn8.b, [x13, #0x8, MUL VL]\n"
      ".inst 0xa063c1b8  // st1w { z24.s-z27.s }, pn8.b, [x13, #0xc, MUL VL]\n"
      "addvl x13, x13, #16\n"
      "blt 13b\n"
      "b 30f\n"
      "14:"  // Store to output array
      "ldr x25, [%x[args], %[offsetof_C]]\n"
      "add x25, x25, x9\n"  // C += n
      "sub x24, x11, x10\n"
      "ld1rw { z8.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_mul]]\n"
      "ldr x23, [%x[args], %[offsetof_ldcb]]\n"
      "madd x25, x10, x23, x25\n"  // C += m * ldc
      "ld1rw { z7.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_right_shift]]\n"
      "ld1rw { z6.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_c_offset]]\n"
      "ld1rw { z5.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_minval]]\n"
      "ld1rw { z4.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_maxval]]\n"
      "tbz x15, #2, 15f\n"
      "ldr w20, [%x[args], %[offsetof_n_0]]\n"
      "add x20, x20, x9\n"
      "ldr x19, [%x[rq], %[offsetof_Requantize32_per_channel_muls]]\n"
      "add x19, x19, x20, LSL #2\n"
      "ld1w { z8.s }, p0/Z, [x19]\n"
      "ldr x19, [%x[rq], %[offsetof_Requantize32_per_channel_right_shifts]]\n"
      "add x19, x19, x20, LSL #2\n"
      "ld1w { z7.s }, p0/Z, [x19]\n"
      "15:"  // Store to output array: Load per-channel parameters: End
      "cntw x22\n"
      "whilelt p0.s, x9, x28\n"
      "cmp x24, x22\n"
      "csel x21, x24, x22, LT\n"
      "lsr x20, x21, #0x2\n"
      "mov x12, #0x0\n"
      "and x19, x21, #0x3\n"
      "cbz x20, 17f\n"
      "16:"  // Store to output array: Accumulator row 0 loop
      ".inst 0xc086040c  // mova { z12.s-z15.s }, za0h.s[x12]\n"
      ".inst 0xc1a8ac0c  // sqdmulh { z12.s-z15.s }, { z12.s-z15.s }, z8.s\n"
      "add x12, x12, #0x4\n"
      ".inst 0xc1a7aa2c  // srshl { z12.s-z15.s }, { z12.s-z15.s }, z7.s\n"
      "cmp x12, x20, LSL #2\n"
      ".inst 0xc1a6ab0c  // add { z12.s-z15.s }, { z12.s-z15.s }, z6.s\n"
      ".inst 0xc1a4ccac  // sclamp { z12.s-z15.s }, z5.s, z4.s\n"
      "st1b { z12.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "st1b { z13.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "st1b { z14.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "st1b { z15.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "blt 16b\n"
      "17:"  // Store to output array: Accumulator row 0 oddments
      "cbz x19, 18f\n"
      ".inst 0xc0860410  // mova { z16.s-z19.s }, za0h.s[x12]\n"
      ".inst 0xc1a8ac10  // sqdmulh { z16.s-z19.s }, { z16.s-z19.s }, z8.s\n"
      "subs x19, x19, #0x1\n"
      ".inst 0xc1a7aa30  // srshl { z16.s-z19.s }, { z16.s-z19.s }, z7.s\n"
      ".inst 0xc1a6ab10  // add { z16.s-z19.s }, { z16.s-z19.s }, z6.s\n"
      ".inst 0xc1a4ccb0  // sclamp { z16.s-z19.s }, z5.s, z4.s\n"
      "st1b { z16.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "beq 18f\n"
      "subs x19, x19, #0x1\n"
      "st1b { z17.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "beq 18f\n"
      "st1b { z18.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "18:"  // Store to output array: Accumulator row 0 oddments: End
      "subs x24, x24, x21\n"
      "beq 28f\n"
      "whilelt p0.s, x9, x28\n"
      "cmp x24, x22\n"
      "csel x21, x24, x22, LT\n"
      "lsr x20, x21, #0x2\n"
      "mov x12, #0x0\n"
      "and x19, x21, #0x3\n"
      "cbz x20, 20f\n"
      "19:"  // Store to output array: Accumulator row 1 loop
      ".inst 0xc0860430  // mova { z16.s-z19.s }, za1h.s[x12]\n"
      ".inst 0xc1a8ac10  // sqdmulh { z16.s-z19.s }, { z16.s-z19.s }, z8.s\n"
      "add x12, x12, #0x4\n"
      ".inst 0xc1a7aa30  // srshl { z16.s-z19.s }, { z16.s-z19.s }, z7.s\n"
      "cmp x12, x20, LSL #2\n"
      ".inst 0xc1a6ab10  // add { z16.s-z19.s }, { z16.s-z19.s }, z6.s\n"
      ".inst 0xc1a4ccb0  // sclamp { z16.s-z19.s }, z5.s, z4.s\n"
      "st1b { z16.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "st1b { z17.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "st1b { z18.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "st1b { z19.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "blt 19b\n"
      "20:"  // Store to output array: Accumulator row 1 oddments
      "cbz x19, 21f\n"
      ".inst 0xc086043c  // mova { z28.s-z31.s }, za1h.s[x12]\n"
      ".inst 0xc1a8ac1c  // sqdmulh { z28.s-z31.s }, { z28.s-z31.s }, z8.s\n"
      "subs x19, x19, #0x1\n"
      ".inst 0xc1a7aa3c  // srshl { z28.s-z31.s }, { z28.s-z31.s }, z7.s\n"
      ".inst 0xc1a6ab1c  // add { z28.s-z31.s }, { z28.s-z31.s }, z6.s\n"
      ".inst 0xc1a4ccbc  // sclamp { z28.s-z31.s }, z5.s, z4.s\n"
      "st1b { z28.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "beq 21f\n"
      "subs x19, x19, #0x1\n"
      "st1b { z29.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "beq 21f\n"
      "st1b { z30.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "21:"  // Store to output array: Accumulator row 1 oddments: End
      "subs x24, x24, x21\n"
      "beq 28f\n"
      "whilelt p0.s, x9, x28\n"
      "cmp x24, x22\n"
      "csel x21, x24, x22, LT\n"
      "lsr x20, x21, #0x2\n"
      "mov x12, #0x0\n"
      "and x19, x21, #0x3\n"
      "cbz x20, 23f\n"
      "22:"  // Store to output array: Accumulator row 2 loop
      ".inst 0xc0860458  // mova { z24.s-z27.s }, za2h.s[x12]\n"
      ".inst 0xc1a8ac18  // sqdmulh { z24.s-z27.s }, { z24.s-z27.s }, z8.s\n"
      "add x12, x12, #0x4\n"
      ".inst 0xc1a7aa38  // srshl { z24.s-z27.s }, { z24.s-z27.s }, z7.s\n"
      "cmp x12, x20, LSL #2\n"
      ".inst 0xc1a6ab18  // add { z24.s-z27.s }, { z24.s-z27.s }, z6.s\n"
      ".inst 0xc1a4ccb8  // sclamp { z24.s-z27.s }, z5.s, z4.s\n"
      "st1b { z24.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "st1b { z25.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "st1b { z26.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "st1b { z27.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "blt 22b\n"
      "23:"  // Store to output array: Accumulator row 2 oddments
      "cbz x19, 24f\n"
      ".inst 0xc086044c  // mova { z12.s-z15.s }, za2h.s[x12]\n"
      ".inst 0xc1a8ac0c  // sqdmulh { z12.s-z15.s }, { z12.s-z15.s }, z8.s\n"
      "subs x19, x19, #0x1\n"
      ".inst 0xc1a7aa2c  // srshl { z12.s-z15.s }, { z12.s-z15.s }, z7.s\n"
      ".inst 0xc1a6ab0c  // add { z12.s-z15.s }, { z12.s-z15.s }, z6.s\n"
      ".inst 0xc1a4ccac  // sclamp { z12.s-z15.s }, z5.s, z4.s\n"
      "st1b { z12.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "beq 24f\n"
      "subs x19, x19, #0x1\n"
      "st1b { z13.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "beq 24f\n"
      "st1b { z14.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "24:"  // Store to output array: Accumulator row 2 oddments: End
      "subs x24, x24, x21\n"
      "beq 28f\n"
      "whilelt p0.s, x9, x28\n"
      "cmp x24, x22\n"
      "csel x19, x24, x22, LT\n"
      "lsr x20, x19, #0x2\n"
      "mov x12, #0x0\n"
      "and x19, x19, #0x3\n"
      "cbz x20, 26f\n"
      "25:"  // Store to output array: Accumulator row 3 loop
      ".inst 0xc0860474  // mova { z20.s-z23.s }, za3h.s[x12]\n"
      ".inst 0xc1a8ac14  // sqdmulh { z20.s-z23.s }, { z20.s-z23.s }, z8.s\n"
      "add x12, x12, #0x4\n"
      ".inst 0xc1a7aa34  // srshl { z20.s-z23.s }, { z20.s-z23.s }, z7.s\n"
      "cmp x12, x20, LSL #2\n"
      ".inst 0xc1a6ab14  // add { z20.s-z23.s }, { z20.s-z23.s }, z6.s\n"
      ".inst 0xc1a4ccb4  // sclamp { z20.s-z23.s }, z5.s, z4.s\n"
      "st1b { z20.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "st1b { z21.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "st1b { z22.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "st1b { z23.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "blt 25b\n"
      "26:"  // Store to output array: Accumulator row 3 oddments
      "cbz x19, 27f\n"
      ".inst 0xc0860460  // mova { z0.s-z3.s }, za3h.s[x12]\n"
      ".inst 0xc1a8ac00  // sqdmulh { z0.s-z3.s }, { z0.s-z3.s }, z8.s\n"
      "subs x19, x19, #0x1\n"
      ".inst 0xc1a7aa20  // srshl { z0.s-z3.s }, { z0.s-z3.s }, z7.s\n"
      ".inst 0xc1a6ab00  // add { z0.s-z3.s }, { z0.s-z3.s }, z6.s\n"
      ".inst 0xc1a4cca0  // sclamp { z0.s-z3.s }, z5.s, z4.s\n"
      "st1b { z0.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "beq 27f\n"
      "subs x19, x19, #0x1\n"
      "st1b { z1.s }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "beq 27f\n"
      "st1b { z2.s }, p0, [x25]\n"
      "27:"  // Store to output array: Accumulator row 3 oddments: End
      "28:"  // Store to output array: End
      "tbz x15, #0, 30f\n"
      "mov x12, #0x0\n"
      "cntw x19\n"
      "29:"  // Store to output array: Refill accumulators: Loop
      ".inst 0xa040c1c4  // ld1w { z4.s-z7.s }, pn8.b/Z, [x14]\n"
      ".inst 0xc0840480  // mova za0h.s[x12], { z4.s-z7.s }\n"
      ".inst 0xa041c1d0  // ld1w { z16.s-z19.s }, pn8.b/Z, [x14, #0x4, MUL VL]\n"
      ".inst 0xc0840601  // mova za1h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xa042c1d0  // ld1w { z16.s-z19.s }, pn8.b/Z, [x14, #0x8, MUL VL]\n"
      ".inst 0xc0840602  // mova za2h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xa043c1c4  // ld1w { z4.s-z7.s }, pn8.b/Z, [x14, #0xc, MUL VL]\n"
      ".inst 0xc0840483  // mova za3h.s[x12], { z4.s-z7.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x19\n"
      "addvl x14, x14, #16\n"
      "blt 29b\n"
      "30:"  // End block
      "incw x9\n"
      "cmp x9, x28\n"
      "blt 3b\n"
      "incw x10, ALL, MUL #4\n"
      "cmp x10, x11\n"
      "mov x9, #0x0\n"
      "mov x27, x26\n"
      "blt 3b\n"
      ".inst 0xd503467f  // SMSTOP\n"
      :
      : [args] "r" (&args), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_Requantize32_c_offset] "I" (offsetof(Requantize32, c_offset)), [offsetof_Requantize32_maxval] "I" (offsetof(Requantize32, maxval)), [offsetof_Requantize32_minval] "I" (offsetof(Requantize32, minval)), [offsetof_Requantize32_per_channel_muls] "I" (offsetof(Requantize32, per_channel_muls)), [offsetof_Requantize32_per_channel_right_shifts] "I" (offsetof(Requantize32, per_channel_right_shifts)), [offsetof_Requantize32_per_layer_mul] "I" (offsetof(Requantize32, per_layer_mul)), [offsetof_Requantize32_per_layer_right_shift] "I" (offsetof(Requantize32, per_layer_right_shift)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_kstride_bytes] "I" (offsetof(KernelArgs, kstride_bytes)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb)), [offsetof_n_0] "I" (offsetof(KernelArgs, n_0)), [rq] "r" (&rq)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

}  // namespace arm_gemm

#endif  // ARM_COMPUTE_ENABLE_SME2
#endif  // __ARM_FEATURE_SVE
