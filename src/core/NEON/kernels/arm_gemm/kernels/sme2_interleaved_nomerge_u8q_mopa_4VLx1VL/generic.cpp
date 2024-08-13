/*
 * Copyright (c) 2022-2024 Arm Limited.
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
        min(0), max(0),

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
    const long M, N, K;
    int32_t min;
    int32_t max;

    const int32_t *const bias;
    const int n_0;


    int32_t *const accumulator_buffer;
    uint64_t flags;
  };

  // Construct arguments for this kernel
  KernelArgs args(A, B, C, ldc, M, N, K, bias, rq, n_0, accumulate, accumulator_buffer);

  __asm__ __volatile__(
      "ldr x16, [%x[args], %[offsetof_flags]]\n"
      ".inst 0xd503477f  // SMSTART ZA\n"
      "ptrue p1.b\n"
      ".inst 0x25207810  // ptrue pn8.b\n"
      "ldr x15, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "ldr x14, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "tbz x16, #0, 2f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "1:"  // Initial accumulator load from buffer: Loop
      ".inst 0xa040c1e8  // ld1w { z8.s-z11.s }, pn8.b/Z, [x15]\n"
      ".inst 0xa041c1ec  // ld1w { z12.s-z15.s }, pn8.b/Z, [x15, #0x4, MUL VL]\n"
      ".inst 0xa042c1e0  // ld1w { z0.s-z3.s }, pn8.b/Z, [x15, #0x8, MUL VL]\n"
      ".inst 0xa043c1e4  // ld1w { z4.s-z7.s }, pn8.b/Z, [x15, #0xc, MUL VL]\n"
      ".inst 0xc0840500  // mova za0h.s[x12], { z8.s-z11.s }\n"
      "addvl x15, x15, #16\n"
      ".inst 0xc0840581  // mova za1h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xc0840402  // mova za2h.s[x12], { z0.s-z3.s }\n"
      ".inst 0xc0840483  // mova za3h.s[x12], { z4.s-z7.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 1b\n"
      "2:"  // Initial accumulator load from buffer: End
      "ldr w13, [%x[args], %[offsetof_M]]\n"
      "mov x11, #0x0\n"
      "mov x10, #0x0\n"
      "ldr w9, [%x[args], %[offsetof_N]]\n"
      "ldr x28, [%x[args], %[offsetof_A]]\n"
      "3:"  // M and N loop
      "mov x27, x28\n"
      "whilelt p0.s, x10, x9\n"
      "tbnz x16, #0, 4f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
      "cbz x20, 5f\n"
      "ld1w { z6.s }, p0/Z, [x20, x10, LSL #2]\n"
      ".inst 0xc09024c0  // addha za0.s, p1/M, p1/M, z6.s\n"
      ".inst 0xc09024c1  // addha za1.s, p1/M, p1/M, z6.s\n"
      ".inst 0xc09024c2  // addha za2.s, p1/M, p1/M, z6.s\n"
      ".inst 0xc09024c3  // addha za3.s, p1/M, p1/M, z6.s\n"
      "4:"  // Prepare accumulators: Test for last block
      "mov x20, x10\n"
      "mov x21, x11\n"
      "incw x20\n"
      "incw x21, ALL, MUL #4\n"
      "cmp x20, x9\n"
      "mov x20, x16\n"
      "csel x21, x11, x21, LT\n"
      "bfm x16, XZR, #0x0, #0x0  // bfc x16, #0x0, #0x1\n"
      "cmp x21, x13\n"
      "csel x16, x20, x16, LT\n"
      "5:"  // Prepare accumulators: End
      "ldr x20, [%x[args], %[offsetof_K]]\n"
      "ldr x23, [%x[args], %[offsetof_B]]\n"
      "ldr x22, [%x[args], %[offsetof_kstride_bytes]]\n"
      "add x20, x20, #0x3\n"
      "lsr x20, x20, #0x2\n"
      "lsr x21, x20, #0x2\n"
      "madd x23, x10, x22, x23\n"  // bptr = B + n * kstride_bytes
      "and x20, x20, #0x3\n"
      "cbz x21, 8f\n"
      "subs x21, x21, #0x1\n"
      ".inst 0xa1408360  // ld1b { z0.b, z4.b, z8.b, z12.b }, pn8.b/Z, [x27]\n"
      "ld1b { z29.b }, p1/Z, [x23]\n"
      ".inst 0xa1418361  // ld1b { z1.b, z5.b, z9.b, z13.b }, pn8.b/Z, [x27, #0x4, MUL VL]\n"
      "ld1b { z19.b }, p1/Z, [x23, #1, MUL VL]\n"
      ".inst 0xa1428363  // ld1b { z3.b, z7.b, z11.b, z15.b }, pn8.b/Z, [x27, #0x8, MUL VL]\n"
      "ld1b { z20.b }, p1/Z, [x23, #2, MUL VL]\n"
      ".inst 0xa0438378  // ld1b { z24.b-z27.b }, pn8.b/Z, [x27, #0xc, MUL VL]\n"
      "addvl x27, x27, #16\n"
      "ld1b { z31.b }, p1/Z, [x23, #3, MUL VL]\n"
      "addvl x23, x23, #4\n"
      "ble 7f\n"
      "6:"  // K loop
      ".inst 0xa1bd2400  // umopa za0.s, p1/M, p1/M, z0.b, z29.b\n"
      "subs x21, x21, #0x1\n"
      ".inst 0xa1bd2481  // umopa za1.s, p1/M, p1/M, z4.b, z29.b\n"
      ".inst 0xa1bd2502  // umopa za2.s, p1/M, p1/M, z8.b, z29.b\n"
      ".inst 0xa1bd2583  // umopa za3.s, p1/M, p1/M, z12.b, z29.b\n"
      ".inst 0xa1408360  // ld1b { z0.b, z4.b, z8.b, z12.b }, pn8.b/Z, [x27]\n"
      ".inst 0xa1b32420  // umopa za0.s, p1/M, p1/M, z1.b, z19.b\n"
      "ld1b { z29.b }, p1/Z, [x23]\n"
      ".inst 0xa1b324a1  // umopa za1.s, p1/M, p1/M, z5.b, z19.b\n"
      ".inst 0xa1b32522  // umopa za2.s, p1/M, p1/M, z9.b, z19.b\n"
      ".inst 0xa1b325a3  // umopa za3.s, p1/M, p1/M, z13.b, z19.b\n"
      ".inst 0xa1418361  // ld1b { z1.b, z5.b, z9.b, z13.b }, pn8.b/Z, [x27, #0x4, MUL VL]\n"
      ".inst 0xa1b42460  // umopa za0.s, p1/M, p1/M, z3.b, z20.b\n"
      "ld1b { z19.b }, p1/Z, [x23, #1, MUL VL]\n"
      ".inst 0xa1b424e1  // umopa za1.s, p1/M, p1/M, z7.b, z20.b\n"
      ".inst 0xa1b42562  // umopa za2.s, p1/M, p1/M, z11.b, z20.b\n"
      ".inst 0xa1b425e3  // umopa za3.s, p1/M, p1/M, z15.b, z20.b\n"
      ".inst 0xa1428363  // ld1b { z3.b, z7.b, z11.b, z15.b }, pn8.b/Z, [x27, #0x8, MUL VL]\n"
      "ld1b { z20.b }, p1/Z, [x23, #2, MUL VL]\n"
      ".inst 0xa1bf2700  // umopa za0.s, p1/M, p1/M, z24.b, z31.b\n"
      ".inst 0xa1bf2721  // umopa za1.s, p1/M, p1/M, z25.b, z31.b\n"
      ".inst 0xa1bf2742  // umopa za2.s, p1/M, p1/M, z26.b, z31.b\n"
      ".inst 0xa1bf2763  // umopa za3.s, p1/M, p1/M, z27.b, z31.b\n"
      ".inst 0xa0438378  // ld1b { z24.b-z27.b }, pn8.b/Z, [x27, #0xc, MUL VL]\n"
      "addvl x27, x27, #16\n"
      "ld1b { z31.b }, p1/Z, [x23, #3, MUL VL]\n"
      "addvl x23, x23, #4\n"
      "bgt 6b\n"
      "7:"  // K loop tail
      ".inst 0xa1bd2400  // umopa za0.s, p1/M, p1/M, z0.b, z29.b\n"
      ".inst 0xa1bd2481  // umopa za1.s, p1/M, p1/M, z4.b, z29.b\n"
      ".inst 0xa1bd2502  // umopa za2.s, p1/M, p1/M, z8.b, z29.b\n"
      ".inst 0xa1bd2583  // umopa za3.s, p1/M, p1/M, z12.b, z29.b\n"
      ".inst 0xa1b32420  // umopa za0.s, p1/M, p1/M, z1.b, z19.b\n"
      ".inst 0xa1b324a1  // umopa za1.s, p1/M, p1/M, z5.b, z19.b\n"
      ".inst 0xa1b32522  // umopa za2.s, p1/M, p1/M, z9.b, z19.b\n"
      ".inst 0xa1b325a3  // umopa za3.s, p1/M, p1/M, z13.b, z19.b\n"
      ".inst 0xa1b42460  // umopa za0.s, p1/M, p1/M, z3.b, z20.b\n"
      ".inst 0xa1b424e1  // umopa za1.s, p1/M, p1/M, z7.b, z20.b\n"
      ".inst 0xa1b42562  // umopa za2.s, p1/M, p1/M, z11.b, z20.b\n"
      ".inst 0xa1b425e3  // umopa za3.s, p1/M, p1/M, z15.b, z20.b\n"
      ".inst 0xa1bf2700  // umopa za0.s, p1/M, p1/M, z24.b, z31.b\n"
      ".inst 0xa1bf2721  // umopa za1.s, p1/M, p1/M, z25.b, z31.b\n"
      ".inst 0xa1bf2742  // umopa za2.s, p1/M, p1/M, z26.b, z31.b\n"
      ".inst 0xa1bf2763  // umopa za3.s, p1/M, p1/M, z27.b, z31.b\n"
      "8:"  // K oddments
      "cbz x20, 10f\n"
      "9:"  // K oddments: Loop
      ".inst 0xa1408372  // ld1b { z18.b, z22.b, z26.b, z30.b }, pn8.b/Z, [x27]\n"
      "subs x20, x20, #0x1\n"
      "addvl x27, x27, #4\n"
      "ld1b { z15.b }, p1/Z, [x23]\n"
      "addvl x23, x23, #1\n"
      ".inst 0xa1af2640  // umopa za0.s, p1/M, p1/M, z18.b, z15.b\n"
      ".inst 0xa1af26c1  // umopa za1.s, p1/M, p1/M, z22.b, z15.b\n"
      ".inst 0xa1af2742  // umopa za2.s, p1/M, p1/M, z26.b, z15.b\n"
      ".inst 0xa1af27c3  // umopa za3.s, p1/M, p1/M, z30.b, z15.b\n"
      "bgt 9b\n"
      "10:"  // K oddments: End
      ".inst 0xa140c363  // ld1w { z3.s, z7.s, z11.s, z15.s }, pn8.b/Z, [x27]\n"
      "addvl x27, x27, #4\n"
      ".inst 0xc0912460  // addva za0.s, p1/M, p1/M, z3.s\n"
      ".inst 0xc09124e1  // addva za1.s, p1/M, p1/M, z7.s\n"
      ".inst 0xc0912562  // addva za2.s, p1/M, p1/M, z11.s\n"
      ".inst 0xc09125e3  // addva za3.s, p1/M, p1/M, z15.s\n"
      "tbz x16, #1, 14f\n"
      "tbz x16, #0, 12f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "11:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xa040c1e4  // ld1w { z4.s-z7.s }, pn8.b/Z, [x15]\n"
      ".inst 0xc0860414  // mova { z20.s-z23.s }, za0h.s[x12]\n"
      ".inst 0xc0860420  // mova { z0.s-z3.s }, za1h.s[x12]\n"
      ".inst 0xa041c1ec  // ld1w { z12.s-z15.s }, pn8.b/Z, [x15, #0x4, MUL VL]\n"
      ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
      ".inst 0xc0860478  // mova { z24.s-z27.s }, za3h.s[x12]\n"
      ".inst 0xa042c1fc  // ld1w { z28.s-z31.s }, pn8.b/Z, [x15, #0x8, MUL VL]\n"
      ".inst 0xa043c1f0  // ld1w { z16.s-z19.s }, pn8.b/Z, [x15, #0xc, MUL VL]\n"
      ".inst 0xc0840480  // mova za0h.s[x12], { z4.s-z7.s }\n"
      "addvl x15, x15, #16\n"
      ".inst 0xc0840581  // mova za1h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xa060c1d4  // st1w { z20.s-z23.s }, pn8.b, [x14]\n"
      ".inst 0xc0840782  // mova za2h.s[x12], { z28.s-z31.s }\n"
      ".inst 0xa061c1c0  // st1w { z0.s-z3.s }, pn8.b, [x14, #0x4, MUL VL]\n"
      ".inst 0xc0840603  // mova za3h.s[x12], { z16.s-z19.s }\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa062c1c8  // st1w { z8.s-z11.s }, pn8.b, [x14, #0x8, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa063c1d8  // st1w { z24.s-z27.s }, pn8.b, [x14, #0xc, MUL VL]\n"
      "addvl x14, x14, #16\n"
      "blt 11b\n"
      "b 30f\n"
      "12:"  // Store to partial result buffer: Store only
      "mov x12, #0x0\n"
      "cntw x20\n"
      "13:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xc0860404  // mova { z4.s-z7.s }, za0h.s[x12]\n"
      ".inst 0xc0860420  // mova { z0.s-z3.s }, za1h.s[x12]\n"
      ".inst 0xc0860454  // mova { z20.s-z23.s }, za2h.s[x12]\n"
      ".inst 0xc0860470  // mova { z16.s-z19.s }, za3h.s[x12]\n"
      ".inst 0xa060c1c4  // st1w { z4.s-z7.s }, pn8.b, [x14]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa061c1c0  // st1w { z0.s-z3.s }, pn8.b, [x14, #0x4, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa062c1d4  // st1w { z20.s-z23.s }, pn8.b, [x14, #0x8, MUL VL]\n"
      ".inst 0xa063c1d0  // st1w { z16.s-z19.s }, pn8.b, [x14, #0xc, MUL VL]\n"
      "addvl x14, x14, #16\n"
      "blt 13b\n"
      "b 30f\n"
      "14:"  // Store to output array
      "ldr x26, [%x[args], %[offsetof_C]]\n"
      "sub x25, x13, x11\n"
      "ld1rw { z2.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_mul]]\n"
      "ldr x24, [%x[args], %[offsetof_ldcb]]\n"
      "ld1rw { z1.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_right_shift]]\n"
      "ld1rw { z0.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_c_offset]]\n"
      "add x26, x26, x10\n"  // C += n
      "ld1rw { z25.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_minval]]\n"
      "madd x26, x11, x24, x26\n"  // C += m * ldc
      "ld1rw { z24.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_maxval]]\n"
      "tbz x16, #2, 15f\n"
      "ldr w22, [%x[args], %[offsetof_n_0]]\n"
      "ldr x21, [%x[rq], %[offsetof_Requantize32_per_channel_muls]]\n"
      "ldr x20, [%x[rq], %[offsetof_Requantize32_per_channel_right_shifts]]\n"
      "add x22, x22, x10\n"
      "add x21, x21, x22, LSL #2\n"
      "add x20, x20, x22, LSL #2\n"
      "ld1w { z2.s }, p0/Z, [x21]\n"
      "ld1w { z1.s }, p0/Z, [x20]\n"
      "15:"  // Store to output array: Load per-channel parameters: End
      "cntw x23\n"
      "whilelt p0.s, x10, x9\n"
      "cmp x25, x23\n"
      "mov x12, #0x0\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 17f\n"
      "16:"  // Store to output array: Accumulator row 0 loop
      ".inst 0xc0860410  // mova { z16.s-z19.s }, za0h.s[x12]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xc1a2ac10  // sqdmulh { z16.s-z19.s }, { z16.s-z19.s }, z2.s\n"
      "cmp x12, x21, LSL #2\n"
      ".inst 0xc1a1aa30  // srshl { z16.s-z19.s }, { z16.s-z19.s }, z1.s\n"
      ".inst 0xc1a0ab10  // add { z16.s-z19.s }, { z16.s-z19.s }, z0.s\n"
      ".inst 0xc1b8cf30  // sclamp { z16.s-z19.s }, z25.s, z24.s\n"
      "st1b { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1b { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1b { z18.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1b { z19.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 16b\n"
      "17:"  // Store to output array: Accumulator row 0 oddments
      "cbz x20, 18f\n"
      ".inst 0xc0860404  // mova { z4.s-z7.s }, za0h.s[x12]\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc1a2ac04  // sqdmulh { z4.s-z7.s }, { z4.s-z7.s }, z2.s\n"
      ".inst 0xc1a1aa24  // srshl { z4.s-z7.s }, { z4.s-z7.s }, z1.s\n"
      ".inst 0xc1a0ab04  // add { z4.s-z7.s }, { z4.s-z7.s }, z0.s\n"
      ".inst 0xc1b8cf24  // sclamp { z4.s-z7.s }, z25.s, z24.s\n"
      "st1b { z4.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 18f\n"
      "subs x20, x20, #0x1\n"
      "st1b { z5.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 18f\n"
      "st1b { z6.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "18:"  // Store to output array: Accumulator row 0 oddments: End
      "subs x25, x25, x22\n"
      "beq 28f\n"
      "cmp x25, x23\n"
      "mov x12, #0x0\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 20f\n"
      "19:"  // Store to output array: Accumulator row 1 loop
      ".inst 0xc0860430  // mova { z16.s-z19.s }, za1h.s[x12]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xc1a2ac10  // sqdmulh { z16.s-z19.s }, { z16.s-z19.s }, z2.s\n"
      "cmp x12, x21, LSL #2\n"
      ".inst 0xc1a1aa30  // srshl { z16.s-z19.s }, { z16.s-z19.s }, z1.s\n"
      ".inst 0xc1a0ab10  // add { z16.s-z19.s }, { z16.s-z19.s }, z0.s\n"
      ".inst 0xc1b8cf30  // sclamp { z16.s-z19.s }, z25.s, z24.s\n"
      "st1b { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1b { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1b { z18.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1b { z19.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 19b\n"
      "20:"  // Store to output array: Accumulator row 1 oddments
      "cbz x20, 21f\n"
      ".inst 0xc0860424  // mova { z4.s-z7.s }, za1h.s[x12]\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc1a2ac04  // sqdmulh { z4.s-z7.s }, { z4.s-z7.s }, z2.s\n"
      ".inst 0xc1a1aa24  // srshl { z4.s-z7.s }, { z4.s-z7.s }, z1.s\n"
      ".inst 0xc1a0ab04  // add { z4.s-z7.s }, { z4.s-z7.s }, z0.s\n"
      ".inst 0xc1b8cf24  // sclamp { z4.s-z7.s }, z25.s, z24.s\n"
      "st1b { z4.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 21f\n"
      "subs x20, x20, #0x1\n"
      "st1b { z5.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 21f\n"
      "st1b { z6.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "21:"  // Store to output array: Accumulator row 1 oddments: End
      "subs x25, x25, x22\n"
      "beq 28f\n"
      "cmp x25, x23\n"
      "mov x12, #0x0\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 23f\n"
      "22:"  // Store to output array: Accumulator row 2 loop
      ".inst 0xc086044c  // mova { z12.s-z15.s }, za2h.s[x12]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xc1a2ac0c  // sqdmulh { z12.s-z15.s }, { z12.s-z15.s }, z2.s\n"
      "cmp x12, x21, LSL #2\n"
      ".inst 0xc1a1aa2c  // srshl { z12.s-z15.s }, { z12.s-z15.s }, z1.s\n"
      ".inst 0xc1a0ab0c  // add { z12.s-z15.s }, { z12.s-z15.s }, z0.s\n"
      ".inst 0xc1b8cf2c  // sclamp { z12.s-z15.s }, z25.s, z24.s\n"
      "st1b { z12.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1b { z13.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1b { z14.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1b { z15.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 22b\n"
      "23:"  // Store to output array: Accumulator row 2 oddments
      "cbz x20, 24f\n"
      ".inst 0xc0860450  // mova { z16.s-z19.s }, za2h.s[x12]\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc1a2ac10  // sqdmulh { z16.s-z19.s }, { z16.s-z19.s }, z2.s\n"
      ".inst 0xc1a1aa30  // srshl { z16.s-z19.s }, { z16.s-z19.s }, z1.s\n"
      ".inst 0xc1a0ab10  // add { z16.s-z19.s }, { z16.s-z19.s }, z0.s\n"
      ".inst 0xc1b8cf30  // sclamp { z16.s-z19.s }, z25.s, z24.s\n"
      "st1b { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 24f\n"
      "subs x20, x20, #0x1\n"
      "st1b { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 24f\n"
      "st1b { z18.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "24:"  // Store to output array: Accumulator row 2 oddments: End
      "subs x25, x25, x22\n"
      "beq 28f\n"
      "cmp x25, x23\n"
      "mov x12, #0x0\n"
      "csel x20, x25, x23, LT\n"
      "lsr x21, x20, #0x2\n"
      "and x20, x20, #0x3\n"
      "cbz x21, 26f\n"
      "25:"  // Store to output array: Accumulator row 3 loop
      ".inst 0xc0860470  // mova { z16.s-z19.s }, za3h.s[x12]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xc1a2ac10  // sqdmulh { z16.s-z19.s }, { z16.s-z19.s }, z2.s\n"
      "cmp x12, x21, LSL #2\n"
      ".inst 0xc1a1aa30  // srshl { z16.s-z19.s }, { z16.s-z19.s }, z1.s\n"
      ".inst 0xc1a0ab10  // add { z16.s-z19.s }, { z16.s-z19.s }, z0.s\n"
      ".inst 0xc1b8cf30  // sclamp { z16.s-z19.s }, z25.s, z24.s\n"
      "st1b { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1b { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1b { z18.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1b { z19.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 25b\n"
      "26:"  // Store to output array: Accumulator row 3 oddments
      "cbz x20, 27f\n"
      ".inst 0xc0860474  // mova { z20.s-z23.s }, za3h.s[x12]\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc1a2ac14  // sqdmulh { z20.s-z23.s }, { z20.s-z23.s }, z2.s\n"
      ".inst 0xc1a1aa34  // srshl { z20.s-z23.s }, { z20.s-z23.s }, z1.s\n"
      ".inst 0xc1a0ab14  // add { z20.s-z23.s }, { z20.s-z23.s }, z0.s\n"
      ".inst 0xc1b8cf34  // sclamp { z20.s-z23.s }, z25.s, z24.s\n"
      "st1b { z20.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 27f\n"
      "subs x20, x20, #0x1\n"
      "st1b { z21.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 27f\n"
      "st1b { z22.s }, p0, [x26]\n"
      "27:"  // Store to output array: Accumulator row 3 oddments: End
      "28:"  // Store to output array: End
      "tbz x16, #0, 30f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "29:"  // Store to output array: Refill accumulators: Loop
      ".inst 0xa040c1f8  // ld1w { z24.s-z27.s }, pn8.b/Z, [x15]\n"
      ".inst 0xa041c1ec  // ld1w { z12.s-z15.s }, pn8.b/Z, [x15, #0x4, MUL VL]\n"
      ".inst 0xa042c1fc  // ld1w { z28.s-z31.s }, pn8.b/Z, [x15, #0x8, MUL VL]\n"
      ".inst 0xa043c1e0  // ld1w { z0.s-z3.s }, pn8.b/Z, [x15, #0xc, MUL VL]\n"
      ".inst 0xc0840700  // mova za0h.s[x12], { z24.s-z27.s }\n"
      "addvl x15, x15, #16\n"
      ".inst 0xc0840581  // mova za1h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xc0840782  // mova za2h.s[x12], { z28.s-z31.s }\n"
      ".inst 0xc0840403  // mova za3h.s[x12], { z0.s-z3.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 29b\n"
      "30:"  // End block
      "incw x10\n"
      "cmp x10, x9\n"
      "blt 3b\n"
      "incw x11, ALL, MUL #4\n"
      "mov x10, #0x0\n"
      "cmp x11, x13\n"
      "mov x28, x27\n"
      "blt 3b\n"
      ".inst 0xd503467f  // SMSTOP\n"
      :
      : [args] "r" (&args), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_Requantize32_c_offset] "I" (offsetof(Requantize32, c_offset)), [offsetof_Requantize32_maxval] "I" (offsetof(Requantize32, maxval)), [offsetof_Requantize32_minval] "I" (offsetof(Requantize32, minval)), [offsetof_Requantize32_per_channel_muls] "I" (offsetof(Requantize32, per_channel_muls)), [offsetof_Requantize32_per_channel_right_shifts] "I" (offsetof(Requantize32, per_channel_right_shifts)), [offsetof_Requantize32_per_layer_mul] "I" (offsetof(Requantize32, per_layer_mul)), [offsetof_Requantize32_per_layer_right_shift] "I" (offsetof(Requantize32, per_layer_right_shift)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_kstride_bytes] "I" (offsetof(KernelArgs, kstride_bytes)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb)), [offsetof_n_0] "I" (offsetof(KernelArgs, n_0)), [rq] "r" (&rq)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

}  // namespace arm_gemm

#endif  // ARM_COMPUTE_ENABLE_SME2
