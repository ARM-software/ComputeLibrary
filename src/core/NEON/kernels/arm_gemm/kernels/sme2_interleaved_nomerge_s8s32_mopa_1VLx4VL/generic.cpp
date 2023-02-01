/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#ifdef __ARM_FEATURE_SVE
#ifdef ARM_COMPUTE_ENABLE_SME2

#include "arm_gemm.hpp"

#include <cstdint>
#include "../../asmlib.hpp"
#include "../../utils.hpp"

namespace arm_gemm {

void sme2_interleaved_nomerge_s8s32_mopa_1VLx4VL(const int8_t *const A, const int8_t *const B, int32_t *const C, int ldc, const int M, const int N, const int K, const int32_t *const bias, const Activation act, bool accumulate, int32_t *const accumulator_buffer)
{
  ARM_COMPUTE_UNUSED(act);

  struct KernelArgs
  {
    KernelArgs(
      const int8_t *const A,
      const int8_t *const B,
      int32_t *const C, const int ldc,
      const int M, const int N, const int K,
      const int32_t *const bias,

      bool accumulate,
      int32_t *const accumulator_buffer
    ) : A(A),
        B(B), kstride_bytes(roundup(K, 4) * sizeof(int8_t)),
        C(C), ldcb(ldc * sizeof(int32_t)),
        M(M), N(N), K(K),
        n_loops(((K / 4) - 1) / 2), n_tail_iters(((K / 4) - 1) % 2),

        bias(bias),
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
      }

    const int8_t *const A;
    const int8_t *const B;
    const long kstride_bytes;
    int32_t *const C;
    const long ldcb;
    const long M, N, K, n_loops, n_tail_iters;

    const int32_t *const bias;

    int32_t *const accumulator_buffer;
    uint64_t flags;
  };

  // Construct arguments for this kernel
  KernelArgs args(A, B, C, ldc, M, N, K, bias, accumulate, accumulator_buffer);

  __asm__ __volatile__(
      "ldr x13, [%x[args], %[offsetof_flags]]\n"
      ".inst 0xd503477f  // SMSTART ZA\n"
      "ptrue p0.b\n"
      ".inst 0x25207811  // ptrue pn9.b\n"
      "ldr x11, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "ldr x10, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "tbz x13, #0, 2f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "1:"  // Initial accumulator load from buffer: Loop
      ".inst 0xa040c564  // ld1w { z4.s-z7.s }, pn9.b/Z, [x11]\n"
      ".inst 0xc0840480  // mova za0h.s[x12], { z4.s-z7.s }\n"
      ".inst 0xa041c57c  // ld1w { z28.s-z31.s }, pn9.b/Z, [x11, #0x4, MUL VL]\n"
      ".inst 0xc0840781  // mova za1h.s[x12], { z28.s-z31.s }\n"
      ".inst 0xa042c570  // ld1w { z16.s-z19.s }, pn9.b/Z, [x11, #0x8, MUL VL]\n"
      ".inst 0xc0840602  // mova za2h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xa043c560  // ld1w { z0.s-z3.s }, pn9.b/Z, [x11, #0xc, MUL VL]\n"
      ".inst 0xc0840403  // mova za3h.s[x12], { z0.s-z3.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "addvl x11, x11, #16\n"
      "blt 1b\n"
      "2:"  // Initial accumulator load from buffer: End
      "ldr w9, [%x[args], %[offsetof_M]]\n"
      "mov x28, #0x0\n"
      "mov x27, #0x0\n"
      "ldr w26, [%x[args], %[offsetof_N]]\n"
      "ldr x25, [%x[args], %[offsetof_A]]\n"
      "3:"  // M and N loop
      "mov x24, x25\n"
      ".inst 0x25ba6770  // whilelt pn8.s, x27, x26, VLx4\n"
      "tbnz x13, #0, 4f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
      "cbz x20, 5f\n"
      ".inst 0xa11bc28a  // ldnt1w { z2.s, z6.s, z10.s, z14.s }, p8/Z, [x20, x27, LSL #2]\n"
      ".inst 0xc0900040  // addha za0.s, p0/M, p0/M, z2.s\n"
      ".inst 0xc09000c1  // addha za1.s, p0/M, p0/M, z6.s\n"
      ".inst 0xc0900142  // addha za2.s, p0/M, p0/M, z10.s\n"
      ".inst 0xc09001c3  // addha za3.s, p0/M, p0/M, z14.s\n"
      "4:"  // Prepare accumulators: Test for last block
      "mov x20, x27\n"
      "mov x21, x28\n"
      "incw x20, ALL, MUL #4\n"
      "incw x21\n"
      "cmp x20, x26\n"
      "csel x21, x28, x21, LT\n"
      "mov x20, x13\n"
      "bfm x13, XZR, #0x0, #0x0  // bfc x13, #0x0, #0x1\n"
      "cmp x21, x9\n"
      "csel x13, x20, x13, LT\n"
      "5:"  // Prepare accumulators: End
      "ldr x20, [%x[args], %[offsetof_K]]\n"
      "add x20, x20, #0x3\n"
      "lsr x20, x20, #0x2\n"
      "ldr x23, [%x[args], %[offsetof_B]]\n"
      "lsr x22, x20, #0x2\n"
      "and x21, x20, #0x3\n"
      "ldr x20, [%x[args], %[offsetof_kstride_bytes]]\n"
      "madd x23, x27, x20, x23\n"  // bptr = B + n * kstride_bytes
      "cbz x22, 8f\n"
      "subs x22, x22, #0x1\n"
      "ld1b { z20.b }, p0/Z, [x24]\n"
      ".inst 0xa14086e9  // ldnt1b { z1.b, z5.b, z9.b, z13.b }, pn9.b/Z, [x23]\n"
      "ld1b { z10.b }, p0/Z, [x24, #1, MUL VL]\n"
      ".inst 0xa14186fa  // ldnt1b { z18.b, z22.b, z26.b, z30.b }, pn9.b/Z, [x23, #0x4, MUL VL]\n"
      "ld1b { z16.b }, p0/Z, [x24, #2, MUL VL]\n"
      ".inst 0xa14286eb  // ldnt1b { z3.b, z7.b, z11.b, z15.b }, pn9.b/Z, [x23, #0x8, MUL VL]\n"
      "ld1b { z25.b }, p0/Z, [x24, #3, MUL VL]\n"
      "addvl x24, x24, #4\n"
      ".inst 0xa14386e8  // ldnt1b { z0.b, z4.b, z8.b, z12.b }, pn9.b/Z, [x23, #0xc, MUL VL]\n"
      "addvl x23, x23, #16\n"
      "ble 7f\n"
      "6:"  // K loop
      ".inst 0xa0810280  // smopa za0.s, p0/M, p0/M, z20.b, z1.b\n"
      "subs x22, x22, #0x1\n"
      ".inst 0xa0850281  // smopa za1.s, p0/M, p0/M, z20.b, z5.b\n"
      ".inst 0xa0890282  // smopa za2.s, p0/M, p0/M, z20.b, z9.b\n"
      ".inst 0xa08d0283  // smopa za3.s, p0/M, p0/M, z20.b, z13.b\n"
      "ld1b { z20.b }, p0/Z, [x24]\n"
      ".inst 0xa0920140  // smopa za0.s, p0/M, p0/M, z10.b, z18.b\n"
      ".inst 0xa14086e9  // ldnt1b { z1.b, z5.b, z9.b, z13.b }, pn9.b/Z, [x23]\n"
      ".inst 0xa0960141  // smopa za1.s, p0/M, p0/M, z10.b, z22.b\n"
      ".inst 0xa09a0142  // smopa za2.s, p0/M, p0/M, z10.b, z26.b\n"
      ".inst 0xa09e0143  // smopa za3.s, p0/M, p0/M, z10.b, z30.b\n"
      "ld1b { z10.b }, p0/Z, [x24, #1, MUL VL]\n"
      ".inst 0xa0830200  // smopa za0.s, p0/M, p0/M, z16.b, z3.b\n"
      ".inst 0xa14186fa  // ldnt1b { z18.b, z22.b, z26.b, z30.b }, pn9.b/Z, [x23, #0x4, MUL VL]\n"
      ".inst 0xa0870201  // smopa za1.s, p0/M, p0/M, z16.b, z7.b\n"
      ".inst 0xa08b0202  // smopa za2.s, p0/M, p0/M, z16.b, z11.b\n"
      ".inst 0xa08f0203  // smopa za3.s, p0/M, p0/M, z16.b, z15.b\n"
      "ld1b { z16.b }, p0/Z, [x24, #2, MUL VL]\n"
      ".inst 0xa14286eb  // ldnt1b { z3.b, z7.b, z11.b, z15.b }, pn9.b/Z, [x23, #0x8, MUL VL]\n"
      ".inst 0xa0800320  // smopa za0.s, p0/M, p0/M, z25.b, z0.b\n"
      ".inst 0xa0840321  // smopa za1.s, p0/M, p0/M, z25.b, z4.b\n"
      ".inst 0xa0880322  // smopa za2.s, p0/M, p0/M, z25.b, z8.b\n"
      ".inst 0xa08c0323  // smopa za3.s, p0/M, p0/M, z25.b, z12.b\n"
      "ld1b { z25.b }, p0/Z, [x24, #3, MUL VL]\n"
      "addvl x24, x24, #4\n"
      ".inst 0xa14386e8  // ldnt1b { z0.b, z4.b, z8.b, z12.b }, pn9.b/Z, [x23, #0xc, MUL VL]\n"
      "addvl x23, x23, #16\n"
      "bgt 6b\n"
      "7:"  // K loop tail
      ".inst 0xa0810280  // smopa za0.s, p0/M, p0/M, z20.b, z1.b\n"
      ".inst 0xa0850281  // smopa za1.s, p0/M, p0/M, z20.b, z5.b\n"
      ".inst 0xa0890282  // smopa za2.s, p0/M, p0/M, z20.b, z9.b\n"
      ".inst 0xa08d0283  // smopa za3.s, p0/M, p0/M, z20.b, z13.b\n"
      ".inst 0xa0920140  // smopa za0.s, p0/M, p0/M, z10.b, z18.b\n"
      ".inst 0xa0960141  // smopa za1.s, p0/M, p0/M, z10.b, z22.b\n"
      ".inst 0xa09a0142  // smopa za2.s, p0/M, p0/M, z10.b, z26.b\n"
      ".inst 0xa09e0143  // smopa za3.s, p0/M, p0/M, z10.b, z30.b\n"
      ".inst 0xa0830200  // smopa za0.s, p0/M, p0/M, z16.b, z3.b\n"
      ".inst 0xa0870201  // smopa za1.s, p0/M, p0/M, z16.b, z7.b\n"
      ".inst 0xa08b0202  // smopa za2.s, p0/M, p0/M, z16.b, z11.b\n"
      ".inst 0xa08f0203  // smopa za3.s, p0/M, p0/M, z16.b, z15.b\n"
      ".inst 0xa0800320  // smopa za0.s, p0/M, p0/M, z25.b, z0.b\n"
      ".inst 0xa0840321  // smopa za1.s, p0/M, p0/M, z25.b, z4.b\n"
      ".inst 0xa0880322  // smopa za2.s, p0/M, p0/M, z25.b, z8.b\n"
      ".inst 0xa08c0323  // smopa za3.s, p0/M, p0/M, z25.b, z12.b\n"
      "8:"  // K oddments
      "cbz x21, 10f\n"
      "9:"  // K oddments: Loop
      "ld1b { z20.b }, p0/Z, [x24]\n"
      "subs x21, x21, #0x1\n"
      "addvl x24, x24, #1\n"
      ".inst 0xa14086e1  // ld1b { z1.b, z5.b, z9.b, z13.b }, pn9.b/Z, [x23]\n"
      "addvl x23, x23, #4\n"
      ".inst 0xa0810280  // smopa za0.s, p0/M, p0/M, z20.b, z1.b\n"
      ".inst 0xa0850281  // smopa za1.s, p0/M, p0/M, z20.b, z5.b\n"
      ".inst 0xa0890282  // smopa za2.s, p0/M, p0/M, z20.b, z9.b\n"
      ".inst 0xa08d0283  // smopa za3.s, p0/M, p0/M, z20.b, z13.b\n"
      "bgt 9b\n"
      "10:"  // K oddments: End
      "tbz x13, #1, 14f\n"
      "tbz x13, #0, 12f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "11:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xa040c57c  // ld1w { z28.s-z31.s }, pn9.b/Z, [x11]\n"
      ".inst 0xc0860404  // mova { z4.s-z7.s }, za0h.s[x12]\n"
      ".inst 0xc0840780  // mova za0h.s[x12], { z28.s-z31.s }\n"
      ".inst 0xc0860434  // mova { z20.s-z23.s }, za1h.s[x12]\n"
      ".inst 0xa041c560  // ld1w { z0.s-z3.s }, pn9.b/Z, [x11, #0x4, MUL VL]\n"
      ".inst 0xc0840401  // mova za1h.s[x12], { z0.s-z3.s }\n"
      ".inst 0xc0860458  // mova { z24.s-z27.s }, za2h.s[x12]\n"
      ".inst 0xc086047c  // mova { z28.s-z31.s }, za3h.s[x12]\n"
      ".inst 0xa042c570  // ld1w { z16.s-z19.s }, pn9.b/Z, [x11, #0x8, MUL VL]\n"
      ".inst 0xc0840602  // mova za2h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xa043c56c  // ld1w { z12.s-z15.s }, pn9.b/Z, [x11, #0xc, MUL VL]\n"
      ".inst 0xc0840583  // mova za3h.s[x12], { z12.s-z15.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      ".inst 0xa060c544  // st1w { z4.s-z7.s }, pn9.b, [x10]\n"
      "addvl x11, x11, #16\n"
      ".inst 0xa061c554  // st1w { z20.s-z23.s }, pn9.b, [x10, #0x4, MUL VL]\n"
      ".inst 0xa062c558  // st1w { z24.s-z27.s }, pn9.b, [x10, #0x8, MUL VL]\n"
      ".inst 0xa063c55c  // st1w { z28.s-z31.s }, pn9.b, [x10, #0xc, MUL VL]\n"
      "addvl x10, x10, #16\n"
      "blt 11b\n"
      "b 20f\n"
      "12:"  // Store to partial result buffer: Store only
      "mov x12, #0x0\n"
      "cntw x20\n"
      "13:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xc0860414  // mova { z20.s-z23.s }, za0h.s[x12]\n"
      ".inst 0xc0860420  // mova { z0.s-z3.s }, za1h.s[x12]\n"
      ".inst 0xa060c554  // st1w { z20.s-z23.s }, pn9.b, [x10]\n"
      ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xa061c540  // st1w { z0.s-z3.s }, pn9.b, [x10, #0x4, MUL VL]\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      ".inst 0xa062c548  // st1w { z8.s-z11.s }, pn9.b, [x10, #0x8, MUL VL]\n"
      ".inst 0xa063c54c  // st1w { z12.s-z15.s }, pn9.b, [x10, #0xc, MUL VL]\n"
      "addvl x10, x10, #16\n"
      "blt 13b\n"
      "b 20f\n"
      "14:"  // Store to output array
      "ldr x23, [%x[args], %[offsetof_C]]\n"
      "sub x21, x9, x28\n"
      "cntw x20\n"
      "ldr x22, [%x[args], %[offsetof_ldcb]]\n"
      "cmp x21, x20\n"
      "csel x20, x21, x20, LT\n"
      "add x23, x23, x27, LSL #2\n"  // C += n
      "lsr x21, x20, #0x2\n"
      "madd x23, x28, x22, x23\n"  // C += m * ldc
      "mov x12, #0x0\n"
      "and x20, x20, #0x3\n"
      "cbz x21, 16f\n"
      "15:"  // Store to output array: Accumulator row 0 loop
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      ".inst 0xc0860424  // mova { z4.s-z7.s }, za1h.s[x12]\n"
      ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xa160c2e0  // st1w { z0.s, z4.s, z8.s, z12.s }, p8, [x23]\n"
      "add x23, x23, x22\n"
      ".inst 0xa160c2e1  // st1w { z1.s, z5.s, z9.s, z13.s }, p8, [x23]\n"
      "add x23, x23, x22\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa160c2e2  // st1w { z2.s, z6.s, z10.s, z14.s }, p8, [x23]\n"
      "add x23, x23, x22\n"
      "cmp x12, x21, LSL #2\n"
      ".inst 0xa160c2e3  // st1w { z3.s, z7.s, z11.s, z15.s }, p8, [x23]\n"
      "add x23, x23, x22\n"
      "blt 15b\n"
      "16:"  // Store to output array: Accumulator row 0 oddments
      "cbz x20, 17f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      ".inst 0xc0860424  // mova { z4.s-z7.s }, za1h.s[x12]\n"
      ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xa160c2e0  // st1w { z0.s, z4.s, z8.s, z12.s }, p8, [x23]\n"
      "add x23, x23, x22\n"
      "beq 17f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xa160c2e1  // st1w { z1.s, z5.s, z9.s, z13.s }, p8, [x23]\n"
      "add x23, x23, x22\n"
      "beq 17f\n"
      ".inst 0xa160c2e2  // st1w { z2.s, z6.s, z10.s, z14.s }, p8, [x23]\n"
      "17:"  // Store to output array: Accumulator row 0 oddments: End
      "18:"  // Store to output array: End
      "tbz x13, #0, 20f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "19:"  // Store to output array: Refill accumulators: Loop
      ".inst 0xa040c56c  // ld1w { z12.s-z15.s }, pn9.b/Z, [x11]\n"
      ".inst 0xc0840580  // mova za0h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xa041c570  // ld1w { z16.s-z19.s }, pn9.b/Z, [x11, #0x4, MUL VL]\n"
      ".inst 0xc0840601  // mova za1h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xa042c570  // ld1w { z16.s-z19.s }, pn9.b/Z, [x11, #0x8, MUL VL]\n"
      ".inst 0xc0840602  // mova za2h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xa043c564  // ld1w { z4.s-z7.s }, pn9.b/Z, [x11, #0xc, MUL VL]\n"
      ".inst 0xc0840483  // mova za3h.s[x12], { z4.s-z7.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "addvl x11, x11, #16\n"
      "blt 19b\n"
      "20:"  // End block
      "incw x27, ALL, MUL #4\n"
      "cmp x27, x26\n"
      "blt 3b\n"
      "incw x28\n"
      "cmp x28, x9\n"
      "mov x27, #0x0\n"
      "mov x25, x24\n"
      "blt 3b\n"
      ".inst 0xd503467f  // SMSTOP\n"
      :
      : [args] "r" (&args), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_kstride_bytes] "I" (offsetof(KernelArgs, kstride_bytes)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb))
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

}  // namespace arm_gemm

#endif  // ARM_COMPUTE_ENABLE_SME2
#endif  // __ARM_FEATURE_SVE
