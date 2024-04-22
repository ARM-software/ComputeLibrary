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
#ifdef ARM_COMPUTE_ENABLE_SME2

#include "arm_gemm.hpp"

#include <cstdint>
#include "../../asmlib.hpp"
#include "../../utils.hpp"

namespace arm_gemm {

void sme2_interleaved_nomerge_s8s32_mopa_1VLx4VL(const int8_t *const A, const int8_t *const B, int32_t *const C, int ldc, const int M, const int N, const int K, const int32_t *const bias, const Activation, bool accumulate, int32_t *const accumulator_buffer)
{
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
      ".inst 0xa041c56c  // ld1w { z12.s-z15.s }, pn9.b/Z, [x11, #0x4, MUL VL]\n"
      ".inst 0xc0840581  // mova za1h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xa042c57c  // ld1w { z28.s-z31.s }, pn9.b/Z, [x11, #0x8, MUL VL]\n"
      ".inst 0xc0840782  // mova za2h.s[x12], { z28.s-z31.s }\n"
      ".inst 0xa043c574  // ld1w { z20.s-z23.s }, pn9.b/Z, [x11, #0xc, MUL VL]\n"
      ".inst 0xc0840683  // mova za3h.s[x12], { z20.s-z23.s }\n"
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
      ".inst 0xa11bc29b  // ldnt1w { z19.s, z23.s, z27.s, z31.s }, p8/Z, [x20, x27, LSL #2]\n"
      ".inst 0xc0900260  // addha za0.s, p0/M, p0/M, z19.s\n"
      ".inst 0xc09002e1  // addha za1.s, p0/M, p0/M, z23.s\n"
      ".inst 0xc0900362  // addha za2.s, p0/M, p0/M, z27.s\n"
      ".inst 0xc09003e3  // addha za3.s, p0/M, p0/M, z31.s\n"
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
      "ld1b { z30.b }, p0/Z, [x24]\n"
      ".inst 0xa04086e1  // ldnt1b { z0.b-z3.b }, pn9.b/Z, [x23]\n"
      "ld1b { z21.b }, p0/Z, [x24, #1, MUL VL]\n"
      ".inst 0xa04186f9  // ldnt1b { z24.b-z27.b }, pn9.b/Z, [x23, #0x4, MUL VL]\n"
      "ld1b { z28.b }, p0/Z, [x24, #2, MUL VL]\n"
      ".inst 0xa04286e5  // ldnt1b { z4.b-z7.b }, pn9.b/Z, [x23, #0x8, MUL VL]\n"
      "ld1b { z11.b }, p0/Z, [x24, #3, MUL VL]\n"
      "addvl x24, x24, #4\n"
      ".inst 0xa04386f1  // ldnt1b { z16.b-z19.b }, pn9.b/Z, [x23, #0xc, MUL VL]\n"
      "addvl x23, x23, #16\n"
      "ble 7f\n"
      "6:"  // K loop
      ".inst 0xa08003c0  // smopa za0.s, p0/M, p0/M, z30.b, z0.b\n"
      "subs x22, x22, #0x1\n"
      ".inst 0xa08103c1  // smopa za1.s, p0/M, p0/M, z30.b, z1.b\n"
      ".inst 0xa08203c2  // smopa za2.s, p0/M, p0/M, z30.b, z2.b\n"
      ".inst 0xa08303c3  // smopa za3.s, p0/M, p0/M, z30.b, z3.b\n"
      "ld1b { z30.b }, p0/Z, [x24]\n"
      ".inst 0xa09802a0  // smopa za0.s, p0/M, p0/M, z21.b, z24.b\n"
      ".inst 0xa04086e1  // ldnt1b { z0.b-z3.b }, pn9.b/Z, [x23]\n"
      ".inst 0xa09902a1  // smopa za1.s, p0/M, p0/M, z21.b, z25.b\n"
      ".inst 0xa09a02a2  // smopa za2.s, p0/M, p0/M, z21.b, z26.b\n"
      ".inst 0xa09b02a3  // smopa za3.s, p0/M, p0/M, z21.b, z27.b\n"
      "ld1b { z21.b }, p0/Z, [x24, #1, MUL VL]\n"
      ".inst 0xa0840380  // smopa za0.s, p0/M, p0/M, z28.b, z4.b\n"
      ".inst 0xa04186f9  // ldnt1b { z24.b-z27.b }, pn9.b/Z, [x23, #0x4, MUL VL]\n"
      ".inst 0xa0850381  // smopa za1.s, p0/M, p0/M, z28.b, z5.b\n"
      ".inst 0xa0860382  // smopa za2.s, p0/M, p0/M, z28.b, z6.b\n"
      ".inst 0xa0870383  // smopa za3.s, p0/M, p0/M, z28.b, z7.b\n"
      "ld1b { z28.b }, p0/Z, [x24, #2, MUL VL]\n"
      ".inst 0xa04286e5  // ldnt1b { z4.b-z7.b }, pn9.b/Z, [x23, #0x8, MUL VL]\n"
      ".inst 0xa0900160  // smopa za0.s, p0/M, p0/M, z11.b, z16.b\n"
      ".inst 0xa0910161  // smopa za1.s, p0/M, p0/M, z11.b, z17.b\n"
      ".inst 0xa0920162  // smopa za2.s, p0/M, p0/M, z11.b, z18.b\n"
      ".inst 0xa0930163  // smopa za3.s, p0/M, p0/M, z11.b, z19.b\n"
      "ld1b { z11.b }, p0/Z, [x24, #3, MUL VL]\n"
      "addvl x24, x24, #4\n"
      ".inst 0xa04386f1  // ldnt1b { z16.b-z19.b }, pn9.b/Z, [x23, #0xc, MUL VL]\n"
      "addvl x23, x23, #16\n"
      "bgt 6b\n"
      "7:"  // K loop tail
      ".inst 0xa08003c0  // smopa za0.s, p0/M, p0/M, z30.b, z0.b\n"
      ".inst 0xa08103c1  // smopa za1.s, p0/M, p0/M, z30.b, z1.b\n"
      ".inst 0xa08203c2  // smopa za2.s, p0/M, p0/M, z30.b, z2.b\n"
      ".inst 0xa08303c3  // smopa za3.s, p0/M, p0/M, z30.b, z3.b\n"
      ".inst 0xa09802a0  // smopa za0.s, p0/M, p0/M, z21.b, z24.b\n"
      ".inst 0xa09902a1  // smopa za1.s, p0/M, p0/M, z21.b, z25.b\n"
      ".inst 0xa09a02a2  // smopa za2.s, p0/M, p0/M, z21.b, z26.b\n"
      ".inst 0xa09b02a3  // smopa za3.s, p0/M, p0/M, z21.b, z27.b\n"
      ".inst 0xa0840380  // smopa za0.s, p0/M, p0/M, z28.b, z4.b\n"
      ".inst 0xa0850381  // smopa za1.s, p0/M, p0/M, z28.b, z5.b\n"
      ".inst 0xa0860382  // smopa za2.s, p0/M, p0/M, z28.b, z6.b\n"
      ".inst 0xa0870383  // smopa za3.s, p0/M, p0/M, z28.b, z7.b\n"
      ".inst 0xa0900160  // smopa za0.s, p0/M, p0/M, z11.b, z16.b\n"
      ".inst 0xa0910161  // smopa za1.s, p0/M, p0/M, z11.b, z17.b\n"
      ".inst 0xa0920162  // smopa za2.s, p0/M, p0/M, z11.b, z18.b\n"
      ".inst 0xa0930163  // smopa za3.s, p0/M, p0/M, z11.b, z19.b\n"
      "8:"  // K oddments
      "cbz x21, 10f\n"
      "9:"  // K oddments: Loop
      "ld1b { z22.b }, p0/Z, [x24]\n"
      "subs x21, x21, #0x1\n"
      "addvl x24, x24, #1\n"
      ".inst 0xa14086f1  // ld1b { z17.b, z21.b, z25.b, z29.b }, pn9.b/Z, [x23]\n"
      "addvl x23, x23, #4\n"
      ".inst 0xa09102c0  // smopa za0.s, p0/M, p0/M, z22.b, z17.b\n"
      ".inst 0xa09502c1  // smopa za1.s, p0/M, p0/M, z22.b, z21.b\n"
      ".inst 0xa09902c2  // smopa za2.s, p0/M, p0/M, z22.b, z25.b\n"
      ".inst 0xa09d02c3  // smopa za3.s, p0/M, p0/M, z22.b, z29.b\n"
      "bgt 9b\n"
      "10:"  // K oddments: End
      "tbz x13, #1, 14f\n"
      "tbz x13, #0, 12f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "11:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xa040c570  // ld1w { z16.s-z19.s }, pn9.b/Z, [x11]\n"
      ".inst 0xc0860404  // mova { z4.s-z7.s }, za0h.s[x12]\n"
      ".inst 0xc0840600  // mova za0h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xc086042c  // mova { z12.s-z15.s }, za1h.s[x12]\n"
      ".inst 0xa041c570  // ld1w { z16.s-z19.s }, pn9.b/Z, [x11, #0x4, MUL VL]\n"
      ".inst 0xc0840601  // mova za1h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xc086045c  // mova { z28.s-z31.s }, za2h.s[x12]\n"
      ".inst 0xc0860470  // mova { z16.s-z19.s }, za3h.s[x12]\n"
      ".inst 0xa042c560  // ld1w { z0.s-z3.s }, pn9.b/Z, [x11, #0x8, MUL VL]\n"
      ".inst 0xc0840402  // mova za2h.s[x12], { z0.s-z3.s }\n"
      ".inst 0xa043c574  // ld1w { z20.s-z23.s }, pn9.b/Z, [x11, #0xc, MUL VL]\n"
      ".inst 0xc0840683  // mova za3h.s[x12], { z20.s-z23.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      ".inst 0xa060c544  // st1w { z4.s-z7.s }, pn9.b, [x10]\n"
      "addvl x11, x11, #16\n"
      ".inst 0xa061c54c  // st1w { z12.s-z15.s }, pn9.b, [x10, #0x4, MUL VL]\n"
      ".inst 0xa062c55c  // st1w { z28.s-z31.s }, pn9.b, [x10, #0x8, MUL VL]\n"
      ".inst 0xa063c550  // st1w { z16.s-z19.s }, pn9.b, [x10, #0xc, MUL VL]\n"
      "addvl x10, x10, #16\n"
      "blt 11b\n"
      "b 20f\n"
      "12:"  // Store to partial result buffer: Store only
      "mov x12, #0x0\n"
      "cntw x20\n"
      "13:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xc086040c  // mova { z12.s-z15.s }, za0h.s[x12]\n"
      ".inst 0xc0860424  // mova { z4.s-z7.s }, za1h.s[x12]\n"
      ".inst 0xa060c54c  // st1w { z12.s-z15.s }, pn9.b, [x10]\n"
      ".inst 0xc0860440  // mova { z0.s-z3.s }, za2h.s[x12]\n"
      ".inst 0xc0860478  // mova { z24.s-z27.s }, za3h.s[x12]\n"
      ".inst 0xa061c544  // st1w { z4.s-z7.s }, pn9.b, [x10, #0x4, MUL VL]\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      ".inst 0xa062c540  // st1w { z0.s-z3.s }, pn9.b, [x10, #0x8, MUL VL]\n"
      ".inst 0xa063c558  // st1w { z24.s-z27.s }, pn9.b, [x10, #0xc, MUL VL]\n"
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
      ".inst 0xc0860410  // mova { z16.s-z19.s }, za0h.s[x12]\n"
      ".inst 0xc0860434  // mova { z20.s-z23.s }, za1h.s[x12]\n"
      ".inst 0xc0860458  // mova { z24.s-z27.s }, za2h.s[x12]\n"
      ".inst 0xc086047c  // mova { z28.s-z31.s }, za3h.s[x12]\n"
      ".inst 0xa160c2f0  // st1w { z16.s, z20.s, z24.s, z28.s }, p8, [x23]\n"
      "add x23, x23, x22\n"
      "beq 17f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xa160c2f1  // st1w { z17.s, z21.s, z25.s, z29.s }, p8, [x23]\n"
      "add x23, x23, x22\n"
      "beq 17f\n"
      ".inst 0xa160c2f2  // st1w { z18.s, z22.s, z26.s, z30.s }, p8, [x23]\n"
      "17:"  // Store to output array: Accumulator row 0 oddments: End
      "18:"  // Store to output array: End
      "tbz x13, #0, 20f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "19:"  // Store to output array: Refill accumulators: Loop
      ".inst 0xa040c568  // ld1w { z8.s-z11.s }, pn9.b/Z, [x11]\n"
      ".inst 0xc0840500  // mova za0h.s[x12], { z8.s-z11.s }\n"
      ".inst 0xa041c570  // ld1w { z16.s-z19.s }, pn9.b/Z, [x11, #0x4, MUL VL]\n"
      ".inst 0xc0840601  // mova za1h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xa042c570  // ld1w { z16.s-z19.s }, pn9.b/Z, [x11, #0x8, MUL VL]\n"
      ".inst 0xc0840602  // mova za2h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xa043c568  // ld1w { z8.s-z11.s }, pn9.b/Z, [x11, #0xc, MUL VL]\n"
      ".inst 0xc0840503  // mova za3h.s[x12], { z8.s-z11.s }\n"
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
