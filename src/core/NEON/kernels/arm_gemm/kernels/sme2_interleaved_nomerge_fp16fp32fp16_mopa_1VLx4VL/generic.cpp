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

#include "arm_gemm.hpp"


#include "../../asmlib.hpp"
#include "../../utils.hpp"

namespace arm_gemm {

void sme2_interleaved_nomerge_fp16fp32fp16_mopa_1VLx4VL(const __fp16 *const A, const __fp16 *const B, __fp16 *const C, int ldc, const int M, const int N, const int K, const __fp16 *const bias, const Activation act, bool accumulate, float *const accumulator_buffer)
{
  struct KernelArgs
  {
    KernelArgs(
      const __fp16 *const A,
      const __fp16 *const B,
      __fp16 *const C, const int ldc,
      const int M, const int N, const int K,
      const __fp16 *const bias,
      const Activation act,
      bool accumulate,
      float *const accumulator_buffer
    ) : A(A),
        B(B), kstride_bytes(roundup(K, 2) * sizeof(__fp16)),
        C(C), ldcb(ldc * sizeof(__fp16)),
        M(M), N(N), K(K),
        min(-static_cast<__fp16>(std::numeric_limits<float>::infinity())),
        max(static_cast<__fp16>(std::numeric_limits<float>::infinity())),
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

      // Initialise the activation values
      switch (act.type)
      {
        default:
        case Activation::Type::None:
            break;
        case Activation::Type::BoundedReLU:
            this->max = static_cast<__fp16>(act.param1);
            /* fall through */
        case Activation::Type::ReLU:
            this->min = static_cast<__fp16>(0);
            break;
      }
    }

    const __fp16 *const A;
    const __fp16 *const B;
    const long kstride_bytes;
    __fp16 *const C;
    const long ldcb;
    const long M, N, K;
    __fp16 min = -static_cast<__fp16>(std::numeric_limits<float>::infinity());
    __fp16 max = static_cast<__fp16>(std::numeric_limits<float>::infinity());

    const __fp16 *const bias;

    float *const accumulator_buffer;
    uint64_t flags;
  };

  // Construct arguments for this kernel
  KernelArgs args(A, B, C, ldc, M, N, K, bias, act, accumulate, accumulator_buffer);

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
      ".inst 0xa040c578  // ld1w { z24.s-z27.s }, pn9.b/Z, [x11]\n"
      ".inst 0xa041c570  // ld1w { z16.s-z19.s }, pn9.b/Z, [x11, #0x4, MUL VL]\n"
      ".inst 0xa042c564  // ld1w { z4.s-z7.s }, pn9.b/Z, [x11, #0x8, MUL VL]\n"
      ".inst 0xa043c574  // ld1w { z20.s-z23.s }, pn9.b/Z, [x11, #0xc, MUL VL]\n"
      ".inst 0xc0840700  // mova za0h.s[x12], { z24.s-z27.s }\n"
      "addvl x11, x11, #16\n"
      ".inst 0xc0840601  // mova za1h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xc0840482  // mova za2h.s[x12], { z4.s-z7.s }\n"
      ".inst 0xc0840683  // mova za3h.s[x12], { z20.s-z23.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 1b\n"
      "2:"  // Initial accumulator load from buffer: End
      "ldr w9, [%x[args], %[offsetof_M]]\n"
      "mov x28, #0x0\n"
      "mov x27, #0x0\n"
      "ldr w26, [%x[args], %[offsetof_N]]\n"
      "ldr x25, [%x[args], %[offsetof_A]]\n"
      "3:"  // M and N loop
      "mov x24, x25\n"
      "tbnz x13, #0, 4f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
      "cbz x20, 5f\n"
      ".inst 0x257a4770  // whilelt pn8.h, x27, x26, VLx2\n"
      "fmov z6.h, #0.0\n"
      "fmov z19.h, #1.0\n"
      ".inst 0xa01b2295  // ldnt1h { z20.h-z21.h }, p8/Z, [x20, x27, LSL #1]\n"
      "zip1 z23.h, z20.h, z6.h\n"
      "zip2 z12.h, z20.h, z6.h\n"
      "zip1 z16.h, z21.h, z6.h\n"
      "zip2 z8.h, z21.h, z6.h\n"
      ".inst 0x81b70260  // fmopa za0.s, p0/M, p0/M, z19.h, z23.h\n"
      ".inst 0x81ac0261  // fmopa za1.s, p0/M, p0/M, z19.h, z12.h\n"
      ".inst 0x81b00262  // fmopa za2.s, p0/M, p0/M, z19.h, z16.h\n"
      ".inst 0x81a80263  // fmopa za3.s, p0/M, p0/M, z19.h, z8.h\n"
      "4:"  // Prepare accumulators: Test for last block
      "mov x20, x27\n"
      "mov x21, x28\n"
      "incw x20, ALL, MUL #4\n"
      "incw x21\n"
      "cmp x20, x26\n"
      "mov x20, x13\n"
      "csel x21, x28, x21, LT\n"
      "bfm x13, XZR, #0x0, #0x0  // bfc x13, #0x0, #0x1\n"
      "cmp x21, x9\n"
      "csel x13, x20, x13, LT\n"
      "5:"  // Prepare accumulators: End
      "ldr x20, [%x[args], %[offsetof_K]]\n"
      "ldr x23, [%x[args], %[offsetof_B]]\n"
      "ldr x22, [%x[args], %[offsetof_kstride_bytes]]\n"
      "add x20, x20, #0x1\n"
      "lsr x20, x20, #0x1\n"
      "lsr x21, x20, #0x2\n"
      "and x20, x20, #0x3\n"
      "madd x23, x27, x22, x23\n"  // bptr = B + n * kstride_bytes
      "cbz x21, 8f\n"
      "subs x21, x21, #0x1\n"
      "ld1h { z21.h }, p0/Z, [x24]\n"
      ".inst 0xa140a6f8  // ldnt1h { z16.h, z20.h, z24.h, z28.h }, pn9.b/Z, [x23]\n"
      "ld1h { z29.h }, p0/Z, [x24, #1, MUL VL]\n"
      ".inst 0xa041a6ed  // ldnt1h { z12.h-z15.h }, pn9.b/Z, [x23, #0x4, MUL VL]\n"
      "ld1h { z4.h }, p0/Z, [x24, #2, MUL VL]\n"
      ".inst 0xa042a6e1  // ldnt1h { z0.h-z3.h }, pn9.b/Z, [x23, #0x8, MUL VL]\n"
      "ld1h { z25.h }, p0/Z, [x24, #3, MUL VL]\n"
      "addvl x24, x24, #4\n"
      ".inst 0xa143a6fb  // ldnt1h { z19.h, z23.h, z27.h, z31.h }, pn9.b/Z, [x23, #0xc, MUL VL]\n"
      "addvl x23, x23, #16\n"
      "ble 7f\n"
      "6:"  // K loop
      ".inst 0x81b002a0  // fmopa za0.s, p0/M, p0/M, z21.h, z16.h\n"
      "subs x21, x21, #0x1\n"
      ".inst 0x81b402a1  // fmopa za1.s, p0/M, p0/M, z21.h, z20.h\n"
      ".inst 0x81b802a2  // fmopa za2.s, p0/M, p0/M, z21.h, z24.h\n"
      ".inst 0x81bc02a3  // fmopa za3.s, p0/M, p0/M, z21.h, z28.h\n"
      "ld1h { z21.h }, p0/Z, [x24]\n"
      ".inst 0x81ac03a0  // fmopa za0.s, p0/M, p0/M, z29.h, z12.h\n"
      ".inst 0xa140a6f0  // ld1h { z16.h, z20.h, z24.h, z28.h }, pn9.b/Z, [x23]\n"
      ".inst 0x81ad03a1  // fmopa za1.s, p0/M, p0/M, z29.h, z13.h\n"
      ".inst 0x81ae03a2  // fmopa za2.s, p0/M, p0/M, z29.h, z14.h\n"
      ".inst 0x81af03a3  // fmopa za3.s, p0/M, p0/M, z29.h, z15.h\n"
      "ld1h { z29.h }, p0/Z, [x24, #1, MUL VL]\n"
      ".inst 0x81a00080  // fmopa za0.s, p0/M, p0/M, z4.h, z0.h\n"
      ".inst 0xa041a6ec  // ld1h { z12.h-z15.h }, pn9.b/Z, [x23, #0x4, MUL VL]\n"
      ".inst 0x81a10081  // fmopa za1.s, p0/M, p0/M, z4.h, z1.h\n"
      ".inst 0x81a20082  // fmopa za2.s, p0/M, p0/M, z4.h, z2.h\n"
      ".inst 0x81a30083  // fmopa za3.s, p0/M, p0/M, z4.h, z3.h\n"
      "ld1h { z4.h }, p0/Z, [x24, #2, MUL VL]\n"
      ".inst 0xa042a6e0  // ld1h { z0.h-z3.h }, pn9.b/Z, [x23, #0x8, MUL VL]\n"
      ".inst 0x81b30320  // fmopa za0.s, p0/M, p0/M, z25.h, z19.h\n"
      ".inst 0x81b70321  // fmopa za1.s, p0/M, p0/M, z25.h, z23.h\n"
      ".inst 0x81bb0322  // fmopa za2.s, p0/M, p0/M, z25.h, z27.h\n"
      ".inst 0x81bf0323  // fmopa za3.s, p0/M, p0/M, z25.h, z31.h\n"
      "ld1h { z25.h }, p0/Z, [x24, #3, MUL VL]\n"
      "addvl x24, x24, #4\n"
      ".inst 0xa143a6f3  // ld1h { z19.h, z23.h, z27.h, z31.h }, pn9.b/Z, [x23, #0xc, MUL VL]\n"
      "addvl x23, x23, #16\n"
      "bgt 6b\n"
      "7:"  // K loop tail
      ".inst 0x81b002a0  // fmopa za0.s, p0/M, p0/M, z21.h, z16.h\n"
      ".inst 0x81b402a1  // fmopa za1.s, p0/M, p0/M, z21.h, z20.h\n"
      ".inst 0x81b802a2  // fmopa za2.s, p0/M, p0/M, z21.h, z24.h\n"
      ".inst 0x81bc02a3  // fmopa za3.s, p0/M, p0/M, z21.h, z28.h\n"
      ".inst 0x81ac03a0  // fmopa za0.s, p0/M, p0/M, z29.h, z12.h\n"
      ".inst 0x81ad03a1  // fmopa za1.s, p0/M, p0/M, z29.h, z13.h\n"
      ".inst 0x81ae03a2  // fmopa za2.s, p0/M, p0/M, z29.h, z14.h\n"
      ".inst 0x81af03a3  // fmopa za3.s, p0/M, p0/M, z29.h, z15.h\n"
      ".inst 0x81a00080  // fmopa za0.s, p0/M, p0/M, z4.h, z0.h\n"
      ".inst 0x81a10081  // fmopa za1.s, p0/M, p0/M, z4.h, z1.h\n"
      ".inst 0x81a20082  // fmopa za2.s, p0/M, p0/M, z4.h, z2.h\n"
      ".inst 0x81a30083  // fmopa za3.s, p0/M, p0/M, z4.h, z3.h\n"
      ".inst 0x81b30320  // fmopa za0.s, p0/M, p0/M, z25.h, z19.h\n"
      ".inst 0x81b70321  // fmopa za1.s, p0/M, p0/M, z25.h, z23.h\n"
      ".inst 0x81bb0322  // fmopa za2.s, p0/M, p0/M, z25.h, z27.h\n"
      ".inst 0x81bf0323  // fmopa za3.s, p0/M, p0/M, z25.h, z31.h\n"
      "8:"  // K oddments
      "cbz x20, 10f\n"
      "9:"  // K oddments: Loop
      "ld1h { z21.h }, p0/Z, [x24]\n"
      "subs x20, x20, #0x1\n"
      "addvl x24, x24, #1\n"
      ".inst 0xa140a6f0  // ld1h { z16.h, z20.h, z24.h, z28.h }, pn9.b/Z, [x23]\n"
      "addvl x23, x23, #4\n"
      ".inst 0x81b002a0  // fmopa za0.s, p0/M, p0/M, z21.h, z16.h\n"
      ".inst 0x81b402a1  // fmopa za1.s, p0/M, p0/M, z21.h, z20.h\n"
      ".inst 0x81b802a2  // fmopa za2.s, p0/M, p0/M, z21.h, z24.h\n"
      ".inst 0x81bc02a3  // fmopa za3.s, p0/M, p0/M, z21.h, z28.h\n"
      "bgt 9b\n"
      "10:"  // K oddments: End
      "tbz x13, #1, 14f\n"
      "tbz x13, #0, 12f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "11:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xa040c574  // ld1w { z20.s-z23.s }, pn9.b/Z, [x11]\n"
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      ".inst 0xc0860438  // mova { z24.s-z27.s }, za1h.s[x12]\n"
      ".inst 0xa041c56c  // ld1w { z12.s-z15.s }, pn9.b/Z, [x11, #0x4, MUL VL]\n"
      ".inst 0xc0860444  // mova { z4.s-z7.s }, za2h.s[x12]\n"
      ".inst 0xc086047c  // mova { z28.s-z31.s }, za3h.s[x12]\n"
      ".inst 0xa042c568  // ld1w { z8.s-z11.s }, pn9.b/Z, [x11, #0x8, MUL VL]\n"
      ".inst 0xa043c570  // ld1w { z16.s-z19.s }, pn9.b/Z, [x11, #0xc, MUL VL]\n"
      ".inst 0xc0840680  // mova za0h.s[x12], { z20.s-z23.s }\n"
      "addvl x11, x11, #16\n"
      ".inst 0xc0840581  // mova za1h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xa060c540  // st1w { z0.s-z3.s }, pn9.b, [x10]\n"
      ".inst 0xc0840502  // mova za2h.s[x12], { z8.s-z11.s }\n"
      ".inst 0xa061c558  // st1w { z24.s-z27.s }, pn9.b, [x10, #0x4, MUL VL]\n"
      ".inst 0xc0840603  // mova za3h.s[x12], { z16.s-z19.s }\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa062c544  // st1w { z4.s-z7.s }, pn9.b, [x10, #0x8, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa063c55c  // st1w { z28.s-z31.s }, pn9.b, [x10, #0xc, MUL VL]\n"
      "addvl x10, x10, #16\n"
      "blt 11b\n"
      "b 18f\n"
      "12:"  // Store to partial result buffer: Store only
      "mov x12, #0x0\n"
      "cntw x20\n"
      "13:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      ".inst 0xc0860428  // mova { z8.s-z11.s }, za1h.s[x12]\n"
      ".inst 0xc086044c  // mova { z12.s-z15.s }, za2h.s[x12]\n"
      ".inst 0xc0860464  // mova { z4.s-z7.s }, za3h.s[x12]\n"
      ".inst 0xa060c540  // st1w { z0.s-z3.s }, pn9.b, [x10]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa061c548  // st1w { z8.s-z11.s }, pn9.b, [x10, #0x4, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa062c54c  // st1w { z12.s-z15.s }, pn9.b, [x10, #0x8, MUL VL]\n"
      ".inst 0xa063c544  // st1w { z4.s-z7.s }, pn9.b, [x10, #0xc, MUL VL]\n"
      "addvl x10, x10, #16\n"
      "blt 13b\n"
      "b 18f\n"
      "14:"  // Store to output array
      "ldr x23, [%x[args], %[offsetof_C]]\n"
      "sub x22, x9, x28\n"
      "cntw x21\n"
      "ld1rh { z17.h }, p0/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
      "ldr x20, [%x[args], %[offsetof_ldcb]]\n"
      ".inst 0x257a4770  // whilelt pn8.h, x27, x26, VLx2\n"
      "cmp x22, x21\n"
      "ld1rh { z16.h }, p0/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
      "mov x12, #0x0\n"
      "csel x22, x22, x21, LT\n"
      "add x23, x23, x27, LSL #1\n"  // C += n
      "madd x23, x28, x20, x23\n"  // C += m * ldc
      "15:"  // Store to output array: Accumulator loop
      ".inst 0xc0060414  // mova { z20.b-z23.b }, za0h.b[x12, 0:3]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xc120e28e  // fcvt z14.h, { z20.s-z21.s }\n"
      ".inst 0xc120e2cf  // fcvt z15.h, { z22.s-z23.s }\n"
      "cmp x12, x22, LSL #2\n"
      ".inst 0xc170c22e  // fclamp { z14.h-z15.h }, z17.h, z16.h\n"
      ".inst 0xa06022ee  // st1h { z14.h-z15.h }, p8, [x23]\n"
      "add x23, x23, x20\n"
      "blt 15b\n"
      "16:"  // Store to output array: End
      "tbz x13, #0, 18f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "17:"  // Store to output array: Refill accumulators: Loop
      ".inst 0xa040c578  // ld1w { z24.s-z27.s }, pn9.b/Z, [x11]\n"
      ".inst 0xa041c574  // ld1w { z20.s-z23.s }, pn9.b/Z, [x11, #0x4, MUL VL]\n"
      ".inst 0xa042c570  // ld1w { z16.s-z19.s }, pn9.b/Z, [x11, #0x8, MUL VL]\n"
      ".inst 0xa043c57c  // ld1w { z28.s-z31.s }, pn9.b/Z, [x11, #0xc, MUL VL]\n"
      ".inst 0xc0840700  // mova za0h.s[x12], { z24.s-z27.s }\n"
      "addvl x11, x11, #16\n"
      ".inst 0xc0840681  // mova za1h.s[x12], { z20.s-z23.s }\n"
      ".inst 0xc0840602  // mova za2h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xc0840783  // mova za3h.s[x12], { z28.s-z31.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 17b\n"
      "18:"  // End block
      "incw x27, ALL, MUL #4\n"
      "cmp x27, x26\n"
      "blt 3b\n"
      "incw x28\n"
      "mov x27, #0x0\n"
      "cmp x28, x9\n"
      "mov x25, x24\n"
      "blt 3b\n"
      ".inst 0xd503467f  // SMSTOP\n"
      :
      : [args] "r" (&args), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_KernelArgs_max] "I" (offsetof(KernelArgs, max)), [offsetof_KernelArgs_min] "I" (offsetof(KernelArgs, min)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_kstride_bytes] "I" (offsetof(KernelArgs, kstride_bytes)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb))
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

}  // namespace arm_gemm

#endif  // __ARM_FEATURE_SVE
