/*
 * Copyright (c) 2023-2024 Arm Limited.
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


#include "../../asmlib.hpp"
#include "../../utils.hpp"

namespace arm_gemm {

void sme2_interleaved_nomerge_fp16fp32fp16_mopa_2VLx2VL(const __fp16 *const A, const __fp16 *const B, __fp16 *const C, int ldc, const int M, const int N, const int K, const __fp16 *const bias, const Activation act, bool accumulate, float *const accumulator_buffer)
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
      "ldr x15, [%x[args], %[offsetof_flags]]\n"
      ".inst 0xd503477f  // SMSTART ZA\n"
      "ptrue p1.b\n"
      ".inst 0x25207810  // ptrue pn8.b\n"
      "ldr x14, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "ldr x13, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "tbz x15, #0, 2f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "1:"  // Initial accumulator load from buffer: Loop
      ".inst 0xa040c1d8  // ld1w { z24.s-z27.s }, pn8.b/Z, [x14]\n"
      ".inst 0xa041c1d4  // ld1w { z20.s-z23.s }, pn8.b/Z, [x14, #0x4, MUL VL]\n"
      ".inst 0xa042c1cc  // ld1w { z12.s-z15.s }, pn8.b/Z, [x14, #0x8, MUL VL]\n"
      ".inst 0xa043c1d0  // ld1w { z16.s-z19.s }, pn8.b/Z, [x14, #0xc, MUL VL]\n"
      ".inst 0xc0840700  // mova za0h.s[x12], { z24.s-z27.s }\n"
      "addvl x14, x14, #16\n"
      ".inst 0xc0840681  // mova za1h.s[x12], { z20.s-z23.s }\n"
      ".inst 0xc0840582  // mova za2h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xc0840603  // mova za3h.s[x12], { z16.s-z19.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 1b\n"
      "2:"  // Initial accumulator load from buffer: End
      "ldr w11, [%x[args], %[offsetof_M]]\n"
      "mov x10, #0x0\n"
      "mov x9, #0x0\n"
      "ldr w28, [%x[args], %[offsetof_N]]\n"
      "ldr x27, [%x[args], %[offsetof_A]]\n"
      "3:"  // M and N loop
      "mov x26, x27\n"
      "tbnz x15, #0, 4f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
      "cbz x20, 5f\n"
      "whilelt p0.h, x9, x28\n"
      "fmov z7.h, #0.0\n"
      "fmov z19.h, #1.0\n"
      "ld1h { z20.h }, p0/Z, [x20, x9, LSL #1]\n"
      "zip1 z21.h, z20.h, z7.h\n"
      "zip2 z30.h, z20.h, z7.h\n"
      ".inst 0x81b52660  // fmopa za0.s, p1/M, p1/M, z19.h, z21.h\n"
      ".inst 0x81be2661  // fmopa za1.s, p1/M, p1/M, z19.h, z30.h\n"
      ".inst 0x81b52662  // fmopa za2.s, p1/M, p1/M, z19.h, z21.h\n"
      ".inst 0x81be2663  // fmopa za3.s, p1/M, p1/M, z19.h, z30.h\n"
      "4:"  // Prepare accumulators: Test for last block
      "mov x20, x9\n"
      "mov x21, x10\n"
      "incw x20, ALL, MUL #2\n"
      "incw x21, ALL, MUL #2\n"
      "cmp x20, x28\n"
      "mov x20, x15\n"
      "csel x21, x10, x21, LT\n"
      "bfm x15, XZR, #0x0, #0x0  // bfc x15, #0x0, #0x1\n"
      "cmp x21, x11\n"
      "csel x15, x20, x15, LT\n"
      "5:"  // Prepare accumulators: End
      "ldr x20, [%x[args], %[offsetof_K]]\n"
      "ldr x23, [%x[args], %[offsetof_B]]\n"
      "ldr x22, [%x[args], %[offsetof_kstride_bytes]]\n"
      "add x20, x20, #0x1\n"
      "lsr x20, x20, #0x1\n"
      "lsr x21, x20, #0x2\n"
      "madd x23, x9, x22, x23\n"  // bptr = B + n * kstride_bytes
      "and x20, x20, #0x3\n"
      "cbz x21, 8f\n"
      "subs x21, x21, #0x1\n"
      ".inst 0xa0402344  // ld1h { z4.h-z5.h }, pn8.b/Z, [x26]\n"
      ".inst 0xa14022f1  // ld1h { z17.h, z25.h }, pn8.b/Z, [x23]\n"
      ".inst 0xa0412352  // ld1h { z18.h-z19.h }, pn8.b/Z, [x26, #0x2, MUL VL]\n"
      ".inst 0xa14122e3  // ld1h { z3.h, z11.h }, pn8.b/Z, [x23, #0x2, MUL VL]\n"
      ".inst 0xa042234c  // ld1h { z12.h-z13.h }, pn8.b/Z, [x26, #0x4, MUL VL]\n"
      ".inst 0xa04222fc  // ld1h { z28.h-z29.h }, pn8.b/Z, [x23, #0x4, MUL VL]\n"
      ".inst 0xa1432347  // ld1h { z7.h, z15.h }, pn8.b/Z, [x26, #0x6, MUL VL]\n"
      "addvl x26, x26, #8\n"
      ".inst 0xa14322f7  // ld1h { z23.h, z31.h }, pn8.b/Z, [x23, #0x6, MUL VL]\n"
      "addvl x23, x23, #8\n"
      "ble 7f\n"
      "6:"  // K loop
      ".inst 0x81b12480  // fmopa za0.s, p1/M, p1/M, z4.h, z17.h\n"
      "subs x21, x21, #0x1\n"
      ".inst 0x81b92481  // fmopa za1.s, p1/M, p1/M, z4.h, z25.h\n"
      ".inst 0x81b124a2  // fmopa za2.s, p1/M, p1/M, z5.h, z17.h\n"
      ".inst 0x81b924a3  // fmopa za3.s, p1/M, p1/M, z5.h, z25.h\n"
      ".inst 0xa0402344  // ld1h { z4.h-z5.h }, pn8.b/Z, [x26]\n"
      ".inst 0x81a32640  // fmopa za0.s, p1/M, p1/M, z18.h, z3.h\n"
      ".inst 0xa14022f1  // ld1h { z17.h, z25.h }, pn8.b/Z, [x23]\n"
      ".inst 0x81ab2641  // fmopa za1.s, p1/M, p1/M, z18.h, z11.h\n"
      ".inst 0x81a32662  // fmopa za2.s, p1/M, p1/M, z19.h, z3.h\n"
      ".inst 0x81ab2663  // fmopa za3.s, p1/M, p1/M, z19.h, z11.h\n"
      ".inst 0xa0412352  // ld1h { z18.h-z19.h }, pn8.b/Z, [x26, #0x2, MUL VL]\n"
      ".inst 0x81bc2580  // fmopa za0.s, p1/M, p1/M, z12.h, z28.h\n"
      ".inst 0xa14122e3  // ld1h { z3.h, z11.h }, pn8.b/Z, [x23, #0x2, MUL VL]\n"
      ".inst 0x81bd2581  // fmopa za1.s, p1/M, p1/M, z12.h, z29.h\n"
      ".inst 0x81bc25a2  // fmopa za2.s, p1/M, p1/M, z13.h, z28.h\n"
      ".inst 0x81bd25a3  // fmopa za3.s, p1/M, p1/M, z13.h, z29.h\n"
      ".inst 0xa042234c  // ld1h { z12.h-z13.h }, pn8.b/Z, [x26, #0x4, MUL VL]\n"
      ".inst 0xa04222fc  // ld1h { z28.h-z29.h }, pn8.b/Z, [x23, #0x4, MUL VL]\n"
      ".inst 0x81b724e0  // fmopa za0.s, p1/M, p1/M, z7.h, z23.h\n"
      ".inst 0x81bf24e1  // fmopa za1.s, p1/M, p1/M, z7.h, z31.h\n"
      ".inst 0x81b725e2  // fmopa za2.s, p1/M, p1/M, z15.h, z23.h\n"
      ".inst 0x81bf25e3  // fmopa za3.s, p1/M, p1/M, z15.h, z31.h\n"
      ".inst 0xa1432347  // ld1h { z7.h, z15.h }, pn8.b/Z, [x26, #0x6, MUL VL]\n"
      "addvl x26, x26, #8\n"
      ".inst 0xa14322f7  // ld1h { z23.h, z31.h }, pn8.b/Z, [x23, #0x6, MUL VL]\n"
      "addvl x23, x23, #8\n"
      "bgt 6b\n"
      "7:"  // K loop tail
      ".inst 0x81b12480  // fmopa za0.s, p1/M, p1/M, z4.h, z17.h\n"
      ".inst 0x81b92481  // fmopa za1.s, p1/M, p1/M, z4.h, z25.h\n"
      ".inst 0x81b124a2  // fmopa za2.s, p1/M, p1/M, z5.h, z17.h\n"
      ".inst 0x81b924a3  // fmopa za3.s, p1/M, p1/M, z5.h, z25.h\n"
      ".inst 0x81a32640  // fmopa za0.s, p1/M, p1/M, z18.h, z3.h\n"
      ".inst 0x81ab2641  // fmopa za1.s, p1/M, p1/M, z18.h, z11.h\n"
      ".inst 0x81a32662  // fmopa za2.s, p1/M, p1/M, z19.h, z3.h\n"
      ".inst 0x81ab2663  // fmopa za3.s, p1/M, p1/M, z19.h, z11.h\n"
      ".inst 0x81bc2580  // fmopa za0.s, p1/M, p1/M, z12.h, z28.h\n"
      ".inst 0x81bd2581  // fmopa za1.s, p1/M, p1/M, z12.h, z29.h\n"
      ".inst 0x81bc25a2  // fmopa za2.s, p1/M, p1/M, z13.h, z28.h\n"
      ".inst 0x81bd25a3  // fmopa za3.s, p1/M, p1/M, z13.h, z29.h\n"
      ".inst 0x81b724e0  // fmopa za0.s, p1/M, p1/M, z7.h, z23.h\n"
      ".inst 0x81bf24e1  // fmopa za1.s, p1/M, p1/M, z7.h, z31.h\n"
      ".inst 0x81b725e2  // fmopa za2.s, p1/M, p1/M, z15.h, z23.h\n"
      ".inst 0x81bf25e3  // fmopa za3.s, p1/M, p1/M, z15.h, z31.h\n"
      "8:"  // K oddments
      "cbz x20, 10f\n"
      "9:"  // K oddments: Loop
      ".inst 0xa0402346  // ld1h { z6.h-z7.h }, pn8.b/Z, [x26]\n"
      "subs x20, x20, #0x1\n"
      "addvl x26, x26, #2\n"
      ".inst 0xa04022e0  // ld1h { z0.h-z1.h }, pn8.b/Z, [x23]\n"
      "addvl x23, x23, #2\n"
      ".inst 0x81a024c0  // fmopa za0.s, p1/M, p1/M, z6.h, z0.h\n"
      ".inst 0x81a124c1  // fmopa za1.s, p1/M, p1/M, z6.h, z1.h\n"
      ".inst 0x81a024e2  // fmopa za2.s, p1/M, p1/M, z7.h, z0.h\n"
      ".inst 0x81a124e3  // fmopa za3.s, p1/M, p1/M, z7.h, z1.h\n"
      "bgt 9b\n"
      "10:"  // K oddments: End
      "tbz x15, #1, 14f\n"
      "tbz x15, #0, 12f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "11:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xa040c1d4  // ld1w { z20.s-z23.s }, pn8.b/Z, [x14]\n"
      ".inst 0xc086040c  // mova { z12.s-z15.s }, za0h.s[x12]\n"
      ".inst 0xc0860430  // mova { z16.s-z19.s }, za1h.s[x12]\n"
      ".inst 0xa041c1c4  // ld1w { z4.s-z7.s }, pn8.b/Z, [x14, #0x4, MUL VL]\n"
      ".inst 0xc0860458  // mova { z24.s-z27.s }, za2h.s[x12]\n"
      ".inst 0xc0860468  // mova { z8.s-z11.s }, za3h.s[x12]\n"
      ".inst 0xa042c1dc  // ld1w { z28.s-z31.s }, pn8.b/Z, [x14, #0x8, MUL VL]\n"
      ".inst 0xa043c1c0  // ld1w { z0.s-z3.s }, pn8.b/Z, [x14, #0xc, MUL VL]\n"
      ".inst 0xc0840680  // mova za0h.s[x12], { z20.s-z23.s }\n"
      "addvl x14, x14, #16\n"
      ".inst 0xc0840481  // mova za1h.s[x12], { z4.s-z7.s }\n"
      ".inst 0xa060c1ac  // st1w { z12.s-z15.s }, pn8.b, [x13]\n"
      ".inst 0xc0840782  // mova za2h.s[x12], { z28.s-z31.s }\n"
      ".inst 0xa061c1b0  // st1w { z16.s-z19.s }, pn8.b, [x13, #0x4, MUL VL]\n"
      ".inst 0xc0840403  // mova za3h.s[x12], { z0.s-z3.s }\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa062c1b8  // st1w { z24.s-z27.s }, pn8.b, [x13, #0x8, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa063c1a8  // st1w { z8.s-z11.s }, pn8.b, [x13, #0xc, MUL VL]\n"
      "addvl x13, x13, #16\n"
      "blt 11b\n"
      "b 18f\n"
      "12:"  // Store to partial result buffer: Store only
      "mov x12, #0x0\n"
      "cntw x20\n"
      "13:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xc0860408  // mova { z8.s-z11.s }, za0h.s[x12]\n"
      ".inst 0xc086042c  // mova { z12.s-z15.s }, za1h.s[x12]\n"
      ".inst 0xc086045c  // mova { z28.s-z31.s }, za2h.s[x12]\n"
      ".inst 0xc0860470  // mova { z16.s-z19.s }, za3h.s[x12]\n"
      ".inst 0xa060c1a8  // st1w { z8.s-z11.s }, pn8.b, [x13]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa061c1ac  // st1w { z12.s-z15.s }, pn8.b, [x13, #0x4, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa062c1bc  // st1w { z28.s-z31.s }, pn8.b, [x13, #0x8, MUL VL]\n"
      ".inst 0xa063c1b0  // st1w { z16.s-z19.s }, pn8.b, [x13, #0xc, MUL VL]\n"
      "addvl x13, x13, #16\n"
      "blt 13b\n"
      "b 18f\n"
      "14:"  // Store to output array
      "ldr x25, [%x[args], %[offsetof_C]]\n"
      "sub x24, x11, x10\n"
      "cntw x23, ALL, MUL #2\n"
      "ld1rh { z18.h }, p1/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
      "ldr x22, [%x[args], %[offsetof_ldcb]]\n"
      "whilelt p0.h, x9, x28\n"
      "cmp x24, x23\n"
      "ld1rh { z17.h }, p1/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
      "mov x12, #0x0\n"
      "mov x21, #0x0\n"
      "add x25, x25, x9, LSL #1\n"  // C += n
      "mov x20, #0x2\n"
      "madd x25, x10, x22, x25\n"  // C += m * ldc
      "csel x24, x24, x23, LT\n"
      "15:"  // Store to output array: Accumulator loop
      ".inst 0xc006000e  // mova { z14.b-z15.b }, za0h.b[x12, 0:1]\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x23, LSL #1\n"
      "add x21, x21, #0x1\n"
      ".inst 0xc120e1d0  // fcvt z16.h, { z14.s-z15.s }\n"
      "csel x12, x12, x20, LT\n"
      "cmp x21, x24\n"
      ".inst 0x64712650  // fclamp z16.h, z18.h, z17.h\n"
      "st1h { z16.h }, p0, [x25]\n"
      "add x25, x25, x22\n"
      "blt 15b\n"
      "16:"  // Store to output array: End
      "tbz x15, #0, 18f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "17:"  // Store to output array: Refill accumulators: Loop
      ".inst 0xa040c1cc  // ld1w { z12.s-z15.s }, pn8.b/Z, [x14]\n"
      ".inst 0xa041c1c4  // ld1w { z4.s-z7.s }, pn8.b/Z, [x14, #0x4, MUL VL]\n"
      ".inst 0xa042c1c8  // ld1w { z8.s-z11.s }, pn8.b/Z, [x14, #0x8, MUL VL]\n"
      ".inst 0xa043c1dc  // ld1w { z28.s-z31.s }, pn8.b/Z, [x14, #0xc, MUL VL]\n"
      ".inst 0xc0840580  // mova za0h.s[x12], { z12.s-z15.s }\n"
      "addvl x14, x14, #16\n"
      ".inst 0xc0840481  // mova za1h.s[x12], { z4.s-z7.s }\n"
      ".inst 0xc0840502  // mova za2h.s[x12], { z8.s-z11.s }\n"
      ".inst 0xc0840783  // mova za3h.s[x12], { z28.s-z31.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 17b\n"
      "18:"  // End block
      "incw x9, ALL, MUL #2\n"
      "cmp x9, x28\n"
      "blt 3b\n"
      "incw x10, ALL, MUL #2\n"
      "mov x9, #0x0\n"
      "cmp x10, x11\n"
      "mov x27, x26\n"
      "blt 3b\n"
      ".inst 0xd503467f  // SMSTOP\n"
      :
      : [args] "r" (&args), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_KernelArgs_max] "I" (offsetof(KernelArgs, max)), [offsetof_KernelArgs_min] "I" (offsetof(KernelArgs, min)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_kstride_bytes] "I" (offsetof(KernelArgs, kstride_bytes)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb))
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

}  // namespace arm_gemm

#endif  // ARM_COMPUTE_ENABLE_SME2
