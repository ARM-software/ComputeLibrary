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

#include "../../bfloat.hpp"
#include "../../asmlib.hpp"
#include "../../utils.hpp"

namespace arm_gemm {

void sme2_interleaved_nomerge_bf16fp32_mopa_1VLx4VL(const bfloat16 *const A, const bfloat16 *const B, float *const C, int ldc, const int M, const int N, const int K, const float *const bias, const Activation act, bool accumulate, float *const accumulator_buffer)
{
  struct KernelArgs
  {
    KernelArgs(
      const bfloat16 *const A,
      const bfloat16 *const B,
      float *const C, const int ldc,
      const int M, const int N, const int K,
      const float *const bias,
      const Activation act,
      bool accumulate,
      float *const accumulator_buffer
    ) : A(A),
        B(B), kstride_bytes(roundup(K, 2) * sizeof(bfloat16)),
        C(C), ldcb(ldc * sizeof(float)),
        M(M), N(N), K(K),
        n_loops(((K / 2) - 1) / 2), n_tail_iters(((K / 2) - 1) % 2),
        min(-std::numeric_limits<float>::infinity()),
        max(std::numeric_limits<float>::infinity()),
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
      if (act.type == Activation::Type::None)
      {
        flags |= 1 << 2;  // SKIP_ACTIVATION
      }

      // Initialise the activation values
      switch (act.type)
      {
        default:
        case Activation::Type::None:
            break;
        case Activation::Type::BoundedReLU:
            this->max = static_cast<float>(act.param1);
            /* fall through */
        case Activation::Type::ReLU:
            this->min = static_cast<float>(0);
            break;
      }
    }

    const bfloat16 *const A;
    const bfloat16 *const B;
    const long kstride_bytes;
    float *const C;
    const long ldcb;
    const long M, N, K, n_loops, n_tail_iters;
    float min = -std::numeric_limits<float>::infinity();
    float max = std::numeric_limits<float>::infinity();

    const float *const bias;

    float *const accumulator_buffer;
    uint64_t flags;
  };

  // Construct arguments for this kernel
  KernelArgs args(A, B, C, ldc, M, N, K, bias, act, accumulate, accumulator_buffer);

  __asm__ __volatile__(
      "ldr x14, [%x[args], %[offsetof_flags]]\n"
      ".inst 0xd503477f  // SMSTART ZA\n"
      "ptrue p0.b\n"
      ".inst 0x25207811  // ptrue pn9.b\n"
      "ldr x13, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "ldr x11, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "tbz x14, #0, 2f\n"
      "mov x12, #0x0\n"
      "cntw x19\n"
      "1:"  // Initial accumulator load from buffer: Loop
      ".inst 0xa040c5ac  // ld1w { z12.s-z15.s }, pn9.b/Z, [x13]\n"
      ".inst 0xc0840580  // mova za0h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xa041c5b0  // ld1w { z16.s-z19.s }, pn9.b/Z, [x13, #0x4, MUL VL]\n"
      ".inst 0xc0840601  // mova za1h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xa042c5a4  // ld1w { z4.s-z7.s }, pn9.b/Z, [x13, #0x8, MUL VL]\n"
      ".inst 0xc0840482  // mova za2h.s[x12], { z4.s-z7.s }\n"
      ".inst 0xa043c5a4  // ld1w { z4.s-z7.s }, pn9.b/Z, [x13, #0xc, MUL VL]\n"
      ".inst 0xc0840483  // mova za3h.s[x12], { z4.s-z7.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x19\n"
      "addvl x13, x13, #16\n"
      "blt 1b\n"
      "2:"  // Initial accumulator load from buffer: End
      "ldr w10, [%x[args], %[offsetof_M]]\n"
      "mov x9, #0x0\n"
      "mov x28, #0x0\n"
      "ldr w27, [%x[args], %[offsetof_N]]\n"
      "ldr x26, [%x[args], %[offsetof_A]]\n"
      "3:"  // M and N loop
      "mov x25, x26\n"
      ".inst 0x25bb6790  // whilelt pn8.s, x28, x27, VLx4\n"
      "tbnz x14, #0, 4f\n"
      "ldr x19, [%x[args], %[offsetof_bias]]\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
      "cbz x19, 5f\n"
      "fmov z21.s, #1.0\n"
      ".inst 0xa01cc27d  // ldnt1w { z28.s-z31.s }, p8/Z, [x19, x28, LSL #2]\n"
      ".inst 0x809c02a0  // fmopa za0.s, p0/M, p0/M, z21.s, z28.s\n"
      ".inst 0x809d02a1  // fmopa za1.s, p0/M, p0/M, z21.s, z29.s\n"
      ".inst 0x809e02a2  // fmopa za2.s, p0/M, p0/M, z21.s, z30.s\n"
      ".inst 0x809f02a3  // fmopa za3.s, p0/M, p0/M, z21.s, z31.s\n"
      "4:"  // Prepare accumulators: Test for last block
      "mov x19, x28\n"
      "mov x20, x9\n"
      "incw x19, ALL, MUL #4\n"
      "incw x20\n"
      "cmp x19, x27\n"
      "csel x20, x9, x20, LT\n"
      "mov x19, x14\n"
      "bfm x14, XZR, #0x0, #0x0  // bfc x14, #0x0, #0x1\n"
      "cmp x20, x10\n"
      "csel x14, x19, x14, LT\n"
      "5:"  // Prepare accumulators: End
      "ldr x19, [%x[args], %[offsetof_K]]\n"
      "add x19, x19, #0x1\n"
      "lsr x19, x19, #0x1\n"
      "ldr x22, [%x[args], %[offsetof_B]]\n"
      "lsr x21, x19, #0x2\n"
      "and x20, x19, #0x3\n"
      "ldr x19, [%x[args], %[offsetof_kstride_bytes]]\n"
      "madd x22, x28, x19, x22\n"  // bptr = B + n * kstride_bytes
      "cbz x21, 8f\n"
      "subs x21, x21, #0x1\n"
      "ld1h { z0.h }, p0/Z, [x25]\n"
      ".inst 0xa140a6db  // ldnt1h { z19.h, z23.h, z27.h, z31.h }, pn9.b/Z, [x22]\n"
      "ld1h { z13.h }, p0/Z, [x25, #1, MUL VL]\n"
      ".inst 0xa141a6ca  // ldnt1h { z2.h, z6.h, z10.h, z14.h }, pn9.b/Z, [x22, #0x4, MUL VL]\n"
      "ld1h { z12.h }, p0/Z, [x25, #2, MUL VL]\n"
      ".inst 0xa142a6cb  // ldnt1h { z3.h, z7.h, z11.h, z15.h }, pn9.b/Z, [x22, #0x8, MUL VL]\n"
      "ld1h { z26.h }, p0/Z, [x25, #3, MUL VL]\n"
      "addvl x25, x25, #4\n"
      ".inst 0xa143a6d8  // ldnt1h { z16.h, z20.h, z24.h, z28.h }, pn9.b/Z, [x22, #0xc, MUL VL]\n"
      "addvl x22, x22, #16\n"
      "ble 7f\n"
      "6:"  // K loop
      ".inst 0x81930000  // bfmopa za0.s, p0/M, p0/M, z0.h, z19.h\n"
      "subs x21, x21, #0x1\n"
      ".inst 0x81970001  // bfmopa za1.s, p0/M, p0/M, z0.h, z23.h\n"
      ".inst 0x819b0002  // bfmopa za2.s, p0/M, p0/M, z0.h, z27.h\n"
      ".inst 0x819f0003  // bfmopa za3.s, p0/M, p0/M, z0.h, z31.h\n"
      "ld1h { z0.h }, p0/Z, [x25]\n"
      ".inst 0x818201a0  // bfmopa za0.s, p0/M, p0/M, z13.h, z2.h\n"
      ".inst 0xa140a6db  // ldnt1h { z19.h, z23.h, z27.h, z31.h }, pn9.b/Z, [x22]\n"
      ".inst 0x818601a1  // bfmopa za1.s, p0/M, p0/M, z13.h, z6.h\n"
      ".inst 0x818a01a2  // bfmopa za2.s, p0/M, p0/M, z13.h, z10.h\n"
      ".inst 0x818e01a3  // bfmopa za3.s, p0/M, p0/M, z13.h, z14.h\n"
      "ld1h { z13.h }, p0/Z, [x25, #1, MUL VL]\n"
      ".inst 0x81830180  // bfmopa za0.s, p0/M, p0/M, z12.h, z3.h\n"
      ".inst 0xa141a6ca  // ldnt1h { z2.h, z6.h, z10.h, z14.h }, pn9.b/Z, [x22, #0x4, MUL VL]\n"
      ".inst 0x81870181  // bfmopa za1.s, p0/M, p0/M, z12.h, z7.h\n"
      ".inst 0x818b0182  // bfmopa za2.s, p0/M, p0/M, z12.h, z11.h\n"
      ".inst 0x818f0183  // bfmopa za3.s, p0/M, p0/M, z12.h, z15.h\n"
      "ld1h { z12.h }, p0/Z, [x25, #2, MUL VL]\n"
      ".inst 0xa142a6cb  // ldnt1h { z3.h, z7.h, z11.h, z15.h }, pn9.b/Z, [x22, #0x8, MUL VL]\n"
      ".inst 0x81900340  // bfmopa za0.s, p0/M, p0/M, z26.h, z16.h\n"
      ".inst 0x81940341  // bfmopa za1.s, p0/M, p0/M, z26.h, z20.h\n"
      ".inst 0x81980342  // bfmopa za2.s, p0/M, p0/M, z26.h, z24.h\n"
      ".inst 0x819c0343  // bfmopa za3.s, p0/M, p0/M, z26.h, z28.h\n"
      "ld1h { z26.h }, p0/Z, [x25, #3, MUL VL]\n"
      "addvl x25, x25, #4\n"
      ".inst 0xa143a6d8  // ldnt1h { z16.h, z20.h, z24.h, z28.h }, pn9.b/Z, [x22, #0xc, MUL VL]\n"
      "addvl x22, x22, #16\n"
      "bgt 6b\n"
      "7:"  // K loop tail
      ".inst 0x81930000  // bfmopa za0.s, p0/M, p0/M, z0.h, z19.h\n"
      ".inst 0x81970001  // bfmopa za1.s, p0/M, p0/M, z0.h, z23.h\n"
      ".inst 0x819b0002  // bfmopa za2.s, p0/M, p0/M, z0.h, z27.h\n"
      ".inst 0x819f0003  // bfmopa za3.s, p0/M, p0/M, z0.h, z31.h\n"
      ".inst 0x818201a0  // bfmopa za0.s, p0/M, p0/M, z13.h, z2.h\n"
      ".inst 0x818601a1  // bfmopa za1.s, p0/M, p0/M, z13.h, z6.h\n"
      ".inst 0x818a01a2  // bfmopa za2.s, p0/M, p0/M, z13.h, z10.h\n"
      ".inst 0x818e01a3  // bfmopa za3.s, p0/M, p0/M, z13.h, z14.h\n"
      ".inst 0x81830180  // bfmopa za0.s, p0/M, p0/M, z12.h, z3.h\n"
      ".inst 0x81870181  // bfmopa za1.s, p0/M, p0/M, z12.h, z7.h\n"
      ".inst 0x818b0182  // bfmopa za2.s, p0/M, p0/M, z12.h, z11.h\n"
      ".inst 0x818f0183  // bfmopa za3.s, p0/M, p0/M, z12.h, z15.h\n"
      ".inst 0x81900340  // bfmopa za0.s, p0/M, p0/M, z26.h, z16.h\n"
      ".inst 0x81940341  // bfmopa za1.s, p0/M, p0/M, z26.h, z20.h\n"
      ".inst 0x81980342  // bfmopa za2.s, p0/M, p0/M, z26.h, z24.h\n"
      ".inst 0x819c0343  // bfmopa za3.s, p0/M, p0/M, z26.h, z28.h\n"
      "8:"  // K oddments
      "cbz x20, 10f\n"
      "9:"  // K oddments: Loop
      "ld1h { z0.h }, p0/Z, [x25]\n"
      "subs x20, x20, #0x1\n"
      "addvl x25, x25, #1\n"
      ".inst 0xa140a6d3  // ld1h { z19.h, z23.h, z27.h, z31.h }, pn9.b/Z, [x22]\n"
      "addvl x22, x22, #4\n"
      ".inst 0x81930000  // bfmopa za0.s, p0/M, p0/M, z0.h, z19.h\n"
      ".inst 0x81970001  // bfmopa za1.s, p0/M, p0/M, z0.h, z23.h\n"
      ".inst 0x819b0002  // bfmopa za2.s, p0/M, p0/M, z0.h, z27.h\n"
      ".inst 0x819f0003  // bfmopa za3.s, p0/M, p0/M, z0.h, z31.h\n"
      "bgt 9b\n"
      "10:"  // K oddments: End
      "tbz x14, #1, 14f\n"
      "tbz x14, #0, 12f\n"
      "mov x12, #0x0\n"
      "cntw x19\n"
      "11:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xa040c5ac  // ld1w { z12.s-z15.s }, pn9.b/Z, [x13]\n"
      ".inst 0xc0860418  // mova { z24.s-z27.s }, za0h.s[x12]\n"
      ".inst 0xc0840580  // mova za0h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xc0860434  // mova { z20.s-z23.s }, za1h.s[x12]\n"
      ".inst 0xa041c5bc  // ld1w { z28.s-z31.s }, pn9.b/Z, [x13, #0x4, MUL VL]\n"
      ".inst 0xc0840781  // mova za1h.s[x12], { z28.s-z31.s }\n"
      ".inst 0xc086045c  // mova { z28.s-z31.s }, za2h.s[x12]\n"
      ".inst 0xc0860470  // mova { z16.s-z19.s }, za3h.s[x12]\n"
      ".inst 0xa042c5ac  // ld1w { z12.s-z15.s }, pn9.b/Z, [x13, #0x8, MUL VL]\n"
      ".inst 0xc0840582  // mova za2h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xa043c5ac  // ld1w { z12.s-z15.s }, pn9.b/Z, [x13, #0xc, MUL VL]\n"
      ".inst 0xc0840583  // mova za3h.s[x12], { z12.s-z15.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x19\n"
      ".inst 0xa060c578  // st1w { z24.s-z27.s }, pn9.b, [x11]\n"
      "addvl x13, x13, #16\n"
      ".inst 0xa061c574  // st1w { z20.s-z23.s }, pn9.b, [x11, #0x4, MUL VL]\n"
      ".inst 0xa062c57c  // st1w { z28.s-z31.s }, pn9.b, [x11, #0x8, MUL VL]\n"
      ".inst 0xa063c570  // st1w { z16.s-z19.s }, pn9.b, [x11, #0xc, MUL VL]\n"
      "addvl x11, x11, #16\n"
      "blt 11b\n"
      "b 24f\n"
      "12:"  // Store to partial result buffer: Store only
      "mov x12, #0x0\n"
      "cntw x19\n"
      "13:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xc086040c  // mova { z12.s-z15.s }, za0h.s[x12]\n"
      ".inst 0xc086043c  // mova { z28.s-z31.s }, za1h.s[x12]\n"
      ".inst 0xa060c56c  // st1w { z12.s-z15.s }, pn9.b, [x11]\n"
      ".inst 0xc0860450  // mova { z16.s-z19.s }, za2h.s[x12]\n"
      ".inst 0xc0860464  // mova { z4.s-z7.s }, za3h.s[x12]\n"
      ".inst 0xa061c57c  // st1w { z28.s-z31.s }, pn9.b, [x11, #0x4, MUL VL]\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x19\n"
      ".inst 0xa062c570  // st1w { z16.s-z19.s }, pn9.b, [x11, #0x8, MUL VL]\n"
      ".inst 0xa063c564  // st1w { z4.s-z7.s }, pn9.b, [x11, #0xc, MUL VL]\n"
      "addvl x11, x11, #16\n"
      "blt 13b\n"
      "b 24f\n"
      "14:"  // Store to output array
      "ldr x24, [%x[args], %[offsetof_C]]\n"
      "add x24, x24, x28, LSL #2\n"  // C += n
      "sub x23, x10, x9\n"
      "ldr x22, [%x[args], %[offsetof_ldcb]]\n"
      "madd x24, x9, x22, x24\n"  // C += m * ldc
      "tbz x14, #2, 18f\n"
      "cntw x19\n"
      "cmp x23, x19\n"
      "csel x21, x23, x19, LT\n"
      "lsr x20, x21, #0x2\n"
      "mov x12, #0x0\n"
      "and x19, x21, #0x3\n"
      "cbz x20, 16f\n"
      "15:"  // Store to output array: Skip activation: Accumulator row 0 loop
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      ".inst 0xc0860424  // mova { z4.s-z7.s }, za1h.s[x12]\n"
      ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xa160c300  // st1w { z0.s, z4.s, z8.s, z12.s }, p8, [x24]\n"
      "add x24, x24, x22\n"
      ".inst 0xa160c301  // st1w { z1.s, z5.s, z9.s, z13.s }, p8, [x24]\n"
      "add x24, x24, x22\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa160c302  // st1w { z2.s, z6.s, z10.s, z14.s }, p8, [x24]\n"
      "add x24, x24, x22\n"
      "cmp x12, x20, LSL #2\n"
      ".inst 0xa160c303  // st1w { z3.s, z7.s, z11.s, z15.s }, p8, [x24]\n"
      "add x24, x24, x22\n"
      "blt 15b\n"
      "16:"  // Store to output array: Skip activation: Accumulator row 0 oddments
      "cbz x19, 17f\n"
      "subs x19, x19, #0x1\n"
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      ".inst 0xc0860424  // mova { z4.s-z7.s }, za1h.s[x12]\n"
      ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xa160c300  // st1w { z0.s, z4.s, z8.s, z12.s }, p8, [x24]\n"
      "add x24, x24, x22\n"
      "beq 17f\n"
      "subs x19, x19, #0x1\n"
      ".inst 0xa160c301  // st1w { z1.s, z5.s, z9.s, z13.s }, p8, [x24]\n"
      "add x24, x24, x22\n"
      "beq 17f\n"
      ".inst 0xa160c302  // st1w { z2.s, z6.s, z10.s, z14.s }, p8, [x24]\n"
      "add x24, x24, x22\n"
      "17:"  // Store to output array: Skip activation: Accumulator row 0 oddments: End
      "subs x23, x23, x21\n"
      "beq 18f\n"
      "b 22f\n"
      "18:"  // Store to output array: Skip activation: End
      "cntw x19\n"
      "cmp x23, x19\n"
      "ld1rw { z23.s }, p0/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
      "csel x19, x23, x19, LT\n"
      "lsr x20, x19, #0x2\n"
      "ld1rw { z16.s }, p0/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
      "mov x12, #0x0\n"
      "and x19, x19, #0x3\n"
      "cbz x20, 20f\n"
      "19:"  // Store to output array: Accumulator row 0 loop
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      ".inst 0xc0860424  // mova { z4.s-z7.s }, za1h.s[x12]\n"
      ".inst 0xc1b0cae0  // fclamp { z0.s-z3.s }, z23.s, z16.s\n"
      ".inst 0xc1b0cae4  // fclamp { z4.s-z7.s }, z23.s, z16.s\n"
      ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xc1b0cae8  // fclamp { z8.s-z11.s }, z23.s, z16.s\n"
      ".inst 0xc1b0caec  // fclamp { z12.s-z15.s }, z23.s, z16.s\n"
      ".inst 0xa160c300  // st1w { z0.s, z4.s, z8.s, z12.s }, p8, [x24]\n"
      "add x24, x24, x22\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa160c301  // st1w { z1.s, z5.s, z9.s, z13.s }, p8, [x24]\n"
      "add x24, x24, x22\n"
      "cmp x12, x20, LSL #2\n"
      ".inst 0xa160c302  // st1w { z2.s, z6.s, z10.s, z14.s }, p8, [x24]\n"
      "add x24, x24, x22\n"
      ".inst 0xa160c303  // st1w { z3.s, z7.s, z11.s, z15.s }, p8, [x24]\n"
      "add x24, x24, x22\n"
      "blt 19b\n"
      "20:"  // Store to output array: Accumulator row 0 oddments
      "cbz x19, 21f\n"
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      ".inst 0xc0860424  // mova { z4.s-z7.s }, za1h.s[x12]\n"
      ".inst 0xc1b0cae0  // fclamp { z0.s-z3.s }, z23.s, z16.s\n"
      ".inst 0xc1b0cae4  // fclamp { z4.s-z7.s }, z23.s, z16.s\n"
      ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xc1b0cae8  // fclamp { z8.s-z11.s }, z23.s, z16.s\n"
      ".inst 0xc1b0caec  // fclamp { z12.s-z15.s }, z23.s, z16.s\n"
      "subs x19, x19, #0x1\n"
      ".inst 0xa160c300  // st1w { z0.s, z4.s, z8.s, z12.s }, p8, [x24]\n"
      "add x24, x24, x22\n"
      "beq 21f\n"
      "subs x19, x19, #0x1\n"
      ".inst 0xa160c301  // st1w { z1.s, z5.s, z9.s, z13.s }, p8, [x24]\n"
      "add x24, x24, x22\n"
      "beq 21f\n"
      ".inst 0xa160c302  // st1w { z2.s, z6.s, z10.s, z14.s }, p8, [x24]\n"
      "21:"  // Store to output array: Accumulator row 0 oddments: End
      "22:"  // Store to output array: End
      "tbz x14, #0, 24f\n"
      "mov x12, #0x0\n"
      "cntw x19\n"
      "23:"  // Store to output array: Refill accumulators: Loop
      ".inst 0xa040c5b0  // ld1w { z16.s-z19.s }, pn9.b/Z, [x13]\n"
      ".inst 0xc0840600  // mova za0h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xa041c5b0  // ld1w { z16.s-z19.s }, pn9.b/Z, [x13, #0x4, MUL VL]\n"
      ".inst 0xc0840601  // mova za1h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xa042c5b0  // ld1w { z16.s-z19.s }, pn9.b/Z, [x13, #0x8, MUL VL]\n"
      ".inst 0xc0840602  // mova za2h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xa043c5a8  // ld1w { z8.s-z11.s }, pn9.b/Z, [x13, #0xc, MUL VL]\n"
      ".inst 0xc0840503  // mova za3h.s[x12], { z8.s-z11.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x19\n"
      "addvl x13, x13, #16\n"
      "blt 23b\n"
      "24:"  // End block
      "incw x28, ALL, MUL #4\n"
      "cmp x28, x27\n"
      "blt 3b\n"
      "incw x9\n"
      "cmp x9, x10\n"
      "mov x28, #0x0\n"
      "mov x26, x25\n"
      "blt 3b\n"
      ".inst 0xd503467f  // SMSTOP\n"
      :
      : [args] "r" (&args), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_KernelArgs_max] "I" (offsetof(KernelArgs, max)), [offsetof_KernelArgs_min] "I" (offsetof(KernelArgs, min)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_kstride_bytes] "I" (offsetof(KernelArgs, kstride_bytes)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb))
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

}  // namespace arm_gemm

#endif  // ARM_COMPUTE_ENABLE_SME2
#endif  // __ARM_FEATURE_SVE
