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


#include "../../asmlib.hpp"
#include "../../utils.hpp"

namespace arm_gemm {

void sme2_interleaved_nomerge_fp32_mopa_1VLx4VL(const float *const A, const float *const B, float *const C, int ldc, const int M, const int N, const int K, const float *const bias, const Activation act, bool accumulate, float *const accumulator_buffer)
{
  struct KernelArgs
  {
    KernelArgs(
      const float *const A,
      const float *const B,
      float *const C, const int ldc,
      const int M, const int N, const int K,
      const float *const bias,
      const Activation act,
      bool accumulate,
      float *const accumulator_buffer
    ) : A(A),
        B(B), kstride_bytes(K * sizeof(float)),
        C(C), ldcb(ldc * sizeof(float)),
        M(M), N(N), K(K),
        n_loops((K - 1) / 2), n_tail_iters((K - 1) % 2),
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

    const float *const A;
    const float *const B;
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
      "ldr x15, [%x[args], %[offsetof_flags]]\n"
      ".inst 0xd503477f  // SMSTART ZA\n"
      "ptrue p0.b\n"
      ".inst 0x25207811  // ptrue pn9.b\n"
      "ldr x14, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "ldr x13, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "tbz x15, #0, 2f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "1:"  // Initial accumulator load from buffer: Loop
      ".inst 0xa040c5cc  // ld1w { z12.s-z15.s }, pn9.b/Z, [x14]\n"
      ".inst 0xc0840580  // mova za0h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xa041c5d0  // ld1w { z16.s-z19.s }, pn9.b/Z, [x14, #0x4, MUL VL]\n"
      ".inst 0xc0840601  // mova za1h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xa042c5c4  // ld1w { z4.s-z7.s }, pn9.b/Z, [x14, #0x8, MUL VL]\n"
      ".inst 0xc0840482  // mova za2h.s[x12], { z4.s-z7.s }\n"
      ".inst 0xa043c5c4  // ld1w { z4.s-z7.s }, pn9.b/Z, [x14, #0xc, MUL VL]\n"
      ".inst 0xc0840483  // mova za3h.s[x12], { z4.s-z7.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
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
      ".inst 0x25bc6530  // whilelt pn8.s, x9, x28, VLx4\n"
      "tbnz x15, #0, 4f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
      "cbz x20, 5f\n"
      "fmov z21.s, #1.0\n"
      ".inst 0xa009c29d  // ldnt1w { z28.s-z31.s }, p8/Z, [x20, x9, LSL #2]\n"
      ".inst 0x809c02a0  // fmopa za0.s, p0/M, p0/M, z21.s, z28.s\n"
      ".inst 0x809d02a1  // fmopa za1.s, p0/M, p0/M, z21.s, z29.s\n"
      ".inst 0x809e02a2  // fmopa za2.s, p0/M, p0/M, z21.s, z30.s\n"
      ".inst 0x809f02a3  // fmopa za3.s, p0/M, p0/M, z21.s, z31.s\n"
      "4:"  // Prepare accumulators: Test for last block
      "mov x20, x9\n"
      "mov x21, x10\n"
      "incw x20, ALL, MUL #4\n"
      "incw x21\n"
      "cmp x20, x28\n"
      "csel x21, x10, x21, LT\n"
      "mov x20, x15\n"
      "bfm x15, XZR, #0x0, #0x0  // bfc x15, #0x0, #0x1\n"
      "cmp x21, x11\n"
      "csel x15, x20, x15, LT\n"
      "5:"  // Prepare accumulators: End
      "ldr x20, [%x[args], %[offsetof_K]]\n"
      "lsr x23, x20, #0x2\n"
      "and x22, x20, #0x3\n"
      "ldr x21, [%x[args], %[offsetof_B]]\n"
      "ldr x20, [%x[args], %[offsetof_kstride_bytes]]\n"
      "madd x21, x9, x20, x21\n"  // bptr = B + n * kstride_bytes
      "cbz x23, 8f\n"
      "subs x23, x23, #0x1\n"
      "ld1w { z0.s }, p0/Z, [x26]\n"
      ".inst 0xa140c6bb  // ldnt1w { z19.s, z23.s, z27.s, z31.s }, pn9.b/Z, [x21]\n"
      "ld1w { z13.s }, p0/Z, [x26, #1, MUL VL]\n"
      ".inst 0xa141c6aa  // ldnt1w { z2.s, z6.s, z10.s, z14.s }, pn9.b/Z, [x21, #0x4, MUL VL]\n"
      "ld1w { z12.s }, p0/Z, [x26, #2, MUL VL]\n"
      ".inst 0xa142c6ab  // ldnt1w { z3.s, z7.s, z11.s, z15.s }, pn9.b/Z, [x21, #0x8, MUL VL]\n"
      "ld1w { z26.s }, p0/Z, [x26, #3, MUL VL]\n"
      "addvl x26, x26, #4\n"
      ".inst 0xa143c6b8  // ldnt1w { z16.s, z20.s, z24.s, z28.s }, pn9.b/Z, [x21, #0xc, MUL VL]\n"
      "addvl x21, x21, #16\n"
      "ble 7f\n"
      "6:"  // K loop
      ".inst 0x80930000  // fmopa za0.s, p0/M, p0/M, z0.s, z19.s\n"
      "subs x23, x23, #0x1\n"
      ".inst 0x80970001  // fmopa za1.s, p0/M, p0/M, z0.s, z23.s\n"
      ".inst 0x809b0002  // fmopa za2.s, p0/M, p0/M, z0.s, z27.s\n"
      ".inst 0x809f0003  // fmopa za3.s, p0/M, p0/M, z0.s, z31.s\n"
      "ld1w { z0.s }, p0/Z, [x26]\n"
      ".inst 0x808201a0  // fmopa za0.s, p0/M, p0/M, z13.s, z2.s\n"
      ".inst 0xa140c6bb  // ldnt1w { z19.s, z23.s, z27.s, z31.s }, pn9.b/Z, [x21]\n"
      ".inst 0x808601a1  // fmopa za1.s, p0/M, p0/M, z13.s, z6.s\n"
      ".inst 0x808a01a2  // fmopa za2.s, p0/M, p0/M, z13.s, z10.s\n"
      ".inst 0x808e01a3  // fmopa za3.s, p0/M, p0/M, z13.s, z14.s\n"
      "ld1w { z13.s }, p0/Z, [x26, #1, MUL VL]\n"
      ".inst 0x80830180  // fmopa za0.s, p0/M, p0/M, z12.s, z3.s\n"
      ".inst 0xa141c6aa  // ldnt1w { z2.s, z6.s, z10.s, z14.s }, pn9.b/Z, [x21, #0x4, MUL VL]\n"
      ".inst 0x80870181  // fmopa za1.s, p0/M, p0/M, z12.s, z7.s\n"
      ".inst 0x808b0182  // fmopa za2.s, p0/M, p0/M, z12.s, z11.s\n"
      ".inst 0x808f0183  // fmopa za3.s, p0/M, p0/M, z12.s, z15.s\n"
      "ld1w { z12.s }, p0/Z, [x26, #2, MUL VL]\n"
      ".inst 0xa142c6ab  // ldnt1w { z3.s, z7.s, z11.s, z15.s }, pn9.b/Z, [x21, #0x8, MUL VL]\n"
      ".inst 0x80900340  // fmopa za0.s, p0/M, p0/M, z26.s, z16.s\n"
      ".inst 0x80940341  // fmopa za1.s, p0/M, p0/M, z26.s, z20.s\n"
      ".inst 0x80980342  // fmopa za2.s, p0/M, p0/M, z26.s, z24.s\n"
      ".inst 0x809c0343  // fmopa za3.s, p0/M, p0/M, z26.s, z28.s\n"
      "ld1w { z26.s }, p0/Z, [x26, #3, MUL VL]\n"
      "addvl x26, x26, #4\n"
      ".inst 0xa143c6b8  // ldnt1w { z16.s, z20.s, z24.s, z28.s }, pn9.b/Z, [x21, #0xc, MUL VL]\n"
      "addvl x21, x21, #16\n"
      "bgt 6b\n"
      "7:"  // K loop tail
      ".inst 0x80930000  // fmopa za0.s, p0/M, p0/M, z0.s, z19.s\n"
      ".inst 0x80970001  // fmopa za1.s, p0/M, p0/M, z0.s, z23.s\n"
      ".inst 0x809b0002  // fmopa za2.s, p0/M, p0/M, z0.s, z27.s\n"
      ".inst 0x809f0003  // fmopa za3.s, p0/M, p0/M, z0.s, z31.s\n"
      ".inst 0x808201a0  // fmopa za0.s, p0/M, p0/M, z13.s, z2.s\n"
      ".inst 0x808601a1  // fmopa za1.s, p0/M, p0/M, z13.s, z6.s\n"
      ".inst 0x808a01a2  // fmopa za2.s, p0/M, p0/M, z13.s, z10.s\n"
      ".inst 0x808e01a3  // fmopa za3.s, p0/M, p0/M, z13.s, z14.s\n"
      ".inst 0x80830180  // fmopa za0.s, p0/M, p0/M, z12.s, z3.s\n"
      ".inst 0x80870181  // fmopa za1.s, p0/M, p0/M, z12.s, z7.s\n"
      ".inst 0x808b0182  // fmopa za2.s, p0/M, p0/M, z12.s, z11.s\n"
      ".inst 0x808f0183  // fmopa za3.s, p0/M, p0/M, z12.s, z15.s\n"
      ".inst 0x80900340  // fmopa za0.s, p0/M, p0/M, z26.s, z16.s\n"
      ".inst 0x80940341  // fmopa za1.s, p0/M, p0/M, z26.s, z20.s\n"
      ".inst 0x80980342  // fmopa za2.s, p0/M, p0/M, z26.s, z24.s\n"
      ".inst 0x809c0343  // fmopa za3.s, p0/M, p0/M, z26.s, z28.s\n"
      "8:"  // K oddments
      "cbz x22, 10f\n"
      "9:"  // K oddments: Loop
      "ld1w { z0.s }, p0/Z, [x26]\n"
      "subs x22, x22, #0x1\n"
      "addvl x26, x26, #1\n"
      ".inst 0xa140c6b3  // ld1w { z19.s, z23.s, z27.s, z31.s }, pn9.b/Z, [x21]\n"
      "addvl x21, x21, #4\n"
      ".inst 0x80930000  // fmopa za0.s, p0/M, p0/M, z0.s, z19.s\n"
      ".inst 0x80970001  // fmopa za1.s, p0/M, p0/M, z0.s, z23.s\n"
      ".inst 0x809b0002  // fmopa za2.s, p0/M, p0/M, z0.s, z27.s\n"
      ".inst 0x809f0003  // fmopa za3.s, p0/M, p0/M, z0.s, z31.s\n"
      "bgt 9b\n"
      "10:"  // K oddments: End
      "tbz x15, #1, 14f\n"
      "tbz x15, #0, 12f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "11:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xa040c5cc  // ld1w { z12.s-z15.s }, pn9.b/Z, [x14]\n"
      ".inst 0xc0860418  // mova { z24.s-z27.s }, za0h.s[x12]\n"
      ".inst 0xc0840580  // mova za0h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xc0860434  // mova { z20.s-z23.s }, za1h.s[x12]\n"
      ".inst 0xa041c5dc  // ld1w { z28.s-z31.s }, pn9.b/Z, [x14, #0x4, MUL VL]\n"
      ".inst 0xc0840781  // mova za1h.s[x12], { z28.s-z31.s }\n"
      ".inst 0xc086045c  // mova { z28.s-z31.s }, za2h.s[x12]\n"
      ".inst 0xc0860470  // mova { z16.s-z19.s }, za3h.s[x12]\n"
      ".inst 0xa042c5cc  // ld1w { z12.s-z15.s }, pn9.b/Z, [x14, #0x8, MUL VL]\n"
      ".inst 0xc0840582  // mova za2h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xa043c5cc  // ld1w { z12.s-z15.s }, pn9.b/Z, [x14, #0xc, MUL VL]\n"
      ".inst 0xc0840583  // mova za3h.s[x12], { z12.s-z15.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      ".inst 0xa060c5b8  // st1w { z24.s-z27.s }, pn9.b, [x13]\n"
      "addvl x14, x14, #16\n"
      ".inst 0xa061c5b4  // st1w { z20.s-z23.s }, pn9.b, [x13, #0x4, MUL VL]\n"
      ".inst 0xa062c5bc  // st1w { z28.s-z31.s }, pn9.b, [x13, #0x8, MUL VL]\n"
      ".inst 0xa063c5b0  // st1w { z16.s-z19.s }, pn9.b, [x13, #0xc, MUL VL]\n"
      "addvl x13, x13, #16\n"
      "blt 11b\n"
      "b 24f\n"
      "12:"  // Store to partial result buffer: Store only
      "mov x12, #0x0\n"
      "cntw x20\n"
      "13:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xc086040c  // mova { z12.s-z15.s }, za0h.s[x12]\n"
      ".inst 0xc086043c  // mova { z28.s-z31.s }, za1h.s[x12]\n"
      ".inst 0xa060c5ac  // st1w { z12.s-z15.s }, pn9.b, [x13]\n"
      ".inst 0xc0860450  // mova { z16.s-z19.s }, za2h.s[x12]\n"
      ".inst 0xc0860464  // mova { z4.s-z7.s }, za3h.s[x12]\n"
      ".inst 0xa061c5bc  // st1w { z28.s-z31.s }, pn9.b, [x13, #0x4, MUL VL]\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      ".inst 0xa062c5b0  // st1w { z16.s-z19.s }, pn9.b, [x13, #0x8, MUL VL]\n"
      ".inst 0xa063c5a4  // st1w { z4.s-z7.s }, pn9.b, [x13, #0xc, MUL VL]\n"
      "addvl x13, x13, #16\n"
      "blt 13b\n"
      "b 24f\n"
      "14:"  // Store to output array
      "ldr x25, [%x[args], %[offsetof_C]]\n"
      "add x25, x25, x9, LSL #2\n"  // C += n
      "sub x24, x11, x10\n"
      "ldr x23, [%x[args], %[offsetof_ldcb]]\n"
      "madd x25, x10, x23, x25\n"  // C += m * ldc
      "tbz x15, #2, 18f\n"
      "cntw x20\n"
      "cmp x24, x20\n"
      "csel x22, x24, x20, LT\n"
      "lsr x21, x22, #0x2\n"
      "mov x12, #0x0\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 16f\n"
      "15:"  // Store to output array: Skip activation: Accumulator row 0 loop
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      ".inst 0xc0860424  // mova { z4.s-z7.s }, za1h.s[x12]\n"
      ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xa160c320  // st1w { z0.s, z4.s, z8.s, z12.s }, p8, [x25]\n"
      "add x25, x25, x23\n"
      ".inst 0xa160c321  // st1w { z1.s, z5.s, z9.s, z13.s }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa160c322  // st1w { z2.s, z6.s, z10.s, z14.s }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "cmp x12, x21, LSL #2\n"
      ".inst 0xa160c323  // st1w { z3.s, z7.s, z11.s, z15.s }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "blt 15b\n"
      "16:"  // Store to output array: Skip activation: Accumulator row 0 oddments
      "cbz x20, 17f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      ".inst 0xc0860424  // mova { z4.s-z7.s }, za1h.s[x12]\n"
      ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xa160c320  // st1w { z0.s, z4.s, z8.s, z12.s }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "beq 17f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xa160c321  // st1w { z1.s, z5.s, z9.s, z13.s }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "beq 17f\n"
      ".inst 0xa160c322  // st1w { z2.s, z6.s, z10.s, z14.s }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "17:"  // Store to output array: Skip activation: Accumulator row 0 oddments: End
      "subs x24, x24, x22\n"
      "beq 18f\n"
      "b 22f\n"
      "18:"  // Store to output array: Skip activation: End
      "cntw x20\n"
      "cmp x24, x20\n"
      "ld1rw { z23.s }, p0/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
      "csel x20, x24, x20, LT\n"
      "lsr x21, x20, #0x2\n"
      "ld1rw { z16.s }, p0/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
      "mov x12, #0x0\n"
      "and x20, x20, #0x3\n"
      "cbz x21, 20f\n"
      "19:"  // Store to output array: Accumulator row 0 loop
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      ".inst 0xc0860424  // mova { z4.s-z7.s }, za1h.s[x12]\n"
      ".inst 0xc1b0cae0  // fclamp { z0.s-z3.s }, z23.s, z16.s\n"
      ".inst 0xc1b0cae4  // fclamp { z4.s-z7.s }, z23.s, z16.s\n"
      ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xc1b0cae8  // fclamp { z8.s-z11.s }, z23.s, z16.s\n"
      ".inst 0xc1b0caec  // fclamp { z12.s-z15.s }, z23.s, z16.s\n"
      ".inst 0xa160c320  // st1w { z0.s, z4.s, z8.s, z12.s }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa160c321  // st1w { z1.s, z5.s, z9.s, z13.s }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "cmp x12, x21, LSL #2\n"
      ".inst 0xa160c322  // st1w { z2.s, z6.s, z10.s, z14.s }, p8, [x25]\n"
      "add x25, x25, x23\n"
      ".inst 0xa160c323  // st1w { z3.s, z7.s, z11.s, z15.s }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "blt 19b\n"
      "20:"  // Store to output array: Accumulator row 0 oddments
      "cbz x20, 21f\n"
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      ".inst 0xc0860424  // mova { z4.s-z7.s }, za1h.s[x12]\n"
      ".inst 0xc1b0cae0  // fclamp { z0.s-z3.s }, z23.s, z16.s\n"
      ".inst 0xc1b0cae4  // fclamp { z4.s-z7.s }, z23.s, z16.s\n"
      ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xc1b0cae8  // fclamp { z8.s-z11.s }, z23.s, z16.s\n"
      ".inst 0xc1b0caec  // fclamp { z12.s-z15.s }, z23.s, z16.s\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xa160c320  // st1w { z0.s, z4.s, z8.s, z12.s }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "beq 21f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xa160c321  // st1w { z1.s, z5.s, z9.s, z13.s }, p8, [x25]\n"
      "add x25, x25, x23\n"
      "beq 21f\n"
      ".inst 0xa160c322  // st1w { z2.s, z6.s, z10.s, z14.s }, p8, [x25]\n"
      "21:"  // Store to output array: Accumulator row 0 oddments: End
      "22:"  // Store to output array: End
      "tbz x15, #0, 24f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "23:"  // Store to output array: Refill accumulators: Loop
      ".inst 0xa040c5d0  // ld1w { z16.s-z19.s }, pn9.b/Z, [x14]\n"
      ".inst 0xc0840600  // mova za0h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xa041c5d0  // ld1w { z16.s-z19.s }, pn9.b/Z, [x14, #0x4, MUL VL]\n"
      ".inst 0xc0840601  // mova za1h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xa042c5d0  // ld1w { z16.s-z19.s }, pn9.b/Z, [x14, #0x8, MUL VL]\n"
      ".inst 0xc0840602  // mova za2h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xa043c5c8  // ld1w { z8.s-z11.s }, pn9.b/Z, [x14, #0xc, MUL VL]\n"
      ".inst 0xc0840503  // mova za3h.s[x12], { z8.s-z11.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "addvl x14, x14, #16\n"
      "blt 23b\n"
      "24:"  // End block
      "incw x9, ALL, MUL #4\n"
      "cmp x9, x28\n"
      "blt 3b\n"
      "incw x10\n"
      "cmp x10, x11\n"
      "mov x9, #0x0\n"
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
#endif  // __ARM_FEATURE_SVE
