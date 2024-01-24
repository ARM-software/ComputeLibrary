/*
 * Copyright (c) 2024 Arm Limited.
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

void sme2_interleaved_nomerge_s8qfp32_mopa_4VLx1VL(const int8_t *const A, const int8_t *const B, float *const C, int ldc, const int M, const int N, const int K, const int32_t *const bias, const DequantizeFloat &dq, const float *const late_bias, const Activation act, bool accumulate, int32_t *const accumulator_buffer)
{
  struct KernelArgs
  {
    KernelArgs(
      const int8_t *const A,
      const int8_t *const B,
      float *const C, const int ldc,
      const int M, const int N, const int K,
      const int32_t *const bias, const float *const late_bias, const Activation act,
      bool accumulate,
      int32_t *const accumulator_buffer
    ) : A(A),
        B(B), kstride_bytes(roundup(K, 4) * sizeof(int8_t)),
        C(C), ldcb(ldc * sizeof(float)),
        M(M), N(N), K(K),
        min(-std::numeric_limits<float>::infinity()),
        max(std::numeric_limits<float>::infinity()),
        bias(bias), late_bias(late_bias),
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
            this->max = static_cast<float>(act.param1);
            /* fall through */
        case Activation::Type::ReLU:
            this->min = static_cast<float>(0);
            break;
      }
    }

    const int8_t *const A;
    const int8_t *const B;
    const long kstride_bytes;
    float *const C;
    const long ldcb;
    const long M, N, K;
    float min = -std::numeric_limits<float>::infinity();
    float max = std::numeric_limits<float>::infinity();

    const int32_t *const bias;
    const float *const late_bias;

    int32_t *const accumulator_buffer;
    uint64_t flags;
  };

  // Construct arguments for this kernel
  KernelArgs args(A, B, C, ldc, M, N, K, bias, late_bias, act, accumulate, accumulator_buffer);

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
      ".inst 0xa040c1f4  // ld1w { z20.s-z23.s }, pn8.b/Z, [x15]\n"
      ".inst 0xa041c1fc  // ld1w { z28.s-z31.s }, pn8.b/Z, [x15, #0x4, MUL VL]\n"
      ".inst 0xa042c1e8  // ld1w { z8.s-z11.s }, pn8.b/Z, [x15, #0x8, MUL VL]\n"
      ".inst 0xa043c1f0  // ld1w { z16.s-z19.s }, pn8.b/Z, [x15, #0xc, MUL VL]\n"
      ".inst 0xc0840680  // mova za0h.s[x12], { z20.s-z23.s }\n"
      "addvl x15, x15, #16\n"
      ".inst 0xc0840781  // mova za1h.s[x12], { z28.s-z31.s }\n"
      ".inst 0xc0840502  // mova za2h.s[x12], { z8.s-z11.s }\n"
      ".inst 0xc0840603  // mova za3h.s[x12], { z16.s-z19.s }\n"
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
      "ld1w { z23.s }, p0/Z, [x20, x10, LSL #2]\n"
      ".inst 0xc09026e0  // addha za0.s, p1/M, p1/M, z23.s\n"
      ".inst 0xc09026e1  // addha za1.s, p1/M, p1/M, z23.s\n"
      ".inst 0xc09026e2  // addha za2.s, p1/M, p1/M, z23.s\n"
      ".inst 0xc09026e3  // addha za3.s, p1/M, p1/M, z23.s\n"
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
      ".inst 0xa0408378  // ld1b { z24.b-z27.b }, pn8.b/Z, [x27]\n"
      "ld1b { z4.b }, p1/Z, [x23]\n"
      ".inst 0xa0418374  // ld1b { z20.b-z23.b }, pn8.b/Z, [x27, #0x4, MUL VL]\n"
      "ld1b { z2.b }, p1/Z, [x23, #1, MUL VL]\n"
      ".inst 0xa042836c  // ld1b { z12.b-z15.b }, pn8.b/Z, [x27, #0x8, MUL VL]\n"
      "ld1b { z11.b }, p1/Z, [x23, #2, MUL VL]\n"
      ".inst 0xa0438370  // ld1b { z16.b-z19.b }, pn8.b/Z, [x27, #0xc, MUL VL]\n"
      "addvl x27, x27, #16\n"
      "ld1b { z28.b }, p1/Z, [x23, #3, MUL VL]\n"
      "addvl x23, x23, #4\n"
      "ble 7f\n"
      "6:"  // K loop
      ".inst 0xa0842700  // smopa za0.s, p1/M, p1/M, z24.b, z4.b\n"
      "subs x21, x21, #0x1\n"
      ".inst 0xa0842721  // smopa za1.s, p1/M, p1/M, z25.b, z4.b\n"
      ".inst 0xa0842742  // smopa za2.s, p1/M, p1/M, z26.b, z4.b\n"
      ".inst 0xa0842763  // smopa za3.s, p1/M, p1/M, z27.b, z4.b\n"
      ".inst 0xa0408378  // ld1b { z24.b-z27.b }, pn8.b/Z, [x27]\n"
      ".inst 0xa0822680  // smopa za0.s, p1/M, p1/M, z20.b, z2.b\n"
      "ld1b { z4.b }, p1/Z, [x23]\n"
      ".inst 0xa08226a1  // smopa za1.s, p1/M, p1/M, z21.b, z2.b\n"
      ".inst 0xa08226c2  // smopa za2.s, p1/M, p1/M, z22.b, z2.b\n"
      ".inst 0xa08226e3  // smopa za3.s, p1/M, p1/M, z23.b, z2.b\n"
      ".inst 0xa0418374  // ld1b { z20.b-z23.b }, pn8.b/Z, [x27, #0x4, MUL VL]\n"
      ".inst 0xa08b2580  // smopa za0.s, p1/M, p1/M, z12.b, z11.b\n"
      "ld1b { z2.b }, p1/Z, [x23, #1, MUL VL]\n"
      ".inst 0xa08b25a1  // smopa za1.s, p1/M, p1/M, z13.b, z11.b\n"
      ".inst 0xa08b25c2  // smopa za2.s, p1/M, p1/M, z14.b, z11.b\n"
      ".inst 0xa08b25e3  // smopa za3.s, p1/M, p1/M, z15.b, z11.b\n"
      ".inst 0xa042836c  // ld1b { z12.b-z15.b }, pn8.b/Z, [x27, #0x8, MUL VL]\n"
      "ld1b { z11.b }, p1/Z, [x23, #2, MUL VL]\n"
      ".inst 0xa09c2600  // smopa za0.s, p1/M, p1/M, z16.b, z28.b\n"
      ".inst 0xa09c2621  // smopa za1.s, p1/M, p1/M, z17.b, z28.b\n"
      ".inst 0xa09c2642  // smopa za2.s, p1/M, p1/M, z18.b, z28.b\n"
      ".inst 0xa09c2663  // smopa za3.s, p1/M, p1/M, z19.b, z28.b\n"
      ".inst 0xa0438370  // ld1b { z16.b-z19.b }, pn8.b/Z, [x27, #0xc, MUL VL]\n"
      "addvl x27, x27, #16\n"
      "ld1b { z28.b }, p1/Z, [x23, #3, MUL VL]\n"
      "addvl x23, x23, #4\n"
      "bgt 6b\n"
      "7:"  // K loop tail
      ".inst 0xa0842700  // smopa za0.s, p1/M, p1/M, z24.b, z4.b\n"
      ".inst 0xa0842721  // smopa za1.s, p1/M, p1/M, z25.b, z4.b\n"
      ".inst 0xa0842742  // smopa za2.s, p1/M, p1/M, z26.b, z4.b\n"
      ".inst 0xa0842763  // smopa za3.s, p1/M, p1/M, z27.b, z4.b\n"
      ".inst 0xa0822680  // smopa za0.s, p1/M, p1/M, z20.b, z2.b\n"
      ".inst 0xa08226a1  // smopa za1.s, p1/M, p1/M, z21.b, z2.b\n"
      ".inst 0xa08226c2  // smopa za2.s, p1/M, p1/M, z22.b, z2.b\n"
      ".inst 0xa08226e3  // smopa za3.s, p1/M, p1/M, z23.b, z2.b\n"
      ".inst 0xa08b2580  // smopa za0.s, p1/M, p1/M, z12.b, z11.b\n"
      ".inst 0xa08b25a1  // smopa za1.s, p1/M, p1/M, z13.b, z11.b\n"
      ".inst 0xa08b25c2  // smopa za2.s, p1/M, p1/M, z14.b, z11.b\n"
      ".inst 0xa08b25e3  // smopa za3.s, p1/M, p1/M, z15.b, z11.b\n"
      ".inst 0xa09c2600  // smopa za0.s, p1/M, p1/M, z16.b, z28.b\n"
      ".inst 0xa09c2621  // smopa za1.s, p1/M, p1/M, z17.b, z28.b\n"
      ".inst 0xa09c2642  // smopa za2.s, p1/M, p1/M, z18.b, z28.b\n"
      ".inst 0xa09c2663  // smopa za3.s, p1/M, p1/M, z19.b, z28.b\n"
      "8:"  // K oddments
      "cbz x20, 10f\n"
      "9:"  // K oddments: Loop
      ".inst 0xa1408373  // ld1b { z19.b, z23.b, z27.b, z31.b }, pn8.b/Z, [x27]\n"
      "subs x20, x20, #0x1\n"
      "addvl x27, x27, #4\n"
      "ld1b { z16.b }, p1/Z, [x23]\n"
      "addvl x23, x23, #1\n"
      ".inst 0xa0902660  // smopa za0.s, p1/M, p1/M, z19.b, z16.b\n"
      ".inst 0xa09026e1  // smopa za1.s, p1/M, p1/M, z23.b, z16.b\n"
      ".inst 0xa0902762  // smopa za2.s, p1/M, p1/M, z27.b, z16.b\n"
      ".inst 0xa09027e3  // smopa za3.s, p1/M, p1/M, z31.b, z16.b\n"
      "bgt 9b\n"
      "10:"  // K oddments: End
      "tbz x16, #1, 14f\n"
      "tbz x16, #0, 12f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "11:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xa040c1e8  // ld1w { z8.s-z11.s }, pn8.b/Z, [x15]\n"
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      ".inst 0xc0860424  // mova { z4.s-z7.s }, za1h.s[x12]\n"
      ".inst 0xa041c1ec  // ld1w { z12.s-z15.s }, pn8.b/Z, [x15, #0x4, MUL VL]\n"
      ".inst 0xc0860458  // mova { z24.s-z27.s }, za2h.s[x12]\n"
      ".inst 0xc0860470  // mova { z16.s-z19.s }, za3h.s[x12]\n"
      ".inst 0xa042c1fc  // ld1w { z28.s-z31.s }, pn8.b/Z, [x15, #0x8, MUL VL]\n"
      ".inst 0xa043c1f4  // ld1w { z20.s-z23.s }, pn8.b/Z, [x15, #0xc, MUL VL]\n"
      ".inst 0xc0840500  // mova za0h.s[x12], { z8.s-z11.s }\n"
      "addvl x15, x15, #16\n"
      ".inst 0xc0840581  // mova za1h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xa060c1c0  // st1w { z0.s-z3.s }, pn8.b, [x14]\n"
      ".inst 0xc0840782  // mova za2h.s[x12], { z28.s-z31.s }\n"
      ".inst 0xa061c1c4  // st1w { z4.s-z7.s }, pn8.b, [x14, #0x4, MUL VL]\n"
      ".inst 0xc0840683  // mova za3h.s[x12], { z20.s-z23.s }\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa062c1d8  // st1w { z24.s-z27.s }, pn8.b, [x14, #0x8, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa063c1d0  // st1w { z16.s-z19.s }, pn8.b, [x14, #0xc, MUL VL]\n"
      "addvl x14, x14, #16\n"
      "blt 11b\n"
      "b 30f\n"
      "12:"  // Store to partial result buffer: Store only
      "mov x12, #0x0\n"
      "cntw x20\n"
      "13:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xc0860408  // mova { z8.s-z11.s }, za0h.s[x12]\n"
      ".inst 0xc086042c  // mova { z12.s-z15.s }, za1h.s[x12]\n"
      ".inst 0xc0860454  // mova { z20.s-z23.s }, za2h.s[x12]\n"
      ".inst 0xc0860470  // mova { z16.s-z19.s }, za3h.s[x12]\n"
      ".inst 0xa060c1c8  // st1w { z8.s-z11.s }, pn8.b, [x14]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa061c1cc  // st1w { z12.s-z15.s }, pn8.b, [x14, #0x4, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa062c1d4  // st1w { z20.s-z23.s }, pn8.b, [x14, #0x8, MUL VL]\n"
      ".inst 0xa063c1d0  // st1w { z16.s-z19.s }, pn8.b, [x14, #0xc, MUL VL]\n"
      "addvl x14, x14, #16\n"
      "blt 13b\n"
      "b 30f\n"
      "14:"  // Store to output array
      "ldr x26, [%x[args], %[offsetof_C]]\n"
      "sub x25, x13, x11\n"
      "ld1rw { z23.s }, p1/Z, [%x[dq], %[offset_DequantizeFloat_scale]]\n"
      "fmov z22.s, #0x0\n"
      "ldr x24, [%x[args], %[offsetof_ldcb]]\n"
      "ldr x20, [%x[args], %[offsetof_late_bias]]\n"
      "add x26, x26, x10, LSL #2\n"  // C += n
      "madd x26, x11, x24, x26\n"  // C += m * ldc
      "cbz x20, 15f\n"
      "add x20, x20, x10, LSL #2\n"
      "ld1w { z22.s }, p0/Z, [x20]\n"
      "15:"  // Store to output array: no late bias
      "cntw x23\n"
      "ld1rw { z21.s }, p1/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
      "mov x12, #0x0\n"
      "cmp x25, x23\n"
      "ld1rw { z20.s }, p1/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 17f\n"
      "16:"  // Store to output array: Accumulator row 0 loop
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xc132e000  // scvtf { z0.s-z3.s }, { z0.s-z3.s }\n"
      "cmp x12, x21, LSL #2\n"
      "fmad z0.s, p1/M, z23.s, z22.s\n"
      "fmad z1.s, p1/M, z23.s, z22.s\n"
      "fmad z2.s, p1/M, z23.s, z22.s\n"
      "fmad z3.s, p1/M, z23.s, z22.s\n"
      ".inst 0xc1b4caa0  // fclamp { z0.s-z3.s }, z21.s, z20.s\n"
      "st1w { z0.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z1.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z2.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z3.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 16b\n"
      "17:"  // Store to output array: Accumulator row 0 oddments
      "cbz x20, 18f\n"
      ".inst 0xc0860410  // mova { z16.s-z19.s }, za0h.s[x12]\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc132e210  // scvtf { z16.s-z19.s }, { z16.s-z19.s }\n"
      "fmad z16.s, p1/M, z23.s, z22.s\n"
      "fmad z17.s, p1/M, z23.s, z22.s\n"
      "fmad z18.s, p1/M, z23.s, z22.s\n"
      "fmad z19.s, p1/M, z23.s, z22.s\n"
      ".inst 0xc1b4cab0  // fclamp { z16.s-z19.s }, z21.s, z20.s\n"
      "st1w { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 18f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 18f\n"
      "st1w { z18.s }, p0, [x26]\n"
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
      ".inst 0xc132e210  // scvtf { z16.s-z19.s }, { z16.s-z19.s }\n"
      "cmp x12, x21, LSL #2\n"
      "fmad z16.s, p1/M, z23.s, z22.s\n"
      "fmad z17.s, p1/M, z23.s, z22.s\n"
      "fmad z18.s, p1/M, z23.s, z22.s\n"
      "fmad z19.s, p1/M, z23.s, z22.s\n"
      ".inst 0xc1b4cab0  // fclamp { z16.s-z19.s }, z21.s, z20.s\n"
      "st1w { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z18.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z19.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 19b\n"
      "20:"  // Store to output array: Accumulator row 1 oddments
      "cbz x20, 21f\n"
      ".inst 0xc086043c  // mova { z28.s-z31.s }, za1h.s[x12]\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc132e39c  // scvtf { z28.s-z31.s }, { z28.s-z31.s }\n"
      "fmad z28.s, p1/M, z23.s, z22.s\n"
      "fmad z29.s, p1/M, z23.s, z22.s\n"
      "fmad z30.s, p1/M, z23.s, z22.s\n"
      "fmad z31.s, p1/M, z23.s, z22.s\n"
      ".inst 0xc1b4cabc  // fclamp { z28.s-z31.s }, z21.s, z20.s\n"
      "st1w { z28.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 21f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z29.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 21f\n"
      "st1w { z30.s }, p0, [x26]\n"
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
      ".inst 0xc132e18c  // scvtf { z12.s-z15.s }, { z12.s-z15.s }\n"
      "cmp x12, x21, LSL #2\n"
      "fmad z12.s, p1/M, z23.s, z22.s\n"
      "fmad z13.s, p1/M, z23.s, z22.s\n"
      "fmad z14.s, p1/M, z23.s, z22.s\n"
      "fmad z15.s, p1/M, z23.s, z22.s\n"
      ".inst 0xc1b4caac  // fclamp { z12.s-z15.s }, z21.s, z20.s\n"
      "st1w { z12.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z13.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z14.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z15.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 22b\n"
      "23:"  // Store to output array: Accumulator row 2 oddments
      "cbz x20, 24f\n"
      ".inst 0xc0860450  // mova { z16.s-z19.s }, za2h.s[x12]\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc132e210  // scvtf { z16.s-z19.s }, { z16.s-z19.s }\n"
      "fmad z16.s, p1/M, z23.s, z22.s\n"
      "fmad z17.s, p1/M, z23.s, z22.s\n"
      "fmad z18.s, p1/M, z23.s, z22.s\n"
      "fmad z19.s, p1/M, z23.s, z22.s\n"
      ".inst 0xc1b4cab0  // fclamp { z16.s-z19.s }, z21.s, z20.s\n"
      "st1w { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 24f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 24f\n"
      "st1w { z18.s }, p0, [x26]\n"
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
      ".inst 0xc0860478  // mova { z24.s-z27.s }, za3h.s[x12]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xc132e318  // scvtf { z24.s-z27.s }, { z24.s-z27.s }\n"
      "cmp x12, x21, LSL #2\n"
      "fmad z24.s, p1/M, z23.s, z22.s\n"
      "fmad z25.s, p1/M, z23.s, z22.s\n"
      "fmad z26.s, p1/M, z23.s, z22.s\n"
      "fmad z27.s, p1/M, z23.s, z22.s\n"
      ".inst 0xc1b4cab8  // fclamp { z24.s-z27.s }, z21.s, z20.s\n"
      "st1w { z24.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z25.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z26.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z27.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 25b\n"
      "26:"  // Store to output array: Accumulator row 3 oddments
      "cbz x20, 27f\n"
      ".inst 0xc0860470  // mova { z16.s-z19.s }, za3h.s[x12]\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc132e210  // scvtf { z16.s-z19.s }, { z16.s-z19.s }\n"
      "fmad z16.s, p1/M, z23.s, z22.s\n"
      "fmad z17.s, p1/M, z23.s, z22.s\n"
      "fmad z18.s, p1/M, z23.s, z22.s\n"
      "fmad z19.s, p1/M, z23.s, z22.s\n"
      ".inst 0xc1b4cab0  // fclamp { z16.s-z19.s }, z21.s, z20.s\n"
      "st1w { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 27f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 27f\n"
      "st1w { z18.s }, p0, [x26]\n"
      "27:"  // Store to output array: Accumulator row 3 oddments: End
      "28:"  // Store to output array: End
      "tbz x16, #0, 30f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "29:"  // Store to output array: Refill accumulators: Loop
      ".inst 0xa040c1fc  // ld1w { z28.s-z31.s }, pn8.b/Z, [x15]\n"
      ".inst 0xa041c1e0  // ld1w { z0.s-z3.s }, pn8.b/Z, [x15, #0x4, MUL VL]\n"
      ".inst 0xa042c1ec  // ld1w { z12.s-z15.s }, pn8.b/Z, [x15, #0x8, MUL VL]\n"
      ".inst 0xa043c1e4  // ld1w { z4.s-z7.s }, pn8.b/Z, [x15, #0xc, MUL VL]\n"
      ".inst 0xc0840780  // mova za0h.s[x12], { z28.s-z31.s }\n"
      "addvl x15, x15, #16\n"
      ".inst 0xc0840401  // mova za1h.s[x12], { z0.s-z3.s }\n"
      ".inst 0xc0840582  // mova za2h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xc0840483  // mova za3h.s[x12], { z4.s-z7.s }\n"
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
      : [args] "r" (&args), [dq] "r" (&dq), [offset_DequantizeFloat_scale] "I" (offsetof(DequantizeFloat, scale)), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_KernelArgs_max] "I" (offsetof(KernelArgs, max)), [offsetof_KernelArgs_min] "I" (offsetof(KernelArgs, min)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_kstride_bytes] "I" (offsetof(KernelArgs, kstride_bytes)), [offsetof_late_bias] "I" (offsetof(KernelArgs, late_bias)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb))
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

}  // namespace arm_gemm

#endif  // ARM_COMPUTE_ENABLE_SME2
