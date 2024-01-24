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

void sme2_interleaved_nomerge_s8qfp32_mopa_1VLx4VL(const int8_t *const A, const int8_t *const B, float *const C, int ldc, const int M, const int N, const int K, const int32_t *const bias, const DequantizeFloat &dq, const float *const late_bias, const Activation act, bool accumulate, int32_t *const accumulator_buffer)
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
      ".inst 0xa040c57c  // ld1w { z28.s-z31.s }, pn9.b/Z, [x11]\n"
      ".inst 0xa041c560  // ld1w { z0.s-z3.s }, pn9.b/Z, [x11, #0x4, MUL VL]\n"
      ".inst 0xa042c578  // ld1w { z24.s-z27.s }, pn9.b/Z, [x11, #0x8, MUL VL]\n"
      ".inst 0xa043c56c  // ld1w { z12.s-z15.s }, pn9.b/Z, [x11, #0xc, MUL VL]\n"
      ".inst 0xc0840780  // mova za0h.s[x12], { z28.s-z31.s }\n"
      "addvl x11, x11, #16\n"
      ".inst 0xc0840401  // mova za1h.s[x12], { z0.s-z3.s }\n"
      ".inst 0xc0840702  // mova za2h.s[x12], { z24.s-z27.s }\n"
      ".inst 0xc0840583  // mova za3h.s[x12], { z12.s-z15.s }\n"
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
      ".inst 0x25ba6770  // whilelt pn8.s, x27, x26, VLx4\n"
      "tbnz x13, #0, 4f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
      "cbz x20, 5f\n"
      ".inst 0xa01bc288  // ld1w { z8.s-z11.s }, p8/Z, [x20, x27, LSL #2]\n"
      ".inst 0xc0900100  // addha za0.s, p0/M, p0/M, z8.s\n"
      ".inst 0xc0900121  // addha za1.s, p0/M, p0/M, z9.s\n"
      ".inst 0xc0900142  // addha za2.s, p0/M, p0/M, z10.s\n"
      ".inst 0xc0900163  // addha za3.s, p0/M, p0/M, z11.s\n"
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
      "add x20, x20, #0x3\n"
      "lsr x20, x20, #0x2\n"
      "lsr x21, x20, #0x2\n"
      "madd x23, x27, x22, x23\n"  // bptr = B + n * kstride_bytes
      "and x20, x20, #0x3\n"
      "cbz x21, 8f\n"
      "subs x21, x21, #0x1\n"
      "ld1b { z31.b }, p0/Z, [x24]\n"
      ".inst 0xa04086e8  // ld1b { z8.b-z11.b }, pn9.b/Z, [x23]\n"
      "ld1b { z1.b }, p0/Z, [x24, #1, MUL VL]\n"
      ".inst 0xa04186e4  // ld1b { z4.b-z7.b }, pn9.b/Z, [x23, #0x4, MUL VL]\n"
      "ld1b { z0.b }, p0/Z, [x24, #2, MUL VL]\n"
      ".inst 0xa04286ec  // ld1b { z12.b-z15.b }, pn9.b/Z, [x23, #0x8, MUL VL]\n"
      "ld1b { z3.b }, p0/Z, [x24, #3, MUL VL]\n"
      "addvl x24, x24, #4\n"
      ".inst 0xa04386f0  // ld1b { z16.b-z19.b }, pn9.b/Z, [x23, #0xc, MUL VL]\n"
      "addvl x23, x23, #16\n"
      "ble 7f\n"
      "6:"  // K loop
      ".inst 0xa08803e0  // smopa za0.s, p0/M, p0/M, z31.b, z8.b\n"
      "subs x21, x21, #0x1\n"
      ".inst 0xa08903e1  // smopa za1.s, p0/M, p0/M, z31.b, z9.b\n"
      ".inst 0xa08a03e2  // smopa za2.s, p0/M, p0/M, z31.b, z10.b\n"
      ".inst 0xa08b03e3  // smopa za3.s, p0/M, p0/M, z31.b, z11.b\n"
      "ld1b { z31.b }, p0/Z, [x24]\n"
      ".inst 0xa0840020  // smopa za0.s, p0/M, p0/M, z1.b, z4.b\n"
      ".inst 0xa04086e8  // ld1b { z8.b-z11.b }, pn9.b/Z, [x23]\n"
      ".inst 0xa0850021  // smopa za1.s, p0/M, p0/M, z1.b, z5.b\n"
      ".inst 0xa0860022  // smopa za2.s, p0/M, p0/M, z1.b, z6.b\n"
      ".inst 0xa0870023  // smopa za3.s, p0/M, p0/M, z1.b, z7.b\n"
      "ld1b { z1.b }, p0/Z, [x24, #1, MUL VL]\n"
      ".inst 0xa08c0000  // smopa za0.s, p0/M, p0/M, z0.b, z12.b\n"
      ".inst 0xa04186e4  // ld1b { z4.b-z7.b }, pn9.b/Z, [x23, #0x4, MUL VL]\n"
      ".inst 0xa08d0001  // smopa za1.s, p0/M, p0/M, z0.b, z13.b\n"
      ".inst 0xa08e0002  // smopa za2.s, p0/M, p0/M, z0.b, z14.b\n"
      ".inst 0xa08f0003  // smopa za3.s, p0/M, p0/M, z0.b, z15.b\n"
      "ld1b { z0.b }, p0/Z, [x24, #2, MUL VL]\n"
      ".inst 0xa04286ec  // ld1b { z12.b-z15.b }, pn9.b/Z, [x23, #0x8, MUL VL]\n"
      ".inst 0xa0900060  // smopa za0.s, p0/M, p0/M, z3.b, z16.b\n"
      ".inst 0xa0910061  // smopa za1.s, p0/M, p0/M, z3.b, z17.b\n"
      ".inst 0xa0920062  // smopa za2.s, p0/M, p0/M, z3.b, z18.b\n"
      ".inst 0xa0930063  // smopa za3.s, p0/M, p0/M, z3.b, z19.b\n"
      "ld1b { z3.b }, p0/Z, [x24, #3, MUL VL]\n"
      "addvl x24, x24, #4\n"
      ".inst 0xa04386f0  // ld1b { z16.b-z19.b }, pn9.b/Z, [x23, #0xc, MUL VL]\n"
      "addvl x23, x23, #16\n"
      "bgt 6b\n"
      "7:"  // K loop tail
      ".inst 0xa08803e0  // smopa za0.s, p0/M, p0/M, z31.b, z8.b\n"
      ".inst 0xa08903e1  // smopa za1.s, p0/M, p0/M, z31.b, z9.b\n"
      ".inst 0xa08a03e2  // smopa za2.s, p0/M, p0/M, z31.b, z10.b\n"
      ".inst 0xa08b03e3  // smopa za3.s, p0/M, p0/M, z31.b, z11.b\n"
      ".inst 0xa0840020  // smopa za0.s, p0/M, p0/M, z1.b, z4.b\n"
      ".inst 0xa0850021  // smopa za1.s, p0/M, p0/M, z1.b, z5.b\n"
      ".inst 0xa0860022  // smopa za2.s, p0/M, p0/M, z1.b, z6.b\n"
      ".inst 0xa0870023  // smopa za3.s, p0/M, p0/M, z1.b, z7.b\n"
      ".inst 0xa08c0000  // smopa za0.s, p0/M, p0/M, z0.b, z12.b\n"
      ".inst 0xa08d0001  // smopa za1.s, p0/M, p0/M, z0.b, z13.b\n"
      ".inst 0xa08e0002  // smopa za2.s, p0/M, p0/M, z0.b, z14.b\n"
      ".inst 0xa08f0003  // smopa za3.s, p0/M, p0/M, z0.b, z15.b\n"
      ".inst 0xa0900060  // smopa za0.s, p0/M, p0/M, z3.b, z16.b\n"
      ".inst 0xa0910061  // smopa za1.s, p0/M, p0/M, z3.b, z17.b\n"
      ".inst 0xa0920062  // smopa za2.s, p0/M, p0/M, z3.b, z18.b\n"
      ".inst 0xa0930063  // smopa za3.s, p0/M, p0/M, z3.b, z19.b\n"
      "8:"  // K oddments
      "cbz x20, 10f\n"
      "9:"  // K oddments: Loop
      "ld1b { z18.b }, p0/Z, [x24]\n"
      "subs x20, x20, #0x1\n"
      "addvl x24, x24, #1\n"
      ".inst 0xa04086fc  // ld1b { z28.b-z31.b }, pn9.b/Z, [x23]\n"
      "addvl x23, x23, #4\n"
      ".inst 0xa09c0240  // smopa za0.s, p0/M, p0/M, z18.b, z28.b\n"
      ".inst 0xa09d0241  // smopa za1.s, p0/M, p0/M, z18.b, z29.b\n"
      ".inst 0xa09e0242  // smopa za2.s, p0/M, p0/M, z18.b, z30.b\n"
      ".inst 0xa09f0243  // smopa za3.s, p0/M, p0/M, z18.b, z31.b\n"
      "bgt 9b\n"
      "10:"  // K oddments: End
      "tbz x13, #1, 14f\n"
      "tbz x13, #0, 12f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "11:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xa040c560  // ld1w { z0.s-z3.s }, pn9.b/Z, [x11]\n"
      ".inst 0xc0860408  // mova { z8.s-z11.s }, za0h.s[x12]\n"
      ".inst 0xc086042c  // mova { z12.s-z15.s }, za1h.s[x12]\n"
      ".inst 0xa041c57c  // ld1w { z28.s-z31.s }, pn9.b/Z, [x11, #0x4, MUL VL]\n"
      ".inst 0xc0860444  // mova { z4.s-z7.s }, za2h.s[x12]\n"
      ".inst 0xc0860470  // mova { z16.s-z19.s }, za3h.s[x12]\n"
      ".inst 0xa042c578  // ld1w { z24.s-z27.s }, pn9.b/Z, [x11, #0x8, MUL VL]\n"
      ".inst 0xa043c574  // ld1w { z20.s-z23.s }, pn9.b/Z, [x11, #0xc, MUL VL]\n"
      ".inst 0xc0840400  // mova za0h.s[x12], { z0.s-z3.s }\n"
      "addvl x11, x11, #16\n"
      ".inst 0xc0840781  // mova za1h.s[x12], { z28.s-z31.s }\n"
      ".inst 0xa060c548  // st1w { z8.s-z11.s }, pn9.b, [x10]\n"
      ".inst 0xc0840702  // mova za2h.s[x12], { z24.s-z27.s }\n"
      ".inst 0xa061c54c  // st1w { z12.s-z15.s }, pn9.b, [x10, #0x4, MUL VL]\n"
      ".inst 0xc0840683  // mova za3h.s[x12], { z20.s-z23.s }\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa062c544  // st1w { z4.s-z7.s }, pn9.b, [x10, #0x8, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa063c550  // st1w { z16.s-z19.s }, pn9.b, [x10, #0xc, MUL VL]\n"
      "addvl x10, x10, #16\n"
      "blt 11b\n"
      "b 21f\n"
      "12:"  // Store to partial result buffer: Store only
      "mov x12, #0x0\n"
      "cntw x20\n"
      "13:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xc0860404  // mova { z4.s-z7.s }, za0h.s[x12]\n"
      ".inst 0xc0860430  // mova { z16.s-z19.s }, za1h.s[x12]\n"
      ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xa060c544  // st1w { z4.s-z7.s }, pn9.b, [x10]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa061c550  // st1w { z16.s-z19.s }, pn9.b, [x10, #0x4, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa062c548  // st1w { z8.s-z11.s }, pn9.b, [x10, #0x8, MUL VL]\n"
      ".inst 0xa063c54c  // st1w { z12.s-z15.s }, pn9.b, [x10, #0xc, MUL VL]\n"
      "addvl x10, x10, #16\n"
      "blt 13b\n"
      "b 21f\n"
      "14:"  // Store to output array
      "ldr x23, [%x[args], %[offsetof_C]]\n"
      "sub x21, x9, x28\n"
      "ld1rw { z18.s }, p0/Z, [%x[dq], %[offset_DequantizeFloat_scale]]\n"
      "fmov z20.s, #0x0\n"
      "ldr x22, [%x[args], %[offsetof_ldcb]]\n"
      "fmov z21.s, #0x0\n"
      "fmov z22.s, #0x0\n"
      "ldr x20, [%x[args], %[offsetof_late_bias]]\n"
      "fmov z23.s, #0x0\n"
      "add x23, x23, x27, LSL #2\n"  // C += n
      "madd x23, x28, x22, x23\n"  // C += m * ldc
      "cbz x20, 15f\n"
      "add x20, x20, x27, LSL #2\n"
      ".inst 0xa040c294  // ld1w { z20.s-z23.s }, p8/Z, [x20]\n"
      "15:"  // Store to output array: no late bias
      "cntw x20\n"
      "ld1rw { z17.s }, p0/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
      "mov x12, #0x0\n"
      "cmp x21, x20\n"
      "ld1rw { z16.s }, p0/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
      "csel x20, x21, x20, LT\n"
      "lsr x21, x20, #0x2\n"
      "and x20, x20, #0x3\n"
      "cbz x21, 17f\n"
      "16:"  // Store to output array: Accumulator row 0 loop
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      ".inst 0xc0860424  // mova { z4.s-z7.s }, za1h.s[x12]\n"
      ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xc132e000  // scvtf { z0.s-z3.s }, { z0.s-z3.s }\n"
      ".inst 0xc132e084  // scvtf { z4.s-z7.s }, { z4.s-z7.s }\n"
      ".inst 0xc132e108  // scvtf { z8.s-z11.s }, { z8.s-z11.s }\n"
      ".inst 0xc132e18c  // scvtf { z12.s-z15.s }, { z12.s-z15.s }\n"
      "fmad z0.s, p0/M, z18.s, z20.s\n"
      "fmad z1.s, p0/M, z18.s, z20.s\n"
      "fmad z2.s, p0/M, z18.s, z20.s\n"
      "fmad z3.s, p0/M, z18.s, z20.s\n"
      "add x12, x12, #0x4\n"
      "fmad z4.s, p0/M, z18.s, z21.s\n"
      "fmad z5.s, p0/M, z18.s, z21.s\n"
      "cmp x12, x21, LSL #2\n"
      "fmad z6.s, p0/M, z18.s, z21.s\n"
      "fmad z7.s, p0/M, z18.s, z21.s\n"
      "fmad z8.s, p0/M, z18.s, z22.s\n"
      "fmad z9.s, p0/M, z18.s, z22.s\n"
      "fmad z10.s, p0/M, z18.s, z22.s\n"
      "fmad z11.s, p0/M, z18.s, z22.s\n"
      "fmad z12.s, p0/M, z18.s, z23.s\n"
      "fmad z13.s, p0/M, z18.s, z23.s\n"
      "fmad z14.s, p0/M, z18.s, z23.s\n"
      "fmad z15.s, p0/M, z18.s, z23.s\n"
      ".inst 0xc1b0ca20  // fclamp { z0.s-z3.s }, z17.s, z16.s\n"
      ".inst 0xc1b0ca24  // fclamp { z4.s-z7.s }, z17.s, z16.s\n"
      ".inst 0xc1b0ca28  // fclamp { z8.s-z11.s }, z17.s, z16.s\n"
      ".inst 0xc1b0ca2c  // fclamp { z12.s-z15.s }, z17.s, z16.s\n"
      ".inst 0xa160c2e0  // st1w { z0.s, z4.s, z8.s, z12.s }, p8, [x23]\n"
      "add x23, x23, x22\n"
      ".inst 0xa160c2e1  // st1w { z1.s, z5.s, z9.s, z13.s }, p8, [x23]\n"
      "add x23, x23, x22\n"
      ".inst 0xa160c2e2  // st1w { z2.s, z6.s, z10.s, z14.s }, p8, [x23]\n"
      "add x23, x23, x22\n"
      ".inst 0xa160c2e3  // st1w { z3.s, z7.s, z11.s, z15.s }, p8, [x23]\n"
      "add x23, x23, x22\n"
      "blt 16b\n"
      "17:"  // Store to output array: Accumulator row 0 oddments
      "cbz x20, 18f\n"
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      ".inst 0xc0860424  // mova { z4.s-z7.s }, za1h.s[x12]\n"
      ".inst 0xc0860448  // mova { z8.s-z11.s }, za2h.s[x12]\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      ".inst 0xc132e000  // scvtf { z0.s-z3.s }, { z0.s-z3.s }\n"
      ".inst 0xc132e084  // scvtf { z4.s-z7.s }, { z4.s-z7.s }\n"
      ".inst 0xc132e108  // scvtf { z8.s-z11.s }, { z8.s-z11.s }\n"
      ".inst 0xc132e18c  // scvtf { z12.s-z15.s }, { z12.s-z15.s }\n"
      "fmad z0.s, p0/M, z18.s, z20.s\n"
      "fmad z1.s, p0/M, z18.s, z20.s\n"
      "fmad z2.s, p0/M, z18.s, z20.s\n"
      "fmad z3.s, p0/M, z18.s, z20.s\n"
      "subs x20, x20, #0x1\n"
      "fmad z4.s, p0/M, z18.s, z21.s\n"
      "fmad z5.s, p0/M, z18.s, z21.s\n"
      "fmad z6.s, p0/M, z18.s, z21.s\n"
      "fmad z7.s, p0/M, z18.s, z21.s\n"
      "fmad z8.s, p0/M, z18.s, z22.s\n"
      "fmad z9.s, p0/M, z18.s, z22.s\n"
      "fmad z10.s, p0/M, z18.s, z22.s\n"
      "fmad z11.s, p0/M, z18.s, z22.s\n"
      "fmad z12.s, p0/M, z18.s, z23.s\n"
      "fmad z13.s, p0/M, z18.s, z23.s\n"
      "fmad z14.s, p0/M, z18.s, z23.s\n"
      "fmad z15.s, p0/M, z18.s, z23.s\n"
      ".inst 0xc1b0ca20  // fclamp { z0.s-z3.s }, z17.s, z16.s\n"
      ".inst 0xc1b0ca24  // fclamp { z4.s-z7.s }, z17.s, z16.s\n"
      ".inst 0xc1b0ca28  // fclamp { z8.s-z11.s }, z17.s, z16.s\n"
      ".inst 0xc1b0ca2c  // fclamp { z12.s-z15.s }, z17.s, z16.s\n"
      ".inst 0xa160c2e0  // st1w { z0.s, z4.s, z8.s, z12.s }, p8, [x23]\n"
      "add x23, x23, x22\n"
      "beq 18f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xa160c2e1  // st1w { z1.s, z5.s, z9.s, z13.s }, p8, [x23]\n"
      "add x23, x23, x22\n"
      "beq 18f\n"
      ".inst 0xa160c2e2  // st1w { z2.s, z6.s, z10.s, z14.s }, p8, [x23]\n"
      "18:"  // Store to output array: Accumulator row 0 oddments: End
      "19:"  // Store to output array: End
      "tbz x13, #0, 21f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "20:"  // Store to output array: Refill accumulators: Loop
      ".inst 0xa040c574  // ld1w { z20.s-z23.s }, pn9.b/Z, [x11]\n"
      ".inst 0xa041c56c  // ld1w { z12.s-z15.s }, pn9.b/Z, [x11, #0x4, MUL VL]\n"
      ".inst 0xa042c560  // ld1w { z0.s-z3.s }, pn9.b/Z, [x11, #0x8, MUL VL]\n"
      ".inst 0xa043c568  // ld1w { z8.s-z11.s }, pn9.b/Z, [x11, #0xc, MUL VL]\n"
      ".inst 0xc0840680  // mova za0h.s[x12], { z20.s-z23.s }\n"
      "addvl x11, x11, #16\n"
      ".inst 0xc0840581  // mova za1h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xc0840402  // mova za2h.s[x12], { z0.s-z3.s }\n"
      ".inst 0xc0840503  // mova za3h.s[x12], { z8.s-z11.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 20b\n"
      "21:"  // End block
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
      : [args] "r" (&args), [dq] "r" (&dq), [offset_DequantizeFloat_scale] "I" (offsetof(DequantizeFloat, scale)), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_KernelArgs_max] "I" (offsetof(KernelArgs, max)), [offsetof_KernelArgs_min] "I" (offsetof(KernelArgs, min)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_kstride_bytes] "I" (offsetof(KernelArgs, kstride_bytes)), [offsetof_late_bias] "I" (offsetof(KernelArgs, late_bias)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb))
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

}  // namespace arm_gemm

#endif  // ARM_COMPUTE_ENABLE_SME2
