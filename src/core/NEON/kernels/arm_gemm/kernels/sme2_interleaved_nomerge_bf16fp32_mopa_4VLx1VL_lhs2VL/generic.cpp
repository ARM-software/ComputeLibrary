/*
 * Copyright (c) 2022-2026 Arm Limited.
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

#if defined(ARM_COMPUTE_ENABLE_BF16) && defined(ARM_COMPUTE_ENABLE_SME2) && defined(__aarch64__)

#include "arm_gemm/arm_gemm.hpp"

#include "arm_common/bfloat.hpp"
#include "arm_common/internal/utils.hpp"

namespace arm_gemm {

void sme2_interleaved_nomerge_bf16fp32_mopa_4VLx1VL_lhs2VL(const bfloat16 *const A, const bfloat16 *const B, float *const C, int ldc, const int M, const int N, const int K, const float *const bias, const Activation act, bool accumulate, float *const accumulator_buffer)
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
    const long M, N, K;
    float min = -std::numeric_limits<float>::infinity();
    float max = std::numeric_limits<float>::infinity();

    const float *const bias;


    float *const accumulator_buffer;
    uint64_t flags;
  };

  // Construct arguments for this kernel
  KernelArgs args(A, B, C, ldc, M, N, K, bias, act, accumulate, accumulator_buffer);

  __asm__ __volatile__(
      "ldr x7, [%x[args], %[offsetof_flags]]\n"
      ".inst 0xd503477f  // SMSTART ZA\n"
      "ptrue p1.b\n"
      ".inst 0x25207810  // ptrue pn8.b\n"
      "ldr x8, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "ldr x17, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "tbz x7, #0, 2f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "1:"  // Initial accumulator load from buffer: Loop
      ".inst 0xa040c100  // ld1w { z0.s-z3.s }, pn8.b/Z, [x8]\n"
      ".inst 0xa041c104  // ld1w { z4.s-z7.s }, pn8.b/Z, [x8, #0x4, MUL VL]\n"
      ".inst 0xa042c114  // ld1w { z20.s-z23.s }, pn8.b/Z, [x8, #0x8, MUL VL]\n"
      ".inst 0xa043c108  // ld1w { z8.s-z11.s }, pn8.b/Z, [x8, #0xc, MUL VL]\n"
      ".inst 0xc0840400  // mova za0h.s[x12], { z0.s-z3.s }\n"
      "addvl x8, x8, #16\n"
      ".inst 0xc0840481  // mova za1h.s[x12], { z4.s-z7.s }\n"
      ".inst 0xc0840682  // mova za2h.s[x12], { z20.s-z23.s }\n"
      ".inst 0xc0840503  // mova za3h.s[x12], { z8.s-z11.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 1b\n"
      "2:"  // Initial accumulator load from buffer: End
      "ldr x16, [%x[args], %[offsetof_K]]\n"
      "cntb x15, ALL, MUL #2\n"
      "mov x14, #0\n"
      "ldr w13, [%x[args], %[offsetof_M]]\n"
      "mov x11, #0\n"
      "ldr w10, [%x[args], %[offsetof_N]]\n"
      "add x16, x16, #0x1\n"
      "ldr x9, [%x[args], %[offsetof_A]]\n"
      "lsr x16, x16, #0x1\n"
      "mul x15, x15, x16\n"
      "3:"  // M loop
      "ldr x28, [%x[args], %[offsetof_B]]\n"
      "4:"  // N loop
      "mov x22, x9\n"
      "whilelt p0.s, x11, x10\n"
      "add x27, x22, x15\n"
      "tbnz x7, #0, 5f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
      "cbz x20, 6f\n"
      "fmov z7.s, #1.0\n"
      "ld1w { z0.s }, p0/Z, [x20, x11, LSL #2]\n"
      ".inst 0x808024e0  // fmopa za0.s, p1/M, p1/M, z7.s, z0.s\n"
      ".inst 0x808024e1  // fmopa za1.s, p1/M, p1/M, z7.s, z0.s\n"
      ".inst 0x808024e2  // fmopa za2.s, p1/M, p1/M, z7.s, z0.s\n"
      ".inst 0x808024e3  // fmopa za3.s, p1/M, p1/M, z7.s, z0.s\n"
      "5:"  // Prepare accumulators: Test for last block
      "mov x20, x11\n"
      "mov x21, x14\n"
      "incw x20\n"
      "incw x21, ALL, MUL #4\n"
      "cmp x20, x10\n"
      "mov x20, x7\n"
      "csel x21, x14, x21, LT\n"
      "bfm x7, XZR, #0, #0  // bfc x7, #0, #0x1\n"
      "cmp x21, x13\n"
      "csel x7, x20, x7, LT\n"
      "6:"  // Prepare accumulators: End
      "lsr x21, x16, #0x2\n"
      "and x20, x16, #0x3\n"
      "cbz x21, 9f\n"
      "subs x21, x21, #0x1\n"
      ".inst 0xa040a2cc  // ld1h { z12.h-z15.h }, pn8.b/Z, [x22]\n"
      ".inst 0xa041a2c8  // ld1h { z8.h-z11.h }, pn8.b/Z, [x22, #0x4, MUL VL]\n"
      "addvl x22, x22, #8\n"
      ".inst 0xa140a373  // ld1h { z19.h, z23.h, z27.h, z31.h }, pn8.b/Z, [x27]\n"
      ".inst 0xa141a371  // ld1h { z17.h, z21.h, z25.h, z29.h }, pn8.b/Z, [x27, #0x4, MUL VL]\n"
      "addvl x27, x27, #8\n"
      ".inst 0xa140a390  // ld1h { z16.h, z20.h, z24.h, z28.h }, pn8.b/Z, [x28]\n"
      "addvl x28, x28, #4\n"
      "ble 8f\n"
      "7:"  // K loop
      ".inst 0x81902580  // bfmopa za0.s, p1/M, p1/M, z12.h, z16.h\n"
      "subs x21, x21, #0x1\n"
      ".inst 0x819025a1  // bfmopa za1.s, p1/M, p1/M, z13.h, z16.h\n"
      ".inst 0x81902662  // bfmopa za2.s, p1/M, p1/M, z19.h, z16.h\n"
      ".inst 0x819026e3  // bfmopa za3.s, p1/M, p1/M, z23.h, z16.h\n"
      ".inst 0x819425c0  // bfmopa za0.s, p1/M, p1/M, z14.h, z20.h\n"
      ".inst 0x819425e1  // bfmopa za1.s, p1/M, p1/M, z15.h, z20.h\n"
      ".inst 0xa040a2cc  // ld1h { z12.h-z15.h }, pn8.b/Z, [x22]\n"
      ".inst 0x81942762  // bfmopa za2.s, p1/M, p1/M, z27.h, z20.h\n"
      ".inst 0x819427e3  // bfmopa za3.s, p1/M, p1/M, z31.h, z20.h\n"
      ".inst 0xa140a373  // ld1h { z19.h, z23.h, z27.h, z31.h }, pn8.b/Z, [x27]\n"
      ".inst 0x81982500  // bfmopa za0.s, p1/M, p1/M, z8.h, z24.h\n"
      ".inst 0x81982521  // bfmopa za1.s, p1/M, p1/M, z9.h, z24.h\n"
      ".inst 0x81982622  // bfmopa za2.s, p1/M, p1/M, z17.h, z24.h\n"
      ".inst 0x819826a3  // bfmopa za3.s, p1/M, p1/M, z21.h, z24.h\n"
      ".inst 0x819c2540  // bfmopa za0.s, p1/M, p1/M, z10.h, z28.h\n"
      ".inst 0x819c2561  // bfmopa za1.s, p1/M, p1/M, z11.h, z28.h\n"
      ".inst 0xa041a2c8  // ld1h { z8.h-z11.h }, pn8.b/Z, [x22, #0x4, MUL VL]\n"
      "addvl x22, x22, #8\n"
      ".inst 0x819c2722  // bfmopa za2.s, p1/M, p1/M, z25.h, z28.h\n"
      ".inst 0x819c27a3  // bfmopa za3.s, p1/M, p1/M, z29.h, z28.h\n"
      ".inst 0xa141a371  // ld1h { z17.h, z21.h, z25.h, z29.h }, pn8.b/Z, [x27, #0x4, MUL VL]\n"
      "addvl x27, x27, #8\n"
      ".inst 0xa140a390  // ld1h { z16.h, z20.h, z24.h, z28.h }, pn8.b/Z, [x28]\n"
      "addvl x28, x28, #4\n"
      "bgt 7b\n"
      "8:"  // K loop tail
      ".inst 0x81902580  // bfmopa za0.s, p1/M, p1/M, z12.h, z16.h\n"
      ".inst 0x819025a1  // bfmopa za1.s, p1/M, p1/M, z13.h, z16.h\n"
      ".inst 0x81902662  // bfmopa za2.s, p1/M, p1/M, z19.h, z16.h\n"
      ".inst 0x819026e3  // bfmopa za3.s, p1/M, p1/M, z23.h, z16.h\n"
      ".inst 0x819425c0  // bfmopa za0.s, p1/M, p1/M, z14.h, z20.h\n"
      ".inst 0x819425e1  // bfmopa za1.s, p1/M, p1/M, z15.h, z20.h\n"
      ".inst 0x81942762  // bfmopa za2.s, p1/M, p1/M, z27.h, z20.h\n"
      ".inst 0x819427e3  // bfmopa za3.s, p1/M, p1/M, z31.h, z20.h\n"
      ".inst 0x81982500  // bfmopa za0.s, p1/M, p1/M, z8.h, z24.h\n"
      ".inst 0x81982521  // bfmopa za1.s, p1/M, p1/M, z9.h, z24.h\n"
      ".inst 0x81982622  // bfmopa za2.s, p1/M, p1/M, z17.h, z24.h\n"
      ".inst 0x819826a3  // bfmopa za3.s, p1/M, p1/M, z21.h, z24.h\n"
      ".inst 0x819c2540  // bfmopa za0.s, p1/M, p1/M, z10.h, z28.h\n"
      ".inst 0x819c2561  // bfmopa za1.s, p1/M, p1/M, z11.h, z28.h\n"
      ".inst 0x819c2722  // bfmopa za2.s, p1/M, p1/M, z25.h, z28.h\n"
      ".inst 0x819c27a3  // bfmopa za3.s, p1/M, p1/M, z29.h, z28.h\n"
      "9:"  // K oddments
      "cbz x20, 11f\n"
      "10:"  // K oddments: Loop
      ".inst 0xa04022c4  // ld1h { z4.h-z5.h }, pn8.b/Z, [x22]\n"
      "subs x20, x20, #0x1\n"
      "addvl x22, x22, #2\n"
      ".inst 0xa1402374  // ld1h { z20.h, z28.h }, pn8.b/Z, [x27]\n"
      "addvl x27, x27, #2\n"
      "ld1h { z0.h }, p1/Z, [x28]\n"
      "addvl x28, x28, #1\n"
      ".inst 0x81802480  // bfmopa za0.s, p1/M, p1/M, z4.h, z0.h\n"
      ".inst 0x818024a1  // bfmopa za1.s, p1/M, p1/M, z5.h, z0.h\n"
      ".inst 0x81802682  // bfmopa za2.s, p1/M, p1/M, z20.h, z0.h\n"
      ".inst 0x81802783  // bfmopa za3.s, p1/M, p1/M, z28.h, z0.h\n"
      "bgt 10b\n"
      "11:"  // K oddments: End
      "tbz x7, #1, 15f\n"
      "tbz x7, #0, 13f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "12:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xa040c114  // ld1w { z20.s-z23.s }, pn8.b/Z, [x8]\n"
      ".inst 0xc0860408  // mova { z8.s-z11.s }, za0h.s[x12]\n"
      ".inst 0xc086043c  // mova { z28.s-z31.s }, za1h.s[x12]\n"
      ".inst 0xa041c110  // ld1w { z16.s-z19.s }, pn8.b/Z, [x8, #0x4, MUL VL]\n"
      ".inst 0xc0860458  // mova { z24.s-z27.s }, za2h.s[x12]\n"
      ".inst 0xc0860464  // mova { z4.s-z7.s }, za3h.s[x12]\n"
      ".inst 0xa042c100  // ld1w { z0.s-z3.s }, pn8.b/Z, [x8, #0x8, MUL VL]\n"
      ".inst 0xa043c10c  // ld1w { z12.s-z15.s }, pn8.b/Z, [x8, #0xc, MUL VL]\n"
      ".inst 0xc0840680  // mova za0h.s[x12], { z20.s-z23.s }\n"
      "addvl x8, x8, #16\n"
      ".inst 0xc0840601  // mova za1h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xa060c228  // st1w { z8.s-z11.s }, pn8.b, [x17]\n"
      ".inst 0xc0840402  // mova za2h.s[x12], { z0.s-z3.s }\n"
      ".inst 0xa061c23c  // st1w { z28.s-z31.s }, pn8.b, [x17, #0x4, MUL VL]\n"
      ".inst 0xc0840583  // mova za3h.s[x12], { z12.s-z15.s }\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa062c238  // st1w { z24.s-z27.s }, pn8.b, [x17, #0x8, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa063c224  // st1w { z4.s-z7.s }, pn8.b, [x17, #0xc, MUL VL]\n"
      "addvl x17, x17, #16\n"
      "blt 12b\n"
      "b 43f\n"
      "13:"  // Store to partial result buffer: Store only
      "mov x12, #0\n"
      "cntw x20\n"
      "14:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xc0860410  // mova { z16.s-z19.s }, za0h.s[x12]\n"
      ".inst 0xc086042c  // mova { z12.s-z15.s }, za1h.s[x12]\n"
      ".inst 0xc0860444  // mova { z4.s-z7.s }, za2h.s[x12]\n"
      ".inst 0xc0860468  // mova { z8.s-z11.s }, za3h.s[x12]\n"
      ".inst 0xa060c230  // st1w { z16.s-z19.s }, pn8.b, [x17]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa061c22c  // st1w { z12.s-z15.s }, pn8.b, [x17, #0x4, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa062c224  // st1w { z4.s-z7.s }, pn8.b, [x17, #0x8, MUL VL]\n"
      ".inst 0xa063c228  // st1w { z8.s-z11.s }, pn8.b, [x17, #0xc, MUL VL]\n"
      "addvl x17, x17, #16\n"
      "blt 14b\n"
      "b 43f\n"
      "15:"  // Store to output array
      "ldr x26, [%x[args], %[offsetof_C]]\n"
      "sub x25, x13, x14\n"
      "ldr x24, [%x[args], %[offsetof_ldcb]]\n"
      "add x26, x26, x11, LSL #2\n"  // C += n
      "madd x26, x14, x24, x26\n"  // C += m * ldc
      "tbz x7, #2, 28f\n"
      "cntw x23\n"
      "mov x12, #0\n"
      "cmp x25, x23\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 17f\n"
      "16:"  // Store to output array: Skip activation: Accumulator row 0 loop
      ".inst 0xc0860400  // mova { z0.s-z3.s }, za0h.s[x12]\n"
      "add x12, x12, #0x4\n"
      "st1w { z0.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "cmp x12, x21, LSL #2\n"
      "st1w { z1.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z2.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z3.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 16b\n"
      "17:"  // Store to output array: Skip activation: Accumulator row 0 oddments
      "cbz x20, 18f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc0860408  // mova { z8.s-z11.s }, za0h.s[x12]\n"
      "st1w { z8.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 18f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z9.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 18f\n"
      "st1w { z10.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "18:"  // Store to output array: Skip activation: Accumulator row 0 oddments: End
      "subs x25, x25, x22\n"
      "beq 28f\n"
      "cmp x25, x23\n"
      "mov x12, #0\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 20f\n"
      "19:"  // Store to output array: Skip activation: Accumulator row 1 loop
      ".inst 0xc086042c  // mova { z12.s-z15.s }, za1h.s[x12]\n"
      "add x12, x12, #0x4\n"
      "st1w { z12.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "cmp x12, x21, LSL #2\n"
      "st1w { z13.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z14.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z15.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 19b\n"
      "20:"  // Store to output array: Skip activation: Accumulator row 1 oddments
      "cbz x20, 21f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc0860428  // mova { z8.s-z11.s }, za1h.s[x12]\n"
      "st1w { z8.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 21f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z9.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 21f\n"
      "st1w { z10.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "21:"  // Store to output array: Skip activation: Accumulator row 1 oddments: End
      "subs x25, x25, x22\n"
      "beq 28f\n"
      "cmp x25, x23\n"
      "mov x12, #0\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 23f\n"
      "22:"  // Store to output array: Skip activation: Accumulator row 2 loop
      ".inst 0xc0860454  // mova { z20.s-z23.s }, za2h.s[x12]\n"
      "add x12, x12, #0x4\n"
      "st1w { z20.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "cmp x12, x21, LSL #2\n"
      "st1w { z21.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z22.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z23.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 22b\n"
      "23:"  // Store to output array: Skip activation: Accumulator row 2 oddments
      "cbz x20, 24f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc086044c  // mova { z12.s-z15.s }, za2h.s[x12]\n"
      "st1w { z12.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 24f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z13.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 24f\n"
      "st1w { z14.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "24:"  // Store to output array: Skip activation: Accumulator row 2 oddments: End
      "subs x25, x25, x22\n"
      "beq 28f\n"
      "cmp x25, x23\n"
      "mov x12, #0\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 26f\n"
      "25:"  // Store to output array: Skip activation: Accumulator row 3 loop
      ".inst 0xc0860478  // mova { z24.s-z27.s }, za3h.s[x12]\n"
      "add x12, x12, #0x4\n"
      "st1w { z24.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "cmp x12, x21, LSL #2\n"
      "st1w { z25.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z26.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z27.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 25b\n"
      "26:"  // Store to output array: Skip activation: Accumulator row 3 oddments
      "cbz x20, 27f\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc0860470  // mova { z16.s-z19.s }, za3h.s[x12]\n"
      "st1w { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 27f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 27f\n"
      "st1w { z18.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "27:"  // Store to output array: Skip activation: Accumulator row 3 oddments: End
      "subs x25, x25, x22\n"
      "beq 28f\n"
      "b 41f\n"
      "28:"  // Store to output array: Skip activation: End
      "cntw x23\n"
      "ld1rw { z25.s }, p1/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
      "mov x12, #0\n"
      "cmp x25, x23\n"
      "ld1rw { z24.s }, p1/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 30f\n"
      "29:"  // Store to output array: Accumulator row 0 loop
      ".inst 0xc0860414  // mova { z20.s-z23.s }, za0h.s[x12]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xc1b8cb34  // fclamp { z20.s-z23.s }, z25.s, z24.s\n"
      "cmp x12, x21, LSL #2\n"
      "st1w { z20.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z21.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z22.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z23.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 29b\n"
      "30:"  // Store to output array: Accumulator row 0 oddments
      "cbz x20, 31f\n"
      ".inst 0xc0860410  // mova { z16.s-z19.s }, za0h.s[x12]\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc1b8cb30  // fclamp { z16.s-z19.s }, z25.s, z24.s\n"
      "st1w { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 31f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 31f\n"
      "st1w { z18.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "31:"  // Store to output array: Accumulator row 0 oddments: End
      "subs x25, x25, x22\n"
      "beq 41f\n"
      "cmp x25, x23\n"
      "mov x12, #0\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 33f\n"
      "32:"  // Store to output array: Accumulator row 1 loop
      ".inst 0xc0860430  // mova { z16.s-z19.s }, za1h.s[x12]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xc1b8cb30  // fclamp { z16.s-z19.s }, z25.s, z24.s\n"
      "cmp x12, x21, LSL #2\n"
      "st1w { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z18.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z19.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 32b\n"
      "33:"  // Store to output array: Accumulator row 1 oddments
      "cbz x20, 34f\n"
      ".inst 0xc0860430  // mova { z16.s-z19.s }, za1h.s[x12]\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc1b8cb30  // fclamp { z16.s-z19.s }, z25.s, z24.s\n"
      "st1w { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 34f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 34f\n"
      "st1w { z18.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "34:"  // Store to output array: Accumulator row 1 oddments: End
      "subs x25, x25, x22\n"
      "beq 41f\n"
      "cmp x25, x23\n"
      "mov x12, #0\n"
      "csel x22, x25, x23, LT\n"
      "lsr x21, x22, #0x2\n"
      "and x20, x22, #0x3\n"
      "cbz x21, 36f\n"
      "35:"  // Store to output array: Accumulator row 2 loop
      ".inst 0xc0860450  // mova { z16.s-z19.s }, za2h.s[x12]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xc1b8cb30  // fclamp { z16.s-z19.s }, z25.s, z24.s\n"
      "cmp x12, x21, LSL #2\n"
      "st1w { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z18.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z19.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 35b\n"
      "36:"  // Store to output array: Accumulator row 2 oddments
      "cbz x20, 37f\n"
      ".inst 0xc086045c  // mova { z28.s-z31.s }, za2h.s[x12]\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc1b8cb3c  // fclamp { z28.s-z31.s }, z25.s, z24.s\n"
      "st1w { z28.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 37f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z29.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 37f\n"
      "st1w { z30.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "37:"  // Store to output array: Accumulator row 2 oddments: End
      "subs x25, x25, x22\n"
      "beq 41f\n"
      "cmp x25, x23\n"
      "mov x12, #0\n"
      "csel x20, x25, x23, LT\n"
      "lsr x21, x20, #0x2\n"
      "and x20, x20, #0x3\n"
      "cbz x21, 39f\n"
      "38:"  // Store to output array: Accumulator row 3 loop
      ".inst 0xc0860470  // mova { z16.s-z19.s }, za3h.s[x12]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xc1b8cb30  // fclamp { z16.s-z19.s }, z25.s, z24.s\n"
      "cmp x12, x21, LSL #2\n"
      "st1w { z16.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z17.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z18.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "st1w { z19.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "blt 38b\n"
      "39:"  // Store to output array: Accumulator row 3 oddments
      "cbz x20, 40f\n"
      ".inst 0xc086046c  // mova { z12.s-z15.s }, za3h.s[x12]\n"
      "subs x20, x20, #0x1\n"
      ".inst 0xc1b8cb2c  // fclamp { z12.s-z15.s }, z25.s, z24.s\n"
      "st1w { z12.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 40f\n"
      "subs x20, x20, #0x1\n"
      "st1w { z13.s }, p0, [x26]\n"
      "add x26, x26, x24\n"
      "beq 40f\n"
      "st1w { z14.s }, p0, [x26]\n"
      "40:"  // Store to output array: Accumulator row 3 oddments: End
      "41:"  // Store to output array: End
      "tbz x7, #0, 43f\n"
      "mov x12, #0\n"
      "cntw x20\n"
      "42:"  // Store to output array: Refill accumulators: Loop
      ".inst 0xa040c10c  // ld1w { z12.s-z15.s }, pn8.b/Z, [x8]\n"
      ".inst 0xa041c11c  // ld1w { z28.s-z31.s }, pn8.b/Z, [x8, #0x4, MUL VL]\n"
      ".inst 0xa042c104  // ld1w { z4.s-z7.s }, pn8.b/Z, [x8, #0x8, MUL VL]\n"
      ".inst 0xa043c108  // ld1w { z8.s-z11.s }, pn8.b/Z, [x8, #0xc, MUL VL]\n"
      ".inst 0xc0840580  // mova za0h.s[x12], { z12.s-z15.s }\n"
      "addvl x8, x8, #16\n"
      ".inst 0xc0840781  // mova za1h.s[x12], { z28.s-z31.s }\n"
      ".inst 0xc0840482  // mova za2h.s[x12], { z4.s-z7.s }\n"
      ".inst 0xc0840503  // mova za3h.s[x12], { z8.s-z11.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 42b\n"
      "43:"  // End block
      "incw x11\n"
      "cmp x11, x10\n"
      "blt 4b\n"
      "incw x14, ALL, MUL #4\n"
      "mov x11, #0\n"
      "cmp x14, x13\n"
      "mov x9, x27\n"
      "blt 3b\n"
      ".inst 0xd503467f  // SMSTOP\n"
      :
      : [args] "r" (&args), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_KernelArgs_max] "I" (offsetof(KernelArgs, max)), [offsetof_KernelArgs_min] "I" (offsetof(KernelArgs, min)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb))
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

}  // namespace arm_gemm

#endif // defined(ARM_COMPUTE_ENABLE_BF16) && defined(ARM_COMPUTE_ENABLE_SME2) && defined(__aarch64__)

