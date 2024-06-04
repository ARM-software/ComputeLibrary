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

void sme2_interleaved_nomerge_s8q_mopa_1VLx4VL(const int8_t *const A, const int8_t *const B, int8_t *const C, int ldc, const int M, const int N, const int K, const int32_t *const bias, const Requantize32 &rq, const int n_0, bool accumulate, int32_t *const accumulator_buffer)
{
  struct KernelArgs
  {
    KernelArgs(
      const int8_t *const A,
      const int8_t *const B,
      int8_t *const C, const int ldc,
      const int M, const int N, const int K,
      const int32_t *const bias,
      const Requantize32 &rq,
      const int n_0,
      bool accumulate,
      int32_t *const accumulator_buffer
    ) : A(A),
        B(B), kstride_bytes(roundup(K, 4) * sizeof(int8_t)),
        C(C), ldcb(ldc * sizeof(int8_t)),
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

    const int8_t *const A;
    const int8_t *const B;
    const long kstride_bytes;
    int8_t *const C;
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
      "ldr x15, [%x[args], %[offsetof_flags]]\n"
      ".inst 0xd503477f  // SMSTART ZA\n"
      "ptrue p1.b\n"
      ".inst 0x25207811  // ptrue pn9.b\n"
      "ldr x14, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "ldr x13, [%x[args], %[offsetof_accumulator_buffer]]\n"
      "tbz x15, #0, 2f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "1:"  // Initial accumulator load from buffer: Loop
      ".inst 0xa040c5d8  // ld1w { z24.s-z27.s }, pn9.b/Z, [x14]\n"
      ".inst 0xa041c5c0  // ld1w { z0.s-z3.s }, pn9.b/Z, [x14, #0x4, MUL VL]\n"
      ".inst 0xa042c5dc  // ld1w { z28.s-z31.s }, pn9.b/Z, [x14, #0x8, MUL VL]\n"
      ".inst 0xa043c5c4  // ld1w { z4.s-z7.s }, pn9.b/Z, [x14, #0xc, MUL VL]\n"
      ".inst 0xc0840700  // mova za0h.s[x12], { z24.s-z27.s }\n"
      "addvl x14, x14, #16\n"
      ".inst 0xc0840401  // mova za1h.s[x12], { z0.s-z3.s }\n"
      ".inst 0xc0840782  // mova za2h.s[x12], { z28.s-z31.s }\n"
      ".inst 0xc0840483  // mova za3h.s[x12], { z4.s-z7.s }\n"
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
      ".inst 0x25bc6530  // whilelt pn8.s, x9, x28, VLx4\n"
      "tbnz x15, #0, 4f\n"
      "ldr x20, [%x[args], %[offsetof_bias]]\n"
      ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
      "cbz x20, 5f\n"
      ".inst 0xa009c290  // ld1w { z16.s-z19.s }, p8/Z, [x20, x9, LSL #2]\n"
      ".inst 0xc0902600  // addha za0.s, p1/M, p1/M, z16.s\n"
      ".inst 0xc0902621  // addha za1.s, p1/M, p1/M, z17.s\n"
      ".inst 0xc0902642  // addha za2.s, p1/M, p1/M, z18.s\n"
      ".inst 0xc0902663  // addha za3.s, p1/M, p1/M, z19.s\n"
      "4:"  // Prepare accumulators: Test for last block
      "mov x20, x9\n"
      "mov x21, x10\n"
      "incw x20, ALL, MUL #4\n"
      "incw x21\n"
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
      "add x20, x20, #0x3\n"
      "lsr x20, x20, #0x2\n"
      "lsr x21, x20, #0x2\n"
      "madd x23, x9, x22, x23\n"  // bptr = B + n * kstride_bytes
      "and x20, x20, #0x3\n"
      "cbz x21, 8f\n"
      "subs x21, x21, #0x1\n"
      "ld1b { z5.b }, p1/Z, [x26]\n"
      ".inst 0xa14086e0  // ld1b { z0.b, z4.b, z8.b, z12.b }, pn9.b/Z, [x23]\n"
      "ld1b { z31.b }, p1/Z, [x26, #1, MUL VL]\n"
      ".inst 0xa14186f2  // ld1b { z18.b, z22.b, z26.b, z30.b }, pn9.b/Z, [x23, #0x4, MUL VL]\n"
      "ld1b { z1.b }, p1/Z, [x26, #2, MUL VL]\n"
      ".inst 0xa14286f0  // ld1b { z16.b, z20.b, z24.b, z28.b }, pn9.b/Z, [x23, #0x8, MUL VL]\n"
      "ld1b { z6.b }, p1/Z, [x26, #3, MUL VL]\n"
      "addvl x26, x26, #4\n"
      ".inst 0xa14386e3  // ld1b { z3.b, z7.b, z11.b, z15.b }, pn9.b/Z, [x23, #0xc, MUL VL]\n"
      "addvl x23, x23, #16\n"
      "ble 7f\n"
      "6:"  // K loop
      ".inst 0xa08024a0  // smopa za0.s, p1/M, p1/M, z5.b, z0.b\n"
      "subs x21, x21, #0x1\n"
      ".inst 0xa08424a1  // smopa za1.s, p1/M, p1/M, z5.b, z4.b\n"
      ".inst 0xa08824a2  // smopa za2.s, p1/M, p1/M, z5.b, z8.b\n"
      ".inst 0xa08c24a3  // smopa za3.s, p1/M, p1/M, z5.b, z12.b\n"
      "ld1b { z5.b }, p1/Z, [x26]\n"
      ".inst 0xa09227e0  // smopa za0.s, p1/M, p1/M, z31.b, z18.b\n"
      ".inst 0xa14086e0  // ld1b { z0.b, z4.b, z8.b, z12.b }, pn9.b/Z, [x23]\n"
      ".inst 0xa09627e1  // smopa za1.s, p1/M, p1/M, z31.b, z22.b\n"
      ".inst 0xa09a27e2  // smopa za2.s, p1/M, p1/M, z31.b, z26.b\n"
      ".inst 0xa09e27e3  // smopa za3.s, p1/M, p1/M, z31.b, z30.b\n"
      "ld1b { z31.b }, p1/Z, [x26, #1, MUL VL]\n"
      ".inst 0xa0902420  // smopa za0.s, p1/M, p1/M, z1.b, z16.b\n"
      ".inst 0xa14186f2  // ld1b { z18.b, z22.b, z26.b, z30.b }, pn9.b/Z, [x23, #0x4, MUL VL]\n"
      ".inst 0xa0942421  // smopa za1.s, p1/M, p1/M, z1.b, z20.b\n"
      ".inst 0xa0982422  // smopa za2.s, p1/M, p1/M, z1.b, z24.b\n"
      ".inst 0xa09c2423  // smopa za3.s, p1/M, p1/M, z1.b, z28.b\n"
      "ld1b { z1.b }, p1/Z, [x26, #2, MUL VL]\n"
      ".inst 0xa14286f0  // ld1b { z16.b, z20.b, z24.b, z28.b }, pn9.b/Z, [x23, #0x8, MUL VL]\n"
      ".inst 0xa08324c0  // smopa za0.s, p1/M, p1/M, z6.b, z3.b\n"
      ".inst 0xa08724c1  // smopa za1.s, p1/M, p1/M, z6.b, z7.b\n"
      ".inst 0xa08b24c2  // smopa za2.s, p1/M, p1/M, z6.b, z11.b\n"
      ".inst 0xa08f24c3  // smopa za3.s, p1/M, p1/M, z6.b, z15.b\n"
      "ld1b { z6.b }, p1/Z, [x26, #3, MUL VL]\n"
      "addvl x26, x26, #4\n"
      ".inst 0xa14386e3  // ld1b { z3.b, z7.b, z11.b, z15.b }, pn9.b/Z, [x23, #0xc, MUL VL]\n"
      "addvl x23, x23, #16\n"
      "bgt 6b\n"
      "7:"  // K loop tail
      ".inst 0xa08024a0  // smopa za0.s, p1/M, p1/M, z5.b, z0.b\n"
      ".inst 0xa08424a1  // smopa za1.s, p1/M, p1/M, z5.b, z4.b\n"
      ".inst 0xa08824a2  // smopa za2.s, p1/M, p1/M, z5.b, z8.b\n"
      ".inst 0xa08c24a3  // smopa za3.s, p1/M, p1/M, z5.b, z12.b\n"
      ".inst 0xa09227e0  // smopa za0.s, p1/M, p1/M, z31.b, z18.b\n"
      ".inst 0xa09627e1  // smopa za1.s, p1/M, p1/M, z31.b, z22.b\n"
      ".inst 0xa09a27e2  // smopa za2.s, p1/M, p1/M, z31.b, z26.b\n"
      ".inst 0xa09e27e3  // smopa za3.s, p1/M, p1/M, z31.b, z30.b\n"
      ".inst 0xa0902420  // smopa za0.s, p1/M, p1/M, z1.b, z16.b\n"
      ".inst 0xa0942421  // smopa za1.s, p1/M, p1/M, z1.b, z20.b\n"
      ".inst 0xa0982422  // smopa za2.s, p1/M, p1/M, z1.b, z24.b\n"
      ".inst 0xa09c2423  // smopa za3.s, p1/M, p1/M, z1.b, z28.b\n"
      ".inst 0xa08324c0  // smopa za0.s, p1/M, p1/M, z6.b, z3.b\n"
      ".inst 0xa08724c1  // smopa za1.s, p1/M, p1/M, z6.b, z7.b\n"
      ".inst 0xa08b24c2  // smopa za2.s, p1/M, p1/M, z6.b, z11.b\n"
      ".inst 0xa08f24c3  // smopa za3.s, p1/M, p1/M, z6.b, z15.b\n"
      "8:"  // K oddments
      "cbz x20, 10f\n"
      "9:"  // K oddments: Loop
      "ld1b { z16.b }, p1/Z, [x26]\n"
      "subs x20, x20, #0x1\n"
      "addvl x26, x26, #1\n"
      ".inst 0xa04086e4  // ld1b { z4.b-z7.b }, pn9.b/Z, [x23]\n"
      "addvl x23, x23, #4\n"
      ".inst 0xa0842600  // smopa za0.s, p1/M, p1/M, z16.b, z4.b\n"
      ".inst 0xa0852601  // smopa za1.s, p1/M, p1/M, z16.b, z5.b\n"
      ".inst 0xa0862602  // smopa za2.s, p1/M, p1/M, z16.b, z6.b\n"
      ".inst 0xa0872603  // smopa za3.s, p1/M, p1/M, z16.b, z7.b\n"
      "bgt 9b\n"
      "10:"  // K oddments: End
      "ld1w { z15.s }, p1/Z, [x26]\n"
      "addvl x26, x26, #1\n"
      ".inst 0xc09125e0  // addva za0.s, p1/M, p1/M, z15.s\n"
      ".inst 0xc09125e1  // addva za1.s, p1/M, p1/M, z15.s\n"
      ".inst 0xc09125e2  // addva za2.s, p1/M, p1/M, z15.s\n"
      ".inst 0xc09125e3  // addva za3.s, p1/M, p1/M, z15.s\n"
      "tbz x15, #1, 14f\n"
      "tbz x15, #0, 12f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "11:"  // Store to partial result buffer: Store and refill: Loop
      ".inst 0xa040c5c4  // ld1w { z4.s-z7.s }, pn9.b/Z, [x14]\n"
      ".inst 0xc086041c  // mova { z28.s-z31.s }, za0h.s[x12]\n"
      ".inst 0xc0860428  // mova { z8.s-z11.s }, za1h.s[x12]\n"
      ".inst 0xa041c5d0  // ld1w { z16.s-z19.s }, pn9.b/Z, [x14, #0x4, MUL VL]\n"
      ".inst 0xc0860454  // mova { z20.s-z23.s }, za2h.s[x12]\n"
      ".inst 0xc0860478  // mova { z24.s-z27.s }, za3h.s[x12]\n"
      ".inst 0xa042c5c0  // ld1w { z0.s-z3.s }, pn9.b/Z, [x14, #0x8, MUL VL]\n"
      ".inst 0xa043c5cc  // ld1w { z12.s-z15.s }, pn9.b/Z, [x14, #0xc, MUL VL]\n"
      ".inst 0xc0840480  // mova za0h.s[x12], { z4.s-z7.s }\n"
      "addvl x14, x14, #16\n"
      ".inst 0xc0840601  // mova za1h.s[x12], { z16.s-z19.s }\n"
      ".inst 0xa060c5bc  // st1w { z28.s-z31.s }, pn9.b, [x13]\n"
      ".inst 0xc0840402  // mova za2h.s[x12], { z0.s-z3.s }\n"
      ".inst 0xa061c5a8  // st1w { z8.s-z11.s }, pn9.b, [x13, #0x4, MUL VL]\n"
      ".inst 0xc0840583  // mova za3h.s[x12], { z12.s-z15.s }\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa062c5b4  // st1w { z20.s-z23.s }, pn9.b, [x13, #0x8, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa063c5b8  // st1w { z24.s-z27.s }, pn9.b, [x13, #0xc, MUL VL]\n"
      "addvl x13, x13, #16\n"
      "blt 11b\n"
      "b 21f\n"
      "12:"  // Store to partial result buffer: Store only
      "mov x12, #0x0\n"
      "cntw x20\n"
      "13:"  // Store to partial result buffer: Store only: Loop
      ".inst 0xc0860408  // mova { z8.s-z11.s }, za0h.s[x12]\n"
      ".inst 0xc0860424  // mova { z4.s-z7.s }, za1h.s[x12]\n"
      ".inst 0xc086044c  // mova { z12.s-z15.s }, za2h.s[x12]\n"
      ".inst 0xc0860470  // mova { z16.s-z19.s }, za3h.s[x12]\n"
      ".inst 0xa060c5a8  // st1w { z8.s-z11.s }, pn9.b, [x13]\n"
      "add x12, x12, #0x4\n"
      ".inst 0xa061c5a4  // st1w { z4.s-z7.s }, pn9.b, [x13, #0x4, MUL VL]\n"
      "cmp x12, x20\n"
      ".inst 0xa062c5ac  // st1w { z12.s-z15.s }, pn9.b, [x13, #0x8, MUL VL]\n"
      ".inst 0xa063c5b0  // st1w { z16.s-z19.s }, pn9.b, [x13, #0xc, MUL VL]\n"
      "addvl x13, x13, #16\n"
      "blt 13b\n"
      "b 21f\n"
      "14:"  // Store to output array
      "ldr x25, [%x[args], %[offsetof_C]]\n"
      "sub x24, x11, x10\n"
      "ld1rw { z4.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_mul]]\n"
      "ldr x23, [%x[args], %[offsetof_ldcb]]\n"
      "ld1rw { z5.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_mul]]\n"
      "ld1rw { z6.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_mul]]\n"
      "add x25, x25, x9\n"  // C += n
      "ld1rw { z7.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_mul]]\n"
      "madd x25, x10, x23, x25\n"  // C += m * ldc
      "ld1rw { z0.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_right_shift]]\n"
      "ld1rw { z1.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_right_shift]]\n"
      "ld1rw { z2.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_right_shift]]\n"
      "ld1rw { z3.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_per_layer_right_shift]]\n"
      "ld1rw { z8.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_c_offset]]\n"
      "ld1rw { z21.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_minval]]\n"
      "ld1rw { z20.s }, p1/Z, [%x[rq], %[offsetof_Requantize32_maxval]]\n"
      "tbz x15, #2, 15f\n"
      "ldr w22, [%x[args], %[offsetof_n_0]]\n"
      "ldr x21, [%x[rq], %[offsetof_Requantize32_per_channel_muls]]\n"
      "ldr x20, [%x[rq], %[offsetof_Requantize32_per_channel_right_shifts]]\n"
      "add x22, x22, x9\n"
      "add x21, x21, x22, LSL #2\n"
      "add x20, x20, x22, LSL #2\n"
      ".inst 0xa040c2a4  // ld1w { z4.s-z7.s }, p8/Z, [x21]\n"
      ".inst 0xa040c280  // ld1w { z0.s-z3.s }, p8/Z, [x20]\n"
      "15:"  // Store to output array: Load per-channel parameters: End
      "cntw x20\n"
      "whilelt p0.b, x9, x28\n"
      "cmp x24, x20\n"
      "mov x12, #0x0\n"
      "csel x20, x24, x20, LT\n"
      "lsr x21, x20, #0x1\n"
      "and x20, x20, #0x1\n"
      "cbz x21, 17f\n"
      "16:"  // Store to output array: Accumulator row 0 loop
      ".inst 0xc0860010  // mova { z16.s-z17.s }, za0h.s[x12, 0:1]\n"
      ".inst 0xc086005e  // mova { z30.s-z31.s }, za1h.s[x12, 0:1]\n"
      ".inst 0xc086009a  // mova { z26.s-z27.s }, za2h.s[x12, 0:1]\n"
      ".inst 0xc08600cc  // mova { z12.s-z13.s }, za3h.s[x12, 0:1]\n"
      ".inst 0xc1a4a410  // sqdmulh { z16.s-z17.s }, { z16.s-z17.s }, z4.s\n"
      ".inst 0xc1a5a41e  // sqdmulh { z30.s-z31.s }, { z30.s-z31.s }, z5.s\n"
      "add x12, x12, #0x2\n"
      ".inst 0xc1a6a41a  // sqdmulh { z26.s-z27.s }, { z26.s-z27.s }, z6.s\n"
      "cmp x12, x21, LSL #1\n"
      ".inst 0xc1a7a40c  // sqdmulh { z12.s-z13.s }, { z12.s-z13.s }, z7.s\n"
      ".inst 0xc1a0a230  // srshl { z16.s-z17.s }, { z16.s-z17.s }, z0.s\n"
      ".inst 0xc1a1a23e  // srshl { z30.s-z31.s }, { z30.s-z31.s }, z1.s\n"
      ".inst 0xc1a2a23a  // srshl { z26.s-z27.s }, { z26.s-z27.s }, z2.s\n"
      ".inst 0xc1a3a22c  // srshl { z12.s-z13.s }, { z12.s-z13.s }, z3.s\n"
      ".inst 0xc1a8a310  // add { z16.s-z17.s }, { z16.s-z17.s }, z8.s\n"
      ".inst 0xc1a8a31e  // add { z30.s-z31.s }, { z30.s-z31.s }, z8.s\n"
      ".inst 0xc1a8a31a  // add { z26.s-z27.s }, { z26.s-z27.s }, z8.s\n"
      ".inst 0xc1a8a30c  // add { z12.s-z13.s }, { z12.s-z13.s }, z8.s\n"
      ".inst 0xc1b4c6b0  // sclamp { z16.s-z17.s }, z21.s, z20.s\n"
      ".inst 0xc1b4c6be  // sclamp { z30.s-z31.s }, z21.s, z20.s\n"
      ".inst 0xc1b4c6ba  // sclamp { z26.s-z27.s }, z21.s, z20.s\n"
      ".inst 0xc1b4c6ac  // sclamp { z12.s-z13.s }, z21.s, z20.s\n"
      "uzp1 z19.b, z16.b, z30.b\n"
      "uzp1 z18.b, z17.b, z31.b\n"
      "uzp1 z17.b, z26.b, z12.b\n"
      "uzp1 z16.b, z27.b, z13.b\n"
      "uzp1 z17.b, z19.b, z17.b\n"
      "uzp1 z16.b, z18.b, z16.b\n"
      "st1b { z17.b }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "st1b { z16.b }, p0, [x25]\n"
      "add x25, x25, x23\n"
      "blt 16b\n"
      "17:"  // Store to output array: Accumulator row 0 oddments
      "cbz x20, 18f\n"
      ".inst 0xc086000a  // mova { z10.s-z11.s }, za0h.s[x12, 0:1]\n"
      ".inst 0xc086005a  // mova { z26.s-z27.s }, za1h.s[x12, 0:1]\n"
      ".inst 0xc086008e  // mova { z14.s-z15.s }, za2h.s[x12, 0:1]\n"
      ".inst 0xc08600d6  // mova { z22.s-z23.s }, za3h.s[x12, 0:1]\n"
      ".inst 0xc1a4a40a  // sqdmulh { z10.s-z11.s }, { z10.s-z11.s }, z4.s\n"
      ".inst 0xc1a5a41a  // sqdmulh { z26.s-z27.s }, { z26.s-z27.s }, z5.s\n"
      ".inst 0xc1a6a40e  // sqdmulh { z14.s-z15.s }, { z14.s-z15.s }, z6.s\n"
      ".inst 0xc1a7a416  // sqdmulh { z22.s-z23.s }, { z22.s-z23.s }, z7.s\n"
      ".inst 0xc1a0a22a  // srshl { z10.s-z11.s }, { z10.s-z11.s }, z0.s\n"
      ".inst 0xc1a1a23a  // srshl { z26.s-z27.s }, { z26.s-z27.s }, z1.s\n"
      ".inst 0xc1a2a22e  // srshl { z14.s-z15.s }, { z14.s-z15.s }, z2.s\n"
      ".inst 0xc1a3a236  // srshl { z22.s-z23.s }, { z22.s-z23.s }, z3.s\n"
      ".inst 0xc1a8a30a  // add { z10.s-z11.s }, { z10.s-z11.s }, z8.s\n"
      ".inst 0xc1a8a31a  // add { z26.s-z27.s }, { z26.s-z27.s }, z8.s\n"
      ".inst 0xc1a8a30e  // add { z14.s-z15.s }, { z14.s-z15.s }, z8.s\n"
      ".inst 0xc1a8a316  // add { z22.s-z23.s }, { z22.s-z23.s }, z8.s\n"
      ".inst 0xc1b4c6aa  // sclamp { z10.s-z11.s }, z21.s, z20.s\n"
      ".inst 0xc1b4c6ba  // sclamp { z26.s-z27.s }, z21.s, z20.s\n"
      ".inst 0xc1b4c6ae  // sclamp { z14.s-z15.s }, z21.s, z20.s\n"
      ".inst 0xc1b4c6b6  // sclamp { z22.s-z23.s }, z21.s, z20.s\n"
      "uzp1 z17.b, z10.b, z26.b\n"
      "uzp1 z16.b, z14.b, z22.b\n"
      "uzp1 z16.b, z17.b, z16.b\n"
      "st1b { z16.b }, p0, [x25]\n"
      "18:"  // Store to output array: Accumulator row 0 oddments: End
      "19:"  // Store to output array: End
      "tbz x15, #0, 21f\n"
      "mov x12, #0x0\n"
      "cntw x20\n"
      "20:"  // Store to output array: Refill accumulators: Loop
      ".inst 0xa040c5d8  // ld1w { z24.s-z27.s }, pn9.b/Z, [x14]\n"
      ".inst 0xa041c5cc  // ld1w { z12.s-z15.s }, pn9.b/Z, [x14, #0x4, MUL VL]\n"
      ".inst 0xa042c5c4  // ld1w { z4.s-z7.s }, pn9.b/Z, [x14, #0x8, MUL VL]\n"
      ".inst 0xa043c5c0  // ld1w { z0.s-z3.s }, pn9.b/Z, [x14, #0xc, MUL VL]\n"
      ".inst 0xc0840700  // mova za0h.s[x12], { z24.s-z27.s }\n"
      "addvl x14, x14, #16\n"
      ".inst 0xc0840581  // mova za1h.s[x12], { z12.s-z15.s }\n"
      ".inst 0xc0840482  // mova za2h.s[x12], { z4.s-z7.s }\n"
      ".inst 0xc0840403  // mova za3h.s[x12], { z0.s-z3.s }\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x20\n"
      "blt 20b\n"
      "21:"  // End block
      "incw x9, ALL, MUL #4\n"
      "cmp x9, x28\n"
      "blt 3b\n"
      "incw x10\n"
      "mov x9, #0x0\n"
      "cmp x10, x11\n"
      "mov x27, x26\n"
      "blt 3b\n"
      ".inst 0xd503467f  // SMSTOP\n"
      :
      : [args] "r" (&args), [offsetof_A] "I" (offsetof(KernelArgs, A)), [offsetof_B] "I" (offsetof(KernelArgs, B)), [offsetof_C] "I" (offsetof(KernelArgs, C)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_M] "I" (offsetof(KernelArgs, M)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_Requantize32_c_offset] "I" (offsetof(Requantize32, c_offset)), [offsetof_Requantize32_maxval] "I" (offsetof(Requantize32, maxval)), [offsetof_Requantize32_minval] "I" (offsetof(Requantize32, minval)), [offsetof_Requantize32_per_channel_muls] "I" (offsetof(Requantize32, per_channel_muls)), [offsetof_Requantize32_per_channel_right_shifts] "I" (offsetof(Requantize32, per_channel_right_shifts)), [offsetof_Requantize32_per_layer_mul] "I" (offsetof(Requantize32, per_layer_mul)), [offsetof_Requantize32_per_layer_right_shift] "I" (offsetof(Requantize32, per_layer_right_shift)), [offsetof_accumulator_buffer] "I" (offsetof(KernelArgs, accumulator_buffer)), [offsetof_bias] "I" (offsetof(KernelArgs, bias)), [offsetof_flags] "I" (offsetof(KernelArgs, flags)), [offsetof_kstride_bytes] "I" (offsetof(KernelArgs, kstride_bytes)), [offsetof_ldcb] "I" (offsetof(KernelArgs, ldcb)), [offsetof_n_0] "I" (offsetof(KernelArgs, n_0)), [rq] "r" (&rq)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

}  // namespace arm_gemm

#endif  // ARM_COMPUTE_ENABLE_SME2
