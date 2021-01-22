/*
 * Copyright (c) 2021 Arm Limited.
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

#include <cstddef>
#include <cstdint>

#if defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

namespace arm_conv {
namespace depthwise {

void a64_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst_direct_impl(
  const unsigned int n_tile_rows,
  const unsigned int n_tile_cols,
  const __fp16 *inptr,
  int64_t ld_input_row,
  int64_t ld_input_col,
  __fp16 *outptr,
  int64_t ld_output_row,
  int64_t ld_output_col,
  const void *params,
  unsigned int n_channels,
  const __fp16 activation_min,
  const __fp16 activation_max
)
{
  struct Args
  {
    const uint64_t n_tile_rows, n_tile_cols;
    const __fp16 *inptr;
    const uint64_t ld_input_row;
    const uint64_t ld_input_col;
    __fp16 *outptr;
    const uint64_t ld_output_row;
    const uint64_t ld_output_col;
    const void *params;
    const __fp16 min, max;

    uint64_t tile_i = 0, tile_j = 0;

    Args(
      const unsigned int n_tile_rows,
      const unsigned int n_tile_cols,
      const __fp16 *inptr,
      int64_t ld_input_row,
      int64_t ld_input_col,
      __fp16 *outptr,
      int64_t ld_output_row,
      int64_t ld_output_col,
      const void *params,
      const float activation_min,
      const float activation_max
    ) : n_tile_rows(n_tile_rows), n_tile_cols(n_tile_cols), inptr(inptr),
        ld_input_row(ld_input_row), ld_input_col(ld_input_col), outptr(outptr),
        ld_output_row(ld_output_row), ld_output_col(ld_output_col),
        params(params), min(activation_min), max(activation_max)
    {
    }
  };

  Args params_struct(
    n_tile_rows, n_tile_cols,
    inptr, ld_input_row, ld_input_col,
    outptr, ld_output_row, ld_output_col,
    params, activation_min, activation_max
  );

  __asm__ __volatile__(
    "mov x6, #0x0\n"
    "mov x27, #0x0\n"
    "1:"  // Tile loop
    "str x6, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "mov x26, #0x4\n"
    "str x27, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "mov x25, #0x2\n"
    "ldr x7, [%x[params_struct], %[offsetof_args_params]]\n"
    "add x24, %x[params_struct], %[offsetof_args_min]\n"
    "ldr x23, [%x[params_struct], %[offsetof_args_ld_input_row]]\n"
    "add x21, %x[params_struct], %[offsetof_args_max]\n"
    "ldr x8, [%x[params_struct], %[offsetof_args_ld_input_col]]\n"
    "mov x22, #0x0\n"
    "ldr x17, [%x[params_struct], %[offsetof_args_inptr]]\n"
    "mul x19, x6, x23\n" // offset = tile_i * ld_input_row
    "ldr x20, [%x[params_struct], %[offsetof_args_ld_output_row]]\n"
    "madd x19, x27, x8, x19\n" // offset += tile_j * ld_input_col
    "ldr x16, [%x[params_struct], %[offsetof_args_ld_output_col]]\n"
    "mul x19, x19, x26\n" // offset *= kernel_stride * output_size
    "ldr x15, [%x[params_struct], %[offsetof_args_outptr]]\n"
    "add x17, x17, x19, LSL #1\n" // inptr[0] += offset * sizeof(__fp16)
    "ld1r { v19.8h }, [x24]\n"
    "add x14, x17, x23, LSL #1\n"
    "ld1r { v18.8h }, [x21]\n"
    "add x13, x14, x23, LSL #1\n"
    "lsl x8, x8, #0x1\n"
    "add x12, x13, x23, LSL #1\n"
    "add x11, x12, x23, LSL #1\n"
    "add x10, x8, x8\n"
    "add x9, x10, x8\n"
    "add x28, x9, x8\n"
    "mul x19, x6, x20\n" // offset = tile_i * ld_output_row
    "madd x19, x27, x16, x19\n" // offset += tile_j * ld_output_col
    "mul x19, x19, x25\n" // offset *= output_tile_size
    "add x15, x15, x19, LSL #1\n" // outptrs[0] += offset * sizeof(__fp16)
    "add x27, x15, x20, LSL #1\n"
    "lsl x16, x16, #0x1\n"
    "mov x21, #0x10\n" // cntb _, ALL, #1
    "sub x20, XZR, x21\n"
    "lsr x19, %x[n_channels], #0x3\n"
    "cbz x19, 4f\n"
    "ldr q17, [x7, #0x0]\n"
    "ldr q0, [x7, #0x10]\n"
    "cmp x21, x19, LSL #4\n"
    "ldr q1, [x7, #0x20]\n"
    "ldr q2, [x7, #0x30]\n"
    "ldr q3, [x7, #0x40]\n"
    "ldr q4, [x7, #0x50]\n"
    "ldr q5, [x7, #0x60]\n"
    "ldr q6, [x7, #0x70]\n"
    "ldr q7, [x7, #0x80]\n"
    "ldr q8, [x7, #0x90]\n"
    "add x7, x7, #0xa0\n"
    "ldr q9, [x13, x10]\n"
    "ld1 { v10.8h }, [x17]\n"
    "ldr q11, [x17, x8]\n"
    "ldr q12, [x17, x9]\n"
    "ldr q13, [x17, x28]\n"
    "ld1 { v14.8h }, [x14]\n"
    "ldr q15, [x14, x8]\n"
    "ldr q16, [x17, x10]\n"
    "bge 3f\n"
    "2:"  // Tile loop: Channel loop
    "mov v31.16b, v17.16b\n fmla v31.8h, v8.8h, v9.8h\n"
    "add x20, x20, #0x10\n"
    "mov v30.16b, v17.16b\n fmla v30.8h, v6.8h, v9.8h\n"
    "add x22, x22, #0x10\n"
    "mov v29.16b, v17.16b\n fmla v29.8h, v2.8h, v9.8h\n"
    "add x17, x17, #0x10\n"
    "mov v28.16b, v17.16b\n fmla v28.8h, v0.8h, v9.8h\n"
    "ldr q17, [x7, #0x0]\n"
    "add x21, x21, #0x10\n"
    "fmla v31.8h, v0.8h, v10.8h\n"
    "ld1 { v10.8h }, [x17]\n"
    "cmp x21, x19, LSL #4\n"
    "fmla v30.8h, v1.8h, v12.8h\n"
    "ldr q12, [x14, x28]\n"
    "fmla v31.8h, v1.8h, v11.8h\n"
    "ldr q11, [x14, x9]\n"
    "fmla v30.8h, v2.8h, v13.8h\n"
    "ldr q13, [x14, x10]\n"
    "add x14, x14, #0x10\n"
    "fmla v31.8h, v3.8h, v14.8h\n"
    "ld1 { v14.8h }, [x12]\n"
    "fmla v30.8h, v0.8h, v16.8h\n"
    "fmla v31.8h, v4.8h, v15.8h\n"
    "ld1 { v15.8h }, [x13]\n"
    "fmla v29.8h, v3.8h, v14.8h\n"
    "ldr q14, [x12, x28]\n"
    "fmla v30.8h, v4.8h, v11.8h\n"
    "ldr q11, [x12, x8]\n"
    "fmla v31.8h, v2.8h, v16.8h\n"
    "ldr q16, [x13, x8]\n"
    "fmla v29.8h, v0.8h, v15.8h\n"
    "ldr q0, [x7, #0x10]\n"
    "fmla v30.8h, v5.8h, v12.8h\n"
    "ldr q12, [x13, x9]\n"
    "fmla v31.8h, v5.8h, v13.8h\n"
    "fmla v29.8h, v4.8h, v11.8h\n"
    "ldr q11, [x13, x28]\n"
    "add x13, x13, #0x10\n"
    "fmla v30.8h, v3.8h, v13.8h\n"
    "ldr q13, [x12, x9]\n"
    "ldr q9, [x13, x10]\n"
    "fmla v31.8h, v6.8h, v15.8h\n"
    "ld1 { v15.8h }, [x11]\n"
    "fmla v29.8h, v1.8h, v16.8h\n"
    "fmla v28.8h, v4.8h, v13.8h\n"
    "ldr q13, [x11, x8]\n"
    "fmla v30.8h, v7.8h, v12.8h\n"
    "ldr q4, [x7, #0x50]\n"
    "fmla v31.8h, v7.8h, v16.8h\n"
    "ldr q16, [x12, x10]\n"
    "add x12, x12, #0x10\n"
    "fmla v29.8h, v6.8h, v15.8h\n"
    "ldr q15, [x11, x10]\n"
    "fmla v28.8h, v1.8h, v12.8h\n"
    "ldr q12, [x17, x9]\n"
    "fmla v30.8h, v8.8h, v11.8h\n"
    "ldr q1, [x7, #0x20]\n"
    "fmax v31.8h, v31.8h, v19.8h\n"
    "fmla v29.8h, v7.8h, v13.8h\n"
    "ldr q13, [x17, x28]\n"
    "fmla v28.8h, v5.8h, v14.8h\n"
    "ldr q14, [x11, x9]\n"
    "fmax v30.8h, v30.8h, v19.8h\n"
    "fmin v31.8h, v31.8h, v18.8h\n"
    "st1 { v31.8h }, [x15]\n"
    "fmla v28.8h, v2.8h, v11.8h\n"
    "fmla v29.8h, v5.8h, v16.8h\n"
    "ldr q11, [x11, x28]\n"
    "add x11, x11, #0x10\n"
    "fmin v30.8h, v30.8h, v18.8h\n"
    "ldr q2, [x7, #0x30]\n"
    "ldr q5, [x7, #0x60]\n"
    "fmla v28.8h, v3.8h, v16.8h\n"
    "ldr q16, [x17, x10]\n"
    "fmla v29.8h, v8.8h, v15.8h\n"
    "str q30, [x15, x16]\n"
    "add x15, x15, #0x10\n"
    "fmla v28.8h, v7.8h, v14.8h\n"
    "ld1 { v14.8h }, [x14]\n"
    "fmax v29.8h, v29.8h, v19.8h\n"
    "ldr q3, [x7, #0x40]\n"
    "ldr q7, [x7, #0x80]\n"
    "fmin v29.8h, v29.8h, v18.8h\n"
    "st1 { v29.8h }, [x27]\n"
    "fmla v28.8h, v6.8h, v15.8h\n"
    "ldr q15, [x14, x8]\n"
    "fmla v28.8h, v8.8h, v11.8h\n"
    "ldr q11, [x17, x8]\n"
    "ldr q6, [x7, #0x70]\n"
    "fmax v28.8h, v28.8h, v19.8h\n"
    "ldr q8, [x7, #0x90]\n"
    "add x7, x7, #0xa0\n"
    "fmin v28.8h, v28.8h, v18.8h\n"
    "str q28, [x27, x16]\n"
    "add x27, x27, #0x10\n"
    "blt 2b\n"
    "3:"  // Tile loop: Channel tail
    "mov v31.16b, v17.16b\n fmla v31.8h, v8.8h, v9.8h\n"
    "add x17, x17, #0x10\n"
    "mov v30.16b, v17.16b\n fmla v30.8h, v6.8h, v9.8h\n"
    "mov v29.16b, v17.16b\n fmla v29.8h, v2.8h, v9.8h\n"
    "mov v28.16b, v17.16b\n fmla v28.8h, v0.8h, v9.8h\n"
    "fmla v31.8h, v0.8h, v10.8h\n"
    "fmla v30.8h, v1.8h, v12.8h\n"
    "ldr q12, [x14, x28]\n"
    "fmla v31.8h, v1.8h, v11.8h\n"
    "ldr q11, [x14, x9]\n"
    "fmla v30.8h, v2.8h, v13.8h\n"
    "ldr q13, [x14, x10]\n"
    "add x14, x14, #0x10\n"
    "fmla v31.8h, v3.8h, v14.8h\n"
    "ld1 { v14.8h }, [x12]\n"
    "fmla v30.8h, v0.8h, v16.8h\n"
    "fmla v31.8h, v4.8h, v15.8h\n"
    "ld1 { v15.8h }, [x13]\n"
    "fmla v30.8h, v4.8h, v11.8h\n"
    "ldr q11, [x12, x8]\n"
    "fmla v29.8h, v3.8h, v14.8h\n"
    "ldr q14, [x12, x28]\n"
    "fmla v31.8h, v2.8h, v16.8h\n"
    "ldr q16, [x13, x8]\n"
    "fmla v30.8h, v5.8h, v12.8h\n"
    "ldr q12, [x13, x9]\n"
    "fmla v29.8h, v0.8h, v15.8h\n"
    "fmla v31.8h, v5.8h, v13.8h\n"
    "fmla v30.8h, v3.8h, v13.8h\n"
    "ldr q13, [x12, x9]\n"
    "fmla v29.8h, v4.8h, v11.8h\n"
    "ldr q11, [x13, x28]\n"
    "add x13, x13, #0x10\n"
    "fmla v31.8h, v6.8h, v15.8h\n"
    "ld1 { v15.8h }, [x11]\n"
    "fmla v30.8h, v7.8h, v12.8h\n"
    "fmla v29.8h, v1.8h, v16.8h\n"
    "fmla v28.8h, v4.8h, v13.8h\n"
    "ldr q13, [x11, x8]\n"
    "fmla v31.8h, v7.8h, v16.8h\n"
    "ldr q16, [x12, x10]\n"
    "add x12, x12, #0x10\n"
    "fmla v29.8h, v6.8h, v15.8h\n"
    "ldr q15, [x11, x10]\n"
    "fmla v30.8h, v8.8h, v11.8h\n"
    "fmla v28.8h, v1.8h, v12.8h\n"
    "fmax v31.8h, v31.8h, v19.8h\n"
    "fmla v29.8h, v7.8h, v13.8h\n"
    "fmax v30.8h, v30.8h, v19.8h\n"
    "fmla v28.8h, v5.8h, v14.8h\n"
    "ldr q14, [x11, x9]\n"
    "fmin v31.8h, v31.8h, v18.8h\n"
    "st1 { v31.8h }, [x15]\n"
    "fmla v28.8h, v2.8h, v11.8h\n"
    "fmla v29.8h, v5.8h, v16.8h\n"
    "ldr q11, [x11, x28]\n"
    "add x11, x11, #0x10\n"
    "fmin v30.8h, v30.8h, v18.8h\n"
    "str q30, [x15, x16]\n"
    "fmla v28.8h, v3.8h, v16.8h\n"
    "add x15, x15, #0x10\n"
    "fmla v29.8h, v8.8h, v15.8h\n"
    "fmla v28.8h, v7.8h, v14.8h\n"
    "fmax v29.8h, v29.8h, v19.8h\n"
    "fmla v28.8h, v6.8h, v15.8h\n"
    "fmin v29.8h, v29.8h, v18.8h\n"
    "st1 { v29.8h }, [x27]\n"
    "fmla v28.8h, v8.8h, v11.8h\n"
    "fmax v28.8h, v28.8h, v19.8h\n"
    "fmin v28.8h, v28.8h, v18.8h\n"
    "str q28, [x27, x16]\n"
    "add x27, x27, #0x10\n"
    "4:"  // Tile loop: Oddments
    "tst %x[n_channels], #0x1\n"
    "beq 43f\n"
    "ldr q17, [x7, #0x0]\n"
    "ldr q0, [x7, #0x10]\n"
    "add x26, x13, x10\n"
    "ldr q1, [x7, #0x20]\n"
    "add x25, x17, XZR\n"
    "ldr q2, [x7, #0x30]\n"
    "add x24, x17, x8\n"
    "ldr q3, [x7, #0x40]\n"
    "add x23, x17, x9\n"
    "ldr q4, [x7, #0x50]\n"
    "add x22, x17, x28\n"
    "ldr q5, [x7, #0x60]\n"
    "add x21, x14, XZR\n"
    "ldr q6, [x7, #0x70]\n"
    "add x20, x14, x8\n"
    "ldr q7, [x7, #0x80]\n"
    "add x19, x17, x10\n"
    "ldr q8, [x7, #0x90]\n"
    "tbz %x[n_channels], #1, 5f\n"
    "ldr s9, [x26], #0x4\n"
    "ldr s10, [x25], #0x4\n"
    "ldr s11, [x24], #0x4\n"
    "ldr s12, [x23], #0x4\n"
    "ldr s13, [x22], #0x4\n"
    "ldr s14, [x21], #0x4\n"
    "ldr s15, [x20], #0x4\n"
    "ldr s16, [x19], #0x4\n"
    "tbz %x[n_channels], #0, 6f\n"
    "ld1 { v9.h }[2], [x26]\n"
    "ld1 { v10.h }[2], [x25]\n"
    "ld1 { v11.h }[2], [x24]\n"
    "ld1 { v12.h }[2], [x23]\n"
    "ld1 { v13.h }[2], [x22]\n"
    "ld1 { v14.h }[2], [x21]\n"
    "ld1 { v15.h }[2], [x20]\n"
    "ld1 { v16.h }[2], [x19]\n"
    "b 6f\n"
    "5:"  // Tile loop: Oddments: Load inputs: (2, 2), (0, 0), (0, 1), (0, 3), (0, 4), (1, 0), (1, 1), (0, 2): Bit 1: Unset
    "ldr h9, [x26, #0x0]\n"
    "ldr h10, [x25, #0x0]\n"
    "ldr h11, [x24, #0x0]\n"
    "ldr h12, [x23, #0x0]\n"
    "ldr h13, [x22, #0x0]\n"
    "ldr h14, [x21, #0x0]\n"
    "ldr h15, [x20, #0x0]\n"
    "ldr h16, [x19, #0x0]\n"
    "6:"  // Tile loop: Oddments: Load inputs: (2, 2), (0, 0), (0, 1), (0, 3), (0, 4), (1, 0), (1, 1), (0, 2): Bit 1: End
    "mov v31.16b, v17.16b\n fmla v31.8h, v8.8h, v9.8h\n"
    "add x19, x14, x9\n"
    "mov v30.16b, v17.16b\n fmla v30.8h, v6.8h, v9.8h\n"
    "mov v29.16b, v17.16b\n fmla v29.8h, v2.8h, v9.8h\n"
    "mov v28.16b, v17.16b\n fmla v28.8h, v0.8h, v9.8h\n"
    "fmla v31.8h, v0.8h, v10.8h\n"
    "fmla v30.8h, v1.8h, v12.8h\n"
    "fmla v31.8h, v1.8h, v11.8h\n"
    "fmla v30.8h, v2.8h, v13.8h\n"
    "fmla v31.8h, v3.8h, v14.8h\n"
    "fmla v30.8h, v0.8h, v16.8h\n"
    "fmla v31.8h, v4.8h, v15.8h\n"
    "fmla v31.8h, v2.8h, v16.8h\n"
    "tbz %x[n_channels], #1, 7f\n"
    "ldr s11, [x19], #0x4\n"
    "tbz %x[n_channels], #0, 8f\n"
    "ld1 { v11.h }[2], [x19]\n"
    "b 8f\n"
    "7:"  // Tile loop: Oddments: Load inputs: (1, 3): Bit 1: Unset
    "ldr h11, [x19, #0x0]\n"
    "8:"  // Tile loop: Oddments: Load inputs: (1, 3): Bit 1: End
    "fmla v30.8h, v4.8h, v11.8h\n"
    "add x19, x14, x28\n"
    "tbz %x[n_channels], #1, 9f\n"
    "ldr s12, [x19], #0x4\n"
    "tbz %x[n_channels], #0, 10f\n"
    "ld1 { v12.h }[2], [x19]\n"
    "b 10f\n"
    "9:"  // Tile loop: Oddments: Load inputs: (1, 4): Bit 1: Unset
    "ldr h12, [x19, #0x0]\n"
    "10:"  // Tile loop: Oddments: Load inputs: (1, 4): Bit 1: End
    "fmla v30.8h, v5.8h, v12.8h\n"
    "add x19, x14, x10\n"
    "tbz %x[n_channels], #1, 11f\n"
    "ldr s13, [x19], #0x4\n"
    "tbz %x[n_channels], #0, 12f\n"
    "ld1 { v13.h }[2], [x19]\n"
    "b 12f\n"
    "11:"  // Tile loop: Oddments: Load inputs: (1, 2): Bit 1: Unset
    "ldr h13, [x19, #0x0]\n"
    "12:"  // Tile loop: Oddments: Load inputs: (1, 2): Bit 1: End
    "fmla v31.8h, v5.8h, v13.8h\n"
    "add x19, x12, XZR\n"
    "fmla v30.8h, v3.8h, v13.8h\n"
    "tbz %x[n_channels], #1, 13f\n"
    "ldr s14, [x19], #0x4\n"
    "tbz %x[n_channels], #0, 14f\n"
    "ld1 { v14.h }[2], [x19]\n"
    "b 14f\n"
    "13:"  // Tile loop: Oddments: Load inputs: (3, 0): Bit 1: Unset
    "ldr h14, [x19, #0x0]\n"
    "14:"  // Tile loop: Oddments: Load inputs: (3, 0): Bit 1: End
    "fmla v29.8h, v3.8h, v14.8h\n"
    "add x19, x13, XZR\n"
    "tbz %x[n_channels], #1, 15f\n"
    "ldr s15, [x19], #0x4\n"
    "tbz %x[n_channels], #0, 16f\n"
    "ld1 { v15.h }[2], [x19]\n"
    "b 16f\n"
    "15:"  // Tile loop: Oddments: Load inputs: (2, 0): Bit 1: Unset
    "ldr h15, [x19, #0x0]\n"
    "16:"  // Tile loop: Oddments: Load inputs: (2, 0): Bit 1: End
    "fmla v31.8h, v6.8h, v15.8h\n"
    "add x19, x12, x8\n"
    "fmla v29.8h, v0.8h, v15.8h\n"
    "tbz %x[n_channels], #1, 17f\n"
    "ldr s11, [x19], #0x4\n"
    "tbz %x[n_channels], #0, 18f\n"
    "ld1 { v11.h }[2], [x19]\n"
    "b 18f\n"
    "17:"  // Tile loop: Oddments: Load inputs: (3, 1): Bit 1: Unset
    "ldr h11, [x19, #0x0]\n"
    "18:"  // Tile loop: Oddments: Load inputs: (3, 1): Bit 1: End
    "fmla v29.8h, v4.8h, v11.8h\n"
    "add x19, x13, x8\n"
    "tbz %x[n_channels], #1, 19f\n"
    "ldr s16, [x19], #0x4\n"
    "tbz %x[n_channels], #0, 20f\n"
    "ld1 { v16.h }[2], [x19]\n"
    "b 20f\n"
    "19:"  // Tile loop: Oddments: Load inputs: (2, 1): Bit 1: Unset
    "ldr h16, [x19, #0x0]\n"
    "20:"  // Tile loop: Oddments: Load inputs: (2, 1): Bit 1: End
    "fmla v31.8h, v7.8h, v16.8h\n"
    "add x19, x12, x9\n"
    "fmla v29.8h, v1.8h, v16.8h\n"
    "tbz %x[n_channels], #1, 21f\n"
    "ldr s13, [x19], #0x4\n"
    "tbz %x[n_channels], #0, 22f\n"
    "ld1 { v13.h }[2], [x19]\n"
    "b 22f\n"
    "21:"  // Tile loop: Oddments: Load inputs: (3, 3): Bit 1: Unset
    "ldr h13, [x19, #0x0]\n"
    "22:"  // Tile loop: Oddments: Load inputs: (3, 3): Bit 1: End
    "fmla v28.8h, v4.8h, v13.8h\n"
    "add x19, x13, x9\n"
    "tbz %x[n_channels], #1, 23f\n"
    "ldr s12, [x19], #0x4\n"
    "tbz %x[n_channels], #0, 24f\n"
    "ld1 { v12.h }[2], [x19]\n"
    "b 24f\n"
    "23:"  // Tile loop: Oddments: Load inputs: (2, 3): Bit 1: Unset
    "ldr h12, [x19, #0x0]\n"
    "24:"  // Tile loop: Oddments: Load inputs: (2, 3): Bit 1: End
    "fmla v30.8h, v7.8h, v12.8h\n"
    "add x19, x12, x28\n"
    "fmla v28.8h, v1.8h, v12.8h\n"
    "tbz %x[n_channels], #1, 25f\n"
    "ldr s14, [x19], #0x4\n"
    "tbz %x[n_channels], #0, 26f\n"
    "ld1 { v14.h }[2], [x19]\n"
    "b 26f\n"
    "25:"  // Tile loop: Oddments: Load inputs: (3, 4): Bit 1: Unset
    "ldr h14, [x19, #0x0]\n"
    "26:"  // Tile loop: Oddments: Load inputs: (3, 4): Bit 1: End
    "fmla v28.8h, v5.8h, v14.8h\n"
    "add x19, x11, XZR\n"
    "tbz %x[n_channels], #1, 27f\n"
    "ldr s15, [x19], #0x4\n"
    "tbz %x[n_channels], #0, 28f\n"
    "ld1 { v15.h }[2], [x19]\n"
    "b 28f\n"
    "27:"  // Tile loop: Oddments: Load inputs: (4, 0): Bit 1: Unset
    "ldr h15, [x19, #0x0]\n"
    "28:"  // Tile loop: Oddments: Load inputs: (4, 0): Bit 1: End
    "fmla v29.8h, v6.8h, v15.8h\n"
    "add x19, x13, x28\n"
    "tbz %x[n_channels], #1, 29f\n"
    "ldr s11, [x19], #0x4\n"
    "tbz %x[n_channels], #0, 30f\n"
    "ld1 { v11.h }[2], [x19]\n"
    "b 30f\n"
    "29:"  // Tile loop: Oddments: Load inputs: (2, 4): Bit 1: Unset
    "ldr h11, [x19, #0x0]\n"
    "30:"  // Tile loop: Oddments: Load inputs: (2, 4): Bit 1: End
    "fmla v30.8h, v8.8h, v11.8h\n"
    "add x19, x11, x8\n"
    "fmla v28.8h, v2.8h, v11.8h\n"
    "tbz %x[n_channels], #1, 31f\n"
    "ldr s13, [x19], #0x4\n"
    "tbz %x[n_channels], #0, 32f\n"
    "ld1 { v13.h }[2], [x19]\n"
    "b 32f\n"
    "31:"  // Tile loop: Oddments: Load inputs: (4, 1): Bit 1: Unset
    "ldr h13, [x19, #0x0]\n"
    "32:"  // Tile loop: Oddments: Load inputs: (4, 1): Bit 1: End
    "fmla v29.8h, v7.8h, v13.8h\n"
    "add x19, x12, x10\n"
    "tbz %x[n_channels], #1, 33f\n"
    "ldr s16, [x19], #0x4\n"
    "tbz %x[n_channels], #0, 34f\n"
    "ld1 { v16.h }[2], [x19]\n"
    "b 34f\n"
    "33:"  // Tile loop: Oddments: Load inputs: (3, 2): Bit 1: Unset
    "ldr h16, [x19, #0x0]\n"
    "34:"  // Tile loop: Oddments: Load inputs: (3, 2): Bit 1: End
    "fmla v29.8h, v5.8h, v16.8h\n"
    "add x19, x11, x9\n"
    "fmla v28.8h, v3.8h, v16.8h\n"
    "tbz %x[n_channels], #1, 35f\n"
    "ldr s14, [x19], #0x4\n"
    "tbz %x[n_channels], #0, 36f\n"
    "ld1 { v14.h }[2], [x19]\n"
    "b 36f\n"
    "35:"  // Tile loop: Oddments: Load inputs: (4, 3): Bit 1: Unset
    "ldr h14, [x19, #0x0]\n"
    "36:"  // Tile loop: Oddments: Load inputs: (4, 3): Bit 1: End
    "fmla v28.8h, v7.8h, v14.8h\n"
    "add x19, x11, x10\n"
    "tbz %x[n_channels], #1, 37f\n"
    "ldr s15, [x19], #0x4\n"
    "tbz %x[n_channels], #0, 38f\n"
    "ld1 { v15.h }[2], [x19]\n"
    "b 38f\n"
    "37:"  // Tile loop: Oddments: Load inputs: (4, 2): Bit 1: Unset
    "ldr h15, [x19, #0x0]\n"
    "38:"  // Tile loop: Oddments: Load inputs: (4, 2): Bit 1: End
    "fmla v29.8h, v8.8h, v15.8h\n"
    "add x19, x11, x28\n"
    "fmla v28.8h, v6.8h, v15.8h\n"
    "tbz %x[n_channels], #1, 39f\n"
    "ldr s11, [x19], #0x4\n"
    "tbz %x[n_channels], #0, 40f\n"
    "ld1 { v11.h }[2], [x19]\n"
    "b 40f\n"
    "39:"  // Tile loop: Oddments: Load inputs: (4, 4): Bit 1: Unset
    "ldr h11, [x19, #0x0]\n"
    "40:"  // Tile loop: Oddments: Load inputs: (4, 4): Bit 1: End
    "fmla v28.8h, v8.8h, v11.8h\n"
    "fmax v31.8h, v31.8h, v19.8h\n"
    "fmax v30.8h, v30.8h, v19.8h\n"
    "fmax v29.8h, v29.8h, v19.8h\n"
    "fmin v31.8h, v31.8h, v18.8h\n"
    "fmin v30.8h, v30.8h, v18.8h\n"
    "fmin v29.8h, v29.8h, v18.8h\n"
    "fmax v28.8h, v28.8h, v19.8h\n"
    "fmin v28.8h, v28.8h, v18.8h\n"
    "tbz %x[n_channels], #1, 41f\n"
    "mov x19, x15\n"
    "st1 { v31.s }[0], [x19], x16\n"
    "add x15, x15, #0x4\n"
    "st1 { v30.s }[0], [x19]\n"
    "mov x19, x27\n"
    "st1 { v29.s }[0], [x19], x16\n"
    "add x27, x27, #0x4\n"
    "st1 { v28.s }[0], [x19]\n"
    "tbz %x[n_channels], #0, 42f\n"
    "mov x20, x15\n"
    "st1 { v31.h }[2], [x20], x16\n"
    "mov x19, x27\n"
    "st1 { v30.h }[2], [x20]\n"
    "st1 { v29.h }[2], [x19], x16\n"
    "st1 { v28.h }[2], [x19]\n"
    "b 42f\n"
    "41:"  // Tile loop: Oddments: Store: Bit 1: Unset
    "mov x20, x15\n"
    "st1 { v31.h }[0], [x20], x16\n"
    "mov x19, x27\n"
    "st1 { v30.h }[0], [x20]\n"
    "st1 { v29.h }[0], [x19], x16\n"
    "st1 { v28.h }[0], [x19]\n"
    "42:"  // Tile loop: Oddments: Store: Bit 1: End

    "43:"  // Tile loop: End
    "ldr x6, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "add x21, x6, #0x1\n"
    "ldr x27, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "ldr x20, [%x[params_struct], %[offsetof_args_n_tile_rows]]\n"
    "add x27, x27, #0x1\n"
    "ldr x19, [%x[params_struct], %[offsetof_args_n_tile_cols]]\n"
    "cmp x27, x19\n"
    "csel x27, x27, XZR, LT\n"
    "csel x6, x6, x21, LT\n"
    "cmp x6, x20\n"
    "blt 1b\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_args_inptr] "I" (offsetof(Args, inptr)), [offsetof_args_ld_input_col] "I" (offsetof(Args, ld_input_col)), [offsetof_args_ld_input_row] "I" (offsetof(Args, ld_input_row)), [offsetof_args_ld_output_col] "I" (offsetof(Args, ld_output_col)), [offsetof_args_ld_output_row] "I" (offsetof(Args, ld_output_row)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_n_tile_cols] "I" (offsetof(Args, n_tile_cols)), [offsetof_args_n_tile_rows] "I" (offsetof(Args, n_tile_rows)), [offsetof_args_outptr] "I" (offsetof(Args, outptr)), [offsetof_args_params] "I" (offsetof(Args, params)), [offsetof_args_tile_i] "I" (offsetof(Args, tile_i)), [offsetof_args_tile_j] "I" (offsetof(Args, tile_j)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v28", "v29", "v30", "v31", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
