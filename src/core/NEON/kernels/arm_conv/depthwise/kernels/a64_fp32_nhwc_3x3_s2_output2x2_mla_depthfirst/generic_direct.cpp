/*
 * Copyright (c) 2021, 2023 Arm Limited.
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

#if defined(__aarch64__)

namespace arm_conv {
namespace depthwise {

void a64_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst_direct_impl(
  const unsigned int n_tile_rows,
  const unsigned int n_tile_cols,
  const float *inptr,
  int64_t ld_input_row,
  int64_t ld_input_col,
  float *outptr,
  int64_t ld_output_row,
  int64_t ld_output_col,
  const void *params,
  unsigned int n_channels,
  const float activation_min,
  const float activation_max
)
{
  struct Args
  {
    const uint64_t n_tile_rows, n_tile_cols;
    const float *inptr;
    const uint64_t ld_input_row;
    const uint64_t ld_input_col;
    float *outptr;
    const uint64_t ld_output_row;
    const uint64_t ld_output_col;
    const void *params;
    const float min, max;

    uint64_t tile_i = 0, tile_j = 0;

    Args(
      const unsigned int n_tile_rows,
      const unsigned int n_tile_cols,
      const float *inptr,
      int64_t ld_input_row,
      int64_t ld_input_col,
      float *outptr,
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
    "mov x23, #0x0\n"
    "mov x27, #0x0\n"
    "1:"  // Tile loop
    "str x23, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "mov x26, #0x4\n"
    "mov x25, #0x2\n"
    "str x27, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "ldr x24, [%x[params_struct], %[offsetof_args_ld_input_row]]\n"
    "ldr x6, [%x[params_struct], %[offsetof_args_ld_input_col]]\n"
    "mul x22, x23, x24\n"  // offset = tile_i * ld_input_row
    "ldr x21, [%x[params_struct], %[offsetof_args_ld_output_row]]\n"
    "madd x22, x27, x6, x22\n"  // offset += tile_j * ld_input_col
    "ldr x7, [%x[params_struct], %[offsetof_args_ld_output_col]]\n"
    "lsl x6, x6, #0x2\n"
    "mul x20, x23, x21\n"  // offset = tile_i * ld_output_row
    "ldr x8, [%x[params_struct], %[offsetof_args_inptr]]\n"
    "ldr x17, [%x[params_struct], %[offsetof_args_outptr]]\n"
    "mov x23, #0x10\n"  // cntb _, ALL, #1
    "mul x22, x22, x26\n"  // offset *= kernel_stride * output_size
    "add x8, x8, x22, LSL #2\n"  // inptr[0] += offset * sizeof(float)
    "add x16, x8, x24, LSL #2\n"
    "ldr x15, [%x[params_struct], %[offsetof_args_params]]\n"
    "madd x20, x27, x7, x20\n"  // offset += tile_j * ld_output_col
    "lsr x22, %x[n_channels], #0x2\n"
    "add x14, x16, x24, LSL #2\n"
    "mul x20, x20, x25\n"  // offset *= output_tile_size
    "add x13, x6, x6\n"
    "add x12, x14, x24, LSL #2\n"
    "add x11, x13, x6\n"
    "add x17, x17, x20, LSL #2\n"  // outptrs[0] += offset * sizeof(float)
    "add x20, %x[params_struct], %[offsetof_args_min]\n"
    "ld1r { v26.4s }, [x20]\n"
    "add x20, %x[params_struct], %[offsetof_args_max]\n"
    "ld1r { v27.4s }, [x20]\n"
    "add x10, x12, x24, LSL #2\n"
    "add x9, x11, x6\n"
    "add x28, x17, x21, LSL #2\n"
    "lsl x7, x7, #0x2\n"
    "mov x21, #0x0\n"
    "sub x20, XZR, x23\n"
    "cbz x22, 4f\n"
    "ldr q31, [x15, #0x0]\n"
    "ldr q0, [x15, #0x10]\n"
    "cmp x23, x22, LSL #4\n"
    "ldr q1, [x15, #0x20]\n"
    "ldr q2, [x15, #0x30]\n"
    "ldr q3, [x15, #0x40]\n"
    "ldr q4, [x15, #0x50]\n"
    "ldr q5, [x15, #0x60]\n"
    "ldr q6, [x15, #0x70]\n"
    "ldr q7, [x15, #0x80]\n"
    "ldr q8, [x15, #0x90]\n"
    "add x15, x15, #0xa0\n"
    "ldr q9, [x14, x13]\n"
    "ld1 { v10.4s }, [x8]\n"
    "ldr q11, [x8, x6]\n"
    "ldr q12, [x8, x11]\n"
    "ldr q13, [x8, x9]\n"
    "ld1 { v14.4s }, [x16]\n"
    "ldr q15, [x16, x6]\n"
    "ldr q16, [x8, x13]\n"
    "bge 3f\n"
    "2:"  // Tile loop: Channel loop
    "mov v29.16b, v31.16b\n fmla v29.4s, v8.4s, v9.4s\n"
    "mov v28.16b, v31.16b\n fmla v28.4s, v6.4s, v9.4s\n"
    "add x23, x23, #0x10\n"
    "add x8, x8, #0x10\n"
    "fmla v29.4s, v0.4s, v10.4s\n"
    "ld1 { v10.4s }, [x8]\n"
    "fmla v28.4s, v1.4s, v12.4s\n"
    "ldr q21, [x16, x9]\n"
    "fmla v29.4s, v1.4s, v11.4s\n"
    "ldr q18, [x16, x11]\n"
    "fmla v28.4s, v2.4s, v13.4s\n"
    "ldr q17, [x16, x13]\n"
    "fmla v29.4s, v3.4s, v14.4s\n"
    "ld1 { v20.4s }, [x12]\n"
    "fmla v28.4s, v0.4s, v16.4s\n"
    "add x16, x16, #0x10\n"
    "fmla v29.4s, v4.4s, v15.4s\n"
    "ld1 { v25.4s }, [x14]\n"
    "fmla v28.4s, v4.4s, v18.4s\n"
    "ldr q19, [x12, x6]\n"
    "fmla v29.4s, v2.4s, v16.4s\n"
    "ldr q18, [x14, x6]\n"
    "fmla v28.4s, v5.4s, v21.4s\n"
    "ldr q24, [x14, x11]\n"
    "mov v23.16b, v31.16b\n fmla v23.4s, v2.4s, v9.4s\n"
    "mov v22.16b, v31.16b\n fmla v22.4s, v0.4s, v9.4s\n"
    "ldr q31, [x15, #0x0]\n"
    "cmp x23, x22, LSL #4\n"
    "fmla v29.4s, v5.4s, v17.4s\n"
    "fmla v28.4s, v3.4s, v17.4s\n"
    "ldr q17, [x12, x11]\n"
    "add x20, x20, #0x10\n"
    "fmla v23.4s, v3.4s, v20.4s\n"
    "ldr q16, [x12, x9]\n"
    "fmla v22.4s, v4.4s, v17.4s\n"
    "ldr q21, [x10, x6]\n"
    "fmla v23.4s, v0.4s, v25.4s\n"
    "ldr q0, [x15, #0x10]\n"
    "fmla v22.4s, v1.4s, v24.4s\n"
    "add x21, x21, #0x10\n"
    "fmla v23.4s, v4.4s, v19.4s\n"
    "ldr q20, [x14, x9]\n"
    "ldr q4, [x15, #0x50]\n"
    "fmla v22.4s, v5.4s, v16.4s\n"
    "ldr q19, [x10, x11]\n"
    "fmla v29.4s, v6.4s, v25.4s\n"
    "ld1 { v17.4s }, [x10]\n"
    "fmla v23.4s, v1.4s, v18.4s\n"
    "ldr q1, [x15, #0x20]\n"
    "fmla v22.4s, v2.4s, v20.4s\n"
    "ldr q2, [x15, #0x30]\n"
    "fmla v29.4s, v7.4s, v18.4s\n"
    "ldr q16, [x12, x13]\n"
    "fmla v23.4s, v6.4s, v17.4s\n"
    "ldr q18, [x10, x13]\n"
    "fmla v22.4s, v3.4s, v16.4s\n"
    "ldr q3, [x15, #0x40]\n"
    "fmla v23.4s, v7.4s, v21.4s\n"
    "ldr q13, [x8, x9]\n"
    "fmla v22.4s, v7.4s, v19.4s\n"
    "ld1 { v14.4s }, [x16]\n"
    "fmla v28.4s, v7.4s, v24.4s\n"
    "ldr q12, [x8, x11]\n"
    "fmla v23.4s, v5.4s, v16.4s\n"
    "ldr q16, [x8, x13]\n"
    "ldr q5, [x15, #0x60]\n"
    "fmla v22.4s, v6.4s, v18.4s\n"
    "fmla v28.4s, v8.4s, v20.4s\n"
    "ldr q17, [x10, x9]\n"
    "ldr q6, [x15, #0x70]\n"
    "fmla v23.4s, v8.4s, v18.4s\n"
    "fmla v22.4s, v8.4s, v17.4s\n"
    "ldr q11, [x8, x6]\n"
    "ldr q15, [x16, x6]\n"
    "fmax v29.4s, v29.4s, v26.4s\n"
    "fmax v28.4s, v28.4s, v26.4s\n"
    "ldr q7, [x15, #0x80]\n"
    "ldr q8, [x15, #0x90]\n"
    "fmax v23.4s, v23.4s, v26.4s\n"
    "fmax v22.4s, v22.4s, v26.4s\n"
    "add x14, x14, #0x10\n"
    "ldr q9, [x14, x13]\n"
    "fmin v29.4s, v29.4s, v27.4s\n"
    "fmin v28.4s, v28.4s, v27.4s\n"
    "fmin v23.4s, v23.4s, v27.4s\n"
    "fmin v22.4s, v22.4s, v27.4s\n"
    "add x12, x12, #0x10\n"
    "add x10, x10, #0x10\n"
    "st1 { v29.4s }, [x17]\n"
    "add x15, x15, #0xa0\n"
    "str q28, [x17, x7]\n"
    "add x17, x17, #0x10\n"
    "st1 { v23.4s }, [x28]\n"
    "str q22, [x28, x7]\n"
    "add x28, x28, #0x10\n"
    "blt 2b\n"
    "3:"  // Tile loop: Channel tail
    "mov v29.16b, v31.16b\n fmla v29.4s, v8.4s, v9.4s\n"
    "mov v28.16b, v31.16b\n fmla v28.4s, v6.4s, v9.4s\n"
    "add x8, x8, #0x10\n"
    "fmla v29.4s, v0.4s, v10.4s\n"
    "fmla v28.4s, v1.4s, v12.4s\n"
    "ldr q20, [x16, x9]\n"
    "fmla v29.4s, v1.4s, v11.4s\n"
    "ldr q18, [x16, x11]\n"
    "fmla v28.4s, v2.4s, v13.4s\n"
    "ldr q17, [x16, x13]\n"
    "fmla v29.4s, v3.4s, v14.4s\n"
    "ld1 { v19.4s }, [x12]\n"
    "fmla v28.4s, v0.4s, v16.4s\n"
    "add x16, x16, #0x10\n"
    "fmla v29.4s, v4.4s, v15.4s\n"
    "ld1 { v25.4s }, [x14]\n"
    "fmla v28.4s, v4.4s, v18.4s\n"
    "ldr q18, [x12, x6]\n"
    "fmla v29.4s, v2.4s, v16.4s\n"
    "ldr q24, [x14, x6]\n"
    "fmla v28.4s, v5.4s, v20.4s\n"
    "ldr q23, [x14, x11]\n"
    "mov v22.16b, v31.16b\n fmla v22.4s, v2.4s, v9.4s\n"
    "mov v21.16b, v31.16b\n fmla v21.4s, v0.4s, v9.4s\n"
    "fmla v29.4s, v5.4s, v17.4s\n"
    "fmla v28.4s, v3.4s, v17.4s\n"
    "ldr q17, [x12, x11]\n"
    "fmla v22.4s, v3.4s, v19.4s\n"
    "ldr q16, [x12, x9]\n"
    "fmla v21.4s, v4.4s, v17.4s\n"
    "ldr q20, [x10, x6]\n"
    "fmla v22.4s, v0.4s, v25.4s\n"
    "fmla v21.4s, v1.4s, v23.4s\n"
    "fmla v22.4s, v4.4s, v18.4s\n"
    "ldr q19, [x14, x9]\n"
    "fmla v21.4s, v5.4s, v16.4s\n"
    "ldr q18, [x10, x11]\n"
    "fmla v29.4s, v6.4s, v25.4s\n"
    "ld1 { v17.4s }, [x10]\n"
    "fmla v22.4s, v1.4s, v24.4s\n"
    "add x14, x14, #0x10\n"
    "fmla v21.4s, v2.4s, v19.4s\n"
    "fmla v29.4s, v7.4s, v24.4s\n"
    "ldr q16, [x12, x13]\n"
    "fmax v29.4s, v29.4s, v26.4s\n"
    "fmla v22.4s, v6.4s, v17.4s\n"
    "ldr q17, [x10, x13]\n"
    "fmla v21.4s, v3.4s, v16.4s\n"
    "fmin v29.4s, v29.4s, v27.4s\n"
    "fmla v22.4s, v7.4s, v20.4s\n"
    "fmla v21.4s, v7.4s, v18.4s\n"
    "st1 { v29.4s }, [x17]\n"
    "add x12, x12, #0x10\n"
    "fmla v28.4s, v7.4s, v23.4s\n"
    "fmla v22.4s, v5.4s, v16.4s\n"
    "fmla v21.4s, v6.4s, v17.4s\n"
    "fmla v28.4s, v8.4s, v19.4s\n"
    "ldr q16, [x10, x9]\n"
    "fmax v28.4s, v28.4s, v26.4s\n"
    "fmla v22.4s, v8.4s, v17.4s\n"
    "fmla v21.4s, v8.4s, v16.4s\n"
    "fmax v22.4s, v22.4s, v26.4s\n"
    "add x10, x10, #0x10\n"
    "fmax v21.4s, v21.4s, v26.4s\n"
    "fmin v28.4s, v28.4s, v27.4s\n"
    "str q28, [x17, x7]\n"
    "add x17, x17, #0x10\n"
    "fmin v22.4s, v22.4s, v27.4s\n"
    "fmin v21.4s, v21.4s, v27.4s\n"
    "st1 { v22.4s }, [x28]\n"
    "str q21, [x28, x7]\n"
    "add x28, x28, #0x10\n"
    "4:"  // Tile loop: Oddments
    "tst %x[n_channels], #0x3\n"
    "beq 43f\n"
    "ldr q31, [x15, #0x0]\n"
    "ldr q0, [x15, #0x10]\n"
    "add x27, x14, x13\n"
    "add x26, x8, XZR\n"
    "ldr q1, [x15, #0x20]\n"
    "ldr q2, [x15, #0x30]\n"
    "add x25, x8, x6\n"
    "add x24, x8, x11\n"
    "ldr q3, [x15, #0x40]\n"
    "ldr q4, [x15, #0x50]\n"
    "add x23, x8, x9\n"
    "add x22, x16, XZR\n"
    "ldr q5, [x15, #0x60]\n"
    "ldr q6, [x15, #0x70]\n"
    "add x21, x16, x6\n"
    "add x20, x8, x13\n"
    "ldr q7, [x15, #0x80]\n"
    "ldr q8, [x15, #0x90]\n"
    "tbz %x[n_channels], #1, 5f\n"
    "ldr d9, [x27], #0x8\n"
    "ldr d10, [x26], #0x8\n"
    "ldr d11, [x25], #0x8\n"
    "ldr d12, [x24], #0x8\n"
    "ldr d13, [x23], #0x8\n"
    "ldr d14, [x22], #0x8\n"
    "ldr d15, [x21], #0x8\n"
    "ldr d16, [x20], #0x8\n"
    "tbz %x[n_channels], #0, 6f\n"
    "ld1 { v9.s }[2], [x27]\n"
    "ld1 { v10.s }[2], [x26]\n"
    "ld1 { v11.s }[2], [x25]\n"
    "ld1 { v12.s }[2], [x24]\n"
    "ld1 { v13.s }[2], [x23]\n"
    "ld1 { v14.s }[2], [x22]\n"
    "ld1 { v15.s }[2], [x21]\n"
    "ld1 { v16.s }[2], [x20]\n"
    "b 6f\n"
    "5:"  // Tile loop: Oddments: Load inputs: (2, 2), (0, 0), (0, 1), (0, 3), (0, 4), (1, 0), (1, 1), (0, 2): Bit 1: Unset
    "ldr s9, [x27, #0x0]\n"
    "ldr s10, [x26, #0x0]\n"
    "ldr s11, [x25, #0x0]\n"
    "ldr s12, [x24, #0x0]\n"
    "ldr s13, [x23, #0x0]\n"
    "ldr s14, [x22, #0x0]\n"
    "ldr s15, [x21, #0x0]\n"
    "ldr s16, [x20, #0x0]\n"
    "6:"  // Tile loop: Oddments: Load inputs: (2, 2), (0, 0), (0, 1), (0, 3), (0, 4), (1, 0), (1, 1), (0, 2): Bit 1: End
    "mov v28.16b, v31.16b\n fmla v28.4s, v8.4s, v9.4s\n"
    "fmla v28.4s, v0.4s, v10.4s\n"
    "add x20, x16, x11\n"
    "mov v29.16b, v31.16b\n fmla v29.4s, v6.4s, v9.4s\n"
    "fmla v28.4s, v1.4s, v11.4s\n"
    "fmla v29.4s, v1.4s, v12.4s\n"
    "fmla v28.4s, v3.4s, v14.4s\n"
    "fmla v29.4s, v2.4s, v13.4s\n"
    "fmla v28.4s, v4.4s, v15.4s\n"
    "mov v30.16b, v31.16b\n fmla v30.4s, v2.4s, v9.4s\n"
    "fmla v31.4s, v0.4s, v9.4s\n"
    "fmla v28.4s, v2.4s, v16.4s\n"
    "fmla v29.4s, v0.4s, v16.4s\n"
    "tbz %x[n_channels], #1, 7f\n"
    "ldr d11, [x20], #0x8\n"
    "tbz %x[n_channels], #0, 8f\n"
    "ld1 { v11.s }[2], [x20]\n"
    "b 8f\n"
    "7:"  // Tile loop: Oddments: Load inputs: (1, 3): Bit 1: Unset
    "ldr s11, [x20, #0x0]\n"
    "8:"  // Tile loop: Oddments: Load inputs: (1, 3): Bit 1: End
    "fmla v29.4s, v4.4s, v11.4s\n"
    "add x20, x16, x9\n"
    "tbz %x[n_channels], #1, 9f\n"
    "ldr d12, [x20], #0x8\n"
    "tbz %x[n_channels], #0, 10f\n"
    "ld1 { v12.s }[2], [x20]\n"
    "b 10f\n"
    "9:"  // Tile loop: Oddments: Load inputs: (1, 4): Bit 1: Unset
    "ldr s12, [x20, #0x0]\n"
    "10:"  // Tile loop: Oddments: Load inputs: (1, 4): Bit 1: End
    "fmla v29.4s, v5.4s, v12.4s\n"
    "add x20, x16, x13\n"
    "tbz %x[n_channels], #1, 11f\n"
    "ldr d13, [x20], #0x8\n"
    "tbz %x[n_channels], #0, 12f\n"
    "ld1 { v13.s }[2], [x20]\n"
    "b 12f\n"
    "11:"  // Tile loop: Oddments: Load inputs: (1, 2): Bit 1: Unset
    "ldr s13, [x20, #0x0]\n"
    "12:"  // Tile loop: Oddments: Load inputs: (1, 2): Bit 1: End
    "fmla v28.4s, v5.4s, v13.4s\n"
    "fmla v29.4s, v3.4s, v13.4s\n"
    "add x20, x12, XZR\n"
    "tbz %x[n_channels], #1, 13f\n"
    "ldr d14, [x20], #0x8\n"
    "tbz %x[n_channels], #0, 14f\n"
    "ld1 { v14.s }[2], [x20]\n"
    "b 14f\n"
    "13:"  // Tile loop: Oddments: Load inputs: (3, 0): Bit 1: Unset
    "ldr s14, [x20, #0x0]\n"
    "14:"  // Tile loop: Oddments: Load inputs: (3, 0): Bit 1: End
    "fmla v30.4s, v3.4s, v14.4s\n"
    "add x20, x14, XZR\n"
    "tbz %x[n_channels], #1, 15f\n"
    "ldr d15, [x20], #0x8\n"
    "tbz %x[n_channels], #0, 16f\n"
    "ld1 { v15.s }[2], [x20]\n"
    "b 16f\n"
    "15:"  // Tile loop: Oddments: Load inputs: (2, 0): Bit 1: Unset
    "ldr s15, [x20, #0x0]\n"
    "16:"  // Tile loop: Oddments: Load inputs: (2, 0): Bit 1: End
    "fmla v28.4s, v6.4s, v15.4s\n"
    "fmla v30.4s, v0.4s, v15.4s\n"
    "add x20, x12, x6\n"
    "tbz %x[n_channels], #1, 17f\n"
    "ldr d11, [x20], #0x8\n"
    "tbz %x[n_channels], #0, 18f\n"
    "ld1 { v11.s }[2], [x20]\n"
    "b 18f\n"
    "17:"  // Tile loop: Oddments: Load inputs: (3, 1): Bit 1: Unset
    "ldr s11, [x20, #0x0]\n"
    "18:"  // Tile loop: Oddments: Load inputs: (3, 1): Bit 1: End
    "fmla v30.4s, v4.4s, v11.4s\n"
    "add x20, x14, x6\n"
    "tbz %x[n_channels], #1, 19f\n"
    "ldr d16, [x20], #0x8\n"
    "tbz %x[n_channels], #0, 20f\n"
    "ld1 { v16.s }[2], [x20]\n"
    "b 20f\n"
    "19:"  // Tile loop: Oddments: Load inputs: (2, 1): Bit 1: Unset
    "ldr s16, [x20, #0x0]\n"
    "20:"  // Tile loop: Oddments: Load inputs: (2, 1): Bit 1: End
    "fmla v28.4s, v7.4s, v16.4s\n"
    "fmla v30.4s, v1.4s, v16.4s\n"
    "add x20, x12, x11\n"
    "tbz %x[n_channels], #1, 21f\n"
    "ldr d13, [x20], #0x8\n"
    "tbz %x[n_channels], #0, 22f\n"
    "ld1 { v13.s }[2], [x20]\n"
    "b 22f\n"
    "21:"  // Tile loop: Oddments: Load inputs: (3, 3): Bit 1: Unset
    "ldr s13, [x20, #0x0]\n"
    "22:"  // Tile loop: Oddments: Load inputs: (3, 3): Bit 1: End
    "fmla v31.4s, v4.4s, v13.4s\n"
    "add x20, x14, x11\n"
    "tbz %x[n_channels], #1, 23f\n"
    "ldr d12, [x20], #0x8\n"
    "tbz %x[n_channels], #0, 24f\n"
    "ld1 { v12.s }[2], [x20]\n"
    "b 24f\n"
    "23:"  // Tile loop: Oddments: Load inputs: (2, 3): Bit 1: Unset
    "ldr s12, [x20, #0x0]\n"
    "24:"  // Tile loop: Oddments: Load inputs: (2, 3): Bit 1: End
    "fmla v29.4s, v7.4s, v12.4s\n"
    "fmla v31.4s, v1.4s, v12.4s\n"
    "add x20, x12, x9\n"
    "tbz %x[n_channels], #1, 25f\n"
    "ldr d14, [x20], #0x8\n"
    "tbz %x[n_channels], #0, 26f\n"
    "ld1 { v14.s }[2], [x20]\n"
    "b 26f\n"
    "25:"  // Tile loop: Oddments: Load inputs: (3, 4): Bit 1: Unset
    "ldr s14, [x20, #0x0]\n"
    "26:"  // Tile loop: Oddments: Load inputs: (3, 4): Bit 1: End
    "fmla v31.4s, v5.4s, v14.4s\n"
    "add x20, x10, XZR\n"
    "tbz %x[n_channels], #1, 27f\n"
    "ldr d15, [x20], #0x8\n"
    "tbz %x[n_channels], #0, 28f\n"
    "ld1 { v15.s }[2], [x20]\n"
    "b 28f\n"
    "27:"  // Tile loop: Oddments: Load inputs: (4, 0): Bit 1: Unset
    "ldr s15, [x20, #0x0]\n"
    "28:"  // Tile loop: Oddments: Load inputs: (4, 0): Bit 1: End
    "fmla v30.4s, v6.4s, v15.4s\n"
    "add x20, x14, x9\n"
    "tbz %x[n_channels], #1, 29f\n"
    "ldr d11, [x20], #0x8\n"
    "tbz %x[n_channels], #0, 30f\n"
    "ld1 { v11.s }[2], [x20]\n"
    "b 30f\n"
    "29:"  // Tile loop: Oddments: Load inputs: (2, 4): Bit 1: Unset
    "ldr s11, [x20, #0x0]\n"
    "30:"  // Tile loop: Oddments: Load inputs: (2, 4): Bit 1: End
    "fmla v29.4s, v8.4s, v11.4s\n"
    "fmla v31.4s, v2.4s, v11.4s\n"
    "add x20, x10, x6\n"
    "tbz %x[n_channels], #1, 31f\n"
    "ldr d13, [x20], #0x8\n"
    "tbz %x[n_channels], #0, 32f\n"
    "ld1 { v13.s }[2], [x20]\n"
    "b 32f\n"
    "31:"  // Tile loop: Oddments: Load inputs: (4, 1): Bit 1: Unset
    "ldr s13, [x20, #0x0]\n"
    "32:"  // Tile loop: Oddments: Load inputs: (4, 1): Bit 1: End
    "fmla v30.4s, v7.4s, v13.4s\n"
    "add x20, x12, x13\n"
    "tbz %x[n_channels], #1, 33f\n"
    "ldr d16, [x20], #0x8\n"
    "tbz %x[n_channels], #0, 34f\n"
    "ld1 { v16.s }[2], [x20]\n"
    "b 34f\n"
    "33:"  // Tile loop: Oddments: Load inputs: (3, 2): Bit 1: Unset
    "ldr s16, [x20, #0x0]\n"
    "34:"  // Tile loop: Oddments: Load inputs: (3, 2): Bit 1: End
    "fmla v30.4s, v5.4s, v16.4s\n"
    "fmla v31.4s, v3.4s, v16.4s\n"
    "add x20, x10, x11\n"
    "tbz %x[n_channels], #1, 35f\n"
    "ldr d14, [x20], #0x8\n"
    "tbz %x[n_channels], #0, 36f\n"
    "ld1 { v14.s }[2], [x20]\n"
    "b 36f\n"
    "35:"  // Tile loop: Oddments: Load inputs: (4, 3): Bit 1: Unset
    "ldr s14, [x20, #0x0]\n"
    "36:"  // Tile loop: Oddments: Load inputs: (4, 3): Bit 1: End
    "fmla v31.4s, v7.4s, v14.4s\n"
    "add x20, x10, x13\n"
    "tbz %x[n_channels], #1, 37f\n"
    "ldr d15, [x20], #0x8\n"
    "tbz %x[n_channels], #0, 38f\n"
    "ld1 { v15.s }[2], [x20]\n"
    "b 38f\n"
    "37:"  // Tile loop: Oddments: Load inputs: (4, 2): Bit 1: Unset
    "ldr s15, [x20, #0x0]\n"
    "38:"  // Tile loop: Oddments: Load inputs: (4, 2): Bit 1: End
    "fmla v30.4s, v8.4s, v15.4s\n"
    "fmla v31.4s, v6.4s, v15.4s\n"
    "add x20, x10, x9\n"
    "tbz %x[n_channels], #1, 39f\n"
    "ldr d11, [x20], #0x8\n"
    "tbz %x[n_channels], #0, 40f\n"
    "ld1 { v11.s }[2], [x20]\n"
    "b 40f\n"
    "39:"  // Tile loop: Oddments: Load inputs: (4, 4): Bit 1: Unset
    "ldr s11, [x20, #0x0]\n"
    "40:"  // Tile loop: Oddments: Load inputs: (4, 4): Bit 1: End
    "fmla v31.4s, v8.4s, v11.4s\n"
    "fmax v28.4s, v28.4s, v26.4s\n"
    "fmax v29.4s, v29.4s, v26.4s\n"
    "fmax v30.4s, v30.4s, v26.4s\n"
    "fmax v31.4s, v31.4s, v26.4s\n"
    "fmin v28.4s, v28.4s, v27.4s\n"
    "fmin v29.4s, v29.4s, v27.4s\n"
    "fmin v30.4s, v30.4s, v27.4s\n"
    "fmin v31.4s, v31.4s, v27.4s\n"
    "tbz %x[n_channels], #1, 41f\n"
    "mov x21, x17\n"
    "mov x20, x28\n"
    "st1 { v28.d }[0], [x21], x7\n"
    "st1 { v30.d }[0], [x20], x7\n"
    "add x17, x17, #0x8\n"
    "add x28, x28, #0x8\n"
    "st1 { v29.d }[0], [x21]\n"
    "st1 { v31.d }[0], [x20]\n"
    "tbz %x[n_channels], #0, 42f\n"
    "mov x21, x17\n"
    "mov x20, x28\n"
    "st1 { v28.s }[2], [x21], x7\n"
    "st1 { v30.s }[2], [x20], x7\n"
    "st1 { v29.s }[2], [x21]\n"
    "st1 { v31.s }[2], [x20]\n"
    "b 42f\n"
    "41:"  // Tile loop: Oddments: Store: Bit 1: Unset
    "mov x21, x17\n"
    "mov x20, x28\n"
    "st1 { v28.s }[0], [x21], x7\n"
    "st1 { v30.s }[0], [x20], x7\n"
    "st1 { v29.s }[0], [x21]\n"
    "st1 { v31.s }[0], [x20]\n"
    "42:"  // Tile loop: Oddments: Store: Bit 1: End
    "43:"  // Tile loop: End
    "ldr x27, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "ldr x23, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "add x27, x27, #0x1\n"
    "add x21, x23, #0x1\n"
    "ldr x20, [%x[params_struct], %[offsetof_args_n_tile_cols]]\n"
    "cmp x27, x20\n"
    "ldr x20, [%x[params_struct], %[offsetof_args_n_tile_rows]]\n"
    "csel x23, x23, x21, LT\n"
    "csel x27, x27, XZR, LT\n"
    "cmp x23, x20\n"
    "blt 1b\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_args_inptr] "I" (offsetof(Args, inptr)), [offsetof_args_ld_input_col] "I" (offsetof(Args, ld_input_col)), [offsetof_args_ld_input_row] "I" (offsetof(Args, ld_input_row)), [offsetof_args_ld_output_col] "I" (offsetof(Args, ld_output_col)), [offsetof_args_ld_output_row] "I" (offsetof(Args, ld_output_row)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_n_tile_cols] "I" (offsetof(Args, n_tile_cols)), [offsetof_args_n_tile_rows] "I" (offsetof(Args, n_tile_rows)), [offsetof_args_outptr] "I" (offsetof(Args, outptr)), [offsetof_args_params] "I" (offsetof(Args, params)), [offsetof_args_tile_i] "I" (offsetof(Args, tile_i)), [offsetof_args_tile_j] "I" (offsetof(Args, tile_j)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(__aarch64__)
