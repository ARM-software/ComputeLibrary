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

namespace arm_conv {
namespace depthwise {

void a64_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst_direct_impl(
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
    "mov x17, #0x0\n"
    "mov x16, #0x0\n"
    "1:"  // Tile loop
    "str x17, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "mov x25, #0x2\n"
    "str x16, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "mov x15, #0x2\n"
    "ldr x14, [%x[params_struct], %[offsetof_args_params]]\n"
    "add x24, %x[params_struct], %[offsetof_args_min]\n"
    "ldr x23, [%x[params_struct], %[offsetof_args_ld_input_row]]\n"
    "add x21, %x[params_struct], %[offsetof_args_max]\n"
    "ldr x13, [%x[params_struct], %[offsetof_args_ld_input_col]]\n"
    "mov x22, #0x0\n"
    "ldr x12, [%x[params_struct], %[offsetof_args_inptr]]\n"
    "mul x19, x17, x23\n" // offset = tile_i * ld_input_row
    "ldr x20, [%x[params_struct], %[offsetof_args_ld_output_row]]\n"
    "madd x19, x16, x13, x19\n" // offset += tile_j * ld_input_col
    "ldr x11, [%x[params_struct], %[offsetof_args_ld_output_col]]\n"
    "mul x19, x19, x25\n" // offset *= kernel_stride * output_size
    "ldr x10, [%x[params_struct], %[offsetof_args_outptr]]\n"
    "add x12, x12, x19, LSL #2\n" // inptr[0] += offset * sizeof(float)
    "ld1r { v18.4s }, [x24]\n"
    "add x9, x12, x23, LSL #2\n"
    "ld1r { v17.4s }, [x21]\n"
    "add x28, x9, x23, LSL #2\n"
    "lsl x13, x13, #0x2\n"
    "add x27, x28, x23, LSL #2\n"
    "add x26, x13, x13\n"
    "add x25, x26, x13\n"
    "mul x19, x17, x20\n" // offset = tile_i * ld_output_row
    "madd x19, x16, x11, x19\n" // offset += tile_j * ld_output_col
    "mul x19, x19, x15\n" // offset *= output_tile_size
    "add x10, x10, x19, LSL #2\n" // outptrs[0] += offset * sizeof(float)
    "add x24, x10, x20, LSL #2\n"
    "lsl x11, x11, #0x2\n"
    "mov x21, #0x10\n" // cntb _, ALL, #1
    "sub x20, XZR, x21\n"
    "lsr x19, %x[n_channels], #0x2\n"
    "cbz x19, 4f\n"
    "ldr q16, [x14, #0x0]\n"
    "ldr q0, [x14, #0x10]\n"
    "cmp x21, x19, LSL #4\n"
    "ldr q1, [x14, #0x20]\n"
    "ldr q2, [x14, #0x30]\n"
    "ldr q3, [x14, #0x40]\n"
    "ldr q4, [x14, #0x50]\n"
    "ldr q5, [x14, #0x60]\n"
    "ldr q6, [x14, #0x70]\n"
    "ldr q7, [x14, #0x80]\n"
    "ldr q8, [x14, #0x90]\n"
    "add x14, x14, #0xa0\n"
    "ldr q9, [x9, x13]\n"
    "ld1 { v10.4s }, [x12]\n"
    "ldr q11, [x12, x25]\n"
    "ldr q12, [x9, x26]\n"
    "ldr q13, [x28, x13]\n"
    "bge 3f\n"
    "2:"  // Tile loop: Channel loop
    "mov v31.16b, v16.16b\n fmla v31.4s, v4.4s, v9.4s\n"
    "add x20, x20, #0x10\n"
    "mov v30.16b, v16.16b\n fmla v30.4s, v3.4s, v9.4s\n"
    "add x22, x22, #0x10\n"
    "mov v29.16b, v16.16b\n fmla v29.4s, v1.4s, v9.4s\n"
    "add x21, x21, #0x10\n"
    "mov v28.16b, v16.16b\n fmla v28.4s, v0.4s, v9.4s\n"
    "ld1 { v9.4s }, [x27]\n"
    "cmp x21, x19, LSL #4\n"
    "fmla v31.4s, v0.4s, v10.4s\n"
    "ldr q10, [x28, x26]\n"
    "fmla v30.4s, v2.4s, v11.4s\n"
    "ldr q11, [x27, x25]\n"
    "fmla v29.4s, v2.4s, v12.4s\n"
    "ldr q16, [x14, #0x0]\n"
    "fmla v28.4s, v1.4s, v12.4s\n"
    "fmla v31.4s, v5.4s, v12.4s\n"
    "fmla v30.4s, v4.4s, v12.4s\n"
    "ldr q12, [x12, x13]\n"
    "fmla v29.4s, v6.4s, v9.4s\n"
    "ldr q9, [x12, x26]\n"
    "add x12, x12, #0x10\n"
    "fmla v28.4s, v3.4s, v13.4s\n"
    "fmla v31.4s, v7.4s, v13.4s\n"
    "fmla v30.4s, v6.4s, v13.4s\n"
    "fmla v29.4s, v4.4s, v13.4s\n"
    "fmla v28.4s, v8.4s, v11.4s\n"
    "ld1 { v11.4s }, [x9]\n"
    "fmla v31.4s, v1.4s, v12.4s\n"
    "fmla v30.4s, v0.4s, v12.4s\n"
    "ldr q12, [x9, x25]\n"
    "add x9, x9, #0x10\n"
    "fmla v29.4s, v5.4s, v10.4s\n"
    "fmla v28.4s, v4.4s, v10.4s\n"
    "ldr q4, [x14, #0x50]\n"
    "fmla v31.4s, v2.4s, v9.4s\n"
    "fmla v30.4s, v1.4s, v9.4s\n"
    "ld1 { v9.4s }, [x28]\n"
    "ldr q1, [x14, #0x20]\n"
    "fmla v29.4s, v0.4s, v11.4s\n"
    "ldr q0, [x14, #0x10]\n"
    "fmla v28.4s, v2.4s, v12.4s\n"
    "ldr q2, [x14, #0x30]\n"
    "fmla v31.4s, v8.4s, v10.4s\n"
    "fmla v30.4s, v7.4s, v10.4s\n"
    "ldr q10, [x28, x25]\n"
    "add x28, x28, #0x10\n"
    "fmla v29.4s, v3.4s, v9.4s\n"
    "ldr q13, [x28, x13]\n"
    "fmla v31.4s, v3.4s, v11.4s\n"
    "ldr q11, [x27, x13]\n"
    "fmla v30.4s, v5.4s, v12.4s\n"
    "ldr q12, [x27, x26]\n"
    "add x27, x27, #0x10\n"
    "fmla v28.4s, v5.4s, v10.4s\n"
    "ldr q3, [x14, #0x40]\n"
    "ldr q5, [x14, #0x60]\n"
    "fmla v31.4s, v6.4s, v9.4s\n"
    "ldr q9, [x9, x13]\n"
    "fmla v30.4s, v8.4s, v10.4s\n"
    "ld1 { v10.4s }, [x12]\n"
    "fmla v29.4s, v7.4s, v11.4s\n"
    "fmla v28.4s, v6.4s, v11.4s\n"
    "ldr q11, [x12, x25]\n"
    "ldr q6, [x14, #0x70]\n"
    "fmax v31.4s, v31.4s, v18.4s\n"
    "fmax v30.4s, v30.4s, v18.4s\n"
    "fmla v29.4s, v8.4s, v12.4s\n"
    "ldr q8, [x14, #0x90]\n"
    "fmla v28.4s, v7.4s, v12.4s\n"
    "ldr q12, [x9, x26]\n"
    "fmin v31.4s, v31.4s, v17.4s\n"
    "ldr q7, [x14, #0x80]\n"
    "add x14, x14, #0xa0\n"
    "fmin v30.4s, v30.4s, v17.4s\n"
    "st1 { v31.4s }, [x10]\n"
    "fmax v29.4s, v29.4s, v18.4s\n"
    "fmax v28.4s, v28.4s, v18.4s\n"
    "str q30, [x10, x11]\n"
    "fmin v29.4s, v29.4s, v17.4s\n"
    "st1 { v29.4s }, [x24]\n"
    "fmin v28.4s, v28.4s, v17.4s\n"
    "add x10, x10, #0x10\n"
    "str q28, [x24, x11]\n"
    "add x24, x24, #0x10\n"
    "blt 2b\n"
    "3:"  // Tile loop: Channel tail
    "mov v31.16b, v16.16b\n fmla v31.4s, v4.4s, v9.4s\n"
    "mov v30.16b, v16.16b\n fmla v30.4s, v3.4s, v9.4s\n"
    "mov v29.16b, v16.16b\n fmla v29.4s, v1.4s, v9.4s\n"
    "mov v28.16b, v16.16b\n fmla v28.4s, v0.4s, v9.4s\n"
    "ld1 { v9.4s }, [x27]\n"
    "fmla v31.4s, v0.4s, v10.4s\n"
    "ldr q10, [x28, x26]\n"
    "fmla v30.4s, v2.4s, v11.4s\n"
    "ldr q11, [x27, x25]\n"
    "fmla v29.4s, v2.4s, v12.4s\n"
    "fmla v28.4s, v1.4s, v12.4s\n"
    "fmla v31.4s, v5.4s, v12.4s\n"
    "fmla v30.4s, v4.4s, v12.4s\n"
    "ldr q12, [x12, x13]\n"
    "fmla v29.4s, v6.4s, v9.4s\n"
    "ldr q9, [x12, x26]\n"
    "add x12, x12, #0x10\n"
    "fmla v28.4s, v3.4s, v13.4s\n"
    "fmla v31.4s, v7.4s, v13.4s\n"
    "fmla v30.4s, v6.4s, v13.4s\n"
    "fmla v29.4s, v4.4s, v13.4s\n"
    "fmla v28.4s, v8.4s, v11.4s\n"
    "ld1 { v11.4s }, [x9]\n"
    "fmla v31.4s, v1.4s, v12.4s\n"
    "fmla v30.4s, v0.4s, v12.4s\n"
    "ldr q12, [x9, x25]\n"
    "add x9, x9, #0x10\n"
    "fmla v29.4s, v5.4s, v10.4s\n"
    "fmla v28.4s, v4.4s, v10.4s\n"
    "fmla v31.4s, v2.4s, v9.4s\n"
    "fmla v30.4s, v1.4s, v9.4s\n"
    "ld1 { v9.4s }, [x28]\n"
    "fmla v29.4s, v0.4s, v11.4s\n"
    "fmla v28.4s, v2.4s, v12.4s\n"
    "fmla v31.4s, v8.4s, v10.4s\n"
    "fmla v30.4s, v7.4s, v10.4s\n"
    "ldr q10, [x28, x25]\n"
    "add x28, x28, #0x10\n"
    "fmla v29.4s, v3.4s, v9.4s\n"
    "fmla v31.4s, v3.4s, v11.4s\n"
    "ldr q11, [x27, x13]\n"
    "fmla v30.4s, v5.4s, v12.4s\n"
    "ldr q12, [x27, x26]\n"
    "add x27, x27, #0x10\n"
    "fmla v28.4s, v5.4s, v10.4s\n"
    "fmla v31.4s, v6.4s, v9.4s\n"
    "fmla v30.4s, v8.4s, v10.4s\n"
    "fmla v29.4s, v7.4s, v11.4s\n"
    "fmla v28.4s, v6.4s, v11.4s\n"
    "fmax v31.4s, v31.4s, v18.4s\n"
    "fmax v30.4s, v30.4s, v18.4s\n"
    "fmla v29.4s, v8.4s, v12.4s\n"
    "fmla v28.4s, v7.4s, v12.4s\n"
    "fmin v31.4s, v31.4s, v17.4s\n"
    "st1 { v31.4s }, [x10]\n"
    "fmin v30.4s, v30.4s, v17.4s\n"
    "fmax v29.4s, v29.4s, v18.4s\n"
    "str q30, [x10, x11]\n"
    "fmin v29.4s, v29.4s, v17.4s\n"
    "add x10, x10, #0x10\n"
    "fmax v28.4s, v28.4s, v18.4s\n"
    "st1 { v29.4s }, [x24]\n"
    "fmin v28.4s, v28.4s, v17.4s\n"
    "str q28, [x24, x11]\n"
    "add x24, x24, #0x10\n"
    "4:"  // Tile loop: Oddments
    "tst %x[n_channels], #0x3\n"
    "beq 31f\n"
    "ldr q16, [x14, #0x0]\n"
    "ldr q0, [x14, #0x10]\n"
    "add x23, x9, x13\n"
    "ldr q1, [x14, #0x20]\n"
    "add x22, x12, XZR\n"
    "ldr q2, [x14, #0x30]\n"
    "add x21, x12, x25\n"
    "ldr q3, [x14, #0x40]\n"
    "add x20, x9, x26\n"
    "ldr q4, [x14, #0x50]\n"
    "add x19, x28, x13\n"
    "ldr q5, [x14, #0x60]\n"
    "ldr q6, [x14, #0x70]\n"
    "ldr q7, [x14, #0x80]\n"
    "ldr q8, [x14, #0x90]\n"
    "tbz %x[n_channels], #1, 5f\n"
    "ldr d9, [x23], #0x8\n"
    "ldr d10, [x22], #0x8\n"
    "ldr d11, [x21], #0x8\n"
    "ldr d12, [x20], #0x8\n"
    "ldr d13, [x19], #0x8\n"
    "tbz %x[n_channels], #0, 6f\n"
    "ld1 { v9.s }[2], [x23]\n"
    "ld1 { v10.s }[2], [x22]\n"
    "ld1 { v11.s }[2], [x21]\n"
    "ld1 { v12.s }[2], [x20]\n"
    "ld1 { v13.s }[2], [x19]\n"
    "b 6f\n"
    "5:"  // Tile loop: Oddments: Load inputs: (1, 1), (0, 0), (0, 3), (1, 2), (2, 1): Bit 1: Unset
    "ldr s9, [x23, #0x0]\n"
    "ldr s10, [x22, #0x0]\n"
    "ldr s11, [x21, #0x0]\n"
    "ldr s12, [x20, #0x0]\n"
    "ldr s13, [x19, #0x0]\n"
    "6:"  // Tile loop: Oddments: Load inputs: (1, 1), (0, 0), (0, 3), (1, 2), (2, 1): Bit 1: End
    "mov v31.16b, v16.16b\n fmla v31.4s, v4.4s, v9.4s\n"
    "add x19, x27, XZR\n"
    "mov v30.16b, v16.16b\n fmla v30.4s, v3.4s, v9.4s\n"
    "mov v29.16b, v16.16b\n fmla v29.4s, v1.4s, v9.4s\n"
    "mov v28.16b, v16.16b\n fmla v28.4s, v0.4s, v9.4s\n"
    "fmla v31.4s, v0.4s, v10.4s\n"
    "fmla v30.4s, v2.4s, v11.4s\n"
    "fmla v29.4s, v2.4s, v12.4s\n"
    "fmla v28.4s, v1.4s, v12.4s\n"
    "fmla v31.4s, v5.4s, v12.4s\n"
    "fmla v30.4s, v4.4s, v12.4s\n"
    "tbz %x[n_channels], #1, 7f\n"
    "ldr d9, [x19], #0x8\n"
    "tbz %x[n_channels], #0, 8f\n"
    "ld1 { v9.s }[2], [x19]\n"
    "b 8f\n"
    "7:"  // Tile loop: Oddments: Load inputs: (3, 0): Bit 1: Unset
    "ldr s9, [x19, #0x0]\n"
    "8:"  // Tile loop: Oddments: Load inputs: (3, 0): Bit 1: End
    "fmla v29.4s, v6.4s, v9.4s\n"
    "add x19, x27, x25\n"
    "fmla v31.4s, v7.4s, v13.4s\n"
    "fmla v30.4s, v6.4s, v13.4s\n"
    "fmla v28.4s, v3.4s, v13.4s\n"
    "fmla v29.4s, v4.4s, v13.4s\n"
    "tbz %x[n_channels], #1, 9f\n"
    "ldr d11, [x19], #0x8\n"
    "tbz %x[n_channels], #0, 10f\n"
    "ld1 { v11.s }[2], [x19]\n"
    "b 10f\n"
    "9:"  // Tile loop: Oddments: Load inputs: (3, 3): Bit 1: Unset
    "ldr s11, [x19, #0x0]\n"
    "10:"  // Tile loop: Oddments: Load inputs: (3, 3): Bit 1: End
    "fmla v28.4s, v8.4s, v11.4s\n"
    "add x19, x12, x13\n"
    "tbz %x[n_channels], #1, 11f\n"
    "ldr d12, [x19], #0x8\n"
    "tbz %x[n_channels], #0, 12f\n"
    "ld1 { v12.s }[2], [x19]\n"
    "b 12f\n"
    "11:"  // Tile loop: Oddments: Load inputs: (0, 1): Bit 1: Unset
    "ldr s12, [x19, #0x0]\n"
    "12:"  // Tile loop: Oddments: Load inputs: (0, 1): Bit 1: End
    "fmla v31.4s, v1.4s, v12.4s\n"
    "add x19, x12, x26\n"
    "fmla v30.4s, v0.4s, v12.4s\n"
    "tbz %x[n_channels], #1, 13f\n"
    "ldr d9, [x19], #0x8\n"
    "tbz %x[n_channels], #0, 14f\n"
    "ld1 { v9.s }[2], [x19]\n"
    "b 14f\n"
    "13:"  // Tile loop: Oddments: Load inputs: (0, 2): Bit 1: Unset
    "ldr s9, [x19, #0x0]\n"
    "14:"  // Tile loop: Oddments: Load inputs: (0, 2): Bit 1: End
    "fmla v31.4s, v2.4s, v9.4s\n"
    "add x19, x28, x26\n"
    "fmla v30.4s, v1.4s, v9.4s\n"
    "tbz %x[n_channels], #1, 15f\n"
    "ldr d10, [x19], #0x8\n"
    "tbz %x[n_channels], #0, 16f\n"
    "ld1 { v10.s }[2], [x19]\n"
    "b 16f\n"
    "15:"  // Tile loop: Oddments: Load inputs: (2, 2): Bit 1: Unset
    "ldr s10, [x19, #0x0]\n"
    "16:"  // Tile loop: Oddments: Load inputs: (2, 2): Bit 1: End
    "fmla v31.4s, v8.4s, v10.4s\n"
    "add x19, x9, XZR\n"
    "fmla v30.4s, v7.4s, v10.4s\n"
    "fmla v29.4s, v5.4s, v10.4s\n"
    "fmla v28.4s, v4.4s, v10.4s\n"
    "tbz %x[n_channels], #1, 17f\n"
    "ldr d11, [x19], #0x8\n"
    "tbz %x[n_channels], #0, 18f\n"
    "ld1 { v11.s }[2], [x19]\n"
    "b 18f\n"
    "17:"  // Tile loop: Oddments: Load inputs: (1, 0): Bit 1: Unset
    "ldr s11, [x19, #0x0]\n"
    "18:"  // Tile loop: Oddments: Load inputs: (1, 0): Bit 1: End
    "fmla v31.4s, v3.4s, v11.4s\n"
    "add x19, x9, x25\n"
    "fmla v29.4s, v0.4s, v11.4s\n"
    "tbz %x[n_channels], #1, 19f\n"
    "ldr d12, [x19], #0x8\n"
    "tbz %x[n_channels], #0, 20f\n"
    "ld1 { v12.s }[2], [x19]\n"
    "b 20f\n"
    "19:"  // Tile loop: Oddments: Load inputs: (1, 3): Bit 1: Unset
    "ldr s12, [x19, #0x0]\n"
    "20:"  // Tile loop: Oddments: Load inputs: (1, 3): Bit 1: End
    "fmla v30.4s, v5.4s, v12.4s\n"
    "add x19, x28, XZR\n"
    "fmla v28.4s, v2.4s, v12.4s\n"
    "tbz %x[n_channels], #1, 21f\n"
    "ldr d9, [x19], #0x8\n"
    "tbz %x[n_channels], #0, 22f\n"
    "ld1 { v9.s }[2], [x19]\n"
    "b 22f\n"
    "21:"  // Tile loop: Oddments: Load inputs: (2, 0): Bit 1: Unset
    "ldr s9, [x19, #0x0]\n"
    "22:"  // Tile loop: Oddments: Load inputs: (2, 0): Bit 1: End
    "fmla v31.4s, v6.4s, v9.4s\n"
    "add x19, x28, x25\n"
    "fmla v29.4s, v3.4s, v9.4s\n"
    "tbz %x[n_channels], #1, 23f\n"
    "ldr d10, [x19], #0x8\n"
    "tbz %x[n_channels], #0, 24f\n"
    "ld1 { v10.s }[2], [x19]\n"
    "b 24f\n"
    "23:"  // Tile loop: Oddments: Load inputs: (2, 3): Bit 1: Unset
    "ldr s10, [x19, #0x0]\n"
    "24:"  // Tile loop: Oddments: Load inputs: (2, 3): Bit 1: End
    "fmla v30.4s, v8.4s, v10.4s\n"
    "add x19, x27, x13\n"
    "fmla v28.4s, v5.4s, v10.4s\n"
    "tbz %x[n_channels], #1, 25f\n"
    "ldr d11, [x19], #0x8\n"
    "tbz %x[n_channels], #0, 26f\n"
    "ld1 { v11.s }[2], [x19]\n"
    "b 26f\n"
    "25:"  // Tile loop: Oddments: Load inputs: (3, 1): Bit 1: Unset
    "ldr s11, [x19, #0x0]\n"
    "26:"  // Tile loop: Oddments: Load inputs: (3, 1): Bit 1: End
    "fmla v29.4s, v7.4s, v11.4s\n"
    "add x19, x27, x26\n"
    "fmla v28.4s, v6.4s, v11.4s\n"
    "tbz %x[n_channels], #1, 27f\n"
    "ldr d12, [x19], #0x8\n"
    "tbz %x[n_channels], #0, 28f\n"
    "ld1 { v12.s }[2], [x19]\n"
    "b 28f\n"
    "27:"  // Tile loop: Oddments: Load inputs: (3, 2): Bit 1: Unset
    "ldr s12, [x19, #0x0]\n"
    "28:"  // Tile loop: Oddments: Load inputs: (3, 2): Bit 1: End
    "fmla v29.4s, v8.4s, v12.4s\n"
    "fmla v28.4s, v7.4s, v12.4s\n"
    "fmax v31.4s, v31.4s, v18.4s\n"
    "fmax v30.4s, v30.4s, v18.4s\n"
    "fmin v31.4s, v31.4s, v17.4s\n"
    "fmax v29.4s, v29.4s, v18.4s\n"
    "fmin v30.4s, v30.4s, v17.4s\n"
    "fmax v28.4s, v28.4s, v18.4s\n"
    "fmin v29.4s, v29.4s, v17.4s\n"
    "fmin v28.4s, v28.4s, v17.4s\n"
    "tbz %x[n_channels], #1, 29f\n"
    "mov x19, x10\n"
    "st1 { v31.d }[0], [x19], x11\n"
    "add x10, x10, #0x8\n"
    "st1 { v30.d }[0], [x19]\n"
    "mov x19, x24\n"
    "st1 { v29.d }[0], [x19], x11\n"
    "add x24, x24, #0x8\n"
    "st1 { v28.d }[0], [x19]\n"
    "tbz %x[n_channels], #0, 30f\n"
    "mov x20, x10\n"
    "st1 { v31.s }[2], [x20], x11\n"
    "mov x19, x24\n"
    "st1 { v30.s }[2], [x20]\n"
    "st1 { v29.s }[2], [x19], x11\n"
    "st1 { v28.s }[2], [x19]\n"
    "b 30f\n"
    "29:"  // Tile loop: Oddments: Store: Bit 1: Unset
    "mov x20, x10\n"
    "st1 { v31.s }[0], [x20], x11\n"
    "mov x19, x24\n"
    "st1 { v30.s }[0], [x20]\n"
    "st1 { v29.s }[0], [x19], x11\n"
    "st1 { v28.s }[0], [x19]\n"
    "30:"  // Tile loop: Oddments: Store: Bit 1: End

    "31:"  // Tile loop: End
    "ldr x17, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "add x21, x17, #0x1\n"
    "ldr x16, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "ldr x20, [%x[params_struct], %[offsetof_args_n_tile_rows]]\n"
    "add x16, x16, #0x1\n"
    "ldr x19, [%x[params_struct], %[offsetof_args_n_tile_cols]]\n"
    "cmp x16, x19\n"
    "csel x16, x16, XZR, LT\n"
    "csel x17, x17, x21, LT\n"
    "cmp x17, x20\n"
    "blt 1b\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_args_inptr] "I" (offsetof(Args, inptr)), [offsetof_args_ld_input_col] "I" (offsetof(Args, ld_input_col)), [offsetof_args_ld_input_row] "I" (offsetof(Args, ld_input_row)), [offsetof_args_ld_output_col] "I" (offsetof(Args, ld_output_col)), [offsetof_args_ld_output_row] "I" (offsetof(Args, ld_output_row)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_n_tile_cols] "I" (offsetof(Args, n_tile_cols)), [offsetof_args_n_tile_rows] "I" (offsetof(Args, n_tile_rows)), [offsetof_args_outptr] "I" (offsetof(Args, outptr)), [offsetof_args_params] "I" (offsetof(Args, params)), [offsetof_args_tile_i] "I" (offsetof(Args, tile_i)), [offsetof_args_tile_j] "I" (offsetof(Args, tile_j)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v16", "v17", "v18", "v28", "v29", "v30", "v31", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
  );
}

}  // namespace depthwise
}  // namespace arm_conv
