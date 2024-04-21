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

#if defined(__aarch64__) && defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

namespace arm_conv {
namespace depthwise {

void a64_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst_direct_impl(
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
    "mov x23, #0x0\n"
    "mov x22, #0x0\n"
    "1:"  // Tile loop
    "str x23, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "mov x27, #0x2\n"
    "mov x26, #0x2\n"
    "str x22, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "ldr x25, [%x[params_struct], %[offsetof_args_ld_input_row]]\n"
    "ldr x24, [%x[params_struct], %[offsetof_args_ld_output_row]]\n"
    "mul x21, x23, x25\n"  // offset = tile_i * ld_input_row
    "ldr x15, [%x[params_struct], %[offsetof_args_ld_input_col]]\n"
    "ldr x14, [%x[params_struct], %[offsetof_args_ld_output_col]]\n"
    "mul x20, x23, x24\n"  // offset = tile_i * ld_output_row
    "mov x23, #0x10\n"  // cntb _, ALL, #1
    "madd x21, x22, x15, x21\n"  // offset += tile_j * ld_input_col
    "ldr x13, [%x[params_struct], %[offsetof_args_inptr]]\n"
    "lsl x15, x15, #0x1\n"
    "ldr x12, [%x[params_struct], %[offsetof_args_outptr]]\n"
    "madd x20, x22, x14, x20\n"  // offset += tile_j * ld_output_col
    "lsr x22, %x[n_channels], #0x3\n"
    "add x11, x15, x15\n"
    "ldr x10, [%x[params_struct], %[offsetof_args_params]]\n"
    "mul x21, x21, x27\n"  // offset *= kernel_stride * output_size
    "add x13, x13, x21, LSL #1\n"  // inptr[0] += offset * sizeof(__fp16)
    "add x9, x13, x25, LSL #1\n"
    "mul x20, x20, x26\n"  // offset *= output_tile_size
    "add x28, x9, x25, LSL #1\n"
    "add x12, x12, x20, LSL #1\n"  // outptrs[0] += offset * sizeof(__fp16)
    "add x20, %x[params_struct], %[offsetof_args_min]\n"
    "ld1r { v27.8h }, [x20]\n"
    "add x20, %x[params_struct], %[offsetof_args_max]\n"
    "ld1r { v26.8h }, [x20]\n"
    "add x27, x28, x25, LSL #1\n"
    "add x26, x11, x15\n"
    "add x25, x12, x24, LSL #1\n"
    "lsl x14, x14, #0x1\n"
    "mov x21, #0x0\n"
    "sub x20, XZR, x23\n"
    "cbz x22, 4f\n"
    "ldr q25, [x10, #0x0]\n"
    "ldr q0, [x10, #0x10]\n"
    "cmp x23, x22, LSL #4\n"
    "ldr q1, [x10, #0x20]\n"
    "ldr q2, [x10, #0x30]\n"
    "ldr q3, [x10, #0x40]\n"
    "ldr q4, [x10, #0x50]\n"
    "ldr q5, [x10, #0x60]\n"
    "ldr q6, [x10, #0x70]\n"
    "ldr q7, [x10, #0x80]\n"
    "ldr q8, [x10, #0x90]\n"
    "add x10, x10, #0xa0\n"
    "ldr q9, [x9, x15]\n"
    "ld1 { v10.8h }, [x13]\n"
    "ldr q11, [x13, x26]\n"
    "ldr q12, [x9, x11]\n"
    "ldr q13, [x28, x15]\n"
    "bge 3f\n"
    "2:"  // Tile loop: Channel loop
    "mov v24.16b, v25.16b\n fmla v24.8h, v4.8h, v9.8h\n"
    "mov v23.16b, v25.16b\n fmla v23.8h, v3.8h, v9.8h\n"
    "add x23, x23, #0x10\n"
    "cmp x23, x22, LSL #4\n"
    "mov v22.16b, v25.16b\n fmla v22.8h, v1.8h, v9.8h\n"
    "mov v21.16b, v25.16b\n fmla v21.8h, v0.8h, v9.8h\n"
    "ld1 { v18.8h }, [x27]\n"
    "ldr q25, [x10, #0x0]\n"
    "fmla v24.8h, v0.8h, v10.8h\n"
    "ldr q20, [x28, x11]\n"
    "fmla v23.8h, v2.8h, v11.8h\n"
    "ldr q17, [x27, x26]\n"
    "fmla v22.8h, v2.8h, v12.8h\n"
    "fmla v21.8h, v1.8h, v12.8h\n"
    "add x20, x20, #0x10\n"
    "add x21, x21, #0x10\n"
    "fmla v24.8h, v5.8h, v12.8h\n"
    "fmla v23.8h, v4.8h, v12.8h\n"
    "ldr q16, [x13, x15]\n"
    "fmla v22.8h, v6.8h, v18.8h\n"
    "ldr q18, [x13, x11]\n"
    "fmla v21.8h, v3.8h, v13.8h\n"
    "add x13, x13, #0x10\n"
    "fmla v24.8h, v7.8h, v13.8h\n"
    "fmla v23.8h, v6.8h, v13.8h\n"
    "fmla v22.8h, v4.8h, v13.8h\n"
    "fmla v21.8h, v8.8h, v17.8h\n"
    "ld1 { v17.8h }, [x9]\n"
    "fmla v24.8h, v1.8h, v16.8h\n"
    "fmla v23.8h, v0.8h, v16.8h\n"
    "ldr q16, [x9, x26]\n"
    "add x9, x9, #0x10\n"
    "fmla v22.8h, v5.8h, v20.8h\n"
    "fmla v21.8h, v4.8h, v20.8h\n"
    "ldr q4, [x10, #0x50]\n"
    "fmla v24.8h, v2.8h, v18.8h\n"
    "fmla v23.8h, v1.8h, v18.8h\n"
    "ld1 { v19.8h }, [x28]\n"
    "ldr q1, [x10, #0x20]\n"
    "fmla v22.8h, v0.8h, v17.8h\n"
    "ldr q0, [x10, #0x10]\n"
    "fmla v21.8h, v2.8h, v16.8h\n"
    "ldr q2, [x10, #0x30]\n"
    "fmla v24.8h, v8.8h, v20.8h\n"
    "fmla v23.8h, v7.8h, v20.8h\n"
    "ldr q18, [x28, x26]\n"
    "add x28, x28, #0x10\n"
    "ldr q13, [x28, x15]\n"
    "fmla v22.8h, v3.8h, v19.8h\n"
    "fmla v21.8h, v5.8h, v18.8h\n"
    "fmla v24.8h, v3.8h, v17.8h\n"
    "ldr q17, [x27, x15]\n"
    "ldr q3, [x10, #0x40]\n"
    "fmla v23.8h, v5.8h, v16.8h\n"
    "ldr q16, [x27, x11]\n"
    "ldr q5, [x10, #0x60]\n"
    "fmla v22.8h, v7.8h, v17.8h\n"
    "fmla v21.8h, v6.8h, v17.8h\n"
    "ldr q11, [x13, x26]\n"
    "fmla v24.8h, v6.8h, v19.8h\n"
    "ldr q9, [x9, x15]\n"
    "fmla v23.8h, v8.8h, v18.8h\n"
    "ld1 { v10.8h }, [x13]\n"
    "ldr q6, [x10, #0x70]\n"
    "fmla v22.8h, v8.8h, v16.8h\n"
    "fmla v21.8h, v7.8h, v16.8h\n"
    "ldr q12, [x9, x11]\n"
    "ldr q7, [x10, #0x80]\n"
    "fmax v24.8h, v24.8h, v27.8h\n"
    "fmax v23.8h, v23.8h, v27.8h\n"
    "ldr q8, [x10, #0x90]\n"
    "fmax v22.8h, v22.8h, v27.8h\n"
    "fmax v21.8h, v21.8h, v27.8h\n"
    "add x27, x27, #0x10\n"
    "fmin v24.8h, v24.8h, v26.8h\n"
    "fmin v23.8h, v23.8h, v26.8h\n"
    "st1 { v24.8h }, [x12]\n"
    "add x10, x10, #0xa0\n"
    "fmin v22.8h, v22.8h, v26.8h\n"
    "fmin v21.8h, v21.8h, v26.8h\n"
    "str q23, [x12, x14]\n"
    "add x12, x12, #0x10\n"
    "st1 { v22.8h }, [x25]\n"
    "str q21, [x25, x14]\n"
    "add x25, x25, #0x10\n"
    "blt 2b\n"
    "3:"  // Tile loop: Channel tail
    "mov v24.16b, v25.16b\n fmla v24.8h, v4.8h, v9.8h\n"
    "mov v23.16b, v25.16b\n fmla v23.8h, v3.8h, v9.8h\n"
    "mov v22.16b, v25.16b\n fmla v22.8h, v1.8h, v9.8h\n"
    "mov v21.16b, v25.16b\n fmla v21.8h, v0.8h, v9.8h\n"
    "ld1 { v18.8h }, [x27]\n"
    "fmla v24.8h, v0.8h, v10.8h\n"
    "ldr q20, [x28, x11]\n"
    "fmla v23.8h, v2.8h, v11.8h\n"
    "ldr q17, [x27, x26]\n"
    "fmla v22.8h, v2.8h, v12.8h\n"
    "fmla v21.8h, v1.8h, v12.8h\n"
    "fmla v24.8h, v5.8h, v12.8h\n"
    "fmla v23.8h, v4.8h, v12.8h\n"
    "ldr q16, [x13, x15]\n"
    "fmla v22.8h, v6.8h, v18.8h\n"
    "ldr q18, [x13, x11]\n"
    "fmla v21.8h, v3.8h, v13.8h\n"
    "add x13, x13, #0x10\n"
    "fmla v24.8h, v7.8h, v13.8h\n"
    "fmla v23.8h, v6.8h, v13.8h\n"
    "fmla v22.8h, v4.8h, v13.8h\n"
    "fmla v21.8h, v8.8h, v17.8h\n"
    "ld1 { v17.8h }, [x9]\n"
    "fmla v24.8h, v1.8h, v16.8h\n"
    "fmla v23.8h, v0.8h, v16.8h\n"
    "ldr q16, [x9, x26]\n"
    "add x9, x9, #0x10\n"
    "fmla v22.8h, v5.8h, v20.8h\n"
    "fmla v21.8h, v4.8h, v20.8h\n"
    "fmla v24.8h, v2.8h, v18.8h\n"
    "fmla v23.8h, v1.8h, v18.8h\n"
    "ld1 { v19.8h }, [x28]\n"
    "fmla v22.8h, v0.8h, v17.8h\n"
    "fmla v21.8h, v2.8h, v16.8h\n"
    "fmla v24.8h, v8.8h, v20.8h\n"
    "fmla v23.8h, v7.8h, v20.8h\n"
    "ldr q18, [x28, x26]\n"
    "add x28, x28, #0x10\n"
    "fmla v22.8h, v3.8h, v19.8h\n"
    "fmla v21.8h, v5.8h, v18.8h\n"
    "fmla v24.8h, v3.8h, v17.8h\n"
    "ldr q17, [x27, x15]\n"
    "fmla v23.8h, v5.8h, v16.8h\n"
    "ldr q16, [x27, x11]\n"
    "fmla v22.8h, v7.8h, v17.8h\n"
    "fmla v21.8h, v6.8h, v17.8h\n"
    "add x27, x27, #0x10\n"
    "fmla v24.8h, v6.8h, v19.8h\n"
    "fmla v23.8h, v8.8h, v18.8h\n"
    "fmax v24.8h, v24.8h, v27.8h\n"
    "fmla v22.8h, v8.8h, v16.8h\n"
    "fmla v21.8h, v7.8h, v16.8h\n"
    "fmax v23.8h, v23.8h, v27.8h\n"
    "fmax v22.8h, v22.8h, v27.8h\n"
    "fmax v21.8h, v21.8h, v27.8h\n"
    "fmin v24.8h, v24.8h, v26.8h\n"
    "fmin v23.8h, v23.8h, v26.8h\n"
    "st1 { v24.8h }, [x12]\n"
    "fmin v22.8h, v22.8h, v26.8h\n"
    "fmin v21.8h, v21.8h, v26.8h\n"
    "str q23, [x12, x14]\n"
    "add x12, x12, #0x10\n"
    "st1 { v22.8h }, [x25]\n"
    "str q21, [x25, x14]\n"
    "add x25, x25, #0x10\n"
    "4:"  // Tile loop: Oddments
    "tst %x[n_channels], #0x7\n"
    "beq 57f\n"
    "ldr q25, [x10, #0x0]\n"
    "ldr q0, [x10, #0x10]\n"
    "add x24, x9, x15\n"
    "add x23, x13, XZR\n"
    "ldr q1, [x10, #0x20]\n"
    "ldr q2, [x10, #0x30]\n"
    "add x22, x13, x26\n"
    "add x21, x9, x11\n"
    "ldr q3, [x10, #0x40]\n"
    "ldr q4, [x10, #0x50]\n"
    "add x20, x28, x15\n"
    "ldr q5, [x10, #0x60]\n"
    "ldr q6, [x10, #0x70]\n"
    "ldr q7, [x10, #0x80]\n"
    "ldr q8, [x10, #0x90]\n"
    "tbz %x[n_channels], #2, 6f\n"
    "ldr d9, [x24], #0x8\n"
    "ldr d10, [x23], #0x8\n"
    "ldr d11, [x22], #0x8\n"
    "ldr d12, [x21], #0x8\n"
    "ldr d13, [x20], #0x8\n"
    "tbz %x[n_channels], #1, 5f\n"
    "ld1 { v9.s }[2], [x24], #0x4\n"
    "ld1 { v10.s }[2], [x23], #0x4\n"
    "ld1 { v11.s }[2], [x22], #0x4\n"
    "ld1 { v12.s }[2], [x21], #0x4\n"
    "ld1 { v13.s }[2], [x20], #0x4\n"
    "tbz %x[n_channels], #0, 8f\n"
    "ld1 { v9.h }[6], [x24]\n"
    "ld1 { v10.h }[6], [x23]\n"
    "ld1 { v11.h }[6], [x22]\n"
    "ld1 { v12.h }[6], [x21]\n"
    "ld1 { v13.h }[6], [x20]\n"
    "b 8f\n"
    "5:"  // Tile loop: Oddments: Load inputs: (1, 1), (0, 0), (0, 3), (1, 2), (2, 1): Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 8f\n"
    "ld1 { v9.h }[4], [x24]\n"
    "ld1 { v10.h }[4], [x23]\n"
    "ld1 { v11.h }[4], [x22]\n"
    "ld1 { v12.h }[4], [x21]\n"
    "ld1 { v13.h }[4], [x20]\n"
    "b 8f\n"
    "6:"  // Tile loop: Oddments: Load inputs: (1, 1), (0, 0), (0, 3), (1, 2), (2, 1): Bit 2: Unset
    "tbz %x[n_channels], #1, 7f\n"
    "ldr s9, [x24], #0x4\n"
    "ldr s10, [x23], #0x4\n"
    "ldr s11, [x22], #0x4\n"
    "ldr s12, [x21], #0x4\n"
    "ldr s13, [x20], #0x4\n"
    "tbz %x[n_channels], #0, 8f\n"
    "ld1 { v9.h }[2], [x24]\n"
    "ld1 { v10.h }[2], [x23]\n"
    "ld1 { v11.h }[2], [x22]\n"
    "ld1 { v12.h }[2], [x21]\n"
    "ld1 { v13.h }[2], [x20]\n"
    "b 8f\n"
    "7:"  // Tile loop: Oddments: Load inputs: (1, 1), (0, 0), (0, 3), (1, 2), (2, 1): Bit 2: Unset: Bit 1: Unset
    "ldr h9, [x24, #0x0]\n"
    "ldr h10, [x23, #0x0]\n"
    "ldr h11, [x22, #0x0]\n"
    "ldr h12, [x21, #0x0]\n"
    "ldr h13, [x20, #0x0]\n"
    "8:"  // Tile loop: Oddments: Load inputs: (1, 1), (0, 0), (0, 3), (1, 2), (2, 1): Bit 2: End
    "mov v28.16b, v25.16b\n fmla v28.8h, v4.8h, v9.8h\n"
    "mov v29.16b, v25.16b\n fmla v29.8h, v3.8h, v9.8h\n"
    "add x20, x27, XZR\n"
    "mov v30.16b, v25.16b\n fmla v30.8h, v1.8h, v9.8h\n"
    "mov v31.16b, v25.16b\n fmla v31.8h, v0.8h, v9.8h\n"
    "fmla v28.8h, v0.8h, v10.8h\n"
    "fmla v29.8h, v2.8h, v11.8h\n"
    "fmla v28.8h, v5.8h, v12.8h\n"
    "fmla v29.8h, v4.8h, v12.8h\n"
    "fmla v30.8h, v2.8h, v12.8h\n"
    "fmla v31.8h, v1.8h, v12.8h\n"
    "tbz %x[n_channels], #2, 10f\n"
    "ldr d9, [x20], #0x8\n"
    "tbz %x[n_channels], #1, 9f\n"
    "ld1 { v9.s }[2], [x20], #0x4\n"
    "tbz %x[n_channels], #0, 12f\n"
    "ld1 { v9.h }[6], [x20]\n"
    "b 12f\n"
    "9:"  // Tile loop: Oddments: Load inputs: (3, 0): Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 12f\n"
    "ld1 { v9.h }[4], [x20]\n"
    "b 12f\n"
    "10:"  // Tile loop: Oddments: Load inputs: (3, 0): Bit 2: Unset
    "tbz %x[n_channels], #1, 11f\n"
    "ldr s9, [x20], #0x4\n"
    "tbz %x[n_channels], #0, 12f\n"
    "ld1 { v9.h }[2], [x20]\n"
    "b 12f\n"
    "11:"  // Tile loop: Oddments: Load inputs: (3, 0): Bit 2: Unset: Bit 1: Unset
    "ldr h9, [x20, #0x0]\n"
    "12:"  // Tile loop: Oddments: Load inputs: (3, 0): Bit 2: End
    "fmla v30.8h, v6.8h, v9.8h\n"
    "fmla v28.8h, v7.8h, v13.8h\n"
    "add x20, x27, x26\n"
    "fmla v29.8h, v6.8h, v13.8h\n"
    "fmla v30.8h, v4.8h, v13.8h\n"
    "fmla v31.8h, v3.8h, v13.8h\n"
    "tbz %x[n_channels], #2, 14f\n"
    "ldr d11, [x20], #0x8\n"
    "tbz %x[n_channels], #1, 13f\n"
    "ld1 { v11.s }[2], [x20], #0x4\n"
    "tbz %x[n_channels], #0, 16f\n"
    "ld1 { v11.h }[6], [x20]\n"
    "b 16f\n"
    "13:"  // Tile loop: Oddments: Load inputs: (3, 3): Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 16f\n"
    "ld1 { v11.h }[4], [x20]\n"
    "b 16f\n"
    "14:"  // Tile loop: Oddments: Load inputs: (3, 3): Bit 2: Unset
    "tbz %x[n_channels], #1, 15f\n"
    "ldr s11, [x20], #0x4\n"
    "tbz %x[n_channels], #0, 16f\n"
    "ld1 { v11.h }[2], [x20]\n"
    "b 16f\n"
    "15:"  // Tile loop: Oddments: Load inputs: (3, 3): Bit 2: Unset: Bit 1: Unset
    "ldr h11, [x20, #0x0]\n"
    "16:"  // Tile loop: Oddments: Load inputs: (3, 3): Bit 2: End
    "fmla v31.8h, v8.8h, v11.8h\n"
    "add x20, x13, x15\n"
    "tbz %x[n_channels], #2, 18f\n"
    "ldr d12, [x20], #0x8\n"
    "tbz %x[n_channels], #1, 17f\n"
    "ld1 { v12.s }[2], [x20], #0x4\n"
    "tbz %x[n_channels], #0, 20f\n"
    "ld1 { v12.h }[6], [x20]\n"
    "b 20f\n"
    "17:"  // Tile loop: Oddments: Load inputs: (0, 1): Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 20f\n"
    "ld1 { v12.h }[4], [x20]\n"
    "b 20f\n"
    "18:"  // Tile loop: Oddments: Load inputs: (0, 1): Bit 2: Unset
    "tbz %x[n_channels], #1, 19f\n"
    "ldr s12, [x20], #0x4\n"
    "tbz %x[n_channels], #0, 20f\n"
    "ld1 { v12.h }[2], [x20]\n"
    "b 20f\n"
    "19:"  // Tile loop: Oddments: Load inputs: (0, 1): Bit 2: Unset: Bit 1: Unset
    "ldr h12, [x20, #0x0]\n"
    "20:"  // Tile loop: Oddments: Load inputs: (0, 1): Bit 2: End
    "fmla v28.8h, v1.8h, v12.8h\n"
    "fmla v29.8h, v0.8h, v12.8h\n"
    "add x20, x13, x11\n"
    "tbz %x[n_channels], #2, 22f\n"
    "ldr d9, [x20], #0x8\n"
    "tbz %x[n_channels], #1, 21f\n"
    "ld1 { v9.s }[2], [x20], #0x4\n"
    "tbz %x[n_channels], #0, 24f\n"
    "ld1 { v9.h }[6], [x20]\n"
    "b 24f\n"
    "21:"  // Tile loop: Oddments: Load inputs: (0, 2): Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 24f\n"
    "ld1 { v9.h }[4], [x20]\n"
    "b 24f\n"
    "22:"  // Tile loop: Oddments: Load inputs: (0, 2): Bit 2: Unset
    "tbz %x[n_channels], #1, 23f\n"
    "ldr s9, [x20], #0x4\n"
    "tbz %x[n_channels], #0, 24f\n"
    "ld1 { v9.h }[2], [x20]\n"
    "b 24f\n"
    "23:"  // Tile loop: Oddments: Load inputs: (0, 2): Bit 2: Unset: Bit 1: Unset
    "ldr h9, [x20, #0x0]\n"
    "24:"  // Tile loop: Oddments: Load inputs: (0, 2): Bit 2: End
    "fmla v28.8h, v2.8h, v9.8h\n"
    "fmla v29.8h, v1.8h, v9.8h\n"
    "add x20, x28, x11\n"
    "tbz %x[n_channels], #2, 26f\n"
    "ldr d10, [x20], #0x8\n"
    "tbz %x[n_channels], #1, 25f\n"
    "ld1 { v10.s }[2], [x20], #0x4\n"
    "tbz %x[n_channels], #0, 28f\n"
    "ld1 { v10.h }[6], [x20]\n"
    "b 28f\n"
    "25:"  // Tile loop: Oddments: Load inputs: (2, 2): Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 28f\n"
    "ld1 { v10.h }[4], [x20]\n"
    "b 28f\n"
    "26:"  // Tile loop: Oddments: Load inputs: (2, 2): Bit 2: Unset
    "tbz %x[n_channels], #1, 27f\n"
    "ldr s10, [x20], #0x4\n"
    "tbz %x[n_channels], #0, 28f\n"
    "ld1 { v10.h }[2], [x20]\n"
    "b 28f\n"
    "27:"  // Tile loop: Oddments: Load inputs: (2, 2): Bit 2: Unset: Bit 1: Unset
    "ldr h10, [x20, #0x0]\n"
    "28:"  // Tile loop: Oddments: Load inputs: (2, 2): Bit 2: End
    "fmla v28.8h, v8.8h, v10.8h\n"
    "fmla v29.8h, v7.8h, v10.8h\n"
    "add x20, x9, XZR\n"
    "fmla v30.8h, v5.8h, v10.8h\n"
    "fmla v31.8h, v4.8h, v10.8h\n"
    "tbz %x[n_channels], #2, 30f\n"
    "ldr d11, [x20], #0x8\n"
    "tbz %x[n_channels], #1, 29f\n"
    "ld1 { v11.s }[2], [x20], #0x4\n"
    "tbz %x[n_channels], #0, 32f\n"
    "ld1 { v11.h }[6], [x20]\n"
    "b 32f\n"
    "29:"  // Tile loop: Oddments: Load inputs: (1, 0): Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 32f\n"
    "ld1 { v11.h }[4], [x20]\n"
    "b 32f\n"
    "30:"  // Tile loop: Oddments: Load inputs: (1, 0): Bit 2: Unset
    "tbz %x[n_channels], #1, 31f\n"
    "ldr s11, [x20], #0x4\n"
    "tbz %x[n_channels], #0, 32f\n"
    "ld1 { v11.h }[2], [x20]\n"
    "b 32f\n"
    "31:"  // Tile loop: Oddments: Load inputs: (1, 0): Bit 2: Unset: Bit 1: Unset
    "ldr h11, [x20, #0x0]\n"
    "32:"  // Tile loop: Oddments: Load inputs: (1, 0): Bit 2: End
    "fmla v28.8h, v3.8h, v11.8h\n"
    "fmla v30.8h, v0.8h, v11.8h\n"
    "add x20, x9, x26\n"
    "tbz %x[n_channels], #2, 34f\n"
    "ldr d12, [x20], #0x8\n"
    "tbz %x[n_channels], #1, 33f\n"
    "ld1 { v12.s }[2], [x20], #0x4\n"
    "tbz %x[n_channels], #0, 36f\n"
    "ld1 { v12.h }[6], [x20]\n"
    "b 36f\n"
    "33:"  // Tile loop: Oddments: Load inputs: (1, 3): Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 36f\n"
    "ld1 { v12.h }[4], [x20]\n"
    "b 36f\n"
    "34:"  // Tile loop: Oddments: Load inputs: (1, 3): Bit 2: Unset
    "tbz %x[n_channels], #1, 35f\n"
    "ldr s12, [x20], #0x4\n"
    "tbz %x[n_channels], #0, 36f\n"
    "ld1 { v12.h }[2], [x20]\n"
    "b 36f\n"
    "35:"  // Tile loop: Oddments: Load inputs: (1, 3): Bit 2: Unset: Bit 1: Unset
    "ldr h12, [x20, #0x0]\n"
    "36:"  // Tile loop: Oddments: Load inputs: (1, 3): Bit 2: End
    "fmla v29.8h, v5.8h, v12.8h\n"
    "fmla v31.8h, v2.8h, v12.8h\n"
    "add x20, x28, XZR\n"
    "tbz %x[n_channels], #2, 38f\n"
    "ldr d9, [x20], #0x8\n"
    "tbz %x[n_channels], #1, 37f\n"
    "ld1 { v9.s }[2], [x20], #0x4\n"
    "tbz %x[n_channels], #0, 40f\n"
    "ld1 { v9.h }[6], [x20]\n"
    "b 40f\n"
    "37:"  // Tile loop: Oddments: Load inputs: (2, 0): Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 40f\n"
    "ld1 { v9.h }[4], [x20]\n"
    "b 40f\n"
    "38:"  // Tile loop: Oddments: Load inputs: (2, 0): Bit 2: Unset
    "tbz %x[n_channels], #1, 39f\n"
    "ldr s9, [x20], #0x4\n"
    "tbz %x[n_channels], #0, 40f\n"
    "ld1 { v9.h }[2], [x20]\n"
    "b 40f\n"
    "39:"  // Tile loop: Oddments: Load inputs: (2, 0): Bit 2: Unset: Bit 1: Unset
    "ldr h9, [x20, #0x0]\n"
    "40:"  // Tile loop: Oddments: Load inputs: (2, 0): Bit 2: End
    "fmla v28.8h, v6.8h, v9.8h\n"
    "fmla v30.8h, v3.8h, v9.8h\n"
    "add x20, x28, x26\n"
    "tbz %x[n_channels], #2, 42f\n"
    "ldr d10, [x20], #0x8\n"
    "tbz %x[n_channels], #1, 41f\n"
    "ld1 { v10.s }[2], [x20], #0x4\n"
    "tbz %x[n_channels], #0, 44f\n"
    "ld1 { v10.h }[6], [x20]\n"
    "b 44f\n"
    "41:"  // Tile loop: Oddments: Load inputs: (2, 3): Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 44f\n"
    "ld1 { v10.h }[4], [x20]\n"
    "b 44f\n"
    "42:"  // Tile loop: Oddments: Load inputs: (2, 3): Bit 2: Unset
    "tbz %x[n_channels], #1, 43f\n"
    "ldr s10, [x20], #0x4\n"
    "tbz %x[n_channels], #0, 44f\n"
    "ld1 { v10.h }[2], [x20]\n"
    "b 44f\n"
    "43:"  // Tile loop: Oddments: Load inputs: (2, 3): Bit 2: Unset: Bit 1: Unset
    "ldr h10, [x20, #0x0]\n"
    "44:"  // Tile loop: Oddments: Load inputs: (2, 3): Bit 2: End
    "fmla v29.8h, v8.8h, v10.8h\n"
    "fmla v31.8h, v5.8h, v10.8h\n"
    "add x20, x27, x15\n"
    "tbz %x[n_channels], #2, 46f\n"
    "ldr d11, [x20], #0x8\n"
    "tbz %x[n_channels], #1, 45f\n"
    "ld1 { v11.s }[2], [x20], #0x4\n"
    "tbz %x[n_channels], #0, 48f\n"
    "ld1 { v11.h }[6], [x20]\n"
    "b 48f\n"
    "45:"  // Tile loop: Oddments: Load inputs: (3, 1): Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 48f\n"
    "ld1 { v11.h }[4], [x20]\n"
    "b 48f\n"
    "46:"  // Tile loop: Oddments: Load inputs: (3, 1): Bit 2: Unset
    "tbz %x[n_channels], #1, 47f\n"
    "ldr s11, [x20], #0x4\n"
    "tbz %x[n_channels], #0, 48f\n"
    "ld1 { v11.h }[2], [x20]\n"
    "b 48f\n"
    "47:"  // Tile loop: Oddments: Load inputs: (3, 1): Bit 2: Unset: Bit 1: Unset
    "ldr h11, [x20, #0x0]\n"
    "48:"  // Tile loop: Oddments: Load inputs: (3, 1): Bit 2: End
    "fmla v30.8h, v7.8h, v11.8h\n"
    "fmla v31.8h, v6.8h, v11.8h\n"
    "add x20, x27, x11\n"
    "tbz %x[n_channels], #2, 50f\n"
    "ldr d12, [x20], #0x8\n"
    "tbz %x[n_channels], #1, 49f\n"
    "ld1 { v12.s }[2], [x20], #0x4\n"
    "tbz %x[n_channels], #0, 52f\n"
    "ld1 { v12.h }[6], [x20]\n"
    "b 52f\n"
    "49:"  // Tile loop: Oddments: Load inputs: (3, 2): Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 52f\n"
    "ld1 { v12.h }[4], [x20]\n"
    "b 52f\n"
    "50:"  // Tile loop: Oddments: Load inputs: (3, 2): Bit 2: Unset
    "tbz %x[n_channels], #1, 51f\n"
    "ldr s12, [x20], #0x4\n"
    "tbz %x[n_channels], #0, 52f\n"
    "ld1 { v12.h }[2], [x20]\n"
    "b 52f\n"
    "51:"  // Tile loop: Oddments: Load inputs: (3, 2): Bit 2: Unset: Bit 1: Unset
    "ldr h12, [x20, #0x0]\n"
    "52:"  // Tile loop: Oddments: Load inputs: (3, 2): Bit 2: End
    "fmla v30.8h, v8.8h, v12.8h\n"
    "fmla v31.8h, v7.8h, v12.8h\n"
    "fmax v28.8h, v28.8h, v27.8h\n"
    "fmax v29.8h, v29.8h, v27.8h\n"
    "fmax v30.8h, v30.8h, v27.8h\n"
    "fmax v31.8h, v31.8h, v27.8h\n"
    "fmin v28.8h, v28.8h, v26.8h\n"
    "fmin v29.8h, v29.8h, v26.8h\n"
    "fmin v30.8h, v30.8h, v26.8h\n"
    "fmin v31.8h, v31.8h, v26.8h\n"
    "tbz %x[n_channels], #2, 54f\n"
    "mov x21, x12\n"
    "mov x20, x25\n"
    "st1 { v28.d }[0], [x21], x14\n"
    "st1 { v30.d }[0], [x20], x14\n"
    "add x12, x12, #0x8\n"
    "add x25, x25, #0x8\n"
    "st1 { v29.d }[0], [x21]\n"
    "st1 { v31.d }[0], [x20]\n"
    "tbz %x[n_channels], #1, 53f\n"
    "mov x21, x12\n"
    "mov x20, x25\n"
    "st1 { v28.s }[2], [x21], x14\n"
    "st1 { v30.s }[2], [x20], x14\n"
    "add x12, x12, #0x4\n"
    "add x25, x25, #0x4\n"
    "st1 { v29.s }[2], [x21]\n"
    "st1 { v31.s }[2], [x20]\n"
    "tbz %x[n_channels], #0, 56f\n"
    "mov x21, x12\n"
    "mov x20, x25\n"
    "st1 { v28.h }[6], [x21], x14\n"
    "st1 { v30.h }[6], [x20], x14\n"
    "st1 { v29.h }[6], [x21]\n"
    "st1 { v31.h }[6], [x20]\n"
    "b 56f\n"
    "53:"  // Tile loop: Oddments: Store: Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 56f\n"
    "mov x21, x12\n"
    "mov x20, x25\n"
    "st1 { v28.h }[4], [x21], x14\n"
    "st1 { v30.h }[4], [x20], x14\n"
    "st1 { v29.h }[4], [x21]\n"
    "st1 { v31.h }[4], [x20]\n"
    "b 56f\n"
    "54:"  // Tile loop: Oddments: Store: Bit 2: Unset
    "tbz %x[n_channels], #1, 55f\n"
    "mov x21, x12\n"
    "mov x20, x25\n"
    "st1 { v28.s }[0], [x21], x14\n"
    "st1 { v30.s }[0], [x20], x14\n"
    "add x12, x12, #0x4\n"
    "add x25, x25, #0x4\n"
    "st1 { v29.s }[0], [x21]\n"
    "st1 { v31.s }[0], [x20]\n"
    "tbz %x[n_channels], #0, 56f\n"
    "mov x21, x12\n"
    "mov x20, x25\n"
    "st1 { v28.h }[2], [x21], x14\n"
    "st1 { v30.h }[2], [x20], x14\n"
    "st1 { v29.h }[2], [x21]\n"
    "st1 { v31.h }[2], [x20]\n"
    "b 56f\n"
    "55:"  // Tile loop: Oddments: Store: Bit 2: Unset: Bit 1: Unset
    "mov x21, x12\n"
    "mov x20, x25\n"
    "st1 { v28.h }[0], [x21], x14\n"
    "st1 { v30.h }[0], [x20], x14\n"
    "st1 { v29.h }[0], [x21]\n"
    "st1 { v31.h }[0], [x20]\n"
    "56:"  // Tile loop: Oddments: Store: Bit 2: End
    "57:"  // Tile loop: End
    "ldr x22, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "ldr x23, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "add x22, x22, #0x1\n"
    "add x21, x23, #0x1\n"
    "ldr x20, [%x[params_struct], %[offsetof_args_n_tile_cols]]\n"
    "cmp x22, x20\n"
    "ldr x20, [%x[params_struct], %[offsetof_args_n_tile_rows]]\n"
    "csel x23, x23, x21, LT\n"
    "csel x22, x22, XZR, LT\n"
    "cmp x23, x20\n"
    "blt 1b\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_args_inptr] "I" (offsetof(Args, inptr)), [offsetof_args_ld_input_col] "I" (offsetof(Args, ld_input_col)), [offsetof_args_ld_input_row] "I" (offsetof(Args, ld_input_row)), [offsetof_args_ld_output_col] "I" (offsetof(Args, ld_output_col)), [offsetof_args_ld_output_row] "I" (offsetof(Args, ld_output_row)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_n_tile_cols] "I" (offsetof(Args, n_tile_cols)), [offsetof_args_n_tile_rows] "I" (offsetof(Args, n_tile_rows)), [offsetof_args_outptr] "I" (offsetof(Args, outptr)), [offsetof_args_params] "I" (offsetof(Args, params)), [offsetof_args_tile_i] "I" (offsetof(Args, tile_i)), [offsetof_args_tile_j] "I" (offsetof(Args, tile_j)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(__aarch64__) && defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
