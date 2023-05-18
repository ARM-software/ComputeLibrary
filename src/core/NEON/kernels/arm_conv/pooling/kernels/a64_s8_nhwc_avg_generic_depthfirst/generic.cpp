/*
 * Copyright (c) 2021-2023 Arm Limited.
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

#if defined(__aarch64__)

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>


namespace arm_conv {
namespace pooling {

namespace {
  struct RescaleParams
  {
    int32_t multiplier, shift;
  };

  constexpr RescaleParams rescale_params[8] = {
    {0x40000000, -0},  // 1/2
    {0x55555556, -1},  // 1/3
    {0x40000000, -1},  // 1/4
    {0x66666666, -2},  // 1/5
    {0x55555556, -2},  // 1/6
    {0x49249249, -2},  // 1/7
    {0x40000000, -2},  // 1/8
    {0x71c71c72, -3},  // 1/9
  };
}

void a64_s8_nhwc_avg_generic_depthfirst_impl(
  const uint64_t window_cells,
  const uint64_t n_valid_cells,
  uint64_t n_channels,
  const int8_t *const *const inptrs,
  int8_t *outptr
)
{
  if (n_valid_cells == 1 && window_cells == 1)
  {
    // In this case, simply copy from the input to the output
    std::memcpy(outptr, *inptrs, n_channels);
    return;
  }

  // Compute (or look up) the rescale values
  int32_t shift_value = 0, rescale_value = 0;
  if (2 <= window_cells && window_cells <= 9)
  {
    auto &params = rescale_params[window_cells - 2];
    rescale_value = params.multiplier;
    shift_value = params.shift;
  }
  else
  {
    auto f_rescale_value = 1.0f / static_cast<float>(window_cells);

    shift_value = 0;
    while (f_rescale_value < 0.5f)
    {
      shift_value--;
      f_rescale_value *= 2.0f;
    }

    int64_t long_rescale_value = round(f_rescale_value * static_cast<float>(1ll << 31));
    if (long_rescale_value == (1ll << 31))
    {
      shift_value++;
      long_rescale_value >>= 1;
    }
    rescale_value = static_cast<int32_t>(long_rescale_value);
  }

  __asm__ __volatile__(
    "cmp %x[n_channels], #0x40\n"
    "mov x27, #0x0\n"
    "mov x26, #0x10\n"  // cntb _, ALL, #1
    "mov x25, #0x20\n"  // cntb _, ALL, #2
    "mov x24, #0x30\n"  // cntb _, ALL, #3
    "blt 7f\n"
    "1:"  // 4-vectors of channels
    "lsr x23, %x[n_valid_cells], #0x1\n"
    "movi v15.4s, #0x0\n"
    "movi v14.4s, #0x0\n"
    "mov x20, %x[inptrs]\n"
    "movi v13.4s, #0x0\n"
    "movi v12.4s, #0x0\n"
    "movi v11.4s, #0x0\n"
    "movi v10.4s, #0x0\n"
    "movi v9.4s, #0x0\n"
    "movi v8.4s, #0x0\n"
    "movi v7.4s, #0x0\n"
    "movi v6.4s, #0x0\n"
    "movi v5.4s, #0x0\n"
    "movi v4.4s, #0x0\n"
    "movi v3.4s, #0x0\n"
    "movi v2.4s, #0x0\n"
    "movi v1.4s, #0x0\n"
    "movi v0.4s, #0x0\n"
    "cbz x23, 4f\n"
    "ldp x22, x21, [x20, #0x0]\n"
    "ldr q31, [x22, x27]\n"
    "subs x23, x23, #0x1\n"
    "add x20, x20, #0x10\n"
    "ldr q30, [x21, x27]\n"
    "ldr q29, [x22, x26]\n"
    "ldr q28, [x21, x26]\n"
    "ldr q27, [x22, x25]\n"
    "ldr q26, [x21, x25]\n"
    "ldr q25, [x22, x24]\n"
    "ldr q24, [x21, x24]\n"
    "beq 3f\n"
    "2:"  // 4-vectors of channels: 2 inputs loop
    "saddl v23.8h, v31.8b, v30.8b\n"
    "saddl2 v22.8h, v31.16b, v30.16b\n"
    "ldp x22, x21, [x20, #0x0]\n"
    "ldr q31, [x22, x27]\n"
    "ldr q30, [x21, x27]\n"
    "saddl v21.8h, v29.8b, v28.8b\n"
    "saddl2 v20.8h, v29.16b, v28.16b\n"
    "ldr q29, [x22, x26]\n"
    "ldr q28, [x21, x26]\n"
    "saddl v19.8h, v27.8b, v26.8b\n"
    "saddl2 v18.8h, v27.16b, v26.16b\n"
    "ldr q27, [x22, x25]\n"
    "ldr q26, [x21, x25]\n"
    "subs x23, x23, #0x1\n"
    "saddw v15.4s, v15.4s, v23.4h\n"
    "saddw2 v14.4s, v14.4s, v23.8h\n"
    "saddl v17.8h, v25.8b, v24.8b\n"
    "saddl2 v16.8h, v25.16b, v24.16b\n"
    "ldr q25, [x22, x24]\n"
    "add x20, x20, #0x10\n"
    "saddw v13.4s, v13.4s, v22.4h\n"
    "saddw2 v12.4s, v12.4s, v22.8h\n"
    "ldr q24, [x21, x24]\n"
    "saddw v11.4s, v11.4s, v21.4h\n"
    "saddw2 v10.4s, v10.4s, v21.8h\n"
    "saddw v9.4s, v9.4s, v20.4h\n"
    "saddw2 v8.4s, v8.4s, v20.8h\n"
    "saddw v7.4s, v7.4s, v19.4h\n"
    "saddw2 v6.4s, v6.4s, v19.8h\n"
    "saddw v5.4s, v5.4s, v18.4h\n"
    "saddw2 v4.4s, v4.4s, v18.8h\n"
    "saddw v3.4s, v3.4s, v17.4h\n"
    "saddw2 v2.4s, v2.4s, v17.8h\n"
    "saddw v1.4s, v1.4s, v16.4h\n"
    "saddw2 v0.4s, v0.4s, v16.8h\n"
    "bgt 2b\n"
    "3:"  // 4-vectors of channels: 2 inputs tail
    "saddl v23.8h, v31.8b, v30.8b\n"
    "saddl2 v22.8h, v31.16b, v30.16b\n"
    "saddl v21.8h, v29.8b, v28.8b\n"
    "saddl2 v20.8h, v29.16b, v28.16b\n"
    "saddl v19.8h, v27.8b, v26.8b\n"
    "saddl2 v18.8h, v27.16b, v26.16b\n"
    "saddl v17.8h, v25.8b, v24.8b\n"
    "saddl2 v16.8h, v25.16b, v24.16b\n"
    "saddw v15.4s, v15.4s, v23.4h\n"
    "saddw2 v14.4s, v14.4s, v23.8h\n"
    "saddw v13.4s, v13.4s, v22.4h\n"
    "saddw2 v12.4s, v12.4s, v22.8h\n"
    "saddw v11.4s, v11.4s, v21.4h\n"
    "saddw2 v10.4s, v10.4s, v21.8h\n"
    "saddw v9.4s, v9.4s, v20.4h\n"
    "saddw2 v8.4s, v8.4s, v20.8h\n"
    "saddw v7.4s, v7.4s, v19.4h\n"
    "saddw2 v6.4s, v6.4s, v19.8h\n"
    "saddw v5.4s, v5.4s, v18.4h\n"
    "saddw2 v4.4s, v4.4s, v18.8h\n"
    "saddw v3.4s, v3.4s, v17.4h\n"
    "saddw2 v2.4s, v2.4s, v17.8h\n"
    "saddw v1.4s, v1.4s, v16.4h\n"
    "saddw2 v0.4s, v0.4s, v16.8h\n"
    "4:"  // 4-vectors of channels: After loop
    "ands x21, %x[n_valid_cells], #0x1\n"
    "beq 6f\n"
    "5:"  // 4-vectors of channels: Single input loop
    "ldr x22, [x20], #0x8\n"
    "ldr q31, [x22, x27]\n"
    "sxtl v23.8h, v31.8b\n"
    "sxtl2 v22.8h, v31.16b\n"
    "ldr q29, [x22, x26]\n"
    "ldr q27, [x22, x25]\n"
    "sxtl v21.8h, v29.8b\n"
    "sxtl2 v20.8h, v29.16b\n"
    "ldr q25, [x22, x24]\n"
    "sxtl v19.8h, v27.8b\n"
    "sxtl2 v18.8h, v27.16b\n"
    "subs x21, x21, #0x1\n"
    "sxtl v17.8h, v25.8b\n"
    "sxtl2 v16.8h, v25.16b\n"
    "saddw v15.4s, v15.4s, v23.4h\n"
    "saddw2 v14.4s, v14.4s, v23.8h\n"
    "saddw v13.4s, v13.4s, v22.4h\n"
    "saddw2 v12.4s, v12.4s, v22.8h\n"
    "saddw v11.4s, v11.4s, v21.4h\n"
    "saddw2 v10.4s, v10.4s, v21.8h\n"
    "saddw v9.4s, v9.4s, v20.4h\n"
    "saddw2 v8.4s, v8.4s, v20.8h\n"
    "saddw v7.4s, v7.4s, v19.4h\n"
    "saddw2 v6.4s, v6.4s, v19.8h\n"
    "saddw v5.4s, v5.4s, v18.4h\n"
    "saddw2 v4.4s, v4.4s, v18.8h\n"
    "saddw v3.4s, v3.4s, v17.4h\n"
    "saddw2 v2.4s, v2.4s, v17.8h\n"
    "saddw v1.4s, v1.4s, v16.4h\n"
    "saddw2 v0.4s, v0.4s, v16.8h\n"
    "bgt 5b\n"
    "6:"  // 4-vectors of channels: Single input loop: End
    "ld1r { v17.4s }, [%x[rescale_ptr]]\n"
    "ld1r { v16.4s }, [%x[shift_ptr]]\n"
    "sqdmulh v15.4s, v15.4s, v17.4s\n"
    "sqdmulh v14.4s, v14.4s, v17.4s\n"
    "sqdmulh v13.4s, v13.4s, v17.4s\n"
    "sqdmulh v12.4s, v12.4s, v17.4s\n"
    "sub %x[n_channels], %x[n_channels], #0x40\n"
    "cmp %x[n_channels], #0x40\n"
    "sqdmulh v11.4s, v11.4s, v17.4s\n"
    "sqdmulh v10.4s, v10.4s, v17.4s\n"
    "sqdmulh v9.4s, v9.4s, v17.4s\n"
    "sqdmulh v8.4s, v8.4s, v17.4s\n"
    "sqdmulh v7.4s, v7.4s, v17.4s\n"
    "sqdmulh v6.4s, v6.4s, v17.4s\n"
    "sqdmulh v5.4s, v5.4s, v17.4s\n"
    "sqdmulh v4.4s, v4.4s, v17.4s\n"
    "sqdmulh v3.4s, v3.4s, v17.4s\n"
    "sqdmulh v2.4s, v2.4s, v17.4s\n"
    "sqdmulh v1.4s, v1.4s, v17.4s\n"
    "sqdmulh v0.4s, v0.4s, v17.4s\n"
    "movi v17.4s, #0x7f\n"
    "srshl v15.4s, v15.4s, v16.4s\n"
    "srshl v14.4s, v14.4s, v16.4s\n"
    "srshl v13.4s, v13.4s, v16.4s\n"
    "srshl v12.4s, v12.4s, v16.4s\n"
    "srshl v11.4s, v11.4s, v16.4s\n"
    "srshl v10.4s, v10.4s, v16.4s\n"
    "srshl v9.4s, v9.4s, v16.4s\n"
    "srshl v8.4s, v8.4s, v16.4s\n"
    "srshl v7.4s, v7.4s, v16.4s\n"
    "srshl v6.4s, v6.4s, v16.4s\n"
    "srshl v5.4s, v5.4s, v16.4s\n"
    "srshl v4.4s, v4.4s, v16.4s\n"
    "srshl v3.4s, v3.4s, v16.4s\n"
    "srshl v2.4s, v2.4s, v16.4s\n"
    "srshl v1.4s, v1.4s, v16.4s\n"
    "srshl v0.4s, v0.4s, v16.4s\n"
    "not v16.16b, v17.16b\n"
    "smax v15.4s, v15.4s, v16.4s\n"
    "smax v14.4s, v14.4s, v16.4s\n"
    "smax v13.4s, v13.4s, v16.4s\n"
    "smax v12.4s, v12.4s, v16.4s\n"
    "smax v11.4s, v11.4s, v16.4s\n"
    "smax v10.4s, v10.4s, v16.4s\n"
    "smax v9.4s, v9.4s, v16.4s\n"
    "smax v8.4s, v8.4s, v16.4s\n"
    "smax v7.4s, v7.4s, v16.4s\n"
    "smax v6.4s, v6.4s, v16.4s\n"
    "smax v5.4s, v5.4s, v16.4s\n"
    "smax v4.4s, v4.4s, v16.4s\n"
    "smax v3.4s, v3.4s, v16.4s\n"
    "smax v2.4s, v2.4s, v16.4s\n"
    "smax v1.4s, v1.4s, v16.4s\n"
    "smax v0.4s, v0.4s, v16.4s\n"
    "smin v15.4s, v15.4s, v17.4s\n"
    "smin v14.4s, v14.4s, v17.4s\n"
    "smin v13.4s, v13.4s, v17.4s\n"
    "smin v12.4s, v12.4s, v17.4s\n"
    "smin v11.4s, v11.4s, v17.4s\n"
    "smin v10.4s, v10.4s, v17.4s\n"
    "smin v9.4s, v9.4s, v17.4s\n"
    "smin v8.4s, v8.4s, v17.4s\n"
    "smin v7.4s, v7.4s, v17.4s\n"
    "smin v6.4s, v6.4s, v17.4s\n"
    "smin v5.4s, v5.4s, v17.4s\n"
    "smin v4.4s, v4.4s, v17.4s\n"
    "smin v3.4s, v3.4s, v17.4s\n"
    "smin v2.4s, v2.4s, v17.4s\n"
    "smin v1.4s, v1.4s, v17.4s\n"
    "smin v0.4s, v0.4s, v17.4s\n"
    "uzp1 v23.16b, v15.16b, v14.16b\n"
    "uzp1 v16.16b, v13.16b, v12.16b\n"
    "uzp1 v22.16b, v11.16b, v10.16b\n"
    "uzp1 v18.16b, v9.16b, v8.16b\n"
    "uzp1 v21.16b, v7.16b, v6.16b\n"
    "uzp1 v17.16b, v5.16b, v4.16b\n"
    "uzp1 v20.16b, v3.16b, v2.16b\n"
    "uzp1 v19.16b, v1.16b, v0.16b\n"
    "uzp1 v16.16b, v23.16b, v16.16b\n"
    "uzp1 v18.16b, v22.16b, v18.16b\n"
    "str q16, [%x[outptr], x27]\n"
    "add x27, x27, #0x40\n"
    "uzp1 v17.16b, v21.16b, v17.16b\n"
    "uzp1 v16.16b, v20.16b, v19.16b\n"
    "str q18, [%x[outptr], x26]\n"
    "add x26, x26, #0x40\n"
    "str q17, [%x[outptr], x25]\n"
    "add x25, x25, #0x40\n"
    "str q16, [%x[outptr], x24]\n"
    "add x24, x24, #0x40\n"
    "bge 1b\n"
    "cbz %x[n_channels], 43f\n"
    "7:"  // Single vector of channels
    "cmp %x[n_channels], #0x10\n"
    "blt 14f\n"
    "8:"  // Single vector of channels: Loop
    "lsr x23, %x[n_valid_cells], #0x1\n"
    "movi v15.4s, #0x0\n"
    "movi v14.4s, #0x0\n"
    "mov x20, %x[inptrs]\n"
    "movi v13.4s, #0x0\n"
    "movi v12.4s, #0x0\n"
    "cbz x23, 11f\n"
    "ldp x22, x21, [x20, #0x0]\n"
    "ldr q31, [x22, x27]\n"
    "subs x23, x23, #0x1\n"
    "add x20, x20, #0x10\n"
    "ldr q30, [x21, x27]\n"
    "beq 10f\n"
    "9:"  // Single vector of channels: Loop: 2 inputs loop
    "saddl v23.8h, v31.8b, v30.8b\n"
    "saddl2 v22.8h, v31.16b, v30.16b\n"
    "ldp x22, x21, [x20, #0x0]\n"
    "ldr q31, [x22, x27]\n"
    "ldr q30, [x21, x27]\n"
    "subs x23, x23, #0x1\n"
    "saddw v15.4s, v15.4s, v23.4h\n"
    "saddw2 v14.4s, v14.4s, v23.8h\n"
    "saddw v13.4s, v13.4s, v22.4h\n"
    "saddw2 v12.4s, v12.4s, v22.8h\n"
    "add x20, x20, #0x10\n"
    "bgt 9b\n"
    "10:"  // Single vector of channels: Loop: 2 inputs tail
    "saddl v23.8h, v31.8b, v30.8b\n"
    "saddl2 v22.8h, v31.16b, v30.16b\n"
    "saddw v15.4s, v15.4s, v23.4h\n"
    "saddw2 v14.4s, v14.4s, v23.8h\n"
    "saddw v13.4s, v13.4s, v22.4h\n"
    "saddw2 v12.4s, v12.4s, v22.8h\n"
    "11:"  // Single vector of channels: Loop: After loop
    "ands x21, %x[n_valid_cells], #0x1\n"
    "beq 13f\n"
    "12:"  // Single vector of channels: Loop: Single input loop
    "ldr x22, [x20], #0x8\n"
    "ldr q31, [x22, x27]\n"
    "sxtl v23.8h, v31.8b\n"
    "sxtl2 v22.8h, v31.16b\n"
    "subs x21, x21, #0x1\n"
    "saddw v15.4s, v15.4s, v23.4h\n"
    "saddw2 v14.4s, v14.4s, v23.8h\n"
    "saddw v13.4s, v13.4s, v22.4h\n"
    "saddw2 v12.4s, v12.4s, v22.8h\n"
    "bgt 12b\n"
    "13:"  // Single vector of channels: Loop: Single input loop: End
    "ld1r { v17.4s }, [%x[rescale_ptr]]\n"
    "ld1r { v16.4s }, [%x[shift_ptr]]\n"
    "sqdmulh v15.4s, v15.4s, v17.4s\n"
    "sqdmulh v14.4s, v14.4s, v17.4s\n"
    "sqdmulh v13.4s, v13.4s, v17.4s\n"
    "sqdmulh v12.4s, v12.4s, v17.4s\n"
    "sub %x[n_channels], %x[n_channels], #0x10\n"
    "cmp %x[n_channels], #0x10\n"
    "movi v17.4s, #0x7f\n"
    "srshl v15.4s, v15.4s, v16.4s\n"
    "srshl v14.4s, v14.4s, v16.4s\n"
    "srshl v13.4s, v13.4s, v16.4s\n"
    "srshl v12.4s, v12.4s, v16.4s\n"
    "not v16.16b, v17.16b\n"
    "smax v15.4s, v15.4s, v16.4s\n"
    "smax v14.4s, v14.4s, v16.4s\n"
    "smax v13.4s, v13.4s, v16.4s\n"
    "smax v12.4s, v12.4s, v16.4s\n"
    "smin v15.4s, v15.4s, v17.4s\n"
    "smin v14.4s, v14.4s, v17.4s\n"
    "smin v13.4s, v13.4s, v17.4s\n"
    "smin v12.4s, v12.4s, v17.4s\n"
    "uzp1 v23.16b, v15.16b, v14.16b\n"
    "uzp1 v16.16b, v13.16b, v12.16b\n"
    "uzp1 v16.16b, v23.16b, v16.16b\n"
    "str q16, [%x[outptr], x27]\n"
    "add x27, x27, #0x10\n"
    "bge 8b\n"
    "cbz %x[n_channels], 43f\n"
    "14:"  // Oddments
    "lsr x23, %x[n_valid_cells], #0x1\n"
    "add %x[outptr], %x[outptr], x27\n"
    "movi v15.4s, #0x0\n"
    "movi v14.4s, #0x0\n"
    "movi v13.4s, #0x0\n"
    "movi v12.4s, #0x0\n"
    "mov x20, %x[inptrs]\n"
    "cbz x23, 24f\n"
    "15:"  // Oddments: 2 inputs loop
    "ldp x22, x21, [x20, #0x0]\n"
    "add x20, x20, #0x10\n"
    "add x22, x22, x27\n"
    "movi v31.16b, #0x0\n"
    "add x21, x21, x27\n"
    "movi v30.16b, #0x0\n"
    "tbz %x[n_channels], #3, 19f\n"
    "ldr d31, [x22], #0x8\n"
    "ldr d30, [x21], #0x8\n"
    "tbz %x[n_channels], #2, 17f\n"
    "ld1 { v31.s }[2], [x22], #0x4\n"
    "ld1 { v30.s }[2], [x21], #0x4\n"
    "tbz %x[n_channels], #1, 16f\n"
    "ld1 { v31.h }[6], [x22], #0x2\n"
    "ld1 { v30.h }[6], [x21], #0x2\n"
    "tbz %x[n_channels], #0, 23f\n"
    "ld1 { v31.b }[14], [x22], #0x1\n"
    "ld1 { v30.b }[14], [x21], #0x1\n"
    "b 23f\n"
    "16:"  // Oddments: 2 inputs loop: Load: Bit 3: Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 23f\n"
    "ld1 { v31.b }[12], [x22], #0x1\n"
    "ld1 { v30.b }[12], [x21], #0x1\n"
    "b 23f\n"
    "17:"  // Oddments: 2 inputs loop: Load: Bit 3: Bit 2: Unset
    "tbz %x[n_channels], #1, 18f\n"
    "ld1 { v31.h }[4], [x22], #0x2\n"
    "ld1 { v30.h }[4], [x21], #0x2\n"
    "tbz %x[n_channels], #0, 23f\n"
    "ld1 { v31.b }[10], [x22], #0x1\n"
    "ld1 { v30.b }[10], [x21], #0x1\n"
    "b 23f\n"
    "18:"  // Oddments: 2 inputs loop: Load: Bit 3: Bit 2: Unset: Bit 1: Unset
    "tbz %x[n_channels], #0, 23f\n"
    "ld1 { v31.b }[8], [x22], #0x1\n"
    "ld1 { v30.b }[8], [x21], #0x1\n"
    "b 23f\n"
    "19:"  // Oddments: 2 inputs loop: Load: Bit 3: Unset
    "tbz %x[n_channels], #2, 21f\n"
    "ldr s31, [x22], #0x4\n"
    "ldr s30, [x21], #0x4\n"
    "tbz %x[n_channels], #1, 20f\n"
    "ld1 { v31.h }[2], [x22], #0x2\n"
    "ld1 { v30.h }[2], [x21], #0x2\n"
    "tbz %x[n_channels], #0, 23f\n"
    "ld1 { v31.b }[6], [x22], #0x1\n"
    "ld1 { v30.b }[6], [x21], #0x1\n"
    "b 23f\n"
    "20:"  // Oddments: 2 inputs loop: Load: Bit 3: Unset: Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 23f\n"
    "ld1 { v31.b }[4], [x22], #0x1\n"
    "ld1 { v30.b }[4], [x21], #0x1\n"
    "b 23f\n"
    "21:"  // Oddments: 2 inputs loop: Load: Bit 3: Unset: Bit 2: Unset
    "tbz %x[n_channels], #1, 22f\n"
    "ldr h31, [x22], #0x2\n"
    "ldr h30, [x21], #0x2\n"
    "tbz %x[n_channels], #0, 23f\n"
    "ld1 { v31.b }[2], [x22], #0x1\n"
    "ld1 { v30.b }[2], [x21], #0x1\n"
    "b 23f\n"
    "22:"  // Oddments: 2 inputs loop: Load: Bit 3: Unset: Bit 2: Unset: Bit 1: Unset
    "tbz %x[n_channels], #0, 23f\n"
    "ldr b31, [x22], #0x1\n"
    "ldr b30, [x21], #0x1\n"
    "23:"  // Oddments: 2 inputs loop: Load: Bit 3: End
    "saddl v23.8h, v31.8b, v30.8b\n"
    "saddl2 v22.8h, v31.16b, v30.16b\n"
    "subs x23, x23, #0x1\n"
    "saddw v15.4s, v15.4s, v23.4h\n"
    "saddw2 v14.4s, v14.4s, v23.8h\n"
    "saddw v13.4s, v13.4s, v22.4h\n"
    "saddw2 v12.4s, v12.4s, v22.8h\n"
    "bgt 15b\n"
    "24:"  // Oddments: After loop
    "ands x21, %x[n_valid_cells], #0x1\n"
    "beq 34f\n"
    "25:"  // Oddments: Single input loop
    "ldr x22, [x20], #0x8\n"
    "add x22, x22, x27\n"
    "movi v31.16b, #0x0\n"
    "tbz %x[n_channels], #3, 29f\n"
    "ldr d31, [x22], #0x8\n"
    "tbz %x[n_channels], #2, 27f\n"
    "ld1 { v31.s }[2], [x22], #0x4\n"
    "tbz %x[n_channels], #1, 26f\n"
    "ld1 { v31.h }[6], [x22], #0x2\n"
    "tbz %x[n_channels], #0, 33f\n"
    "ld1 { v31.b }[14], [x22], #0x1\n"
    "b 33f\n"
    "26:"  // Oddments: Single input loop: Load: Bit 3: Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 33f\n"
    "ld1 { v31.b }[12], [x22], #0x1\n"
    "b 33f\n"
    "27:"  // Oddments: Single input loop: Load: Bit 3: Bit 2: Unset
    "tbz %x[n_channels], #1, 28f\n"
    "ld1 { v31.h }[4], [x22], #0x2\n"
    "tbz %x[n_channels], #0, 33f\n"
    "ld1 { v31.b }[10], [x22], #0x1\n"
    "b 33f\n"
    "28:"  // Oddments: Single input loop: Load: Bit 3: Bit 2: Unset: Bit 1: Unset
    "tbz %x[n_channels], #0, 33f\n"
    "ld1 { v31.b }[8], [x22], #0x1\n"
    "b 33f\n"
    "29:"  // Oddments: Single input loop: Load: Bit 3: Unset
    "tbz %x[n_channels], #2, 31f\n"
    "ldr s31, [x22], #0x4\n"
    "tbz %x[n_channels], #1, 30f\n"
    "ld1 { v31.h }[2], [x22], #0x2\n"
    "tbz %x[n_channels], #0, 33f\n"
    "ld1 { v31.b }[6], [x22], #0x1\n"
    "b 33f\n"
    "30:"  // Oddments: Single input loop: Load: Bit 3: Unset: Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 33f\n"
    "ld1 { v31.b }[4], [x22], #0x1\n"
    "b 33f\n"
    "31:"  // Oddments: Single input loop: Load: Bit 3: Unset: Bit 2: Unset
    "tbz %x[n_channels], #1, 32f\n"
    "ldr h31, [x22], #0x2\n"
    "tbz %x[n_channels], #0, 33f\n"
    "ld1 { v31.b }[2], [x22], #0x1\n"
    "b 33f\n"
    "32:"  // Oddments: Single input loop: Load: Bit 3: Unset: Bit 2: Unset: Bit 1: Unset
    "tbz %x[n_channels], #0, 33f\n"
    "ldr b31, [x22], #0x1\n"
    "33:"  // Oddments: Single input loop: Load: Bit 3: End
    "sxtl v23.8h, v31.8b\n"
    "sxtl2 v22.8h, v31.16b\n"
    "subs x21, x21, #0x1\n"
    "saddw v15.4s, v15.4s, v23.4h\n"
    "saddw2 v14.4s, v14.4s, v23.8h\n"
    "saddw v13.4s, v13.4s, v22.4h\n"
    "saddw2 v12.4s, v12.4s, v22.8h\n"
    "bgt 25b\n"
    "34:"  // Oddments: Single input loop: End
    "ld1r { v17.4s }, [%x[rescale_ptr]]\n"
    "ld1r { v16.4s }, [%x[shift_ptr]]\n"
    "sqdmulh v15.4s, v15.4s, v17.4s\n"
    "sqdmulh v14.4s, v14.4s, v17.4s\n"
    "sqdmulh v13.4s, v13.4s, v17.4s\n"
    "sqdmulh v12.4s, v12.4s, v17.4s\n"
    "movi v17.4s, #0x7f\n"
    "srshl v15.4s, v15.4s, v16.4s\n"
    "srshl v14.4s, v14.4s, v16.4s\n"
    "srshl v13.4s, v13.4s, v16.4s\n"
    "srshl v12.4s, v12.4s, v16.4s\n"
    "not v16.16b, v17.16b\n"
    "smax v15.4s, v15.4s, v16.4s\n"
    "smax v14.4s, v14.4s, v16.4s\n"
    "smax v13.4s, v13.4s, v16.4s\n"
    "smax v12.4s, v12.4s, v16.4s\n"
    "smin v15.4s, v15.4s, v17.4s\n"
    "smin v14.4s, v14.4s, v17.4s\n"
    "smin v13.4s, v13.4s, v17.4s\n"
    "smin v12.4s, v12.4s, v17.4s\n"
    "uzp1 v23.16b, v15.16b, v14.16b\n"
    "uzp1 v16.16b, v13.16b, v12.16b\n"
    "uzp1 v16.16b, v23.16b, v16.16b\n"
    "tbz %x[n_channels], #3, 38f\n"
    "st1 { v16.d }[0], [%x[outptr]], #0x8\n"
    "tbz %x[n_channels], #2, 36f\n"
    "st1 { v16.s }[2], [%x[outptr]], #0x4\n"
    "tbz %x[n_channels], #1, 35f\n"
    "st1 { v16.h }[6], [%x[outptr]], #0x2\n"
    "tbz %x[n_channels], #0, 42f\n"
    "st1 { v16.b }[14], [%x[outptr]], #0x1\n"
    "b 42f\n"
    "35:"  // Oddments: Store: Bit 3: Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 42f\n"
    "st1 { v16.b }[12], [%x[outptr]], #0x1\n"
    "b 42f\n"
    "36:"  // Oddments: Store: Bit 3: Bit 2: Unset
    "tbz %x[n_channels], #1, 37f\n"
    "st1 { v16.h }[4], [%x[outptr]], #0x2\n"
    "tbz %x[n_channels], #0, 42f\n"
    "st1 { v16.b }[10], [%x[outptr]], #0x1\n"
    "b 42f\n"
    "37:"  // Oddments: Store: Bit 3: Bit 2: Unset: Bit 1: Unset
    "tbz %x[n_channels], #0, 42f\n"
    "st1 { v16.b }[8], [%x[outptr]], #0x1\n"
    "b 42f\n"
    "38:"  // Oddments: Store: Bit 3: Unset
    "tbz %x[n_channels], #2, 40f\n"
    "st1 { v16.s }[0], [%x[outptr]], #0x4\n"
    "tbz %x[n_channels], #1, 39f\n"
    "st1 { v16.h }[2], [%x[outptr]], #0x2\n"
    "tbz %x[n_channels], #0, 42f\n"
    "st1 { v16.b }[6], [%x[outptr]], #0x1\n"
    "b 42f\n"
    "39:"  // Oddments: Store: Bit 3: Unset: Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 42f\n"
    "st1 { v16.b }[4], [%x[outptr]], #0x1\n"
    "b 42f\n"
    "40:"  // Oddments: Store: Bit 3: Unset: Bit 2: Unset
    "tbz %x[n_channels], #1, 41f\n"
    "st1 { v16.h }[0], [%x[outptr]], #0x2\n"
    "tbz %x[n_channels], #0, 42f\n"
    "st1 { v16.b }[2], [%x[outptr]], #0x1\n"
    "b 42f\n"
    "41:"  // Oddments: Store: Bit 3: Unset: Bit 2: Unset: Bit 1: Unset
    "tbz %x[n_channels], #0, 42f\n"
    "st1 { v16.b }[0], [%x[outptr]], #0x1\n"
    "42:"  // Oddments: Store: Bit 3: End
    "43:"  // End
    : [n_channels] "+&r" (n_channels), [outptr] "+&r" (outptr)
    : [inptrs] "r" (inptrs), [n_valid_cells] "r" (n_valid_cells), [rescale_ptr] "r" (&rescale_value), [shift_ptr] "r" (&shift_value)
    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27"
  );
}

}  // namespace pooling
}  // namespace arm_conv
#endif  // defined(__aarch64__)
