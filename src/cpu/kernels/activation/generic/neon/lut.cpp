/*
 * Copyright (c) 2022 Arm Limited.
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

#include "arm_compute/core/Helpers.h"

#include <arm_neon.h>
#include <cstdint>

namespace arm_compute
{
namespace cpu
{
namespace
{
#ifdef __aarch64__

void substitute_bytes_neon(
    const uint8_t        *table,
    size_t                num_strings,
    size_t                string_length,
    const uint8_t *const *input,
    uint8_t *const       *output)
{
    __asm__ __volatile__(
        "ldr q16, [%x[table], #0x0]\n"
        "ldr q17, [%x[table], #0x10]\n"
        "mov x22, #0x0\n"
        "ldr q18, [%x[table], #0x20]\n"
        "ldr q19, [%x[table], #0x30]\n"
        "ldr q20, [%x[table], #0x40]\n"
        "ldr q21, [%x[table], #0x50]\n"
        "ldr q22, [%x[table], #0x60]\n"
        "ldr q23, [%x[table], #0x70]\n"
        "ldr q24, [%x[table], #0x80]\n"
        "ldr q25, [%x[table], #0x90]\n"
        "ldr q26, [%x[table], #0xa0]\n"
        "ldr q27, [%x[table], #0xb0]\n"
        "ldr q28, [%x[table], #0xc0]\n"
        "ldr q29, [%x[table], #0xd0]\n"
        "ldr q30, [%x[table], #0xe0]\n"
        "ldr q31, [%x[table], #0xf0]\n"
        "1:" // string loop
        "ldr x21, [%x[input], x22, LSL #0x3]\n"
        "ldr x20, [%x[output], x22, LSL #0x3]\n"
        "movi v12.16b, #0x40\n"
        "movi v11.16b, #0x80\n"
        "movi v10.16b, #0xc0\n"
        "mov x19, %x[string_length]\n"
        "2:" // 4 rounds: width loop
        "cmp x19, #0x30\n"
        "bge 27f\n"
        "tbz x19, #5, 10f\n"
        "ld1 { v9.16b }, [x21], #0x10\n"
        "ld1 { v13.16b }, [x21], #0x10\n"
        "tbz x19, #3, 6f\n"
        "ldr d14, [x21], #0x8\n"
        "tbz x19, #2, 4f\n"
        "ld1 { v14.s }[2], [x21], #0x4\n"
        "tbz x19, #1, 3f\n"
        "ld1 { v14.h }[6], [x21], #0x2\n"
        "tbz x19, #0, 26f\n"
        "ld1 { v14.b }[14], [x21]\n"
        "b 26f\n"
        "3:" // 4 rounds: Partial load: partial_1_44
        "tbz x19, #0, 26f\n"
        "ld1 { v14.b }[12], [x21]\n"
        "b 26f\n"
        "4:" // 4 rounds: Partial load: partial_2_40
        "tbz x19, #1, 5f\n"
        "ld1 { v14.h }[4], [x21], #0x2\n"
        "tbz x19, #0, 26f\n"
        "ld1 { v14.b }[10], [x21]\n"
        "b 26f\n"
        "5:" // 4 rounds: Partial load: partial_1_40
        "tbz x19, #0, 26f\n"
        "ld1 { v14.b }[8], [x21]\n"
        "b 26f\n"
        "6:" // 4 rounds: Partial load: partial_4_32
        "tbz x19, #2, 8f\n"
        "ldr s14, [x21], #0x4\n"
        "tbz x19, #1, 7f\n"
        "ld1 { v14.h }[2], [x21], #0x2\n"
        "tbz x19, #0, 26f\n"
        "ld1 { v14.b }[6], [x21]\n"
        "b 26f\n"
        "7:" // 4 rounds: Partial load: partial_1_36
        "tbz x19, #0, 26f\n"
        "ld1 { v14.b }[4], [x21]\n"
        "b 26f\n"
        "8:" // 4 rounds: Partial load: partial_2_32
        "tbz x19, #1, 9f\n"
        "ldr h14, [x21], #0x2\n"
        "tbz x19, #0, 26f\n"
        "ld1 { v14.b }[2], [x21]\n"
        "b 26f\n"
        "9:" // 4 rounds: Partial load: partial_1_32
        "tbz x19, #0, 26f\n"
        "ldr b14, [x21, #0x0]\n"
        "b 26f\n"
        "10:" // 4 rounds: Partial load: partial_16_0
        "tbz x19, #4, 18f\n"
        "ld1 { v9.16b }, [x21], #0x10\n"
        "tbz x19, #3, 14f\n"
        "ldr d13, [x21], #0x8\n"
        "tbz x19, #2, 12f\n"
        "ld1 { v13.s }[2], [x21], #0x4\n"
        "tbz x19, #1, 11f\n"
        "ld1 { v13.h }[6], [x21], #0x2\n"
        "tbz x19, #0, 26f\n"
        "ld1 { v13.b }[14], [x21]\n"
        "b 26f\n"
        "11:" // 4 rounds: Partial load: partial_1_28
        "tbz x19, #0, 26f\n"
        "ld1 { v13.b }[12], [x21]\n"
        "b 26f\n"
        "12:" // 4 rounds: Partial load: partial_2_24
        "tbz x19, #1, 13f\n"
        "ld1 { v13.h }[4], [x21], #0x2\n"
        "tbz x19, #0, 26f\n"
        "ld1 { v13.b }[10], [x21]\n"
        "b 26f\n"
        "13:" // 4 rounds: Partial load: partial_1_24
        "tbz x19, #0, 26f\n"
        "ld1 { v13.b }[8], [x21]\n"
        "b 26f\n"
        "14:" // 4 rounds: Partial load: partial_4_16
        "tbz x19, #2, 16f\n"
        "ldr s13, [x21], #0x4\n"
        "tbz x19, #1, 15f\n"
        "ld1 { v13.h }[2], [x21], #0x2\n"
        "tbz x19, #0, 26f\n"
        "ld1 { v13.b }[6], [x21]\n"
        "b 26f\n"
        "15:" // 4 rounds: Partial load: partial_1_20
        "tbz x19, #0, 26f\n"
        "ld1 { v13.b }[4], [x21]\n"
        "b 26f\n"
        "16:" // 4 rounds: Partial load: partial_2_16
        "tbz x19, #1, 17f\n"
        "ldr h13, [x21], #0x2\n"
        "tbz x19, #0, 26f\n"
        "ld1 { v13.b }[2], [x21]\n"
        "b 26f\n"
        "17:" // 4 rounds: Partial load: partial_1_16
        "tbz x19, #0, 26f\n"
        "ldr b13, [x21, #0x0]\n"
        "b 26f\n"
        "18:" // 4 rounds: Partial load: partial_8_0
        "tbz x19, #3, 22f\n"
        "ldr d9, [x21], #0x8\n"
        "tbz x19, #2, 20f\n"
        "ld1 { v9.s }[2], [x21], #0x4\n"
        "tbz x19, #1, 19f\n"
        "ld1 { v9.h }[6], [x21], #0x2\n"
        "tbz x19, #0, 26f\n"
        "ld1 { v9.b }[14], [x21]\n"
        "b 26f\n"
        "19:" // 4 rounds: Partial load: partial_1_12
        "tbz x19, #0, 26f\n"
        "ld1 { v9.b }[12], [x21]\n"
        "b 26f\n"
        "20:" // 4 rounds: Partial load: partial_2_8
        "tbz x19, #1, 21f\n"
        "ld1 { v9.h }[4], [x21], #0x2\n"
        "tbz x19, #0, 26f\n"
        "ld1 { v9.b }[10], [x21]\n"
        "b 26f\n"
        "21:" // 4 rounds: Partial load: partial_1_8
        "tbz x19, #0, 26f\n"
        "ld1 { v9.b }[8], [x21]\n"
        "b 26f\n"
        "22:" // 4 rounds: Partial load: partial_4_0
        "tbz x19, #2, 24f\n"
        "ldr s9, [x21], #0x4\n"
        "tbz x19, #1, 23f\n"
        "ld1 { v9.h }[2], [x21], #0x2\n"
        "tbz x19, #0, 26f\n"
        "ld1 { v9.b }[6], [x21]\n"
        "b 26f\n"
        "23:" // 4 rounds: Partial load: partial_1_4
        "tbz x19, #0, 26f\n"
        "ld1 { v9.b }[4], [x21]\n"
        "b 26f\n"
        "24:" // 4 rounds: Partial load: partial_2_0
        "tbz x19, #1, 25f\n"
        "ldr h9, [x21], #0x2\n"
        "tbz x19, #0, 26f\n"
        "ld1 { v9.b }[2], [x21]\n"
        "b 26f\n"
        "25:" // 4 rounds: Partial load: partial_1_0
        "ldr b9, [x21, #0x0]\n"
        "26:" // 4 rounds: Partial load: Done
        "b 28f\n"
        "27:" // 4 rounds: Full load
        "ldr q9, [x21, #0x0]\n"
        "ldr q13, [x21, #0x10]\n"
        "ldr q14, [x21, #0x20]\n"
        "add x21, x21, #0x30\n"
        "28:" // 4 rounds: Load done
        "sub v8.16b, v9.16b, v12.16b\n"
        "sub v7.16b, v9.16b, v11.16b\n"
        "tbl v8.16b, { v20.16b, v21.16b, v22.16b, v23.16b }, v8.16b\n"
        "sub v6.16b, v9.16b, v10.16b\n"
        "sub v5.16b, v13.16b, v12.16b\n"
        "tbl v9.16b, { v16.16b, v17.16b, v18.16b, v19.16b }, v9.16b\n"
        "sub v4.16b, v13.16b, v11.16b\n"
        "sub v3.16b, v13.16b, v10.16b\n"
        "tbl v7.16b, { v24.16b, v25.16b, v26.16b, v27.16b }, v7.16b\n"
        "sub v2.16b, v14.16b, v12.16b\n"
        "sub v1.16b, v14.16b, v11.16b\n"
        "tbl v6.16b, { v28.16b, v29.16b, v30.16b, v31.16b }, v6.16b\n"
        "sub v0.16b, v14.16b, v10.16b\n"
        "tbl v13.16b, { v16.16b, v17.16b, v18.16b, v19.16b }, v13.16b\n"
        "tbl v5.16b, { v20.16b, v21.16b, v22.16b, v23.16b }, v5.16b\n"
        "tbl v4.16b, { v24.16b, v25.16b, v26.16b, v27.16b }, v4.16b\n"
        "tbl v3.16b, { v28.16b, v29.16b, v30.16b, v31.16b }, v3.16b\n"
        "orr v9.16b, v9.16b, v8.16b\n"
        "tbl v14.16b, { v16.16b, v17.16b, v18.16b, v19.16b }, v14.16b\n"
        "tbl v2.16b, { v20.16b, v21.16b, v22.16b, v23.16b }, v2.16b\n"
        "orr v7.16b, v7.16b, v6.16b\n"
        "tbl v1.16b, { v24.16b, v25.16b, v26.16b, v27.16b }, v1.16b\n"
        "tbl v0.16b, { v28.16b, v29.16b, v30.16b, v31.16b }, v0.16b\n"
        "orr v13.16b, v13.16b, v5.16b\n"
        "orr v4.16b, v4.16b, v3.16b\n"
        "orr v14.16b, v14.16b, v2.16b\n"
        "cmp x19, #0x30\n"
        "orr v1.16b, v1.16b, v0.16b\n"
        "orr v9.16b, v9.16b, v7.16b\n"
        "orr v13.16b, v13.16b, v4.16b\n"
        "orr v14.16b, v14.16b, v1.16b\n"
        "bge 53f\n"
        "tbz x19, #5, 36f\n"
        "st1 { v9.16b }, [x20], #0x10\n"
        "st1 { v13.16b }, [x20], #0x10\n"
        "tbz x19, #3, 32f\n"
        "str d14, [x20], #0x8\n"
        "tbz x19, #2, 30f\n"
        "st1 { v14.s }[2], [x20], #0x4\n"
        "tbz x19, #1, 29f\n"
        "st1 { v14.h }[6], [x20], #0x2\n"
        "tbz x19, #0, 52f\n"
        "st1 { v14.b }[14], [x20]\n"
        "b 52f\n"
        "29:" // 4 rounds: Partial writeback: partial_1_44
        "tbz x19, #0, 52f\n"
        "st1 { v14.b }[12], [x20]\n"
        "b 52f\n"
        "30:" // 4 rounds: Partial writeback: partial_2_40
        "tbz x19, #1, 31f\n"
        "st1 { v14.h }[4], [x20], #0x2\n"
        "tbz x19, #0, 52f\n"
        "st1 { v14.b }[10], [x20]\n"
        "b 52f\n"
        "31:" // 4 rounds: Partial writeback: partial_1_40
        "tbz x19, #0, 52f\n"
        "st1 { v14.b }[8], [x20]\n"
        "b 52f\n"
        "32:" // 4 rounds: Partial writeback: partial_4_32
        "tbz x19, #2, 34f\n"
        "str s14, [x20], #0x4\n"
        "tbz x19, #1, 33f\n"
        "st1 { v14.h }[2], [x20], #0x2\n"
        "tbz x19, #0, 52f\n"
        "st1 { v14.b }[6], [x20]\n"
        "b 52f\n"
        "33:" // 4 rounds: Partial writeback: partial_1_36
        "tbz x19, #0, 52f\n"
        "st1 { v14.b }[4], [x20]\n"
        "b 52f\n"
        "34:" // 4 rounds: Partial writeback: partial_2_32
        "tbz x19, #1, 35f\n"
        "str h14, [x20], #0x2\n"
        "tbz x19, #0, 52f\n"
        "st1 { v14.b }[2], [x20]\n"
        "b 52f\n"
        "35:" // 4 rounds: Partial writeback: partial_1_32
        "tbz x19, #0, 52f\n"
        "str b14, [x20, #0x0]\n"
        "b 52f\n"
        "36:" // 4 rounds: Partial writeback: partial_16_0
        "tbz x19, #4, 44f\n"
        "st1 { v9.16b }, [x20], #0x10\n"
        "tbz x19, #3, 40f\n"
        "str d13, [x20], #0x8\n"
        "tbz x19, #2, 38f\n"
        "st1 { v13.s }[2], [x20], #0x4\n"
        "tbz x19, #1, 37f\n"
        "st1 { v13.h }[6], [x20], #0x2\n"
        "tbz x19, #0, 52f\n"
        "st1 { v13.b }[14], [x20]\n"
        "b 52f\n"
        "37:" // 4 rounds: Partial writeback: partial_1_28
        "tbz x19, #0, 52f\n"
        "st1 { v13.b }[12], [x20]\n"
        "b 52f\n"
        "38:" // 4 rounds: Partial writeback: partial_2_24
        "tbz x19, #1, 39f\n"
        "st1 { v13.h }[4], [x20], #0x2\n"
        "tbz x19, #0, 52f\n"
        "st1 { v13.b }[10], [x20]\n"
        "b 52f\n"
        "39:" // 4 rounds: Partial writeback: partial_1_24
        "tbz x19, #0, 52f\n"
        "st1 { v13.b }[8], [x20]\n"
        "b 52f\n"
        "40:" // 4 rounds: Partial writeback: partial_4_16
        "tbz x19, #2, 42f\n"
        "str s13, [x20], #0x4\n"
        "tbz x19, #1, 41f\n"
        "st1 { v13.h }[2], [x20], #0x2\n"
        "tbz x19, #0, 52f\n"
        "st1 { v13.b }[6], [x20]\n"
        "b 52f\n"
        "41:" // 4 rounds: Partial writeback: partial_1_20
        "tbz x19, #0, 52f\n"
        "st1 { v13.b }[4], [x20]\n"
        "b 52f\n"
        "42:" // 4 rounds: Partial writeback: partial_2_16
        "tbz x19, #1, 43f\n"
        "str h13, [x20], #0x2\n"
        "tbz x19, #0, 52f\n"
        "st1 { v13.b }[2], [x20]\n"
        "b 52f\n"
        "43:" // 4 rounds: Partial writeback: partial_1_16
        "tbz x19, #0, 52f\n"
        "str b13, [x20, #0x0]\n"
        "b 52f\n"
        "44:" // 4 rounds: Partial writeback: partial_8_0
        "tbz x19, #3, 48f\n"
        "str d9, [x20], #0x8\n"
        "tbz x19, #2, 46f\n"
        "st1 { v9.s }[2], [x20], #0x4\n"
        "tbz x19, #1, 45f\n"
        "st1 { v9.h }[6], [x20], #0x2\n"
        "tbz x19, #0, 52f\n"
        "st1 { v9.b }[14], [x20]\n"
        "b 52f\n"
        "45:" // 4 rounds: Partial writeback: partial_1_12
        "tbz x19, #0, 52f\n"
        "st1 { v9.b }[12], [x20]\n"
        "b 52f\n"
        "46:" // 4 rounds: Partial writeback: partial_2_8
        "tbz x19, #1, 47f\n"
        "st1 { v9.h }[4], [x20], #0x2\n"
        "tbz x19, #0, 52f\n"
        "st1 { v9.b }[10], [x20]\n"
        "b 52f\n"
        "47:" // 4 rounds: Partial writeback: partial_1_8
        "tbz x19, #0, 52f\n"
        "st1 { v9.b }[8], [x20]\n"
        "b 52f\n"
        "48:" // 4 rounds: Partial writeback: partial_4_0
        "tbz x19, #2, 50f\n"
        "str s9, [x20], #0x4\n"
        "tbz x19, #1, 49f\n"
        "st1 { v9.h }[2], [x20], #0x2\n"
        "tbz x19, #0, 52f\n"
        "st1 { v9.b }[6], [x20]\n"
        "b 52f\n"
        "49:" // 4 rounds: Partial writeback: partial_1_4
        "tbz x19, #0, 52f\n"
        "st1 { v9.b }[4], [x20]\n"
        "b 52f\n"
        "50:" // 4 rounds: Partial writeback: partial_2_0
        "tbz x19, #1, 51f\n"
        "str h9, [x20], #0x2\n"
        "tbz x19, #0, 52f\n"
        "st1 { v9.b }[2], [x20]\n"
        "b 52f\n"
        "51:" // 4 rounds: Partial writeback: partial_1_0
        "str b9, [x20, #0x0]\n"
        "52:" // 4 rounds: Partial writeback: Done
        "b 54f\n"
        "53:" // 4 rounds: Full writeback
        "str q9, [x20, #0x0]\n"
        "str q13, [x20, #0x10]\n"
        "str q14, [x20, #0x20]\n"
        "add x20, x20, #0x30\n"
        "54:" // 4 rounds: Writeback done
        "subs x19, x19, #0x30\n"
        "bgt 2b\n"
        "add x22, x22, #0x1\n"
        "cmp x22, %x[num_strings]\n"
        "bne 1b\n"
        :
        : [input] "r"(input), [num_strings] "r"(num_strings), [output] "r"(output), [string_length] "r"(string_length), [table] "r"(table)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x19", "x20", "x21", "x22");
}

#endif // __aarch64__
} // namespace

#ifdef __aarch64__
void neon_q8_activation_lut(const ITensor *src, ITensor *dst, const ActivationLayerInfo &act_info, const Window &window)
{
    ARM_COMPUTE_ERROR_ON(!ActivationLayerInfo::is_lut_supported(act_info.activation(), src->info()->data_type()));
    const auto window_end_x = window.x().end();
    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));
    Iterator input(src, win_collapsed);
    Iterator output(dst, win_collapsed);
    execute_window_loop(win_collapsed, [&](const Coordinates &)
    {
        const auto input_ptr  = reinterpret_cast<const uint8_t *>(input.ptr());
        auto       output_ptr = reinterpret_cast<uint8_t *>(output.ptr());
        substitute_bytes_neon(act_info.lut().data(), 1u, window_end_x, &input_ptr, &output_ptr);
    },
    input, output);
}
#endif // __aarch64__
} // namespace cpu
} // namespace arm_compute
