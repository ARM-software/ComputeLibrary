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
#include "arm_compute/core/Error.h"

#include "src/cpu/kernels/lut/list.h"

#ifdef __aarch64__
#ifdef ARM_COMPUTE_ENABLE_SVE

#include <arm_sve.h>

namespace arm_compute
{
namespace cpu
{
void lut_u16_sve(const uint16_t *table, size_t num_strings, size_t size, const uint16_t *input, uint16_t *output)
{
    int64_t cnth  = svcnth();
    int64_t tail  = size & (4 * cnth - 1);
    int64_t count = size - tail;
    int64_t pos   = 0;
    ARM_COMPUTE_UNUSED(num_strings);
    __asm __volatile("cbz %[count], 2f\n"
                     "mov z31.s, #0\n"
                     "cnth x7, ALL, MUL #4\n"
                     "cntb x8, ALL, MUL #4\n"
                     "ptrue p0.b\n"
                     "1:"
                     "ld1h z0.h, p0/z, [%[input]]\n"
                     "ld1h z1.h, p0/z, [%[input], #1, MUL VL]\n"
                     "ld1h z2.h, p0/z, [%[input], #2, MUL VL]\n"
                     "ld1h z3.h, p0/z, [%[input], #3, MUL VL]\n"
                     "add %[input], %[input], x8\n"

                     "zip1 z8.h, z0.h, z31.h\n"
                     "ld1h z8.s, p0/z, [%[table], z8.s, UXTW #1]\n"
                     "zip2 z0.h, z0.h, z31.h\n"
                     "ld1h z0.s, p0/z, [%[table], z0.s, UXTW #1]\n"
                     "uzp1 z0.h, z8.h, z0.h\n"
                     "st1h z0.h, p0, [%[output]]\n"

                     "zip1 z10.h, z1.h, z31.h\n"
                     "ld1h z10.s, p0/z, [%[table], z10.s, UXTW #1]\n"
                     "zip2 z1.h, z1.h, z31.h\n"
                     "ld1h z1.s, p0/z, [%[table], z1.s, UXTW #1]\n"
                     "uzp1 z1.h, z10.h, z1.h\n"
                     "st1h z1.h, p0, [%[output], #1, MUL VL]\n"

                     "zip1 z12.h, z2.h, z31.h\n"
                     "ld1h z12.s, p0/z, [%[table], z12.s, UXTW #1]\n"
                     "zip2 z2.h, z2.h, z31.h\n"
                     "ld1h z2.s, p0/z, [%[table], z2.s, UXTW #1]\n"
                     "uzp1 z2.h, z12.h, z2.h\n"
                     "st1h z2.h, p0, [%[output], #2, MUL VL]\n"

                     "zip1 z14.h, z3.h, z31.h\n"
                     "ld1h z14.s, p0/z, [%[table], z14.s, UXTW #1]\n"
                     "zip2 z3.h, z3.h, z31.h\n"
                     "ld1h z3.s, p0/z, [%[table], z3.s, UXTW #1]\n"
                     "uzp1 z3.h, z14.h, z3.h\n"
                     "st1h z3.h, p0, [%[output], #3, MUL VL]\n"

                     "add %[pos], %[pos], x7\n"
                     "add %[output], %[output], x8\n"
                     "cmp %[pos], %[count]\n"
                     "blt 1b\n"
                     "2:\n"
                     : [count] "+r"(count), [input] "+r"(input), [output] "+r"(output), [pos] "+r"(pos)
                     : [table] "r"(table)
                     : "memory", "cc", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12",
                       "z14", "z31", "p0", "p1", "z2", "z3", "z4", "x7", "x8");
    for (int i = 0; i < tail; i++)
    {
        output[i] = table[input[i]];
    }
}

} // namespace cpu
} // namespace arm_compute

#endif // ARM_COMPUTE_ENABLE_SVE
#endif // __aarch64__
