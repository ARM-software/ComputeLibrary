/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_CPP_TYPES_H__
#define __ARM_COMPUTE_CPP_TYPES_H__

namespace arm_compute
{
/** Available CPU Targets */
enum class CPUTarget
{
    ARCH_MASK  = 0x0F00,
    CPU_MODEL  = 0x00FF,
    INTRINSICS = 0x0100,
    ARMV7      = 0x0200,
    ARMV8      = 0x0300,
    ARMV8_2    = 0x0400,
    A7x        = 0x0070,
    A5x        = 0x0050,
    DOT        = 0x1000,

    A53     = (ARMV8 | A7x | 0x3),
    A55     = (ARMV8_2 | A5x | 0x5),
    A55_DOT = (A55 | DOT),
    A72     = (ARMV8 | A7x | 0x2),
    A73     = (ARMV8 | A7x | 0x3),
    A75     = (ARMV8_2 | A7x | 0x5),
    A75_DOT = (A75 | DOT),
};

struct CPUInfo
{
    CPUTarget CPU{ CPUTarget::INTRINSICS };
    int       L1_size{ 0 };
    int       L2_size{ 0 };
};

struct ThreadInfo
{
    int     thread_id{ 0 };
    int     num_threads{ 1 };
    CPUInfo cpu_info{};
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CPP_TYPES_H__ */
