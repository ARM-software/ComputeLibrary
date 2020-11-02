/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#pragma once

#ifdef __aarch64__

#include <cstdint>

namespace arm_gemm
{

// Actual kernel implementations
void a64_smallK_hybrid_u8u32_dot_8x4(const uint8_t *, int, const uint8_t *, uint32_t *, int, int, int, int, const uint32_t *, Activation, bool);
void a64_smallK_hybrid_u8u32_dot_8x4_a55(const uint8_t *, int, const uint8_t *, uint32_t *, int, int, int, int, const uint32_t *, Activation, bool);

class cls_a64_smallK_hybrid_u8u32_dot_8x4
{
public:
    typedef uint8_t operand_type;
    typedef uint32_t result_type;

    typedef void (*kern_type)(const uint8_t *, int, const uint8_t *, uint32_t *, int, int, int, int, const uint32_t *, Activation, bool);

    /* Kernel blocking parameters */
    static constexpr unsigned int out_height()
    {
        return 8;
    }

    static unsigned int out_width()
    {
        return 4;
    }

    static constexpr unsigned int k_unroll()
    {
        return 4;
    }

    static constexpr bool supports_accumulate()
    {
        return false;
    }

    static constexpr bool supports_bias()
    {
        return false;
    }

    static constexpr bool supports_activation()
    {
        return false;
    }

    StdTransformsFixed<operand_type, result_type, 8, 4, 4> transforms = {};

    // Default to the generic kernel
    kern_type kernel=a64_smallK_hybrid_u8u32_dot_8x4;

    cls_a64_smallK_hybrid_u8u32_dot_8x4(const CPUInfo *ci)
    {
        if (ci->get_cpu_model() == CPUModel::A55r1) {
            kernel = a64_smallK_hybrid_u8u32_dot_8x4_a55;
        }
    }
};

} // namespace arm_gemm

#endif // __aarch64__
