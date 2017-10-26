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
#include "arm_compute/core/NEON/kernels/arm64/NEGEMMLowpAArch64V8P4Kernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "support/ToolchainSupport.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

// Enable only if compiled for AArch64-V8.2-A targets
#ifdef ARM_COMPUTE_AARCH64_V8_2

#define ASM_PREFETCH(address) "PRFM PLDL1KEEP, " address "\n"
#define ASM_PREFETCHL2(address) "PRFM PLDL2KEEP, " address "\n"
#define ASM_PREFETCHW(address) "PRFM PSTL1KEEP, " address "\n"
#define ASM_PREFETCHWL2(address) "PRFM PSTL2KEEP, " address "\n"

static inline void stincpld(uint32x4_t v0, uint32x4_t v1, uint32x4_t v2, uint32x4_t v3,
                            uint32x4_t v4, uint32x4_t v5, uint32x4_t v6, uint32x4_t v7,
                            uint32_t *&ptr0, uint32_t *&ptr1, uint32_t *&ptr2, uint32_t *&ptr3,
                            uint32_t *&ptr4, uint32_t *&ptr5, uint32_t *&ptr6, uint32_t *&ptr7)
{
    __asm __volatile(
        "LDR    q0, [%[ptr0]]\n"
        "LDR    q1, [%[ptr1]]\n"
        "LDR    q2, [%[ptr2]]\n"
        "LDR    q3, [%[ptr3]]\n"
        "LDR    q4, [%[ptr4]]\n"
        "LDR    q5, [%[ptr5]]\n"
        "LDR    q6, [%[ptr6]]\n"
        "LDR    q7, [%[ptr7]]\n"
        "ADD    v0.4s, v0.4s, %[v0].4s\n" ASM_PREFETCH("[%[ptr0], #80]") "ADD    v1.4s, v1.4s, %[v1].4s\n" ASM_PREFETCH("[%[ptr1], #80]") "ADD    v2.4s, v2.4s, %[v2].4s\n" ASM_PREFETCH("[%[ptr2], #80]")
        "ADD    v3.4s, v3.4s, %[v3].4s\n" ASM_PREFETCH("[%[ptr3], #80]") "ADD    v4.4s, v4.4s, %[v4].4s\n" ASM_PREFETCH("[%[ptr4], #80]") "ADD    v5.4s, v5.4s, %[v5].4s\n" ASM_PREFETCH("[%[ptr5], #80]")
        "ADD    v6.4s, v6.4s, %[v6].4s\n" ASM_PREFETCH("[%[ptr6], #80]") "ADD    v7.4s, v7.4s, %[v7].4s\n" ASM_PREFETCH("[%[ptr7], #80]")
        "STR    q0, [%[ptr0]], #16\n"
        "STR    q1, [%[ptr1]], #16\n"
        "STR    q2, [%[ptr2]], #16\n"
        "STR    q3, [%[ptr3]], #16\n"
        "STR    q4, [%[ptr4]], #16\n"
        "STR    q5, [%[ptr5]], #16\n"
        "STR    q6, [%[ptr6]], #16\n"
        "STR    q7, [%[ptr7]], #16\n"
        : [ptr0] "+r"(ptr0), [ptr1] "+r"(ptr1), [ptr2] "+r"(ptr2), [ptr3] "+r"(ptr3),
        [ptr4] "+r"(ptr4), [ptr5] "+r"(ptr5), [ptr6] "+r"(ptr6), [ptr7] "+r"(ptr7)
        : [v0] "w"(v0), [v1] "w"(v1), [v2] "w"(v2), [v3] "w"(v3),
        [v4] "w"(v4), [v5] "w"(v5), [v6] "w"(v6), [v7] "w"(v7)
        : "x20", "x21", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory");
}

static inline void stinc(uint32x4_t v0, uint32x4_t v1, uint32x4_t v2, uint32x4_t v3,
                         uint32x4_t v4, uint32x4_t v5, uint32x4_t v6, uint32x4_t v7,
                         uint32_t *&ptr0, uint32_t *&ptr1, uint32_t *&ptr2, uint32_t *&ptr3,
                         uint32_t *&ptr4, uint32_t *&ptr5, uint32_t *&ptr6, uint32_t *&ptr7)
{
    __asm __volatile(
        "LDR    q0, [%[ptr0]]\n"
        "LDR    q1, [%[ptr1]]\n"
        "LDR    q2, [%[ptr2]]\n"
        "LDR    q3, [%[ptr3]]\n"
        "LDR    q4, [%[ptr4]]\n"
        "LDR    q5, [%[ptr5]]\n"
        "LDR    q6, [%[ptr6]]\n"
        "LDR    q7, [%[ptr7]]\n"
        "ADD    v0.4s, v0.4s, %[v0].4s\n"
        "ADD    v1.4s, v1.4s, %[v1].4s\n"
        "ADD    v2.4s, v2.4s, %[v2].4s\n"
        "ADD    v3.4s, v3.4s, %[v3].4s\n"
        "ADD    v4.4s, v4.4s, %[v4].4s\n"
        "ADD    v5.4s, v5.4s, %[v5].4s\n"
        "ADD    v6.4s, v6.4s, %[v6].4s\n"
        "ADD    v7.4s, v7.4s, %[v7].4s\n"
        "STR    q0, [%[ptr0]], #16\n"
        "STR    q1, [%[ptr1]], #16\n"
        "STR    q2, [%[ptr2]], #16\n"
        "STR    q3, [%[ptr3]], #16\n"
        "STR    q4, [%[ptr4]], #16\n"
        "STR    q5, [%[ptr5]], #16\n"
        "STR    q6, [%[ptr6]], #16\n"
        "STR    q7, [%[ptr7]], #16\n"
        : [ptr0] "+r"(ptr0), [ptr1] "+r"(ptr1), [ptr2] "+r"(ptr2), [ptr3] "+r"(ptr3),
        [ptr4] "+r"(ptr4), [ptr5] "+r"(ptr5), [ptr6] "+r"(ptr6), [ptr7] "+r"(ptr7)
        : [v0] "w"(v0), [v1] "w"(v1), [v2] "w"(v2), [v3] "w"(v3),
        [v4] "w"(v4), [v5] "w"(v5), [v6] "w"(v6), [v7] "w"(v7)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory");
}

namespace arm_compute
{
void NEGEMMLowpAArch64V8P4Kernel::internal_configure(const ITensor *input0, const ITensor *input1, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input0, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input0, input1);

    _input0 = input0;
    _input1 = input1;
    _output = output;

    // Configure kernel window
    Window win = calculate_max_window(*output->info());

    AccessWindowRectangle output_access(output->info(), 0, 0, 12, 8);

    const int input0_access_end = ceil_to_multiple(input0->info()->tensor_shape().x(), 8);
    const int input1_access_end = ceil_to_multiple(input1->info()->tensor_shape().x(), 12);

    update_window_and_padding(win,
                              AccessWindowStatic(input0->info(), 0, 0, input0_access_end, input0->info()->tensor_shape().y()),
                              AccessWindowStatic(input1->info(), 0, 0, input1_access_end, input1->info()->tensor_shape().y()),
                              output_access);

    INEKernel::configure(win);
}

bool NEGEMMLowpAArch64V8P4Kernel::is_parallelisable() const
{
    return false;
}

#define _UDOT_MACRO                                                                                    \
    ".altmacro\n"                                                                                      \
    ".macro udot opd:req, opn:req, opm:req\n"                                                          \
    "local vd, vn, vm, h, l\n"                                                                         \
    ".irp reg,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31\n" \
    ".ifeqs \"\\opd\",\"v\\reg\\.4s\"\n"                                                               \
    ".set vd,\\reg\n"                                                                                  \
    ".endif\n"                                                                                         \
    ".ifeqs \"\\opn\",\"v\\reg\\.16b\"\n"                                                              \
    ".set vn,\\reg\n"                                                                                  \
    ".endif\n"                                                                                         \
    ".irp idx,0,1,2,3\n"                                                                               \
    ".ifeqs \"\\opm\",\"v\\reg\\.4b[\\idx\\]\"\n"                                                      \
    ".set vm,\\reg\n"                                                                                  \
    ".set h,\\idx / 2\n"                                                                               \
    ".set l,\\idx %% 2\n"                                                                              \
    ".endif\n"                                                                                         \
    ".endr\n"                                                                                          \
    ".endr\n"                                                                                          \
    ".ifndef vd\n"                                                                                     \
    ".error \"Bad operand \\opd\"\n"                                                                   \
    ".exitm\n"                                                                                         \
    ".endif\n"                                                                                         \
    ".ifndef vn\n"                                                                                     \
    ".error \"Bad operand \\opn\"\n"                                                                   \
    ".exitm\n"                                                                                         \
    ".endif\n"                                                                                         \
    ".ifndef vm\n"                                                                                     \
    ".error \"Bad operand \\opm\"\n"                                                                   \
    ".exitm\n"                                                                                         \
    ".endif\n"                                                                                         \
    ".ifndef h\n"                                                                                      \
    ".error \"Bad operand \\opm\"\n"                                                                   \
    ".exitm\n"                                                                                         \
    ".endif\n"                                                                                         \
    ".ifndef l\n"                                                                                      \
    ".error \"Bad operand \\opm\"\n"                                                                   \
    ".exitm\n"                                                                                         \
    ".endif\n"                                                                                         \
    ".int    0x6f80e000 | vd | (vn << 5) | (vm << 16) | (l << 21) | (h << 11)\n"                       \
    ".endm\n"

#define _PREFETCH_                                     \
    __asm __volatile(                                  \
                                                       "" ASM_PREFETCH("[%[a_ptr], #64]")             \
                                                       ASM_PREFETCH("[%[a_ptr], #128]")           \
                                                       ASM_PREFETCH("[%[a_ptr], #192]")       \
                                                       :                                              \
                                                       :                                              \
                                                       [a_ptr] "r"(a_ptr), [b_ptr] "r"(b_ptr)         \
                                                       : "x20", "x21", "memory");                     \
    __asm __volatile(                                  \
                                                       "" ASM_PREFETCH("[%[b_ptr]]")                  \
                                                       ASM_PREFETCH("[%[b_ptr], #64]")            \
                                                       ASM_PREFETCH("[%[b_ptr], #128]")       \
                                                       ASM_PREFETCH("[%[b_ptr], #192]")   \
                                                       :                                              \
                                                       :                                              \
                                                       [b_ptr] "r"(b_ptr)                             \
                                                       : "x20", "x21");                               \
    __asm __volatile(                                  \
                                                       ""                                             \
                                                       : [r00] "+w"(r00), [r01] "+w"(r01),            \
                                                       [r10] "+w"(r10), [r11] "+w"(r11),            \
                                                       [r20] "+w"(r20), [r21] "+w"(r21),            \
                                                       [r30] "+w"(r30), [r31] "+w"(r31),            \
                                                       [a0] "+w"(a0), [a1] "+w"(a1),                \
                                                       [b0] "+w"(b0), [b1] "+w"(b1), [b2] "=w"(b2), \
                                                       [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr)     \
                                                       :                                              \
                                                       :);                                            \
    __asm __volatile(                                  \
                                                       ""                                             \
                                                       : [r02] "+w"(r02),                             \
                                                       [r12] "+w"(r12),                             \
                                                       [r22] "+w"(r22),                             \
                                                       [r32] "+w"(r32),                             \
                                                       [r40] "+w"(r40),                             \
                                                       [r50] "+w"(r50),                             \
                                                       [r60] "+w"(r60),                             \
                                                       [r70] "+w"(r70),                             \
                                                       [a0a] "=w"(a0a), [a1a] "=w"(a1a),            \
                                                       [b0] "+w"(b0), [b2] "+w"(b2), [b5] "=&w"(b5) \
                                                       :                                              \
                                                       :);                                            \
    __asm __volatile(                                  \
                                                       ""                                             \
                                                       :                                              \
                                                       [r41] "+w"(r41), [r42] "+w"(r42),              \
                                                       [r51] "+w"(r51), [r52] "+w"(r52),              \
                                                       [r61] "+w"(r61), [r62] "+w"(r62),              \
                                                       [r71] "+w"(r71), [r72] "+w"(r72),              \
                                                       [a1] "+w"(a1),                                 \
                                                       [b0] "+w"(b0), [b1] "+w"(b1), [b2] "+w"(b2),   \
                                                       [b_ptr] "+r"(b_ptr), [k] "+r"(k)               \
                                                       :                                              \
                                                       :);

void NEGEMMLowpAArch64V8P4Kernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const int x_block  = 348;
    const int k_block  = 1664;
    const int nthreads = 1;
    const int M        = _output->info()->tensor_shape().y();
    const int N        = _output->info()->tensor_shape().x();
    const int K        = _input0->info()->tensor_shape().x() >> 3;

    int yblocksperthread = ((M / nthreads) + 7) / 8;

    if(yblocksperthread < 1)
    {
        yblocksperthread = 1;
    }

    const int lda  = _input0->info()->strides_in_bytes().y();
    const int ldb  = _input1->info()->strides_in_bytes().y();
    const int ldc  = _output->info()->strides_in_bytes().y();
    const int ldc2 = _output->info()->strides_in_bytes().x();
    const int ldc3 = ldc / sizeof(uint32_t);

    const int threadid = 0;
    int       y0       = threadid * yblocksperthread * 8;
    int       ymax     = (threadid + 1) * yblocksperthread * 8;
    if(y0 >= M)
    {
        return;
    }
    if(ymax > M)
    {
        ymax = M;
    }
    for(int k0 = 0; k0 < K; k0 += k_block)
    {
        int kmax = k0 + k_block;
        if(kmax > K)
        {
            kmax = K;
        }

        for(int x0 = 0; x0 < N; x0 += x_block)
        {
            int xmax = x0 + x_block;
            if(xmax > N)
            {
                xmax = N;
            }

            for(int y = y0; y < ymax; y += 8)
            {
                auto      c_ptr0 = reinterpret_cast<uint32_t *>(_output->buffer() + (y * ldc) + x0 * ldc2);
                uint32_t *c_ptr1 = c_ptr0 + ldc3;
                uint32_t *c_ptr2 = c_ptr1 + ldc3;
                uint32_t *c_ptr3 = c_ptr2 + ldc3;
                uint32_t *c_ptr4 = c_ptr3 + ldc3;
                uint32_t *c_ptr5 = c_ptr4 + ldc3;
                uint32_t *c_ptr6 = c_ptr5 + ldc3;
                uint32_t *c_ptr7 = c_ptr6 + ldc3;

                __asm __volatile(
                    "" ASM_PREFETCH("[%[c_ptr0]]")
                    ASM_PREFETCH("[%[c_ptr1]]")
                    ASM_PREFETCH("[%[c_ptr2]]")
                    ASM_PREFETCH("[%[c_ptr3]]")
                    ASM_PREFETCH("[%[c_ptr4]]")
                    ASM_PREFETCH("[%[c_ptr5]]")
                    ASM_PREFETCH("[%[c_ptr6]]")
                    ASM_PREFETCH("[%[c_ptr7]]")
                    :
                    : [c_ptr0] "r"(c_ptr0), [c_ptr1] "r"(c_ptr1), [c_ptr2] "r"(c_ptr2), [c_ptr3] "r"(c_ptr3),
                    [c_ptr4] "r"(c_ptr4), [c_ptr5] "r"(c_ptr5), [c_ptr6] "r"(c_ptr6), [c_ptr7] "r"(c_ptr7)
                    : "x20", "x21");

                for(int x = x0; x < xmax; x += 12)
                {
                    register uint32x4_t r00 asm("v8");
                    register uint32x4_t r10 asm("v9");
                    register uint32x4_t r20 asm("v10");
                    register uint32x4_t r30 asm("v11");
                    register uint32x4_t r40 asm("v12");
                    register uint32x4_t r50 asm("v13");
                    register uint32x4_t r60 asm("v14");
                    register uint32x4_t r70 asm("v15");
                    register uint32x4_t r01 asm("v16");
                    register uint32x4_t r11 asm("v17");
                    register uint32x4_t r21 asm("v18");
                    register uint32x4_t r31 asm("v19");
                    register uint32x4_t r41 asm("v20");
                    register uint32x4_t r51 asm("v21");
                    register uint32x4_t r61 asm("v22");
                    register uint32x4_t r71 asm("v23");
                    register uint32x4_t r02 asm("v24");
                    register uint32x4_t r12 asm("v25");
                    register uint32x4_t r22 asm("v26");
                    register uint32x4_t r32 asm("v27");
                    register uint32x4_t r42 asm("v28");
                    register uint32x4_t r52 asm("v29");
                    register uint32x4_t r62 asm("v30");
                    register uint32x4_t r72 asm("v31");

                    register uint8x16_t a0 asm("v0");
                    register uint8x16_t a1 asm("v1");
                    register uint8x16_t b0 asm("v2");
                    register uint8x16_t b1 asm("v3");
                    register uint8x16_t b2 asm("v4");
                    register uint8x16_t a0a asm("v5");
                    register uint8x16_t a1a asm("v6");
                    register uint8x16_t b5 asm("v7");
                    const uint8_t      *a_ptr = _input0->buffer() + ((y / 8) * lda) + (k0 * 8);
                    const uint8_t      *b_ptr = _input1->buffer() + ((x / 12) * ldb) + (k0 * 12);

                    r00 = r01 = r02 = r10 = r11 = r12 = r20 = r21 = r22 = r30 = r31 = r32 = vdupq_n_u32(0);
                    r40 = r41 = r42 = r50 = r51 = r52 = r60 = r61 = r62 = r70 = r71 = r72 = vdupq_n_u32(0);

                    int k = ((kmax - k0) / 8) - 1;

                    a0 = vld1q_u8(a_ptr);
                    b0 = vld1q_u8(b_ptr);
                    a1 = vld1q_u8(a_ptr + 16);
                    b1 = vld1q_u8(b_ptr + 16);

                    _PREFETCH_

                    __asm __volatile(
                        _UDOT_MACRO
                        "1:\n"
                        "udot    v8.4s , %[b0].16b, %[a0].4b[0]\n"
                        "udot    v9.4s , %[b0].16b, %[a0].4b[1]\n"
                        "ldr    %q[b2], [%[b_ptr], #32]\n"
                        "udot    v10.4s, %[b0].16b, %[a0].4b[2]\n"
                        "udot    v11.4s, %[b0].16b, %[a0].4b[3]\n"
                        "ldr    %q[a0a], [%[a_ptr], #32]\n"
                        "udot    v12.4s, %[b0].16b, %[a1].4b[0]\n"
                        "udot    v13.4s, %[b0].16b, %[a1].4b[1]\n"
                        "ldr    %q[a1a], [%[a_ptr], #48]\n"
                        "udot    v14.4s, %[b0].16b, %[a1].4b[2]\n"
                        "udot    v15.4s, %[b0].16b, %[a1].4b[3]\n"
                        "ldr    %q[b0], [%[b_ptr], #48]\n"

                        "udot    v16.4s, %[b1].16b, %[a0].4b[0]\n"
                        "udot    v17.4s, %[b1].16b, %[a0].4b[1]\n" ASM_PREFETCH("[%[a_ptr], #256]")
                        "udot    v18.4s, %[b1].16b, %[a0].4b[2]\n"
                        "udot    v19.4s, %[b1].16b, %[a0].4b[3]\n"
                        "udot    v20.4s, %[b1].16b, %[a1].4b[0]\n"
                        "udot    v21.4s, %[b1].16b, %[a1].4b[1]\n"
                        "udot    v22.4s, %[b1].16b, %[a1].4b[2]\n"
                        "udot    v23.4s, %[b1].16b, %[a1].4b[3]\n"
                        "ldr    %q[b1], [%[b_ptr], #64]\n"

                        "udot    v24.4s, %[b2].16b, %[a0].4b[0]\n"
                        "udot    v25.4s, %[b2].16b, %[a0].4b[1]\n" ASM_PREFETCH("[%[b_ptr], #256]")
                        "udot    v26.4s, %[b2].16b, %[a0].4b[2]\n"
                        "udot    v27.4s, %[b2].16b, %[a0].4b[3]\n"
                        "udot    v28.4s, %[b2].16b, %[a1].4b[0]\n"
                        "udot    v29.4s, %[b2].16b, %[a1].4b[1]\n"
                        "udot    v30.4s, %[b2].16b, %[a1].4b[2]\n"
                        "udot    v31.4s, %[b2].16b, %[a1].4b[3]\n"
                        "ldr    %q[b2], [%[b_ptr], #80]\n"

                        "udot    v8.4s , %[b0].16b, %[a0a].4b[0]\n"
                        "udot    v9.4s , %[b0].16b, %[a0a].4b[1]\n"
                        "ldr    %q[a0], [%[a_ptr], #64]\n"
                        "udot    v10.4s, %[b0].16b, %[a0a].4b[2]\n"
                        "udot    v11.4s, %[b0].16b, %[a0a].4b[3]\n"
                        "udot    v12.4s, %[b0].16b, %[a1a].4b[0]\n"
                        "ldr    %q[a1], [%[a_ptr], #80]\n"
                        "udot    v13.4s, %[b0].16b, %[a1a].4b[1]\n"
                        "udot    v14.4s, %[b0].16b, %[a1a].4b[2]\n"
                        "udot    v15.4s, %[b0].16b, %[a1a].4b[3]\n"
                        "ldr    %q[b0], [%[b_ptr], #96]\n"

                        "udot    v16.4s, %[b1].16b, %[a0a].4b[0]\n"
                        "udot    v17.4s, %[b1].16b, %[a0a].4b[1]\n" ASM_PREFETCH("[%[b_ptr], #320]")
                        "udot    v18.4s, %[b1].16b, %[a0a].4b[2]\n"
                        "udot    v19.4s, %[b1].16b, %[a0a].4b[3]\n"
                        "udot    v20.4s, %[b1].16b, %[a1a].4b[0]\n"
                        "udot    v21.4s, %[b1].16b, %[a1a].4b[1]\n"
                        "udot    v22.4s, %[b1].16b, %[a1a].4b[2]\n"
                        "udot    v23.4s, %[b1].16b, %[a1a].4b[3]\n"
                        "ldr    %q[b1], [%[b_ptr], #112]\n"

                        "udot    v24.4s, %[b2].16b, %[a0a].4b[0]\n"
                        "udot    v25.4s, %[b2].16b, %[a0a].4b[1]\n"
                        "add    %[a_ptr], %[a_ptr], #64\n"
                        "udot    v26.4s, %[b2].16b, %[a0a].4b[2]\n"
                        "udot    v27.4s, %[b2].16b, %[a0a].4b[3]\n"
                        "add    %[b_ptr], %[b_ptr], #96\n"
                        "udot    v28.4s, %[b2].16b, %[a1a].4b[0]\n"
                        "udot    v29.4s, %[b2].16b, %[a1a].4b[1]\n"
                        "subs    %w[k], %w[k], #1\n"
                        "udot    v30.4s, %[b2].16b, %[a1a].4b[2]\n"
                        "udot    v31.4s, %[b2].16b, %[a1a].4b[3]\n"

                        "bne    1b\n"

                        "udot    v8.4s , %[b0].16b, %[a0].4b[0]\n"
                        "udot    v9.4s , %[b0].16b, %[a0].4b[1]\n"
                        "ldr    %q[b2], [%[b_ptr], #32]\n"
                        "udot    v10.4s, %[b0].16b, %[a0].4b[2]\n"
                        "udot    v11.4s, %[b0].16b, %[a0].4b[3]\n"
                        "ldr    %q[a0a], [%[a_ptr], #32]\n"
                        "udot    v12.4s, %[b0].16b, %[a1].4b[0]\n"
                        "udot    v13.4s, %[b0].16b, %[a1].4b[1]\n"
                        "ldr    %q[a1a], [%[a_ptr], #48]\n"
                        "udot    v14.4s, %[b0].16b, %[a1].4b[2]\n"
                        "udot    v15.4s, %[b0].16b, %[a1].4b[3]\n"
                        "ldr    %q[b0], [%[b_ptr], #48]\n"

                        "udot    v16.4s, %[b1].16b, %[a0].4b[0]\n"
                        "udot    v17.4s, %[b1].16b, %[a0].4b[1]\n"
                        "udot    v18.4s, %[b1].16b, %[a0].4b[2]\n"
                        "udot    v19.4s, %[b1].16b, %[a0].4b[3]\n"
                        "udot    v20.4s, %[b1].16b, %[a1].4b[0]\n"
                        "udot    v21.4s, %[b1].16b, %[a1].4b[1]\n"
                        "udot    v22.4s, %[b1].16b, %[a1].4b[2]\n"
                        "udot    v23.4s, %[b1].16b, %[a1].4b[3]\n"
                        "ldr    %q[b1], [%[b_ptr], #64]\n"

                        "udot    v24.4s, %[b2].16b, %[a0].4b[0]\n"
                        "udot    v25.4s, %[b2].16b, %[a0].4b[1]\n"
                        "udot    v26.4s, %[b2].16b, %[a0].4b[2]\n"
                        "udot    v27.4s, %[b2].16b, %[a0].4b[3]\n"
                        "udot    v28.4s, %[b2].16b, %[a1].4b[0]\n"
                        "udot    v29.4s, %[b2].16b, %[a1].4b[1]\n"
                        "udot    v30.4s, %[b2].16b, %[a1].4b[2]\n"
                        "udot    v31.4s, %[b2].16b, %[a1].4b[3]\n"
                        "ldr    %q[b2], [%[b_ptr], #80]\n"

                        "udot    v8.4s , %[b0].16b, %[a0a].4b[0]\n" ASM_PREFETCH("[%[c_ptr0]]") "udot    v9.4s , %[b0].16b, %[a0a].4b[1]\n" ASM_PREFETCH("[%[c_ptr1]]") "udot    v10.4s, %[b0].16b, %[a0a].4b[2]\n"
                        ASM_PREFETCH("[%[c_ptr2]]") "udot    v11.4s, %[b0].16b, %[a0a].4b[3]\n" ASM_PREFETCH("[%[c_ptr3]]") "udot    v12.4s, %[b0].16b, %[a1a].4b[0]\n" ASM_PREFETCH("[%[c_ptr4]]")
                        "udot    v13.4s, %[b0].16b, %[a1a].4b[1]\n" ASM_PREFETCH("[%[c_ptr5]]") "udot    v14.4s, %[b0].16b, %[a1a].4b[2]\n" ASM_PREFETCH("[%[c_ptr6]]") "udot    v15.4s, %[b0].16b, %[a1a].4b[3]\n"
                        ASM_PREFETCH("[%[c_ptr7]]")

                        "udot    v16.4s, %[b1].16b, %[a0a].4b[0]\n" ASM_PREFETCH("[%[c_ptr0], #48]") "udot    v17.4s, %[b1].16b, %[a0a].4b[1]\n" ASM_PREFETCH("[%[c_ptr1], #48]") "udot    v18.4s, %[b1].16b, %[a0a].4b[2]\n"
                        ASM_PREFETCH("[%[c_ptr2], #48]") "udot    v19.4s, %[b1].16b, %[a0a].4b[3]\n" ASM_PREFETCH("[%[c_ptr3], #48]") "udot    v20.4s, %[b1].16b, %[a1a].4b[0]\n" ASM_PREFETCH("[%[c_ptr4], #48]")
                        "udot    v21.4s, %[b1].16b, %[a1a].4b[1]\n" ASM_PREFETCH("[%[c_ptr5], #48]") "udot    v22.4s, %[b1].16b, %[a1a].4b[2]\n" ASM_PREFETCH("[%[c_ptr6], #48]") "udot    v23.4s, %[b1].16b, %[a1a].4b[3]\n"
                        ASM_PREFETCH("[%[c_ptr7], #48]")

                        "udot    v24.4s, %[b2].16b, %[a0a].4b[0]\n"
                        "udot    v25.4s, %[b2].16b, %[a0a].4b[1]\n"
                        "udot    v26.4s, %[b2].16b, %[a0a].4b[2]\n"
                        "udot    v27.4s, %[b2].16b, %[a0a].4b[3]\n"
                        "add    %[b_ptr], %[b_ptr], #96\n"
                        "udot    v28.4s, %[b2].16b, %[a1a].4b[0]\n"
                        "udot    v29.4s, %[b2].16b, %[a1a].4b[1]\n"
                        "udot    v30.4s, %[b2].16b, %[a1a].4b[2]\n"
                        "udot    v31.4s, %[b2].16b, %[a1a].4b[3]\n"

                        // Clean up macro namespace
                        ".purgem udot\n"

                        :
                        [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
                        [a0] "+w"(a0), [a1] "+w"(a1), [a0a] "+w"(a0a), [a1a] "+w"(a1a),
                        [b0] "+w"(b0), [b1] "+w"(b1), [b2] "+w"(b2), [k] "+r"(k)
                        : [c_ptr0] "r"(c_ptr0), [c_ptr1] "r"(c_ptr1), [c_ptr2] "r"(c_ptr2), [c_ptr3] "r"(c_ptr3),
                        [c_ptr4] "r"(c_ptr4), [c_ptr5] "r"(c_ptr5), [c_ptr6] "r"(c_ptr6), [c_ptr7] "r"(c_ptr7)
                        : "x20", "x21");

                    stincpld(r00, r10, r20, r30, r40, r50, r60, r70, c_ptr0, c_ptr1, c_ptr2, c_ptr3, c_ptr4, c_ptr5, c_ptr6, c_ptr7);
                    stinc(r01, r11, r21, r31, r41, r51, r61, r71, c_ptr0, c_ptr1, c_ptr2, c_ptr3, c_ptr4, c_ptr5, c_ptr6, c_ptr7);
                    stinc(r02, r12, r22, r32, r42, r52, r62, r72, c_ptr0, c_ptr1, c_ptr2, c_ptr3, c_ptr4, c_ptr5, c_ptr6, c_ptr7);
                }
            }
        }
    }
}
} // namespace arm_compute
#endif /* ARM_COMPUTE_AARCH64_V8_2 */
