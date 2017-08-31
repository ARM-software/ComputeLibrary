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
#pragma once

#ifdef __aarch64__
// Macro to use in assembler to get a preload.  Needed because of various
// workarounds needed to get working preload behaviour.
//
// Code using these macros needs to clobber x20 and x21 as they might be
// used by the workaround.

#define ASM_PREFETCH(address)    "PRFM PLDL1KEEP, " address "\n"
#define ASM_PREFETCHL2(address)  "PRFM PLDL2KEEP, " address "\n"
#define ASM_PREFETCHW(address)   "PRFM PSTL1KEEP, " address "\n"
#define ASM_PREFETCHWL2(address) "PRFM PSTL2KEEP, " address "\n"

#else

#define ASM_PREFETCH(address)     "PLD " address "\n"
#define ASM_PREFETCHW(address)    "PLDW " address "\n"

#endif

/*
 * Do some prefetches.
 */
template <typename T>
static inline void prefetch_6x(const T *pfp) {
    __asm __volatile (
        ASM_PREFETCH("[%[pfp]]")
        ASM_PREFETCH("[%[pfp], #64]")
        ASM_PREFETCH("[%[pfp], #128]")
        ASM_PREFETCH("[%[pfp], #192]")
        ASM_PREFETCH("[%[pfp], #256]")
        ASM_PREFETCH("[%[pfp], #320]")
    :
    : [pfp] "r" (pfp)
    : "memory"
    );
}

template <typename T>
static inline void prefetch_5x(const T *pfp) {
    __asm __volatile (
        ASM_PREFETCH("[%[pfp]]")
        ASM_PREFETCH("[%[pfp], #64]")
        ASM_PREFETCH("[%[pfp], #128]")
        ASM_PREFETCH("[%[pfp], #192]")
        ASM_PREFETCH("[%[pfp], #256]")
    :
    : [pfp] "r" (pfp)
    : "memory"
    );
}

template <typename T>
static inline void prefetch_4x(const T *pfp) {
    __asm __volatile (
        ASM_PREFETCH("[%[pfp]]")
        ASM_PREFETCH("[%[pfp], #64]")
        ASM_PREFETCH("[%[pfp], #128]")
        ASM_PREFETCH("[%[pfp], #192]")
    :
    : [pfp] "r" (pfp)
    : "memory"
    );
}

template <typename T>
static inline void prefetch_3x(const T *pfp) {
    __asm __volatile (
        ASM_PREFETCH("[%[pfp]]")
        ASM_PREFETCH("[%[pfp], #64]")
        ASM_PREFETCH("[%[pfp], #128]")
    :
    : [pfp] "r" (pfp)
    : "memory"
    );
}

template <typename T>
static inline void prefetch_2x(const T *pfp) {
    __asm __volatile (
        ASM_PREFETCH("[%[pfp]]")
        ASM_PREFETCH("[%[pfp], #64]")
    :
    : [pfp] "r" (pfp)
    : "memory"
    );
}

template <typename T>
static inline void prefetch_1x(const T *pfp) {
    __asm __volatile (
        ASM_PREFETCH("[%[pfp]]")
    :
    : [pfp] "r" (pfp)
    : "memory"
    );
}
