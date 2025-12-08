//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if defined(_MSC_VER)
#  define KAI_ASM_CODE(name) AREA name, CODE, READONLY
#  define KAI_ASM_LABEL(label) label
#  define KAI_ASM_LABEL_GLOBAL(label) label
#  define KAI_ASM_GLOBAL(symbol) global symbol
#  define KAI_ASM_END end
#elif defined(__APPLE__)
#  define KAI_ASM_CODE(name) .text
#  define KAI_ASM_LABEL(label) _##label:
#  define KAI_ASM_LABEL_GLOBAL(label) _##label:
#  define KAI_ASM_GLOBAL(symbol) .global _##symbol
#  define KAI_ASM_END
#else
#  define KAI_ASM_CODE(name) .text
#  define KAI_ASM_LABEL(label) label:
#  define KAI_ASM_LABEL_GLOBAL(label) label:
#  define KAI_ASM_GLOBAL(symbol) .global symbol
#  define KAI_ASM_END
#endif

    KAI_ASM_CODE(kai_test_common_float16)

    KAI_ASM_GLOBAL(kai_test_float16_from_float)
    KAI_ASM_GLOBAL(kai_test_float_from_float16)
    KAI_ASM_GLOBAL(kai_test_float16_add)
    KAI_ASM_GLOBAL(kai_test_float16_sub)
    KAI_ASM_GLOBAL(kai_test_float16_mul)
    KAI_ASM_GLOBAL(kai_test_float16_div)
    KAI_ASM_GLOBAL(kai_test_float16_lt)
    KAI_ASM_GLOBAL(kai_test_float16_gt)

KAI_ASM_LABEL_GLOBAL(kai_test_float16_from_float)
    fcvt h0, s0
    fmov w0, s0  // Use s0 instead of h0 to avoid unnecessary need for FEAT_FP16.
    ret

KAI_ASM_LABEL_GLOBAL(kai_test_float_from_float16)
    fmov s0, w0  // Use s0 instead of h0 to avoid unnecessary need for FEAT_FP16.
    fcvt s0, h0
    ret

KAI_ASM_LABEL_GLOBAL(kai_test_float16_add)
    fmov h0, w0
    fmov h1, w1
    fadd h0, h0, h1
    fmov w0, h0
    ret

KAI_ASM_LABEL_GLOBAL(kai_test_float16_sub)
    fmov h0, w0
    fmov h1, w1
    fsub h0, h0, h1
    fmov w0, h0
    ret

KAI_ASM_LABEL_GLOBAL(kai_test_float16_mul)
    fmov h0, w0
    fmov h1, w1
    fmul h0, h0, h1
    fmov w0, h0
    ret

KAI_ASM_LABEL_GLOBAL(kai_test_float16_div)
    fmov h0, w0
    fmov h1, w1
    fdiv h0, h0, h1
    fmov w0, h0
    ret

KAI_ASM_LABEL_GLOBAL(kai_test_float16_lt)
    fmov h0, w0
    fmov h1, w1
    fcmp h0, h1
    cset x0, lt
    ret

KAI_ASM_LABEL_GLOBAL(kai_test_float16_gt)
    fmov h0, w0
    fmov h1, w1
    fcmp h0, h1
    cset x0, gt
    ret

    KAI_ASM_END
