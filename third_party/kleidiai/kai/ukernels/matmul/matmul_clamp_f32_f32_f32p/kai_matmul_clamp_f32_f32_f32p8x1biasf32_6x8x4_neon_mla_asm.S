//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if defined(_MSC_VER) && !defined(__clang__)

#define KAI_ASM_HEADER AREA |.text|, CODE, READONLY, ALIGN=4
#define KAI_ASM_LABEL(label) |label|
#define KAI_ASM_FUNCTION(label) |label|
#define KAI_ASM_EXPORT(label) global label
#define KAI_ASM_FOOTER end
#define KAI_ASM_INST(num) dcd num

#elif defined(_MSC_VER) && defined(__clang__)

#define KAI_ASM_HEADER        .text
#define KAI_ASM_LABEL(label)  label:
#define KAI_ASM_FUNCTION(label) label:
#define KAI_ASM_EXPORT(label) .globl label
#define KAI_ASM_FOOTER
#define KAI_ASM_INST(num)     .inst num

#else  // _MSC_VER

#define KAI_ASM_HEADER .text
#define KAI_ASM_LABEL(label) label:

#ifdef __APPLE__
#define KAI_ASM_FUNCTION(label) _##label:
#define KAI_ASM_EXPORT(label) \
    .global _##label;         \
    .type _##label, %function
#else  // __APPLE__
#define KAI_ASM_FUNCTION(label) label:
#define KAI_ASM_EXPORT(label) \
    .global label;            \
    .type label, %function
#endif  // __APPLE__

#define KAI_ASM_FOOTER
#define KAI_ASM_INST(num) .inst num

#endif  // _MSC_VER

    KAI_ASM_HEADER

    KAI_ASM_EXPORT(kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla_impl)

KAI_ASM_FUNCTION(kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla_impl)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d10, d11, [sp, 72]
    stp d12, d13, [sp, 88]
    stp d14, d15, [sp, 104]
    stp d8, d9, [sp, 120]
KAI_ASM_LABEL(label_1)  // Row loop
    cmp x1, #0x6
    bge label_126
    cmp x1, #0x4
    bgt label_101
    beq label_76
    cmp x1, #0x2
    bgt label_51
    beq label_26
    ldr x11, [x2, #0x18]
    ldr x10, [x2, #0x20]
    ldr x9, [x2, #0x40]
KAI_ASM_LABEL(label_2)  // Height 1: Column loop
    cbz x10, label_3
    ldr q20, [x10, #0x0]
    ldr q21, [x10, #0x10]
    add x10, x10, #0x20
    b label_10
KAI_ASM_LABEL(label_3)  // Height 1: no bias
    tbz x3, #0, label_9
    cmp x11, #0x8
    bge label_8
    tbz x11, #2, label_5
    ld1 { v20.4s }, [x9], #0x10
    tbz x11, #1, label_4
    ldr d21, [x9], #0x8
    mov x20, #0x18
    tbz x11, #0, label_7
    ld1 { v21.s }[2], [x9]
    b label_7
KAI_ASM_LABEL(label_4)  // Height 1: Partial accumulate: partial_1_4
    mov x20, #0x10
    tbz x11, #0, label_7
    ldr s21, [x9, #0x0]
    b label_7
KAI_ASM_LABEL(label_5)  // Height 1: Partial accumulate: partial_2_0
    tbz x11, #1, label_6
    ldr d20, [x9], #0x8
    mov x20, #0x8
    tbz x11, #0, label_7
    ld1 { v20.s }[2], [x9]
    b label_7
KAI_ASM_LABEL(label_6)  // Height 1: Partial accumulate: partial_1_0
    ldr s20, [x9, #0x0]
    mov x20, #0x0
KAI_ASM_LABEL(label_7)  // Height 1: Partial accumulate: Done
    sub x9, x9, x20
    b label_10
KAI_ASM_LABEL(label_8)  // Height 1: full accumulate
    ldr q20, [x9, #0x0]
    ldr q21, [x9, #0x10]
    b label_10
KAI_ASM_LABEL(label_9)  // Height 1: no accumulate
    movi v20.16b, #0x0
    movi v21.16b, #0x0
KAI_ASM_LABEL(label_10)  // Height 1: setup done
    mov x28, #0x0
KAI_ASM_LABEL(label_11)  // Height 1: String loop
    ldr x20, [x2, #0x10]
    ldr x21, [x2, #0x38]
    ldr w27, [x20, x28, LSL #0x2]
    tbz x3, #3, label_12
    ldr x20, [x0, x28, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x26, [x20, #0x0]
    cbnz x28, label_13
    ldr x20, [x2, #0x30]
    add x26, x26, x20, LSL #2
    b label_13
KAI_ASM_LABEL(label_12)  // Height 1: setup direct input
    mov x26, x0
KAI_ASM_LABEL(label_13)  // Height 1: input setup done
    cmp x27, #0x4
    blt label_16
    ldr q0, [x26, #0x0]
    ldr q6, [x10, #0x0]
    cmp x27, #0x8
    ldr q7, [x10, #0x10]
    ldr q8, [x10, #0x20]
    ldr q9, [x10, #0x30]
    ldr q10, [x10, #0x40]
    ldr q11, [x10, #0x50]
    ldr q12, [x10, #0x60]
    ldr q13, [x10, #0x70]
    blt label_15
KAI_ASM_LABEL(label_14)  // Height 1: Multiply loop: Main loop head
    fmla v20.4s, v6.4s, v0.s[0]
    fmla v21.4s, v7.4s, v0.s[0]
    sub x27, x27, #0x4
    add x26, x26, #0x10
    cmp x27, #0x8
    add x10, x10, #0x80
    prfm pldl1keep, [x26, #0x80]
    ldr q6, [x10, #0x0]
    ldr q7, [x10, #0x10]
    fmla v20.4s, v8.4s, v0.s[1]
    ldr q8, [x10, #0x20]
    fmla v21.4s, v9.4s, v0.s[1]
    ldr q9, [x10, #0x30]
    fmla v20.4s, v10.4s, v0.s[2]
    ldr q10, [x10, #0x40]
    fmla v21.4s, v11.4s, v0.s[2]
    ldr q11, [x10, #0x50]
    fmla v20.4s, v12.4s, v0.s[3]
    ldr q12, [x10, #0x60]
    fmla v21.4s, v13.4s, v0.s[3]
    ldr q0, [x26, #0x0]
    ldr q13, [x10, #0x70]
    bge label_14
KAI_ASM_LABEL(label_15)  // Height 1: Multiply loop: Single iteration only
    fmla v20.4s, v6.4s, v0.s[0]
    fmla v21.4s, v7.4s, v0.s[0]
    add x26, x26, #0x10
    sub x27, x27, #0x4
    add x10, x10, #0x80
    prfm pldl1keep, [x26, #0x80]
    fmla v20.4s, v8.4s, v0.s[1]
    fmla v21.4s, v9.4s, v0.s[1]
    fmla v20.4s, v10.4s, v0.s[2]
    fmla v21.4s, v11.4s, v0.s[2]
    fmla v20.4s, v12.4s, v0.s[3]
    fmla v21.4s, v13.4s, v0.s[3]
KAI_ASM_LABEL(label_16)  // Height 1: Multiply loop: Main loop skip
    cbz x27, label_18
KAI_ASM_LABEL(label_17)  // Height 1: Multiply loop: Odd block loop
    ldr s0, [x26], #0x4
    ldr q14, [x10, #0x0]
    sub x27, x27, #0x1
    ldr q15, [x10, #0x10]
    add x10, x10, #0x20
    fmla v20.4s, v14.4s, v0.s[0]
    fmla v21.4s, v15.4s, v0.s[0]
    cbnz x27, label_17
KAI_ASM_LABEL(label_18)  // Height 1: Multiply loop: No odd multiplies
    ldr w20, [x2, #0x8]
    add x28, x28, #0x1
    cmp x28, x20
    bne label_11
    prfm pstl1keep, [x9, #0x0]
    tbz x3, #1, label_19
    add x21, x2, #0x0
    add x20, x2, #0x4
    ld1r { v17.4s }, [x21]
    ld1r { v16.4s }, [x20]
    fmin v20.4s, v20.4s, v17.4s
    fmin v21.4s, v21.4s, v17.4s
    fmax v20.4s, v20.4s, v16.4s
    fmax v21.4s, v21.4s, v16.4s
KAI_ASM_LABEL(label_19)  // Height 1: No activation
    cmp x11, #0x8
    bge label_24
    tbz x11, #2, label_21
    st1 { v20.4s }, [x9], #0x10
    tbz x11, #1, label_20
    str d21, [x9], #0x8
    tbz x11, #0, label_23
    st1 { v21.s }[2], [x9]
    b label_23
KAI_ASM_LABEL(label_20)  // Height 1: Partial direct writeback: partial_1_4
    tbz x11, #0, label_23
    str s21, [x9, #0x0]
    b label_23
KAI_ASM_LABEL(label_21)  // Height 1: Partial direct writeback: partial_2_0
    tbz x11, #1, label_22
    str d20, [x9], #0x8
    tbz x11, #0, label_23
    st1 { v20.s }[2], [x9]
    b label_23
KAI_ASM_LABEL(label_22)  // Height 1: Partial direct writeback: partial_1_0
    str s20, [x9, #0x0]
KAI_ASM_LABEL(label_23)  // Height 1: Partial direct writeback: Done
    b label_25
KAI_ASM_LABEL(label_24)  // Height 1: Full writeback
    str q20, [x9, #0x0]
    str q21, [x9, #0x10]
    add x9, x9, #0x20
KAI_ASM_LABEL(label_25)  // Height 1: Writeback done
    subs x11, x11, #0x8
    bgt label_2
    b label_152
KAI_ASM_LABEL(label_26)  // Height 2
    ldr x11, [x2, #0x18]
    ldr x10, [x2, #0x20]
    ldr x9, [x2, #0x40]
KAI_ASM_LABEL(label_27)  // Height 2: Column loop
    cbz x10, label_28
    ldr q20, [x10, #0x0]
    ldr q21, [x10, #0x10]
    add x10, x10, #0x20
    mov v22.16b, v20.16b
    mov v23.16b, v21.16b
    b label_35
KAI_ASM_LABEL(label_28)  // Height 2: no bias
    tbz x3, #0, label_34
    ldr x20, [x2, #0x28]
    cmp x11, #0x8
    add x26, x9, x20, LSL #2
    bge label_33
    tbz x11, #2, label_30
    ld1 { v20.4s }, [x9], #0x10
    ld1 { v22.4s }, [x26], #0x10
    tbz x11, #1, label_29
    ldr d21, [x9], #0x8
    ldr d23, [x26], #0x8
    mov x20, #0x18
    tbz x11, #0, label_32
    ld1 { v21.s }[2], [x9]
    ld1 { v23.s }[2], [x26]
    b label_32
KAI_ASM_LABEL(label_29)  // Height 2: Partial accumulate: partial_1_4
    mov x20, #0x10
    tbz x11, #0, label_32
    ldr s21, [x9, #0x0]
    ldr s23, [x26, #0x0]
    b label_32
KAI_ASM_LABEL(label_30)  // Height 2: Partial accumulate: partial_2_0
    tbz x11, #1, label_31
    ldr d20, [x9], #0x8
    ldr d22, [x26], #0x8
    mov x20, #0x8
    tbz x11, #0, label_32
    ld1 { v20.s }[2], [x9]
    ld1 { v22.s }[2], [x26]
    b label_32
KAI_ASM_LABEL(label_31)  // Height 2: Partial accumulate: partial_1_0
    ldr s20, [x9, #0x0]
    ldr s22, [x26, #0x0]
    mov x20, #0x0
KAI_ASM_LABEL(label_32)  // Height 2: Partial accumulate: Done
    sub x9, x9, x20
    b label_35
KAI_ASM_LABEL(label_33)  // Height 2: full accumulate
    ldr q20, [x9, #0x0]
    ldr q21, [x9, #0x10]
    ldr q22, [x26, #0x0]
    ldr q23, [x26, #0x10]
    b label_35
KAI_ASM_LABEL(label_34)  // Height 2: no accumulate
    movi v20.16b, #0x0
    movi v21.16b, #0x0
    movi v22.16b, #0x0
    movi v23.16b, #0x0
KAI_ASM_LABEL(label_35)  // Height 2: setup done
    mov x28, #0x0
KAI_ASM_LABEL(label_36)  // Height 2: String loop
    ldr x20, [x2, #0x10]
    ldr x21, [x2, #0x38]
    ldr w27, [x20, x28, LSL #0x2]
    tbz x3, #3, label_37
    ldr x20, [x0, x28, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x26, [x20, #0x0]
    ldr x25, [x20, #0x8]
    cbnz x28, label_38
    ldr x20, [x2, #0x30]
    add x26, x26, x20, LSL #2
    add x25, x25, x20, LSL #2
    b label_38
KAI_ASM_LABEL(label_37)  // Height 2: setup direct input
    mov x26, x0
    add x25, x26, x21, LSL #2
KAI_ASM_LABEL(label_38)  // Height 2: input setup done
    cmp x27, #0x4
    blt label_41
    ldr q0, [x26, #0x0]
    ldr q1, [x25, #0x0]
    cmp x27, #0x8
    ldr q6, [x10, #0x0]
    ldr q7, [x10, #0x10]
    ldr q8, [x10, #0x20]
    ldr q9, [x10, #0x30]
    ldr q10, [x10, #0x40]
    ldr q11, [x10, #0x50]
    ldr q12, [x10, #0x60]
    ldr q13, [x10, #0x70]
    blt label_40
KAI_ASM_LABEL(label_39)  // Height 2: Multiply loop: Main loop head
    fmla v20.4s, v6.4s, v0.s[0]
    fmla v22.4s, v6.4s, v1.s[0]
    sub x27, x27, #0x4
    add x26, x26, #0x10
    fmla v21.4s, v7.4s, v0.s[0]
    fmla v23.4s, v7.4s, v1.s[0]
    add x25, x25, #0x10
    cmp x27, #0x8
    add x10, x10, #0x80
    prfm pldl1keep, [x26, #0x80]
    prfm pldl1keep, [x25, #0x80]
    ldr q6, [x10, #0x0]
    ldr q7, [x10, #0x10]
    fmla v20.4s, v8.4s, v0.s[1]
    fmla v22.4s, v8.4s, v1.s[1]
    ldr q8, [x10, #0x20]
    fmla v21.4s, v9.4s, v0.s[1]
    fmla v23.4s, v9.4s, v1.s[1]
    ldr q9, [x10, #0x30]
    fmla v20.4s, v10.4s, v0.s[2]
    fmla v22.4s, v10.4s, v1.s[2]
    ldr q10, [x10, #0x40]
    fmla v21.4s, v11.4s, v0.s[2]
    fmla v23.4s, v11.4s, v1.s[2]
    ldr q11, [x10, #0x50]
    fmla v20.4s, v12.4s, v0.s[3]
    fmla v22.4s, v12.4s, v1.s[3]
    ldr q12, [x10, #0x60]
    fmla v21.4s, v13.4s, v0.s[3]
    ldr q0, [x26, #0x0]
    fmla v23.4s, v13.4s, v1.s[3]
    ldr q1, [x25, #0x0]
    ldr q13, [x10, #0x70]
    bge label_39
KAI_ASM_LABEL(label_40)  // Height 2: Multiply loop: Single iteration only
    fmla v20.4s, v6.4s, v0.s[0]
    fmla v22.4s, v6.4s, v1.s[0]
    add x26, x26, #0x10
    add x25, x25, #0x10
    fmla v21.4s, v7.4s, v0.s[0]
    fmla v23.4s, v7.4s, v1.s[0]
    sub x27, x27, #0x4
    prfm pldl1keep, [x26, #0x80]
    prfm pldl1keep, [x25, #0x80]
    add x10, x10, #0x80
    fmla v20.4s, v8.4s, v0.s[1]
    fmla v22.4s, v8.4s, v1.s[1]
    fmla v21.4s, v9.4s, v0.s[1]
    fmla v23.4s, v9.4s, v1.s[1]
    fmla v20.4s, v10.4s, v0.s[2]
    fmla v22.4s, v10.4s, v1.s[2]
    fmla v21.4s, v11.4s, v0.s[2]
    fmla v23.4s, v11.4s, v1.s[2]
    fmla v20.4s, v12.4s, v0.s[3]
    fmla v22.4s, v12.4s, v1.s[3]
    fmla v21.4s, v13.4s, v0.s[3]
    fmla v23.4s, v13.4s, v1.s[3]
KAI_ASM_LABEL(label_41)  // Height 2: Multiply loop: Main loop skip
    cbz x27, label_43
KAI_ASM_LABEL(label_42)  // Height 2: Multiply loop: Odd block loop
    ldr s0, [x26], #0x4
    ldr s1, [x25], #0x4
    sub x27, x27, #0x1
    ldr q14, [x10, #0x0]
    ldr q15, [x10, #0x10]
    add x10, x10, #0x20
    fmla v20.4s, v14.4s, v0.s[0]
    fmla v22.4s, v14.4s, v1.s[0]
    fmla v21.4s, v15.4s, v0.s[0]
    fmla v23.4s, v15.4s, v1.s[0]
    cbnz x27, label_42
KAI_ASM_LABEL(label_43)  // Height 2: Multiply loop: No odd multiplies
    ldr w20, [x2, #0x8]
    add x28, x28, #0x1
    cmp x28, x20
    bne label_36
    ldr x20, [x2, #0x28]
    prfm pstl1keep, [x9, #0x0]
    add x26, x9, x20, LSL #2
    prfm pstl1keep, [x26, #0x0]
    tbz x3, #1, label_44
    add x21, x2, #0x0
    add x20, x2, #0x4
    ld1r { v17.4s }, [x21]
    ld1r { v16.4s }, [x20]
    fmin v20.4s, v20.4s, v17.4s
    fmin v21.4s, v21.4s, v17.4s
    fmin v22.4s, v22.4s, v17.4s
    fmin v23.4s, v23.4s, v17.4s
    fmax v20.4s, v20.4s, v16.4s
    fmax v21.4s, v21.4s, v16.4s
    fmax v22.4s, v22.4s, v16.4s
    fmax v23.4s, v23.4s, v16.4s
KAI_ASM_LABEL(label_44)  // Height 2: No activation
    cmp x11, #0x8
    bge label_49
    tbz x11, #2, label_46
    st1 { v20.4s }, [x9], #0x10
    st1 { v22.4s }, [x26], #0x10
    tbz x11, #1, label_45
    str d21, [x9], #0x8
    str d23, [x26], #0x8
    tbz x11, #0, label_48
    st1 { v21.s }[2], [x9]
    st1 { v23.s }[2], [x26]
    b label_48
KAI_ASM_LABEL(label_45)  // Height 2: Partial direct writeback: partial_1_4
    tbz x11, #0, label_48
    str s21, [x9, #0x0]
    str s23, [x26, #0x0]
    b label_48
KAI_ASM_LABEL(label_46)  // Height 2: Partial direct writeback: partial_2_0
    tbz x11, #1, label_47
    str d20, [x9], #0x8
    str d22, [x26], #0x8
    tbz x11, #0, label_48
    st1 { v20.s }[2], [x9]
    st1 { v22.s }[2], [x26]
    b label_48
KAI_ASM_LABEL(label_47)  // Height 2: Partial direct writeback: partial_1_0
    str s20, [x9, #0x0]
    str s22, [x26, #0x0]
KAI_ASM_LABEL(label_48)  // Height 2: Partial direct writeback: Done
    b label_50
KAI_ASM_LABEL(label_49)  // Height 2: Full writeback
    str q20, [x9, #0x0]
    str q21, [x9, #0x10]
    add x9, x9, #0x20
    str q22, [x26, #0x0]
    str q23, [x26, #0x10]
KAI_ASM_LABEL(label_50)  // Height 2: Writeback done
    subs x11, x11, #0x8
    bgt label_27
    b label_152
KAI_ASM_LABEL(label_51)  // Height 3
    ldr x11, [x2, #0x18]
    ldr x10, [x2, #0x20]
    ldr x9, [x2, #0x40]
KAI_ASM_LABEL(label_52)  // Height 3: Column loop
    cbz x10, label_53
    ldr q20, [x10, #0x0]
    ldr q21, [x10, #0x10]
    add x10, x10, #0x20
    mov v22.16b, v20.16b
    mov v23.16b, v21.16b
    mov v24.16b, v20.16b
    mov v25.16b, v21.16b
    b label_60
KAI_ASM_LABEL(label_53)  // Height 3: no bias
    tbz x3, #0, label_59
    ldr x20, [x2, #0x28]
    cmp x11, #0x8
    add x26, x9, x20, LSL #2
    add x25, x26, x20, LSL #2
    bge label_58
    tbz x11, #2, label_55
    ld1 { v20.4s }, [x9], #0x10
    ld1 { v22.4s }, [x26], #0x10
    ld1 { v24.4s }, [x25], #0x10
    tbz x11, #1, label_54
    ldr d21, [x9], #0x8
    ldr d23, [x26], #0x8
    mov x20, #0x18
    ldr d25, [x25], #0x8
    tbz x11, #0, label_57
    ld1 { v21.s }[2], [x9]
    ld1 { v23.s }[2], [x26]
    ld1 { v25.s }[2], [x25]
    b label_57
KAI_ASM_LABEL(label_54)  // Height 3: Partial accumulate: partial_1_4
    mov x20, #0x10
    tbz x11, #0, label_57
    ldr s21, [x9, #0x0]
    ldr s23, [x26, #0x0]
    ldr s25, [x25, #0x0]
    b label_57
KAI_ASM_LABEL(label_55)  // Height 3: Partial accumulate: partial_2_0
    tbz x11, #1, label_56
    ldr d20, [x9], #0x8
    ldr d22, [x26], #0x8
    mov x20, #0x8
    ldr d24, [x25], #0x8
    tbz x11, #0, label_57
    ld1 { v20.s }[2], [x9]
    ld1 { v22.s }[2], [x26]
    ld1 { v24.s }[2], [x25]
    b label_57
KAI_ASM_LABEL(label_56)  // Height 3: Partial accumulate: partial_1_0
    ldr s20, [x9, #0x0]
    ldr s22, [x26, #0x0]
    mov x20, #0x0
    ldr s24, [x25, #0x0]
KAI_ASM_LABEL(label_57)  // Height 3: Partial accumulate: Done
    sub x9, x9, x20
    b label_60
KAI_ASM_LABEL(label_58)  // Height 3: full accumulate
    ldr q20, [x9, #0x0]
    ldr q21, [x9, #0x10]
    ldr q22, [x26, #0x0]
    ldr q23, [x26, #0x10]
    ldr q24, [x25, #0x0]
    ldr q25, [x25, #0x10]
    b label_60
KAI_ASM_LABEL(label_59)  // Height 3: no accumulate
    movi v20.16b, #0x0
    movi v21.16b, #0x0
    movi v22.16b, #0x0
    movi v23.16b, #0x0
    movi v24.16b, #0x0
    movi v25.16b, #0x0
KAI_ASM_LABEL(label_60)  // Height 3: setup done
    mov x28, #0x0
KAI_ASM_LABEL(label_61)  // Height 3: String loop
    ldr x20, [x2, #0x10]
    ldr x21, [x2, #0x38]
    ldr w27, [x20, x28, LSL #0x2]
    tbz x3, #3, label_62
    ldr x20, [x0, x28, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x26, [x20, #0x0]
    ldr x25, [x20, #0x8]
    ldr x24, [x20, #0x10]
    cbnz x28, label_63
    ldr x20, [x2, #0x30]
    add x26, x26, x20, LSL #2
    add x25, x25, x20, LSL #2
    add x24, x24, x20, LSL #2
    b label_63
KAI_ASM_LABEL(label_62)  // Height 3: setup direct input
    mov x26, x0
    add x25, x26, x21, LSL #2
    add x24, x25, x21, LSL #2
KAI_ASM_LABEL(label_63)  // Height 3: input setup done
    cmp x27, #0x4
    blt label_66
    ldr q0, [x26, #0x0]
    ldr q1, [x25, #0x0]
    cmp x27, #0x8
    ldr q2, [x24, #0x0]
    ldr q6, [x10, #0x0]
    ldr q7, [x10, #0x10]
    ldr q8, [x10, #0x20]
    ldr q9, [x10, #0x30]
    ldr q10, [x10, #0x40]
    ldr q11, [x10, #0x50]
    ldr q12, [x10, #0x60]
    ldr q13, [x10, #0x70]
    blt label_65
KAI_ASM_LABEL(label_64)  // Height 3: Multiply loop: Main loop head
    fmla v20.4s, v6.4s, v0.s[0]
    fmla v22.4s, v6.4s, v1.s[0]
    sub x27, x27, #0x4
    add x26, x26, #0x10
    fmla v24.4s, v6.4s, v2.s[0]
    fmla v21.4s, v7.4s, v0.s[0]
    add x25, x25, #0x10
    add x24, x24, #0x10
    fmla v23.4s, v7.4s, v1.s[0]
    fmla v25.4s, v7.4s, v2.s[0]
    cmp x27, #0x8
    add x10, x10, #0x80
    ldr q6, [x10, #0x0]
    ldr q7, [x10, #0x10]
    fmla v20.4s, v8.4s, v0.s[1]
    fmla v22.4s, v8.4s, v1.s[1]
    prfm pldl1keep, [x26, #0x80]
    prfm pldl1keep, [x25, #0x80]
    fmla v24.4s, v8.4s, v2.s[1]
    ldr q8, [x10, #0x20]
    fmla v21.4s, v9.4s, v0.s[1]
    prfm pldl1keep, [x24, #0x80]
    fmla v23.4s, v9.4s, v1.s[1]
    fmla v25.4s, v9.4s, v2.s[1]
    ldr q9, [x10, #0x30]
    fmla v20.4s, v10.4s, v0.s[2]
    fmla v22.4s, v10.4s, v1.s[2]
    fmla v24.4s, v10.4s, v2.s[2]
    ldr q10, [x10, #0x40]
    fmla v21.4s, v11.4s, v0.s[2]
    fmla v23.4s, v11.4s, v1.s[2]
    fmla v25.4s, v11.4s, v2.s[2]
    ldr q11, [x10, #0x50]
    fmla v20.4s, v12.4s, v0.s[3]
    fmla v22.4s, v12.4s, v1.s[3]
    fmla v24.4s, v12.4s, v2.s[3]
    ldr q12, [x10, #0x60]
    fmla v21.4s, v13.4s, v0.s[3]
    ldr q0, [x26, #0x0]
    fmla v23.4s, v13.4s, v1.s[3]
    ldr q1, [x25, #0x0]
    fmla v25.4s, v13.4s, v2.s[3]
    ldr q2, [x24, #0x0]
    ldr q13, [x10, #0x70]
    bge label_64
KAI_ASM_LABEL(label_65)  // Height 3: Multiply loop: Single iteration only
    fmla v20.4s, v6.4s, v0.s[0]
    fmla v22.4s, v6.4s, v1.s[0]
    add x26, x26, #0x10
    add x25, x25, #0x10
    fmla v24.4s, v6.4s, v2.s[0]
    fmla v21.4s, v7.4s, v0.s[0]
    add x24, x24, #0x10
    prfm pldl1keep, [x26, #0x80]
    fmla v23.4s, v7.4s, v1.s[0]
    fmla v25.4s, v7.4s, v2.s[0]
    sub x27, x27, #0x4
    prfm pldl1keep, [x25, #0x80]
    add x10, x10, #0x80
    prfm pldl1keep, [x24, #0x80]
    fmla v20.4s, v8.4s, v0.s[1]
    fmla v22.4s, v8.4s, v1.s[1]
    fmla v24.4s, v8.4s, v2.s[1]
    fmla v21.4s, v9.4s, v0.s[1]
    fmla v23.4s, v9.4s, v1.s[1]
    fmla v25.4s, v9.4s, v2.s[1]
    fmla v20.4s, v10.4s, v0.s[2]
    fmla v22.4s, v10.4s, v1.s[2]
    fmla v24.4s, v10.4s, v2.s[2]
    fmla v21.4s, v11.4s, v0.s[2]
    fmla v23.4s, v11.4s, v1.s[2]
    fmla v25.4s, v11.4s, v2.s[2]
    fmla v20.4s, v12.4s, v0.s[3]
    fmla v22.4s, v12.4s, v1.s[3]
    fmla v24.4s, v12.4s, v2.s[3]
    fmla v21.4s, v13.4s, v0.s[3]
    fmla v23.4s, v13.4s, v1.s[3]
    fmla v25.4s, v13.4s, v2.s[3]
KAI_ASM_LABEL(label_66)  // Height 3: Multiply loop: Main loop skip
    cbz x27, label_68
KAI_ASM_LABEL(label_67)  // Height 3: Multiply loop: Odd block loop
    ldr s0, [x26], #0x4
    ldr s1, [x25], #0x4
    sub x27, x27, #0x1
    ldr s2, [x24], #0x4
    ldr q14, [x10, #0x0]
    ldr q15, [x10, #0x10]
    add x10, x10, #0x20
    fmla v20.4s, v14.4s, v0.s[0]
    fmla v22.4s, v14.4s, v1.s[0]
    fmla v24.4s, v14.4s, v2.s[0]
    fmla v21.4s, v15.4s, v0.s[0]
    fmla v23.4s, v15.4s, v1.s[0]
    fmla v25.4s, v15.4s, v2.s[0]
    cbnz x27, label_67
KAI_ASM_LABEL(label_68)  // Height 3: Multiply loop: No odd multiplies
    ldr w20, [x2, #0x8]
    add x28, x28, #0x1
    cmp x28, x20
    bne label_61
    ldr x20, [x2, #0x28]
    prfm pstl1keep, [x9, #0x0]
    add x26, x9, x20, LSL #2
    prfm pstl1keep, [x26, #0x0]
    add x25, x26, x20, LSL #2
    prfm pstl1keep, [x25, #0x0]
    tbz x3, #1, label_69
    add x21, x2, #0x0
    add x20, x2, #0x4
    ld1r { v17.4s }, [x21]
    ld1r { v16.4s }, [x20]
    fmin v20.4s, v20.4s, v17.4s
    fmin v21.4s, v21.4s, v17.4s
    fmin v22.4s, v22.4s, v17.4s
    fmin v23.4s, v23.4s, v17.4s
    fmin v24.4s, v24.4s, v17.4s
    fmin v25.4s, v25.4s, v17.4s
    fmax v20.4s, v20.4s, v16.4s
    fmax v21.4s, v21.4s, v16.4s
    fmax v22.4s, v22.4s, v16.4s
    fmax v23.4s, v23.4s, v16.4s
    fmax v24.4s, v24.4s, v16.4s
    fmax v25.4s, v25.4s, v16.4s
KAI_ASM_LABEL(label_69)  // Height 3: No activation
    cmp x11, #0x8
    bge label_74
    tbz x11, #2, label_71
    st1 { v20.4s }, [x9], #0x10
    st1 { v22.4s }, [x26], #0x10
    st1 { v24.4s }, [x25], #0x10
    tbz x11, #1, label_70
    str d21, [x9], #0x8
    str d23, [x26], #0x8
    str d25, [x25], #0x8
    tbz x11, #0, label_73
    st1 { v21.s }[2], [x9]
    st1 { v23.s }[2], [x26]
    st1 { v25.s }[2], [x25]
    b label_73
KAI_ASM_LABEL(label_70)  // Height 3: Partial direct writeback: partial_1_4
    tbz x11, #0, label_73
    str s21, [x9, #0x0]
    str s23, [x26, #0x0]
    str s25, [x25, #0x0]
    b label_73
KAI_ASM_LABEL(label_71)  // Height 3: Partial direct writeback: partial_2_0
    tbz x11, #1, label_72
    str d20, [x9], #0x8
    str d22, [x26], #0x8
    str d24, [x25], #0x8
    tbz x11, #0, label_73
    st1 { v20.s }[2], [x9]
    st1 { v22.s }[2], [x26]
    st1 { v24.s }[2], [x25]
    b label_73
KAI_ASM_LABEL(label_72)  // Height 3: Partial direct writeback: partial_1_0
    str s20, [x9, #0x0]
    str s22, [x26, #0x0]
    str s24, [x25, #0x0]
KAI_ASM_LABEL(label_73)  // Height 3: Partial direct writeback: Done
    b label_75
KAI_ASM_LABEL(label_74)  // Height 3: Full writeback
    str q20, [x9, #0x0]
    str q21, [x9, #0x10]
    add x9, x9, #0x20
    str q22, [x26, #0x0]
    str q23, [x26, #0x10]
    str q24, [x25, #0x0]
    str q25, [x25, #0x10]
KAI_ASM_LABEL(label_75)  // Height 3: Writeback done
    subs x11, x11, #0x8
    bgt label_52
    b label_152
KAI_ASM_LABEL(label_76)  // Height 4
    ldr x11, [x2, #0x18]
    ldr x10, [x2, #0x20]
    ldr x9, [x2, #0x40]
KAI_ASM_LABEL(label_77)  // Height 4: Column loop
    cbz x10, label_78
    ldr q20, [x10, #0x0]
    ldr q21, [x10, #0x10]
    add x10, x10, #0x20
    mov v22.16b, v20.16b
    mov v23.16b, v21.16b
    mov v24.16b, v20.16b
    mov v25.16b, v21.16b
    mov v26.16b, v20.16b
    mov v27.16b, v21.16b
    b label_85
KAI_ASM_LABEL(label_78)  // Height 4: no bias
    tbz x3, #0, label_84
    ldr x20, [x2, #0x28]
    cmp x11, #0x8
    add x26, x9, x20, LSL #2
    add x25, x26, x20, LSL #2
    add x24, x25, x20, LSL #2
    bge label_83
    tbz x11, #2, label_80
    ld1 { v20.4s }, [x9], #0x10
    ld1 { v22.4s }, [x26], #0x10
    ld1 { v24.4s }, [x25], #0x10
    ld1 { v26.4s }, [x24], #0x10
    tbz x11, #1, label_79
    ldr d21, [x9], #0x8
    ldr d23, [x26], #0x8
    mov x20, #0x18
    ldr d25, [x25], #0x8
    ldr d27, [x24], #0x8
    tbz x11, #0, label_82
    ld1 { v21.s }[2], [x9]
    ld1 { v23.s }[2], [x26]
    ld1 { v25.s }[2], [x25]
    ld1 { v27.s }[2], [x24]
    b label_82
KAI_ASM_LABEL(label_79)  // Height 4: Partial accumulate: partial_1_4
    mov x20, #0x10
    tbz x11, #0, label_82
    ldr s21, [x9, #0x0]
    ldr s23, [x26, #0x0]
    ldr s25, [x25, #0x0]
    ldr s27, [x24, #0x0]
    b label_82
KAI_ASM_LABEL(label_80)  // Height 4: Partial accumulate: partial_2_0
    tbz x11, #1, label_81
    ldr d20, [x9], #0x8
    ldr d22, [x26], #0x8
    mov x20, #0x8
    ldr d24, [x25], #0x8
    ldr d26, [x24], #0x8
    tbz x11, #0, label_82
    ld1 { v20.s }[2], [x9]
    ld1 { v22.s }[2], [x26]
    ld1 { v24.s }[2], [x25]
    ld1 { v26.s }[2], [x24]
    b label_82
KAI_ASM_LABEL(label_81)  // Height 4: Partial accumulate: partial_1_0
    ldr s20, [x9, #0x0]
    ldr s22, [x26, #0x0]
    mov x20, #0x0
    ldr s24, [x25, #0x0]
    ldr s26, [x24, #0x0]
KAI_ASM_LABEL(label_82)  // Height 4: Partial accumulate: Done
    sub x9, x9, x20
    b label_85
KAI_ASM_LABEL(label_83)  // Height 4: full accumulate
    ldr q20, [x9, #0x0]
    ldr q21, [x9, #0x10]
    ldr q22, [x26, #0x0]
    ldr q23, [x26, #0x10]
    ldr q24, [x25, #0x0]
    ldr q25, [x25, #0x10]
    ldr q26, [x24, #0x0]
    ldr q27, [x24, #0x10]
    b label_85
KAI_ASM_LABEL(label_84)  // Height 4: no accumulate
    movi v20.16b, #0x0
    movi v21.16b, #0x0
    movi v22.16b, #0x0
    movi v23.16b, #0x0
    movi v24.16b, #0x0
    movi v25.16b, #0x0
    movi v26.16b, #0x0
    movi v27.16b, #0x0
KAI_ASM_LABEL(label_85)  // Height 4: setup done
    mov x28, #0x0
KAI_ASM_LABEL(label_86)  // Height 4: String loop
    ldr x20, [x2, #0x10]
    ldr x21, [x2, #0x38]
    ldr w27, [x20, x28, LSL #0x2]
    tbz x3, #3, label_87
    ldr x20, [x0, x28, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x26, [x20, #0x0]
    ldr x25, [x20, #0x8]
    ldr x24, [x20, #0x10]
    ldr x23, [x20, #0x18]
    cbnz x28, label_88
    ldr x20, [x2, #0x30]
    add x26, x26, x20, LSL #2
    add x25, x25, x20, LSL #2
    add x24, x24, x20, LSL #2
    add x23, x23, x20, LSL #2
    b label_88
KAI_ASM_LABEL(label_87)  // Height 4: setup direct input
    mov x26, x0
    add x25, x26, x21, LSL #2
    add x24, x25, x21, LSL #2
    add x23, x24, x21, LSL #2
KAI_ASM_LABEL(label_88)  // Height 4: input setup done
    cmp x27, #0x4
    blt label_91
    ldr q0, [x26, #0x0]
    ldr q1, [x25, #0x0]
    cmp x27, #0x8
    ldr q2, [x24, #0x0]
    ldr q3, [x23, #0x0]
    ldr q6, [x10, #0x0]
    ldr q7, [x10, #0x10]
    ldr q8, [x10, #0x20]
    ldr q9, [x10, #0x30]
    ldr q10, [x10, #0x40]
    ldr q11, [x10, #0x50]
    ldr q12, [x10, #0x60]
    ldr q13, [x10, #0x70]
    blt label_90
KAI_ASM_LABEL(label_89)  // Height 4: Multiply loop: Main loop head
    fmla v20.4s, v6.4s, v0.s[0]
    fmla v22.4s, v6.4s, v1.s[0]
    sub x27, x27, #0x4
    add x26, x26, #0x10
    fmla v24.4s, v6.4s, v2.s[0]
    fmla v26.4s, v6.4s, v3.s[0]
    add x25, x25, #0x10
    add x24, x24, #0x10
    fmla v21.4s, v7.4s, v0.s[0]
    fmla v23.4s, v7.4s, v1.s[0]
    add x23, x23, #0x10
    cmp x27, #0x8
    fmla v25.4s, v7.4s, v2.s[0]
    fmla v27.4s, v7.4s, v3.s[0]
    add x10, x10, #0x80
    prfm pldl1keep, [x26, #0x80]
    ldr q6, [x10, #0x0]
    ldr q7, [x10, #0x10]
    fmla v20.4s, v8.4s, v0.s[1]
    fmla v22.4s, v8.4s, v1.s[1]
    fmla v24.4s, v8.4s, v2.s[1]
    fmla v26.4s, v8.4s, v3.s[1]
    ldr q8, [x10, #0x20]
    prfm pldl1keep, [x25, #0x80]
    fmla v21.4s, v9.4s, v0.s[1]
    fmla v23.4s, v9.4s, v1.s[1]
    prfm pldl1keep, [x24, #0x80]
    prfm pldl1keep, [x23, #0x80]
    fmla v25.4s, v9.4s, v2.s[1]
    fmla v27.4s, v9.4s, v3.s[1]
    ldr q9, [x10, #0x30]
    fmla v20.4s, v10.4s, v0.s[2]
    fmla v22.4s, v10.4s, v1.s[2]
    fmla v24.4s, v10.4s, v2.s[2]
    fmla v26.4s, v10.4s, v3.s[2]
    ldr q10, [x10, #0x40]
    fmla v21.4s, v11.4s, v0.s[2]
    fmla v23.4s, v11.4s, v1.s[2]
    fmla v25.4s, v11.4s, v2.s[2]
    fmla v27.4s, v11.4s, v3.s[2]
    ldr q11, [x10, #0x50]
    fmla v20.4s, v12.4s, v0.s[3]
    fmla v22.4s, v12.4s, v1.s[3]
    fmla v24.4s, v12.4s, v2.s[3]
    fmla v26.4s, v12.4s, v3.s[3]
    ldr q12, [x10, #0x60]
    fmla v21.4s, v13.4s, v0.s[3]
    ldr q0, [x26, #0x0]
    fmla v23.4s, v13.4s, v1.s[3]
    ldr q1, [x25, #0x0]
    fmla v25.4s, v13.4s, v2.s[3]
    ldr q2, [x24, #0x0]
    fmla v27.4s, v13.4s, v3.s[3]
    ldr q3, [x23, #0x0]
    ldr q13, [x10, #0x70]
    bge label_89
KAI_ASM_LABEL(label_90)  // Height 4: Multiply loop: Single iteration only
    fmla v20.4s, v6.4s, v0.s[0]
    fmla v22.4s, v6.4s, v1.s[0]
    add x26, x26, #0x10
    add x25, x25, #0x10
    fmla v24.4s, v6.4s, v2.s[0]
    fmla v26.4s, v6.4s, v3.s[0]
    add x24, x24, #0x10
    add x23, x23, #0x10
    fmla v21.4s, v7.4s, v0.s[0]
    fmla v23.4s, v7.4s, v1.s[0]
    sub x27, x27, #0x4
    prfm pldl1keep, [x26, #0x80]
    fmla v25.4s, v7.4s, v2.s[0]
    fmla v27.4s, v7.4s, v3.s[0]
    prfm pldl1keep, [x25, #0x80]
    prfm pldl1keep, [x24, #0x80]
    fmla v20.4s, v8.4s, v0.s[1]
    fmla v22.4s, v8.4s, v1.s[1]
    prfm pldl1keep, [x23, #0x80]
    add x10, x10, #0x80
    fmla v24.4s, v8.4s, v2.s[1]
    fmla v26.4s, v8.4s, v3.s[1]
    fmla v21.4s, v9.4s, v0.s[1]
    fmla v23.4s, v9.4s, v1.s[1]
    fmla v25.4s, v9.4s, v2.s[1]
    fmla v27.4s, v9.4s, v3.s[1]
    fmla v20.4s, v10.4s, v0.s[2]
    fmla v22.4s, v10.4s, v1.s[2]
    fmla v24.4s, v10.4s, v2.s[2]
    fmla v26.4s, v10.4s, v3.s[2]
    fmla v21.4s, v11.4s, v0.s[2]
    fmla v23.4s, v11.4s, v1.s[2]
    fmla v25.4s, v11.4s, v2.s[2]
    fmla v27.4s, v11.4s, v3.s[2]
    fmla v20.4s, v12.4s, v0.s[3]
    fmla v22.4s, v12.4s, v1.s[3]
    fmla v24.4s, v12.4s, v2.s[3]
    fmla v26.4s, v12.4s, v3.s[3]
    fmla v21.4s, v13.4s, v0.s[3]
    fmla v23.4s, v13.4s, v1.s[3]
    fmla v25.4s, v13.4s, v2.s[3]
    fmla v27.4s, v13.4s, v3.s[3]
KAI_ASM_LABEL(label_91)  // Height 4: Multiply loop: Main loop skip
    cbz x27, label_93
KAI_ASM_LABEL(label_92)  // Height 4: Multiply loop: Odd block loop
    ldr s0, [x26], #0x4
    ldr s1, [x25], #0x4
    sub x27, x27, #0x1
    ldr s2, [x24], #0x4
    ldr s3, [x23], #0x4
    ldr q14, [x10, #0x0]
    ldr q15, [x10, #0x10]
    add x10, x10, #0x20
    fmla v20.4s, v14.4s, v0.s[0]
    fmla v22.4s, v14.4s, v1.s[0]
    fmla v24.4s, v14.4s, v2.s[0]
    fmla v26.4s, v14.4s, v3.s[0]
    fmla v21.4s, v15.4s, v0.s[0]
    fmla v23.4s, v15.4s, v1.s[0]
    fmla v25.4s, v15.4s, v2.s[0]
    fmla v27.4s, v15.4s, v3.s[0]
    cbnz x27, label_92
KAI_ASM_LABEL(label_93)  // Height 4: Multiply loop: No odd multiplies
    ldr w20, [x2, #0x8]
    add x28, x28, #0x1
    cmp x28, x20
    bne label_86
    ldr x20, [x2, #0x28]
    prfm pstl1keep, [x9, #0x0]
    add x26, x9, x20, LSL #2
    prfm pstl1keep, [x26, #0x0]
    add x25, x26, x20, LSL #2
    prfm pstl1keep, [x25, #0x0]
    add x24, x25, x20, LSL #2
    prfm pstl1keep, [x24, #0x0]
    tbz x3, #1, label_94
    add x21, x2, #0x0
    add x20, x2, #0x4
    ld1r { v17.4s }, [x21]
    ld1r { v16.4s }, [x20]
    fmin v20.4s, v20.4s, v17.4s
    fmin v21.4s, v21.4s, v17.4s
    fmin v22.4s, v22.4s, v17.4s
    fmin v23.4s, v23.4s, v17.4s
    fmin v24.4s, v24.4s, v17.4s
    fmin v25.4s, v25.4s, v17.4s
    fmin v26.4s, v26.4s, v17.4s
    fmin v27.4s, v27.4s, v17.4s
    fmax v20.4s, v20.4s, v16.4s
    fmax v21.4s, v21.4s, v16.4s
    fmax v22.4s, v22.4s, v16.4s
    fmax v23.4s, v23.4s, v16.4s
    fmax v24.4s, v24.4s, v16.4s
    fmax v25.4s, v25.4s, v16.4s
    fmax v26.4s, v26.4s, v16.4s
    fmax v27.4s, v27.4s, v16.4s
KAI_ASM_LABEL(label_94)  // Height 4: No activation
    cmp x11, #0x8
    bge label_99
    tbz x11, #2, label_96
    st1 { v20.4s }, [x9], #0x10
    st1 { v22.4s }, [x26], #0x10
    st1 { v24.4s }, [x25], #0x10
    st1 { v26.4s }, [x24], #0x10
    tbz x11, #1, label_95
    str d21, [x9], #0x8
    str d23, [x26], #0x8
    str d25, [x25], #0x8
    str d27, [x24], #0x8
    tbz x11, #0, label_98
    st1 { v21.s }[2], [x9]
    st1 { v23.s }[2], [x26]
    st1 { v25.s }[2], [x25]
    st1 { v27.s }[2], [x24]
    b label_98
KAI_ASM_LABEL(label_95)  // Height 4: Partial direct writeback: partial_1_4
    tbz x11, #0, label_98
    str s21, [x9, #0x0]
    str s23, [x26, #0x0]
    str s25, [x25, #0x0]
    str s27, [x24, #0x0]
    b label_98
KAI_ASM_LABEL(label_96)  // Height 4: Partial direct writeback: partial_2_0
    tbz x11, #1, label_97
    str d20, [x9], #0x8
    str d22, [x26], #0x8
    str d24, [x25], #0x8
    str d26, [x24], #0x8
    tbz x11, #0, label_98
    st1 { v20.s }[2], [x9]
    st1 { v22.s }[2], [x26]
    st1 { v24.s }[2], [x25]
    st1 { v26.s }[2], [x24]
    b label_98
KAI_ASM_LABEL(label_97)  // Height 4: Partial direct writeback: partial_1_0
    str s20, [x9, #0x0]
    str s22, [x26, #0x0]
    str s24, [x25, #0x0]
    str s26, [x24, #0x0]
KAI_ASM_LABEL(label_98)  // Height 4: Partial direct writeback: Done
    b label_100
KAI_ASM_LABEL(label_99)  // Height 4: Full writeback
    str q20, [x9, #0x0]
    str q21, [x9, #0x10]
    add x9, x9, #0x20
    str q22, [x26, #0x0]
    str q23, [x26, #0x10]
    str q24, [x25, #0x0]
    str q25, [x25, #0x10]
    str q26, [x24, #0x0]
    str q27, [x24, #0x10]
KAI_ASM_LABEL(label_100)  // Height 4: Writeback done
    subs x11, x11, #0x8
    bgt label_77
    b label_152
KAI_ASM_LABEL(label_101)  // Height 5
    ldr x11, [x2, #0x18]
    ldr x10, [x2, #0x20]
    ldr x9, [x2, #0x40]
KAI_ASM_LABEL(label_102)  // Height 5: Column loop
    cbz x10, label_103
    ldr q20, [x10, #0x0]
    ldr q21, [x10, #0x10]
    add x10, x10, #0x20
    mov v22.16b, v20.16b
    mov v23.16b, v21.16b
    mov v24.16b, v20.16b
    mov v25.16b, v21.16b
    mov v26.16b, v20.16b
    mov v27.16b, v21.16b
    mov v28.16b, v20.16b
    mov v29.16b, v21.16b
    b label_110
KAI_ASM_LABEL(label_103)  // Height 5: no bias
    tbz x3, #0, label_109
    ldr x20, [x2, #0x28]
    cmp x11, #0x8
    add x26, x9, x20, LSL #2
    add x25, x26, x20, LSL #2
    add x24, x25, x20, LSL #2
    add x23, x24, x20, LSL #2
    bge label_108
    tbz x11, #2, label_105
    ld1 { v20.4s }, [x9], #0x10
    ld1 { v22.4s }, [x26], #0x10
    ld1 { v24.4s }, [x25], #0x10
    ld1 { v26.4s }, [x24], #0x10
    ld1 { v28.4s }, [x23], #0x10
    tbz x11, #1, label_104
    ldr d21, [x9], #0x8
    ldr d23, [x26], #0x8
    mov x20, #0x18
    ldr d25, [x25], #0x8
    ldr d27, [x24], #0x8
    ldr d29, [x23], #0x8
    tbz x11, #0, label_107
    ld1 { v21.s }[2], [x9]
    ld1 { v23.s }[2], [x26]
    ld1 { v25.s }[2], [x25]
    ld1 { v27.s }[2], [x24]
    ld1 { v29.s }[2], [x23]
    b label_107
KAI_ASM_LABEL(label_104)  // Height 5: Partial accumulate: partial_1_4
    mov x20, #0x10
    tbz x11, #0, label_107
    ldr s21, [x9, #0x0]
    ldr s23, [x26, #0x0]
    ldr s25, [x25, #0x0]
    ldr s27, [x24, #0x0]
    ldr s29, [x23, #0x0]
    b label_107
KAI_ASM_LABEL(label_105)  // Height 5: Partial accumulate: partial_2_0
    tbz x11, #1, label_106
    ldr d20, [x9], #0x8
    ldr d22, [x26], #0x8
    mov x20, #0x8
    ldr d24, [x25], #0x8
    ldr d26, [x24], #0x8
    ldr d28, [x23], #0x8
    tbz x11, #0, label_107
    ld1 { v20.s }[2], [x9]
    ld1 { v22.s }[2], [x26]
    ld1 { v24.s }[2], [x25]
    ld1 { v26.s }[2], [x24]
    ld1 { v28.s }[2], [x23]
    b label_107
KAI_ASM_LABEL(label_106)  // Height 5: Partial accumulate: partial_1_0
    ldr s20, [x9, #0x0]
    ldr s22, [x26, #0x0]
    mov x20, #0x0
    ldr s24, [x25, #0x0]
    ldr s26, [x24, #0x0]
    ldr s28, [x23, #0x0]
KAI_ASM_LABEL(label_107)  // Height 5: Partial accumulate: Done
    sub x9, x9, x20
    b label_110
KAI_ASM_LABEL(label_108)  // Height 5: full accumulate
    ldr q20, [x9, #0x0]
    ldr q21, [x9, #0x10]
    ldr q22, [x26, #0x0]
    ldr q23, [x26, #0x10]
    ldr q24, [x25, #0x0]
    ldr q25, [x25, #0x10]
    ldr q26, [x24, #0x0]
    ldr q27, [x24, #0x10]
    ldr q28, [x23, #0x0]
    ldr q29, [x23, #0x10]
    b label_110
KAI_ASM_LABEL(label_109)  // Height 5: no accumulate
    movi v20.16b, #0x0
    movi v21.16b, #0x0
    movi v22.16b, #0x0
    movi v23.16b, #0x0
    movi v24.16b, #0x0
    movi v25.16b, #0x0
    movi v26.16b, #0x0
    movi v27.16b, #0x0
    movi v28.16b, #0x0
    movi v29.16b, #0x0
KAI_ASM_LABEL(label_110)  // Height 5: setup done
    mov x28, #0x0
KAI_ASM_LABEL(label_111)  // Height 5: String loop
    ldr x20, [x2, #0x10]
    ldr x21, [x2, #0x38]
    ldr w27, [x20, x28, LSL #0x2]
    tbz x3, #3, label_112
    ldr x20, [x0, x28, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x26, [x20, #0x0]
    ldr x25, [x20, #0x8]
    ldr x24, [x20, #0x10]
    ldr x23, [x20, #0x18]
    ldr x22, [x20, #0x20]
    cbnz x28, label_113
    ldr x20, [x2, #0x30]
    add x26, x26, x20, LSL #2
    add x25, x25, x20, LSL #2
    add x24, x24, x20, LSL #2
    add x23, x23, x20, LSL #2
    add x22, x22, x20, LSL #2
    b label_113
KAI_ASM_LABEL(label_112)  // Height 5: setup direct input
    mov x26, x0
    add x25, x26, x21, LSL #2
    add x24, x25, x21, LSL #2
    add x23, x24, x21, LSL #2
    add x22, x23, x21, LSL #2
KAI_ASM_LABEL(label_113)  // Height 5: input setup done
    cmp x27, #0x4
    blt label_116
    ldr q0, [x26, #0x0]
    ldr q1, [x25, #0x0]
    cmp x27, #0x8
    ldr q2, [x24, #0x0]
    ldr q3, [x23, #0x0]
    ldr q4, [x22, #0x0]
    ldr q6, [x10, #0x0]
    ldr q7, [x10, #0x10]
    ldr q8, [x10, #0x20]
    ldr q9, [x10, #0x30]
    ldr q10, [x10, #0x40]
    ldr q11, [x10, #0x50]
    ldr q12, [x10, #0x60]
    ldr q13, [x10, #0x70]
    blt label_115
KAI_ASM_LABEL(label_114)  // Height 5: Multiply loop: Main loop head
    fmla v20.4s, v6.4s, v0.s[0]
    fmla v22.4s, v6.4s, v1.s[0]
    sub x27, x27, #0x4
    add x26, x26, #0x10
    fmla v24.4s, v6.4s, v2.s[0]
    fmla v26.4s, v6.4s, v3.s[0]
    add x25, x25, #0x10
    add x24, x24, #0x10
    fmla v28.4s, v6.4s, v4.s[0]
    fmla v21.4s, v7.4s, v0.s[0]
    add x23, x23, #0x10
    add x22, x22, #0x10
    fmla v23.4s, v7.4s, v1.s[0]
    fmla v25.4s, v7.4s, v2.s[0]
    cmp x27, #0x8
    add x10, x10, #0x80
    ldr q6, [x10, #0x0]
    fmla v27.4s, v7.4s, v3.s[0]
    fmla v29.4s, v7.4s, v4.s[0]
    ldr q7, [x10, #0x10]
    fmla v20.4s, v8.4s, v0.s[1]
    fmla v22.4s, v8.4s, v1.s[1]
    prfm pldl1keep, [x26, #0x80]
    prfm pldl1keep, [x25, #0x80]
    fmla v24.4s, v8.4s, v2.s[1]
    fmla v26.4s, v8.4s, v3.s[1]
    prfm pldl1keep, [x24, #0x80]
    prfm pldl1keep, [x23, #0x80]
    fmla v28.4s, v8.4s, v4.s[1]
    ldr q8, [x10, #0x20]
    fmla v21.4s, v9.4s, v0.s[1]
    prfm pldl1keep, [x22, #0x80]
    fmla v23.4s, v9.4s, v1.s[1]
    fmla v25.4s, v9.4s, v2.s[1]
    fmla v27.4s, v9.4s, v3.s[1]
    fmla v29.4s, v9.4s, v4.s[1]
    ldr q9, [x10, #0x30]
    fmla v20.4s, v10.4s, v0.s[2]
    fmla v22.4s, v10.4s, v1.s[2]
    fmla v24.4s, v10.4s, v2.s[2]
    fmla v26.4s, v10.4s, v3.s[2]
    fmla v28.4s, v10.4s, v4.s[2]
    ldr q10, [x10, #0x40]
    fmla v21.4s, v11.4s, v0.s[2]
    fmla v23.4s, v11.4s, v1.s[2]
    fmla v25.4s, v11.4s, v2.s[2]
    fmla v27.4s, v11.4s, v3.s[2]
    fmla v29.4s, v11.4s, v4.s[2]
    ldr q11, [x10, #0x50]
    fmla v20.4s, v12.4s, v0.s[3]
    fmla v22.4s, v12.4s, v1.s[3]
    fmla v24.4s, v12.4s, v2.s[3]
    fmla v26.4s, v12.4s, v3.s[3]
    fmla v28.4s, v12.4s, v4.s[3]
    ldr q12, [x10, #0x60]
    fmla v21.4s, v13.4s, v0.s[3]
    ldr q0, [x26, #0x0]
    fmla v23.4s, v13.4s, v1.s[3]
    ldr q1, [x25, #0x0]
    fmla v25.4s, v13.4s, v2.s[3]
    ldr q2, [x24, #0x0]
    fmla v27.4s, v13.4s, v3.s[3]
    ldr q3, [x23, #0x0]
    fmla v29.4s, v13.4s, v4.s[3]
    ldr q4, [x22, #0x0]
    ldr q13, [x10, #0x70]
    bge label_114
KAI_ASM_LABEL(label_115)  // Height 5: Multiply loop: Single iteration only
    fmla v20.4s, v6.4s, v0.s[0]
    fmla v22.4s, v6.4s, v1.s[0]
    add x26, x26, #0x10
    add x25, x25, #0x10
    fmla v24.4s, v6.4s, v2.s[0]
    fmla v26.4s, v6.4s, v3.s[0]
    add x24, x24, #0x10
    add x23, x23, #0x10
    fmla v28.4s, v6.4s, v4.s[0]
    fmla v21.4s, v7.4s, v0.s[0]
    add x22, x22, #0x10
    sub x27, x27, #0x4
    fmla v23.4s, v7.4s, v1.s[0]
    fmla v25.4s, v7.4s, v2.s[0]
    prfm pldl1keep, [x26, #0x80]
    prfm pldl1keep, [x25, #0x80]
    fmla v27.4s, v7.4s, v3.s[0]
    fmla v29.4s, v7.4s, v4.s[0]
    prfm pldl1keep, [x24, #0x80]
    prfm pldl1keep, [x23, #0x80]
    fmla v20.4s, v8.4s, v0.s[1]
    fmla v22.4s, v8.4s, v1.s[1]
    prfm pldl1keep, [x22, #0x80]
    add x10, x10, #0x80
    fmla v24.4s, v8.4s, v2.s[1]
    fmla v26.4s, v8.4s, v3.s[1]
    fmla v28.4s, v8.4s, v4.s[1]
    fmla v21.4s, v9.4s, v0.s[1]
    fmla v23.4s, v9.4s, v1.s[1]
    fmla v25.4s, v9.4s, v2.s[1]
    fmla v27.4s, v9.4s, v3.s[1]
    fmla v29.4s, v9.4s, v4.s[1]
    fmla v20.4s, v10.4s, v0.s[2]
    fmla v22.4s, v10.4s, v1.s[2]
    fmla v24.4s, v10.4s, v2.s[2]
    fmla v26.4s, v10.4s, v3.s[2]
    fmla v28.4s, v10.4s, v4.s[2]
    fmla v21.4s, v11.4s, v0.s[2]
    fmla v23.4s, v11.4s, v1.s[2]
    fmla v25.4s, v11.4s, v2.s[2]
    fmla v27.4s, v11.4s, v3.s[2]
    fmla v29.4s, v11.4s, v4.s[2]
    fmla v20.4s, v12.4s, v0.s[3]
    fmla v22.4s, v12.4s, v1.s[3]
    fmla v24.4s, v12.4s, v2.s[3]
    fmla v26.4s, v12.4s, v3.s[3]
    fmla v28.4s, v12.4s, v4.s[3]
    fmla v21.4s, v13.4s, v0.s[3]
    fmla v23.4s, v13.4s, v1.s[3]
    fmla v25.4s, v13.4s, v2.s[3]
    fmla v27.4s, v13.4s, v3.s[3]
    fmla v29.4s, v13.4s, v4.s[3]
KAI_ASM_LABEL(label_116)  // Height 5: Multiply loop: Main loop skip
    cbz x27, label_118
KAI_ASM_LABEL(label_117)  // Height 5: Multiply loop: Odd block loop
    ldr s0, [x26], #0x4
    ldr s1, [x25], #0x4
    sub x27, x27, #0x1
    ldr s2, [x24], #0x4
    ldr s3, [x23], #0x4
    ldr s4, [x22], #0x4
    ldr q14, [x10, #0x0]
    ldr q15, [x10, #0x10]
    add x10, x10, #0x20
    fmla v20.4s, v14.4s, v0.s[0]
    fmla v22.4s, v14.4s, v1.s[0]
    fmla v24.4s, v14.4s, v2.s[0]
    fmla v26.4s, v14.4s, v3.s[0]
    fmla v28.4s, v14.4s, v4.s[0]
    fmla v21.4s, v15.4s, v0.s[0]
    fmla v23.4s, v15.4s, v1.s[0]
    fmla v25.4s, v15.4s, v2.s[0]
    fmla v27.4s, v15.4s, v3.s[0]
    fmla v29.4s, v15.4s, v4.s[0]
    cbnz x27, label_117
KAI_ASM_LABEL(label_118)  // Height 5: Multiply loop: No odd multiplies
    ldr w20, [x2, #0x8]
    add x28, x28, #0x1
    cmp x28, x20
    bne label_111
    ldr x20, [x2, #0x28]
    prfm pstl1keep, [x9, #0x0]
    add x26, x9, x20, LSL #2
    prfm pstl1keep, [x26, #0x0]
    add x25, x26, x20, LSL #2
    prfm pstl1keep, [x25, #0x0]
    add x24, x25, x20, LSL #2
    prfm pstl1keep, [x24, #0x0]
    add x23, x24, x20, LSL #2
    prfm pstl1keep, [x23, #0x0]
    tbz x3, #1, label_119
    add x21, x2, #0x0
    add x20, x2, #0x4
    ld1r { v17.4s }, [x21]
    ld1r { v16.4s }, [x20]
    fmin v20.4s, v20.4s, v17.4s
    fmin v21.4s, v21.4s, v17.4s
    fmin v22.4s, v22.4s, v17.4s
    fmin v23.4s, v23.4s, v17.4s
    fmin v24.4s, v24.4s, v17.4s
    fmin v25.4s, v25.4s, v17.4s
    fmin v26.4s, v26.4s, v17.4s
    fmin v27.4s, v27.4s, v17.4s
    fmin v28.4s, v28.4s, v17.4s
    fmin v29.4s, v29.4s, v17.4s
    fmax v20.4s, v20.4s, v16.4s
    fmax v21.4s, v21.4s, v16.4s
    fmax v22.4s, v22.4s, v16.4s
    fmax v23.4s, v23.4s, v16.4s
    fmax v24.4s, v24.4s, v16.4s
    fmax v25.4s, v25.4s, v16.4s
    fmax v26.4s, v26.4s, v16.4s
    fmax v27.4s, v27.4s, v16.4s
    fmax v28.4s, v28.4s, v16.4s
    fmax v29.4s, v29.4s, v16.4s
KAI_ASM_LABEL(label_119)  // Height 5: No activation
    cmp x11, #0x8
    bge label_124
    tbz x11, #2, label_121
    st1 { v20.4s }, [x9], #0x10
    st1 { v22.4s }, [x26], #0x10
    st1 { v24.4s }, [x25], #0x10
    st1 { v26.4s }, [x24], #0x10
    st1 { v28.4s }, [x23], #0x10
    tbz x11, #1, label_120
    str d21, [x9], #0x8
    str d23, [x26], #0x8
    str d25, [x25], #0x8
    str d27, [x24], #0x8
    str d29, [x23], #0x8
    tbz x11, #0, label_123
    st1 { v21.s }[2], [x9]
    st1 { v23.s }[2], [x26]
    st1 { v25.s }[2], [x25]
    st1 { v27.s }[2], [x24]
    st1 { v29.s }[2], [x23]
    b label_123
KAI_ASM_LABEL(label_120)  // Height 5: Partial direct writeback: partial_1_4
    tbz x11, #0, label_123
    str s21, [x9, #0x0]
    str s23, [x26, #0x0]
    str s25, [x25, #0x0]
    str s27, [x24, #0x0]
    str s29, [x23, #0x0]
    b label_123
KAI_ASM_LABEL(label_121)  // Height 5: Partial direct writeback: partial_2_0
    tbz x11, #1, label_122
    str d20, [x9], #0x8
    str d22, [x26], #0x8
    str d24, [x25], #0x8
    str d26, [x24], #0x8
    str d28, [x23], #0x8
    tbz x11, #0, label_123
    st1 { v20.s }[2], [x9]
    st1 { v22.s }[2], [x26]
    st1 { v24.s }[2], [x25]
    st1 { v26.s }[2], [x24]
    st1 { v28.s }[2], [x23]
    b label_123
KAI_ASM_LABEL(label_122)  // Height 5: Partial direct writeback: partial_1_0
    str s20, [x9, #0x0]
    str s22, [x26, #0x0]
    str s24, [x25, #0x0]
    str s26, [x24, #0x0]
    str s28, [x23, #0x0]
KAI_ASM_LABEL(label_123)  // Height 5: Partial direct writeback: Done
    b label_125
KAI_ASM_LABEL(label_124)  // Height 5: Full writeback
    str q20, [x9, #0x0]
    str q21, [x9, #0x10]
    add x9, x9, #0x20
    str q22, [x26, #0x0]
    str q23, [x26, #0x10]
    str q24, [x25, #0x0]
    str q25, [x25, #0x10]
    str q26, [x24, #0x0]
    str q27, [x24, #0x10]
    str q28, [x23, #0x0]
    str q29, [x23, #0x10]
KAI_ASM_LABEL(label_125)  // Height 5: Writeback done
    subs x11, x11, #0x8
    bgt label_102
    b label_152
KAI_ASM_LABEL(label_126)  // Height 6
    ldr x21, [x2, #0x28]
    ldr x9, [x2, #0x40]
    mov x20, #0x18
    ldr x11, [x2, #0x18]
    ldr x10, [x2, #0x20]
    madd x20, x21, x20, x9
    str x20, [x2, #0x40]
KAI_ASM_LABEL(label_127)  // Height 6: Column loop
    cbz x10, label_128
    ldr q20, [x10, #0x0]
    ldr q21, [x10, #0x10]
    add x10, x10, #0x20
    mov v22.16b, v20.16b
    mov v23.16b, v21.16b
    mov v24.16b, v20.16b
    mov v25.16b, v21.16b
    mov v26.16b, v20.16b
    mov v27.16b, v21.16b
    mov v28.16b, v20.16b
    mov v29.16b, v21.16b
    mov v30.16b, v20.16b
    mov v31.16b, v21.16b
    b label_135
KAI_ASM_LABEL(label_128)  // Height 6: no bias
    tbz x3, #0, label_134
    ldr x20, [x2, #0x28]
    cmp x11, #0x8
    add x26, x9, x20, LSL #2
    add x25, x26, x20, LSL #2
    add x24, x25, x20, LSL #2
    add x23, x24, x20, LSL #2
    add x22, x23, x20, LSL #2
    bge label_133
    tbz x11, #2, label_130
    ld1 { v20.4s }, [x9], #0x10
    ld1 { v22.4s }, [x26], #0x10
    ld1 { v24.4s }, [x25], #0x10
    ld1 { v26.4s }, [x24], #0x10
    ld1 { v28.4s }, [x23], #0x10
    ld1 { v30.4s }, [x22], #0x10
    tbz x11, #1, label_129
    ldr d21, [x9], #0x8
    ldr d23, [x26], #0x8
    mov x20, #0x18
    ldr d25, [x25], #0x8
    ldr d27, [x24], #0x8
    ldr d29, [x23], #0x8
    ldr d31, [x22], #0x8
    tbz x11, #0, label_132
    ld1 { v21.s }[2], [x9]
    ld1 { v23.s }[2], [x26]
    ld1 { v25.s }[2], [x25]
    ld1 { v27.s }[2], [x24]
    ld1 { v29.s }[2], [x23]
    ld1 { v31.s }[2], [x22]
    b label_132
KAI_ASM_LABEL(label_129)  // Height 6: Partial accumulate: partial_1_4
    mov x20, #0x10
    tbz x11, #0, label_132
    ldr s21, [x9, #0x0]
    ldr s23, [x26, #0x0]
    ldr s25, [x25, #0x0]
    ldr s27, [x24, #0x0]
    ldr s29, [x23, #0x0]
    ldr s31, [x22, #0x0]
    b label_132
KAI_ASM_LABEL(label_130)  // Height 6: Partial accumulate: partial_2_0
    tbz x11, #1, label_131
    ldr d20, [x9], #0x8
    ldr d22, [x26], #0x8
    mov x20, #0x8
    ldr d24, [x25], #0x8
    ldr d26, [x24], #0x8
    ldr d28, [x23], #0x8
    ldr d30, [x22], #0x8
    tbz x11, #0, label_132
    ld1 { v20.s }[2], [x9]
    ld1 { v22.s }[2], [x26]
    ld1 { v24.s }[2], [x25]
    ld1 { v26.s }[2], [x24]
    ld1 { v28.s }[2], [x23]
    ld1 { v30.s }[2], [x22]
    b label_132
KAI_ASM_LABEL(label_131)  // Height 6: Partial accumulate: partial_1_0
    ldr s20, [x9, #0x0]
    ldr s22, [x26, #0x0]
    mov x20, #0x0
    ldr s24, [x25, #0x0]
    ldr s26, [x24, #0x0]
    ldr s28, [x23, #0x0]
    ldr s30, [x22, #0x0]
KAI_ASM_LABEL(label_132)  // Height 6: Partial accumulate: Done
    sub x9, x9, x20
    b label_135
KAI_ASM_LABEL(label_133)  // Height 6: full accumulate
    ldr q20, [x9, #0x0]
    ldr q21, [x9, #0x10]
    ldr q22, [x26, #0x0]
    ldr q23, [x26, #0x10]
    ldr q24, [x25, #0x0]
    ldr q25, [x25, #0x10]
    ldr q26, [x24, #0x0]
    ldr q27, [x24, #0x10]
    ldr q28, [x23, #0x0]
    ldr q29, [x23, #0x10]
    ldr q30, [x22, #0x0]
    ldr q31, [x22, #0x10]
    b label_135
KAI_ASM_LABEL(label_134)  // Height 6: no accumulate
    movi v20.16b, #0x0
    movi v21.16b, #0x0
    movi v22.16b, #0x0
    movi v23.16b, #0x0
    movi v24.16b, #0x0
    movi v25.16b, #0x0
    movi v26.16b, #0x0
    movi v27.16b, #0x0
    movi v28.16b, #0x0
    movi v29.16b, #0x0
    movi v30.16b, #0x0
    movi v31.16b, #0x0
KAI_ASM_LABEL(label_135)  // Height 6: setup done
    mov x28, #0x0
KAI_ASM_LABEL(label_136)  // Height 6: String loop
    ldr x20, [x2, #0x10]
    ldr x21, [x2, #0x38]
    ldr w27, [x20, x28, LSL #0x2]
    tbz x3, #3, label_137
    ldr x20, [x0, x28, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x26, [x20, #0x0]
    ldr x25, [x20, #0x8]
    ldr x24, [x20, #0x10]
    ldr x23, [x20, #0x18]
    ldr x22, [x20, #0x20]
    ldr x21, [x20, #0x28]
    cbnz x28, label_138
    ldr x20, [x2, #0x30]
    add x26, x26, x20, LSL #2
    add x25, x25, x20, LSL #2
    add x24, x24, x20, LSL #2
    add x23, x23, x20, LSL #2
    add x22, x22, x20, LSL #2
    add x21, x21, x20, LSL #2
    b label_138
KAI_ASM_LABEL(label_137)  // Height 6: setup direct input
    mov x26, x0
    add x25, x26, x21, LSL #2
    add x24, x25, x21, LSL #2
    add x23, x24, x21, LSL #2
    add x22, x23, x21, LSL #2
    add x21, x22, x21, LSL #2
KAI_ASM_LABEL(label_138)  // Height 6: input setup done
    cmp x27, #0x4
    blt label_141
    ldr q0, [x26, #0x0]
    ldr q1, [x25, #0x0]
    cmp x27, #0x8
    ldr q2, [x24, #0x0]
    ldr q3, [x23, #0x0]
    ldr q4, [x22, #0x0]
    ldr q5, [x21, #0x0]
    ldr q6, [x10, #0x0]
    ldr q7, [x10, #0x10]
    ldr q8, [x10, #0x20]
    ldr q9, [x10, #0x30]
    ldr q10, [x10, #0x40]
    ldr q11, [x10, #0x50]
    ldr q12, [x10, #0x60]
    ldr q13, [x10, #0x70]
    blt label_140
KAI_ASM_LABEL(label_139)  // Height 6: Multiply loop: Main loop head
    fmla v20.4s, v6.4s, v0.s[0]
    fmla v22.4s, v6.4s, v1.s[0]
    sub x27, x27, #0x4
    add x26, x26, #0x10
    fmla v24.4s, v6.4s, v2.s[0]
    fmla v26.4s, v6.4s, v3.s[0]
    add x25, x25, #0x10
    add x24, x24, #0x10
    fmla v28.4s, v6.4s, v4.s[0]
    fmla v30.4s, v6.4s, v5.s[0]
    add x23, x23, #0x10
    add x22, x22, #0x10
    fmla v21.4s, v7.4s, v0.s[0]
    fmla v23.4s, v7.4s, v1.s[0]
    add x21, x21, #0x10
    cmp x27, #0x8
    fmla v25.4s, v7.4s, v2.s[0]
    fmla v27.4s, v7.4s, v3.s[0]
    add x10, x10, #0x80
    prfm pldl1keep, [x26, #0x80]
    ldr q6, [x10, #0x0]
    fmla v29.4s, v7.4s, v4.s[0]
    fmla v31.4s, v7.4s, v5.s[0]
    ldr q7, [x10, #0x10]
    fmla v20.4s, v8.4s, v0.s[1]
    fmla v22.4s, v8.4s, v1.s[1]
    prfm pldl1keep, [x25, #0x80]
    prfm pldl1keep, [x24, #0x80]
    fmla v24.4s, v8.4s, v2.s[1]
    fmla v26.4s, v8.4s, v3.s[1]
    prfm pldl1keep, [x23, #0x80]
    prfm pldl1keep, [x22, #0x80]
    fmla v28.4s, v8.4s, v4.s[1]
    fmla v30.4s, v8.4s, v5.s[1]
    ldr q8, [x10, #0x20]
    prfm pldl1keep, [x21, #0x80]
    fmla v21.4s, v9.4s, v0.s[1]
    fmla v23.4s, v9.4s, v1.s[1]
    fmla v25.4s, v9.4s, v2.s[1]
    fmla v27.4s, v9.4s, v3.s[1]
    fmla v29.4s, v9.4s, v4.s[1]
    fmla v31.4s, v9.4s, v5.s[1]
    ldr q9, [x10, #0x30]
    fmla v20.4s, v10.4s, v0.s[2]
    fmla v22.4s, v10.4s, v1.s[2]
    fmla v24.4s, v10.4s, v2.s[2]
    fmla v26.4s, v10.4s, v3.s[2]
    fmla v28.4s, v10.4s, v4.s[2]
    fmla v30.4s, v10.4s, v5.s[2]
    ldr q10, [x10, #0x40]
    fmla v21.4s, v11.4s, v0.s[2]
    fmla v23.4s, v11.4s, v1.s[2]
    fmla v25.4s, v11.4s, v2.s[2]
    fmla v27.4s, v11.4s, v3.s[2]
    fmla v29.4s, v11.4s, v4.s[2]
    fmla v31.4s, v11.4s, v5.s[2]
    ldr q11, [x10, #0x50]
    fmla v20.4s, v12.4s, v0.s[3]
    fmla v22.4s, v12.4s, v1.s[3]
    fmla v24.4s, v12.4s, v2.s[3]
    fmla v26.4s, v12.4s, v3.s[3]
    fmla v28.4s, v12.4s, v4.s[3]
    fmla v30.4s, v12.4s, v5.s[3]
    ldr q12, [x10, #0x60]
    fmla v21.4s, v13.4s, v0.s[3]
    ldr q0, [x26, #0x0]
    fmla v23.4s, v13.4s, v1.s[3]
    ldr q1, [x25, #0x0]
    fmla v25.4s, v13.4s, v2.s[3]
    ldr q2, [x24, #0x0]
    fmla v27.4s, v13.4s, v3.s[3]
    ldr q3, [x23, #0x0]
    fmla v29.4s, v13.4s, v4.s[3]
    ldr q4, [x22, #0x0]
    fmla v31.4s, v13.4s, v5.s[3]
    ldr q5, [x21, #0x0]
    ldr q13, [x10, #0x70]
    bge label_139
KAI_ASM_LABEL(label_140)  // Height 6: Multiply loop: Single iteration only
    fmla v20.4s, v6.4s, v0.s[0]
    fmla v22.4s, v6.4s, v1.s[0]
    add x26, x26, #0x10
    add x25, x25, #0x10
    fmla v24.4s, v6.4s, v2.s[0]
    fmla v26.4s, v6.4s, v3.s[0]
    add x24, x24, #0x10
    add x23, x23, #0x10
    fmla v28.4s, v6.4s, v4.s[0]
    fmla v30.4s, v6.4s, v5.s[0]
    add x22, x22, #0x10
    add x21, x21, #0x10
    fmla v21.4s, v7.4s, v0.s[0]
    fmla v23.4s, v7.4s, v1.s[0]
    sub x27, x27, #0x4
    prfm pldl1keep, [x26, #0x80]
    fmla v25.4s, v7.4s, v2.s[0]
    fmla v27.4s, v7.4s, v3.s[0]
    prfm pldl1keep, [x25, #0x80]
    prfm pldl1keep, [x24, #0x80]
    fmla v29.4s, v7.4s, v4.s[0]
    fmla v31.4s, v7.4s, v5.s[0]
    prfm pldl1keep, [x23, #0x80]
    prfm pldl1keep, [x22, #0x80]
    fmla v20.4s, v8.4s, v0.s[1]
    fmla v22.4s, v8.4s, v1.s[1]
    prfm pldl1keep, [x21, #0x80]
    add x10, x10, #0x80
    fmla v24.4s, v8.4s, v2.s[1]
    fmla v26.4s, v8.4s, v3.s[1]
    fmla v28.4s, v8.4s, v4.s[1]
    fmla v30.4s, v8.4s, v5.s[1]
    fmla v21.4s, v9.4s, v0.s[1]
    fmla v23.4s, v9.4s, v1.s[1]
    fmla v25.4s, v9.4s, v2.s[1]
    fmla v27.4s, v9.4s, v3.s[1]
    fmla v29.4s, v9.4s, v4.s[1]
    fmla v31.4s, v9.4s, v5.s[1]
    fmla v20.4s, v10.4s, v0.s[2]
    fmla v22.4s, v10.4s, v1.s[2]
    fmla v24.4s, v10.4s, v2.s[2]
    fmla v26.4s, v10.4s, v3.s[2]
    fmla v28.4s, v10.4s, v4.s[2]
    fmla v30.4s, v10.4s, v5.s[2]
    fmla v21.4s, v11.4s, v0.s[2]
    fmla v23.4s, v11.4s, v1.s[2]
    fmla v25.4s, v11.4s, v2.s[2]
    fmla v27.4s, v11.4s, v3.s[2]
    fmla v29.4s, v11.4s, v4.s[2]
    fmla v31.4s, v11.4s, v5.s[2]
    fmla v20.4s, v12.4s, v0.s[3]
    fmla v22.4s, v12.4s, v1.s[3]
    fmla v24.4s, v12.4s, v2.s[3]
    fmla v26.4s, v12.4s, v3.s[3]
    fmla v28.4s, v12.4s, v4.s[3]
    fmla v30.4s, v12.4s, v5.s[3]
    fmla v21.4s, v13.4s, v0.s[3]
    fmla v23.4s, v13.4s, v1.s[3]
    fmla v25.4s, v13.4s, v2.s[3]
    fmla v27.4s, v13.4s, v3.s[3]
    fmla v29.4s, v13.4s, v4.s[3]
    fmla v31.4s, v13.4s, v5.s[3]
KAI_ASM_LABEL(label_141)  // Height 6: Multiply loop: Main loop skip
    cbz x27, label_143
KAI_ASM_LABEL(label_142)  // Height 6: Multiply loop: Odd block loop
    ldr s0, [x26], #0x4
    ldr s1, [x25], #0x4
    sub x27, x27, #0x1
    ldr s2, [x24], #0x4
    ldr s3, [x23], #0x4
    ldr s4, [x22], #0x4
    ldr s5, [x21], #0x4
    ldr q14, [x10, #0x0]
    ldr q15, [x10, #0x10]
    add x10, x10, #0x20
    fmla v20.4s, v14.4s, v0.s[0]
    fmla v22.4s, v14.4s, v1.s[0]
    fmla v24.4s, v14.4s, v2.s[0]
    fmla v26.4s, v14.4s, v3.s[0]
    fmla v28.4s, v14.4s, v4.s[0]
    fmla v30.4s, v14.4s, v5.s[0]
    fmla v21.4s, v15.4s, v0.s[0]
    fmla v23.4s, v15.4s, v1.s[0]
    fmla v25.4s, v15.4s, v2.s[0]
    fmla v27.4s, v15.4s, v3.s[0]
    fmla v29.4s, v15.4s, v4.s[0]
    fmla v31.4s, v15.4s, v5.s[0]
    cbnz x27, label_142
KAI_ASM_LABEL(label_143)  // Height 6: Multiply loop: No odd multiplies
    ldr w20, [x2, #0x8]
    add x28, x28, #0x1
    cmp x28, x20
    bne label_136
    ldr x20, [x2, #0x28]
    prfm pstl1keep, [x9, #0x0]
    add x26, x9, x20, LSL #2
    prfm pstl1keep, [x26, #0x0]
    add x25, x26, x20, LSL #2
    prfm pstl1keep, [x25, #0x0]
    add x24, x25, x20, LSL #2
    prfm pstl1keep, [x24, #0x0]
    add x23, x24, x20, LSL #2
    add x22, x23, x20, LSL #2
    prfm pstl1keep, [x23, #0x0]
    prfm pstl1keep, [x22, #0x0]
    tbz x3, #1, label_144
    add x21, x2, #0x0
    add x20, x2, #0x4
    ld1r { v17.4s }, [x21]
    ld1r { v16.4s }, [x20]
    fmin v20.4s, v20.4s, v17.4s
    fmin v21.4s, v21.4s, v17.4s
    fmin v22.4s, v22.4s, v17.4s
    fmin v23.4s, v23.4s, v17.4s
    fmin v24.4s, v24.4s, v17.4s
    fmin v25.4s, v25.4s, v17.4s
    fmin v26.4s, v26.4s, v17.4s
    fmin v27.4s, v27.4s, v17.4s
    fmin v28.4s, v28.4s, v17.4s
    fmin v29.4s, v29.4s, v17.4s
    fmin v30.4s, v30.4s, v17.4s
    fmin v31.4s, v31.4s, v17.4s
    fmax v20.4s, v20.4s, v16.4s
    fmax v21.4s, v21.4s, v16.4s
    fmax v22.4s, v22.4s, v16.4s
    fmax v23.4s, v23.4s, v16.4s
    fmax v24.4s, v24.4s, v16.4s
    fmax v25.4s, v25.4s, v16.4s
    fmax v26.4s, v26.4s, v16.4s
    fmax v27.4s, v27.4s, v16.4s
    fmax v28.4s, v28.4s, v16.4s
    fmax v29.4s, v29.4s, v16.4s
    fmax v30.4s, v30.4s, v16.4s
    fmax v31.4s, v31.4s, v16.4s
KAI_ASM_LABEL(label_144)  // Height 6: No activation
    cmp x11, #0x8
    bge label_149
    tbz x11, #2, label_146
    st1 { v20.4s }, [x9], #0x10
    st1 { v22.4s }, [x26], #0x10
    st1 { v24.4s }, [x25], #0x10
    st1 { v26.4s }, [x24], #0x10
    st1 { v28.4s }, [x23], #0x10
    st1 { v30.4s }, [x22], #0x10
    tbz x11, #1, label_145
    str d21, [x9], #0x8
    str d23, [x26], #0x8
    str d25, [x25], #0x8
    str d27, [x24], #0x8
    str d29, [x23], #0x8
    str d31, [x22], #0x8
    tbz x11, #0, label_148
    st1 { v21.s }[2], [x9]
    st1 { v23.s }[2], [x26]
    st1 { v25.s }[2], [x25]
    st1 { v27.s }[2], [x24]
    st1 { v29.s }[2], [x23]
    st1 { v31.s }[2], [x22]
    b label_148
KAI_ASM_LABEL(label_145)  // Height 6: Partial direct writeback: partial_1_4
    tbz x11, #0, label_148
    str s21, [x9, #0x0]
    str s23, [x26, #0x0]
    str s25, [x25, #0x0]
    str s27, [x24, #0x0]
    str s29, [x23, #0x0]
    str s31, [x22, #0x0]
    b label_148
KAI_ASM_LABEL(label_146)  // Height 6: Partial direct writeback: partial_2_0
    tbz x11, #1, label_147
    str d20, [x9], #0x8
    str d22, [x26], #0x8
    str d24, [x25], #0x8
    str d26, [x24], #0x8
    str d28, [x23], #0x8
    str d30, [x22], #0x8
    tbz x11, #0, label_148
    st1 { v20.s }[2], [x9]
    st1 { v22.s }[2], [x26]
    st1 { v24.s }[2], [x25]
    st1 { v26.s }[2], [x24]
    st1 { v28.s }[2], [x23]
    st1 { v30.s }[2], [x22]
    b label_148
KAI_ASM_LABEL(label_147)  // Height 6: Partial direct writeback: partial_1_0
    str s20, [x9, #0x0]
    str s22, [x26, #0x0]
    str s24, [x25, #0x0]
    str s26, [x24, #0x0]
    str s28, [x23, #0x0]
    str s30, [x22, #0x0]
KAI_ASM_LABEL(label_148)  // Height 6: Partial direct writeback: Done
    b label_150
KAI_ASM_LABEL(label_149)  // Height 6: Full writeback
    str q20, [x9, #0x0]
    str q21, [x9, #0x10]
    add x9, x9, #0x20
    str q22, [x26, #0x0]
    str q23, [x26, #0x10]
    str q24, [x25, #0x0]
    str q25, [x25, #0x10]
    str q26, [x24, #0x0]
    str q27, [x24, #0x10]
    str q28, [x23, #0x0]
    str q29, [x23, #0x10]
    str q30, [x22, #0x0]
    str q31, [x22, #0x10]
KAI_ASM_LABEL(label_150)  // Height 6: Writeback done
    subs x11, x11, #0x8
    bgt label_127
    subs x1, x1, #0x6
    beq label_152
    ldr x21, [x2, #0x38]
    tbz x3, #3, label_151
    add x21, x21, #0x6
    str x21, [x2, #0x38]
    b label_1
KAI_ASM_LABEL(label_151)  // Update direct input
    mov x20, #0x18
    madd x0, x20, x21, x0
    b label_1
KAI_ASM_LABEL(label_152)  // Exit
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp d10, d11, [sp, 72]
    ldp d12, d13, [sp, 88]
    ldp d14, d15, [sp, 104]
    ldp d8, d9, [sp, 120]
    ldp x20, x21, [sp], 144
    ret

    KAI_ASM_FOOTER
