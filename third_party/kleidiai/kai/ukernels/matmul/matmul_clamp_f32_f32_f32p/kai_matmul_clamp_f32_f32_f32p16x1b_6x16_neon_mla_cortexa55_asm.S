//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if defined(_MSC_VER)
    #define KAI_ASM_GLOBAL(name) GLOBAL name
    #define KAI_ASM_FUNCTION_TYPE(name)
    #define KAI_ASM_FUNCTION_LABEL(name) name PROC
    #define KAI_ASM_FUNCTION_END(name) ENDP

    #define KAI_ASM_CODE(name) AREA name, CODE, READONLY
    #define KAI_ASM_ALIGN
    #define KAI_ASM_LABEL(name) name
    #define KAI_ASM_INST(hex) DCD hex
    #define KAI_ASM_END END
#else
    #if defined(__APPLE__)
        #define KAI_ASM_GLOBAL(name) .globl _##name
        #define KAI_ASM_FUNCTION_TYPE(name)
        #define KAI_ASM_FUNCTION_LABEL(name) _##name:
        #define KAI_ASM_FUNCTION_END(name)
    #else
        #define KAI_ASM_GLOBAL(name) .global name
        #define KAI_ASM_FUNCTION_TYPE(name) .type name, %function
        #define KAI_ASM_FUNCTION_LABEL(name) name:
        #define KAI_ASM_FUNCTION_END(name) .size name, .-name
    #endif

    #define KAI_ASM_CODE(name) .text
    #define KAI_ASM_ALIGN .p2align 4,,11
    #define KAI_ASM_LABEL(name) name:
    #define KAI_ASM_INST(hex) .inst hex
    #define KAI_ASM_END
#endif

    KAI_ASM_CODE(matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_cortexa55)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_cortexa55)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_cortexa55)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_cortexa55)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
KAI_ASM_LABEL(label_1)  // Row loop
    cmp x1, #0x6
    bge label_166
    cmp x1, #0x4
    bgt label_133
    beq label_100
    cmp x1, #0x2
    bgt label_67
    beq label_34
    ldr x8, [x2, #0x18]
    ldr x17, [x2, #0x20]
    ldr x16, [x2, #0x40]
KAI_ASM_LABEL(label_2)  // Height 1: Column loop
    cbz x17, label_3
    ldr q8, [x17, #0x0]
    ldr q9, [x17, #0x10]
    ldr q10, [x17, #0x20]
    ldr q11, [x17, #0x30]
    add x17, x17, #0x40
    b label_14
KAI_ASM_LABEL(label_3)  // Height 1: no bias
    tbz x3, #0, label_13
    cmp x8, #0x10
    bge label_12
    tbz x8, #3, label_7
    ld1 { v8.4s }, [x16], #0x10
    ld1 { v9.4s }, [x16], #0x10
    tbz x8, #2, label_5
    ld1 { v10.4s }, [x16], #0x10
    tbz x8, #1, label_4
    ldr d11, [x16], #0x8
    mov x20, #0x38
    tbz x8, #0, label_11
    ld1 { v11.s }[2], [x16]
    b label_11
KAI_ASM_LABEL(label_4)  // Height 1: Partial accumulate: partial_1_12
    mov x20, #0x30
    tbz x8, #0, label_11
    ldr s11, [x16, #0x0]
    b label_11
KAI_ASM_LABEL(label_5)  // Height 1: Partial accumulate: partial_2_8
    tbz x8, #1, label_6
    ldr d10, [x16], #0x8
    mov x20, #0x28
    tbz x8, #0, label_11
    ld1 { v10.s }[2], [x16]
    b label_11
KAI_ASM_LABEL(label_6)  // Height 1: Partial accumulate: partial_1_8
    mov x20, #0x20
    tbz x8, #0, label_11
    ldr s10, [x16, #0x0]
    b label_11
KAI_ASM_LABEL(label_7)  // Height 1: Partial accumulate: partial_4_0
    tbz x8, #2, label_9
    ld1 { v8.4s }, [x16], #0x10
    tbz x8, #1, label_8
    ldr d9, [x16], #0x8
    mov x20, #0x18
    tbz x8, #0, label_11
    ld1 { v9.s }[2], [x16]
    b label_11
KAI_ASM_LABEL(label_8)  // Height 1: Partial accumulate: partial_1_4
    mov x20, #0x10
    tbz x8, #0, label_11
    ldr s9, [x16, #0x0]
    b label_11
KAI_ASM_LABEL(label_9)  // Height 1: Partial accumulate: partial_2_0
    tbz x8, #1, label_10
    ldr d8, [x16], #0x8
    mov x20, #0x8
    tbz x8, #0, label_11
    ld1 { v8.s }[2], [x16]
    b label_11
KAI_ASM_LABEL(label_10)  // Height 1: Partial accumulate: partial_1_0
    ldr s8, [x16, #0x0]
    mov x20, #0x0
KAI_ASM_LABEL(label_11)  // Height 1: Partial accumulate: Done
    sub x16, x16, x20
    b label_14
KAI_ASM_LABEL(label_12)  // Height 1: full accumulate
    ldr q8, [x16, #0x0]
    ldr q9, [x16, #0x10]
    ldr q10, [x16, #0x20]
    ldr q11, [x16, #0x30]
    b label_14
KAI_ASM_LABEL(label_13)  // Height 1: no accumulate
    movi v8.16b, #0x0
    movi v9.16b, #0x0
    movi v10.16b, #0x0
    movi v11.16b, #0x0
KAI_ASM_LABEL(label_14)  // Height 1: setup done
    mov x15, #0x0
KAI_ASM_LABEL(label_15)  // Height 1: String loop
    ldr x20, [x2, #0x10]
    ldr x21, [x2, #0x38]
    ldr w14, [x20, x15, LSL #0x2]
    tbz x3, #3, label_16
    ldr x20, [x0, x15, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x13, [x20, #0x0]
    cbnz x15, label_17
    ldr x20, [x2, #0x30]
    add x13, x13, x20, LSL #2
    b label_17
KAI_ASM_LABEL(label_16)  // Height 1: setup direct input
    mov x13, x0
KAI_ASM_LABEL(label_17)  // Height 1: input setup done
    cmp x14, #0x4
    blt label_20
    ldr q0, [x13, #0x0]
    cmp x14, #0x8
    ldr q6, [x17, #0x0]
    ldr q7, [x17, #0x10]
    blt label_19
KAI_ASM_LABEL(label_18)  // Height 1: Multiply loop: Main loop head
    fmla v8.4s, v6.4s, v0.s[0]
    ldr d6, [x17, #0x20]
    ldr x21, [x17, #0x28]
    fmla v9.4s, v7.4s, v0.s[0]
    ldr d7, [x17, #0x30]
    add x13, x13, #0x10
    ldr x20, [x17, #0x38]
    sub x14, x14, #0x4
    mov v6.d[1], x21
    ldr x21, [x17, #0x48]
    ldr x22, [x13, #0x8]
    cmp x14, #0x8
    mov v7.d[1], x20
    fmla v10.4s, v6.4s, v0.s[0]
    ldr d6, [x17, #0x40]
    fmla v11.4s, v7.4s, v0.s[0]
    ldr d7, [x17, #0x50]
    ldr x20, [x17, #0x58]
    mov v6.d[1], x21
    ldr x21, [x17, #0x68]
    prfm pldl1keep, [x13, #0x80]
    mov v7.d[1], x20
    fmla v8.4s, v6.4s, v0.s[1]
    ldr d6, [x17, #0x60]
    fmla v9.4s, v7.4s, v0.s[1]
    ldr d7, [x17, #0x70]
    ldr x20, [x17, #0x78]
    mov v6.d[1], x21
    ldr x21, [x17, #0x88]
    mov v7.d[1], x20
    fmla v10.4s, v6.4s, v0.s[1]
    ldr d6, [x17, #0x80]
    fmla v11.4s, v7.4s, v0.s[1]
    ldr d7, [x17, #0x90]
    ldr x20, [x17, #0x98]
    mov v6.d[1], x21
    ldr x21, [x17, #0xa8]
    mov v7.d[1], x20
    fmla v8.4s, v6.4s, v0.s[2]
    ldr d6, [x17, #0xa0]
    fmla v9.4s, v7.4s, v0.s[2]
    ldr d7, [x17, #0xb0]
    ldr x20, [x17, #0xb8]
    mov v6.d[1], x21
    ldr x21, [x17, #0xc8]
    mov v7.d[1], x20
    fmla v10.4s, v6.4s, v0.s[2]
    ldr d6, [x17, #0xc0]
    fmla v11.4s, v7.4s, v0.s[2]
    ldr d7, [x17, #0xd0]
    ldr x20, [x17, #0xd8]
    mov v6.d[1], x21
    ldr x21, [x17, #0xe8]
    mov v7.d[1], x20
    fmla v8.4s, v6.4s, v0.s[3]
    ldr d6, [x17, #0xe0]
    fmla v9.4s, v7.4s, v0.s[3]
    ldr d7, [x17, #0xf0]
    ldr x20, [x17, #0xf8]
    add x17, x17, #0x100
    mov v6.d[1], x21
    ldr x21, [x17, #0x8]
    mov v7.d[1], x20
    fmla v10.4s, v6.4s, v0.s[3]
    ldr d6, [x17, #0x0]
    fmla v11.4s, v7.4s, v0.s[3]
    ldr d0, [x13, #0x0]
    ldr d7, [x17, #0x10]
    ldr x20, [x17, #0x18]
    mov v6.d[1], x21
    mov v0.d[1], x22
    mov v7.d[1], x20
    bge label_18
KAI_ASM_LABEL(label_19)  // Height 1: Multiply loop: Single iteration only
    fmla v8.4s, v6.4s, v0.s[0]
    ldr q6, [x17, #0x20]
    fmla v9.4s, v7.4s, v0.s[0]
    ldr q7, [x17, #0x30]
    add x13, x13, #0x10
    sub x14, x14, #0x4
    fmla v10.4s, v6.4s, v0.s[0]
    ldr q6, [x17, #0x40]
    fmla v11.4s, v7.4s, v0.s[0]
    ldr q7, [x17, #0x50]
    prfm pldl1keep, [x13, #0x80]
    fmla v8.4s, v6.4s, v0.s[1]
    ldr q6, [x17, #0x60]
    fmla v9.4s, v7.4s, v0.s[1]
    ldr q7, [x17, #0x70]
    fmla v10.4s, v6.4s, v0.s[1]
    ldr q6, [x17, #0x80]
    fmla v11.4s, v7.4s, v0.s[1]
    ldr q7, [x17, #0x90]
    fmla v8.4s, v6.4s, v0.s[2]
    ldr q6, [x17, #0xa0]
    fmla v9.4s, v7.4s, v0.s[2]
    ldr q7, [x17, #0xb0]
    fmla v10.4s, v6.4s, v0.s[2]
    ldr q6, [x17, #0xc0]
    fmla v11.4s, v7.4s, v0.s[2]
    ldr q7, [x17, #0xd0]
    fmla v8.4s, v6.4s, v0.s[3]
    ldr q6, [x17, #0xe0]
    fmla v9.4s, v7.4s, v0.s[3]
    ldr q7, [x17, #0xf0]
    add x17, x17, #0x100
    fmla v10.4s, v6.4s, v0.s[3]
    fmla v11.4s, v7.4s, v0.s[3]
KAI_ASM_LABEL(label_20)  // Height 1: Multiply loop: Main loop skip
    cbz x14, label_22
KAI_ASM_LABEL(label_21)  // Height 1: Multiply loop: Odd block loop
    ldr s0, [x13], #0x4
    sub x14, x14, #0x1
    ldr q6, [x17, #0x0]
    ldr q7, [x17, #0x10]
    fmla v8.4s, v6.4s, v0.s[0]
    ldr q6, [x17, #0x20]
    fmla v9.4s, v7.4s, v0.s[0]
    ldr q7, [x17, #0x30]
    add x17, x17, #0x40
    fmla v10.4s, v6.4s, v0.s[0]
    fmla v11.4s, v7.4s, v0.s[0]
    cbnz x14, label_21
KAI_ASM_LABEL(label_22)  // Height 1: Multiply loop: No odd multiplies
    ldr w20, [x2, #0x8]
    add x15, x15, #0x1
    cmp x15, x20
    bne label_15
    prfm pstl1keep, [x16, #0x0]
    tbz x3, #1, label_23
    add x21, x2, #0x0
    add x20, x2, #0x4
    ld1r { v17.4s }, [x21]
    ld1r { v16.4s }, [x20]
    fmin v8.4s, v8.4s, v17.4s
    fmin v9.4s, v9.4s, v17.4s
    fmin v10.4s, v10.4s, v17.4s
    fmin v11.4s, v11.4s, v17.4s
    fmax v8.4s, v8.4s, v16.4s
    fmax v9.4s, v9.4s, v16.4s
    fmax v10.4s, v10.4s, v16.4s
    fmax v11.4s, v11.4s, v16.4s
KAI_ASM_LABEL(label_23)  // Height 1: No activation
    cmp x8, #0x10
    bge label_32
    tbz x8, #3, label_27
    st1 { v8.4s }, [x16], #0x10
    st1 { v9.4s }, [x16], #0x10
    tbz x8, #2, label_25
    st1 { v10.4s }, [x16], #0x10
    tbz x8, #1, label_24
    str d11, [x16], #0x8
    tbz x8, #0, label_31
    st1 { v11.s }[2], [x16]
    b label_31
KAI_ASM_LABEL(label_24)  // Height 1: Partial direct writeback: partial_1_12
    tbz x8, #0, label_31
    str s11, [x16, #0x0]
    b label_31
KAI_ASM_LABEL(label_25)  // Height 1: Partial direct writeback: partial_2_8
    tbz x8, #1, label_26
    str d10, [x16], #0x8
    tbz x8, #0, label_31
    st1 { v10.s }[2], [x16]
    b label_31
KAI_ASM_LABEL(label_26)  // Height 1: Partial direct writeback: partial_1_8
    tbz x8, #0, label_31
    str s10, [x16, #0x0]
    b label_31
KAI_ASM_LABEL(label_27)  // Height 1: Partial direct writeback: partial_4_0
    tbz x8, #2, label_29
    st1 { v8.4s }, [x16], #0x10
    tbz x8, #1, label_28
    str d9, [x16], #0x8
    tbz x8, #0, label_31
    st1 { v9.s }[2], [x16]
    b label_31
KAI_ASM_LABEL(label_28)  // Height 1: Partial direct writeback: partial_1_4
    tbz x8, #0, label_31
    str s9, [x16, #0x0]
    b label_31
KAI_ASM_LABEL(label_29)  // Height 1: Partial direct writeback: partial_2_0
    tbz x8, #1, label_30
    str d8, [x16], #0x8
    tbz x8, #0, label_31
    st1 { v8.s }[2], [x16]
    b label_31
KAI_ASM_LABEL(label_30)  // Height 1: Partial direct writeback: partial_1_0
    str s8, [x16, #0x0]
KAI_ASM_LABEL(label_31)  // Height 1: Partial direct writeback: Done
    b label_33
KAI_ASM_LABEL(label_32)  // Height 1: Full writeback
    str q8, [x16, #0x0]
    str q9, [x16, #0x10]
    str q10, [x16, #0x20]
    str q11, [x16, #0x30]
    add x16, x16, #0x40
KAI_ASM_LABEL(label_33)  // Height 1: Writeback done
    subs x8, x8, #0x10
    bgt label_2
    b label_200
KAI_ASM_LABEL(label_34)  // Height 2
    ldr x8, [x2, #0x18]
    ldr x17, [x2, #0x20]
    ldr x16, [x2, #0x40]
KAI_ASM_LABEL(label_35)  // Height 2: Column loop
    cbz x17, label_36
    ldr q8, [x17, #0x0]
    ldr q9, [x17, #0x10]
    ldr q10, [x17, #0x20]
    ldr q11, [x17, #0x30]
    mov v12.16b, v8.16b
    mov v13.16b, v9.16b
    add x17, x17, #0x40
    mov v14.16b, v10.16b
    mov v15.16b, v11.16b
    b label_47
KAI_ASM_LABEL(label_36)  // Height 2: no bias
    tbz x3, #0, label_46
    ldr x20, [x2, #0x28]
    cmp x8, #0x10
    add x26, x16, x20, LSL #2
    bge label_45
    tbz x8, #3, label_40
    ld1 { v8.4s }, [x16], #0x10
    ld1 { v12.4s }, [x26], #0x10
    ld1 { v9.4s }, [x16], #0x10
    ld1 { v13.4s }, [x26], #0x10
    tbz x8, #2, label_38
    ld1 { v10.4s }, [x16], #0x10
    ld1 { v14.4s }, [x26], #0x10
    tbz x8, #1, label_37
    ldr d11, [x16], #0x8
    mov x20, #0x38
    ldr d15, [x26], #0x8
    tbz x8, #0, label_44
    ld1 { v11.s }[2], [x16]
    ld1 { v15.s }[2], [x26]
    b label_44
KAI_ASM_LABEL(label_37)  // Height 2: Partial accumulate: partial_1_12
    mov x20, #0x30
    tbz x8, #0, label_44
    ldr s11, [x16, #0x0]
    ldr s15, [x26, #0x0]
    b label_44
KAI_ASM_LABEL(label_38)  // Height 2: Partial accumulate: partial_2_8
    tbz x8, #1, label_39
    ldr d10, [x16], #0x8
    mov x20, #0x28
    ldr d14, [x26], #0x8
    tbz x8, #0, label_44
    ld1 { v10.s }[2], [x16]
    ld1 { v14.s }[2], [x26]
    b label_44
KAI_ASM_LABEL(label_39)  // Height 2: Partial accumulate: partial_1_8
    mov x20, #0x20
    tbz x8, #0, label_44
    ldr s10, [x16, #0x0]
    ldr s14, [x26, #0x0]
    b label_44
KAI_ASM_LABEL(label_40)  // Height 2: Partial accumulate: partial_4_0
    tbz x8, #2, label_42
    ld1 { v8.4s }, [x16], #0x10
    ld1 { v12.4s }, [x26], #0x10
    tbz x8, #1, label_41
    ldr d9, [x16], #0x8
    mov x20, #0x18
    ldr d13, [x26], #0x8
    tbz x8, #0, label_44
    ld1 { v9.s }[2], [x16]
    ld1 { v13.s }[2], [x26]
    b label_44
KAI_ASM_LABEL(label_41)  // Height 2: Partial accumulate: partial_1_4
    mov x20, #0x10
    tbz x8, #0, label_44
    ldr s9, [x16, #0x0]
    ldr s13, [x26, #0x0]
    b label_44
KAI_ASM_LABEL(label_42)  // Height 2: Partial accumulate: partial_2_0
    tbz x8, #1, label_43
    ldr d8, [x16], #0x8
    mov x20, #0x8
    ldr d12, [x26], #0x8
    tbz x8, #0, label_44
    ld1 { v8.s }[2], [x16]
    ld1 { v12.s }[2], [x26]
    b label_44
KAI_ASM_LABEL(label_43)  // Height 2: Partial accumulate: partial_1_0
    ldr s8, [x16, #0x0]
    mov x20, #0x0
    ldr s12, [x26, #0x0]
KAI_ASM_LABEL(label_44)  // Height 2: Partial accumulate: Done
    sub x16, x16, x20
    b label_47
KAI_ASM_LABEL(label_45)  // Height 2: full accumulate
    ldr q8, [x16, #0x0]
    ldr q9, [x16, #0x10]
    ldr q10, [x16, #0x20]
    ldr q11, [x16, #0x30]
    ldr q12, [x26, #0x0]
    ldr q13, [x26, #0x10]
    ldr q14, [x26, #0x20]
    ldr q15, [x26, #0x30]
    b label_47
KAI_ASM_LABEL(label_46)  // Height 2: no accumulate
    movi v8.16b, #0x0
    movi v9.16b, #0x0
    movi v10.16b, #0x0
    movi v11.16b, #0x0
    movi v12.16b, #0x0
    movi v13.16b, #0x0
    movi v14.16b, #0x0
    movi v15.16b, #0x0
KAI_ASM_LABEL(label_47)  // Height 2: setup done
    mov x15, #0x0
KAI_ASM_LABEL(label_48)  // Height 2: String loop
    ldr x20, [x2, #0x10]
    ldr x21, [x2, #0x38]
    ldr w14, [x20, x15, LSL #0x2]
    tbz x3, #3, label_49
    ldr x20, [x0, x15, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x13, [x20, #0x0]
    ldr x12, [x20, #0x8]
    cbnz x15, label_50
    ldr x20, [x2, #0x30]
    add x13, x13, x20, LSL #2
    add x12, x12, x20, LSL #2
    b label_50
KAI_ASM_LABEL(label_49)  // Height 2: setup direct input
    mov x13, x0
    add x12, x13, x21, LSL #2
KAI_ASM_LABEL(label_50)  // Height 2: input setup done
    cmp x14, #0x4
    blt label_53
    ldr q0, [x13, #0x0]
    cmp x14, #0x8
    ldr q1, [x12, #0x0]
    ldr q6, [x17, #0x0]
    ldr q7, [x17, #0x10]
    blt label_52
KAI_ASM_LABEL(label_51)  // Height 2: Multiply loop: Main loop head
    fmla v8.4s, v6.4s, v0.s[0]
    ldr x20, [x17, #0x28]
    fmla v12.4s, v6.4s, v1.s[0]
    ldr d6, [x17, #0x20]
    fmla v9.4s, v7.4s, v0.s[0]
    ldr x21, [x17, #0x38]
    fmla v13.4s, v7.4s, v1.s[0]
    ldr d7, [x17, #0x30]
    mov v6.d[1], x20
    ldr x20, [x17, #0x48]
    add x13, x13, #0x10
    add x12, x12, #0x10
    mov v7.d[1], x21
    fmla v10.4s, v6.4s, v0.s[0]
    fmla v14.4s, v6.4s, v1.s[0]
    ldr d6, [x17, #0x40]
    fmla v11.4s, v7.4s, v0.s[0]
    ldr x21, [x17, #0x58]
    fmla v15.4s, v7.4s, v1.s[0]
    ldr d7, [x17, #0x50]
    mov v6.d[1], x20
    ldr x20, [x17, #0x68]
    ldr x23, [x13, #0x8]
    sub x14, x14, #0x4
    mov v7.d[1], x21
    fmla v8.4s, v6.4s, v0.s[1]
    fmla v12.4s, v6.4s, v1.s[1]
    ldr d6, [x17, #0x60]
    fmla v9.4s, v7.4s, v0.s[1]
    ldr x21, [x17, #0x78]
    fmla v13.4s, v7.4s, v1.s[1]
    ldr d7, [x17, #0x70]
    mov v6.d[1], x20
    ldr x20, [x17, #0x88]
    ldr x22, [x12, #0x8]
    cmp x14, #0x8
    mov v7.d[1], x21
    fmla v10.4s, v6.4s, v0.s[1]
    fmla v14.4s, v6.4s, v1.s[1]
    ldr d6, [x17, #0x80]
    fmla v11.4s, v7.4s, v0.s[1]
    ldr x21, [x17, #0x98]
    fmla v15.4s, v7.4s, v1.s[1]
    ldr d7, [x17, #0x90]
    mov v6.d[1], x20
    ldr x20, [x17, #0xa8]
    prfm pldl1keep, [x13, #0x80]
    mov v7.d[1], x21
    fmla v8.4s, v6.4s, v0.s[2]
    fmla v12.4s, v6.4s, v1.s[2]
    ldr d6, [x17, #0xa0]
    fmla v9.4s, v7.4s, v0.s[2]
    ldr x21, [x17, #0xb8]
    fmla v13.4s, v7.4s, v1.s[2]
    ldr d7, [x17, #0xb0]
    mov v6.d[1], x20
    ldr x20, [x17, #0xc8]
    prfm pldl1keep, [x12, #0x80]
    mov v7.d[1], x21
    fmla v10.4s, v6.4s, v0.s[2]
    fmla v14.4s, v6.4s, v1.s[2]
    ldr d6, [x17, #0xc0]
    fmla v11.4s, v7.4s, v0.s[2]
    ldr x21, [x17, #0xd8]
    fmla v15.4s, v7.4s, v1.s[2]
    ldr d7, [x17, #0xd0]
    mov v6.d[1], x20
    ldr x20, [x17, #0xe8]
    mov v7.d[1], x21
    fmla v8.4s, v6.4s, v0.s[3]
    fmla v12.4s, v6.4s, v1.s[3]
    ldr d6, [x17, #0xe0]
    fmla v9.4s, v7.4s, v0.s[3]
    ldr x21, [x17, #0xf8]
    fmla v13.4s, v7.4s, v1.s[3]
    ldr d7, [x17, #0xf0]
    mov v6.d[1], x20
    add x17, x17, #0x100
    ldr x20, [x17, #0x8]
    mov v7.d[1], x21
    fmla v10.4s, v6.4s, v0.s[3]
    fmla v14.4s, v6.4s, v1.s[3]
    ldr d6, [x17, #0x0]
    fmla v11.4s, v7.4s, v0.s[3]
    ldr d0, [x13, #0x0]
    fmla v15.4s, v7.4s, v1.s[3]
    ldr d1, [x12, #0x0]
    ldr d7, [x17, #0x10]
    mov v6.d[1], x20
    ldr x20, [x17, #0x18]
    mov v0.d[1], x23
    mov v1.d[1], x22
    mov v7.d[1], x20
    bge label_51
KAI_ASM_LABEL(label_52)  // Height 2: Multiply loop: Single iteration only
    fmla v8.4s, v6.4s, v0.s[0]
    add x13, x13, #0x10
    fmla v12.4s, v6.4s, v1.s[0]
    ldr q6, [x17, #0x20]
    fmla v9.4s, v7.4s, v0.s[0]
    add x12, x12, #0x10
    fmla v13.4s, v7.4s, v1.s[0]
    ldr q7, [x17, #0x30]
    fmla v10.4s, v6.4s, v0.s[0]
    sub x14, x14, #0x4
    fmla v14.4s, v6.4s, v1.s[0]
    ldr q6, [x17, #0x40]
    fmla v11.4s, v7.4s, v0.s[0]
    prfm pldl1keep, [x13, #0x80]
    fmla v15.4s, v7.4s, v1.s[0]
    ldr q7, [x17, #0x50]
    fmla v8.4s, v6.4s, v0.s[1]
    prfm pldl1keep, [x12, #0x80]
    fmla v12.4s, v6.4s, v1.s[1]
    ldr q6, [x17, #0x60]
    fmla v9.4s, v7.4s, v0.s[1]
    fmla v13.4s, v7.4s, v1.s[1]
    ldr q7, [x17, #0x70]
    fmla v10.4s, v6.4s, v0.s[1]
    fmla v14.4s, v6.4s, v1.s[1]
    ldr q6, [x17, #0x80]
    fmla v11.4s, v7.4s, v0.s[1]
    fmla v15.4s, v7.4s, v1.s[1]
    ldr q7, [x17, #0x90]
    fmla v8.4s, v6.4s, v0.s[2]
    fmla v12.4s, v6.4s, v1.s[2]
    ldr q6, [x17, #0xa0]
    fmla v9.4s, v7.4s, v0.s[2]
    fmla v13.4s, v7.4s, v1.s[2]
    ldr q7, [x17, #0xb0]
    fmla v10.4s, v6.4s, v0.s[2]
    fmla v14.4s, v6.4s, v1.s[2]
    ldr q6, [x17, #0xc0]
    fmla v11.4s, v7.4s, v0.s[2]
    fmla v15.4s, v7.4s, v1.s[2]
    ldr q7, [x17, #0xd0]
    fmla v8.4s, v6.4s, v0.s[3]
    fmla v12.4s, v6.4s, v1.s[3]
    ldr q6, [x17, #0xe0]
    fmla v9.4s, v7.4s, v0.s[3]
    fmla v13.4s, v7.4s, v1.s[3]
    ldr q7, [x17, #0xf0]
    fmla v10.4s, v6.4s, v0.s[3]
    add x17, x17, #0x100
    fmla v14.4s, v6.4s, v1.s[3]
    fmla v11.4s, v7.4s, v0.s[3]
    fmla v15.4s, v7.4s, v1.s[3]
KAI_ASM_LABEL(label_53)  // Height 2: Multiply loop: Main loop skip
    cbz x14, label_55
KAI_ASM_LABEL(label_54)  // Height 2: Multiply loop: Odd block loop
    ldr s0, [x13], #0x4
    sub x14, x14, #0x1
    ldr s1, [x12], #0x4
    ldr q6, [x17, #0x0]
    ldr q7, [x17, #0x10]
    fmla v8.4s, v6.4s, v0.s[0]
    fmla v12.4s, v6.4s, v1.s[0]
    ldr q6, [x17, #0x20]
    fmla v9.4s, v7.4s, v0.s[0]
    fmla v13.4s, v7.4s, v1.s[0]
    ldr q7, [x17, #0x30]
    fmla v10.4s, v6.4s, v0.s[0]
    add x17, x17, #0x40
    fmla v14.4s, v6.4s, v1.s[0]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v15.4s, v7.4s, v1.s[0]
    cbnz x14, label_54
KAI_ASM_LABEL(label_55)  // Height 2: Multiply loop: No odd multiplies
    ldr w20, [x2, #0x8]
    add x15, x15, #0x1
    cmp x15, x20
    bne label_48
    ldr x20, [x2, #0x28]
    prfm pstl1keep, [x16, #0x0]
    add x26, x16, x20, LSL #2
    prfm pstl1keep, [x26, #0x0]
    tbz x3, #1, label_56
    add x21, x2, #0x0
    add x20, x2, #0x4
    ld1r { v17.4s }, [x21]
    ld1r { v16.4s }, [x20]
    fmin v8.4s, v8.4s, v17.4s
    fmin v9.4s, v9.4s, v17.4s
    fmin v10.4s, v10.4s, v17.4s
    fmin v11.4s, v11.4s, v17.4s
    fmin v12.4s, v12.4s, v17.4s
    fmin v13.4s, v13.4s, v17.4s
    fmin v14.4s, v14.4s, v17.4s
    fmin v15.4s, v15.4s, v17.4s
    fmax v8.4s, v8.4s, v16.4s
    fmax v9.4s, v9.4s, v16.4s
    fmax v10.4s, v10.4s, v16.4s
    fmax v11.4s, v11.4s, v16.4s
    fmax v12.4s, v12.4s, v16.4s
    fmax v13.4s, v13.4s, v16.4s
    fmax v14.4s, v14.4s, v16.4s
    fmax v15.4s, v15.4s, v16.4s
KAI_ASM_LABEL(label_56)  // Height 2: No activation
    cmp x8, #0x10
    bge label_65
    tbz x8, #3, label_60
    st1 { v8.4s }, [x16], #0x10
    st1 { v9.4s }, [x16], #0x10
    st1 { v12.4s }, [x26], #0x10
    st1 { v13.4s }, [x26], #0x10
    tbz x8, #2, label_58
    st1 { v10.4s }, [x16], #0x10
    st1 { v14.4s }, [x26], #0x10
    tbz x8, #1, label_57
    str d11, [x16], #0x8
    str d15, [x26], #0x8
    tbz x8, #0, label_64
    st1 { v11.s }[2], [x16]
    st1 { v15.s }[2], [x26]
    b label_64
KAI_ASM_LABEL(label_57)  // Height 2: Partial direct writeback: partial_1_12
    tbz x8, #0, label_64
    str s11, [x16, #0x0]
    str s15, [x26, #0x0]
    b label_64
KAI_ASM_LABEL(label_58)  // Height 2: Partial direct writeback: partial_2_8
    tbz x8, #1, label_59
    str d10, [x16], #0x8
    str d14, [x26], #0x8
    tbz x8, #0, label_64
    st1 { v10.s }[2], [x16]
    st1 { v14.s }[2], [x26]
    b label_64
KAI_ASM_LABEL(label_59)  // Height 2: Partial direct writeback: partial_1_8
    tbz x8, #0, label_64
    str s10, [x16, #0x0]
    str s14, [x26, #0x0]
    b label_64
KAI_ASM_LABEL(label_60)  // Height 2: Partial direct writeback: partial_4_0
    tbz x8, #2, label_62
    st1 { v8.4s }, [x16], #0x10
    st1 { v12.4s }, [x26], #0x10
    tbz x8, #1, label_61
    str d9, [x16], #0x8
    str d13, [x26], #0x8
    tbz x8, #0, label_64
    st1 { v9.s }[2], [x16]
    st1 { v13.s }[2], [x26]
    b label_64
KAI_ASM_LABEL(label_61)  // Height 2: Partial direct writeback: partial_1_4
    tbz x8, #0, label_64
    str s9, [x16, #0x0]
    str s13, [x26, #0x0]
    b label_64
KAI_ASM_LABEL(label_62)  // Height 2: Partial direct writeback: partial_2_0
    tbz x8, #1, label_63
    str d8, [x16], #0x8
    str d12, [x26], #0x8
    tbz x8, #0, label_64
    st1 { v8.s }[2], [x16]
    st1 { v12.s }[2], [x26]
    b label_64
KAI_ASM_LABEL(label_63)  // Height 2: Partial direct writeback: partial_1_0
    str s8, [x16, #0x0]
    str s12, [x26, #0x0]
KAI_ASM_LABEL(label_64)  // Height 2: Partial direct writeback: Done
    b label_66
KAI_ASM_LABEL(label_65)  // Height 2: Full writeback
    str q8, [x16, #0x0]
    str q9, [x16, #0x10]
    str q10, [x16, #0x20]
    str q11, [x16, #0x30]
    add x16, x16, #0x40
    str q12, [x26, #0x0]
    str q13, [x26, #0x10]
    str q14, [x26, #0x20]
    str q15, [x26, #0x30]
KAI_ASM_LABEL(label_66)  // Height 2: Writeback done
    subs x8, x8, #0x10
    bgt label_35
    b label_200
KAI_ASM_LABEL(label_67)  // Height 3
    ldr x8, [x2, #0x18]
    ldr x17, [x2, #0x20]
    ldr x16, [x2, #0x40]
KAI_ASM_LABEL(label_68)  // Height 3: Column loop
    cbz x17, label_69
    ldr q8, [x17, #0x0]
    ldr q9, [x17, #0x10]
    ldr q10, [x17, #0x20]
    ldr q11, [x17, #0x30]
    mov v12.16b, v8.16b
    mov v13.16b, v9.16b
    add x17, x17, #0x40
    mov v14.16b, v10.16b
    mov v15.16b, v11.16b
    mov v16.16b, v8.16b
    mov v17.16b, v9.16b
    mov v18.16b, v10.16b
    mov v19.16b, v11.16b
    b label_80
KAI_ASM_LABEL(label_69)  // Height 3: no bias
    tbz x3, #0, label_79
    ldr x20, [x2, #0x28]
    cmp x8, #0x10
    add x26, x16, x20, LSL #2
    add x25, x26, x20, LSL #2
    bge label_78
    tbz x8, #3, label_73
    ld1 { v8.4s }, [x16], #0x10
    ld1 { v12.4s }, [x26], #0x10
    ld1 { v16.4s }, [x25], #0x10
    ld1 { v9.4s }, [x16], #0x10
    ld1 { v13.4s }, [x26], #0x10
    ld1 { v17.4s }, [x25], #0x10
    tbz x8, #2, label_71
    ld1 { v10.4s }, [x16], #0x10
    ld1 { v14.4s }, [x26], #0x10
    ld1 { v18.4s }, [x25], #0x10
    tbz x8, #1, label_70
    ldr d11, [x16], #0x8
    mov x20, #0x38
    ldr d15, [x26], #0x8
    ldr d19, [x25], #0x8
    tbz x8, #0, label_77
    ld1 { v11.s }[2], [x16]
    ld1 { v15.s }[2], [x26]
    ld1 { v19.s }[2], [x25]
    b label_77
KAI_ASM_LABEL(label_70)  // Height 3: Partial accumulate: partial_1_12
    mov x20, #0x30
    tbz x8, #0, label_77
    ldr s11, [x16, #0x0]
    ldr s15, [x26, #0x0]
    ldr s19, [x25, #0x0]
    b label_77
KAI_ASM_LABEL(label_71)  // Height 3: Partial accumulate: partial_2_8
    tbz x8, #1, label_72
    ldr d10, [x16], #0x8
    mov x20, #0x28
    ldr d14, [x26], #0x8
    ldr d18, [x25], #0x8
    tbz x8, #0, label_77
    ld1 { v10.s }[2], [x16]
    ld1 { v14.s }[2], [x26]
    ld1 { v18.s }[2], [x25]
    b label_77
KAI_ASM_LABEL(label_72)  // Height 3: Partial accumulate: partial_1_8
    mov x20, #0x20
    tbz x8, #0, label_77
    ldr s10, [x16, #0x0]
    ldr s14, [x26, #0x0]
    ldr s18, [x25, #0x0]
    b label_77
KAI_ASM_LABEL(label_73)  // Height 3: Partial accumulate: partial_4_0
    tbz x8, #2, label_75
    ld1 { v8.4s }, [x16], #0x10
    ld1 { v12.4s }, [x26], #0x10
    ld1 { v16.4s }, [x25], #0x10
    tbz x8, #1, label_74
    ldr d9, [x16], #0x8
    mov x20, #0x18
    ldr d13, [x26], #0x8
    ldr d17, [x25], #0x8
    tbz x8, #0, label_77
    ld1 { v9.s }[2], [x16]
    ld1 { v13.s }[2], [x26]
    ld1 { v17.s }[2], [x25]
    b label_77
KAI_ASM_LABEL(label_74)  // Height 3: Partial accumulate: partial_1_4
    mov x20, #0x10
    tbz x8, #0, label_77
    ldr s9, [x16, #0x0]
    ldr s13, [x26, #0x0]
    ldr s17, [x25, #0x0]
    b label_77
KAI_ASM_LABEL(label_75)  // Height 3: Partial accumulate: partial_2_0
    tbz x8, #1, label_76
    ldr d8, [x16], #0x8
    mov x20, #0x8
    ldr d12, [x26], #0x8
    ldr d16, [x25], #0x8
    tbz x8, #0, label_77
    ld1 { v8.s }[2], [x16]
    ld1 { v12.s }[2], [x26]
    ld1 { v16.s }[2], [x25]
    b label_77
KAI_ASM_LABEL(label_76)  // Height 3: Partial accumulate: partial_1_0
    ldr s8, [x16, #0x0]
    mov x20, #0x0
    ldr s12, [x26, #0x0]
    ldr s16, [x25, #0x0]
KAI_ASM_LABEL(label_77)  // Height 3: Partial accumulate: Done
    sub x16, x16, x20
    b label_80
KAI_ASM_LABEL(label_78)  // Height 3: full accumulate
    ldr q8, [x16, #0x0]
    ldr q9, [x16, #0x10]
    ldr q10, [x16, #0x20]
    ldr q11, [x16, #0x30]
    ldr q12, [x26, #0x0]
    ldr q13, [x26, #0x10]
    ldr q14, [x26, #0x20]
    ldr q15, [x26, #0x30]
    ldr q16, [x25, #0x0]
    ldr q17, [x25, #0x10]
    ldr q18, [x25, #0x20]
    ldr q19, [x25, #0x30]
    b label_80
KAI_ASM_LABEL(label_79)  // Height 3: no accumulate
    movi v8.16b, #0x0
    movi v9.16b, #0x0
    movi v10.16b, #0x0
    movi v11.16b, #0x0
    movi v12.16b, #0x0
    movi v13.16b, #0x0
    movi v14.16b, #0x0
    movi v15.16b, #0x0
    movi v16.16b, #0x0
    movi v17.16b, #0x0
    movi v18.16b, #0x0
    movi v19.16b, #0x0
KAI_ASM_LABEL(label_80)  // Height 3: setup done
    mov x15, #0x0
KAI_ASM_LABEL(label_81)  // Height 3: String loop
    ldr x20, [x2, #0x10]
    ldr x21, [x2, #0x38]
    ldr w14, [x20, x15, LSL #0x2]
    tbz x3, #3, label_82
    ldr x20, [x0, x15, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x13, [x20, #0x0]
    ldr x12, [x20, #0x8]
    ldr x11, [x20, #0x10]
    cbnz x15, label_83
    ldr x20, [x2, #0x30]
    add x13, x13, x20, LSL #2
    add x12, x12, x20, LSL #2
    add x11, x11, x20, LSL #2
    b label_83
KAI_ASM_LABEL(label_82)  // Height 3: setup direct input
    mov x13, x0
    add x12, x13, x21, LSL #2
    add x11, x12, x21, LSL #2
KAI_ASM_LABEL(label_83)  // Height 3: input setup done
    cmp x14, #0x4
    blt label_86
    ldr q0, [x13, #0x0]
    cmp x14, #0x8
    ldr q1, [x12, #0x0]
    ldr q2, [x11, #0x0]
    ldr q6, [x17, #0x0]
    ldr q7, [x17, #0x10]
    blt label_85
KAI_ASM_LABEL(label_84)  // Height 3: Multiply loop: Main loop head
    fmla v8.4s, v6.4s, v0.s[0]
    ldr x21, [x17, #0x28]
    fmla v12.4s, v6.4s, v1.s[0]
    ldr x20, [x17, #0x38]
    fmla v16.4s, v6.4s, v2.s[0]
    ldr d6, [x17, #0x20]
    fmla v9.4s, v7.4s, v0.s[0]
    add x13, x13, #0x10
    fmla v13.4s, v7.4s, v1.s[0]
    add x12, x12, #0x10
    fmla v17.4s, v7.4s, v2.s[0]
    ldr d7, [x17, #0x30]
    mov v6.d[1], x21
    ldr x21, [x17, #0x48]
    add x11, x11, #0x10
    ldr x24, [x13, #0x8]
    mov v7.d[1], x20
    fmla v10.4s, v6.4s, v0.s[0]
    fmla v14.4s, v6.4s, v1.s[0]
    ldr x20, [x17, #0x58]
    fmla v18.4s, v6.4s, v2.s[0]
    ldr d6, [x17, #0x40]
    fmla v11.4s, v7.4s, v0.s[0]
    ldr x23, [x12, #0x8]
    fmla v15.4s, v7.4s, v1.s[0]
    ldr x22, [x11, #0x8]
    fmla v19.4s, v7.4s, v2.s[0]
    ldr d7, [x17, #0x50]
    mov v6.d[1], x21
    ldr x21, [x17, #0x68]
    sub x14, x14, #0x4
    prfm pldl1keep, [x13, #0x80]
    mov v7.d[1], x20
    fmla v8.4s, v6.4s, v0.s[1]
    fmla v12.4s, v6.4s, v1.s[1]
    ldr x20, [x17, #0x78]
    fmla v16.4s, v6.4s, v2.s[1]
    ldr d6, [x17, #0x60]
    fmla v9.4s, v7.4s, v0.s[1]
    cmp x14, #0x8
    fmla v13.4s, v7.4s, v1.s[1]
    prfm pldl1keep, [x12, #0x80]
    fmla v17.4s, v7.4s, v2.s[1]
    ldr d7, [x17, #0x70]
    mov v6.d[1], x21
    ldr x21, [x17, #0x88]
    prfm pldl1keep, [x11, #0x80]
    mov v7.d[1], x20
    fmla v10.4s, v6.4s, v0.s[1]
    fmla v14.4s, v6.4s, v1.s[1]
    ldr x20, [x17, #0x98]
    fmla v18.4s, v6.4s, v2.s[1]
    ldr d6, [x17, #0x80]
    fmla v11.4s, v7.4s, v0.s[1]
    fmla v15.4s, v7.4s, v1.s[1]
    fmla v19.4s, v7.4s, v2.s[1]
    ldr d7, [x17, #0x90]
    mov v6.d[1], x21
    ldr x21, [x17, #0xa8]
    mov v7.d[1], x20
    fmla v8.4s, v6.4s, v0.s[2]
    fmla v12.4s, v6.4s, v1.s[2]
    ldr x20, [x17, #0xb8]
    fmla v16.4s, v6.4s, v2.s[2]
    ldr d6, [x17, #0xa0]
    fmla v9.4s, v7.4s, v0.s[2]
    fmla v13.4s, v7.4s, v1.s[2]
    fmla v17.4s, v7.4s, v2.s[2]
    ldr d7, [x17, #0xb0]
    mov v6.d[1], x21
    ldr x21, [x17, #0xc8]
    mov v7.d[1], x20
    fmla v10.4s, v6.4s, v0.s[2]
    fmla v14.4s, v6.4s, v1.s[2]
    ldr x20, [x17, #0xd8]
    fmla v18.4s, v6.4s, v2.s[2]
    ldr d6, [x17, #0xc0]
    fmla v11.4s, v7.4s, v0.s[2]
    fmla v15.4s, v7.4s, v1.s[2]
    fmla v19.4s, v7.4s, v2.s[2]
    ldr d7, [x17, #0xd0]
    mov v6.d[1], x21
    ldr x21, [x17, #0xe8]
    mov v7.d[1], x20
    fmla v8.4s, v6.4s, v0.s[3]
    fmla v12.4s, v6.4s, v1.s[3]
    ldr x20, [x17, #0xf8]
    fmla v16.4s, v6.4s, v2.s[3]
    ldr d6, [x17, #0xe0]
    fmla v9.4s, v7.4s, v0.s[3]
    fmla v13.4s, v7.4s, v1.s[3]
    fmla v17.4s, v7.4s, v2.s[3]
    ldr d7, [x17, #0xf0]
    mov v6.d[1], x21
    add x17, x17, #0x100
    ldr x21, [x17, #0x8]
    mov v7.d[1], x20
    fmla v10.4s, v6.4s, v0.s[3]
    fmla v14.4s, v6.4s, v1.s[3]
    ldr x20, [x17, #0x18]
    fmla v18.4s, v6.4s, v2.s[3]
    ldr d6, [x17, #0x0]
    fmla v11.4s, v7.4s, v0.s[3]
    ldr d0, [x13, #0x0]
    fmla v15.4s, v7.4s, v1.s[3]
    ldr d1, [x12, #0x0]
    fmla v19.4s, v7.4s, v2.s[3]
    ldr d2, [x11, #0x0]
    ldr d7, [x17, #0x10]
    mov v6.d[1], x21
    mov v0.d[1], x24
    mov v1.d[1], x23
    mov v2.d[1], x22
    mov v7.d[1], x20
    bge label_84
KAI_ASM_LABEL(label_85)  // Height 3: Multiply loop: Single iteration only
    fmla v8.4s, v6.4s, v0.s[0]
    add x13, x13, #0x10
    fmla v12.4s, v6.4s, v1.s[0]
    add x12, x12, #0x10
    fmla v16.4s, v6.4s, v2.s[0]
    ldr q6, [x17, #0x20]
    fmla v9.4s, v7.4s, v0.s[0]
    add x11, x11, #0x10
    fmla v13.4s, v7.4s, v1.s[0]
    sub x14, x14, #0x4
    fmla v17.4s, v7.4s, v2.s[0]
    ldr q7, [x17, #0x30]
    fmla v10.4s, v6.4s, v0.s[0]
    prfm pldl1keep, [x13, #0x80]
    fmla v14.4s, v6.4s, v1.s[0]
    prfm pldl1keep, [x12, #0x80]
    fmla v18.4s, v6.4s, v2.s[0]
    ldr q6, [x17, #0x40]
    fmla v11.4s, v7.4s, v0.s[0]
    prfm pldl1keep, [x11, #0x80]
    fmla v15.4s, v7.4s, v1.s[0]
    fmla v19.4s, v7.4s, v2.s[0]
    ldr q7, [x17, #0x50]
    fmla v8.4s, v6.4s, v0.s[1]
    fmla v12.4s, v6.4s, v1.s[1]
    fmla v16.4s, v6.4s, v2.s[1]
    ldr q6, [x17, #0x60]
    fmla v9.4s, v7.4s, v0.s[1]
    fmla v13.4s, v7.4s, v1.s[1]
    fmla v17.4s, v7.4s, v2.s[1]
    ldr q7, [x17, #0x70]
    fmla v10.4s, v6.4s, v0.s[1]
    fmla v14.4s, v6.4s, v1.s[1]
    fmla v18.4s, v6.4s, v2.s[1]
    ldr q6, [x17, #0x80]
    fmla v11.4s, v7.4s, v0.s[1]
    fmla v15.4s, v7.4s, v1.s[1]
    fmla v19.4s, v7.4s, v2.s[1]
    ldr q7, [x17, #0x90]
    fmla v8.4s, v6.4s, v0.s[2]
    fmla v12.4s, v6.4s, v1.s[2]
    fmla v16.4s, v6.4s, v2.s[2]
    ldr q6, [x17, #0xa0]
    fmla v9.4s, v7.4s, v0.s[2]
    fmla v13.4s, v7.4s, v1.s[2]
    fmla v17.4s, v7.4s, v2.s[2]
    ldr q7, [x17, #0xb0]
    fmla v10.4s, v6.4s, v0.s[2]
    fmla v14.4s, v6.4s, v1.s[2]
    fmla v18.4s, v6.4s, v2.s[2]
    ldr q6, [x17, #0xc0]
    fmla v11.4s, v7.4s, v0.s[2]
    fmla v15.4s, v7.4s, v1.s[2]
    fmla v19.4s, v7.4s, v2.s[2]
    ldr q7, [x17, #0xd0]
    fmla v8.4s, v6.4s, v0.s[3]
    fmla v12.4s, v6.4s, v1.s[3]
    fmla v16.4s, v6.4s, v2.s[3]
    ldr q6, [x17, #0xe0]
    fmla v9.4s, v7.4s, v0.s[3]
    fmla v13.4s, v7.4s, v1.s[3]
    fmla v17.4s, v7.4s, v2.s[3]
    ldr q7, [x17, #0xf0]
    fmla v10.4s, v6.4s, v0.s[3]
    add x17, x17, #0x100
    fmla v14.4s, v6.4s, v1.s[3]
    fmla v18.4s, v6.4s, v2.s[3]
    fmla v11.4s, v7.4s, v0.s[3]
    fmla v15.4s, v7.4s, v1.s[3]
    fmla v19.4s, v7.4s, v2.s[3]
KAI_ASM_LABEL(label_86)  // Height 3: Multiply loop: Main loop skip
    cbz x14, label_88
KAI_ASM_LABEL(label_87)  // Height 3: Multiply loop: Odd block loop
    ldr s0, [x13], #0x4
    sub x14, x14, #0x1
    ldr s1, [x12], #0x4
    ldr s2, [x11], #0x4
    ldr q6, [x17, #0x0]
    ldr q7, [x17, #0x10]
    fmla v8.4s, v6.4s, v0.s[0]
    fmla v12.4s, v6.4s, v1.s[0]
    fmla v16.4s, v6.4s, v2.s[0]
    ldr q6, [x17, #0x20]
    fmla v9.4s, v7.4s, v0.s[0]
    fmla v13.4s, v7.4s, v1.s[0]
    fmla v17.4s, v7.4s, v2.s[0]
    ldr q7, [x17, #0x30]
    fmla v10.4s, v6.4s, v0.s[0]
    add x17, x17, #0x40
    fmla v14.4s, v6.4s, v1.s[0]
    fmla v18.4s, v6.4s, v2.s[0]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v15.4s, v7.4s, v1.s[0]
    fmla v19.4s, v7.4s, v2.s[0]
    cbnz x14, label_87
KAI_ASM_LABEL(label_88)  // Height 3: Multiply loop: No odd multiplies
    ldr w20, [x2, #0x8]
    add x15, x15, #0x1
    cmp x15, x20
    bne label_81
    ldr x20, [x2, #0x28]
    prfm pstl1keep, [x16, #0x0]
    add x26, x16, x20, LSL #2
    add x25, x26, x20, LSL #2
    prfm pstl1keep, [x26, #0x0]
    prfm pstl1keep, [x25, #0x0]
    tbz x3, #1, label_89
    add x21, x2, #0x0
    add x20, x2, #0x4
    ld1r { v21.4s }, [x21]
    ld1r { v20.4s }, [x20]
    fmin v8.4s, v8.4s, v21.4s
    fmin v9.4s, v9.4s, v21.4s
    fmin v10.4s, v10.4s, v21.4s
    fmin v11.4s, v11.4s, v21.4s
    fmin v12.4s, v12.4s, v21.4s
    fmin v13.4s, v13.4s, v21.4s
    fmin v14.4s, v14.4s, v21.4s
    fmin v15.4s, v15.4s, v21.4s
    fmin v16.4s, v16.4s, v21.4s
    fmin v17.4s, v17.4s, v21.4s
    fmin v18.4s, v18.4s, v21.4s
    fmin v19.4s, v19.4s, v21.4s
    fmax v8.4s, v8.4s, v20.4s
    fmax v9.4s, v9.4s, v20.4s
    fmax v10.4s, v10.4s, v20.4s
    fmax v11.4s, v11.4s, v20.4s
    fmax v12.4s, v12.4s, v20.4s
    fmax v13.4s, v13.4s, v20.4s
    fmax v14.4s, v14.4s, v20.4s
    fmax v15.4s, v15.4s, v20.4s
    fmax v16.4s, v16.4s, v20.4s
    fmax v17.4s, v17.4s, v20.4s
    fmax v18.4s, v18.4s, v20.4s
    fmax v19.4s, v19.4s, v20.4s
KAI_ASM_LABEL(label_89)  // Height 3: No activation
    cmp x8, #0x10
    bge label_98
    tbz x8, #3, label_93
    st1 { v8.4s }, [x16], #0x10
    st1 { v9.4s }, [x16], #0x10
    st1 { v12.4s }, [x26], #0x10
    st1 { v13.4s }, [x26], #0x10
    st1 { v16.4s }, [x25], #0x10
    st1 { v17.4s }, [x25], #0x10
    tbz x8, #2, label_91
    st1 { v10.4s }, [x16], #0x10
    st1 { v14.4s }, [x26], #0x10
    st1 { v18.4s }, [x25], #0x10
    tbz x8, #1, label_90
    str d11, [x16], #0x8
    str d15, [x26], #0x8
    str d19, [x25], #0x8
    tbz x8, #0, label_97
    st1 { v11.s }[2], [x16]
    st1 { v15.s }[2], [x26]
    st1 { v19.s }[2], [x25]
    b label_97
KAI_ASM_LABEL(label_90)  // Height 3: Partial direct writeback: partial_1_12
    tbz x8, #0, label_97
    str s11, [x16, #0x0]
    str s15, [x26, #0x0]
    str s19, [x25, #0x0]
    b label_97
KAI_ASM_LABEL(label_91)  // Height 3: Partial direct writeback: partial_2_8
    tbz x8, #1, label_92
    str d10, [x16], #0x8
    str d14, [x26], #0x8
    str d18, [x25], #0x8
    tbz x8, #0, label_97
    st1 { v10.s }[2], [x16]
    st1 { v14.s }[2], [x26]
    st1 { v18.s }[2], [x25]
    b label_97
KAI_ASM_LABEL(label_92)  // Height 3: Partial direct writeback: partial_1_8
    tbz x8, #0, label_97
    str s10, [x16, #0x0]
    str s14, [x26, #0x0]
    str s18, [x25, #0x0]
    b label_97
KAI_ASM_LABEL(label_93)  // Height 3: Partial direct writeback: partial_4_0
    tbz x8, #2, label_95
    st1 { v8.4s }, [x16], #0x10
    st1 { v12.4s }, [x26], #0x10
    st1 { v16.4s }, [x25], #0x10
    tbz x8, #1, label_94
    str d9, [x16], #0x8
    str d13, [x26], #0x8
    str d17, [x25], #0x8
    tbz x8, #0, label_97
    st1 { v9.s }[2], [x16]
    st1 { v13.s }[2], [x26]
    st1 { v17.s }[2], [x25]
    b label_97
KAI_ASM_LABEL(label_94)  // Height 3: Partial direct writeback: partial_1_4
    tbz x8, #0, label_97
    str s9, [x16, #0x0]
    str s13, [x26, #0x0]
    str s17, [x25, #0x0]
    b label_97
KAI_ASM_LABEL(label_95)  // Height 3: Partial direct writeback: partial_2_0
    tbz x8, #1, label_96
    str d8, [x16], #0x8
    str d12, [x26], #0x8
    str d16, [x25], #0x8
    tbz x8, #0, label_97
    st1 { v8.s }[2], [x16]
    st1 { v12.s }[2], [x26]
    st1 { v16.s }[2], [x25]
    b label_97
KAI_ASM_LABEL(label_96)  // Height 3: Partial direct writeback: partial_1_0
    str s8, [x16, #0x0]
    str s12, [x26, #0x0]
    str s16, [x25, #0x0]
KAI_ASM_LABEL(label_97)  // Height 3: Partial direct writeback: Done
    b label_99
KAI_ASM_LABEL(label_98)  // Height 3: Full writeback
    str q8, [x16, #0x0]
    str q9, [x16, #0x10]
    str q10, [x16, #0x20]
    str q11, [x16, #0x30]
    add x16, x16, #0x40
    str q12, [x26, #0x0]
    str q13, [x26, #0x10]
    str q14, [x26, #0x20]
    str q15, [x26, #0x30]
    str q16, [x25, #0x0]
    str q17, [x25, #0x10]
    str q18, [x25, #0x20]
    str q19, [x25, #0x30]
KAI_ASM_LABEL(label_99)  // Height 3: Writeback done
    subs x8, x8, #0x10
    bgt label_68
    b label_200
KAI_ASM_LABEL(label_100)  // Height 4
    ldr x8, [x2, #0x18]
    ldr x17, [x2, #0x20]
    ldr x16, [x2, #0x40]
KAI_ASM_LABEL(label_101)  // Height 4: Column loop
    cbz x17, label_102
    ldr q8, [x17, #0x0]
    ldr q9, [x17, #0x10]
    ldr q10, [x17, #0x20]
    ldr q11, [x17, #0x30]
    mov v12.16b, v8.16b
    mov v13.16b, v9.16b
    add x17, x17, #0x40
    mov v14.16b, v10.16b
    mov v15.16b, v11.16b
    mov v16.16b, v8.16b
    mov v17.16b, v9.16b
    mov v18.16b, v10.16b
    mov v19.16b, v11.16b
    mov v20.16b, v8.16b
    mov v21.16b, v9.16b
    mov v22.16b, v10.16b
    mov v23.16b, v11.16b
    b label_113
KAI_ASM_LABEL(label_102)  // Height 4: no bias
    tbz x3, #0, label_112
    ldr x20, [x2, #0x28]
    cmp x8, #0x10
    add x26, x16, x20, LSL #2
    add x25, x26, x20, LSL #2
    add x24, x25, x20, LSL #2
    bge label_111
    tbz x8, #3, label_106
    ld1 { v8.4s }, [x16], #0x10
    ld1 { v12.4s }, [x26], #0x10
    ld1 { v16.4s }, [x25], #0x10
    ld1 { v20.4s }, [x24], #0x10
    ld1 { v9.4s }, [x16], #0x10
    ld1 { v13.4s }, [x26], #0x10
    ld1 { v17.4s }, [x25], #0x10
    ld1 { v21.4s }, [x24], #0x10
    tbz x8, #2, label_104
    ld1 { v10.4s }, [x16], #0x10
    ld1 { v14.4s }, [x26], #0x10
    ld1 { v18.4s }, [x25], #0x10
    ld1 { v22.4s }, [x24], #0x10
    tbz x8, #1, label_103
    ldr d11, [x16], #0x8
    mov x20, #0x38
    ldr d15, [x26], #0x8
    ldr d19, [x25], #0x8
    ldr d23, [x24], #0x8
    tbz x8, #0, label_110
    ld1 { v11.s }[2], [x16]
    ld1 { v15.s }[2], [x26]
    ld1 { v19.s }[2], [x25]
    ld1 { v23.s }[2], [x24]
    b label_110
KAI_ASM_LABEL(label_103)  // Height 4: Partial accumulate: partial_1_12
    mov x20, #0x30
    tbz x8, #0, label_110
    ldr s11, [x16, #0x0]
    ldr s15, [x26, #0x0]
    ldr s19, [x25, #0x0]
    ldr s23, [x24, #0x0]
    b label_110
KAI_ASM_LABEL(label_104)  // Height 4: Partial accumulate: partial_2_8
    tbz x8, #1, label_105
    ldr d10, [x16], #0x8
    mov x20, #0x28
    ldr d14, [x26], #0x8
    ldr d18, [x25], #0x8
    ldr d22, [x24], #0x8
    tbz x8, #0, label_110
    ld1 { v10.s }[2], [x16]
    ld1 { v14.s }[2], [x26]
    ld1 { v18.s }[2], [x25]
    ld1 { v22.s }[2], [x24]
    b label_110
KAI_ASM_LABEL(label_105)  // Height 4: Partial accumulate: partial_1_8
    mov x20, #0x20
    tbz x8, #0, label_110
    ldr s10, [x16, #0x0]
    ldr s14, [x26, #0x0]
    ldr s18, [x25, #0x0]
    ldr s22, [x24, #0x0]
    b label_110
KAI_ASM_LABEL(label_106)  // Height 4: Partial accumulate: partial_4_0
    tbz x8, #2, label_108
    ld1 { v8.4s }, [x16], #0x10
    ld1 { v12.4s }, [x26], #0x10
    ld1 { v16.4s }, [x25], #0x10
    ld1 { v20.4s }, [x24], #0x10
    tbz x8, #1, label_107
    ldr d9, [x16], #0x8
    mov x20, #0x18
    ldr d13, [x26], #0x8
    ldr d17, [x25], #0x8
    ldr d21, [x24], #0x8
    tbz x8, #0, label_110
    ld1 { v9.s }[2], [x16]
    ld1 { v13.s }[2], [x26]
    ld1 { v17.s }[2], [x25]
    ld1 { v21.s }[2], [x24]
    b label_110
KAI_ASM_LABEL(label_107)  // Height 4: Partial accumulate: partial_1_4
    mov x20, #0x10
    tbz x8, #0, label_110
    ldr s9, [x16, #0x0]
    ldr s13, [x26, #0x0]
    ldr s17, [x25, #0x0]
    ldr s21, [x24, #0x0]
    b label_110
KAI_ASM_LABEL(label_108)  // Height 4: Partial accumulate: partial_2_0
    tbz x8, #1, label_109
    ldr d8, [x16], #0x8
    mov x20, #0x8
    ldr d12, [x26], #0x8
    ldr d16, [x25], #0x8
    ldr d20, [x24], #0x8
    tbz x8, #0, label_110
    ld1 { v8.s }[2], [x16]
    ld1 { v12.s }[2], [x26]
    ld1 { v16.s }[2], [x25]
    ld1 { v20.s }[2], [x24]
    b label_110
KAI_ASM_LABEL(label_109)  // Height 4: Partial accumulate: partial_1_0
    ldr s8, [x16, #0x0]
    mov x20, #0x0
    ldr s12, [x26, #0x0]
    ldr s16, [x25, #0x0]
    ldr s20, [x24, #0x0]
KAI_ASM_LABEL(label_110)  // Height 4: Partial accumulate: Done
    sub x16, x16, x20
    b label_113
KAI_ASM_LABEL(label_111)  // Height 4: full accumulate
    ldr q8, [x16, #0x0]
    ldr q9, [x16, #0x10]
    ldr q10, [x16, #0x20]
    ldr q11, [x16, #0x30]
    ldr q12, [x26, #0x0]
    ldr q13, [x26, #0x10]
    ldr q14, [x26, #0x20]
    ldr q15, [x26, #0x30]
    ldr q16, [x25, #0x0]
    ldr q17, [x25, #0x10]
    ldr q18, [x25, #0x20]
    ldr q19, [x25, #0x30]
    ldr q20, [x24, #0x0]
    ldr q21, [x24, #0x10]
    ldr q22, [x24, #0x20]
    ldr q23, [x24, #0x30]
    b label_113
KAI_ASM_LABEL(label_112)  // Height 4: no accumulate
    movi v8.16b, #0x0
    movi v9.16b, #0x0
    movi v10.16b, #0x0
    movi v11.16b, #0x0
    movi v12.16b, #0x0
    movi v13.16b, #0x0
    movi v14.16b, #0x0
    movi v15.16b, #0x0
    movi v16.16b, #0x0
    movi v17.16b, #0x0
    movi v18.16b, #0x0
    movi v19.16b, #0x0
    movi v20.16b, #0x0
    movi v21.16b, #0x0
    movi v22.16b, #0x0
    movi v23.16b, #0x0
KAI_ASM_LABEL(label_113)  // Height 4: setup done
    mov x15, #0x0
KAI_ASM_LABEL(label_114)  // Height 4: String loop
    ldr x20, [x2, #0x10]
    ldr x21, [x2, #0x38]
    ldr w14, [x20, x15, LSL #0x2]
    tbz x3, #3, label_115
    ldr x20, [x0, x15, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x13, [x20, #0x0]
    ldr x12, [x20, #0x8]
    ldr x11, [x20, #0x10]
    ldr x10, [x20, #0x18]
    cbnz x15, label_116
    ldr x20, [x2, #0x30]
    add x13, x13, x20, LSL #2
    add x12, x12, x20, LSL #2
    add x11, x11, x20, LSL #2
    add x10, x10, x20, LSL #2
    b label_116
KAI_ASM_LABEL(label_115)  // Height 4: setup direct input
    mov x13, x0
    add x12, x13, x21, LSL #2
    add x11, x12, x21, LSL #2
    add x10, x11, x21, LSL #2
KAI_ASM_LABEL(label_116)  // Height 4: input setup done
    cmp x14, #0x4
    blt label_119
    ldr q0, [x13, #0x0]
    cmp x14, #0x8
    ldr q1, [x12, #0x0]
    ldr q2, [x11, #0x0]
    ldr q3, [x10, #0x0]
    ldr q6, [x17, #0x0]
    ldr q7, [x17, #0x10]
    blt label_118
KAI_ASM_LABEL(label_117)  // Height 4: Multiply loop: Main loop head
    fmla v8.4s, v6.4s, v0.s[0]
    ldr x20, [x17, #0x28]
    fmla v12.4s, v6.4s, v1.s[0]
    ldr x21, [x17, #0x38]
    fmla v16.4s, v6.4s, v2.s[0]
    add x13, x13, #0x10
    fmla v20.4s, v6.4s, v3.s[0]
    ldr d6, [x17, #0x20]
    fmla v9.4s, v7.4s, v0.s[0]
    add x12, x12, #0x10
    fmla v13.4s, v7.4s, v1.s[0]
    add x11, x11, #0x10
    fmla v17.4s, v7.4s, v2.s[0]
    mov v6.d[1], x20
    fmla v21.4s, v7.4s, v3.s[0]
    ldr d7, [x17, #0x30]
    fmla v10.4s, v6.4s, v0.s[0]
    ldr x20, [x17, #0x48]
    fmla v14.4s, v6.4s, v1.s[0]
    add x10, x10, #0x10
    mov v7.d[1], x21
    fmla v18.4s, v6.4s, v2.s[0]
    fmla v22.4s, v6.4s, v3.s[0]
    ldr d6, [x17, #0x40]
    fmla v11.4s, v7.4s, v0.s[0]
    ldr x21, [x17, #0x58]
    fmla v15.4s, v7.4s, v1.s[0]
    ldr x25, [x13, #0x8]
    fmla v19.4s, v7.4s, v2.s[0]
    mov v6.d[1], x20
    fmla v23.4s, v7.4s, v3.s[0]
    ldr d7, [x17, #0x50]
    fmla v8.4s, v6.4s, v0.s[1]
    ldr x20, [x17, #0x68]
    fmla v12.4s, v6.4s, v1.s[1]
    ldr x24, [x12, #0x8]
    mov v7.d[1], x21
    fmla v16.4s, v6.4s, v2.s[1]
    fmla v20.4s, v6.4s, v3.s[1]
    ldr d6, [x17, #0x60]
    fmla v9.4s, v7.4s, v0.s[1]
    ldr x21, [x17, #0x78]
    fmla v13.4s, v7.4s, v1.s[1]
    ldr x23, [x11, #0x8]
    fmla v17.4s, v7.4s, v2.s[1]
    mov v6.d[1], x20
    fmla v21.4s, v7.4s, v3.s[1]
    ldr d7, [x17, #0x70]
    fmla v10.4s, v6.4s, v0.s[1]
    ldr x20, [x17, #0x88]
    fmla v14.4s, v6.4s, v1.s[1]
    ldr x22, [x10, #0x8]
    mov v7.d[1], x21
    fmla v18.4s, v6.4s, v2.s[1]
    fmla v22.4s, v6.4s, v3.s[1]
    ldr d6, [x17, #0x80]
    fmla v11.4s, v7.4s, v0.s[1]
    ldr x21, [x17, #0x98]
    fmla v15.4s, v7.4s, v1.s[1]
    sub x14, x14, #0x4
    fmla v19.4s, v7.4s, v2.s[1]
    mov v6.d[1], x20
    fmla v23.4s, v7.4s, v3.s[1]
    ldr d7, [x17, #0x90]
    fmla v8.4s, v6.4s, v0.s[2]
    ldr x20, [x17, #0xa8]
    fmla v12.4s, v6.4s, v1.s[2]
    cmp x14, #0x8
    mov v7.d[1], x21
    fmla v16.4s, v6.4s, v2.s[2]
    fmla v20.4s, v6.4s, v3.s[2]
    ldr d6, [x17, #0xa0]
    fmla v9.4s, v7.4s, v0.s[2]
    ldr x21, [x17, #0xb8]
    fmla v13.4s, v7.4s, v1.s[2]
    prfm pldl1keep, [x13, #0x80]
    fmla v17.4s, v7.4s, v2.s[2]
    mov v6.d[1], x20
    fmla v21.4s, v7.4s, v3.s[2]
    ldr d7, [x17, #0xb0]
    fmla v10.4s, v6.4s, v0.s[2]
    ldr x20, [x17, #0xc8]
    fmla v14.4s, v6.4s, v1.s[2]
    prfm pldl1keep, [x12, #0x80]
    mov v7.d[1], x21
    fmla v18.4s, v6.4s, v2.s[2]
    fmla v22.4s, v6.4s, v3.s[2]
    ldr d6, [x17, #0xc0]
    fmla v11.4s, v7.4s, v0.s[2]
    ldr x21, [x17, #0xd8]
    fmla v15.4s, v7.4s, v1.s[2]
    prfm pldl1keep, [x11, #0x80]
    fmla v19.4s, v7.4s, v2.s[2]
    mov v6.d[1], x20
    fmla v23.4s, v7.4s, v3.s[2]
    ldr d7, [x17, #0xd0]
    fmla v8.4s, v6.4s, v0.s[3]
    ldr x20, [x17, #0xe8]
    fmla v12.4s, v6.4s, v1.s[3]
    prfm pldl1keep, [x10, #0x80]
    mov v7.d[1], x21
    fmla v16.4s, v6.4s, v2.s[3]
    fmla v20.4s, v6.4s, v3.s[3]
    ldr d6, [x17, #0xe0]
    fmla v9.4s, v7.4s, v0.s[3]
    ldr x21, [x17, #0xf8]
    fmla v13.4s, v7.4s, v1.s[3]
    fmla v17.4s, v7.4s, v2.s[3]
    mov v6.d[1], x20
    fmla v21.4s, v7.4s, v3.s[3]
    ldr d7, [x17, #0xf0]
    add x17, x17, #0x100
    fmla v10.4s, v6.4s, v0.s[3]
    fmla v14.4s, v6.4s, v1.s[3]
    ldr x20, [x17, #0x8]
    mov v7.d[1], x21
    fmla v18.4s, v6.4s, v2.s[3]
    fmla v22.4s, v6.4s, v3.s[3]
    ldr d6, [x17, #0x0]
    fmla v11.4s, v7.4s, v0.s[3]
    ldr d0, [x13, #0x0]
    fmla v15.4s, v7.4s, v1.s[3]
    ldr d1, [x12, #0x0]
    fmla v19.4s, v7.4s, v2.s[3]
    ldr d2, [x11, #0x0]
    fmla v23.4s, v7.4s, v3.s[3]
    ldr d3, [x10, #0x0]
    ldr d7, [x17, #0x10]
    mov v6.d[1], x20
    ldr x20, [x17, #0x18]
    mov v0.d[1], x25
    mov v1.d[1], x24
    mov v2.d[1], x23
    mov v3.d[1], x22
    mov v7.d[1], x20
    bge label_117
KAI_ASM_LABEL(label_118)  // Height 4: Multiply loop: Single iteration only
    fmla v8.4s, v6.4s, v0.s[0]
    add x13, x13, #0x10
    fmla v12.4s, v6.4s, v1.s[0]
    add x12, x12, #0x10
    fmla v16.4s, v6.4s, v2.s[0]
    add x11, x11, #0x10
    fmla v20.4s, v6.4s, v3.s[0]
    ldr q6, [x17, #0x20]
    fmla v9.4s, v7.4s, v0.s[0]
    add x10, x10, #0x10
    fmla v13.4s, v7.4s, v1.s[0]
    sub x14, x14, #0x4
    fmla v17.4s, v7.4s, v2.s[0]
    prfm pldl1keep, [x13, #0x80]
    fmla v21.4s, v7.4s, v3.s[0]
    ldr q7, [x17, #0x30]
    fmla v10.4s, v6.4s, v0.s[0]
    prfm pldl1keep, [x12, #0x80]
    fmla v14.4s, v6.4s, v1.s[0]
    prfm pldl1keep, [x11, #0x80]
    fmla v18.4s, v6.4s, v2.s[0]
    prfm pldl1keep, [x10, #0x80]
    fmla v22.4s, v6.4s, v3.s[0]
    ldr q6, [x17, #0x40]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v15.4s, v7.4s, v1.s[0]
    fmla v19.4s, v7.4s, v2.s[0]
    fmla v23.4s, v7.4s, v3.s[0]
    ldr q7, [x17, #0x50]
    fmla v8.4s, v6.4s, v0.s[1]
    fmla v12.4s, v6.4s, v1.s[1]
    fmla v16.4s, v6.4s, v2.s[1]
    fmla v20.4s, v6.4s, v3.s[1]
    ldr q6, [x17, #0x60]
    fmla v9.4s, v7.4s, v0.s[1]
    fmla v13.4s, v7.4s, v1.s[1]
    fmla v17.4s, v7.4s, v2.s[1]
    fmla v21.4s, v7.4s, v3.s[1]
    ldr q7, [x17, #0x70]
    fmla v10.4s, v6.4s, v0.s[1]
    fmla v14.4s, v6.4s, v1.s[1]
    fmla v18.4s, v6.4s, v2.s[1]
    fmla v22.4s, v6.4s, v3.s[1]
    ldr q6, [x17, #0x80]
    fmla v11.4s, v7.4s, v0.s[1]
    fmla v15.4s, v7.4s, v1.s[1]
    fmla v19.4s, v7.4s, v2.s[1]
    fmla v23.4s, v7.4s, v3.s[1]
    ldr q7, [x17, #0x90]
    fmla v8.4s, v6.4s, v0.s[2]
    fmla v12.4s, v6.4s, v1.s[2]
    fmla v16.4s, v6.4s, v2.s[2]
    fmla v20.4s, v6.4s, v3.s[2]
    ldr q6, [x17, #0xa0]
    fmla v9.4s, v7.4s, v0.s[2]
    fmla v13.4s, v7.4s, v1.s[2]
    fmla v17.4s, v7.4s, v2.s[2]
    fmla v21.4s, v7.4s, v3.s[2]
    ldr q7, [x17, #0xb0]
    fmla v10.4s, v6.4s, v0.s[2]
    fmla v14.4s, v6.4s, v1.s[2]
    fmla v18.4s, v6.4s, v2.s[2]
    fmla v22.4s, v6.4s, v3.s[2]
    ldr q6, [x17, #0xc0]
    fmla v11.4s, v7.4s, v0.s[2]
    fmla v15.4s, v7.4s, v1.s[2]
    fmla v19.4s, v7.4s, v2.s[2]
    fmla v23.4s, v7.4s, v3.s[2]
    ldr q7, [x17, #0xd0]
    fmla v8.4s, v6.4s, v0.s[3]
    fmla v12.4s, v6.4s, v1.s[3]
    fmla v16.4s, v6.4s, v2.s[3]
    fmla v20.4s, v6.4s, v3.s[3]
    ldr q6, [x17, #0xe0]
    fmla v9.4s, v7.4s, v0.s[3]
    fmla v13.4s, v7.4s, v1.s[3]
    fmla v17.4s, v7.4s, v2.s[3]
    fmla v21.4s, v7.4s, v3.s[3]
    ldr q7, [x17, #0xf0]
    fmla v10.4s, v6.4s, v0.s[3]
    add x17, x17, #0x100
    fmla v14.4s, v6.4s, v1.s[3]
    fmla v18.4s, v6.4s, v2.s[3]
    fmla v22.4s, v6.4s, v3.s[3]
    fmla v11.4s, v7.4s, v0.s[3]
    fmla v15.4s, v7.4s, v1.s[3]
    fmla v19.4s, v7.4s, v2.s[3]
    fmla v23.4s, v7.4s, v3.s[3]
KAI_ASM_LABEL(label_119)  // Height 4: Multiply loop: Main loop skip
    cbz x14, label_121
KAI_ASM_LABEL(label_120)  // Height 4: Multiply loop: Odd block loop
    ldr s0, [x13], #0x4
    sub x14, x14, #0x1
    ldr s1, [x12], #0x4
    ldr s2, [x11], #0x4
    ldr s3, [x10], #0x4
    ldr q6, [x17, #0x0]
    ldr q7, [x17, #0x10]
    fmla v8.4s, v6.4s, v0.s[0]
    fmla v12.4s, v6.4s, v1.s[0]
    fmla v16.4s, v6.4s, v2.s[0]
    fmla v20.4s, v6.4s, v3.s[0]
    ldr q6, [x17, #0x20]
    fmla v9.4s, v7.4s, v0.s[0]
    fmla v13.4s, v7.4s, v1.s[0]
    fmla v17.4s, v7.4s, v2.s[0]
    fmla v21.4s, v7.4s, v3.s[0]
    ldr q7, [x17, #0x30]
    fmla v10.4s, v6.4s, v0.s[0]
    add x17, x17, #0x40
    fmla v14.4s, v6.4s, v1.s[0]
    fmla v18.4s, v6.4s, v2.s[0]
    fmla v22.4s, v6.4s, v3.s[0]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v15.4s, v7.4s, v1.s[0]
    fmla v19.4s, v7.4s, v2.s[0]
    fmla v23.4s, v7.4s, v3.s[0]
    cbnz x14, label_120
KAI_ASM_LABEL(label_121)  // Height 4: Multiply loop: No odd multiplies
    ldr w20, [x2, #0x8]
    add x15, x15, #0x1
    cmp x15, x20
    bne label_114
    ldr x20, [x2, #0x28]
    prfm pstl1keep, [x16, #0x0]
    add x26, x16, x20, LSL #2
    add x25, x26, x20, LSL #2
    prfm pstl1keep, [x26, #0x0]
    add x24, x25, x20, LSL #2
    prfm pstl1keep, [x25, #0x0]
    prfm pstl1keep, [x24, #0x0]
    tbz x3, #1, label_122
    add x21, x2, #0x0
    add x20, x2, #0x4
    ld1r { v25.4s }, [x21]
    ld1r { v24.4s }, [x20]
    fmin v8.4s, v8.4s, v25.4s
    fmin v9.4s, v9.4s, v25.4s
    fmin v10.4s, v10.4s, v25.4s
    fmin v11.4s, v11.4s, v25.4s
    fmin v12.4s, v12.4s, v25.4s
    fmin v13.4s, v13.4s, v25.4s
    fmin v14.4s, v14.4s, v25.4s
    fmin v15.4s, v15.4s, v25.4s
    fmin v16.4s, v16.4s, v25.4s
    fmin v17.4s, v17.4s, v25.4s
    fmin v18.4s, v18.4s, v25.4s
    fmin v19.4s, v19.4s, v25.4s
    fmin v20.4s, v20.4s, v25.4s
    fmin v21.4s, v21.4s, v25.4s
    fmin v22.4s, v22.4s, v25.4s
    fmin v23.4s, v23.4s, v25.4s
    fmax v8.4s, v8.4s, v24.4s
    fmax v9.4s, v9.4s, v24.4s
    fmax v10.4s, v10.4s, v24.4s
    fmax v11.4s, v11.4s, v24.4s
    fmax v12.4s, v12.4s, v24.4s
    fmax v13.4s, v13.4s, v24.4s
    fmax v14.4s, v14.4s, v24.4s
    fmax v15.4s, v15.4s, v24.4s
    fmax v16.4s, v16.4s, v24.4s
    fmax v17.4s, v17.4s, v24.4s
    fmax v18.4s, v18.4s, v24.4s
    fmax v19.4s, v19.4s, v24.4s
    fmax v20.4s, v20.4s, v24.4s
    fmax v21.4s, v21.4s, v24.4s
    fmax v22.4s, v22.4s, v24.4s
    fmax v23.4s, v23.4s, v24.4s
KAI_ASM_LABEL(label_122)  // Height 4: No activation
    cmp x8, #0x10
    bge label_131
    tbz x8, #3, label_126
    st1 { v8.4s }, [x16], #0x10
    st1 { v9.4s }, [x16], #0x10
    st1 { v12.4s }, [x26], #0x10
    st1 { v13.4s }, [x26], #0x10
    st1 { v16.4s }, [x25], #0x10
    st1 { v17.4s }, [x25], #0x10
    st1 { v20.4s }, [x24], #0x10
    st1 { v21.4s }, [x24], #0x10
    tbz x8, #2, label_124
    st1 { v10.4s }, [x16], #0x10
    st1 { v14.4s }, [x26], #0x10
    st1 { v18.4s }, [x25], #0x10
    st1 { v22.4s }, [x24], #0x10
    tbz x8, #1, label_123
    str d11, [x16], #0x8
    str d15, [x26], #0x8
    str d19, [x25], #0x8
    str d23, [x24], #0x8
    tbz x8, #0, label_130
    st1 { v11.s }[2], [x16]
    st1 { v15.s }[2], [x26]
    st1 { v19.s }[2], [x25]
    st1 { v23.s }[2], [x24]
    b label_130
KAI_ASM_LABEL(label_123)  // Height 4: Partial direct writeback: partial_1_12
    tbz x8, #0, label_130
    str s11, [x16, #0x0]
    str s15, [x26, #0x0]
    str s19, [x25, #0x0]
    str s23, [x24, #0x0]
    b label_130
KAI_ASM_LABEL(label_124)  // Height 4: Partial direct writeback: partial_2_8
    tbz x8, #1, label_125
    str d10, [x16], #0x8
    str d14, [x26], #0x8
    str d18, [x25], #0x8
    str d22, [x24], #0x8
    tbz x8, #0, label_130
    st1 { v10.s }[2], [x16]
    st1 { v14.s }[2], [x26]
    st1 { v18.s }[2], [x25]
    st1 { v22.s }[2], [x24]
    b label_130
KAI_ASM_LABEL(label_125)  // Height 4: Partial direct writeback: partial_1_8
    tbz x8, #0, label_130
    str s10, [x16, #0x0]
    str s14, [x26, #0x0]
    str s18, [x25, #0x0]
    str s22, [x24, #0x0]
    b label_130
KAI_ASM_LABEL(label_126)  // Height 4: Partial direct writeback: partial_4_0
    tbz x8, #2, label_128
    st1 { v8.4s }, [x16], #0x10
    st1 { v12.4s }, [x26], #0x10
    st1 { v16.4s }, [x25], #0x10
    st1 { v20.4s }, [x24], #0x10
    tbz x8, #1, label_127
    str d9, [x16], #0x8
    str d13, [x26], #0x8
    str d17, [x25], #0x8
    str d21, [x24], #0x8
    tbz x8, #0, label_130
    st1 { v9.s }[2], [x16]
    st1 { v13.s }[2], [x26]
    st1 { v17.s }[2], [x25]
    st1 { v21.s }[2], [x24]
    b label_130
KAI_ASM_LABEL(label_127)  // Height 4: Partial direct writeback: partial_1_4
    tbz x8, #0, label_130
    str s9, [x16, #0x0]
    str s13, [x26, #0x0]
    str s17, [x25, #0x0]
    str s21, [x24, #0x0]
    b label_130
KAI_ASM_LABEL(label_128)  // Height 4: Partial direct writeback: partial_2_0
    tbz x8, #1, label_129
    str d8, [x16], #0x8
    str d12, [x26], #0x8
    str d16, [x25], #0x8
    str d20, [x24], #0x8
    tbz x8, #0, label_130
    st1 { v8.s }[2], [x16]
    st1 { v12.s }[2], [x26]
    st1 { v16.s }[2], [x25]
    st1 { v20.s }[2], [x24]
    b label_130
KAI_ASM_LABEL(label_129)  // Height 4: Partial direct writeback: partial_1_0
    str s8, [x16, #0x0]
    str s12, [x26, #0x0]
    str s16, [x25, #0x0]
    str s20, [x24, #0x0]
KAI_ASM_LABEL(label_130)  // Height 4: Partial direct writeback: Done
    b label_132
KAI_ASM_LABEL(label_131)  // Height 4: Full writeback
    str q8, [x16, #0x0]
    str q9, [x16, #0x10]
    str q10, [x16, #0x20]
    str q11, [x16, #0x30]
    add x16, x16, #0x40
    str q12, [x26, #0x0]
    str q13, [x26, #0x10]
    str q14, [x26, #0x20]
    str q15, [x26, #0x30]
    str q16, [x25, #0x0]
    str q17, [x25, #0x10]
    str q18, [x25, #0x20]
    str q19, [x25, #0x30]
    str q20, [x24, #0x0]
    str q21, [x24, #0x10]
    str q22, [x24, #0x20]
    str q23, [x24, #0x30]
KAI_ASM_LABEL(label_132)  // Height 4: Writeback done
    subs x8, x8, #0x10
    bgt label_101
    b label_200
KAI_ASM_LABEL(label_133)  // Height 5
    ldr x8, [x2, #0x18]
    ldr x17, [x2, #0x20]
    ldr x16, [x2, #0x40]
KAI_ASM_LABEL(label_134)  // Height 5: Column loop
    cbz x17, label_135
    ldr q8, [x17, #0x0]
    ldr q9, [x17, #0x10]
    ldr q10, [x17, #0x20]
    ldr q11, [x17, #0x30]
    mov v12.16b, v8.16b
    mov v13.16b, v9.16b
    add x17, x17, #0x40
    mov v14.16b, v10.16b
    mov v15.16b, v11.16b
    mov v16.16b, v8.16b
    mov v17.16b, v9.16b
    mov v18.16b, v10.16b
    mov v19.16b, v11.16b
    mov v20.16b, v8.16b
    mov v21.16b, v9.16b
    mov v22.16b, v10.16b
    mov v23.16b, v11.16b
    mov v24.16b, v8.16b
    mov v25.16b, v9.16b
    mov v26.16b, v10.16b
    mov v27.16b, v11.16b
    b label_146
KAI_ASM_LABEL(label_135)  // Height 5: no bias
    tbz x3, #0, label_145
    ldr x20, [x2, #0x28]
    cmp x8, #0x10
    add x26, x16, x20, LSL #2
    add x25, x26, x20, LSL #2
    add x24, x25, x20, LSL #2
    add x23, x24, x20, LSL #2
    bge label_144
    tbz x8, #3, label_139
    ld1 { v8.4s }, [x16], #0x10
    ld1 { v12.4s }, [x26], #0x10
    ld1 { v16.4s }, [x25], #0x10
    ld1 { v20.4s }, [x24], #0x10
    ld1 { v24.4s }, [x23], #0x10
    ld1 { v9.4s }, [x16], #0x10
    ld1 { v13.4s }, [x26], #0x10
    ld1 { v17.4s }, [x25], #0x10
    ld1 { v21.4s }, [x24], #0x10
    ld1 { v25.4s }, [x23], #0x10
    tbz x8, #2, label_137
    ld1 { v10.4s }, [x16], #0x10
    ld1 { v14.4s }, [x26], #0x10
    ld1 { v18.4s }, [x25], #0x10
    ld1 { v22.4s }, [x24], #0x10
    ld1 { v26.4s }, [x23], #0x10
    tbz x8, #1, label_136
    ldr d11, [x16], #0x8
    mov x20, #0x38
    ldr d15, [x26], #0x8
    ldr d19, [x25], #0x8
    ldr d23, [x24], #0x8
    ldr d27, [x23], #0x8
    tbz x8, #0, label_143
    ld1 { v11.s }[2], [x16]
    ld1 { v15.s }[2], [x26]
    ld1 { v19.s }[2], [x25]
    ld1 { v23.s }[2], [x24]
    ld1 { v27.s }[2], [x23]
    b label_143
KAI_ASM_LABEL(label_136)  // Height 5: Partial accumulate: partial_1_12
    mov x20, #0x30
    tbz x8, #0, label_143
    ldr s11, [x16, #0x0]
    ldr s15, [x26, #0x0]
    ldr s19, [x25, #0x0]
    ldr s23, [x24, #0x0]
    ldr s27, [x23, #0x0]
    b label_143
KAI_ASM_LABEL(label_137)  // Height 5: Partial accumulate: partial_2_8
    tbz x8, #1, label_138
    ldr d10, [x16], #0x8
    mov x20, #0x28
    ldr d14, [x26], #0x8
    ldr d18, [x25], #0x8
    ldr d22, [x24], #0x8
    ldr d26, [x23], #0x8
    tbz x8, #0, label_143
    ld1 { v10.s }[2], [x16]
    ld1 { v14.s }[2], [x26]
    ld1 { v18.s }[2], [x25]
    ld1 { v22.s }[2], [x24]
    ld1 { v26.s }[2], [x23]
    b label_143
KAI_ASM_LABEL(label_138)  // Height 5: Partial accumulate: partial_1_8
    mov x20, #0x20
    tbz x8, #0, label_143
    ldr s10, [x16, #0x0]
    ldr s14, [x26, #0x0]
    ldr s18, [x25, #0x0]
    ldr s22, [x24, #0x0]
    ldr s26, [x23, #0x0]
    b label_143
KAI_ASM_LABEL(label_139)  // Height 5: Partial accumulate: partial_4_0
    tbz x8, #2, label_141
    ld1 { v8.4s }, [x16], #0x10
    ld1 { v12.4s }, [x26], #0x10
    ld1 { v16.4s }, [x25], #0x10
    ld1 { v20.4s }, [x24], #0x10
    ld1 { v24.4s }, [x23], #0x10
    tbz x8, #1, label_140
    ldr d9, [x16], #0x8
    mov x20, #0x18
    ldr d13, [x26], #0x8
    ldr d17, [x25], #0x8
    ldr d21, [x24], #0x8
    ldr d25, [x23], #0x8
    tbz x8, #0, label_143
    ld1 { v9.s }[2], [x16]
    ld1 { v13.s }[2], [x26]
    ld1 { v17.s }[2], [x25]
    ld1 { v21.s }[2], [x24]
    ld1 { v25.s }[2], [x23]
    b label_143
KAI_ASM_LABEL(label_140)  // Height 5: Partial accumulate: partial_1_4
    mov x20, #0x10
    tbz x8, #0, label_143
    ldr s9, [x16, #0x0]
    ldr s13, [x26, #0x0]
    ldr s17, [x25, #0x0]
    ldr s21, [x24, #0x0]
    ldr s25, [x23, #0x0]
    b label_143
KAI_ASM_LABEL(label_141)  // Height 5: Partial accumulate: partial_2_0
    tbz x8, #1, label_142
    ldr d8, [x16], #0x8
    mov x20, #0x8
    ldr d12, [x26], #0x8
    ldr d16, [x25], #0x8
    ldr d20, [x24], #0x8
    ldr d24, [x23], #0x8
    tbz x8, #0, label_143
    ld1 { v8.s }[2], [x16]
    ld1 { v12.s }[2], [x26]
    ld1 { v16.s }[2], [x25]
    ld1 { v20.s }[2], [x24]
    ld1 { v24.s }[2], [x23]
    b label_143
KAI_ASM_LABEL(label_142)  // Height 5: Partial accumulate: partial_1_0
    ldr s8, [x16, #0x0]
    mov x20, #0x0
    ldr s12, [x26, #0x0]
    ldr s16, [x25, #0x0]
    ldr s20, [x24, #0x0]
    ldr s24, [x23, #0x0]
KAI_ASM_LABEL(label_143)  // Height 5: Partial accumulate: Done
    sub x16, x16, x20
    b label_146
KAI_ASM_LABEL(label_144)  // Height 5: full accumulate
    ldr q8, [x16, #0x0]
    ldr q9, [x16, #0x10]
    ldr q10, [x16, #0x20]
    ldr q11, [x16, #0x30]
    ldr q12, [x26, #0x0]
    ldr q13, [x26, #0x10]
    ldr q14, [x26, #0x20]
    ldr q15, [x26, #0x30]
    ldr q16, [x25, #0x0]
    ldr q17, [x25, #0x10]
    ldr q18, [x25, #0x20]
    ldr q19, [x25, #0x30]
    ldr q20, [x24, #0x0]
    ldr q21, [x24, #0x10]
    ldr q22, [x24, #0x20]
    ldr q23, [x24, #0x30]
    ldr q24, [x23, #0x0]
    ldr q25, [x23, #0x10]
    ldr q26, [x23, #0x20]
    ldr q27, [x23, #0x30]
    b label_146
KAI_ASM_LABEL(label_145)  // Height 5: no accumulate
    movi v8.16b, #0x0
    movi v9.16b, #0x0
    movi v10.16b, #0x0
    movi v11.16b, #0x0
    movi v12.16b, #0x0
    movi v13.16b, #0x0
    movi v14.16b, #0x0
    movi v15.16b, #0x0
    movi v16.16b, #0x0
    movi v17.16b, #0x0
    movi v18.16b, #0x0
    movi v19.16b, #0x0
    movi v20.16b, #0x0
    movi v21.16b, #0x0
    movi v22.16b, #0x0
    movi v23.16b, #0x0
    movi v24.16b, #0x0
    movi v25.16b, #0x0
    movi v26.16b, #0x0
    movi v27.16b, #0x0
KAI_ASM_LABEL(label_146)  // Height 5: setup done
    mov x15, #0x0
KAI_ASM_LABEL(label_147)  // Height 5: String loop
    ldr x20, [x2, #0x10]
    ldr x21, [x2, #0x38]
    ldr w14, [x20, x15, LSL #0x2]
    tbz x3, #3, label_148
    ldr x20, [x0, x15, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x13, [x20, #0x0]
    ldr x12, [x20, #0x8]
    ldr x11, [x20, #0x10]
    ldr x10, [x20, #0x18]
    ldr x9, [x20, #0x20]
    cbnz x15, label_149
    ldr x20, [x2, #0x30]
    add x13, x13, x20, LSL #2
    add x12, x12, x20, LSL #2
    add x11, x11, x20, LSL #2
    add x10, x10, x20, LSL #2
    add x9, x9, x20, LSL #2
    b label_149
KAI_ASM_LABEL(label_148)  // Height 5: setup direct input
    mov x13, x0
    add x12, x13, x21, LSL #2
    add x11, x12, x21, LSL #2
    add x10, x11, x21, LSL #2
    add x9, x10, x21, LSL #2
KAI_ASM_LABEL(label_149)  // Height 5: input setup done
    cmp x14, #0x4
    blt label_152
    ldr q0, [x13, #0x0]
    cmp x14, #0x8
    ldr q1, [x12, #0x0]
    ldr q2, [x11, #0x0]
    ldr q3, [x10, #0x0]
    ldr q4, [x9, #0x0]
    ldr q6, [x17, #0x0]
    ldr q7, [x17, #0x10]
    blt label_151
KAI_ASM_LABEL(label_150)  // Height 5: Multiply loop: Main loop head
    fmla v8.4s, v6.4s, v0.s[0]
    ldr x21, [x17, #0x28]
    fmla v12.4s, v6.4s, v1.s[0]
    ldr x20, [x17, #0x38]
    fmla v16.4s, v6.4s, v2.s[0]
    add x13, x13, #0x10
    fmla v20.4s, v6.4s, v3.s[0]
    add x12, x12, #0x10
    fmla v24.4s, v6.4s, v4.s[0]
    ldr d6, [x17, #0x20]
    fmla v9.4s, v7.4s, v0.s[0]
    add x11, x11, #0x10
    fmla v13.4s, v7.4s, v1.s[0]
    add x10, x10, #0x10
    fmla v17.4s, v7.4s, v2.s[0]
    mov v6.d[1], x21
    fmla v21.4s, v7.4s, v3.s[0]
    ldr x21, [x17, #0x48]
    fmla v25.4s, v7.4s, v4.s[0]
    ldr d7, [x17, #0x30]
    fmla v10.4s, v6.4s, v0.s[0]
    add x9, x9, #0x10
    fmla v14.4s, v6.4s, v1.s[0]
    ldr x26, [x13, #0x8]
    mov v7.d[1], x20
    fmla v18.4s, v6.4s, v2.s[0]
    fmla v22.4s, v6.4s, v3.s[0]
    ldr x20, [x17, #0x58]
    fmla v26.4s, v6.4s, v4.s[0]
    ldr d6, [x17, #0x40]
    fmla v11.4s, v7.4s, v0.s[0]
    ldr x25, [x12, #0x8]
    fmla v15.4s, v7.4s, v1.s[0]
    ldr x24, [x11, #0x8]
    fmla v19.4s, v7.4s, v2.s[0]
    mov v6.d[1], x21
    fmla v23.4s, v7.4s, v3.s[0]
    ldr x21, [x17, #0x68]
    fmla v27.4s, v7.4s, v4.s[0]
    ldr d7, [x17, #0x50]
    fmla v8.4s, v6.4s, v0.s[1]
    ldr x23, [x10, #0x8]
    fmla v12.4s, v6.4s, v1.s[1]
    ldr x22, [x9, #0x8]
    mov v7.d[1], x20
    fmla v16.4s, v6.4s, v2.s[1]
    fmla v20.4s, v6.4s, v3.s[1]
    ldr x20, [x17, #0x78]
    fmla v24.4s, v6.4s, v4.s[1]
    ldr d6, [x17, #0x60]
    fmla v9.4s, v7.4s, v0.s[1]
    sub x14, x14, #0x4
    fmla v13.4s, v7.4s, v1.s[1]
    cmp x14, #0x8
    fmla v17.4s, v7.4s, v2.s[1]
    mov v6.d[1], x21
    fmla v21.4s, v7.4s, v3.s[1]
    ldr x21, [x17, #0x88]
    fmla v25.4s, v7.4s, v4.s[1]
    ldr d7, [x17, #0x70]
    fmla v10.4s, v6.4s, v0.s[1]
    prfm pldl1keep, [x13, #0x80]
    fmla v14.4s, v6.4s, v1.s[1]
    prfm pldl1keep, [x12, #0x80]
    mov v7.d[1], x20
    fmla v18.4s, v6.4s, v2.s[1]
    fmla v22.4s, v6.4s, v3.s[1]
    ldr x20, [x17, #0x98]
    fmla v26.4s, v6.4s, v4.s[1]
    ldr d6, [x17, #0x80]
    fmla v11.4s, v7.4s, v0.s[1]
    prfm pldl1keep, [x11, #0x80]
    fmla v15.4s, v7.4s, v1.s[1]
    prfm pldl1keep, [x10, #0x80]
    fmla v19.4s, v7.4s, v2.s[1]
    mov v6.d[1], x21
    fmla v23.4s, v7.4s, v3.s[1]
    ldr x21, [x17, #0xa8]
    fmla v27.4s, v7.4s, v4.s[1]
    ldr d7, [x17, #0x90]
    fmla v8.4s, v6.4s, v0.s[2]
    prfm pldl1keep, [x9, #0x80]
    fmla v12.4s, v6.4s, v1.s[2]
    mov v7.d[1], x20
    fmla v16.4s, v6.4s, v2.s[2]
    fmla v20.4s, v6.4s, v3.s[2]
    ldr x20, [x17, #0xb8]
    fmla v24.4s, v6.4s, v4.s[2]
    ldr d6, [x17, #0xa0]
    fmla v9.4s, v7.4s, v0.s[2]
    fmla v13.4s, v7.4s, v1.s[2]
    fmla v17.4s, v7.4s, v2.s[2]
    mov v6.d[1], x21
    fmla v21.4s, v7.4s, v3.s[2]
    ldr x21, [x17, #0xc8]
    fmla v25.4s, v7.4s, v4.s[2]
    ldr d7, [x17, #0xb0]
    fmla v10.4s, v6.4s, v0.s[2]
    fmla v14.4s, v6.4s, v1.s[2]
    mov v7.d[1], x20
    fmla v18.4s, v6.4s, v2.s[2]
    fmla v22.4s, v6.4s, v3.s[2]
    ldr x20, [x17, #0xd8]
    fmla v26.4s, v6.4s, v4.s[2]
    ldr d6, [x17, #0xc0]
    fmla v11.4s, v7.4s, v0.s[2]
    fmla v15.4s, v7.4s, v1.s[2]
    fmla v19.4s, v7.4s, v2.s[2]
    mov v6.d[1], x21
    fmla v23.4s, v7.4s, v3.s[2]
    ldr x21, [x17, #0xe8]
    fmla v27.4s, v7.4s, v4.s[2]
    ldr d7, [x17, #0xd0]
    fmla v8.4s, v6.4s, v0.s[3]
    fmla v12.4s, v6.4s, v1.s[3]
    mov v7.d[1], x20
    fmla v16.4s, v6.4s, v2.s[3]
    fmla v20.4s, v6.4s, v3.s[3]
    ldr x20, [x17, #0xf8]
    fmla v24.4s, v6.4s, v4.s[3]
    ldr d6, [x17, #0xe0]
    fmla v9.4s, v7.4s, v0.s[3]
    fmla v13.4s, v7.4s, v1.s[3]
    fmla v17.4s, v7.4s, v2.s[3]
    mov v6.d[1], x21
    fmla v21.4s, v7.4s, v3.s[3]
    fmla v25.4s, v7.4s, v4.s[3]
    ldr d7, [x17, #0xf0]
    add x17, x17, #0x100
    fmla v10.4s, v6.4s, v0.s[3]
    fmla v14.4s, v6.4s, v1.s[3]
    ldr x21, [x17, #0x8]
    mov v7.d[1], x20
    fmla v18.4s, v6.4s, v2.s[3]
    fmla v22.4s, v6.4s, v3.s[3]
    ldr x20, [x17, #0x18]
    fmla v26.4s, v6.4s, v4.s[3]
    ldr d6, [x17, #0x0]
    fmla v11.4s, v7.4s, v0.s[3]
    ldr d0, [x13, #0x0]
    fmla v15.4s, v7.4s, v1.s[3]
    ldr d1, [x12, #0x0]
    fmla v19.4s, v7.4s, v2.s[3]
    ldr d2, [x11, #0x0]
    fmla v23.4s, v7.4s, v3.s[3]
    ldr d3, [x10, #0x0]
    fmla v27.4s, v7.4s, v4.s[3]
    ldr d4, [x9, #0x0]
    ldr d7, [x17, #0x10]
    mov v6.d[1], x21
    mov v0.d[1], x26
    mov v1.d[1], x25
    mov v2.d[1], x24
    mov v3.d[1], x23
    mov v4.d[1], x22
    mov v7.d[1], x20
    bge label_150
KAI_ASM_LABEL(label_151)  // Height 5: Multiply loop: Single iteration only
    fmla v8.4s, v6.4s, v0.s[0]
    add x13, x13, #0x10
    fmla v12.4s, v6.4s, v1.s[0]
    add x12, x12, #0x10
    fmla v16.4s, v6.4s, v2.s[0]
    add x11, x11, #0x10
    fmla v20.4s, v6.4s, v3.s[0]
    add x10, x10, #0x10
    fmla v24.4s, v6.4s, v4.s[0]
    ldr q6, [x17, #0x20]
    fmla v9.4s, v7.4s, v0.s[0]
    add x9, x9, #0x10
    fmla v13.4s, v7.4s, v1.s[0]
    sub x14, x14, #0x4
    fmla v17.4s, v7.4s, v2.s[0]
    prfm pldl1keep, [x13, #0x80]
    fmla v21.4s, v7.4s, v3.s[0]
    prfm pldl1keep, [x12, #0x80]
    fmla v25.4s, v7.4s, v4.s[0]
    ldr q7, [x17, #0x30]
    fmla v10.4s, v6.4s, v0.s[0]
    prfm pldl1keep, [x11, #0x80]
    fmla v14.4s, v6.4s, v1.s[0]
    prfm pldl1keep, [x10, #0x80]
    fmla v18.4s, v6.4s, v2.s[0]
    prfm pldl1keep, [x9, #0x80]
    fmla v22.4s, v6.4s, v3.s[0]
    fmla v26.4s, v6.4s, v4.s[0]
    ldr q6, [x17, #0x40]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v15.4s, v7.4s, v1.s[0]
    fmla v19.4s, v7.4s, v2.s[0]
    fmla v23.4s, v7.4s, v3.s[0]
    fmla v27.4s, v7.4s, v4.s[0]
    ldr q7, [x17, #0x50]
    fmla v8.4s, v6.4s, v0.s[1]
    fmla v12.4s, v6.4s, v1.s[1]
    fmla v16.4s, v6.4s, v2.s[1]
    fmla v20.4s, v6.4s, v3.s[1]
    fmla v24.4s, v6.4s, v4.s[1]
    ldr q6, [x17, #0x60]
    fmla v9.4s, v7.4s, v0.s[1]
    fmla v13.4s, v7.4s, v1.s[1]
    fmla v17.4s, v7.4s, v2.s[1]
    fmla v21.4s, v7.4s, v3.s[1]
    fmla v25.4s, v7.4s, v4.s[1]
    ldr q7, [x17, #0x70]
    fmla v10.4s, v6.4s, v0.s[1]
    fmla v14.4s, v6.4s, v1.s[1]
    fmla v18.4s, v6.4s, v2.s[1]
    fmla v22.4s, v6.4s, v3.s[1]
    fmla v26.4s, v6.4s, v4.s[1]
    ldr q6, [x17, #0x80]
    fmla v11.4s, v7.4s, v0.s[1]
    fmla v15.4s, v7.4s, v1.s[1]
    fmla v19.4s, v7.4s, v2.s[1]
    fmla v23.4s, v7.4s, v3.s[1]
    fmla v27.4s, v7.4s, v4.s[1]
    ldr q7, [x17, #0x90]
    fmla v8.4s, v6.4s, v0.s[2]
    fmla v12.4s, v6.4s, v1.s[2]
    fmla v16.4s, v6.4s, v2.s[2]
    fmla v20.4s, v6.4s, v3.s[2]
    fmla v24.4s, v6.4s, v4.s[2]
    ldr q6, [x17, #0xa0]
    fmla v9.4s, v7.4s, v0.s[2]
    fmla v13.4s, v7.4s, v1.s[2]
    fmla v17.4s, v7.4s, v2.s[2]
    fmla v21.4s, v7.4s, v3.s[2]
    fmla v25.4s, v7.4s, v4.s[2]
    ldr q7, [x17, #0xb0]
    fmla v10.4s, v6.4s, v0.s[2]
    fmla v14.4s, v6.4s, v1.s[2]
    fmla v18.4s, v6.4s, v2.s[2]
    fmla v22.4s, v6.4s, v3.s[2]
    fmla v26.4s, v6.4s, v4.s[2]
    ldr q6, [x17, #0xc0]
    fmla v11.4s, v7.4s, v0.s[2]
    fmla v15.4s, v7.4s, v1.s[2]
    fmla v19.4s, v7.4s, v2.s[2]
    fmla v23.4s, v7.4s, v3.s[2]
    fmla v27.4s, v7.4s, v4.s[2]
    ldr q7, [x17, #0xd0]
    fmla v8.4s, v6.4s, v0.s[3]
    fmla v12.4s, v6.4s, v1.s[3]
    fmla v16.4s, v6.4s, v2.s[3]
    fmla v20.4s, v6.4s, v3.s[3]
    fmla v24.4s, v6.4s, v4.s[3]
    ldr q6, [x17, #0xe0]
    fmla v9.4s, v7.4s, v0.s[3]
    fmla v13.4s, v7.4s, v1.s[3]
    fmla v17.4s, v7.4s, v2.s[3]
    fmla v21.4s, v7.4s, v3.s[3]
    fmla v25.4s, v7.4s, v4.s[3]
    ldr q7, [x17, #0xf0]
    fmla v10.4s, v6.4s, v0.s[3]
    add x17, x17, #0x100
    fmla v14.4s, v6.4s, v1.s[3]
    fmla v18.4s, v6.4s, v2.s[3]
    fmla v22.4s, v6.4s, v3.s[3]
    fmla v26.4s, v6.4s, v4.s[3]
    fmla v11.4s, v7.4s, v0.s[3]
    fmla v15.4s, v7.4s, v1.s[3]
    fmla v19.4s, v7.4s, v2.s[3]
    fmla v23.4s, v7.4s, v3.s[3]
    fmla v27.4s, v7.4s, v4.s[3]
KAI_ASM_LABEL(label_152)  // Height 5: Multiply loop: Main loop skip
    cbz x14, label_154
KAI_ASM_LABEL(label_153)  // Height 5: Multiply loop: Odd block loop
    ldr s0, [x13], #0x4
    sub x14, x14, #0x1
    ldr s1, [x12], #0x4
    ldr s2, [x11], #0x4
    ldr s3, [x10], #0x4
    ldr s4, [x9], #0x4
    ldr q6, [x17, #0x0]
    ldr q7, [x17, #0x10]
    fmla v8.4s, v6.4s, v0.s[0]
    fmla v12.4s, v6.4s, v1.s[0]
    fmla v16.4s, v6.4s, v2.s[0]
    fmla v20.4s, v6.4s, v3.s[0]
    fmla v24.4s, v6.4s, v4.s[0]
    ldr q6, [x17, #0x20]
    fmla v9.4s, v7.4s, v0.s[0]
    fmla v13.4s, v7.4s, v1.s[0]
    fmla v17.4s, v7.4s, v2.s[0]
    fmla v21.4s, v7.4s, v3.s[0]
    fmla v25.4s, v7.4s, v4.s[0]
    ldr q7, [x17, #0x30]
    fmla v10.4s, v6.4s, v0.s[0]
    add x17, x17, #0x40
    fmla v14.4s, v6.4s, v1.s[0]
    fmla v18.4s, v6.4s, v2.s[0]
    fmla v22.4s, v6.4s, v3.s[0]
    fmla v26.4s, v6.4s, v4.s[0]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v15.4s, v7.4s, v1.s[0]
    fmla v19.4s, v7.4s, v2.s[0]
    fmla v23.4s, v7.4s, v3.s[0]
    fmla v27.4s, v7.4s, v4.s[0]
    cbnz x14, label_153
KAI_ASM_LABEL(label_154)  // Height 5: Multiply loop: No odd multiplies
    ldr w20, [x2, #0x8]
    add x15, x15, #0x1
    cmp x15, x20
    bne label_147
    ldr x20, [x2, #0x28]
    prfm pstl1keep, [x16, #0x0]
    add x26, x16, x20, LSL #2
    add x25, x26, x20, LSL #2
    prfm pstl1keep, [x26, #0x0]
    add x24, x25, x20, LSL #2
    prfm pstl1keep, [x25, #0x0]
    add x23, x24, x20, LSL #2
    prfm pstl1keep, [x24, #0x0]
    prfm pstl1keep, [x23, #0x0]
    tbz x3, #1, label_155
    add x21, x2, #0x0
    add x20, x2, #0x4
    ld1r { v29.4s }, [x21]
    ld1r { v28.4s }, [x20]
    fmin v8.4s, v8.4s, v29.4s
    fmin v9.4s, v9.4s, v29.4s
    fmin v10.4s, v10.4s, v29.4s
    fmin v11.4s, v11.4s, v29.4s
    fmin v12.4s, v12.4s, v29.4s
    fmin v13.4s, v13.4s, v29.4s
    fmin v14.4s, v14.4s, v29.4s
    fmin v15.4s, v15.4s, v29.4s
    fmin v16.4s, v16.4s, v29.4s
    fmin v17.4s, v17.4s, v29.4s
    fmin v18.4s, v18.4s, v29.4s
    fmin v19.4s, v19.4s, v29.4s
    fmin v20.4s, v20.4s, v29.4s
    fmin v21.4s, v21.4s, v29.4s
    fmin v22.4s, v22.4s, v29.4s
    fmin v23.4s, v23.4s, v29.4s
    fmin v24.4s, v24.4s, v29.4s
    fmin v25.4s, v25.4s, v29.4s
    fmin v26.4s, v26.4s, v29.4s
    fmin v27.4s, v27.4s, v29.4s
    fmax v8.4s, v8.4s, v28.4s
    fmax v9.4s, v9.4s, v28.4s
    fmax v10.4s, v10.4s, v28.4s
    fmax v11.4s, v11.4s, v28.4s
    fmax v12.4s, v12.4s, v28.4s
    fmax v13.4s, v13.4s, v28.4s
    fmax v14.4s, v14.4s, v28.4s
    fmax v15.4s, v15.4s, v28.4s
    fmax v16.4s, v16.4s, v28.4s
    fmax v17.4s, v17.4s, v28.4s
    fmax v18.4s, v18.4s, v28.4s
    fmax v19.4s, v19.4s, v28.4s
    fmax v20.4s, v20.4s, v28.4s
    fmax v21.4s, v21.4s, v28.4s
    fmax v22.4s, v22.4s, v28.4s
    fmax v23.4s, v23.4s, v28.4s
    fmax v24.4s, v24.4s, v28.4s
    fmax v25.4s, v25.4s, v28.4s
    fmax v26.4s, v26.4s, v28.4s
    fmax v27.4s, v27.4s, v28.4s
KAI_ASM_LABEL(label_155)  // Height 5: No activation
    cmp x8, #0x10
    bge label_164
    tbz x8, #3, label_159
    st1 { v8.4s }, [x16], #0x10
    st1 { v9.4s }, [x16], #0x10
    st1 { v12.4s }, [x26], #0x10
    st1 { v13.4s }, [x26], #0x10
    st1 { v16.4s }, [x25], #0x10
    st1 { v17.4s }, [x25], #0x10
    st1 { v20.4s }, [x24], #0x10
    st1 { v21.4s }, [x24], #0x10
    st1 { v24.4s }, [x23], #0x10
    st1 { v25.4s }, [x23], #0x10
    tbz x8, #2, label_157
    st1 { v10.4s }, [x16], #0x10
    st1 { v14.4s }, [x26], #0x10
    st1 { v18.4s }, [x25], #0x10
    st1 { v22.4s }, [x24], #0x10
    st1 { v26.4s }, [x23], #0x10
    tbz x8, #1, label_156
    str d11, [x16], #0x8
    str d15, [x26], #0x8
    str d19, [x25], #0x8
    str d23, [x24], #0x8
    str d27, [x23], #0x8
    tbz x8, #0, label_163
    st1 { v11.s }[2], [x16]
    st1 { v15.s }[2], [x26]
    st1 { v19.s }[2], [x25]
    st1 { v23.s }[2], [x24]
    st1 { v27.s }[2], [x23]
    b label_163
KAI_ASM_LABEL(label_156)  // Height 5: Partial direct writeback: partial_1_12
    tbz x8, #0, label_163
    str s11, [x16, #0x0]
    str s15, [x26, #0x0]
    str s19, [x25, #0x0]
    str s23, [x24, #0x0]
    str s27, [x23, #0x0]
    b label_163
KAI_ASM_LABEL(label_157)  // Height 5: Partial direct writeback: partial_2_8
    tbz x8, #1, label_158
    str d10, [x16], #0x8
    str d14, [x26], #0x8
    str d18, [x25], #0x8
    str d22, [x24], #0x8
    str d26, [x23], #0x8
    tbz x8, #0, label_163
    st1 { v10.s }[2], [x16]
    st1 { v14.s }[2], [x26]
    st1 { v18.s }[2], [x25]
    st1 { v22.s }[2], [x24]
    st1 { v26.s }[2], [x23]
    b label_163
KAI_ASM_LABEL(label_158)  // Height 5: Partial direct writeback: partial_1_8
    tbz x8, #0, label_163
    str s10, [x16, #0x0]
    str s14, [x26, #0x0]
    str s18, [x25, #0x0]
    str s22, [x24, #0x0]
    str s26, [x23, #0x0]
    b label_163
KAI_ASM_LABEL(label_159)  // Height 5: Partial direct writeback: partial_4_0
    tbz x8, #2, label_161
    st1 { v8.4s }, [x16], #0x10
    st1 { v12.4s }, [x26], #0x10
    st1 { v16.4s }, [x25], #0x10
    st1 { v20.4s }, [x24], #0x10
    st1 { v24.4s }, [x23], #0x10
    tbz x8, #1, label_160
    str d9, [x16], #0x8
    str d13, [x26], #0x8
    str d17, [x25], #0x8
    str d21, [x24], #0x8
    str d25, [x23], #0x8
    tbz x8, #0, label_163
    st1 { v9.s }[2], [x16]
    st1 { v13.s }[2], [x26]
    st1 { v17.s }[2], [x25]
    st1 { v21.s }[2], [x24]
    st1 { v25.s }[2], [x23]
    b label_163
KAI_ASM_LABEL(label_160)  // Height 5: Partial direct writeback: partial_1_4
    tbz x8, #0, label_163
    str s9, [x16, #0x0]
    str s13, [x26, #0x0]
    str s17, [x25, #0x0]
    str s21, [x24, #0x0]
    str s25, [x23, #0x0]
    b label_163
KAI_ASM_LABEL(label_161)  // Height 5: Partial direct writeback: partial_2_0
    tbz x8, #1, label_162
    str d8, [x16], #0x8
    str d12, [x26], #0x8
    str d16, [x25], #0x8
    str d20, [x24], #0x8
    str d24, [x23], #0x8
    tbz x8, #0, label_163
    st1 { v8.s }[2], [x16]
    st1 { v12.s }[2], [x26]
    st1 { v16.s }[2], [x25]
    st1 { v20.s }[2], [x24]
    st1 { v24.s }[2], [x23]
    b label_163
KAI_ASM_LABEL(label_162)  // Height 5: Partial direct writeback: partial_1_0
    str s8, [x16, #0x0]
    str s12, [x26, #0x0]
    str s16, [x25, #0x0]
    str s20, [x24, #0x0]
    str s24, [x23, #0x0]
KAI_ASM_LABEL(label_163)  // Height 5: Partial direct writeback: Done
    b label_165
KAI_ASM_LABEL(label_164)  // Height 5: Full writeback
    str q8, [x16, #0x0]
    str q9, [x16, #0x10]
    str q10, [x16, #0x20]
    str q11, [x16, #0x30]
    add x16, x16, #0x40
    str q12, [x26, #0x0]
    str q13, [x26, #0x10]
    str q14, [x26, #0x20]
    str q15, [x26, #0x30]
    str q16, [x25, #0x0]
    str q17, [x25, #0x10]
    str q18, [x25, #0x20]
    str q19, [x25, #0x30]
    str q20, [x24, #0x0]
    str q21, [x24, #0x10]
    str q22, [x24, #0x20]
    str q23, [x24, #0x30]
    str q24, [x23, #0x0]
    str q25, [x23, #0x10]
    str q26, [x23, #0x20]
    str q27, [x23, #0x30]
KAI_ASM_LABEL(label_165)  // Height 5: Writeback done
    subs x8, x8, #0x10
    bgt label_134
    b label_200
KAI_ASM_LABEL(label_166)  // Height 6
    ldr x21, [x2, #0x28]
    mov x20, #0x18
    ldr x16, [x2, #0x40]
    ldr x8, [x2, #0x18]
    ldr x17, [x2, #0x20]
    madd x20, x21, x20, x16
    str x20, [x2, #0x40]
KAI_ASM_LABEL(label_167)  // Height 6: Column loop
    cbz x17, label_168
    ldr q8, [x17, #0x0]
    ldr q9, [x17, #0x10]
    ldr q10, [x17, #0x20]
    ldr q11, [x17, #0x30]
    mov v12.16b, v8.16b
    mov v13.16b, v9.16b
    add x17, x17, #0x40
    mov v14.16b, v10.16b
    mov v15.16b, v11.16b
    mov v16.16b, v8.16b
    mov v17.16b, v9.16b
    mov v18.16b, v10.16b
    mov v19.16b, v11.16b
    mov v20.16b, v8.16b
    mov v21.16b, v9.16b
    mov v22.16b, v10.16b
    mov v23.16b, v11.16b
    mov v24.16b, v8.16b
    mov v25.16b, v9.16b
    mov v26.16b, v10.16b
    mov v27.16b, v11.16b
    mov v28.16b, v8.16b
    mov v29.16b, v9.16b
    mov v30.16b, v10.16b
    mov v31.16b, v11.16b
    b label_179
KAI_ASM_LABEL(label_168)  // Height 6: no bias
    tbz x3, #0, label_178
    ldr x20, [x2, #0x28]
    cmp x8, #0x10
    add x26, x16, x20, LSL #2
    add x25, x26, x20, LSL #2
    add x24, x25, x20, LSL #2
    add x23, x24, x20, LSL #2
    add x22, x23, x20, LSL #2
    bge label_177
    tbz x8, #3, label_172
    ld1 { v8.4s }, [x16], #0x10
    ld1 { v12.4s }, [x26], #0x10
    ld1 { v16.4s }, [x25], #0x10
    ld1 { v20.4s }, [x24], #0x10
    ld1 { v24.4s }, [x23], #0x10
    ld1 { v28.4s }, [x22], #0x10
    ld1 { v9.4s }, [x16], #0x10
    ld1 { v13.4s }, [x26], #0x10
    ld1 { v17.4s }, [x25], #0x10
    ld1 { v21.4s }, [x24], #0x10
    ld1 { v25.4s }, [x23], #0x10
    ld1 { v29.4s }, [x22], #0x10
    tbz x8, #2, label_170
    ld1 { v10.4s }, [x16], #0x10
    ld1 { v14.4s }, [x26], #0x10
    ld1 { v18.4s }, [x25], #0x10
    ld1 { v22.4s }, [x24], #0x10
    ld1 { v26.4s }, [x23], #0x10
    ld1 { v30.4s }, [x22], #0x10
    tbz x8, #1, label_169
    ldr d11, [x16], #0x8
    mov x20, #0x38
    ldr d15, [x26], #0x8
    ldr d19, [x25], #0x8
    ldr d23, [x24], #0x8
    ldr d27, [x23], #0x8
    ldr d31, [x22], #0x8
    tbz x8, #0, label_176
    ld1 { v11.s }[2], [x16]
    ld1 { v15.s }[2], [x26]
    ld1 { v19.s }[2], [x25]
    ld1 { v23.s }[2], [x24]
    ld1 { v27.s }[2], [x23]
    ld1 { v31.s }[2], [x22]
    b label_176
KAI_ASM_LABEL(label_169)  // Height 6: Partial accumulate: partial_1_12
    mov x20, #0x30
    tbz x8, #0, label_176
    ldr s11, [x16, #0x0]
    ldr s15, [x26, #0x0]
    ldr s19, [x25, #0x0]
    ldr s23, [x24, #0x0]
    ldr s27, [x23, #0x0]
    ldr s31, [x22, #0x0]
    b label_176
KAI_ASM_LABEL(label_170)  // Height 6: Partial accumulate: partial_2_8
    tbz x8, #1, label_171
    ldr d10, [x16], #0x8
    mov x20, #0x28
    ldr d14, [x26], #0x8
    ldr d18, [x25], #0x8
    ldr d22, [x24], #0x8
    ldr d26, [x23], #0x8
    ldr d30, [x22], #0x8
    tbz x8, #0, label_176
    ld1 { v10.s }[2], [x16]
    ld1 { v14.s }[2], [x26]
    ld1 { v18.s }[2], [x25]
    ld1 { v22.s }[2], [x24]
    ld1 { v26.s }[2], [x23]
    ld1 { v30.s }[2], [x22]
    b label_176
KAI_ASM_LABEL(label_171)  // Height 6: Partial accumulate: partial_1_8
    mov x20, #0x20
    tbz x8, #0, label_176
    ldr s10, [x16, #0x0]
    ldr s14, [x26, #0x0]
    ldr s18, [x25, #0x0]
    ldr s22, [x24, #0x0]
    ldr s26, [x23, #0x0]
    ldr s30, [x22, #0x0]
    b label_176
KAI_ASM_LABEL(label_172)  // Height 6: Partial accumulate: partial_4_0
    tbz x8, #2, label_174
    ld1 { v8.4s }, [x16], #0x10
    ld1 { v12.4s }, [x26], #0x10
    ld1 { v16.4s }, [x25], #0x10
    ld1 { v20.4s }, [x24], #0x10
    ld1 { v24.4s }, [x23], #0x10
    ld1 { v28.4s }, [x22], #0x10
    tbz x8, #1, label_173
    ldr d9, [x16], #0x8
    mov x20, #0x18
    ldr d13, [x26], #0x8
    ldr d17, [x25], #0x8
    ldr d21, [x24], #0x8
    ldr d25, [x23], #0x8
    ldr d29, [x22], #0x8
    tbz x8, #0, label_176
    ld1 { v9.s }[2], [x16]
    ld1 { v13.s }[2], [x26]
    ld1 { v17.s }[2], [x25]
    ld1 { v21.s }[2], [x24]
    ld1 { v25.s }[2], [x23]
    ld1 { v29.s }[2], [x22]
    b label_176
KAI_ASM_LABEL(label_173)  // Height 6: Partial accumulate: partial_1_4
    mov x20, #0x10
    tbz x8, #0, label_176
    ldr s9, [x16, #0x0]
    ldr s13, [x26, #0x0]
    ldr s17, [x25, #0x0]
    ldr s21, [x24, #0x0]
    ldr s25, [x23, #0x0]
    ldr s29, [x22, #0x0]
    b label_176
KAI_ASM_LABEL(label_174)  // Height 6: Partial accumulate: partial_2_0
    tbz x8, #1, label_175
    ldr d8, [x16], #0x8
    mov x20, #0x8
    ldr d12, [x26], #0x8
    ldr d16, [x25], #0x8
    ldr d20, [x24], #0x8
    ldr d24, [x23], #0x8
    ldr d28, [x22], #0x8
    tbz x8, #0, label_176
    ld1 { v8.s }[2], [x16]
    ld1 { v12.s }[2], [x26]
    ld1 { v16.s }[2], [x25]
    ld1 { v20.s }[2], [x24]
    ld1 { v24.s }[2], [x23]
    ld1 { v28.s }[2], [x22]
    b label_176
KAI_ASM_LABEL(label_175)  // Height 6: Partial accumulate: partial_1_0
    ldr s8, [x16, #0x0]
    mov x20, #0x0
    ldr s12, [x26, #0x0]
    ldr s16, [x25, #0x0]
    ldr s20, [x24, #0x0]
    ldr s24, [x23, #0x0]
    ldr s28, [x22, #0x0]
KAI_ASM_LABEL(label_176)  // Height 6: Partial accumulate: Done
    sub x16, x16, x20
    b label_179
KAI_ASM_LABEL(label_177)  // Height 6: full accumulate
    ldr q8, [x16, #0x0]
    ldr q9, [x16, #0x10]
    ldr q10, [x16, #0x20]
    ldr q11, [x16, #0x30]
    ldr q12, [x26, #0x0]
    ldr q13, [x26, #0x10]
    ldr q14, [x26, #0x20]
    ldr q15, [x26, #0x30]
    ldr q16, [x25, #0x0]
    ldr q17, [x25, #0x10]
    ldr q18, [x25, #0x20]
    ldr q19, [x25, #0x30]
    ldr q20, [x24, #0x0]
    ldr q21, [x24, #0x10]
    ldr q22, [x24, #0x20]
    ldr q23, [x24, #0x30]
    ldr q24, [x23, #0x0]
    ldr q25, [x23, #0x10]
    ldr q26, [x23, #0x20]
    ldr q27, [x23, #0x30]
    ldr q28, [x22, #0x0]
    ldr q29, [x22, #0x10]
    ldr q30, [x22, #0x20]
    ldr q31, [x22, #0x30]
    b label_179
KAI_ASM_LABEL(label_178)  // Height 6: no accumulate
    movi v8.16b, #0x0
    movi v9.16b, #0x0
    movi v10.16b, #0x0
    movi v11.16b, #0x0
    movi v12.16b, #0x0
    movi v13.16b, #0x0
    movi v14.16b, #0x0
    movi v15.16b, #0x0
    movi v16.16b, #0x0
    movi v17.16b, #0x0
    movi v18.16b, #0x0
    movi v19.16b, #0x0
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
KAI_ASM_LABEL(label_179)  // Height 6: setup done
    mov x15, #0x0
KAI_ASM_LABEL(label_180)  // Height 6: String loop
    ldr x20, [x2, #0x10]
    ldr x21, [x2, #0x38]
    ldr w14, [x20, x15, LSL #0x2]
    tbz x3, #3, label_181
    ldr x20, [x0, x15, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x13, [x20, #0x0]
    ldr x12, [x20, #0x8]
    ldr x11, [x20, #0x10]
    ldr x10, [x20, #0x18]
    ldr x9, [x20, #0x20]
    ldr x28, [x20, #0x28]
    cbnz x15, label_182
    ldr x20, [x2, #0x30]
    add x13, x13, x20, LSL #2
    add x12, x12, x20, LSL #2
    add x11, x11, x20, LSL #2
    add x10, x10, x20, LSL #2
    add x9, x9, x20, LSL #2
    add x28, x28, x20, LSL #2
    b label_182
KAI_ASM_LABEL(label_181)  // Height 6: setup direct input
    mov x13, x0
    add x12, x13, x21, LSL #2
    add x11, x12, x21, LSL #2
    add x10, x11, x21, LSL #2
    add x9, x10, x21, LSL #2
    add x28, x9, x21, LSL #2
KAI_ASM_LABEL(label_182)  // Height 6: input setup done
    cmp x14, #0x4
    blt label_185
    ldr q0, [x13, #0x0]
    cmp x14, #0x8
    ldr q1, [x12, #0x0]
    ldr q2, [x11, #0x0]
    ldr q3, [x10, #0x0]
    ldr q4, [x9, #0x0]
    ldr q5, [x28, #0x0]
    ldr q6, [x17, #0x0]
    ldr q7, [x17, #0x10]
    blt label_184
KAI_ASM_LABEL(label_183)  // Height 6: Multiply loop: Main loop head
    fmla v8.4s, v6.4s, v0.s[0]
    ldr x21, [x17, #0x28]
    fmla v12.4s, v6.4s, v1.s[0]
    ldr x20, [x17, #0x38]
    fmla v16.4s, v6.4s, v2.s[0]
    add x13, x13, #0x10
    fmla v20.4s, v6.4s, v3.s[0]
    add x12, x12, #0x10
    fmla v24.4s, v6.4s, v4.s[0]
    add x11, x11, #0x10
    fmla v28.4s, v6.4s, v5.s[0]
    ldr d6, [x17, #0x20]
    fmla v9.4s, v7.4s, v0.s[0]
    add x10, x10, #0x10
    fmla v13.4s, v7.4s, v1.s[0]
    add x9, x9, #0x10
    fmla v17.4s, v7.4s, v2.s[0]
    mov v6.d[1], x21
    fmla v21.4s, v7.4s, v3.s[0]
    ldr x21, [x17, #0x48]
    fmla v25.4s, v7.4s, v4.s[0]
    add x28, x28, #0x10
    fmla v29.4s, v7.4s, v5.s[0]
    ldr d7, [x17, #0x30]
    fmla v10.4s, v6.4s, v0.s[0]
    ldr x27, [x13, #0x8]
    fmla v14.4s, v6.4s, v1.s[0]
    ldr x26, [x12, #0x8]
    mov v7.d[1], x20
    fmla v18.4s, v6.4s, v2.s[0]
    fmla v22.4s, v6.4s, v3.s[0]
    ldr x20, [x17, #0x58]
    fmla v26.4s, v6.4s, v4.s[0]
    ldr x25, [x11, #0x8]
    fmla v30.4s, v6.4s, v5.s[0]
    ldr d6, [x17, #0x40]
    fmla v11.4s, v7.4s, v0.s[0]
    ldr x24, [x10, #0x8]
    fmla v15.4s, v7.4s, v1.s[0]
    ldr x23, [x9, #0x8]
    fmla v19.4s, v7.4s, v2.s[0]
    mov v6.d[1], x21
    fmla v23.4s, v7.4s, v3.s[0]
    ldr x21, [x17, #0x68]
    fmla v27.4s, v7.4s, v4.s[0]
    ldr x22, [x28, #0x8]
    fmla v31.4s, v7.4s, v5.s[0]
    ldr d7, [x17, #0x50]
    fmla v8.4s, v6.4s, v0.s[1]
    sub x14, x14, #0x4
    fmla v12.4s, v6.4s, v1.s[1]
    cmp x14, #0x8
    mov v7.d[1], x20
    fmla v16.4s, v6.4s, v2.s[1]
    fmla v20.4s, v6.4s, v3.s[1]
    ldr x20, [x17, #0x78]
    fmla v24.4s, v6.4s, v4.s[1]
    prfm pldl1keep, [x13, #0x80]
    fmla v28.4s, v6.4s, v5.s[1]
    ldr d6, [x17, #0x60]
    fmla v9.4s, v7.4s, v0.s[1]
    prfm pldl1keep, [x12, #0x80]
    fmla v13.4s, v7.4s, v1.s[1]
    prfm pldl1keep, [x11, #0x80]
    fmla v17.4s, v7.4s, v2.s[1]
    mov v6.d[1], x21
    fmla v21.4s, v7.4s, v3.s[1]
    ldr x21, [x17, #0x88]
    fmla v25.4s, v7.4s, v4.s[1]
    prfm pldl1keep, [x10, #0x80]
    fmla v29.4s, v7.4s, v5.s[1]
    ldr d7, [x17, #0x70]
    fmla v10.4s, v6.4s, v0.s[1]
    prfm pldl1keep, [x9, #0x80]
    fmla v14.4s, v6.4s, v1.s[1]
    prfm pldl1keep, [x28, #0x80]
    mov v7.d[1], x20
    fmla v18.4s, v6.4s, v2.s[1]
    fmla v22.4s, v6.4s, v3.s[1]
    ldr x20, [x17, #0x98]
    fmla v26.4s, v6.4s, v4.s[1]
    fmla v30.4s, v6.4s, v5.s[1]
    ldr d6, [x17, #0x80]
    fmla v11.4s, v7.4s, v0.s[1]
    fmla v15.4s, v7.4s, v1.s[1]
    fmla v19.4s, v7.4s, v2.s[1]
    mov v6.d[1], x21
    fmla v23.4s, v7.4s, v3.s[1]
    ldr x21, [x17, #0xa8]
    fmla v27.4s, v7.4s, v4.s[1]
    fmla v31.4s, v7.4s, v5.s[1]
    ldr d7, [x17, #0x90]
    fmla v8.4s, v6.4s, v0.s[2]
    fmla v12.4s, v6.4s, v1.s[2]
    mov v7.d[1], x20
    fmla v16.4s, v6.4s, v2.s[2]
    fmla v20.4s, v6.4s, v3.s[2]
    ldr x20, [x17, #0xb8]
    fmla v24.4s, v6.4s, v4.s[2]
    fmla v28.4s, v6.4s, v5.s[2]
    ldr d6, [x17, #0xa0]
    fmla v9.4s, v7.4s, v0.s[2]
    fmla v13.4s, v7.4s, v1.s[2]
    fmla v17.4s, v7.4s, v2.s[2]
    mov v6.d[1], x21
    fmla v21.4s, v7.4s, v3.s[2]
    ldr x21, [x17, #0xc8]
    fmla v25.4s, v7.4s, v4.s[2]
    fmla v29.4s, v7.4s, v5.s[2]
    ldr d7, [x17, #0xb0]
    fmla v10.4s, v6.4s, v0.s[2]
    fmla v14.4s, v6.4s, v1.s[2]
    mov v7.d[1], x20
    fmla v18.4s, v6.4s, v2.s[2]
    fmla v22.4s, v6.4s, v3.s[2]
    ldr x20, [x17, #0xd8]
    fmla v26.4s, v6.4s, v4.s[2]
    fmla v30.4s, v6.4s, v5.s[2]
    ldr d6, [x17, #0xc0]
    fmla v11.4s, v7.4s, v0.s[2]
    fmla v15.4s, v7.4s, v1.s[2]
    fmla v19.4s, v7.4s, v2.s[2]
    mov v6.d[1], x21
    fmla v23.4s, v7.4s, v3.s[2]
    ldr x21, [x17, #0xe8]
    fmla v27.4s, v7.4s, v4.s[2]
    fmla v31.4s, v7.4s, v5.s[2]
    ldr d7, [x17, #0xd0]
    fmla v8.4s, v6.4s, v0.s[3]
    fmla v12.4s, v6.4s, v1.s[3]
    mov v7.d[1], x20
    fmla v16.4s, v6.4s, v2.s[3]
    fmla v20.4s, v6.4s, v3.s[3]
    ldr x20, [x17, #0xf8]
    fmla v24.4s, v6.4s, v4.s[3]
    fmla v28.4s, v6.4s, v5.s[3]
    ldr d6, [x17, #0xe0]
    fmla v9.4s, v7.4s, v0.s[3]
    fmla v13.4s, v7.4s, v1.s[3]
    fmla v17.4s, v7.4s, v2.s[3]
    mov v6.d[1], x21
    fmla v21.4s, v7.4s, v3.s[3]
    fmla v25.4s, v7.4s, v4.s[3]
    fmla v29.4s, v7.4s, v5.s[3]
    ldr d7, [x17, #0xf0]
    add x17, x17, #0x100
    fmla v10.4s, v6.4s, v0.s[3]
    fmla v14.4s, v6.4s, v1.s[3]
    ldr x21, [x17, #0x8]
    mov v7.d[1], x20
    fmla v18.4s, v6.4s, v2.s[3]
    fmla v22.4s, v6.4s, v3.s[3]
    ldr x20, [x17, #0x18]
    fmla v26.4s, v6.4s, v4.s[3]
    fmla v30.4s, v6.4s, v5.s[3]
    ldr d6, [x17, #0x0]
    fmla v11.4s, v7.4s, v0.s[3]
    ldr d0, [x13, #0x0]
    fmla v15.4s, v7.4s, v1.s[3]
    ldr d1, [x12, #0x0]
    fmla v19.4s, v7.4s, v2.s[3]
    ldr d2, [x11, #0x0]
    fmla v23.4s, v7.4s, v3.s[3]
    ldr d3, [x10, #0x0]
    fmla v27.4s, v7.4s, v4.s[3]
    ldr d4, [x9, #0x0]
    fmla v31.4s, v7.4s, v5.s[3]
    ldr d5, [x28, #0x0]
    ldr d7, [x17, #0x10]
    mov v6.d[1], x21
    mov v0.d[1], x27
    mov v1.d[1], x26
    mov v2.d[1], x25
    mov v3.d[1], x24
    mov v4.d[1], x23
    mov v5.d[1], x22
    mov v7.d[1], x20
    bge label_183
KAI_ASM_LABEL(label_184)  // Height 6: Multiply loop: Single iteration only
    fmla v8.4s, v6.4s, v0.s[0]
    add x13, x13, #0x10
    fmla v12.4s, v6.4s, v1.s[0]
    add x12, x12, #0x10
    fmla v16.4s, v6.4s, v2.s[0]
    add x11, x11, #0x10
    fmla v20.4s, v6.4s, v3.s[0]
    add x10, x10, #0x10
    fmla v24.4s, v6.4s, v4.s[0]
    add x9, x9, #0x10
    fmla v28.4s, v6.4s, v5.s[0]
    ldr q6, [x17, #0x20]
    fmla v9.4s, v7.4s, v0.s[0]
    add x28, x28, #0x10
    fmla v13.4s, v7.4s, v1.s[0]
    sub x14, x14, #0x4
    fmla v17.4s, v7.4s, v2.s[0]
    prfm pldl1keep, [x13, #0x80]
    fmla v21.4s, v7.4s, v3.s[0]
    prfm pldl1keep, [x12, #0x80]
    fmla v25.4s, v7.4s, v4.s[0]
    prfm pldl1keep, [x11, #0x80]
    fmla v29.4s, v7.4s, v5.s[0]
    ldr q7, [x17, #0x30]
    fmla v10.4s, v6.4s, v0.s[0]
    prfm pldl1keep, [x10, #0x80]
    fmla v14.4s, v6.4s, v1.s[0]
    prfm pldl1keep, [x9, #0x80]
    fmla v18.4s, v6.4s, v2.s[0]
    prfm pldl1keep, [x28, #0x80]
    fmla v22.4s, v6.4s, v3.s[0]
    fmla v26.4s, v6.4s, v4.s[0]
    fmla v30.4s, v6.4s, v5.s[0]
    ldr q6, [x17, #0x40]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v15.4s, v7.4s, v1.s[0]
    fmla v19.4s, v7.4s, v2.s[0]
    fmla v23.4s, v7.4s, v3.s[0]
    fmla v27.4s, v7.4s, v4.s[0]
    fmla v31.4s, v7.4s, v5.s[0]
    ldr q7, [x17, #0x50]
    fmla v8.4s, v6.4s, v0.s[1]
    fmla v12.4s, v6.4s, v1.s[1]
    fmla v16.4s, v6.4s, v2.s[1]
    fmla v20.4s, v6.4s, v3.s[1]
    fmla v24.4s, v6.4s, v4.s[1]
    fmla v28.4s, v6.4s, v5.s[1]
    ldr q6, [x17, #0x60]
    fmla v9.4s, v7.4s, v0.s[1]
    fmla v13.4s, v7.4s, v1.s[1]
    fmla v17.4s, v7.4s, v2.s[1]
    fmla v21.4s, v7.4s, v3.s[1]
    fmla v25.4s, v7.4s, v4.s[1]
    fmla v29.4s, v7.4s, v5.s[1]
    ldr q7, [x17, #0x70]
    fmla v10.4s, v6.4s, v0.s[1]
    fmla v14.4s, v6.4s, v1.s[1]
    fmla v18.4s, v6.4s, v2.s[1]
    fmla v22.4s, v6.4s, v3.s[1]
    fmla v26.4s, v6.4s, v4.s[1]
    fmla v30.4s, v6.4s, v5.s[1]
    ldr q6, [x17, #0x80]
    fmla v11.4s, v7.4s, v0.s[1]
    fmla v15.4s, v7.4s, v1.s[1]
    fmla v19.4s, v7.4s, v2.s[1]
    fmla v23.4s, v7.4s, v3.s[1]
    fmla v27.4s, v7.4s, v4.s[1]
    fmla v31.4s, v7.4s, v5.s[1]
    ldr q7, [x17, #0x90]
    fmla v8.4s, v6.4s, v0.s[2]
    fmla v12.4s, v6.4s, v1.s[2]
    fmla v16.4s, v6.4s, v2.s[2]
    fmla v20.4s, v6.4s, v3.s[2]
    fmla v24.4s, v6.4s, v4.s[2]
    fmla v28.4s, v6.4s, v5.s[2]
    ldr q6, [x17, #0xa0]
    fmla v9.4s, v7.4s, v0.s[2]
    fmla v13.4s, v7.4s, v1.s[2]
    fmla v17.4s, v7.4s, v2.s[2]
    fmla v21.4s, v7.4s, v3.s[2]
    fmla v25.4s, v7.4s, v4.s[2]
    fmla v29.4s, v7.4s, v5.s[2]
    ldr q7, [x17, #0xb0]
    fmla v10.4s, v6.4s, v0.s[2]
    fmla v14.4s, v6.4s, v1.s[2]
    fmla v18.4s, v6.4s, v2.s[2]
    fmla v22.4s, v6.4s, v3.s[2]
    fmla v26.4s, v6.4s, v4.s[2]
    fmla v30.4s, v6.4s, v5.s[2]
    ldr q6, [x17, #0xc0]
    fmla v11.4s, v7.4s, v0.s[2]
    fmla v15.4s, v7.4s, v1.s[2]
    fmla v19.4s, v7.4s, v2.s[2]
    fmla v23.4s, v7.4s, v3.s[2]
    fmla v27.4s, v7.4s, v4.s[2]
    fmla v31.4s, v7.4s, v5.s[2]
    ldr q7, [x17, #0xd0]
    fmla v8.4s, v6.4s, v0.s[3]
    fmla v12.4s, v6.4s, v1.s[3]
    fmla v16.4s, v6.4s, v2.s[3]
    fmla v20.4s, v6.4s, v3.s[3]
    fmla v24.4s, v6.4s, v4.s[3]
    fmla v28.4s, v6.4s, v5.s[3]
    ldr q6, [x17, #0xe0]
    fmla v9.4s, v7.4s, v0.s[3]
    fmla v13.4s, v7.4s, v1.s[3]
    fmla v17.4s, v7.4s, v2.s[3]
    fmla v21.4s, v7.4s, v3.s[3]
    fmla v25.4s, v7.4s, v4.s[3]
    fmla v29.4s, v7.4s, v5.s[3]
    ldr q7, [x17, #0xf0]
    fmla v10.4s, v6.4s, v0.s[3]
    add x17, x17, #0x100
    fmla v14.4s, v6.4s, v1.s[3]
    fmla v18.4s, v6.4s, v2.s[3]
    fmla v22.4s, v6.4s, v3.s[3]
    fmla v26.4s, v6.4s, v4.s[3]
    fmla v30.4s, v6.4s, v5.s[3]
    fmla v11.4s, v7.4s, v0.s[3]
    fmla v15.4s, v7.4s, v1.s[3]
    fmla v19.4s, v7.4s, v2.s[3]
    fmla v23.4s, v7.4s, v3.s[3]
    fmla v27.4s, v7.4s, v4.s[3]
    fmla v31.4s, v7.4s, v5.s[3]
KAI_ASM_LABEL(label_185)  // Height 6: Multiply loop: Main loop skip
    cbz x14, label_187
KAI_ASM_LABEL(label_186)  // Height 6: Multiply loop: Odd block loop
    ldr s0, [x13], #0x4
    sub x14, x14, #0x1
    ldr s1, [x12], #0x4
    ldr s2, [x11], #0x4
    ldr s3, [x10], #0x4
    ldr s4, [x9], #0x4
    ldr s5, [x28], #0x4
    ldr q6, [x17, #0x0]
    ldr q7, [x17, #0x10]
    fmla v8.4s, v6.4s, v0.s[0]
    fmla v12.4s, v6.4s, v1.s[0]
    fmla v16.4s, v6.4s, v2.s[0]
    fmla v20.4s, v6.4s, v3.s[0]
    fmla v24.4s, v6.4s, v4.s[0]
    fmla v28.4s, v6.4s, v5.s[0]
    ldr q6, [x17, #0x20]
    fmla v9.4s, v7.4s, v0.s[0]
    fmla v13.4s, v7.4s, v1.s[0]
    fmla v17.4s, v7.4s, v2.s[0]
    fmla v21.4s, v7.4s, v3.s[0]
    fmla v25.4s, v7.4s, v4.s[0]
    fmla v29.4s, v7.4s, v5.s[0]
    ldr q7, [x17, #0x30]
    fmla v10.4s, v6.4s, v0.s[0]
    add x17, x17, #0x40
    fmla v14.4s, v6.4s, v1.s[0]
    fmla v18.4s, v6.4s, v2.s[0]
    fmla v22.4s, v6.4s, v3.s[0]
    fmla v26.4s, v6.4s, v4.s[0]
    fmla v30.4s, v6.4s, v5.s[0]
    fmla v11.4s, v7.4s, v0.s[0]
    fmla v15.4s, v7.4s, v1.s[0]
    fmla v19.4s, v7.4s, v2.s[0]
    fmla v23.4s, v7.4s, v3.s[0]
    fmla v27.4s, v7.4s, v4.s[0]
    fmla v31.4s, v7.4s, v5.s[0]
    cbnz x14, label_186
KAI_ASM_LABEL(label_187)  // Height 6: Multiply loop: No odd multiplies
    ldr w20, [x2, #0x8]
    add x15, x15, #0x1
    cmp x15, x20
    bne label_180
    ldr x20, [x2, #0x28]
    prfm pstl1keep, [x16, #0x0]
    add x26, x16, x20, LSL #2
    add x25, x26, x20, LSL #2
    prfm pstl1keep, [x26, #0x0]
    add x24, x25, x20, LSL #2
    prfm pstl1keep, [x25, #0x0]
    add x23, x24, x20, LSL #2
    prfm pstl1keep, [x24, #0x0]
    add x22, x23, x20, LSL #2
    prfm pstl1keep, [x23, #0x0]
    prfm pstl1keep, [x22, #0x0]
    tbz x3, #1, label_188
    add x21, x2, #0x0
    add x20, x2, #0x4
    ld1r { v1.4s }, [x21]
    ld1r { v0.4s }, [x20]
    fmin v8.4s, v8.4s, v1.4s
    fmin v9.4s, v9.4s, v1.4s
    fmin v10.4s, v10.4s, v1.4s
    fmin v11.4s, v11.4s, v1.4s
    fmin v12.4s, v12.4s, v1.4s
    fmin v13.4s, v13.4s, v1.4s
    fmin v14.4s, v14.4s, v1.4s
    fmin v15.4s, v15.4s, v1.4s
    fmin v16.4s, v16.4s, v1.4s
    fmin v17.4s, v17.4s, v1.4s
    fmin v18.4s, v18.4s, v1.4s
    fmin v19.4s, v19.4s, v1.4s
    fmin v20.4s, v20.4s, v1.4s
    fmin v21.4s, v21.4s, v1.4s
    fmin v22.4s, v22.4s, v1.4s
    fmin v23.4s, v23.4s, v1.4s
    fmin v24.4s, v24.4s, v1.4s
    fmin v25.4s, v25.4s, v1.4s
    fmin v26.4s, v26.4s, v1.4s
    fmin v27.4s, v27.4s, v1.4s
    fmin v28.4s, v28.4s, v1.4s
    fmin v29.4s, v29.4s, v1.4s
    fmin v30.4s, v30.4s, v1.4s
    fmin v31.4s, v31.4s, v1.4s
    fmax v8.4s, v8.4s, v0.4s
    fmax v9.4s, v9.4s, v0.4s
    fmax v10.4s, v10.4s, v0.4s
    fmax v11.4s, v11.4s, v0.4s
    fmax v12.4s, v12.4s, v0.4s
    fmax v13.4s, v13.4s, v0.4s
    fmax v14.4s, v14.4s, v0.4s
    fmax v15.4s, v15.4s, v0.4s
    fmax v16.4s, v16.4s, v0.4s
    fmax v17.4s, v17.4s, v0.4s
    fmax v18.4s, v18.4s, v0.4s
    fmax v19.4s, v19.4s, v0.4s
    fmax v20.4s, v20.4s, v0.4s
    fmax v21.4s, v21.4s, v0.4s
    fmax v22.4s, v22.4s, v0.4s
    fmax v23.4s, v23.4s, v0.4s
    fmax v24.4s, v24.4s, v0.4s
    fmax v25.4s, v25.4s, v0.4s
    fmax v26.4s, v26.4s, v0.4s
    fmax v27.4s, v27.4s, v0.4s
    fmax v28.4s, v28.4s, v0.4s
    fmax v29.4s, v29.4s, v0.4s
    fmax v30.4s, v30.4s, v0.4s
    fmax v31.4s, v31.4s, v0.4s
KAI_ASM_LABEL(label_188)  // Height 6: No activation
    cmp x8, #0x10
    bge label_197
    tbz x8, #3, label_192
    st1 { v8.4s }, [x16], #0x10
    st1 { v9.4s }, [x16], #0x10
    st1 { v12.4s }, [x26], #0x10
    st1 { v13.4s }, [x26], #0x10
    st1 { v16.4s }, [x25], #0x10
    st1 { v17.4s }, [x25], #0x10
    st1 { v20.4s }, [x24], #0x10
    st1 { v21.4s }, [x24], #0x10
    st1 { v24.4s }, [x23], #0x10
    st1 { v25.4s }, [x23], #0x10
    st1 { v28.4s }, [x22], #0x10
    st1 { v29.4s }, [x22], #0x10
    tbz x8, #2, label_190
    st1 { v10.4s }, [x16], #0x10
    st1 { v14.4s }, [x26], #0x10
    st1 { v18.4s }, [x25], #0x10
    st1 { v22.4s }, [x24], #0x10
    st1 { v26.4s }, [x23], #0x10
    st1 { v30.4s }, [x22], #0x10
    tbz x8, #1, label_189
    str d11, [x16], #0x8
    str d15, [x26], #0x8
    str d19, [x25], #0x8
    str d23, [x24], #0x8
    str d27, [x23], #0x8
    str d31, [x22], #0x8
    tbz x8, #0, label_196
    st1 { v11.s }[2], [x16]
    st1 { v15.s }[2], [x26]
    st1 { v19.s }[2], [x25]
    st1 { v23.s }[2], [x24]
    st1 { v27.s }[2], [x23]
    st1 { v31.s }[2], [x22]
    b label_196
KAI_ASM_LABEL(label_189)  // Height 6: Partial direct writeback: partial_1_12
    tbz x8, #0, label_196
    str s11, [x16, #0x0]
    str s15, [x26, #0x0]
    str s19, [x25, #0x0]
    str s23, [x24, #0x0]
    str s27, [x23, #0x0]
    str s31, [x22, #0x0]
    b label_196
KAI_ASM_LABEL(label_190)  // Height 6: Partial direct writeback: partial_2_8
    tbz x8, #1, label_191
    str d10, [x16], #0x8
    str d14, [x26], #0x8
    str d18, [x25], #0x8
    str d22, [x24], #0x8
    str d26, [x23], #0x8
    str d30, [x22], #0x8
    tbz x8, #0, label_196
    st1 { v10.s }[2], [x16]
    st1 { v14.s }[2], [x26]
    st1 { v18.s }[2], [x25]
    st1 { v22.s }[2], [x24]
    st1 { v26.s }[2], [x23]
    st1 { v30.s }[2], [x22]
    b label_196
KAI_ASM_LABEL(label_191)  // Height 6: Partial direct writeback: partial_1_8
    tbz x8, #0, label_196
    str s10, [x16, #0x0]
    str s14, [x26, #0x0]
    str s18, [x25, #0x0]
    str s22, [x24, #0x0]
    str s26, [x23, #0x0]
    str s30, [x22, #0x0]
    b label_196
KAI_ASM_LABEL(label_192)  // Height 6: Partial direct writeback: partial_4_0
    tbz x8, #2, label_194
    st1 { v8.4s }, [x16], #0x10
    st1 { v12.4s }, [x26], #0x10
    st1 { v16.4s }, [x25], #0x10
    st1 { v20.4s }, [x24], #0x10
    st1 { v24.4s }, [x23], #0x10
    st1 { v28.4s }, [x22], #0x10
    tbz x8, #1, label_193
    str d9, [x16], #0x8
    str d13, [x26], #0x8
    str d17, [x25], #0x8
    str d21, [x24], #0x8
    str d25, [x23], #0x8
    str d29, [x22], #0x8
    tbz x8, #0, label_196
    st1 { v9.s }[2], [x16]
    st1 { v13.s }[2], [x26]
    st1 { v17.s }[2], [x25]
    st1 { v21.s }[2], [x24]
    st1 { v25.s }[2], [x23]
    st1 { v29.s }[2], [x22]
    b label_196
KAI_ASM_LABEL(label_193)  // Height 6: Partial direct writeback: partial_1_4
    tbz x8, #0, label_196
    str s9, [x16, #0x0]
    str s13, [x26, #0x0]
    str s17, [x25, #0x0]
    str s21, [x24, #0x0]
    str s25, [x23, #0x0]
    str s29, [x22, #0x0]
    b label_196
KAI_ASM_LABEL(label_194)  // Height 6: Partial direct writeback: partial_2_0
    tbz x8, #1, label_195
    str d8, [x16], #0x8
    str d12, [x26], #0x8
    str d16, [x25], #0x8
    str d20, [x24], #0x8
    str d24, [x23], #0x8
    str d28, [x22], #0x8
    tbz x8, #0, label_196
    st1 { v8.s }[2], [x16]
    st1 { v12.s }[2], [x26]
    st1 { v16.s }[2], [x25]
    st1 { v20.s }[2], [x24]
    st1 { v24.s }[2], [x23]
    st1 { v28.s }[2], [x22]
    b label_196
KAI_ASM_LABEL(label_195)  // Height 6: Partial direct writeback: partial_1_0
    str s8, [x16, #0x0]
    str s12, [x26, #0x0]
    str s16, [x25, #0x0]
    str s20, [x24, #0x0]
    str s24, [x23, #0x0]
    str s28, [x22, #0x0]
KAI_ASM_LABEL(label_196)  // Height 6: Partial direct writeback: Done
    b label_198
KAI_ASM_LABEL(label_197)  // Height 6: Full writeback
    str q8, [x16, #0x0]
    str q9, [x16, #0x10]
    str q10, [x16, #0x20]
    str q11, [x16, #0x30]
    add x16, x16, #0x40
    str q12, [x26, #0x0]
    str q13, [x26, #0x10]
    str q14, [x26, #0x20]
    str q15, [x26, #0x30]
    str q16, [x25, #0x0]
    str q17, [x25, #0x10]
    str q18, [x25, #0x20]
    str q19, [x25, #0x30]
    str q20, [x24, #0x0]
    str q21, [x24, #0x10]
    str q22, [x24, #0x20]
    str q23, [x24, #0x30]
    str q24, [x23, #0x0]
    str q25, [x23, #0x10]
    str q26, [x23, #0x20]
    str q27, [x23, #0x30]
    str q28, [x22, #0x0]
    str q29, [x22, #0x10]
    str q30, [x22, #0x20]
    str q31, [x22, #0x30]
KAI_ASM_LABEL(label_198)  // Height 6: Writeback done
    subs x8, x8, #0x10
    bgt label_167
    subs x1, x1, #0x6
    beq label_200
    ldr x21, [x2, #0x38]
    tbz x3, #3, label_199
    add x21, x21, #0x6
    str x21, [x2, #0x38]
    b label_1
KAI_ASM_LABEL(label_199)  // Update direct input
    mov x20, #0x18
    madd x0, x20, x21, x0
    b label_1
KAI_ASM_LABEL(label_200)  // Exit
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp d8, d9, [sp, 72]
    ldp d10, d11, [sp, 88]
    ldp d12, d13, [sp, 104]
    ldp d14, d15, [sp, 120]
    ldp x20, x21, [sp], 144
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla_cortexa55)

    KAI_ASM_END
