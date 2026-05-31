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

    KAI_ASM_CODE(matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55)

    KAI_ASM_GLOBAL(kai_f16_from_float_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55)

KAI_ASM_FUNCTION_TYPE(kai_f16_from_float_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55)
KAI_ASM_FUNCTION_LABEL(kai_f16_from_float_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55)
    fcvt h0, s0
    fmov w0, h0
    ret
    KAI_ASM_FUNCTION_END(kai_f16_from_float_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55)
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
    bge label_246
    cmp x1, #0x4
    bgt label_197
    beq label_148
    cmp x1, #0x2
    bgt label_99
    beq label_50
    ldr x8, [x2, #0x10]
    ldr x17, [x2, #0x18]
    ldr x16, [x2, #0x38]
KAI_ASM_LABEL(label_2)  // Height 1: Column loop
    cbz x17, label_3
    ldr q8, [x17, #0x0]
    ldr q9, [x17, #0x10]
    ldr q10, [x17, #0x20]
    ldr q11, [x17, #0x30]
    add x17, x17, #0x40
    b label_22
KAI_ASM_LABEL(label_3)  // Height 1: no bias
    tbz x3, #0, label_21
    cmp x8, #0x20
    bge label_20
    tbz x8, #4, label_11
    ld1 { v8.8h }, [x16], #0x10
    ld1 { v9.8h }, [x16], #0x10
    tbz x8, #3, label_7
    ld1 { v10.8h }, [x16], #0x10
    tbz x8, #2, label_5
    ldr d11, [x16], #0x8
    tbz x8, #1, label_4
    ld1 { v11.s }[2], [x16], #0x4
    mov x20, #0x3c
    tbz x8, #0, label_19
    ld1 { v11.h }[6], [x16]
    b label_19
KAI_ASM_LABEL(label_4)  // Height 1: Partial accumulate: partial_1_28
    mov x20, #0x38
    tbz x8, #0, label_19
    ld1 { v11.h }[4], [x16]
    b label_19
KAI_ASM_LABEL(label_5)  // Height 1: Partial accumulate: partial_2_24
    tbz x8, #1, label_6
    ldr s11, [x16], #0x4
    mov x20, #0x34
    tbz x8, #0, label_19
    ld1 { v11.h }[2], [x16]
    b label_19
KAI_ASM_LABEL(label_6)  // Height 1: Partial accumulate: partial_1_24
    mov x20, #0x30
    tbz x8, #0, label_19
    ldr h11, [x16, #0x0]
    b label_19
KAI_ASM_LABEL(label_7)  // Height 1: Partial accumulate: partial_4_16
    tbz x8, #2, label_9
    ldr d10, [x16], #0x8
    tbz x8, #1, label_8
    ld1 { v10.s }[2], [x16], #0x4
    mov x20, #0x2c
    tbz x8, #0, label_19
    ld1 { v10.h }[6], [x16]
    b label_19
KAI_ASM_LABEL(label_8)  // Height 1: Partial accumulate: partial_1_20
    mov x20, #0x28
    tbz x8, #0, label_19
    ld1 { v10.h }[4], [x16]
    b label_19
KAI_ASM_LABEL(label_9)  // Height 1: Partial accumulate: partial_2_16
    tbz x8, #1, label_10
    ldr s10, [x16], #0x4
    mov x20, #0x24
    tbz x8, #0, label_19
    ld1 { v10.h }[2], [x16]
    b label_19
KAI_ASM_LABEL(label_10)  // Height 1: Partial accumulate: partial_1_16
    mov x20, #0x20
    tbz x8, #0, label_19
    ldr h10, [x16, #0x0]
    b label_19
KAI_ASM_LABEL(label_11)  // Height 1: Partial accumulate: partial_8_0
    tbz x8, #3, label_15
    ld1 { v8.8h }, [x16], #0x10
    tbz x8, #2, label_13
    ldr d9, [x16], #0x8
    tbz x8, #1, label_12
    ld1 { v9.s }[2], [x16], #0x4
    mov x20, #0x1c
    tbz x8, #0, label_19
    ld1 { v9.h }[6], [x16]
    b label_19
KAI_ASM_LABEL(label_12)  // Height 1: Partial accumulate: partial_1_12
    mov x20, #0x18
    tbz x8, #0, label_19
    ld1 { v9.h }[4], [x16]
    b label_19
KAI_ASM_LABEL(label_13)  // Height 1: Partial accumulate: partial_2_8
    tbz x8, #1, label_14
    ldr s9, [x16], #0x4
    mov x20, #0x14
    tbz x8, #0, label_19
    ld1 { v9.h }[2], [x16]
    b label_19
KAI_ASM_LABEL(label_14)  // Height 1: Partial accumulate: partial_1_8
    mov x20, #0x10
    tbz x8, #0, label_19
    ldr h9, [x16, #0x0]
    b label_19
KAI_ASM_LABEL(label_15)  // Height 1: Partial accumulate: partial_4_0
    tbz x8, #2, label_17
    ldr d8, [x16], #0x8
    tbz x8, #1, label_16
    ld1 { v8.s }[2], [x16], #0x4
    mov x20, #0xc
    tbz x8, #0, label_19
    ld1 { v8.h }[6], [x16]
    b label_19
KAI_ASM_LABEL(label_16)  // Height 1: Partial accumulate: partial_1_4
    mov x20, #0x8
    tbz x8, #0, label_19
    ld1 { v8.h }[4], [x16]
    b label_19
KAI_ASM_LABEL(label_17)  // Height 1: Partial accumulate: partial_2_0
    tbz x8, #1, label_18
    ldr s8, [x16], #0x4
    mov x20, #0x4
    tbz x8, #0, label_19
    ld1 { v8.h }[2], [x16]
    b label_19
KAI_ASM_LABEL(label_18)  // Height 1: Partial accumulate: partial_1_0
    ldr h8, [x16, #0x0]
    mov x20, #0x0
KAI_ASM_LABEL(label_19)  // Height 1: Partial accumulate: Done
    sub x16, x16, x20
    b label_22
KAI_ASM_LABEL(label_20)  // Height 1: full accumulate
    ldr q8, [x16, #0x0]
    ldr q9, [x16, #0x10]
    ldr q10, [x16, #0x20]
    ldr q11, [x16, #0x30]
    b label_22
KAI_ASM_LABEL(label_21)  // Height 1: no accumulate
    movi v8.16b, #0x0
    movi v9.16b, #0x0
    movi v10.16b, #0x0
    movi v11.16b, #0x0
KAI_ASM_LABEL(label_22)  // Height 1: setup done
    mov x15, #0x0
KAI_ASM_LABEL(label_23)  // Height 1: String loop
    ldr x20, [x2, #0x8]
    ldr x21, [x2, #0x30]
    ldr w14, [x20, x15, LSL #0x2]
    tbz x3, #3, label_24
    ldr x20, [x0, x15, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x13, [x20, #0x0]
    cbnz x15, label_25
    ldr x20, [x2, #0x28]
    add x13, x13, x20, LSL #1
    b label_25
KAI_ASM_LABEL(label_24)  // Height 1: setup direct input
    mov x13, x0
KAI_ASM_LABEL(label_25)  // Height 1: input setup done
    cmp x14, #0x8
    blt label_28
    ldr q0, [x13, #0x0]
    cmp x14, #0x10
    ldr q6, [x17, #0x0]
    ldr q7, [x17, #0x10]
    blt label_27
KAI_ASM_LABEL(label_26)  // Height 1: Multiply loop: Main loop head
    fmla v8.8h, v6.8h, v0.h[0]
    ldr d6, [x17, #0x20]
    ldr x21, [x17, #0x28]
    fmla v9.8h, v7.8h, v0.h[0]
    ldr d7, [x17, #0x30]
    add x13, x13, #0x10
    ldr x20, [x17, #0x38]
    sub x14, x14, #0x8
    mov v6.d[1], x21
    ldr x21, [x17, #0x48]
    ldr x22, [x13, #0x8]
    cmp x14, #0x10
    mov v7.d[1], x20
    fmla v10.8h, v6.8h, v0.h[0]
    ldr d6, [x17, #0x40]
    fmla v11.8h, v7.8h, v0.h[0]
    ldr d7, [x17, #0x50]
    ldr x20, [x17, #0x58]
    mov v6.d[1], x21
    ldr x21, [x17, #0x68]
    prfm pldl1keep, [x13, #0x80]
    mov v7.d[1], x20
    fmla v8.8h, v6.8h, v0.h[1]
    ldr d6, [x17, #0x60]
    fmla v9.8h, v7.8h, v0.h[1]
    ldr d7, [x17, #0x70]
    ldr x20, [x17, #0x78]
    mov v6.d[1], x21
    ldr x21, [x17, #0x88]
    mov v7.d[1], x20
    fmla v10.8h, v6.8h, v0.h[1]
    ldr d6, [x17, #0x80]
    fmla v11.8h, v7.8h, v0.h[1]
    ldr d7, [x17, #0x90]
    ldr x20, [x17, #0x98]
    mov v6.d[1], x21
    ldr x21, [x17, #0xa8]
    mov v7.d[1], x20
    fmla v8.8h, v6.8h, v0.h[2]
    ldr d6, [x17, #0xa0]
    fmla v9.8h, v7.8h, v0.h[2]
    ldr d7, [x17, #0xb0]
    ldr x20, [x17, #0xb8]
    mov v6.d[1], x21
    ldr x21, [x17, #0xc8]
    mov v7.d[1], x20
    fmla v10.8h, v6.8h, v0.h[2]
    ldr d6, [x17, #0xc0]
    fmla v11.8h, v7.8h, v0.h[2]
    ldr d7, [x17, #0xd0]
    ldr x20, [x17, #0xd8]
    mov v6.d[1], x21
    ldr x21, [x17, #0xe8]
    mov v7.d[1], x20
    fmla v8.8h, v6.8h, v0.h[3]
    ldr d6, [x17, #0xe0]
    fmla v9.8h, v7.8h, v0.h[3]
    ldr d7, [x17, #0xf0]
    ldr x20, [x17, #0xf8]
    mov v6.d[1], x21
    ldr x21, [x17, #0x108]
    mov v7.d[1], x20
    fmla v10.8h, v6.8h, v0.h[3]
    ldr d6, [x17, #0x100]
    fmla v11.8h, v7.8h, v0.h[3]
    ldr d7, [x17, #0x110]
    ldr x20, [x17, #0x118]
    mov v6.d[1], x21
    ldr x21, [x17, #0x128]
    mov v7.d[1], x20
    fmla v8.8h, v6.8h, v0.h[4]
    ldr d6, [x17, #0x120]
    fmla v9.8h, v7.8h, v0.h[4]
    ldr d7, [x17, #0x130]
    ldr x20, [x17, #0x138]
    mov v6.d[1], x21
    ldr x21, [x17, #0x148]
    mov v7.d[1], x20
    fmla v10.8h, v6.8h, v0.h[4]
    ldr d6, [x17, #0x140]
    fmla v11.8h, v7.8h, v0.h[4]
    ldr d7, [x17, #0x150]
    ldr x20, [x17, #0x158]
    mov v6.d[1], x21
    ldr x21, [x17, #0x168]
    mov v7.d[1], x20
    fmla v8.8h, v6.8h, v0.h[5]
    ldr d6, [x17, #0x160]
    fmla v9.8h, v7.8h, v0.h[5]
    ldr d7, [x17, #0x170]
    ldr x20, [x17, #0x178]
    mov v6.d[1], x21
    ldr x21, [x17, #0x188]
    mov v7.d[1], x20
    fmla v10.8h, v6.8h, v0.h[5]
    ldr d6, [x17, #0x180]
    fmla v11.8h, v7.8h, v0.h[5]
    ldr d7, [x17, #0x190]
    ldr x20, [x17, #0x198]
    mov v6.d[1], x21
    ldr x21, [x17, #0x1a8]
    mov v7.d[1], x20
    fmla v8.8h, v6.8h, v0.h[6]
    ldr d6, [x17, #0x1a0]
    fmla v9.8h, v7.8h, v0.h[6]
    ldr d7, [x17, #0x1b0]
    ldr x20, [x17, #0x1b8]
    mov v6.d[1], x21
    ldr x21, [x17, #0x1c8]
    mov v7.d[1], x20
    fmla v10.8h, v6.8h, v0.h[6]
    ldr d6, [x17, #0x1c0]
    fmla v11.8h, v7.8h, v0.h[6]
    ldr d7, [x17, #0x1d0]
    ldr x20, [x17, #0x1d8]
    mov v6.d[1], x21
    ldr x21, [x17, #0x1e8]
    mov v7.d[1], x20
    fmla v8.8h, v6.8h, v0.h[7]
    ldr d6, [x17, #0x1e0]
    fmla v9.8h, v7.8h, v0.h[7]
    ldr d7, [x17, #0x1f0]
    ldr x20, [x17, #0x1f8]
    add x17, x17, #0x200
    mov v6.d[1], x21
    ldr x21, [x17, #0x8]
    mov v7.d[1], x20
    fmla v10.8h, v6.8h, v0.h[7]
    ldr d6, [x17, #0x0]
    fmla v11.8h, v7.8h, v0.h[7]
    ldr d0, [x13, #0x0]
    ldr d7, [x17, #0x10]
    ldr x20, [x17, #0x18]
    mov v6.d[1], x21
    mov v0.d[1], x22
    mov v7.d[1], x20
    bge label_26
KAI_ASM_LABEL(label_27)  // Height 1: Multiply loop: Single iteration only
    fmla v8.8h, v6.8h, v0.h[0]
    ldr q6, [x17, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    ldr q7, [x17, #0x30]
    add x13, x13, #0x10
    sub x14, x14, #0x8
    fmla v10.8h, v6.8h, v0.h[0]
    ldr q6, [x17, #0x40]
    fmla v11.8h, v7.8h, v0.h[0]
    ldr q7, [x17, #0x50]
    prfm pldl1keep, [x13, #0x80]
    fmla v8.8h, v6.8h, v0.h[1]
    ldr q6, [x17, #0x60]
    fmla v9.8h, v7.8h, v0.h[1]
    ldr q7, [x17, #0x70]
    fmla v10.8h, v6.8h, v0.h[1]
    ldr q6, [x17, #0x80]
    fmla v11.8h, v7.8h, v0.h[1]
    ldr q7, [x17, #0x90]
    fmla v8.8h, v6.8h, v0.h[2]
    ldr q6, [x17, #0xa0]
    fmla v9.8h, v7.8h, v0.h[2]
    ldr q7, [x17, #0xb0]
    fmla v10.8h, v6.8h, v0.h[2]
    ldr q6, [x17, #0xc0]
    fmla v11.8h, v7.8h, v0.h[2]
    ldr q7, [x17, #0xd0]
    fmla v8.8h, v6.8h, v0.h[3]
    ldr q6, [x17, #0xe0]
    fmla v9.8h, v7.8h, v0.h[3]
    ldr q7, [x17, #0xf0]
    fmla v10.8h, v6.8h, v0.h[3]
    ldr q6, [x17, #0x100]
    fmla v11.8h, v7.8h, v0.h[3]
    ldr q7, [x17, #0x110]
    fmla v8.8h, v6.8h, v0.h[4]
    ldr q6, [x17, #0x120]
    fmla v9.8h, v7.8h, v0.h[4]
    ldr q7, [x17, #0x130]
    fmla v10.8h, v6.8h, v0.h[4]
    ldr q6, [x17, #0x140]
    fmla v11.8h, v7.8h, v0.h[4]
    ldr q7, [x17, #0x150]
    fmla v8.8h, v6.8h, v0.h[5]
    ldr q6, [x17, #0x160]
    fmla v9.8h, v7.8h, v0.h[5]
    ldr q7, [x17, #0x170]
    fmla v10.8h, v6.8h, v0.h[5]
    ldr q6, [x17, #0x180]
    fmla v11.8h, v7.8h, v0.h[5]
    ldr q7, [x17, #0x190]
    fmla v8.8h, v6.8h, v0.h[6]
    ldr q6, [x17, #0x1a0]
    fmla v9.8h, v7.8h, v0.h[6]
    ldr q7, [x17, #0x1b0]
    fmla v10.8h, v6.8h, v0.h[6]
    ldr q6, [x17, #0x1c0]
    fmla v11.8h, v7.8h, v0.h[6]
    ldr q7, [x17, #0x1d0]
    fmla v8.8h, v6.8h, v0.h[7]
    ldr q6, [x17, #0x1e0]
    fmla v9.8h, v7.8h, v0.h[7]
    ldr q7, [x17, #0x1f0]
    add x17, x17, #0x200
    fmla v10.8h, v6.8h, v0.h[7]
    fmla v11.8h, v7.8h, v0.h[7]
KAI_ASM_LABEL(label_28)  // Height 1: Multiply loop: Main loop skip
    cbz x14, label_30
KAI_ASM_LABEL(label_29)  // Height 1: Multiply loop: Odd block loop
    ldr h0, [x13], #0x2
    sub x14, x14, #0x1
    ldr q6, [x17, #0x0]
    ldr q7, [x17, #0x10]
    fmla v8.8h, v6.8h, v0.h[0]
    ldr q6, [x17, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    ldr q7, [x17, #0x30]
    add x17, x17, #0x40
    fmla v10.8h, v6.8h, v0.h[0]
    fmla v11.8h, v7.8h, v0.h[0]
    cbnz x14, label_29
KAI_ASM_LABEL(label_30)  // Height 1: Multiply loop: No odd multiplies
    ldr w20, [x2, #0x4]
    add x15, x15, #0x1
    cmp x15, x20
    bne label_23
    prfm pstl1keep, [x16, #0x0]
    tbz x3, #1, label_31
    add x21, x2, #0x0
    add x20, x2, #0x2
    ld1r { v17.8h }, [x21]
    ld1r { v16.8h }, [x20]
    fmin v8.8h, v8.8h, v17.8h
    fmin v9.8h, v9.8h, v17.8h
    fmin v10.8h, v10.8h, v17.8h
    fmin v11.8h, v11.8h, v17.8h
    fmax v8.8h, v8.8h, v16.8h
    fmax v9.8h, v9.8h, v16.8h
    fmax v10.8h, v10.8h, v16.8h
    fmax v11.8h, v11.8h, v16.8h
KAI_ASM_LABEL(label_31)  // Height 1: No activation
    cmp x8, #0x20
    bge label_48
    tbz x8, #4, label_39
    st1 { v8.8h }, [x16], #0x10
    st1 { v9.8h }, [x16], #0x10
    tbz x8, #3, label_35
    st1 { v10.8h }, [x16], #0x10
    tbz x8, #2, label_33
    str d11, [x16], #0x8
    tbz x8, #1, label_32
    st1 { v11.s }[2], [x16], #0x4
    tbz x8, #0, label_47
    st1 { v11.h }[6], [x16]
    b label_47
KAI_ASM_LABEL(label_32)  // Height 1: Partial direct writeback: partial_1_28
    tbz x8, #0, label_47
    st1 { v11.h }[4], [x16]
    b label_47
KAI_ASM_LABEL(label_33)  // Height 1: Partial direct writeback: partial_2_24
    tbz x8, #1, label_34
    str s11, [x16], #0x4
    tbz x8, #0, label_47
    st1 { v11.h }[2], [x16]
    b label_47
KAI_ASM_LABEL(label_34)  // Height 1: Partial direct writeback: partial_1_24
    tbz x8, #0, label_47
    str h11, [x16, #0x0]
    b label_47
KAI_ASM_LABEL(label_35)  // Height 1: Partial direct writeback: partial_4_16
    tbz x8, #2, label_37
    str d10, [x16], #0x8
    tbz x8, #1, label_36
    st1 { v10.s }[2], [x16], #0x4
    tbz x8, #0, label_47
    st1 { v10.h }[6], [x16]
    b label_47
KAI_ASM_LABEL(label_36)  // Height 1: Partial direct writeback: partial_1_20
    tbz x8, #0, label_47
    st1 { v10.h }[4], [x16]
    b label_47
KAI_ASM_LABEL(label_37)  // Height 1: Partial direct writeback: partial_2_16
    tbz x8, #1, label_38
    str s10, [x16], #0x4
    tbz x8, #0, label_47
    st1 { v10.h }[2], [x16]
    b label_47
KAI_ASM_LABEL(label_38)  // Height 1: Partial direct writeback: partial_1_16
    tbz x8, #0, label_47
    str h10, [x16, #0x0]
    b label_47
KAI_ASM_LABEL(label_39)  // Height 1: Partial direct writeback: partial_8_0
    tbz x8, #3, label_43
    st1 { v8.8h }, [x16], #0x10
    tbz x8, #2, label_41
    str d9, [x16], #0x8
    tbz x8, #1, label_40
    st1 { v9.s }[2], [x16], #0x4
    tbz x8, #0, label_47
    st1 { v9.h }[6], [x16]
    b label_47
KAI_ASM_LABEL(label_40)  // Height 1: Partial direct writeback: partial_1_12
    tbz x8, #0, label_47
    st1 { v9.h }[4], [x16]
    b label_47
KAI_ASM_LABEL(label_41)  // Height 1: Partial direct writeback: partial_2_8
    tbz x8, #1, label_42
    str s9, [x16], #0x4
    tbz x8, #0, label_47
    st1 { v9.h }[2], [x16]
    b label_47
KAI_ASM_LABEL(label_42)  // Height 1: Partial direct writeback: partial_1_8
    tbz x8, #0, label_47
    str h9, [x16, #0x0]
    b label_47
KAI_ASM_LABEL(label_43)  // Height 1: Partial direct writeback: partial_4_0
    tbz x8, #2, label_45
    str d8, [x16], #0x8
    tbz x8, #1, label_44
    st1 { v8.s }[2], [x16], #0x4
    tbz x8, #0, label_47
    st1 { v8.h }[6], [x16]
    b label_47
KAI_ASM_LABEL(label_44)  // Height 1: Partial direct writeback: partial_1_4
    tbz x8, #0, label_47
    st1 { v8.h }[4], [x16]
    b label_47
KAI_ASM_LABEL(label_45)  // Height 1: Partial direct writeback: partial_2_0
    tbz x8, #1, label_46
    str s8, [x16], #0x4
    tbz x8, #0, label_47
    st1 { v8.h }[2], [x16]
    b label_47
KAI_ASM_LABEL(label_46)  // Height 1: Partial direct writeback: partial_1_0
    str h8, [x16, #0x0]
KAI_ASM_LABEL(label_47)  // Height 1: Partial direct writeback: Done
    b label_49
KAI_ASM_LABEL(label_48)  // Height 1: Full writeback
    str q8, [x16, #0x0]
    str q9, [x16, #0x10]
    str q10, [x16, #0x20]
    str q11, [x16, #0x30]
    add x16, x16, #0x40
KAI_ASM_LABEL(label_49)  // Height 1: Writeback done
    subs x8, x8, #0x20
    bgt label_2
    b label_296
KAI_ASM_LABEL(label_50)  // Height 2
    ldr x8, [x2, #0x10]
    ldr x17, [x2, #0x18]
    ldr x16, [x2, #0x38]
KAI_ASM_LABEL(label_51)  // Height 2: Column loop
    cbz x17, label_52
    ldr q8, [x17, #0x0]
    ldr q9, [x17, #0x10]
    ldr q10, [x17, #0x20]
    ldr q11, [x17, #0x30]
    mov v12.16b, v8.16b
    mov v13.16b, v9.16b
    add x17, x17, #0x40
    mov v14.16b, v10.16b
    mov v15.16b, v11.16b
    b label_71
KAI_ASM_LABEL(label_52)  // Height 2: no bias
    tbz x3, #0, label_70
    ldr x20, [x2, #0x20]
    cmp x8, #0x20
    add x26, x16, x20, LSL #1
    bge label_69
    tbz x8, #4, label_60
    ld1 { v8.8h }, [x16], #0x10
    ld1 { v12.8h }, [x26], #0x10
    ld1 { v9.8h }, [x16], #0x10
    ld1 { v13.8h }, [x26], #0x10
    tbz x8, #3, label_56
    ld1 { v10.8h }, [x16], #0x10
    ld1 { v14.8h }, [x26], #0x10
    tbz x8, #2, label_54
    ldr d11, [x16], #0x8
    ldr d15, [x26], #0x8
    tbz x8, #1, label_53
    ld1 { v11.s }[2], [x16], #0x4
    mov x20, #0x3c
    ld1 { v15.s }[2], [x26], #0x4
    tbz x8, #0, label_68
    ld1 { v11.h }[6], [x16]
    ld1 { v15.h }[6], [x26]
    b label_68
KAI_ASM_LABEL(label_53)  // Height 2: Partial accumulate: partial_1_28
    mov x20, #0x38
    tbz x8, #0, label_68
    ld1 { v11.h }[4], [x16]
    ld1 { v15.h }[4], [x26]
    b label_68
KAI_ASM_LABEL(label_54)  // Height 2: Partial accumulate: partial_2_24
    tbz x8, #1, label_55
    ldr s11, [x16], #0x4
    mov x20, #0x34
    ldr s15, [x26], #0x4
    tbz x8, #0, label_68
    ld1 { v11.h }[2], [x16]
    ld1 { v15.h }[2], [x26]
    b label_68
KAI_ASM_LABEL(label_55)  // Height 2: Partial accumulate: partial_1_24
    mov x20, #0x30
    tbz x8, #0, label_68
    ldr h11, [x16, #0x0]
    ldr h15, [x26, #0x0]
    b label_68
KAI_ASM_LABEL(label_56)  // Height 2: Partial accumulate: partial_4_16
    tbz x8, #2, label_58
    ldr d10, [x16], #0x8
    ldr d14, [x26], #0x8
    tbz x8, #1, label_57
    ld1 { v10.s }[2], [x16], #0x4
    mov x20, #0x2c
    ld1 { v14.s }[2], [x26], #0x4
    tbz x8, #0, label_68
    ld1 { v10.h }[6], [x16]
    ld1 { v14.h }[6], [x26]
    b label_68
KAI_ASM_LABEL(label_57)  // Height 2: Partial accumulate: partial_1_20
    mov x20, #0x28
    tbz x8, #0, label_68
    ld1 { v10.h }[4], [x16]
    ld1 { v14.h }[4], [x26]
    b label_68
KAI_ASM_LABEL(label_58)  // Height 2: Partial accumulate: partial_2_16
    tbz x8, #1, label_59
    ldr s10, [x16], #0x4
    mov x20, #0x24
    ldr s14, [x26], #0x4
    tbz x8, #0, label_68
    ld1 { v10.h }[2], [x16]
    ld1 { v14.h }[2], [x26]
    b label_68
KAI_ASM_LABEL(label_59)  // Height 2: Partial accumulate: partial_1_16
    mov x20, #0x20
    tbz x8, #0, label_68
    ldr h10, [x16, #0x0]
    ldr h14, [x26, #0x0]
    b label_68
KAI_ASM_LABEL(label_60)  // Height 2: Partial accumulate: partial_8_0
    tbz x8, #3, label_64
    ld1 { v8.8h }, [x16], #0x10
    ld1 { v12.8h }, [x26], #0x10
    tbz x8, #2, label_62
    ldr d9, [x16], #0x8
    ldr d13, [x26], #0x8
    tbz x8, #1, label_61
    ld1 { v9.s }[2], [x16], #0x4
    mov x20, #0x1c
    ld1 { v13.s }[2], [x26], #0x4
    tbz x8, #0, label_68
    ld1 { v9.h }[6], [x16]
    ld1 { v13.h }[6], [x26]
    b label_68
KAI_ASM_LABEL(label_61)  // Height 2: Partial accumulate: partial_1_12
    mov x20, #0x18
    tbz x8, #0, label_68
    ld1 { v9.h }[4], [x16]
    ld1 { v13.h }[4], [x26]
    b label_68
KAI_ASM_LABEL(label_62)  // Height 2: Partial accumulate: partial_2_8
    tbz x8, #1, label_63
    ldr s9, [x16], #0x4
    mov x20, #0x14
    ldr s13, [x26], #0x4
    tbz x8, #0, label_68
    ld1 { v9.h }[2], [x16]
    ld1 { v13.h }[2], [x26]
    b label_68
KAI_ASM_LABEL(label_63)  // Height 2: Partial accumulate: partial_1_8
    mov x20, #0x10
    tbz x8, #0, label_68
    ldr h9, [x16, #0x0]
    ldr h13, [x26, #0x0]
    b label_68
KAI_ASM_LABEL(label_64)  // Height 2: Partial accumulate: partial_4_0
    tbz x8, #2, label_66
    ldr d8, [x16], #0x8
    ldr d12, [x26], #0x8
    tbz x8, #1, label_65
    ld1 { v8.s }[2], [x16], #0x4
    mov x20, #0xc
    ld1 { v12.s }[2], [x26], #0x4
    tbz x8, #0, label_68
    ld1 { v8.h }[6], [x16]
    ld1 { v12.h }[6], [x26]
    b label_68
KAI_ASM_LABEL(label_65)  // Height 2: Partial accumulate: partial_1_4
    mov x20, #0x8
    tbz x8, #0, label_68
    ld1 { v8.h }[4], [x16]
    ld1 { v12.h }[4], [x26]
    b label_68
KAI_ASM_LABEL(label_66)  // Height 2: Partial accumulate: partial_2_0
    tbz x8, #1, label_67
    ldr s8, [x16], #0x4
    mov x20, #0x4
    ldr s12, [x26], #0x4
    tbz x8, #0, label_68
    ld1 { v8.h }[2], [x16]
    ld1 { v12.h }[2], [x26]
    b label_68
KAI_ASM_LABEL(label_67)  // Height 2: Partial accumulate: partial_1_0
    ldr h8, [x16, #0x0]
    mov x20, #0x0
    ldr h12, [x26, #0x0]
KAI_ASM_LABEL(label_68)  // Height 2: Partial accumulate: Done
    sub x16, x16, x20
    b label_71
KAI_ASM_LABEL(label_69)  // Height 2: full accumulate
    ldr q8, [x16, #0x0]
    ldr q9, [x16, #0x10]
    ldr q10, [x16, #0x20]
    ldr q11, [x16, #0x30]
    ldr q12, [x26, #0x0]
    ldr q13, [x26, #0x10]
    ldr q14, [x26, #0x20]
    ldr q15, [x26, #0x30]
    b label_71
KAI_ASM_LABEL(label_70)  // Height 2: no accumulate
    movi v8.16b, #0x0
    movi v9.16b, #0x0
    movi v10.16b, #0x0
    movi v11.16b, #0x0
    movi v12.16b, #0x0
    movi v13.16b, #0x0
    movi v14.16b, #0x0
    movi v15.16b, #0x0
KAI_ASM_LABEL(label_71)  // Height 2: setup done
    mov x15, #0x0
KAI_ASM_LABEL(label_72)  // Height 2: String loop
    ldr x20, [x2, #0x8]
    ldr x21, [x2, #0x30]
    ldr w14, [x20, x15, LSL #0x2]
    tbz x3, #3, label_73
    ldr x20, [x0, x15, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x13, [x20, #0x0]
    ldr x12, [x20, #0x8]
    cbnz x15, label_74
    ldr x20, [x2, #0x28]
    add x13, x13, x20, LSL #1
    add x12, x12, x20, LSL #1
    b label_74
KAI_ASM_LABEL(label_73)  // Height 2: setup direct input
    mov x13, x0
    add x12, x13, x21, LSL #1
KAI_ASM_LABEL(label_74)  // Height 2: input setup done
    cmp x14, #0x8
    blt label_77
    ldr q0, [x13, #0x0]
    cmp x14, #0x10
    ldr q1, [x12, #0x0]
    ldr q6, [x17, #0x0]
    ldr q7, [x17, #0x10]
    blt label_76
KAI_ASM_LABEL(label_75)  // Height 2: Multiply loop: Main loop head
    fmla v8.8h, v6.8h, v0.h[0]
    ldr x20, [x17, #0x28]
    fmla v12.8h, v6.8h, v1.h[0]
    ldr d6, [x17, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    ldr x21, [x17, #0x38]
    fmla v13.8h, v7.8h, v1.h[0]
    ldr d7, [x17, #0x30]
    mov v6.d[1], x20
    ldr x20, [x17, #0x48]
    add x13, x13, #0x10
    add x12, x12, #0x10
    mov v7.d[1], x21
    fmla v10.8h, v6.8h, v0.h[0]
    fmla v14.8h, v6.8h, v1.h[0]
    ldr d6, [x17, #0x40]
    fmla v11.8h, v7.8h, v0.h[0]
    ldr x21, [x17, #0x58]
    fmla v15.8h, v7.8h, v1.h[0]
    ldr d7, [x17, #0x50]
    mov v6.d[1], x20
    ldr x20, [x17, #0x68]
    ldr x23, [x13, #0x8]
    sub x14, x14, #0x8
    mov v7.d[1], x21
    fmla v8.8h, v6.8h, v0.h[1]
    fmla v12.8h, v6.8h, v1.h[1]
    ldr d6, [x17, #0x60]
    fmla v9.8h, v7.8h, v0.h[1]
    ldr x21, [x17, #0x78]
    fmla v13.8h, v7.8h, v1.h[1]
    ldr d7, [x17, #0x70]
    mov v6.d[1], x20
    ldr x20, [x17, #0x88]
    ldr x22, [x12, #0x8]
    cmp x14, #0x10
    mov v7.d[1], x21
    fmla v10.8h, v6.8h, v0.h[1]
    fmla v14.8h, v6.8h, v1.h[1]
    ldr d6, [x17, #0x80]
    fmla v11.8h, v7.8h, v0.h[1]
    ldr x21, [x17, #0x98]
    fmla v15.8h, v7.8h, v1.h[1]
    ldr d7, [x17, #0x90]
    mov v6.d[1], x20
    ldr x20, [x17, #0xa8]
    prfm pldl1keep, [x13, #0x80]
    mov v7.d[1], x21
    fmla v8.8h, v6.8h, v0.h[2]
    fmla v12.8h, v6.8h, v1.h[2]
    ldr d6, [x17, #0xa0]
    fmla v9.8h, v7.8h, v0.h[2]
    ldr x21, [x17, #0xb8]
    fmla v13.8h, v7.8h, v1.h[2]
    ldr d7, [x17, #0xb0]
    mov v6.d[1], x20
    ldr x20, [x17, #0xc8]
    prfm pldl1keep, [x12, #0x80]
    mov v7.d[1], x21
    fmla v10.8h, v6.8h, v0.h[2]
    fmla v14.8h, v6.8h, v1.h[2]
    ldr d6, [x17, #0xc0]
    fmla v11.8h, v7.8h, v0.h[2]
    ldr x21, [x17, #0xd8]
    fmla v15.8h, v7.8h, v1.h[2]
    ldr d7, [x17, #0xd0]
    mov v6.d[1], x20
    ldr x20, [x17, #0xe8]
    mov v7.d[1], x21
    fmla v8.8h, v6.8h, v0.h[3]
    fmla v12.8h, v6.8h, v1.h[3]
    ldr d6, [x17, #0xe0]
    fmla v9.8h, v7.8h, v0.h[3]
    ldr x21, [x17, #0xf8]
    fmla v13.8h, v7.8h, v1.h[3]
    ldr d7, [x17, #0xf0]
    mov v6.d[1], x20
    ldr x20, [x17, #0x108]
    mov v7.d[1], x21
    fmla v10.8h, v6.8h, v0.h[3]
    fmla v14.8h, v6.8h, v1.h[3]
    ldr d6, [x17, #0x100]
    fmla v11.8h, v7.8h, v0.h[3]
    ldr x21, [x17, #0x118]
    fmla v15.8h, v7.8h, v1.h[3]
    ldr d7, [x17, #0x110]
    mov v6.d[1], x20
    ldr x20, [x17, #0x128]
    mov v7.d[1], x21
    fmla v8.8h, v6.8h, v0.h[4]
    fmla v12.8h, v6.8h, v1.h[4]
    ldr d6, [x17, #0x120]
    fmla v9.8h, v7.8h, v0.h[4]
    ldr x21, [x17, #0x138]
    fmla v13.8h, v7.8h, v1.h[4]
    ldr d7, [x17, #0x130]
    mov v6.d[1], x20
    ldr x20, [x17, #0x148]
    mov v7.d[1], x21
    fmla v10.8h, v6.8h, v0.h[4]
    fmla v14.8h, v6.8h, v1.h[4]
    ldr d6, [x17, #0x140]
    fmla v11.8h, v7.8h, v0.h[4]
    ldr x21, [x17, #0x158]
    fmla v15.8h, v7.8h, v1.h[4]
    ldr d7, [x17, #0x150]
    mov v6.d[1], x20
    ldr x20, [x17, #0x168]
    mov v7.d[1], x21
    fmla v8.8h, v6.8h, v0.h[5]
    fmla v12.8h, v6.8h, v1.h[5]
    ldr d6, [x17, #0x160]
    fmla v9.8h, v7.8h, v0.h[5]
    ldr x21, [x17, #0x178]
    fmla v13.8h, v7.8h, v1.h[5]
    ldr d7, [x17, #0x170]
    mov v6.d[1], x20
    ldr x20, [x17, #0x188]
    mov v7.d[1], x21
    fmla v10.8h, v6.8h, v0.h[5]
    fmla v14.8h, v6.8h, v1.h[5]
    ldr d6, [x17, #0x180]
    fmla v11.8h, v7.8h, v0.h[5]
    ldr x21, [x17, #0x198]
    fmla v15.8h, v7.8h, v1.h[5]
    ldr d7, [x17, #0x190]
    mov v6.d[1], x20
    ldr x20, [x17, #0x1a8]
    mov v7.d[1], x21
    fmla v8.8h, v6.8h, v0.h[6]
    fmla v12.8h, v6.8h, v1.h[6]
    ldr d6, [x17, #0x1a0]
    fmla v9.8h, v7.8h, v0.h[6]
    ldr x21, [x17, #0x1b8]
    fmla v13.8h, v7.8h, v1.h[6]
    ldr d7, [x17, #0x1b0]
    mov v6.d[1], x20
    ldr x20, [x17, #0x1c8]
    mov v7.d[1], x21
    fmla v10.8h, v6.8h, v0.h[6]
    fmla v14.8h, v6.8h, v1.h[6]
    ldr d6, [x17, #0x1c0]
    fmla v11.8h, v7.8h, v0.h[6]
    ldr x21, [x17, #0x1d8]
    fmla v15.8h, v7.8h, v1.h[6]
    ldr d7, [x17, #0x1d0]
    mov v6.d[1], x20
    ldr x20, [x17, #0x1e8]
    mov v7.d[1], x21
    fmla v8.8h, v6.8h, v0.h[7]
    fmla v12.8h, v6.8h, v1.h[7]
    ldr d6, [x17, #0x1e0]
    fmla v9.8h, v7.8h, v0.h[7]
    ldr x21, [x17, #0x1f8]
    fmla v13.8h, v7.8h, v1.h[7]
    ldr d7, [x17, #0x1f0]
    mov v6.d[1], x20
    add x17, x17, #0x200
    ldr x20, [x17, #0x8]
    mov v7.d[1], x21
    fmla v10.8h, v6.8h, v0.h[7]
    fmla v14.8h, v6.8h, v1.h[7]
    ldr d6, [x17, #0x0]
    fmla v11.8h, v7.8h, v0.h[7]
    ldr d0, [x13, #0x0]
    fmla v15.8h, v7.8h, v1.h[7]
    ldr d1, [x12, #0x0]
    ldr d7, [x17, #0x10]
    mov v6.d[1], x20
    ldr x20, [x17, #0x18]
    mov v0.d[1], x23
    mov v1.d[1], x22
    mov v7.d[1], x20
    bge label_75
KAI_ASM_LABEL(label_76)  // Height 2: Multiply loop: Single iteration only
    fmla v8.8h, v6.8h, v0.h[0]
    add x13, x13, #0x10
    fmla v12.8h, v6.8h, v1.h[0]
    ldr q6, [x17, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    add x12, x12, #0x10
    fmla v13.8h, v7.8h, v1.h[0]
    ldr q7, [x17, #0x30]
    fmla v10.8h, v6.8h, v0.h[0]
    sub x14, x14, #0x8
    fmla v14.8h, v6.8h, v1.h[0]
    ldr q6, [x17, #0x40]
    fmla v11.8h, v7.8h, v0.h[0]
    prfm pldl1keep, [x13, #0x80]
    fmla v15.8h, v7.8h, v1.h[0]
    ldr q7, [x17, #0x50]
    fmla v8.8h, v6.8h, v0.h[1]
    prfm pldl1keep, [x12, #0x80]
    fmla v12.8h, v6.8h, v1.h[1]
    ldr q6, [x17, #0x60]
    fmla v9.8h, v7.8h, v0.h[1]
    fmla v13.8h, v7.8h, v1.h[1]
    ldr q7, [x17, #0x70]
    fmla v10.8h, v6.8h, v0.h[1]
    fmla v14.8h, v6.8h, v1.h[1]
    ldr q6, [x17, #0x80]
    fmla v11.8h, v7.8h, v0.h[1]
    fmla v15.8h, v7.8h, v1.h[1]
    ldr q7, [x17, #0x90]
    fmla v8.8h, v6.8h, v0.h[2]
    fmla v12.8h, v6.8h, v1.h[2]
    ldr q6, [x17, #0xa0]
    fmla v9.8h, v7.8h, v0.h[2]
    fmla v13.8h, v7.8h, v1.h[2]
    ldr q7, [x17, #0xb0]
    fmla v10.8h, v6.8h, v0.h[2]
    fmla v14.8h, v6.8h, v1.h[2]
    ldr q6, [x17, #0xc0]
    fmla v11.8h, v7.8h, v0.h[2]
    fmla v15.8h, v7.8h, v1.h[2]
    ldr q7, [x17, #0xd0]
    fmla v8.8h, v6.8h, v0.h[3]
    fmla v12.8h, v6.8h, v1.h[3]
    ldr q6, [x17, #0xe0]
    fmla v9.8h, v7.8h, v0.h[3]
    fmla v13.8h, v7.8h, v1.h[3]
    ldr q7, [x17, #0xf0]
    fmla v10.8h, v6.8h, v0.h[3]
    fmla v14.8h, v6.8h, v1.h[3]
    ldr q6, [x17, #0x100]
    fmla v11.8h, v7.8h, v0.h[3]
    fmla v15.8h, v7.8h, v1.h[3]
    ldr q7, [x17, #0x110]
    fmla v8.8h, v6.8h, v0.h[4]
    fmla v12.8h, v6.8h, v1.h[4]
    ldr q6, [x17, #0x120]
    fmla v9.8h, v7.8h, v0.h[4]
    fmla v13.8h, v7.8h, v1.h[4]
    ldr q7, [x17, #0x130]
    fmla v10.8h, v6.8h, v0.h[4]
    fmla v14.8h, v6.8h, v1.h[4]
    ldr q6, [x17, #0x140]
    fmla v11.8h, v7.8h, v0.h[4]
    fmla v15.8h, v7.8h, v1.h[4]
    ldr q7, [x17, #0x150]
    fmla v8.8h, v6.8h, v0.h[5]
    fmla v12.8h, v6.8h, v1.h[5]
    ldr q6, [x17, #0x160]
    fmla v9.8h, v7.8h, v0.h[5]
    fmla v13.8h, v7.8h, v1.h[5]
    ldr q7, [x17, #0x170]
    fmla v10.8h, v6.8h, v0.h[5]
    fmla v14.8h, v6.8h, v1.h[5]
    ldr q6, [x17, #0x180]
    fmla v11.8h, v7.8h, v0.h[5]
    fmla v15.8h, v7.8h, v1.h[5]
    ldr q7, [x17, #0x190]
    fmla v8.8h, v6.8h, v0.h[6]
    fmla v12.8h, v6.8h, v1.h[6]
    ldr q6, [x17, #0x1a0]
    fmla v9.8h, v7.8h, v0.h[6]
    fmla v13.8h, v7.8h, v1.h[6]
    ldr q7, [x17, #0x1b0]
    fmla v10.8h, v6.8h, v0.h[6]
    fmla v14.8h, v6.8h, v1.h[6]
    ldr q6, [x17, #0x1c0]
    fmla v11.8h, v7.8h, v0.h[6]
    fmla v15.8h, v7.8h, v1.h[6]
    ldr q7, [x17, #0x1d0]
    fmla v8.8h, v6.8h, v0.h[7]
    fmla v12.8h, v6.8h, v1.h[7]
    ldr q6, [x17, #0x1e0]
    fmla v9.8h, v7.8h, v0.h[7]
    fmla v13.8h, v7.8h, v1.h[7]
    ldr q7, [x17, #0x1f0]
    fmla v10.8h, v6.8h, v0.h[7]
    add x17, x17, #0x200
    fmla v14.8h, v6.8h, v1.h[7]
    fmla v11.8h, v7.8h, v0.h[7]
    fmla v15.8h, v7.8h, v1.h[7]
KAI_ASM_LABEL(label_77)  // Height 2: Multiply loop: Main loop skip
    cbz x14, label_79
KAI_ASM_LABEL(label_78)  // Height 2: Multiply loop: Odd block loop
    ldr h0, [x13], #0x2
    sub x14, x14, #0x1
    ldr h1, [x12], #0x2
    ldr q6, [x17, #0x0]
    ldr q7, [x17, #0x10]
    fmla v8.8h, v6.8h, v0.h[0]
    fmla v12.8h, v6.8h, v1.h[0]
    ldr q6, [x17, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    fmla v13.8h, v7.8h, v1.h[0]
    ldr q7, [x17, #0x30]
    fmla v10.8h, v6.8h, v0.h[0]
    add x17, x17, #0x40
    fmla v14.8h, v6.8h, v1.h[0]
    fmla v11.8h, v7.8h, v0.h[0]
    fmla v15.8h, v7.8h, v1.h[0]
    cbnz x14, label_78
KAI_ASM_LABEL(label_79)  // Height 2: Multiply loop: No odd multiplies
    ldr w20, [x2, #0x4]
    add x15, x15, #0x1
    cmp x15, x20
    bne label_72
    ldr x20, [x2, #0x20]
    prfm pstl1keep, [x16, #0x0]
    add x26, x16, x20, LSL #1
    prfm pstl1keep, [x26, #0x0]
    tbz x3, #1, label_80
    add x21, x2, #0x0
    add x20, x2, #0x2
    ld1r { v17.8h }, [x21]
    ld1r { v16.8h }, [x20]
    fmin v8.8h, v8.8h, v17.8h
    fmin v9.8h, v9.8h, v17.8h
    fmin v10.8h, v10.8h, v17.8h
    fmin v11.8h, v11.8h, v17.8h
    fmin v12.8h, v12.8h, v17.8h
    fmin v13.8h, v13.8h, v17.8h
    fmin v14.8h, v14.8h, v17.8h
    fmin v15.8h, v15.8h, v17.8h
    fmax v8.8h, v8.8h, v16.8h
    fmax v9.8h, v9.8h, v16.8h
    fmax v10.8h, v10.8h, v16.8h
    fmax v11.8h, v11.8h, v16.8h
    fmax v12.8h, v12.8h, v16.8h
    fmax v13.8h, v13.8h, v16.8h
    fmax v14.8h, v14.8h, v16.8h
    fmax v15.8h, v15.8h, v16.8h
KAI_ASM_LABEL(label_80)  // Height 2: No activation
    cmp x8, #0x20
    bge label_97
    tbz x8, #4, label_88
    st1 { v8.8h }, [x16], #0x10
    st1 { v9.8h }, [x16], #0x10
    st1 { v12.8h }, [x26], #0x10
    st1 { v13.8h }, [x26], #0x10
    tbz x8, #3, label_84
    st1 { v10.8h }, [x16], #0x10
    st1 { v14.8h }, [x26], #0x10
    tbz x8, #2, label_82
    str d11, [x16], #0x8
    str d15, [x26], #0x8
    tbz x8, #1, label_81
    st1 { v11.s }[2], [x16], #0x4
    st1 { v15.s }[2], [x26], #0x4
    tbz x8, #0, label_96
    st1 { v11.h }[6], [x16]
    st1 { v15.h }[6], [x26]
    b label_96
KAI_ASM_LABEL(label_81)  // Height 2: Partial direct writeback: partial_1_28
    tbz x8, #0, label_96
    st1 { v11.h }[4], [x16]
    st1 { v15.h }[4], [x26]
    b label_96
KAI_ASM_LABEL(label_82)  // Height 2: Partial direct writeback: partial_2_24
    tbz x8, #1, label_83
    str s11, [x16], #0x4
    str s15, [x26], #0x4
    tbz x8, #0, label_96
    st1 { v11.h }[2], [x16]
    st1 { v15.h }[2], [x26]
    b label_96
KAI_ASM_LABEL(label_83)  // Height 2: Partial direct writeback: partial_1_24
    tbz x8, #0, label_96
    str h11, [x16, #0x0]
    str h15, [x26, #0x0]
    b label_96
KAI_ASM_LABEL(label_84)  // Height 2: Partial direct writeback: partial_4_16
    tbz x8, #2, label_86
    str d10, [x16], #0x8
    str d14, [x26], #0x8
    tbz x8, #1, label_85
    st1 { v10.s }[2], [x16], #0x4
    st1 { v14.s }[2], [x26], #0x4
    tbz x8, #0, label_96
    st1 { v10.h }[6], [x16]
    st1 { v14.h }[6], [x26]
    b label_96
KAI_ASM_LABEL(label_85)  // Height 2: Partial direct writeback: partial_1_20
    tbz x8, #0, label_96
    st1 { v10.h }[4], [x16]
    st1 { v14.h }[4], [x26]
    b label_96
KAI_ASM_LABEL(label_86)  // Height 2: Partial direct writeback: partial_2_16
    tbz x8, #1, label_87
    str s10, [x16], #0x4
    str s14, [x26], #0x4
    tbz x8, #0, label_96
    st1 { v10.h }[2], [x16]
    st1 { v14.h }[2], [x26]
    b label_96
KAI_ASM_LABEL(label_87)  // Height 2: Partial direct writeback: partial_1_16
    tbz x8, #0, label_96
    str h10, [x16, #0x0]
    str h14, [x26, #0x0]
    b label_96
KAI_ASM_LABEL(label_88)  // Height 2: Partial direct writeback: partial_8_0
    tbz x8, #3, label_92
    st1 { v8.8h }, [x16], #0x10
    st1 { v12.8h }, [x26], #0x10
    tbz x8, #2, label_90
    str d9, [x16], #0x8
    str d13, [x26], #0x8
    tbz x8, #1, label_89
    st1 { v9.s }[2], [x16], #0x4
    st1 { v13.s }[2], [x26], #0x4
    tbz x8, #0, label_96
    st1 { v9.h }[6], [x16]
    st1 { v13.h }[6], [x26]
    b label_96
KAI_ASM_LABEL(label_89)  // Height 2: Partial direct writeback: partial_1_12
    tbz x8, #0, label_96
    st1 { v9.h }[4], [x16]
    st1 { v13.h }[4], [x26]
    b label_96
KAI_ASM_LABEL(label_90)  // Height 2: Partial direct writeback: partial_2_8
    tbz x8, #1, label_91
    str s9, [x16], #0x4
    str s13, [x26], #0x4
    tbz x8, #0, label_96
    st1 { v9.h }[2], [x16]
    st1 { v13.h }[2], [x26]
    b label_96
KAI_ASM_LABEL(label_91)  // Height 2: Partial direct writeback: partial_1_8
    tbz x8, #0, label_96
    str h9, [x16, #0x0]
    str h13, [x26, #0x0]
    b label_96
KAI_ASM_LABEL(label_92)  // Height 2: Partial direct writeback: partial_4_0
    tbz x8, #2, label_94
    str d8, [x16], #0x8
    str d12, [x26], #0x8
    tbz x8, #1, label_93
    st1 { v8.s }[2], [x16], #0x4
    st1 { v12.s }[2], [x26], #0x4
    tbz x8, #0, label_96
    st1 { v8.h }[6], [x16]
    st1 { v12.h }[6], [x26]
    b label_96
KAI_ASM_LABEL(label_93)  // Height 2: Partial direct writeback: partial_1_4
    tbz x8, #0, label_96
    st1 { v8.h }[4], [x16]
    st1 { v12.h }[4], [x26]
    b label_96
KAI_ASM_LABEL(label_94)  // Height 2: Partial direct writeback: partial_2_0
    tbz x8, #1, label_95
    str s8, [x16], #0x4
    str s12, [x26], #0x4
    tbz x8, #0, label_96
    st1 { v8.h }[2], [x16]
    st1 { v12.h }[2], [x26]
    b label_96
KAI_ASM_LABEL(label_95)  // Height 2: Partial direct writeback: partial_1_0
    str h8, [x16, #0x0]
    str h12, [x26, #0x0]
KAI_ASM_LABEL(label_96)  // Height 2: Partial direct writeback: Done
    b label_98
KAI_ASM_LABEL(label_97)  // Height 2: Full writeback
    str q8, [x16, #0x0]
    str q9, [x16, #0x10]
    str q10, [x16, #0x20]
    str q11, [x16, #0x30]
    add x16, x16, #0x40
    str q12, [x26, #0x0]
    str q13, [x26, #0x10]
    str q14, [x26, #0x20]
    str q15, [x26, #0x30]
KAI_ASM_LABEL(label_98)  // Height 2: Writeback done
    subs x8, x8, #0x20
    bgt label_51
    b label_296
KAI_ASM_LABEL(label_99)  // Height 3
    ldr x8, [x2, #0x10]
    ldr x17, [x2, #0x18]
    ldr x16, [x2, #0x38]
KAI_ASM_LABEL(label_100)  // Height 3: Column loop
    cbz x17, label_101
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
    b label_120
KAI_ASM_LABEL(label_101)  // Height 3: no bias
    tbz x3, #0, label_119
    ldr x20, [x2, #0x20]
    cmp x8, #0x20
    add x26, x16, x20, LSL #1
    add x25, x26, x20, LSL #1
    bge label_118
    tbz x8, #4, label_109
    ld1 { v8.8h }, [x16], #0x10
    ld1 { v12.8h }, [x26], #0x10
    ld1 { v16.8h }, [x25], #0x10
    ld1 { v9.8h }, [x16], #0x10
    ld1 { v13.8h }, [x26], #0x10
    ld1 { v17.8h }, [x25], #0x10
    tbz x8, #3, label_105
    ld1 { v10.8h }, [x16], #0x10
    ld1 { v14.8h }, [x26], #0x10
    ld1 { v18.8h }, [x25], #0x10
    tbz x8, #2, label_103
    ldr d11, [x16], #0x8
    ldr d15, [x26], #0x8
    ldr d19, [x25], #0x8
    tbz x8, #1, label_102
    ld1 { v11.s }[2], [x16], #0x4
    mov x20, #0x3c
    ld1 { v15.s }[2], [x26], #0x4
    ld1 { v19.s }[2], [x25], #0x4
    tbz x8, #0, label_117
    ld1 { v11.h }[6], [x16]
    ld1 { v15.h }[6], [x26]
    ld1 { v19.h }[6], [x25]
    b label_117
KAI_ASM_LABEL(label_102)  // Height 3: Partial accumulate: partial_1_28
    mov x20, #0x38
    tbz x8, #0, label_117
    ld1 { v11.h }[4], [x16]
    ld1 { v15.h }[4], [x26]
    ld1 { v19.h }[4], [x25]
    b label_117
KAI_ASM_LABEL(label_103)  // Height 3: Partial accumulate: partial_2_24
    tbz x8, #1, label_104
    ldr s11, [x16], #0x4
    mov x20, #0x34
    ldr s15, [x26], #0x4
    ldr s19, [x25], #0x4
    tbz x8, #0, label_117
    ld1 { v11.h }[2], [x16]
    ld1 { v15.h }[2], [x26]
    ld1 { v19.h }[2], [x25]
    b label_117
KAI_ASM_LABEL(label_104)  // Height 3: Partial accumulate: partial_1_24
    mov x20, #0x30
    tbz x8, #0, label_117
    ldr h11, [x16, #0x0]
    ldr h15, [x26, #0x0]
    ldr h19, [x25, #0x0]
    b label_117
KAI_ASM_LABEL(label_105)  // Height 3: Partial accumulate: partial_4_16
    tbz x8, #2, label_107
    ldr d10, [x16], #0x8
    ldr d14, [x26], #0x8
    ldr d18, [x25], #0x8
    tbz x8, #1, label_106
    ld1 { v10.s }[2], [x16], #0x4
    mov x20, #0x2c
    ld1 { v14.s }[2], [x26], #0x4
    ld1 { v18.s }[2], [x25], #0x4
    tbz x8, #0, label_117
    ld1 { v10.h }[6], [x16]
    ld1 { v14.h }[6], [x26]
    ld1 { v18.h }[6], [x25]
    b label_117
KAI_ASM_LABEL(label_106)  // Height 3: Partial accumulate: partial_1_20
    mov x20, #0x28
    tbz x8, #0, label_117
    ld1 { v10.h }[4], [x16]
    ld1 { v14.h }[4], [x26]
    ld1 { v18.h }[4], [x25]
    b label_117
KAI_ASM_LABEL(label_107)  // Height 3: Partial accumulate: partial_2_16
    tbz x8, #1, label_108
    ldr s10, [x16], #0x4
    mov x20, #0x24
    ldr s14, [x26], #0x4
    ldr s18, [x25], #0x4
    tbz x8, #0, label_117
    ld1 { v10.h }[2], [x16]
    ld1 { v14.h }[2], [x26]
    ld1 { v18.h }[2], [x25]
    b label_117
KAI_ASM_LABEL(label_108)  // Height 3: Partial accumulate: partial_1_16
    mov x20, #0x20
    tbz x8, #0, label_117
    ldr h10, [x16, #0x0]
    ldr h14, [x26, #0x0]
    ldr h18, [x25, #0x0]
    b label_117
KAI_ASM_LABEL(label_109)  // Height 3: Partial accumulate: partial_8_0
    tbz x8, #3, label_113
    ld1 { v8.8h }, [x16], #0x10
    ld1 { v12.8h }, [x26], #0x10
    ld1 { v16.8h }, [x25], #0x10
    tbz x8, #2, label_111
    ldr d9, [x16], #0x8
    ldr d13, [x26], #0x8
    ldr d17, [x25], #0x8
    tbz x8, #1, label_110
    ld1 { v9.s }[2], [x16], #0x4
    mov x20, #0x1c
    ld1 { v13.s }[2], [x26], #0x4
    ld1 { v17.s }[2], [x25], #0x4
    tbz x8, #0, label_117
    ld1 { v9.h }[6], [x16]
    ld1 { v13.h }[6], [x26]
    ld1 { v17.h }[6], [x25]
    b label_117
KAI_ASM_LABEL(label_110)  // Height 3: Partial accumulate: partial_1_12
    mov x20, #0x18
    tbz x8, #0, label_117
    ld1 { v9.h }[4], [x16]
    ld1 { v13.h }[4], [x26]
    ld1 { v17.h }[4], [x25]
    b label_117
KAI_ASM_LABEL(label_111)  // Height 3: Partial accumulate: partial_2_8
    tbz x8, #1, label_112
    ldr s9, [x16], #0x4
    mov x20, #0x14
    ldr s13, [x26], #0x4
    ldr s17, [x25], #0x4
    tbz x8, #0, label_117
    ld1 { v9.h }[2], [x16]
    ld1 { v13.h }[2], [x26]
    ld1 { v17.h }[2], [x25]
    b label_117
KAI_ASM_LABEL(label_112)  // Height 3: Partial accumulate: partial_1_8
    mov x20, #0x10
    tbz x8, #0, label_117
    ldr h9, [x16, #0x0]
    ldr h13, [x26, #0x0]
    ldr h17, [x25, #0x0]
    b label_117
KAI_ASM_LABEL(label_113)  // Height 3: Partial accumulate: partial_4_0
    tbz x8, #2, label_115
    ldr d8, [x16], #0x8
    ldr d12, [x26], #0x8
    ldr d16, [x25], #0x8
    tbz x8, #1, label_114
    ld1 { v8.s }[2], [x16], #0x4
    mov x20, #0xc
    ld1 { v12.s }[2], [x26], #0x4
    ld1 { v16.s }[2], [x25], #0x4
    tbz x8, #0, label_117
    ld1 { v8.h }[6], [x16]
    ld1 { v12.h }[6], [x26]
    ld1 { v16.h }[6], [x25]
    b label_117
KAI_ASM_LABEL(label_114)  // Height 3: Partial accumulate: partial_1_4
    mov x20, #0x8
    tbz x8, #0, label_117
    ld1 { v8.h }[4], [x16]
    ld1 { v12.h }[4], [x26]
    ld1 { v16.h }[4], [x25]
    b label_117
KAI_ASM_LABEL(label_115)  // Height 3: Partial accumulate: partial_2_0
    tbz x8, #1, label_116
    ldr s8, [x16], #0x4
    mov x20, #0x4
    ldr s12, [x26], #0x4
    ldr s16, [x25], #0x4
    tbz x8, #0, label_117
    ld1 { v8.h }[2], [x16]
    ld1 { v12.h }[2], [x26]
    ld1 { v16.h }[2], [x25]
    b label_117
KAI_ASM_LABEL(label_116)  // Height 3: Partial accumulate: partial_1_0
    ldr h8, [x16, #0x0]
    mov x20, #0x0
    ldr h12, [x26, #0x0]
    ldr h16, [x25, #0x0]
KAI_ASM_LABEL(label_117)  // Height 3: Partial accumulate: Done
    sub x16, x16, x20
    b label_120
KAI_ASM_LABEL(label_118)  // Height 3: full accumulate
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
    b label_120
KAI_ASM_LABEL(label_119)  // Height 3: no accumulate
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
KAI_ASM_LABEL(label_120)  // Height 3: setup done
    mov x15, #0x0
KAI_ASM_LABEL(label_121)  // Height 3: String loop
    ldr x20, [x2, #0x8]
    ldr x21, [x2, #0x30]
    ldr w14, [x20, x15, LSL #0x2]
    tbz x3, #3, label_122
    ldr x20, [x0, x15, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x13, [x20, #0x0]
    ldr x12, [x20, #0x8]
    ldr x11, [x20, #0x10]
    cbnz x15, label_123
    ldr x20, [x2, #0x28]
    add x13, x13, x20, LSL #1
    add x12, x12, x20, LSL #1
    add x11, x11, x20, LSL #1
    b label_123
KAI_ASM_LABEL(label_122)  // Height 3: setup direct input
    mov x13, x0
    add x12, x13, x21, LSL #1
    add x11, x12, x21, LSL #1
KAI_ASM_LABEL(label_123)  // Height 3: input setup done
    cmp x14, #0x8
    blt label_126
    ldr q0, [x13, #0x0]
    cmp x14, #0x10
    ldr q1, [x12, #0x0]
    ldr q2, [x11, #0x0]
    ldr q6, [x17, #0x0]
    ldr q7, [x17, #0x10]
    blt label_125
KAI_ASM_LABEL(label_124)  // Height 3: Multiply loop: Main loop head
    fmla v8.8h, v6.8h, v0.h[0]
    ldr x21, [x17, #0x28]
    fmla v12.8h, v6.8h, v1.h[0]
    ldr x20, [x17, #0x38]
    fmla v16.8h, v6.8h, v2.h[0]
    ldr d6, [x17, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    add x13, x13, #0x10
    fmla v13.8h, v7.8h, v1.h[0]
    add x12, x12, #0x10
    fmla v17.8h, v7.8h, v2.h[0]
    ldr d7, [x17, #0x30]
    mov v6.d[1], x21
    ldr x21, [x17, #0x48]
    add x11, x11, #0x10
    ldr x24, [x13, #0x8]
    mov v7.d[1], x20
    fmla v10.8h, v6.8h, v0.h[0]
    fmla v14.8h, v6.8h, v1.h[0]
    ldr x20, [x17, #0x58]
    fmla v18.8h, v6.8h, v2.h[0]
    ldr d6, [x17, #0x40]
    fmla v11.8h, v7.8h, v0.h[0]
    ldr x23, [x12, #0x8]
    fmla v15.8h, v7.8h, v1.h[0]
    ldr x22, [x11, #0x8]
    fmla v19.8h, v7.8h, v2.h[0]
    ldr d7, [x17, #0x50]
    mov v6.d[1], x21
    ldr x21, [x17, #0x68]
    sub x14, x14, #0x8
    prfm pldl1keep, [x13, #0x80]
    mov v7.d[1], x20
    fmla v8.8h, v6.8h, v0.h[1]
    fmla v12.8h, v6.8h, v1.h[1]
    ldr x20, [x17, #0x78]
    fmla v16.8h, v6.8h, v2.h[1]
    ldr d6, [x17, #0x60]
    fmla v9.8h, v7.8h, v0.h[1]
    cmp x14, #0x10
    fmla v13.8h, v7.8h, v1.h[1]
    prfm pldl1keep, [x12, #0x80]
    fmla v17.8h, v7.8h, v2.h[1]
    ldr d7, [x17, #0x70]
    mov v6.d[1], x21
    ldr x21, [x17, #0x88]
    prfm pldl1keep, [x11, #0x80]
    mov v7.d[1], x20
    fmla v10.8h, v6.8h, v0.h[1]
    fmla v14.8h, v6.8h, v1.h[1]
    ldr x20, [x17, #0x98]
    fmla v18.8h, v6.8h, v2.h[1]
    ldr d6, [x17, #0x80]
    fmla v11.8h, v7.8h, v0.h[1]
    fmla v15.8h, v7.8h, v1.h[1]
    fmla v19.8h, v7.8h, v2.h[1]
    ldr d7, [x17, #0x90]
    mov v6.d[1], x21
    ldr x21, [x17, #0xa8]
    mov v7.d[1], x20
    fmla v8.8h, v6.8h, v0.h[2]
    fmla v12.8h, v6.8h, v1.h[2]
    ldr x20, [x17, #0xb8]
    fmla v16.8h, v6.8h, v2.h[2]
    ldr d6, [x17, #0xa0]
    fmla v9.8h, v7.8h, v0.h[2]
    fmla v13.8h, v7.8h, v1.h[2]
    fmla v17.8h, v7.8h, v2.h[2]
    ldr d7, [x17, #0xb0]
    mov v6.d[1], x21
    ldr x21, [x17, #0xc8]
    mov v7.d[1], x20
    fmla v10.8h, v6.8h, v0.h[2]
    fmla v14.8h, v6.8h, v1.h[2]
    ldr x20, [x17, #0xd8]
    fmla v18.8h, v6.8h, v2.h[2]
    ldr d6, [x17, #0xc0]
    fmla v11.8h, v7.8h, v0.h[2]
    fmla v15.8h, v7.8h, v1.h[2]
    fmla v19.8h, v7.8h, v2.h[2]
    ldr d7, [x17, #0xd0]
    mov v6.d[1], x21
    ldr x21, [x17, #0xe8]
    mov v7.d[1], x20
    fmla v8.8h, v6.8h, v0.h[3]
    fmla v12.8h, v6.8h, v1.h[3]
    ldr x20, [x17, #0xf8]
    fmla v16.8h, v6.8h, v2.h[3]
    ldr d6, [x17, #0xe0]
    fmla v9.8h, v7.8h, v0.h[3]
    fmla v13.8h, v7.8h, v1.h[3]
    fmla v17.8h, v7.8h, v2.h[3]
    ldr d7, [x17, #0xf0]
    mov v6.d[1], x21
    ldr x21, [x17, #0x108]
    mov v7.d[1], x20
    fmla v10.8h, v6.8h, v0.h[3]
    fmla v14.8h, v6.8h, v1.h[3]
    ldr x20, [x17, #0x118]
    fmla v18.8h, v6.8h, v2.h[3]
    ldr d6, [x17, #0x100]
    fmla v11.8h, v7.8h, v0.h[3]
    fmla v15.8h, v7.8h, v1.h[3]
    fmla v19.8h, v7.8h, v2.h[3]
    ldr d7, [x17, #0x110]
    mov v6.d[1], x21
    ldr x21, [x17, #0x128]
    mov v7.d[1], x20
    fmla v8.8h, v6.8h, v0.h[4]
    fmla v12.8h, v6.8h, v1.h[4]
    ldr x20, [x17, #0x138]
    fmla v16.8h, v6.8h, v2.h[4]
    ldr d6, [x17, #0x120]
    fmla v9.8h, v7.8h, v0.h[4]
    fmla v13.8h, v7.8h, v1.h[4]
    fmla v17.8h, v7.8h, v2.h[4]
    ldr d7, [x17, #0x130]
    mov v6.d[1], x21
    ldr x21, [x17, #0x148]
    mov v7.d[1], x20
    fmla v10.8h, v6.8h, v0.h[4]
    fmla v14.8h, v6.8h, v1.h[4]
    ldr x20, [x17, #0x158]
    fmla v18.8h, v6.8h, v2.h[4]
    ldr d6, [x17, #0x140]
    fmla v11.8h, v7.8h, v0.h[4]
    fmla v15.8h, v7.8h, v1.h[4]
    fmla v19.8h, v7.8h, v2.h[4]
    ldr d7, [x17, #0x150]
    mov v6.d[1], x21
    ldr x21, [x17, #0x168]
    mov v7.d[1], x20
    fmla v8.8h, v6.8h, v0.h[5]
    fmla v12.8h, v6.8h, v1.h[5]
    ldr x20, [x17, #0x178]
    fmla v16.8h, v6.8h, v2.h[5]
    ldr d6, [x17, #0x160]
    fmla v9.8h, v7.8h, v0.h[5]
    fmla v13.8h, v7.8h, v1.h[5]
    fmla v17.8h, v7.8h, v2.h[5]
    ldr d7, [x17, #0x170]
    mov v6.d[1], x21
    ldr x21, [x17, #0x188]
    mov v7.d[1], x20
    fmla v10.8h, v6.8h, v0.h[5]
    fmla v14.8h, v6.8h, v1.h[5]
    ldr x20, [x17, #0x198]
    fmla v18.8h, v6.8h, v2.h[5]
    ldr d6, [x17, #0x180]
    fmla v11.8h, v7.8h, v0.h[5]
    fmla v15.8h, v7.8h, v1.h[5]
    fmla v19.8h, v7.8h, v2.h[5]
    ldr d7, [x17, #0x190]
    mov v6.d[1], x21
    ldr x21, [x17, #0x1a8]
    mov v7.d[1], x20
    fmla v8.8h, v6.8h, v0.h[6]
    fmla v12.8h, v6.8h, v1.h[6]
    ldr x20, [x17, #0x1b8]
    fmla v16.8h, v6.8h, v2.h[6]
    ldr d6, [x17, #0x1a0]
    fmla v9.8h, v7.8h, v0.h[6]
    fmla v13.8h, v7.8h, v1.h[6]
    fmla v17.8h, v7.8h, v2.h[6]
    ldr d7, [x17, #0x1b0]
    mov v6.d[1], x21
    ldr x21, [x17, #0x1c8]
    mov v7.d[1], x20
    fmla v10.8h, v6.8h, v0.h[6]
    fmla v14.8h, v6.8h, v1.h[6]
    ldr x20, [x17, #0x1d8]
    fmla v18.8h, v6.8h, v2.h[6]
    ldr d6, [x17, #0x1c0]
    fmla v11.8h, v7.8h, v0.h[6]
    fmla v15.8h, v7.8h, v1.h[6]
    fmla v19.8h, v7.8h, v2.h[6]
    ldr d7, [x17, #0x1d0]
    mov v6.d[1], x21
    ldr x21, [x17, #0x1e8]
    mov v7.d[1], x20
    fmla v8.8h, v6.8h, v0.h[7]
    fmla v12.8h, v6.8h, v1.h[7]
    ldr x20, [x17, #0x1f8]
    fmla v16.8h, v6.8h, v2.h[7]
    ldr d6, [x17, #0x1e0]
    fmla v9.8h, v7.8h, v0.h[7]
    fmla v13.8h, v7.8h, v1.h[7]
    fmla v17.8h, v7.8h, v2.h[7]
    ldr d7, [x17, #0x1f0]
    mov v6.d[1], x21
    add x17, x17, #0x200
    ldr x21, [x17, #0x8]
    mov v7.d[1], x20
    fmla v10.8h, v6.8h, v0.h[7]
    fmla v14.8h, v6.8h, v1.h[7]
    ldr x20, [x17, #0x18]
    fmla v18.8h, v6.8h, v2.h[7]
    ldr d6, [x17, #0x0]
    fmla v11.8h, v7.8h, v0.h[7]
    ldr d0, [x13, #0x0]
    fmla v15.8h, v7.8h, v1.h[7]
    ldr d1, [x12, #0x0]
    fmla v19.8h, v7.8h, v2.h[7]
    ldr d2, [x11, #0x0]
    ldr d7, [x17, #0x10]
    mov v6.d[1], x21
    mov v0.d[1], x24
    mov v1.d[1], x23
    mov v2.d[1], x22
    mov v7.d[1], x20
    bge label_124
KAI_ASM_LABEL(label_125)  // Height 3: Multiply loop: Single iteration only
    fmla v8.8h, v6.8h, v0.h[0]
    add x13, x13, #0x10
    fmla v12.8h, v6.8h, v1.h[0]
    add x12, x12, #0x10
    fmla v16.8h, v6.8h, v2.h[0]
    ldr q6, [x17, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    add x11, x11, #0x10
    fmla v13.8h, v7.8h, v1.h[0]
    sub x14, x14, #0x8
    fmla v17.8h, v7.8h, v2.h[0]
    ldr q7, [x17, #0x30]
    fmla v10.8h, v6.8h, v0.h[0]
    prfm pldl1keep, [x13, #0x80]
    fmla v14.8h, v6.8h, v1.h[0]
    prfm pldl1keep, [x12, #0x80]
    fmla v18.8h, v6.8h, v2.h[0]
    ldr q6, [x17, #0x40]
    fmla v11.8h, v7.8h, v0.h[0]
    prfm pldl1keep, [x11, #0x80]
    fmla v15.8h, v7.8h, v1.h[0]
    fmla v19.8h, v7.8h, v2.h[0]
    ldr q7, [x17, #0x50]
    fmla v8.8h, v6.8h, v0.h[1]
    fmla v12.8h, v6.8h, v1.h[1]
    fmla v16.8h, v6.8h, v2.h[1]
    ldr q6, [x17, #0x60]
    fmla v9.8h, v7.8h, v0.h[1]
    fmla v13.8h, v7.8h, v1.h[1]
    fmla v17.8h, v7.8h, v2.h[1]
    ldr q7, [x17, #0x70]
    fmla v10.8h, v6.8h, v0.h[1]
    fmla v14.8h, v6.8h, v1.h[1]
    fmla v18.8h, v6.8h, v2.h[1]
    ldr q6, [x17, #0x80]
    fmla v11.8h, v7.8h, v0.h[1]
    fmla v15.8h, v7.8h, v1.h[1]
    fmla v19.8h, v7.8h, v2.h[1]
    ldr q7, [x17, #0x90]
    fmla v8.8h, v6.8h, v0.h[2]
    fmla v12.8h, v6.8h, v1.h[2]
    fmla v16.8h, v6.8h, v2.h[2]
    ldr q6, [x17, #0xa0]
    fmla v9.8h, v7.8h, v0.h[2]
    fmla v13.8h, v7.8h, v1.h[2]
    fmla v17.8h, v7.8h, v2.h[2]
    ldr q7, [x17, #0xb0]
    fmla v10.8h, v6.8h, v0.h[2]
    fmla v14.8h, v6.8h, v1.h[2]
    fmla v18.8h, v6.8h, v2.h[2]
    ldr q6, [x17, #0xc0]
    fmla v11.8h, v7.8h, v0.h[2]
    fmla v15.8h, v7.8h, v1.h[2]
    fmla v19.8h, v7.8h, v2.h[2]
    ldr q7, [x17, #0xd0]
    fmla v8.8h, v6.8h, v0.h[3]
    fmla v12.8h, v6.8h, v1.h[3]
    fmla v16.8h, v6.8h, v2.h[3]
    ldr q6, [x17, #0xe0]
    fmla v9.8h, v7.8h, v0.h[3]
    fmla v13.8h, v7.8h, v1.h[3]
    fmla v17.8h, v7.8h, v2.h[3]
    ldr q7, [x17, #0xf0]
    fmla v10.8h, v6.8h, v0.h[3]
    fmla v14.8h, v6.8h, v1.h[3]
    fmla v18.8h, v6.8h, v2.h[3]
    ldr q6, [x17, #0x100]
    fmla v11.8h, v7.8h, v0.h[3]
    fmla v15.8h, v7.8h, v1.h[3]
    fmla v19.8h, v7.8h, v2.h[3]
    ldr q7, [x17, #0x110]
    fmla v8.8h, v6.8h, v0.h[4]
    fmla v12.8h, v6.8h, v1.h[4]
    fmla v16.8h, v6.8h, v2.h[4]
    ldr q6, [x17, #0x120]
    fmla v9.8h, v7.8h, v0.h[4]
    fmla v13.8h, v7.8h, v1.h[4]
    fmla v17.8h, v7.8h, v2.h[4]
    ldr q7, [x17, #0x130]
    fmla v10.8h, v6.8h, v0.h[4]
    fmla v14.8h, v6.8h, v1.h[4]
    fmla v18.8h, v6.8h, v2.h[4]
    ldr q6, [x17, #0x140]
    fmla v11.8h, v7.8h, v0.h[4]
    fmla v15.8h, v7.8h, v1.h[4]
    fmla v19.8h, v7.8h, v2.h[4]
    ldr q7, [x17, #0x150]
    fmla v8.8h, v6.8h, v0.h[5]
    fmla v12.8h, v6.8h, v1.h[5]
    fmla v16.8h, v6.8h, v2.h[5]
    ldr q6, [x17, #0x160]
    fmla v9.8h, v7.8h, v0.h[5]
    fmla v13.8h, v7.8h, v1.h[5]
    fmla v17.8h, v7.8h, v2.h[5]
    ldr q7, [x17, #0x170]
    fmla v10.8h, v6.8h, v0.h[5]
    fmla v14.8h, v6.8h, v1.h[5]
    fmla v18.8h, v6.8h, v2.h[5]
    ldr q6, [x17, #0x180]
    fmla v11.8h, v7.8h, v0.h[5]
    fmla v15.8h, v7.8h, v1.h[5]
    fmla v19.8h, v7.8h, v2.h[5]
    ldr q7, [x17, #0x190]
    fmla v8.8h, v6.8h, v0.h[6]
    fmla v12.8h, v6.8h, v1.h[6]
    fmla v16.8h, v6.8h, v2.h[6]
    ldr q6, [x17, #0x1a0]
    fmla v9.8h, v7.8h, v0.h[6]
    fmla v13.8h, v7.8h, v1.h[6]
    fmla v17.8h, v7.8h, v2.h[6]
    ldr q7, [x17, #0x1b0]
    fmla v10.8h, v6.8h, v0.h[6]
    fmla v14.8h, v6.8h, v1.h[6]
    fmla v18.8h, v6.8h, v2.h[6]
    ldr q6, [x17, #0x1c0]
    fmla v11.8h, v7.8h, v0.h[6]
    fmla v15.8h, v7.8h, v1.h[6]
    fmla v19.8h, v7.8h, v2.h[6]
    ldr q7, [x17, #0x1d0]
    fmla v8.8h, v6.8h, v0.h[7]
    fmla v12.8h, v6.8h, v1.h[7]
    fmla v16.8h, v6.8h, v2.h[7]
    ldr q6, [x17, #0x1e0]
    fmla v9.8h, v7.8h, v0.h[7]
    fmla v13.8h, v7.8h, v1.h[7]
    fmla v17.8h, v7.8h, v2.h[7]
    ldr q7, [x17, #0x1f0]
    fmla v10.8h, v6.8h, v0.h[7]
    add x17, x17, #0x200
    fmla v14.8h, v6.8h, v1.h[7]
    fmla v18.8h, v6.8h, v2.h[7]
    fmla v11.8h, v7.8h, v0.h[7]
    fmla v15.8h, v7.8h, v1.h[7]
    fmla v19.8h, v7.8h, v2.h[7]
KAI_ASM_LABEL(label_126)  // Height 3: Multiply loop: Main loop skip
    cbz x14, label_128
KAI_ASM_LABEL(label_127)  // Height 3: Multiply loop: Odd block loop
    ldr h0, [x13], #0x2
    sub x14, x14, #0x1
    ldr h1, [x12], #0x2
    ldr h2, [x11], #0x2
    ldr q6, [x17, #0x0]
    ldr q7, [x17, #0x10]
    fmla v8.8h, v6.8h, v0.h[0]
    fmla v12.8h, v6.8h, v1.h[0]
    fmla v16.8h, v6.8h, v2.h[0]
    ldr q6, [x17, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    fmla v13.8h, v7.8h, v1.h[0]
    fmla v17.8h, v7.8h, v2.h[0]
    ldr q7, [x17, #0x30]
    fmla v10.8h, v6.8h, v0.h[0]
    add x17, x17, #0x40
    fmla v14.8h, v6.8h, v1.h[0]
    fmla v18.8h, v6.8h, v2.h[0]
    fmla v11.8h, v7.8h, v0.h[0]
    fmla v15.8h, v7.8h, v1.h[0]
    fmla v19.8h, v7.8h, v2.h[0]
    cbnz x14, label_127
KAI_ASM_LABEL(label_128)  // Height 3: Multiply loop: No odd multiplies
    ldr w20, [x2, #0x4]
    add x15, x15, #0x1
    cmp x15, x20
    bne label_121
    ldr x20, [x2, #0x20]
    prfm pstl1keep, [x16, #0x0]
    add x26, x16, x20, LSL #1
    add x25, x26, x20, LSL #1
    prfm pstl1keep, [x26, #0x0]
    prfm pstl1keep, [x25, #0x0]
    tbz x3, #1, label_129
    add x21, x2, #0x0
    add x20, x2, #0x2
    ld1r { v21.8h }, [x21]
    ld1r { v20.8h }, [x20]
    fmin v8.8h, v8.8h, v21.8h
    fmin v9.8h, v9.8h, v21.8h
    fmin v10.8h, v10.8h, v21.8h
    fmin v11.8h, v11.8h, v21.8h
    fmin v12.8h, v12.8h, v21.8h
    fmin v13.8h, v13.8h, v21.8h
    fmin v14.8h, v14.8h, v21.8h
    fmin v15.8h, v15.8h, v21.8h
    fmin v16.8h, v16.8h, v21.8h
    fmin v17.8h, v17.8h, v21.8h
    fmin v18.8h, v18.8h, v21.8h
    fmin v19.8h, v19.8h, v21.8h
    fmax v8.8h, v8.8h, v20.8h
    fmax v9.8h, v9.8h, v20.8h
    fmax v10.8h, v10.8h, v20.8h
    fmax v11.8h, v11.8h, v20.8h
    fmax v12.8h, v12.8h, v20.8h
    fmax v13.8h, v13.8h, v20.8h
    fmax v14.8h, v14.8h, v20.8h
    fmax v15.8h, v15.8h, v20.8h
    fmax v16.8h, v16.8h, v20.8h
    fmax v17.8h, v17.8h, v20.8h
    fmax v18.8h, v18.8h, v20.8h
    fmax v19.8h, v19.8h, v20.8h
KAI_ASM_LABEL(label_129)  // Height 3: No activation
    cmp x8, #0x20
    bge label_146
    tbz x8, #4, label_137
    st1 { v8.8h }, [x16], #0x10
    st1 { v9.8h }, [x16], #0x10
    st1 { v12.8h }, [x26], #0x10
    st1 { v13.8h }, [x26], #0x10
    st1 { v16.8h }, [x25], #0x10
    st1 { v17.8h }, [x25], #0x10
    tbz x8, #3, label_133
    st1 { v10.8h }, [x16], #0x10
    st1 { v14.8h }, [x26], #0x10
    st1 { v18.8h }, [x25], #0x10
    tbz x8, #2, label_131
    str d11, [x16], #0x8
    str d15, [x26], #0x8
    str d19, [x25], #0x8
    tbz x8, #1, label_130
    st1 { v11.s }[2], [x16], #0x4
    st1 { v15.s }[2], [x26], #0x4
    st1 { v19.s }[2], [x25], #0x4
    tbz x8, #0, label_145
    st1 { v11.h }[6], [x16]
    st1 { v15.h }[6], [x26]
    st1 { v19.h }[6], [x25]
    b label_145
KAI_ASM_LABEL(label_130)  // Height 3: Partial direct writeback: partial_1_28
    tbz x8, #0, label_145
    st1 { v11.h }[4], [x16]
    st1 { v15.h }[4], [x26]
    st1 { v19.h }[4], [x25]
    b label_145
KAI_ASM_LABEL(label_131)  // Height 3: Partial direct writeback: partial_2_24
    tbz x8, #1, label_132
    str s11, [x16], #0x4
    str s15, [x26], #0x4
    str s19, [x25], #0x4
    tbz x8, #0, label_145
    st1 { v11.h }[2], [x16]
    st1 { v15.h }[2], [x26]
    st1 { v19.h }[2], [x25]
    b label_145
KAI_ASM_LABEL(label_132)  // Height 3: Partial direct writeback: partial_1_24
    tbz x8, #0, label_145
    str h11, [x16, #0x0]
    str h15, [x26, #0x0]
    str h19, [x25, #0x0]
    b label_145
KAI_ASM_LABEL(label_133)  // Height 3: Partial direct writeback: partial_4_16
    tbz x8, #2, label_135
    str d10, [x16], #0x8
    str d14, [x26], #0x8
    str d18, [x25], #0x8
    tbz x8, #1, label_134
    st1 { v10.s }[2], [x16], #0x4
    st1 { v14.s }[2], [x26], #0x4
    st1 { v18.s }[2], [x25], #0x4
    tbz x8, #0, label_145
    st1 { v10.h }[6], [x16]
    st1 { v14.h }[6], [x26]
    st1 { v18.h }[6], [x25]
    b label_145
KAI_ASM_LABEL(label_134)  // Height 3: Partial direct writeback: partial_1_20
    tbz x8, #0, label_145
    st1 { v10.h }[4], [x16]
    st1 { v14.h }[4], [x26]
    st1 { v18.h }[4], [x25]
    b label_145
KAI_ASM_LABEL(label_135)  // Height 3: Partial direct writeback: partial_2_16
    tbz x8, #1, label_136
    str s10, [x16], #0x4
    str s14, [x26], #0x4
    str s18, [x25], #0x4
    tbz x8, #0, label_145
    st1 { v10.h }[2], [x16]
    st1 { v14.h }[2], [x26]
    st1 { v18.h }[2], [x25]
    b label_145
KAI_ASM_LABEL(label_136)  // Height 3: Partial direct writeback: partial_1_16
    tbz x8, #0, label_145
    str h10, [x16, #0x0]
    str h14, [x26, #0x0]
    str h18, [x25, #0x0]
    b label_145
KAI_ASM_LABEL(label_137)  // Height 3: Partial direct writeback: partial_8_0
    tbz x8, #3, label_141
    st1 { v8.8h }, [x16], #0x10
    st1 { v12.8h }, [x26], #0x10
    st1 { v16.8h }, [x25], #0x10
    tbz x8, #2, label_139
    str d9, [x16], #0x8
    str d13, [x26], #0x8
    str d17, [x25], #0x8
    tbz x8, #1, label_138
    st1 { v9.s }[2], [x16], #0x4
    st1 { v13.s }[2], [x26], #0x4
    st1 { v17.s }[2], [x25], #0x4
    tbz x8, #0, label_145
    st1 { v9.h }[6], [x16]
    st1 { v13.h }[6], [x26]
    st1 { v17.h }[6], [x25]
    b label_145
KAI_ASM_LABEL(label_138)  // Height 3: Partial direct writeback: partial_1_12
    tbz x8, #0, label_145
    st1 { v9.h }[4], [x16]
    st1 { v13.h }[4], [x26]
    st1 { v17.h }[4], [x25]
    b label_145
KAI_ASM_LABEL(label_139)  // Height 3: Partial direct writeback: partial_2_8
    tbz x8, #1, label_140
    str s9, [x16], #0x4
    str s13, [x26], #0x4
    str s17, [x25], #0x4
    tbz x8, #0, label_145
    st1 { v9.h }[2], [x16]
    st1 { v13.h }[2], [x26]
    st1 { v17.h }[2], [x25]
    b label_145
KAI_ASM_LABEL(label_140)  // Height 3: Partial direct writeback: partial_1_8
    tbz x8, #0, label_145
    str h9, [x16, #0x0]
    str h13, [x26, #0x0]
    str h17, [x25, #0x0]
    b label_145
KAI_ASM_LABEL(label_141)  // Height 3: Partial direct writeback: partial_4_0
    tbz x8, #2, label_143
    str d8, [x16], #0x8
    str d12, [x26], #0x8
    str d16, [x25], #0x8
    tbz x8, #1, label_142
    st1 { v8.s }[2], [x16], #0x4
    st1 { v12.s }[2], [x26], #0x4
    st1 { v16.s }[2], [x25], #0x4
    tbz x8, #0, label_145
    st1 { v8.h }[6], [x16]
    st1 { v12.h }[6], [x26]
    st1 { v16.h }[6], [x25]
    b label_145
KAI_ASM_LABEL(label_142)  // Height 3: Partial direct writeback: partial_1_4
    tbz x8, #0, label_145
    st1 { v8.h }[4], [x16]
    st1 { v12.h }[4], [x26]
    st1 { v16.h }[4], [x25]
    b label_145
KAI_ASM_LABEL(label_143)  // Height 3: Partial direct writeback: partial_2_0
    tbz x8, #1, label_144
    str s8, [x16], #0x4
    str s12, [x26], #0x4
    str s16, [x25], #0x4
    tbz x8, #0, label_145
    st1 { v8.h }[2], [x16]
    st1 { v12.h }[2], [x26]
    st1 { v16.h }[2], [x25]
    b label_145
KAI_ASM_LABEL(label_144)  // Height 3: Partial direct writeback: partial_1_0
    str h8, [x16, #0x0]
    str h12, [x26, #0x0]
    str h16, [x25, #0x0]
KAI_ASM_LABEL(label_145)  // Height 3: Partial direct writeback: Done
    b label_147
KAI_ASM_LABEL(label_146)  // Height 3: Full writeback
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
KAI_ASM_LABEL(label_147)  // Height 3: Writeback done
    subs x8, x8, #0x20
    bgt label_100
    b label_296
KAI_ASM_LABEL(label_148)  // Height 4
    ldr x8, [x2, #0x10]
    ldr x17, [x2, #0x18]
    ldr x16, [x2, #0x38]
KAI_ASM_LABEL(label_149)  // Height 4: Column loop
    cbz x17, label_150
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
    b label_169
KAI_ASM_LABEL(label_150)  // Height 4: no bias
    tbz x3, #0, label_168
    ldr x20, [x2, #0x20]
    cmp x8, #0x20
    add x26, x16, x20, LSL #1
    add x25, x26, x20, LSL #1
    add x24, x25, x20, LSL #1
    bge label_167
    tbz x8, #4, label_158
    ld1 { v8.8h }, [x16], #0x10
    ld1 { v12.8h }, [x26], #0x10
    ld1 { v16.8h }, [x25], #0x10
    ld1 { v20.8h }, [x24], #0x10
    ld1 { v9.8h }, [x16], #0x10
    ld1 { v13.8h }, [x26], #0x10
    ld1 { v17.8h }, [x25], #0x10
    ld1 { v21.8h }, [x24], #0x10
    tbz x8, #3, label_154
    ld1 { v10.8h }, [x16], #0x10
    ld1 { v14.8h }, [x26], #0x10
    ld1 { v18.8h }, [x25], #0x10
    ld1 { v22.8h }, [x24], #0x10
    tbz x8, #2, label_152
    ldr d11, [x16], #0x8
    ldr d15, [x26], #0x8
    ldr d19, [x25], #0x8
    ldr d23, [x24], #0x8
    tbz x8, #1, label_151
    ld1 { v11.s }[2], [x16], #0x4
    mov x20, #0x3c
    ld1 { v15.s }[2], [x26], #0x4
    ld1 { v19.s }[2], [x25], #0x4
    ld1 { v23.s }[2], [x24], #0x4
    tbz x8, #0, label_166
    ld1 { v11.h }[6], [x16]
    ld1 { v15.h }[6], [x26]
    ld1 { v19.h }[6], [x25]
    ld1 { v23.h }[6], [x24]
    b label_166
KAI_ASM_LABEL(label_151)  // Height 4: Partial accumulate: partial_1_28
    mov x20, #0x38
    tbz x8, #0, label_166
    ld1 { v11.h }[4], [x16]
    ld1 { v15.h }[4], [x26]
    ld1 { v19.h }[4], [x25]
    ld1 { v23.h }[4], [x24]
    b label_166
KAI_ASM_LABEL(label_152)  // Height 4: Partial accumulate: partial_2_24
    tbz x8, #1, label_153
    ldr s11, [x16], #0x4
    mov x20, #0x34
    ldr s15, [x26], #0x4
    ldr s19, [x25], #0x4
    ldr s23, [x24], #0x4
    tbz x8, #0, label_166
    ld1 { v11.h }[2], [x16]
    ld1 { v15.h }[2], [x26]
    ld1 { v19.h }[2], [x25]
    ld1 { v23.h }[2], [x24]
    b label_166
KAI_ASM_LABEL(label_153)  // Height 4: Partial accumulate: partial_1_24
    mov x20, #0x30
    tbz x8, #0, label_166
    ldr h11, [x16, #0x0]
    ldr h15, [x26, #0x0]
    ldr h19, [x25, #0x0]
    ldr h23, [x24, #0x0]
    b label_166
KAI_ASM_LABEL(label_154)  // Height 4: Partial accumulate: partial_4_16
    tbz x8, #2, label_156
    ldr d10, [x16], #0x8
    ldr d14, [x26], #0x8
    ldr d18, [x25], #0x8
    ldr d22, [x24], #0x8
    tbz x8, #1, label_155
    ld1 { v10.s }[2], [x16], #0x4
    mov x20, #0x2c
    ld1 { v14.s }[2], [x26], #0x4
    ld1 { v18.s }[2], [x25], #0x4
    ld1 { v22.s }[2], [x24], #0x4
    tbz x8, #0, label_166
    ld1 { v10.h }[6], [x16]
    ld1 { v14.h }[6], [x26]
    ld1 { v18.h }[6], [x25]
    ld1 { v22.h }[6], [x24]
    b label_166
KAI_ASM_LABEL(label_155)  // Height 4: Partial accumulate: partial_1_20
    mov x20, #0x28
    tbz x8, #0, label_166
    ld1 { v10.h }[4], [x16]
    ld1 { v14.h }[4], [x26]
    ld1 { v18.h }[4], [x25]
    ld1 { v22.h }[4], [x24]
    b label_166
KAI_ASM_LABEL(label_156)  // Height 4: Partial accumulate: partial_2_16
    tbz x8, #1, label_157
    ldr s10, [x16], #0x4
    mov x20, #0x24
    ldr s14, [x26], #0x4
    ldr s18, [x25], #0x4
    ldr s22, [x24], #0x4
    tbz x8, #0, label_166
    ld1 { v10.h }[2], [x16]
    ld1 { v14.h }[2], [x26]
    ld1 { v18.h }[2], [x25]
    ld1 { v22.h }[2], [x24]
    b label_166
KAI_ASM_LABEL(label_157)  // Height 4: Partial accumulate: partial_1_16
    mov x20, #0x20
    tbz x8, #0, label_166
    ldr h10, [x16, #0x0]
    ldr h14, [x26, #0x0]
    ldr h18, [x25, #0x0]
    ldr h22, [x24, #0x0]
    b label_166
KAI_ASM_LABEL(label_158)  // Height 4: Partial accumulate: partial_8_0
    tbz x8, #3, label_162
    ld1 { v8.8h }, [x16], #0x10
    ld1 { v12.8h }, [x26], #0x10
    ld1 { v16.8h }, [x25], #0x10
    ld1 { v20.8h }, [x24], #0x10
    tbz x8, #2, label_160
    ldr d9, [x16], #0x8
    ldr d13, [x26], #0x8
    ldr d17, [x25], #0x8
    ldr d21, [x24], #0x8
    tbz x8, #1, label_159
    ld1 { v9.s }[2], [x16], #0x4
    mov x20, #0x1c
    ld1 { v13.s }[2], [x26], #0x4
    ld1 { v17.s }[2], [x25], #0x4
    ld1 { v21.s }[2], [x24], #0x4
    tbz x8, #0, label_166
    ld1 { v9.h }[6], [x16]
    ld1 { v13.h }[6], [x26]
    ld1 { v17.h }[6], [x25]
    ld1 { v21.h }[6], [x24]
    b label_166
KAI_ASM_LABEL(label_159)  // Height 4: Partial accumulate: partial_1_12
    mov x20, #0x18
    tbz x8, #0, label_166
    ld1 { v9.h }[4], [x16]
    ld1 { v13.h }[4], [x26]
    ld1 { v17.h }[4], [x25]
    ld1 { v21.h }[4], [x24]
    b label_166
KAI_ASM_LABEL(label_160)  // Height 4: Partial accumulate: partial_2_8
    tbz x8, #1, label_161
    ldr s9, [x16], #0x4
    mov x20, #0x14
    ldr s13, [x26], #0x4
    ldr s17, [x25], #0x4
    ldr s21, [x24], #0x4
    tbz x8, #0, label_166
    ld1 { v9.h }[2], [x16]
    ld1 { v13.h }[2], [x26]
    ld1 { v17.h }[2], [x25]
    ld1 { v21.h }[2], [x24]
    b label_166
KAI_ASM_LABEL(label_161)  // Height 4: Partial accumulate: partial_1_8
    mov x20, #0x10
    tbz x8, #0, label_166
    ldr h9, [x16, #0x0]
    ldr h13, [x26, #0x0]
    ldr h17, [x25, #0x0]
    ldr h21, [x24, #0x0]
    b label_166
KAI_ASM_LABEL(label_162)  // Height 4: Partial accumulate: partial_4_0
    tbz x8, #2, label_164
    ldr d8, [x16], #0x8
    ldr d12, [x26], #0x8
    ldr d16, [x25], #0x8
    ldr d20, [x24], #0x8
    tbz x8, #1, label_163
    ld1 { v8.s }[2], [x16], #0x4
    mov x20, #0xc
    ld1 { v12.s }[2], [x26], #0x4
    ld1 { v16.s }[2], [x25], #0x4
    ld1 { v20.s }[2], [x24], #0x4
    tbz x8, #0, label_166
    ld1 { v8.h }[6], [x16]
    ld1 { v12.h }[6], [x26]
    ld1 { v16.h }[6], [x25]
    ld1 { v20.h }[6], [x24]
    b label_166
KAI_ASM_LABEL(label_163)  // Height 4: Partial accumulate: partial_1_4
    mov x20, #0x8
    tbz x8, #0, label_166
    ld1 { v8.h }[4], [x16]
    ld1 { v12.h }[4], [x26]
    ld1 { v16.h }[4], [x25]
    ld1 { v20.h }[4], [x24]
    b label_166
KAI_ASM_LABEL(label_164)  // Height 4: Partial accumulate: partial_2_0
    tbz x8, #1, label_165
    ldr s8, [x16], #0x4
    mov x20, #0x4
    ldr s12, [x26], #0x4
    ldr s16, [x25], #0x4
    ldr s20, [x24], #0x4
    tbz x8, #0, label_166
    ld1 { v8.h }[2], [x16]
    ld1 { v12.h }[2], [x26]
    ld1 { v16.h }[2], [x25]
    ld1 { v20.h }[2], [x24]
    b label_166
KAI_ASM_LABEL(label_165)  // Height 4: Partial accumulate: partial_1_0
    ldr h8, [x16, #0x0]
    mov x20, #0x0
    ldr h12, [x26, #0x0]
    ldr h16, [x25, #0x0]
    ldr h20, [x24, #0x0]
KAI_ASM_LABEL(label_166)  // Height 4: Partial accumulate: Done
    sub x16, x16, x20
    b label_169
KAI_ASM_LABEL(label_167)  // Height 4: full accumulate
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
    b label_169
KAI_ASM_LABEL(label_168)  // Height 4: no accumulate
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
KAI_ASM_LABEL(label_169)  // Height 4: setup done
    mov x15, #0x0
KAI_ASM_LABEL(label_170)  // Height 4: String loop
    ldr x20, [x2, #0x8]
    ldr x21, [x2, #0x30]
    ldr w14, [x20, x15, LSL #0x2]
    tbz x3, #3, label_171
    ldr x20, [x0, x15, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x13, [x20, #0x0]
    ldr x12, [x20, #0x8]
    ldr x11, [x20, #0x10]
    ldr x10, [x20, #0x18]
    cbnz x15, label_172
    ldr x20, [x2, #0x28]
    add x13, x13, x20, LSL #1
    add x12, x12, x20, LSL #1
    add x11, x11, x20, LSL #1
    add x10, x10, x20, LSL #1
    b label_172
KAI_ASM_LABEL(label_171)  // Height 4: setup direct input
    mov x13, x0
    add x12, x13, x21, LSL #1
    add x11, x12, x21, LSL #1
    add x10, x11, x21, LSL #1
KAI_ASM_LABEL(label_172)  // Height 4: input setup done
    cmp x14, #0x8
    blt label_175
    ldr q0, [x13, #0x0]
    cmp x14, #0x10
    ldr q1, [x12, #0x0]
    ldr q2, [x11, #0x0]
    ldr q3, [x10, #0x0]
    ldr q6, [x17, #0x0]
    ldr q7, [x17, #0x10]
    blt label_174
KAI_ASM_LABEL(label_173)  // Height 4: Multiply loop: Main loop head
    fmla v8.8h, v6.8h, v0.h[0]
    ldr x20, [x17, #0x28]
    fmla v12.8h, v6.8h, v1.h[0]
    ldr x21, [x17, #0x38]
    fmla v16.8h, v6.8h, v2.h[0]
    add x13, x13, #0x10
    fmla v20.8h, v6.8h, v3.h[0]
    ldr d6, [x17, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    add x12, x12, #0x10
    fmla v13.8h, v7.8h, v1.h[0]
    add x11, x11, #0x10
    fmla v17.8h, v7.8h, v2.h[0]
    mov v6.d[1], x20
    fmla v21.8h, v7.8h, v3.h[0]
    ldr d7, [x17, #0x30]
    fmla v10.8h, v6.8h, v0.h[0]
    ldr x20, [x17, #0x48]
    fmla v14.8h, v6.8h, v1.h[0]
    add x10, x10, #0x10
    mov v7.d[1], x21
    fmla v18.8h, v6.8h, v2.h[0]
    fmla v22.8h, v6.8h, v3.h[0]
    ldr d6, [x17, #0x40]
    fmla v11.8h, v7.8h, v0.h[0]
    ldr x21, [x17, #0x58]
    fmla v15.8h, v7.8h, v1.h[0]
    ldr x25, [x13, #0x8]
    fmla v19.8h, v7.8h, v2.h[0]
    mov v6.d[1], x20
    fmla v23.8h, v7.8h, v3.h[0]
    ldr d7, [x17, #0x50]
    fmla v8.8h, v6.8h, v0.h[1]
    ldr x20, [x17, #0x68]
    fmla v12.8h, v6.8h, v1.h[1]
    ldr x24, [x12, #0x8]
    mov v7.d[1], x21
    fmla v16.8h, v6.8h, v2.h[1]
    fmla v20.8h, v6.8h, v3.h[1]
    ldr d6, [x17, #0x60]
    fmla v9.8h, v7.8h, v0.h[1]
    ldr x21, [x17, #0x78]
    fmla v13.8h, v7.8h, v1.h[1]
    ldr x23, [x11, #0x8]
    fmla v17.8h, v7.8h, v2.h[1]
    mov v6.d[1], x20
    fmla v21.8h, v7.8h, v3.h[1]
    ldr d7, [x17, #0x70]
    fmla v10.8h, v6.8h, v0.h[1]
    ldr x20, [x17, #0x88]
    fmla v14.8h, v6.8h, v1.h[1]
    ldr x22, [x10, #0x8]
    mov v7.d[1], x21
    fmla v18.8h, v6.8h, v2.h[1]
    fmla v22.8h, v6.8h, v3.h[1]
    ldr d6, [x17, #0x80]
    fmla v11.8h, v7.8h, v0.h[1]
    ldr x21, [x17, #0x98]
    fmla v15.8h, v7.8h, v1.h[1]
    sub x14, x14, #0x8
    fmla v19.8h, v7.8h, v2.h[1]
    mov v6.d[1], x20
    fmla v23.8h, v7.8h, v3.h[1]
    ldr d7, [x17, #0x90]
    fmla v8.8h, v6.8h, v0.h[2]
    ldr x20, [x17, #0xa8]
    fmla v12.8h, v6.8h, v1.h[2]
    cmp x14, #0x10
    mov v7.d[1], x21
    fmla v16.8h, v6.8h, v2.h[2]
    fmla v20.8h, v6.8h, v3.h[2]
    ldr d6, [x17, #0xa0]
    fmla v9.8h, v7.8h, v0.h[2]
    ldr x21, [x17, #0xb8]
    fmla v13.8h, v7.8h, v1.h[2]
    prfm pldl1keep, [x13, #0x80]
    fmla v17.8h, v7.8h, v2.h[2]
    mov v6.d[1], x20
    fmla v21.8h, v7.8h, v3.h[2]
    ldr d7, [x17, #0xb0]
    fmla v10.8h, v6.8h, v0.h[2]
    ldr x20, [x17, #0xc8]
    fmla v14.8h, v6.8h, v1.h[2]
    prfm pldl1keep, [x12, #0x80]
    mov v7.d[1], x21
    fmla v18.8h, v6.8h, v2.h[2]
    fmla v22.8h, v6.8h, v3.h[2]
    ldr d6, [x17, #0xc0]
    fmla v11.8h, v7.8h, v0.h[2]
    ldr x21, [x17, #0xd8]
    fmla v15.8h, v7.8h, v1.h[2]
    prfm pldl1keep, [x11, #0x80]
    fmla v19.8h, v7.8h, v2.h[2]
    mov v6.d[1], x20
    fmla v23.8h, v7.8h, v3.h[2]
    ldr d7, [x17, #0xd0]
    fmla v8.8h, v6.8h, v0.h[3]
    ldr x20, [x17, #0xe8]
    fmla v12.8h, v6.8h, v1.h[3]
    prfm pldl1keep, [x10, #0x80]
    mov v7.d[1], x21
    fmla v16.8h, v6.8h, v2.h[3]
    fmla v20.8h, v6.8h, v3.h[3]
    ldr d6, [x17, #0xe0]
    fmla v9.8h, v7.8h, v0.h[3]
    ldr x21, [x17, #0xf8]
    fmla v13.8h, v7.8h, v1.h[3]
    fmla v17.8h, v7.8h, v2.h[3]
    mov v6.d[1], x20
    fmla v21.8h, v7.8h, v3.h[3]
    ldr d7, [x17, #0xf0]
    fmla v10.8h, v6.8h, v0.h[3]
    ldr x20, [x17, #0x108]
    fmla v14.8h, v6.8h, v1.h[3]
    mov v7.d[1], x21
    fmla v18.8h, v6.8h, v2.h[3]
    fmla v22.8h, v6.8h, v3.h[3]
    ldr d6, [x17, #0x100]
    fmla v11.8h, v7.8h, v0.h[3]
    ldr x21, [x17, #0x118]
    fmla v15.8h, v7.8h, v1.h[3]
    fmla v19.8h, v7.8h, v2.h[3]
    mov v6.d[1], x20
    fmla v23.8h, v7.8h, v3.h[3]
    ldr d7, [x17, #0x110]
    fmla v8.8h, v6.8h, v0.h[4]
    ldr x20, [x17, #0x128]
    fmla v12.8h, v6.8h, v1.h[4]
    mov v7.d[1], x21
    fmla v16.8h, v6.8h, v2.h[4]
    fmla v20.8h, v6.8h, v3.h[4]
    ldr d6, [x17, #0x120]
    fmla v9.8h, v7.8h, v0.h[4]
    ldr x21, [x17, #0x138]
    fmla v13.8h, v7.8h, v1.h[4]
    fmla v17.8h, v7.8h, v2.h[4]
    mov v6.d[1], x20
    fmla v21.8h, v7.8h, v3.h[4]
    ldr d7, [x17, #0x130]
    fmla v10.8h, v6.8h, v0.h[4]
    ldr x20, [x17, #0x148]
    fmla v14.8h, v6.8h, v1.h[4]
    mov v7.d[1], x21
    fmla v18.8h, v6.8h, v2.h[4]
    fmla v22.8h, v6.8h, v3.h[4]
    ldr d6, [x17, #0x140]
    fmla v11.8h, v7.8h, v0.h[4]
    ldr x21, [x17, #0x158]
    fmla v15.8h, v7.8h, v1.h[4]
    fmla v19.8h, v7.8h, v2.h[4]
    mov v6.d[1], x20
    fmla v23.8h, v7.8h, v3.h[4]
    ldr d7, [x17, #0x150]
    fmla v8.8h, v6.8h, v0.h[5]
    ldr x20, [x17, #0x168]
    fmla v12.8h, v6.8h, v1.h[5]
    mov v7.d[1], x21
    fmla v16.8h, v6.8h, v2.h[5]
    fmla v20.8h, v6.8h, v3.h[5]
    ldr d6, [x17, #0x160]
    fmla v9.8h, v7.8h, v0.h[5]
    ldr x21, [x17, #0x178]
    fmla v13.8h, v7.8h, v1.h[5]
    fmla v17.8h, v7.8h, v2.h[5]
    mov v6.d[1], x20
    fmla v21.8h, v7.8h, v3.h[5]
    ldr d7, [x17, #0x170]
    fmla v10.8h, v6.8h, v0.h[5]
    ldr x20, [x17, #0x188]
    fmla v14.8h, v6.8h, v1.h[5]
    mov v7.d[1], x21
    fmla v18.8h, v6.8h, v2.h[5]
    fmla v22.8h, v6.8h, v3.h[5]
    ldr d6, [x17, #0x180]
    fmla v11.8h, v7.8h, v0.h[5]
    ldr x21, [x17, #0x198]
    fmla v15.8h, v7.8h, v1.h[5]
    fmla v19.8h, v7.8h, v2.h[5]
    mov v6.d[1], x20
    fmla v23.8h, v7.8h, v3.h[5]
    ldr d7, [x17, #0x190]
    fmla v8.8h, v6.8h, v0.h[6]
    ldr x20, [x17, #0x1a8]
    fmla v12.8h, v6.8h, v1.h[6]
    mov v7.d[1], x21
    fmla v16.8h, v6.8h, v2.h[6]
    fmla v20.8h, v6.8h, v3.h[6]
    ldr d6, [x17, #0x1a0]
    fmla v9.8h, v7.8h, v0.h[6]
    ldr x21, [x17, #0x1b8]
    fmla v13.8h, v7.8h, v1.h[6]
    fmla v17.8h, v7.8h, v2.h[6]
    mov v6.d[1], x20
    fmla v21.8h, v7.8h, v3.h[6]
    ldr d7, [x17, #0x1b0]
    fmla v10.8h, v6.8h, v0.h[6]
    ldr x20, [x17, #0x1c8]
    fmla v14.8h, v6.8h, v1.h[6]
    mov v7.d[1], x21
    fmla v18.8h, v6.8h, v2.h[6]
    fmla v22.8h, v6.8h, v3.h[6]
    ldr d6, [x17, #0x1c0]
    fmla v11.8h, v7.8h, v0.h[6]
    ldr x21, [x17, #0x1d8]
    fmla v15.8h, v7.8h, v1.h[6]
    fmla v19.8h, v7.8h, v2.h[6]
    mov v6.d[1], x20
    fmla v23.8h, v7.8h, v3.h[6]
    ldr d7, [x17, #0x1d0]
    fmla v8.8h, v6.8h, v0.h[7]
    ldr x20, [x17, #0x1e8]
    fmla v12.8h, v6.8h, v1.h[7]
    mov v7.d[1], x21
    fmla v16.8h, v6.8h, v2.h[7]
    fmla v20.8h, v6.8h, v3.h[7]
    ldr d6, [x17, #0x1e0]
    fmla v9.8h, v7.8h, v0.h[7]
    ldr x21, [x17, #0x1f8]
    fmla v13.8h, v7.8h, v1.h[7]
    fmla v17.8h, v7.8h, v2.h[7]
    mov v6.d[1], x20
    fmla v21.8h, v7.8h, v3.h[7]
    ldr d7, [x17, #0x1f0]
    add x17, x17, #0x200
    fmla v10.8h, v6.8h, v0.h[7]
    fmla v14.8h, v6.8h, v1.h[7]
    ldr x20, [x17, #0x8]
    mov v7.d[1], x21
    fmla v18.8h, v6.8h, v2.h[7]
    fmla v22.8h, v6.8h, v3.h[7]
    ldr d6, [x17, #0x0]
    fmla v11.8h, v7.8h, v0.h[7]
    ldr d0, [x13, #0x0]
    fmla v15.8h, v7.8h, v1.h[7]
    ldr d1, [x12, #0x0]
    fmla v19.8h, v7.8h, v2.h[7]
    ldr d2, [x11, #0x0]
    fmla v23.8h, v7.8h, v3.h[7]
    ldr d3, [x10, #0x0]
    ldr d7, [x17, #0x10]
    mov v6.d[1], x20
    ldr x20, [x17, #0x18]
    mov v0.d[1], x25
    mov v1.d[1], x24
    mov v2.d[1], x23
    mov v3.d[1], x22
    mov v7.d[1], x20
    bge label_173
KAI_ASM_LABEL(label_174)  // Height 4: Multiply loop: Single iteration only
    fmla v8.8h, v6.8h, v0.h[0]
    add x13, x13, #0x10
    fmla v12.8h, v6.8h, v1.h[0]
    add x12, x12, #0x10
    fmla v16.8h, v6.8h, v2.h[0]
    add x11, x11, #0x10
    fmla v20.8h, v6.8h, v3.h[0]
    ldr q6, [x17, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    add x10, x10, #0x10
    fmla v13.8h, v7.8h, v1.h[0]
    sub x14, x14, #0x8
    fmla v17.8h, v7.8h, v2.h[0]
    prfm pldl1keep, [x13, #0x80]
    fmla v21.8h, v7.8h, v3.h[0]
    ldr q7, [x17, #0x30]
    fmla v10.8h, v6.8h, v0.h[0]
    prfm pldl1keep, [x12, #0x80]
    fmla v14.8h, v6.8h, v1.h[0]
    prfm pldl1keep, [x11, #0x80]
    fmla v18.8h, v6.8h, v2.h[0]
    prfm pldl1keep, [x10, #0x80]
    fmla v22.8h, v6.8h, v3.h[0]
    ldr q6, [x17, #0x40]
    fmla v11.8h, v7.8h, v0.h[0]
    fmla v15.8h, v7.8h, v1.h[0]
    fmla v19.8h, v7.8h, v2.h[0]
    fmla v23.8h, v7.8h, v3.h[0]
    ldr q7, [x17, #0x50]
    fmla v8.8h, v6.8h, v0.h[1]
    fmla v12.8h, v6.8h, v1.h[1]
    fmla v16.8h, v6.8h, v2.h[1]
    fmla v20.8h, v6.8h, v3.h[1]
    ldr q6, [x17, #0x60]
    fmla v9.8h, v7.8h, v0.h[1]
    fmla v13.8h, v7.8h, v1.h[1]
    fmla v17.8h, v7.8h, v2.h[1]
    fmla v21.8h, v7.8h, v3.h[1]
    ldr q7, [x17, #0x70]
    fmla v10.8h, v6.8h, v0.h[1]
    fmla v14.8h, v6.8h, v1.h[1]
    fmla v18.8h, v6.8h, v2.h[1]
    fmla v22.8h, v6.8h, v3.h[1]
    ldr q6, [x17, #0x80]
    fmla v11.8h, v7.8h, v0.h[1]
    fmla v15.8h, v7.8h, v1.h[1]
    fmla v19.8h, v7.8h, v2.h[1]
    fmla v23.8h, v7.8h, v3.h[1]
    ldr q7, [x17, #0x90]
    fmla v8.8h, v6.8h, v0.h[2]
    fmla v12.8h, v6.8h, v1.h[2]
    fmla v16.8h, v6.8h, v2.h[2]
    fmla v20.8h, v6.8h, v3.h[2]
    ldr q6, [x17, #0xa0]
    fmla v9.8h, v7.8h, v0.h[2]
    fmla v13.8h, v7.8h, v1.h[2]
    fmla v17.8h, v7.8h, v2.h[2]
    fmla v21.8h, v7.8h, v3.h[2]
    ldr q7, [x17, #0xb0]
    fmla v10.8h, v6.8h, v0.h[2]
    fmla v14.8h, v6.8h, v1.h[2]
    fmla v18.8h, v6.8h, v2.h[2]
    fmla v22.8h, v6.8h, v3.h[2]
    ldr q6, [x17, #0xc0]
    fmla v11.8h, v7.8h, v0.h[2]
    fmla v15.8h, v7.8h, v1.h[2]
    fmla v19.8h, v7.8h, v2.h[2]
    fmla v23.8h, v7.8h, v3.h[2]
    ldr q7, [x17, #0xd0]
    fmla v8.8h, v6.8h, v0.h[3]
    fmla v12.8h, v6.8h, v1.h[3]
    fmla v16.8h, v6.8h, v2.h[3]
    fmla v20.8h, v6.8h, v3.h[3]
    ldr q6, [x17, #0xe0]
    fmla v9.8h, v7.8h, v0.h[3]
    fmla v13.8h, v7.8h, v1.h[3]
    fmla v17.8h, v7.8h, v2.h[3]
    fmla v21.8h, v7.8h, v3.h[3]
    ldr q7, [x17, #0xf0]
    fmla v10.8h, v6.8h, v0.h[3]
    fmla v14.8h, v6.8h, v1.h[3]
    fmla v18.8h, v6.8h, v2.h[3]
    fmla v22.8h, v6.8h, v3.h[3]
    ldr q6, [x17, #0x100]
    fmla v11.8h, v7.8h, v0.h[3]
    fmla v15.8h, v7.8h, v1.h[3]
    fmla v19.8h, v7.8h, v2.h[3]
    fmla v23.8h, v7.8h, v3.h[3]
    ldr q7, [x17, #0x110]
    fmla v8.8h, v6.8h, v0.h[4]
    fmla v12.8h, v6.8h, v1.h[4]
    fmla v16.8h, v6.8h, v2.h[4]
    fmla v20.8h, v6.8h, v3.h[4]
    ldr q6, [x17, #0x120]
    fmla v9.8h, v7.8h, v0.h[4]
    fmla v13.8h, v7.8h, v1.h[4]
    fmla v17.8h, v7.8h, v2.h[4]
    fmla v21.8h, v7.8h, v3.h[4]
    ldr q7, [x17, #0x130]
    fmla v10.8h, v6.8h, v0.h[4]
    fmla v14.8h, v6.8h, v1.h[4]
    fmla v18.8h, v6.8h, v2.h[4]
    fmla v22.8h, v6.8h, v3.h[4]
    ldr q6, [x17, #0x140]
    fmla v11.8h, v7.8h, v0.h[4]
    fmla v15.8h, v7.8h, v1.h[4]
    fmla v19.8h, v7.8h, v2.h[4]
    fmla v23.8h, v7.8h, v3.h[4]
    ldr q7, [x17, #0x150]
    fmla v8.8h, v6.8h, v0.h[5]
    fmla v12.8h, v6.8h, v1.h[5]
    fmla v16.8h, v6.8h, v2.h[5]
    fmla v20.8h, v6.8h, v3.h[5]
    ldr q6, [x17, #0x160]
    fmla v9.8h, v7.8h, v0.h[5]
    fmla v13.8h, v7.8h, v1.h[5]
    fmla v17.8h, v7.8h, v2.h[5]
    fmla v21.8h, v7.8h, v3.h[5]
    ldr q7, [x17, #0x170]
    fmla v10.8h, v6.8h, v0.h[5]
    fmla v14.8h, v6.8h, v1.h[5]
    fmla v18.8h, v6.8h, v2.h[5]
    fmla v22.8h, v6.8h, v3.h[5]
    ldr q6, [x17, #0x180]
    fmla v11.8h, v7.8h, v0.h[5]
    fmla v15.8h, v7.8h, v1.h[5]
    fmla v19.8h, v7.8h, v2.h[5]
    fmla v23.8h, v7.8h, v3.h[5]
    ldr q7, [x17, #0x190]
    fmla v8.8h, v6.8h, v0.h[6]
    fmla v12.8h, v6.8h, v1.h[6]
    fmla v16.8h, v6.8h, v2.h[6]
    fmla v20.8h, v6.8h, v3.h[6]
    ldr q6, [x17, #0x1a0]
    fmla v9.8h, v7.8h, v0.h[6]
    fmla v13.8h, v7.8h, v1.h[6]
    fmla v17.8h, v7.8h, v2.h[6]
    fmla v21.8h, v7.8h, v3.h[6]
    ldr q7, [x17, #0x1b0]
    fmla v10.8h, v6.8h, v0.h[6]
    fmla v14.8h, v6.8h, v1.h[6]
    fmla v18.8h, v6.8h, v2.h[6]
    fmla v22.8h, v6.8h, v3.h[6]
    ldr q6, [x17, #0x1c0]
    fmla v11.8h, v7.8h, v0.h[6]
    fmla v15.8h, v7.8h, v1.h[6]
    fmla v19.8h, v7.8h, v2.h[6]
    fmla v23.8h, v7.8h, v3.h[6]
    ldr q7, [x17, #0x1d0]
    fmla v8.8h, v6.8h, v0.h[7]
    fmla v12.8h, v6.8h, v1.h[7]
    fmla v16.8h, v6.8h, v2.h[7]
    fmla v20.8h, v6.8h, v3.h[7]
    ldr q6, [x17, #0x1e0]
    fmla v9.8h, v7.8h, v0.h[7]
    fmla v13.8h, v7.8h, v1.h[7]
    fmla v17.8h, v7.8h, v2.h[7]
    fmla v21.8h, v7.8h, v3.h[7]
    ldr q7, [x17, #0x1f0]
    fmla v10.8h, v6.8h, v0.h[7]
    add x17, x17, #0x200
    fmla v14.8h, v6.8h, v1.h[7]
    fmla v18.8h, v6.8h, v2.h[7]
    fmla v22.8h, v6.8h, v3.h[7]
    fmla v11.8h, v7.8h, v0.h[7]
    fmla v15.8h, v7.8h, v1.h[7]
    fmla v19.8h, v7.8h, v2.h[7]
    fmla v23.8h, v7.8h, v3.h[7]
KAI_ASM_LABEL(label_175)  // Height 4: Multiply loop: Main loop skip
    cbz x14, label_177
KAI_ASM_LABEL(label_176)  // Height 4: Multiply loop: Odd block loop
    ldr h0, [x13], #0x2
    sub x14, x14, #0x1
    ldr h1, [x12], #0x2
    ldr h2, [x11], #0x2
    ldr h3, [x10], #0x2
    ldr q6, [x17, #0x0]
    ldr q7, [x17, #0x10]
    fmla v8.8h, v6.8h, v0.h[0]
    fmla v12.8h, v6.8h, v1.h[0]
    fmla v16.8h, v6.8h, v2.h[0]
    fmla v20.8h, v6.8h, v3.h[0]
    ldr q6, [x17, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    fmla v13.8h, v7.8h, v1.h[0]
    fmla v17.8h, v7.8h, v2.h[0]
    fmla v21.8h, v7.8h, v3.h[0]
    ldr q7, [x17, #0x30]
    fmla v10.8h, v6.8h, v0.h[0]
    add x17, x17, #0x40
    fmla v14.8h, v6.8h, v1.h[0]
    fmla v18.8h, v6.8h, v2.h[0]
    fmla v22.8h, v6.8h, v3.h[0]
    fmla v11.8h, v7.8h, v0.h[0]
    fmla v15.8h, v7.8h, v1.h[0]
    fmla v19.8h, v7.8h, v2.h[0]
    fmla v23.8h, v7.8h, v3.h[0]
    cbnz x14, label_176
KAI_ASM_LABEL(label_177)  // Height 4: Multiply loop: No odd multiplies
    ldr w20, [x2, #0x4]
    add x15, x15, #0x1
    cmp x15, x20
    bne label_170
    ldr x20, [x2, #0x20]
    prfm pstl1keep, [x16, #0x0]
    add x26, x16, x20, LSL #1
    add x25, x26, x20, LSL #1
    prfm pstl1keep, [x26, #0x0]
    add x24, x25, x20, LSL #1
    prfm pstl1keep, [x25, #0x0]
    prfm pstl1keep, [x24, #0x0]
    tbz x3, #1, label_178
    add x21, x2, #0x0
    add x20, x2, #0x2
    ld1r { v25.8h }, [x21]
    ld1r { v24.8h }, [x20]
    fmin v8.8h, v8.8h, v25.8h
    fmin v9.8h, v9.8h, v25.8h
    fmin v10.8h, v10.8h, v25.8h
    fmin v11.8h, v11.8h, v25.8h
    fmin v12.8h, v12.8h, v25.8h
    fmin v13.8h, v13.8h, v25.8h
    fmin v14.8h, v14.8h, v25.8h
    fmin v15.8h, v15.8h, v25.8h
    fmin v16.8h, v16.8h, v25.8h
    fmin v17.8h, v17.8h, v25.8h
    fmin v18.8h, v18.8h, v25.8h
    fmin v19.8h, v19.8h, v25.8h
    fmin v20.8h, v20.8h, v25.8h
    fmin v21.8h, v21.8h, v25.8h
    fmin v22.8h, v22.8h, v25.8h
    fmin v23.8h, v23.8h, v25.8h
    fmax v8.8h, v8.8h, v24.8h
    fmax v9.8h, v9.8h, v24.8h
    fmax v10.8h, v10.8h, v24.8h
    fmax v11.8h, v11.8h, v24.8h
    fmax v12.8h, v12.8h, v24.8h
    fmax v13.8h, v13.8h, v24.8h
    fmax v14.8h, v14.8h, v24.8h
    fmax v15.8h, v15.8h, v24.8h
    fmax v16.8h, v16.8h, v24.8h
    fmax v17.8h, v17.8h, v24.8h
    fmax v18.8h, v18.8h, v24.8h
    fmax v19.8h, v19.8h, v24.8h
    fmax v20.8h, v20.8h, v24.8h
    fmax v21.8h, v21.8h, v24.8h
    fmax v22.8h, v22.8h, v24.8h
    fmax v23.8h, v23.8h, v24.8h
KAI_ASM_LABEL(label_178)  // Height 4: No activation
    cmp x8, #0x20
    bge label_195
    tbz x8, #4, label_186
    st1 { v8.8h }, [x16], #0x10
    st1 { v9.8h }, [x16], #0x10
    st1 { v12.8h }, [x26], #0x10
    st1 { v13.8h }, [x26], #0x10
    st1 { v16.8h }, [x25], #0x10
    st1 { v17.8h }, [x25], #0x10
    st1 { v20.8h }, [x24], #0x10
    st1 { v21.8h }, [x24], #0x10
    tbz x8, #3, label_182
    st1 { v10.8h }, [x16], #0x10
    st1 { v14.8h }, [x26], #0x10
    st1 { v18.8h }, [x25], #0x10
    st1 { v22.8h }, [x24], #0x10
    tbz x8, #2, label_180
    str d11, [x16], #0x8
    str d15, [x26], #0x8
    str d19, [x25], #0x8
    str d23, [x24], #0x8
    tbz x8, #1, label_179
    st1 { v11.s }[2], [x16], #0x4
    st1 { v15.s }[2], [x26], #0x4
    st1 { v19.s }[2], [x25], #0x4
    st1 { v23.s }[2], [x24], #0x4
    tbz x8, #0, label_194
    st1 { v11.h }[6], [x16]
    st1 { v15.h }[6], [x26]
    st1 { v19.h }[6], [x25]
    st1 { v23.h }[6], [x24]
    b label_194
KAI_ASM_LABEL(label_179)  // Height 4: Partial direct writeback: partial_1_28
    tbz x8, #0, label_194
    st1 { v11.h }[4], [x16]
    st1 { v15.h }[4], [x26]
    st1 { v19.h }[4], [x25]
    st1 { v23.h }[4], [x24]
    b label_194
KAI_ASM_LABEL(label_180)  // Height 4: Partial direct writeback: partial_2_24
    tbz x8, #1, label_181
    str s11, [x16], #0x4
    str s15, [x26], #0x4
    str s19, [x25], #0x4
    str s23, [x24], #0x4
    tbz x8, #0, label_194
    st1 { v11.h }[2], [x16]
    st1 { v15.h }[2], [x26]
    st1 { v19.h }[2], [x25]
    st1 { v23.h }[2], [x24]
    b label_194
KAI_ASM_LABEL(label_181)  // Height 4: Partial direct writeback: partial_1_24
    tbz x8, #0, label_194
    str h11, [x16, #0x0]
    str h15, [x26, #0x0]
    str h19, [x25, #0x0]
    str h23, [x24, #0x0]
    b label_194
KAI_ASM_LABEL(label_182)  // Height 4: Partial direct writeback: partial_4_16
    tbz x8, #2, label_184
    str d10, [x16], #0x8
    str d14, [x26], #0x8
    str d18, [x25], #0x8
    str d22, [x24], #0x8
    tbz x8, #1, label_183
    st1 { v10.s }[2], [x16], #0x4
    st1 { v14.s }[2], [x26], #0x4
    st1 { v18.s }[2], [x25], #0x4
    st1 { v22.s }[2], [x24], #0x4
    tbz x8, #0, label_194
    st1 { v10.h }[6], [x16]
    st1 { v14.h }[6], [x26]
    st1 { v18.h }[6], [x25]
    st1 { v22.h }[6], [x24]
    b label_194
KAI_ASM_LABEL(label_183)  // Height 4: Partial direct writeback: partial_1_20
    tbz x8, #0, label_194
    st1 { v10.h }[4], [x16]
    st1 { v14.h }[4], [x26]
    st1 { v18.h }[4], [x25]
    st1 { v22.h }[4], [x24]
    b label_194
KAI_ASM_LABEL(label_184)  // Height 4: Partial direct writeback: partial_2_16
    tbz x8, #1, label_185
    str s10, [x16], #0x4
    str s14, [x26], #0x4
    str s18, [x25], #0x4
    str s22, [x24], #0x4
    tbz x8, #0, label_194
    st1 { v10.h }[2], [x16]
    st1 { v14.h }[2], [x26]
    st1 { v18.h }[2], [x25]
    st1 { v22.h }[2], [x24]
    b label_194
KAI_ASM_LABEL(label_185)  // Height 4: Partial direct writeback: partial_1_16
    tbz x8, #0, label_194
    str h10, [x16, #0x0]
    str h14, [x26, #0x0]
    str h18, [x25, #0x0]
    str h22, [x24, #0x0]
    b label_194
KAI_ASM_LABEL(label_186)  // Height 4: Partial direct writeback: partial_8_0
    tbz x8, #3, label_190
    st1 { v8.8h }, [x16], #0x10
    st1 { v12.8h }, [x26], #0x10
    st1 { v16.8h }, [x25], #0x10
    st1 { v20.8h }, [x24], #0x10
    tbz x8, #2, label_188
    str d9, [x16], #0x8
    str d13, [x26], #0x8
    str d17, [x25], #0x8
    str d21, [x24], #0x8
    tbz x8, #1, label_187
    st1 { v9.s }[2], [x16], #0x4
    st1 { v13.s }[2], [x26], #0x4
    st1 { v17.s }[2], [x25], #0x4
    st1 { v21.s }[2], [x24], #0x4
    tbz x8, #0, label_194
    st1 { v9.h }[6], [x16]
    st1 { v13.h }[6], [x26]
    st1 { v17.h }[6], [x25]
    st1 { v21.h }[6], [x24]
    b label_194
KAI_ASM_LABEL(label_187)  // Height 4: Partial direct writeback: partial_1_12
    tbz x8, #0, label_194
    st1 { v9.h }[4], [x16]
    st1 { v13.h }[4], [x26]
    st1 { v17.h }[4], [x25]
    st1 { v21.h }[4], [x24]
    b label_194
KAI_ASM_LABEL(label_188)  // Height 4: Partial direct writeback: partial_2_8
    tbz x8, #1, label_189
    str s9, [x16], #0x4
    str s13, [x26], #0x4
    str s17, [x25], #0x4
    str s21, [x24], #0x4
    tbz x8, #0, label_194
    st1 { v9.h }[2], [x16]
    st1 { v13.h }[2], [x26]
    st1 { v17.h }[2], [x25]
    st1 { v21.h }[2], [x24]
    b label_194
KAI_ASM_LABEL(label_189)  // Height 4: Partial direct writeback: partial_1_8
    tbz x8, #0, label_194
    str h9, [x16, #0x0]
    str h13, [x26, #0x0]
    str h17, [x25, #0x0]
    str h21, [x24, #0x0]
    b label_194
KAI_ASM_LABEL(label_190)  // Height 4: Partial direct writeback: partial_4_0
    tbz x8, #2, label_192
    str d8, [x16], #0x8
    str d12, [x26], #0x8
    str d16, [x25], #0x8
    str d20, [x24], #0x8
    tbz x8, #1, label_191
    st1 { v8.s }[2], [x16], #0x4
    st1 { v12.s }[2], [x26], #0x4
    st1 { v16.s }[2], [x25], #0x4
    st1 { v20.s }[2], [x24], #0x4
    tbz x8, #0, label_194
    st1 { v8.h }[6], [x16]
    st1 { v12.h }[6], [x26]
    st1 { v16.h }[6], [x25]
    st1 { v20.h }[6], [x24]
    b label_194
KAI_ASM_LABEL(label_191)  // Height 4: Partial direct writeback: partial_1_4
    tbz x8, #0, label_194
    st1 { v8.h }[4], [x16]
    st1 { v12.h }[4], [x26]
    st1 { v16.h }[4], [x25]
    st1 { v20.h }[4], [x24]
    b label_194
KAI_ASM_LABEL(label_192)  // Height 4: Partial direct writeback: partial_2_0
    tbz x8, #1, label_193
    str s8, [x16], #0x4
    str s12, [x26], #0x4
    str s16, [x25], #0x4
    str s20, [x24], #0x4
    tbz x8, #0, label_194
    st1 { v8.h }[2], [x16]
    st1 { v12.h }[2], [x26]
    st1 { v16.h }[2], [x25]
    st1 { v20.h }[2], [x24]
    b label_194
KAI_ASM_LABEL(label_193)  // Height 4: Partial direct writeback: partial_1_0
    str h8, [x16, #0x0]
    str h12, [x26, #0x0]
    str h16, [x25, #0x0]
    str h20, [x24, #0x0]
KAI_ASM_LABEL(label_194)  // Height 4: Partial direct writeback: Done
    b label_196
KAI_ASM_LABEL(label_195)  // Height 4: Full writeback
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
KAI_ASM_LABEL(label_196)  // Height 4: Writeback done
    subs x8, x8, #0x20
    bgt label_149
    b label_296
KAI_ASM_LABEL(label_197)  // Height 5
    ldr x8, [x2, #0x10]
    ldr x17, [x2, #0x18]
    ldr x16, [x2, #0x38]
KAI_ASM_LABEL(label_198)  // Height 5: Column loop
    cbz x17, label_199
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
    b label_218
KAI_ASM_LABEL(label_199)  // Height 5: no bias
    tbz x3, #0, label_217
    ldr x20, [x2, #0x20]
    cmp x8, #0x20
    add x26, x16, x20, LSL #1
    add x25, x26, x20, LSL #1
    add x24, x25, x20, LSL #1
    add x23, x24, x20, LSL #1
    bge label_216
    tbz x8, #4, label_207
    ld1 { v8.8h }, [x16], #0x10
    ld1 { v12.8h }, [x26], #0x10
    ld1 { v16.8h }, [x25], #0x10
    ld1 { v20.8h }, [x24], #0x10
    ld1 { v24.8h }, [x23], #0x10
    ld1 { v9.8h }, [x16], #0x10
    ld1 { v13.8h }, [x26], #0x10
    ld1 { v17.8h }, [x25], #0x10
    ld1 { v21.8h }, [x24], #0x10
    ld1 { v25.8h }, [x23], #0x10
    tbz x8, #3, label_203
    ld1 { v10.8h }, [x16], #0x10
    ld1 { v14.8h }, [x26], #0x10
    ld1 { v18.8h }, [x25], #0x10
    ld1 { v22.8h }, [x24], #0x10
    ld1 { v26.8h }, [x23], #0x10
    tbz x8, #2, label_201
    ldr d11, [x16], #0x8
    ldr d15, [x26], #0x8
    ldr d19, [x25], #0x8
    ldr d23, [x24], #0x8
    ldr d27, [x23], #0x8
    tbz x8, #1, label_200
    ld1 { v11.s }[2], [x16], #0x4
    mov x20, #0x3c
    ld1 { v15.s }[2], [x26], #0x4
    ld1 { v19.s }[2], [x25], #0x4
    ld1 { v23.s }[2], [x24], #0x4
    ld1 { v27.s }[2], [x23], #0x4
    tbz x8, #0, label_215
    ld1 { v11.h }[6], [x16]
    ld1 { v15.h }[6], [x26]
    ld1 { v19.h }[6], [x25]
    ld1 { v23.h }[6], [x24]
    ld1 { v27.h }[6], [x23]
    b label_215
KAI_ASM_LABEL(label_200)  // Height 5: Partial accumulate: partial_1_28
    mov x20, #0x38
    tbz x8, #0, label_215
    ld1 { v11.h }[4], [x16]
    ld1 { v15.h }[4], [x26]
    ld1 { v19.h }[4], [x25]
    ld1 { v23.h }[4], [x24]
    ld1 { v27.h }[4], [x23]
    b label_215
KAI_ASM_LABEL(label_201)  // Height 5: Partial accumulate: partial_2_24
    tbz x8, #1, label_202
    ldr s11, [x16], #0x4
    mov x20, #0x34
    ldr s15, [x26], #0x4
    ldr s19, [x25], #0x4
    ldr s23, [x24], #0x4
    ldr s27, [x23], #0x4
    tbz x8, #0, label_215
    ld1 { v11.h }[2], [x16]
    ld1 { v15.h }[2], [x26]
    ld1 { v19.h }[2], [x25]
    ld1 { v23.h }[2], [x24]
    ld1 { v27.h }[2], [x23]
    b label_215
KAI_ASM_LABEL(label_202)  // Height 5: Partial accumulate: partial_1_24
    mov x20, #0x30
    tbz x8, #0, label_215
    ldr h11, [x16, #0x0]
    ldr h15, [x26, #0x0]
    ldr h19, [x25, #0x0]
    ldr h23, [x24, #0x0]
    ldr h27, [x23, #0x0]
    b label_215
KAI_ASM_LABEL(label_203)  // Height 5: Partial accumulate: partial_4_16
    tbz x8, #2, label_205
    ldr d10, [x16], #0x8
    ldr d14, [x26], #0x8
    ldr d18, [x25], #0x8
    ldr d22, [x24], #0x8
    ldr d26, [x23], #0x8
    tbz x8, #1, label_204
    ld1 { v10.s }[2], [x16], #0x4
    mov x20, #0x2c
    ld1 { v14.s }[2], [x26], #0x4
    ld1 { v18.s }[2], [x25], #0x4
    ld1 { v22.s }[2], [x24], #0x4
    ld1 { v26.s }[2], [x23], #0x4
    tbz x8, #0, label_215
    ld1 { v10.h }[6], [x16]
    ld1 { v14.h }[6], [x26]
    ld1 { v18.h }[6], [x25]
    ld1 { v22.h }[6], [x24]
    ld1 { v26.h }[6], [x23]
    b label_215
KAI_ASM_LABEL(label_204)  // Height 5: Partial accumulate: partial_1_20
    mov x20, #0x28
    tbz x8, #0, label_215
    ld1 { v10.h }[4], [x16]
    ld1 { v14.h }[4], [x26]
    ld1 { v18.h }[4], [x25]
    ld1 { v22.h }[4], [x24]
    ld1 { v26.h }[4], [x23]
    b label_215
KAI_ASM_LABEL(label_205)  // Height 5: Partial accumulate: partial_2_16
    tbz x8, #1, label_206
    ldr s10, [x16], #0x4
    mov x20, #0x24
    ldr s14, [x26], #0x4
    ldr s18, [x25], #0x4
    ldr s22, [x24], #0x4
    ldr s26, [x23], #0x4
    tbz x8, #0, label_215
    ld1 { v10.h }[2], [x16]
    ld1 { v14.h }[2], [x26]
    ld1 { v18.h }[2], [x25]
    ld1 { v22.h }[2], [x24]
    ld1 { v26.h }[2], [x23]
    b label_215
KAI_ASM_LABEL(label_206)  // Height 5: Partial accumulate: partial_1_16
    mov x20, #0x20
    tbz x8, #0, label_215
    ldr h10, [x16, #0x0]
    ldr h14, [x26, #0x0]
    ldr h18, [x25, #0x0]
    ldr h22, [x24, #0x0]
    ldr h26, [x23, #0x0]
    b label_215
KAI_ASM_LABEL(label_207)  // Height 5: Partial accumulate: partial_8_0
    tbz x8, #3, label_211
    ld1 { v8.8h }, [x16], #0x10
    ld1 { v12.8h }, [x26], #0x10
    ld1 { v16.8h }, [x25], #0x10
    ld1 { v20.8h }, [x24], #0x10
    ld1 { v24.8h }, [x23], #0x10
    tbz x8, #2, label_209
    ldr d9, [x16], #0x8
    ldr d13, [x26], #0x8
    ldr d17, [x25], #0x8
    ldr d21, [x24], #0x8
    ldr d25, [x23], #0x8
    tbz x8, #1, label_208
    ld1 { v9.s }[2], [x16], #0x4
    mov x20, #0x1c
    ld1 { v13.s }[2], [x26], #0x4
    ld1 { v17.s }[2], [x25], #0x4
    ld1 { v21.s }[2], [x24], #0x4
    ld1 { v25.s }[2], [x23], #0x4
    tbz x8, #0, label_215
    ld1 { v9.h }[6], [x16]
    ld1 { v13.h }[6], [x26]
    ld1 { v17.h }[6], [x25]
    ld1 { v21.h }[6], [x24]
    ld1 { v25.h }[6], [x23]
    b label_215
KAI_ASM_LABEL(label_208)  // Height 5: Partial accumulate: partial_1_12
    mov x20, #0x18
    tbz x8, #0, label_215
    ld1 { v9.h }[4], [x16]
    ld1 { v13.h }[4], [x26]
    ld1 { v17.h }[4], [x25]
    ld1 { v21.h }[4], [x24]
    ld1 { v25.h }[4], [x23]
    b label_215
KAI_ASM_LABEL(label_209)  // Height 5: Partial accumulate: partial_2_8
    tbz x8, #1, label_210
    ldr s9, [x16], #0x4
    mov x20, #0x14
    ldr s13, [x26], #0x4
    ldr s17, [x25], #0x4
    ldr s21, [x24], #0x4
    ldr s25, [x23], #0x4
    tbz x8, #0, label_215
    ld1 { v9.h }[2], [x16]
    ld1 { v13.h }[2], [x26]
    ld1 { v17.h }[2], [x25]
    ld1 { v21.h }[2], [x24]
    ld1 { v25.h }[2], [x23]
    b label_215
KAI_ASM_LABEL(label_210)  // Height 5: Partial accumulate: partial_1_8
    mov x20, #0x10
    tbz x8, #0, label_215
    ldr h9, [x16, #0x0]
    ldr h13, [x26, #0x0]
    ldr h17, [x25, #0x0]
    ldr h21, [x24, #0x0]
    ldr h25, [x23, #0x0]
    b label_215
KAI_ASM_LABEL(label_211)  // Height 5: Partial accumulate: partial_4_0
    tbz x8, #2, label_213
    ldr d8, [x16], #0x8
    ldr d12, [x26], #0x8
    ldr d16, [x25], #0x8
    ldr d20, [x24], #0x8
    ldr d24, [x23], #0x8
    tbz x8, #1, label_212
    ld1 { v8.s }[2], [x16], #0x4
    mov x20, #0xc
    ld1 { v12.s }[2], [x26], #0x4
    ld1 { v16.s }[2], [x25], #0x4
    ld1 { v20.s }[2], [x24], #0x4
    ld1 { v24.s }[2], [x23], #0x4
    tbz x8, #0, label_215
    ld1 { v8.h }[6], [x16]
    ld1 { v12.h }[6], [x26]
    ld1 { v16.h }[6], [x25]
    ld1 { v20.h }[6], [x24]
    ld1 { v24.h }[6], [x23]
    b label_215
KAI_ASM_LABEL(label_212)  // Height 5: Partial accumulate: partial_1_4
    mov x20, #0x8
    tbz x8, #0, label_215
    ld1 { v8.h }[4], [x16]
    ld1 { v12.h }[4], [x26]
    ld1 { v16.h }[4], [x25]
    ld1 { v20.h }[4], [x24]
    ld1 { v24.h }[4], [x23]
    b label_215
KAI_ASM_LABEL(label_213)  // Height 5: Partial accumulate: partial_2_0
    tbz x8, #1, label_214
    ldr s8, [x16], #0x4
    mov x20, #0x4
    ldr s12, [x26], #0x4
    ldr s16, [x25], #0x4
    ldr s20, [x24], #0x4
    ldr s24, [x23], #0x4
    tbz x8, #0, label_215
    ld1 { v8.h }[2], [x16]
    ld1 { v12.h }[2], [x26]
    ld1 { v16.h }[2], [x25]
    ld1 { v20.h }[2], [x24]
    ld1 { v24.h }[2], [x23]
    b label_215
KAI_ASM_LABEL(label_214)  // Height 5: Partial accumulate: partial_1_0
    ldr h8, [x16, #0x0]
    mov x20, #0x0
    ldr h12, [x26, #0x0]
    ldr h16, [x25, #0x0]
    ldr h20, [x24, #0x0]
    ldr h24, [x23, #0x0]
KAI_ASM_LABEL(label_215)  // Height 5: Partial accumulate: Done
    sub x16, x16, x20
    b label_218
KAI_ASM_LABEL(label_216)  // Height 5: full accumulate
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
    b label_218
KAI_ASM_LABEL(label_217)  // Height 5: no accumulate
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
KAI_ASM_LABEL(label_218)  // Height 5: setup done
    mov x15, #0x0
KAI_ASM_LABEL(label_219)  // Height 5: String loop
    ldr x20, [x2, #0x8]
    ldr x21, [x2, #0x30]
    ldr w14, [x20, x15, LSL #0x2]
    tbz x3, #3, label_220
    ldr x20, [x0, x15, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x13, [x20, #0x0]
    ldr x12, [x20, #0x8]
    ldr x11, [x20, #0x10]
    ldr x10, [x20, #0x18]
    ldr x9, [x20, #0x20]
    cbnz x15, label_221
    ldr x20, [x2, #0x28]
    add x13, x13, x20, LSL #1
    add x12, x12, x20, LSL #1
    add x11, x11, x20, LSL #1
    add x10, x10, x20, LSL #1
    add x9, x9, x20, LSL #1
    b label_221
KAI_ASM_LABEL(label_220)  // Height 5: setup direct input
    mov x13, x0
    add x12, x13, x21, LSL #1
    add x11, x12, x21, LSL #1
    add x10, x11, x21, LSL #1
    add x9, x10, x21, LSL #1
KAI_ASM_LABEL(label_221)  // Height 5: input setup done
    cmp x14, #0x8
    blt label_224
    ldr q0, [x13, #0x0]
    cmp x14, #0x10
    ldr q1, [x12, #0x0]
    ldr q2, [x11, #0x0]
    ldr q3, [x10, #0x0]
    ldr q4, [x9, #0x0]
    ldr q6, [x17, #0x0]
    ldr q7, [x17, #0x10]
    blt label_223
KAI_ASM_LABEL(label_222)  // Height 5: Multiply loop: Main loop head
    fmla v8.8h, v6.8h, v0.h[0]
    ldr x21, [x17, #0x28]
    fmla v12.8h, v6.8h, v1.h[0]
    ldr x20, [x17, #0x38]
    fmla v16.8h, v6.8h, v2.h[0]
    add x13, x13, #0x10
    fmla v20.8h, v6.8h, v3.h[0]
    add x12, x12, #0x10
    fmla v24.8h, v6.8h, v4.h[0]
    ldr d6, [x17, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    add x11, x11, #0x10
    fmla v13.8h, v7.8h, v1.h[0]
    add x10, x10, #0x10
    fmla v17.8h, v7.8h, v2.h[0]
    mov v6.d[1], x21
    fmla v21.8h, v7.8h, v3.h[0]
    ldr x21, [x17, #0x48]
    fmla v25.8h, v7.8h, v4.h[0]
    ldr d7, [x17, #0x30]
    fmla v10.8h, v6.8h, v0.h[0]
    add x9, x9, #0x10
    fmla v14.8h, v6.8h, v1.h[0]
    ldr x26, [x13, #0x8]
    mov v7.d[1], x20
    fmla v18.8h, v6.8h, v2.h[0]
    fmla v22.8h, v6.8h, v3.h[0]
    ldr x20, [x17, #0x58]
    fmla v26.8h, v6.8h, v4.h[0]
    ldr d6, [x17, #0x40]
    fmla v11.8h, v7.8h, v0.h[0]
    ldr x25, [x12, #0x8]
    fmla v15.8h, v7.8h, v1.h[0]
    ldr x24, [x11, #0x8]
    fmla v19.8h, v7.8h, v2.h[0]
    mov v6.d[1], x21
    fmla v23.8h, v7.8h, v3.h[0]
    ldr x21, [x17, #0x68]
    fmla v27.8h, v7.8h, v4.h[0]
    ldr d7, [x17, #0x50]
    fmla v8.8h, v6.8h, v0.h[1]
    ldr x23, [x10, #0x8]
    fmla v12.8h, v6.8h, v1.h[1]
    ldr x22, [x9, #0x8]
    mov v7.d[1], x20
    fmla v16.8h, v6.8h, v2.h[1]
    fmla v20.8h, v6.8h, v3.h[1]
    ldr x20, [x17, #0x78]
    fmla v24.8h, v6.8h, v4.h[1]
    ldr d6, [x17, #0x60]
    fmla v9.8h, v7.8h, v0.h[1]
    sub x14, x14, #0x8
    fmla v13.8h, v7.8h, v1.h[1]
    cmp x14, #0x10
    fmla v17.8h, v7.8h, v2.h[1]
    mov v6.d[1], x21
    fmla v21.8h, v7.8h, v3.h[1]
    ldr x21, [x17, #0x88]
    fmla v25.8h, v7.8h, v4.h[1]
    ldr d7, [x17, #0x70]
    fmla v10.8h, v6.8h, v0.h[1]
    prfm pldl1keep, [x13, #0x80]
    fmla v14.8h, v6.8h, v1.h[1]
    prfm pldl1keep, [x12, #0x80]
    mov v7.d[1], x20
    fmla v18.8h, v6.8h, v2.h[1]
    fmla v22.8h, v6.8h, v3.h[1]
    ldr x20, [x17, #0x98]
    fmla v26.8h, v6.8h, v4.h[1]
    ldr d6, [x17, #0x80]
    fmla v11.8h, v7.8h, v0.h[1]
    prfm pldl1keep, [x11, #0x80]
    fmla v15.8h, v7.8h, v1.h[1]
    prfm pldl1keep, [x10, #0x80]
    fmla v19.8h, v7.8h, v2.h[1]
    mov v6.d[1], x21
    fmla v23.8h, v7.8h, v3.h[1]
    ldr x21, [x17, #0xa8]
    fmla v27.8h, v7.8h, v4.h[1]
    ldr d7, [x17, #0x90]
    fmla v8.8h, v6.8h, v0.h[2]
    prfm pldl1keep, [x9, #0x80]
    fmla v12.8h, v6.8h, v1.h[2]
    mov v7.d[1], x20
    fmla v16.8h, v6.8h, v2.h[2]
    fmla v20.8h, v6.8h, v3.h[2]
    ldr x20, [x17, #0xb8]
    fmla v24.8h, v6.8h, v4.h[2]
    ldr d6, [x17, #0xa0]
    fmla v9.8h, v7.8h, v0.h[2]
    fmla v13.8h, v7.8h, v1.h[2]
    fmla v17.8h, v7.8h, v2.h[2]
    mov v6.d[1], x21
    fmla v21.8h, v7.8h, v3.h[2]
    ldr x21, [x17, #0xc8]
    fmla v25.8h, v7.8h, v4.h[2]
    ldr d7, [x17, #0xb0]
    fmla v10.8h, v6.8h, v0.h[2]
    fmla v14.8h, v6.8h, v1.h[2]
    mov v7.d[1], x20
    fmla v18.8h, v6.8h, v2.h[2]
    fmla v22.8h, v6.8h, v3.h[2]
    ldr x20, [x17, #0xd8]
    fmla v26.8h, v6.8h, v4.h[2]
    ldr d6, [x17, #0xc0]
    fmla v11.8h, v7.8h, v0.h[2]
    fmla v15.8h, v7.8h, v1.h[2]
    fmla v19.8h, v7.8h, v2.h[2]
    mov v6.d[1], x21
    fmla v23.8h, v7.8h, v3.h[2]
    ldr x21, [x17, #0xe8]
    fmla v27.8h, v7.8h, v4.h[2]
    ldr d7, [x17, #0xd0]
    fmla v8.8h, v6.8h, v0.h[3]
    fmla v12.8h, v6.8h, v1.h[3]
    mov v7.d[1], x20
    fmla v16.8h, v6.8h, v2.h[3]
    fmla v20.8h, v6.8h, v3.h[3]
    ldr x20, [x17, #0xf8]
    fmla v24.8h, v6.8h, v4.h[3]
    ldr d6, [x17, #0xe0]
    fmla v9.8h, v7.8h, v0.h[3]
    fmla v13.8h, v7.8h, v1.h[3]
    fmla v17.8h, v7.8h, v2.h[3]
    mov v6.d[1], x21
    fmla v21.8h, v7.8h, v3.h[3]
    ldr x21, [x17, #0x108]
    fmla v25.8h, v7.8h, v4.h[3]
    ldr d7, [x17, #0xf0]
    fmla v10.8h, v6.8h, v0.h[3]
    fmla v14.8h, v6.8h, v1.h[3]
    mov v7.d[1], x20
    fmla v18.8h, v6.8h, v2.h[3]
    fmla v22.8h, v6.8h, v3.h[3]
    ldr x20, [x17, #0x118]
    fmla v26.8h, v6.8h, v4.h[3]
    ldr d6, [x17, #0x100]
    fmla v11.8h, v7.8h, v0.h[3]
    fmla v15.8h, v7.8h, v1.h[3]
    fmla v19.8h, v7.8h, v2.h[3]
    mov v6.d[1], x21
    fmla v23.8h, v7.8h, v3.h[3]
    ldr x21, [x17, #0x128]
    fmla v27.8h, v7.8h, v4.h[3]
    ldr d7, [x17, #0x110]
    fmla v8.8h, v6.8h, v0.h[4]
    fmla v12.8h, v6.8h, v1.h[4]
    mov v7.d[1], x20
    fmla v16.8h, v6.8h, v2.h[4]
    fmla v20.8h, v6.8h, v3.h[4]
    ldr x20, [x17, #0x138]
    fmla v24.8h, v6.8h, v4.h[4]
    ldr d6, [x17, #0x120]
    fmla v9.8h, v7.8h, v0.h[4]
    fmla v13.8h, v7.8h, v1.h[4]
    fmla v17.8h, v7.8h, v2.h[4]
    mov v6.d[1], x21
    fmla v21.8h, v7.8h, v3.h[4]
    ldr x21, [x17, #0x148]
    fmla v25.8h, v7.8h, v4.h[4]
    ldr d7, [x17, #0x130]
    fmla v10.8h, v6.8h, v0.h[4]
    fmla v14.8h, v6.8h, v1.h[4]
    mov v7.d[1], x20
    fmla v18.8h, v6.8h, v2.h[4]
    fmla v22.8h, v6.8h, v3.h[4]
    ldr x20, [x17, #0x158]
    fmla v26.8h, v6.8h, v4.h[4]
    ldr d6, [x17, #0x140]
    fmla v11.8h, v7.8h, v0.h[4]
    fmla v15.8h, v7.8h, v1.h[4]
    fmla v19.8h, v7.8h, v2.h[4]
    mov v6.d[1], x21
    fmla v23.8h, v7.8h, v3.h[4]
    ldr x21, [x17, #0x168]
    fmla v27.8h, v7.8h, v4.h[4]
    ldr d7, [x17, #0x150]
    fmla v8.8h, v6.8h, v0.h[5]
    fmla v12.8h, v6.8h, v1.h[5]
    mov v7.d[1], x20
    fmla v16.8h, v6.8h, v2.h[5]
    fmla v20.8h, v6.8h, v3.h[5]
    ldr x20, [x17, #0x178]
    fmla v24.8h, v6.8h, v4.h[5]
    ldr d6, [x17, #0x160]
    fmla v9.8h, v7.8h, v0.h[5]
    fmla v13.8h, v7.8h, v1.h[5]
    fmla v17.8h, v7.8h, v2.h[5]
    mov v6.d[1], x21
    fmla v21.8h, v7.8h, v3.h[5]
    ldr x21, [x17, #0x188]
    fmla v25.8h, v7.8h, v4.h[5]
    ldr d7, [x17, #0x170]
    fmla v10.8h, v6.8h, v0.h[5]
    fmla v14.8h, v6.8h, v1.h[5]
    mov v7.d[1], x20
    fmla v18.8h, v6.8h, v2.h[5]
    fmla v22.8h, v6.8h, v3.h[5]
    ldr x20, [x17, #0x198]
    fmla v26.8h, v6.8h, v4.h[5]
    ldr d6, [x17, #0x180]
    fmla v11.8h, v7.8h, v0.h[5]
    fmla v15.8h, v7.8h, v1.h[5]
    fmla v19.8h, v7.8h, v2.h[5]
    mov v6.d[1], x21
    fmla v23.8h, v7.8h, v3.h[5]
    ldr x21, [x17, #0x1a8]
    fmla v27.8h, v7.8h, v4.h[5]
    ldr d7, [x17, #0x190]
    fmla v8.8h, v6.8h, v0.h[6]
    fmla v12.8h, v6.8h, v1.h[6]
    mov v7.d[1], x20
    fmla v16.8h, v6.8h, v2.h[6]
    fmla v20.8h, v6.8h, v3.h[6]
    ldr x20, [x17, #0x1b8]
    fmla v24.8h, v6.8h, v4.h[6]
    ldr d6, [x17, #0x1a0]
    fmla v9.8h, v7.8h, v0.h[6]
    fmla v13.8h, v7.8h, v1.h[6]
    fmla v17.8h, v7.8h, v2.h[6]
    mov v6.d[1], x21
    fmla v21.8h, v7.8h, v3.h[6]
    ldr x21, [x17, #0x1c8]
    fmla v25.8h, v7.8h, v4.h[6]
    ldr d7, [x17, #0x1b0]
    fmla v10.8h, v6.8h, v0.h[6]
    fmla v14.8h, v6.8h, v1.h[6]
    mov v7.d[1], x20
    fmla v18.8h, v6.8h, v2.h[6]
    fmla v22.8h, v6.8h, v3.h[6]
    ldr x20, [x17, #0x1d8]
    fmla v26.8h, v6.8h, v4.h[6]
    ldr d6, [x17, #0x1c0]
    fmla v11.8h, v7.8h, v0.h[6]
    fmla v15.8h, v7.8h, v1.h[6]
    fmla v19.8h, v7.8h, v2.h[6]
    mov v6.d[1], x21
    fmla v23.8h, v7.8h, v3.h[6]
    ldr x21, [x17, #0x1e8]
    fmla v27.8h, v7.8h, v4.h[6]
    ldr d7, [x17, #0x1d0]
    fmla v8.8h, v6.8h, v0.h[7]
    fmla v12.8h, v6.8h, v1.h[7]
    mov v7.d[1], x20
    fmla v16.8h, v6.8h, v2.h[7]
    fmla v20.8h, v6.8h, v3.h[7]
    ldr x20, [x17, #0x1f8]
    fmla v24.8h, v6.8h, v4.h[7]
    ldr d6, [x17, #0x1e0]
    fmla v9.8h, v7.8h, v0.h[7]
    fmla v13.8h, v7.8h, v1.h[7]
    fmla v17.8h, v7.8h, v2.h[7]
    mov v6.d[1], x21
    fmla v21.8h, v7.8h, v3.h[7]
    fmla v25.8h, v7.8h, v4.h[7]
    ldr d7, [x17, #0x1f0]
    add x17, x17, #0x200
    fmla v10.8h, v6.8h, v0.h[7]
    fmla v14.8h, v6.8h, v1.h[7]
    ldr x21, [x17, #0x8]
    mov v7.d[1], x20
    fmla v18.8h, v6.8h, v2.h[7]
    fmla v22.8h, v6.8h, v3.h[7]
    ldr x20, [x17, #0x18]
    fmla v26.8h, v6.8h, v4.h[7]
    ldr d6, [x17, #0x0]
    fmla v11.8h, v7.8h, v0.h[7]
    ldr d0, [x13, #0x0]
    fmla v15.8h, v7.8h, v1.h[7]
    ldr d1, [x12, #0x0]
    fmla v19.8h, v7.8h, v2.h[7]
    ldr d2, [x11, #0x0]
    fmla v23.8h, v7.8h, v3.h[7]
    ldr d3, [x10, #0x0]
    fmla v27.8h, v7.8h, v4.h[7]
    ldr d4, [x9, #0x0]
    ldr d7, [x17, #0x10]
    mov v6.d[1], x21
    mov v0.d[1], x26
    mov v1.d[1], x25
    mov v2.d[1], x24
    mov v3.d[1], x23
    mov v4.d[1], x22
    mov v7.d[1], x20
    bge label_222
KAI_ASM_LABEL(label_223)  // Height 5: Multiply loop: Single iteration only
    fmla v8.8h, v6.8h, v0.h[0]
    add x13, x13, #0x10
    fmla v12.8h, v6.8h, v1.h[0]
    add x12, x12, #0x10
    fmla v16.8h, v6.8h, v2.h[0]
    add x11, x11, #0x10
    fmla v20.8h, v6.8h, v3.h[0]
    add x10, x10, #0x10
    fmla v24.8h, v6.8h, v4.h[0]
    ldr q6, [x17, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    add x9, x9, #0x10
    fmla v13.8h, v7.8h, v1.h[0]
    sub x14, x14, #0x8
    fmla v17.8h, v7.8h, v2.h[0]
    prfm pldl1keep, [x13, #0x80]
    fmla v21.8h, v7.8h, v3.h[0]
    prfm pldl1keep, [x12, #0x80]
    fmla v25.8h, v7.8h, v4.h[0]
    ldr q7, [x17, #0x30]
    fmla v10.8h, v6.8h, v0.h[0]
    prfm pldl1keep, [x11, #0x80]
    fmla v14.8h, v6.8h, v1.h[0]
    prfm pldl1keep, [x10, #0x80]
    fmla v18.8h, v6.8h, v2.h[0]
    prfm pldl1keep, [x9, #0x80]
    fmla v22.8h, v6.8h, v3.h[0]
    fmla v26.8h, v6.8h, v4.h[0]
    ldr q6, [x17, #0x40]
    fmla v11.8h, v7.8h, v0.h[0]
    fmla v15.8h, v7.8h, v1.h[0]
    fmla v19.8h, v7.8h, v2.h[0]
    fmla v23.8h, v7.8h, v3.h[0]
    fmla v27.8h, v7.8h, v4.h[0]
    ldr q7, [x17, #0x50]
    fmla v8.8h, v6.8h, v0.h[1]
    fmla v12.8h, v6.8h, v1.h[1]
    fmla v16.8h, v6.8h, v2.h[1]
    fmla v20.8h, v6.8h, v3.h[1]
    fmla v24.8h, v6.8h, v4.h[1]
    ldr q6, [x17, #0x60]
    fmla v9.8h, v7.8h, v0.h[1]
    fmla v13.8h, v7.8h, v1.h[1]
    fmla v17.8h, v7.8h, v2.h[1]
    fmla v21.8h, v7.8h, v3.h[1]
    fmla v25.8h, v7.8h, v4.h[1]
    ldr q7, [x17, #0x70]
    fmla v10.8h, v6.8h, v0.h[1]
    fmla v14.8h, v6.8h, v1.h[1]
    fmla v18.8h, v6.8h, v2.h[1]
    fmla v22.8h, v6.8h, v3.h[1]
    fmla v26.8h, v6.8h, v4.h[1]
    ldr q6, [x17, #0x80]
    fmla v11.8h, v7.8h, v0.h[1]
    fmla v15.8h, v7.8h, v1.h[1]
    fmla v19.8h, v7.8h, v2.h[1]
    fmla v23.8h, v7.8h, v3.h[1]
    fmla v27.8h, v7.8h, v4.h[1]
    ldr q7, [x17, #0x90]
    fmla v8.8h, v6.8h, v0.h[2]
    fmla v12.8h, v6.8h, v1.h[2]
    fmla v16.8h, v6.8h, v2.h[2]
    fmla v20.8h, v6.8h, v3.h[2]
    fmla v24.8h, v6.8h, v4.h[2]
    ldr q6, [x17, #0xa0]
    fmla v9.8h, v7.8h, v0.h[2]
    fmla v13.8h, v7.8h, v1.h[2]
    fmla v17.8h, v7.8h, v2.h[2]
    fmla v21.8h, v7.8h, v3.h[2]
    fmla v25.8h, v7.8h, v4.h[2]
    ldr q7, [x17, #0xb0]
    fmla v10.8h, v6.8h, v0.h[2]
    fmla v14.8h, v6.8h, v1.h[2]
    fmla v18.8h, v6.8h, v2.h[2]
    fmla v22.8h, v6.8h, v3.h[2]
    fmla v26.8h, v6.8h, v4.h[2]
    ldr q6, [x17, #0xc0]
    fmla v11.8h, v7.8h, v0.h[2]
    fmla v15.8h, v7.8h, v1.h[2]
    fmla v19.8h, v7.8h, v2.h[2]
    fmla v23.8h, v7.8h, v3.h[2]
    fmla v27.8h, v7.8h, v4.h[2]
    ldr q7, [x17, #0xd0]
    fmla v8.8h, v6.8h, v0.h[3]
    fmla v12.8h, v6.8h, v1.h[3]
    fmla v16.8h, v6.8h, v2.h[3]
    fmla v20.8h, v6.8h, v3.h[3]
    fmla v24.8h, v6.8h, v4.h[3]
    ldr q6, [x17, #0xe0]
    fmla v9.8h, v7.8h, v0.h[3]
    fmla v13.8h, v7.8h, v1.h[3]
    fmla v17.8h, v7.8h, v2.h[3]
    fmla v21.8h, v7.8h, v3.h[3]
    fmla v25.8h, v7.8h, v4.h[3]
    ldr q7, [x17, #0xf0]
    fmla v10.8h, v6.8h, v0.h[3]
    fmla v14.8h, v6.8h, v1.h[3]
    fmla v18.8h, v6.8h, v2.h[3]
    fmla v22.8h, v6.8h, v3.h[3]
    fmla v26.8h, v6.8h, v4.h[3]
    ldr q6, [x17, #0x100]
    fmla v11.8h, v7.8h, v0.h[3]
    fmla v15.8h, v7.8h, v1.h[3]
    fmla v19.8h, v7.8h, v2.h[3]
    fmla v23.8h, v7.8h, v3.h[3]
    fmla v27.8h, v7.8h, v4.h[3]
    ldr q7, [x17, #0x110]
    fmla v8.8h, v6.8h, v0.h[4]
    fmla v12.8h, v6.8h, v1.h[4]
    fmla v16.8h, v6.8h, v2.h[4]
    fmla v20.8h, v6.8h, v3.h[4]
    fmla v24.8h, v6.8h, v4.h[4]
    ldr q6, [x17, #0x120]
    fmla v9.8h, v7.8h, v0.h[4]
    fmla v13.8h, v7.8h, v1.h[4]
    fmla v17.8h, v7.8h, v2.h[4]
    fmla v21.8h, v7.8h, v3.h[4]
    fmla v25.8h, v7.8h, v4.h[4]
    ldr q7, [x17, #0x130]
    fmla v10.8h, v6.8h, v0.h[4]
    fmla v14.8h, v6.8h, v1.h[4]
    fmla v18.8h, v6.8h, v2.h[4]
    fmla v22.8h, v6.8h, v3.h[4]
    fmla v26.8h, v6.8h, v4.h[4]
    ldr q6, [x17, #0x140]
    fmla v11.8h, v7.8h, v0.h[4]
    fmla v15.8h, v7.8h, v1.h[4]
    fmla v19.8h, v7.8h, v2.h[4]
    fmla v23.8h, v7.8h, v3.h[4]
    fmla v27.8h, v7.8h, v4.h[4]
    ldr q7, [x17, #0x150]
    fmla v8.8h, v6.8h, v0.h[5]
    fmla v12.8h, v6.8h, v1.h[5]
    fmla v16.8h, v6.8h, v2.h[5]
    fmla v20.8h, v6.8h, v3.h[5]
    fmla v24.8h, v6.8h, v4.h[5]
    ldr q6, [x17, #0x160]
    fmla v9.8h, v7.8h, v0.h[5]
    fmla v13.8h, v7.8h, v1.h[5]
    fmla v17.8h, v7.8h, v2.h[5]
    fmla v21.8h, v7.8h, v3.h[5]
    fmla v25.8h, v7.8h, v4.h[5]
    ldr q7, [x17, #0x170]
    fmla v10.8h, v6.8h, v0.h[5]
    fmla v14.8h, v6.8h, v1.h[5]
    fmla v18.8h, v6.8h, v2.h[5]
    fmla v22.8h, v6.8h, v3.h[5]
    fmla v26.8h, v6.8h, v4.h[5]
    ldr q6, [x17, #0x180]
    fmla v11.8h, v7.8h, v0.h[5]
    fmla v15.8h, v7.8h, v1.h[5]
    fmla v19.8h, v7.8h, v2.h[5]
    fmla v23.8h, v7.8h, v3.h[5]
    fmla v27.8h, v7.8h, v4.h[5]
    ldr q7, [x17, #0x190]
    fmla v8.8h, v6.8h, v0.h[6]
    fmla v12.8h, v6.8h, v1.h[6]
    fmla v16.8h, v6.8h, v2.h[6]
    fmla v20.8h, v6.8h, v3.h[6]
    fmla v24.8h, v6.8h, v4.h[6]
    ldr q6, [x17, #0x1a0]
    fmla v9.8h, v7.8h, v0.h[6]
    fmla v13.8h, v7.8h, v1.h[6]
    fmla v17.8h, v7.8h, v2.h[6]
    fmla v21.8h, v7.8h, v3.h[6]
    fmla v25.8h, v7.8h, v4.h[6]
    ldr q7, [x17, #0x1b0]
    fmla v10.8h, v6.8h, v0.h[6]
    fmla v14.8h, v6.8h, v1.h[6]
    fmla v18.8h, v6.8h, v2.h[6]
    fmla v22.8h, v6.8h, v3.h[6]
    fmla v26.8h, v6.8h, v4.h[6]
    ldr q6, [x17, #0x1c0]
    fmla v11.8h, v7.8h, v0.h[6]
    fmla v15.8h, v7.8h, v1.h[6]
    fmla v19.8h, v7.8h, v2.h[6]
    fmla v23.8h, v7.8h, v3.h[6]
    fmla v27.8h, v7.8h, v4.h[6]
    ldr q7, [x17, #0x1d0]
    fmla v8.8h, v6.8h, v0.h[7]
    fmla v12.8h, v6.8h, v1.h[7]
    fmla v16.8h, v6.8h, v2.h[7]
    fmla v20.8h, v6.8h, v3.h[7]
    fmla v24.8h, v6.8h, v4.h[7]
    ldr q6, [x17, #0x1e0]
    fmla v9.8h, v7.8h, v0.h[7]
    fmla v13.8h, v7.8h, v1.h[7]
    fmla v17.8h, v7.8h, v2.h[7]
    fmla v21.8h, v7.8h, v3.h[7]
    fmla v25.8h, v7.8h, v4.h[7]
    ldr q7, [x17, #0x1f0]
    fmla v10.8h, v6.8h, v0.h[7]
    add x17, x17, #0x200
    fmla v14.8h, v6.8h, v1.h[7]
    fmla v18.8h, v6.8h, v2.h[7]
    fmla v22.8h, v6.8h, v3.h[7]
    fmla v26.8h, v6.8h, v4.h[7]
    fmla v11.8h, v7.8h, v0.h[7]
    fmla v15.8h, v7.8h, v1.h[7]
    fmla v19.8h, v7.8h, v2.h[7]
    fmla v23.8h, v7.8h, v3.h[7]
    fmla v27.8h, v7.8h, v4.h[7]
KAI_ASM_LABEL(label_224)  // Height 5: Multiply loop: Main loop skip
    cbz x14, label_226
KAI_ASM_LABEL(label_225)  // Height 5: Multiply loop: Odd block loop
    ldr h0, [x13], #0x2
    sub x14, x14, #0x1
    ldr h1, [x12], #0x2
    ldr h2, [x11], #0x2
    ldr h3, [x10], #0x2
    ldr h4, [x9], #0x2
    ldr q6, [x17, #0x0]
    ldr q7, [x17, #0x10]
    fmla v8.8h, v6.8h, v0.h[0]
    fmla v12.8h, v6.8h, v1.h[0]
    fmla v16.8h, v6.8h, v2.h[0]
    fmla v20.8h, v6.8h, v3.h[0]
    fmla v24.8h, v6.8h, v4.h[0]
    ldr q6, [x17, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    fmla v13.8h, v7.8h, v1.h[0]
    fmla v17.8h, v7.8h, v2.h[0]
    fmla v21.8h, v7.8h, v3.h[0]
    fmla v25.8h, v7.8h, v4.h[0]
    ldr q7, [x17, #0x30]
    fmla v10.8h, v6.8h, v0.h[0]
    add x17, x17, #0x40
    fmla v14.8h, v6.8h, v1.h[0]
    fmla v18.8h, v6.8h, v2.h[0]
    fmla v22.8h, v6.8h, v3.h[0]
    fmla v26.8h, v6.8h, v4.h[0]
    fmla v11.8h, v7.8h, v0.h[0]
    fmla v15.8h, v7.8h, v1.h[0]
    fmla v19.8h, v7.8h, v2.h[0]
    fmla v23.8h, v7.8h, v3.h[0]
    fmla v27.8h, v7.8h, v4.h[0]
    cbnz x14, label_225
KAI_ASM_LABEL(label_226)  // Height 5: Multiply loop: No odd multiplies
    ldr w20, [x2, #0x4]
    add x15, x15, #0x1
    cmp x15, x20
    bne label_219
    ldr x20, [x2, #0x20]
    prfm pstl1keep, [x16, #0x0]
    add x26, x16, x20, LSL #1
    add x25, x26, x20, LSL #1
    prfm pstl1keep, [x26, #0x0]
    add x24, x25, x20, LSL #1
    prfm pstl1keep, [x25, #0x0]
    add x23, x24, x20, LSL #1
    prfm pstl1keep, [x24, #0x0]
    prfm pstl1keep, [x23, #0x0]
    tbz x3, #1, label_227
    add x21, x2, #0x0
    add x20, x2, #0x2
    ld1r { v29.8h }, [x21]
    ld1r { v28.8h }, [x20]
    fmin v8.8h, v8.8h, v29.8h
    fmin v9.8h, v9.8h, v29.8h
    fmin v10.8h, v10.8h, v29.8h
    fmin v11.8h, v11.8h, v29.8h
    fmin v12.8h, v12.8h, v29.8h
    fmin v13.8h, v13.8h, v29.8h
    fmin v14.8h, v14.8h, v29.8h
    fmin v15.8h, v15.8h, v29.8h
    fmin v16.8h, v16.8h, v29.8h
    fmin v17.8h, v17.8h, v29.8h
    fmin v18.8h, v18.8h, v29.8h
    fmin v19.8h, v19.8h, v29.8h
    fmin v20.8h, v20.8h, v29.8h
    fmin v21.8h, v21.8h, v29.8h
    fmin v22.8h, v22.8h, v29.8h
    fmin v23.8h, v23.8h, v29.8h
    fmin v24.8h, v24.8h, v29.8h
    fmin v25.8h, v25.8h, v29.8h
    fmin v26.8h, v26.8h, v29.8h
    fmin v27.8h, v27.8h, v29.8h
    fmax v8.8h, v8.8h, v28.8h
    fmax v9.8h, v9.8h, v28.8h
    fmax v10.8h, v10.8h, v28.8h
    fmax v11.8h, v11.8h, v28.8h
    fmax v12.8h, v12.8h, v28.8h
    fmax v13.8h, v13.8h, v28.8h
    fmax v14.8h, v14.8h, v28.8h
    fmax v15.8h, v15.8h, v28.8h
    fmax v16.8h, v16.8h, v28.8h
    fmax v17.8h, v17.8h, v28.8h
    fmax v18.8h, v18.8h, v28.8h
    fmax v19.8h, v19.8h, v28.8h
    fmax v20.8h, v20.8h, v28.8h
    fmax v21.8h, v21.8h, v28.8h
    fmax v22.8h, v22.8h, v28.8h
    fmax v23.8h, v23.8h, v28.8h
    fmax v24.8h, v24.8h, v28.8h
    fmax v25.8h, v25.8h, v28.8h
    fmax v26.8h, v26.8h, v28.8h
    fmax v27.8h, v27.8h, v28.8h
KAI_ASM_LABEL(label_227)  // Height 5: No activation
    cmp x8, #0x20
    bge label_244
    tbz x8, #4, label_235
    st1 { v8.8h }, [x16], #0x10
    st1 { v9.8h }, [x16], #0x10
    st1 { v12.8h }, [x26], #0x10
    st1 { v13.8h }, [x26], #0x10
    st1 { v16.8h }, [x25], #0x10
    st1 { v17.8h }, [x25], #0x10
    st1 { v20.8h }, [x24], #0x10
    st1 { v21.8h }, [x24], #0x10
    st1 { v24.8h }, [x23], #0x10
    st1 { v25.8h }, [x23], #0x10
    tbz x8, #3, label_231
    st1 { v10.8h }, [x16], #0x10
    st1 { v14.8h }, [x26], #0x10
    st1 { v18.8h }, [x25], #0x10
    st1 { v22.8h }, [x24], #0x10
    st1 { v26.8h }, [x23], #0x10
    tbz x8, #2, label_229
    str d11, [x16], #0x8
    str d15, [x26], #0x8
    str d19, [x25], #0x8
    str d23, [x24], #0x8
    str d27, [x23], #0x8
    tbz x8, #1, label_228
    st1 { v11.s }[2], [x16], #0x4
    st1 { v15.s }[2], [x26], #0x4
    st1 { v19.s }[2], [x25], #0x4
    st1 { v23.s }[2], [x24], #0x4
    st1 { v27.s }[2], [x23], #0x4
    tbz x8, #0, label_243
    st1 { v11.h }[6], [x16]
    st1 { v15.h }[6], [x26]
    st1 { v19.h }[6], [x25]
    st1 { v23.h }[6], [x24]
    st1 { v27.h }[6], [x23]
    b label_243
KAI_ASM_LABEL(label_228)  // Height 5: Partial direct writeback: partial_1_28
    tbz x8, #0, label_243
    st1 { v11.h }[4], [x16]
    st1 { v15.h }[4], [x26]
    st1 { v19.h }[4], [x25]
    st1 { v23.h }[4], [x24]
    st1 { v27.h }[4], [x23]
    b label_243
KAI_ASM_LABEL(label_229)  // Height 5: Partial direct writeback: partial_2_24
    tbz x8, #1, label_230
    str s11, [x16], #0x4
    str s15, [x26], #0x4
    str s19, [x25], #0x4
    str s23, [x24], #0x4
    str s27, [x23], #0x4
    tbz x8, #0, label_243
    st1 { v11.h }[2], [x16]
    st1 { v15.h }[2], [x26]
    st1 { v19.h }[2], [x25]
    st1 { v23.h }[2], [x24]
    st1 { v27.h }[2], [x23]
    b label_243
KAI_ASM_LABEL(label_230)  // Height 5: Partial direct writeback: partial_1_24
    tbz x8, #0, label_243
    str h11, [x16, #0x0]
    str h15, [x26, #0x0]
    str h19, [x25, #0x0]
    str h23, [x24, #0x0]
    str h27, [x23, #0x0]
    b label_243
KAI_ASM_LABEL(label_231)  // Height 5: Partial direct writeback: partial_4_16
    tbz x8, #2, label_233
    str d10, [x16], #0x8
    str d14, [x26], #0x8
    str d18, [x25], #0x8
    str d22, [x24], #0x8
    str d26, [x23], #0x8
    tbz x8, #1, label_232
    st1 { v10.s }[2], [x16], #0x4
    st1 { v14.s }[2], [x26], #0x4
    st1 { v18.s }[2], [x25], #0x4
    st1 { v22.s }[2], [x24], #0x4
    st1 { v26.s }[2], [x23], #0x4
    tbz x8, #0, label_243
    st1 { v10.h }[6], [x16]
    st1 { v14.h }[6], [x26]
    st1 { v18.h }[6], [x25]
    st1 { v22.h }[6], [x24]
    st1 { v26.h }[6], [x23]
    b label_243
KAI_ASM_LABEL(label_232)  // Height 5: Partial direct writeback: partial_1_20
    tbz x8, #0, label_243
    st1 { v10.h }[4], [x16]
    st1 { v14.h }[4], [x26]
    st1 { v18.h }[4], [x25]
    st1 { v22.h }[4], [x24]
    st1 { v26.h }[4], [x23]
    b label_243
KAI_ASM_LABEL(label_233)  // Height 5: Partial direct writeback: partial_2_16
    tbz x8, #1, label_234
    str s10, [x16], #0x4
    str s14, [x26], #0x4
    str s18, [x25], #0x4
    str s22, [x24], #0x4
    str s26, [x23], #0x4
    tbz x8, #0, label_243
    st1 { v10.h }[2], [x16]
    st1 { v14.h }[2], [x26]
    st1 { v18.h }[2], [x25]
    st1 { v22.h }[2], [x24]
    st1 { v26.h }[2], [x23]
    b label_243
KAI_ASM_LABEL(label_234)  // Height 5: Partial direct writeback: partial_1_16
    tbz x8, #0, label_243
    str h10, [x16, #0x0]
    str h14, [x26, #0x0]
    str h18, [x25, #0x0]
    str h22, [x24, #0x0]
    str h26, [x23, #0x0]
    b label_243
KAI_ASM_LABEL(label_235)  // Height 5: Partial direct writeback: partial_8_0
    tbz x8, #3, label_239
    st1 { v8.8h }, [x16], #0x10
    st1 { v12.8h }, [x26], #0x10
    st1 { v16.8h }, [x25], #0x10
    st1 { v20.8h }, [x24], #0x10
    st1 { v24.8h }, [x23], #0x10
    tbz x8, #2, label_237
    str d9, [x16], #0x8
    str d13, [x26], #0x8
    str d17, [x25], #0x8
    str d21, [x24], #0x8
    str d25, [x23], #0x8
    tbz x8, #1, label_236
    st1 { v9.s }[2], [x16], #0x4
    st1 { v13.s }[2], [x26], #0x4
    st1 { v17.s }[2], [x25], #0x4
    st1 { v21.s }[2], [x24], #0x4
    st1 { v25.s }[2], [x23], #0x4
    tbz x8, #0, label_243
    st1 { v9.h }[6], [x16]
    st1 { v13.h }[6], [x26]
    st1 { v17.h }[6], [x25]
    st1 { v21.h }[6], [x24]
    st1 { v25.h }[6], [x23]
    b label_243
KAI_ASM_LABEL(label_236)  // Height 5: Partial direct writeback: partial_1_12
    tbz x8, #0, label_243
    st1 { v9.h }[4], [x16]
    st1 { v13.h }[4], [x26]
    st1 { v17.h }[4], [x25]
    st1 { v21.h }[4], [x24]
    st1 { v25.h }[4], [x23]
    b label_243
KAI_ASM_LABEL(label_237)  // Height 5: Partial direct writeback: partial_2_8
    tbz x8, #1, label_238
    str s9, [x16], #0x4
    str s13, [x26], #0x4
    str s17, [x25], #0x4
    str s21, [x24], #0x4
    str s25, [x23], #0x4
    tbz x8, #0, label_243
    st1 { v9.h }[2], [x16]
    st1 { v13.h }[2], [x26]
    st1 { v17.h }[2], [x25]
    st1 { v21.h }[2], [x24]
    st1 { v25.h }[2], [x23]
    b label_243
KAI_ASM_LABEL(label_238)  // Height 5: Partial direct writeback: partial_1_8
    tbz x8, #0, label_243
    str h9, [x16, #0x0]
    str h13, [x26, #0x0]
    str h17, [x25, #0x0]
    str h21, [x24, #0x0]
    str h25, [x23, #0x0]
    b label_243
KAI_ASM_LABEL(label_239)  // Height 5: Partial direct writeback: partial_4_0
    tbz x8, #2, label_241
    str d8, [x16], #0x8
    str d12, [x26], #0x8
    str d16, [x25], #0x8
    str d20, [x24], #0x8
    str d24, [x23], #0x8
    tbz x8, #1, label_240
    st1 { v8.s }[2], [x16], #0x4
    st1 { v12.s }[2], [x26], #0x4
    st1 { v16.s }[2], [x25], #0x4
    st1 { v20.s }[2], [x24], #0x4
    st1 { v24.s }[2], [x23], #0x4
    tbz x8, #0, label_243
    st1 { v8.h }[6], [x16]
    st1 { v12.h }[6], [x26]
    st1 { v16.h }[6], [x25]
    st1 { v20.h }[6], [x24]
    st1 { v24.h }[6], [x23]
    b label_243
KAI_ASM_LABEL(label_240)  // Height 5: Partial direct writeback: partial_1_4
    tbz x8, #0, label_243
    st1 { v8.h }[4], [x16]
    st1 { v12.h }[4], [x26]
    st1 { v16.h }[4], [x25]
    st1 { v20.h }[4], [x24]
    st1 { v24.h }[4], [x23]
    b label_243
KAI_ASM_LABEL(label_241)  // Height 5: Partial direct writeback: partial_2_0
    tbz x8, #1, label_242
    str s8, [x16], #0x4
    str s12, [x26], #0x4
    str s16, [x25], #0x4
    str s20, [x24], #0x4
    str s24, [x23], #0x4
    tbz x8, #0, label_243
    st1 { v8.h }[2], [x16]
    st1 { v12.h }[2], [x26]
    st1 { v16.h }[2], [x25]
    st1 { v20.h }[2], [x24]
    st1 { v24.h }[2], [x23]
    b label_243
KAI_ASM_LABEL(label_242)  // Height 5: Partial direct writeback: partial_1_0
    str h8, [x16, #0x0]
    str h12, [x26, #0x0]
    str h16, [x25, #0x0]
    str h20, [x24, #0x0]
    str h24, [x23, #0x0]
KAI_ASM_LABEL(label_243)  // Height 5: Partial direct writeback: Done
    b label_245
KAI_ASM_LABEL(label_244)  // Height 5: Full writeback
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
KAI_ASM_LABEL(label_245)  // Height 5: Writeback done
    subs x8, x8, #0x20
    bgt label_198
    b label_296
KAI_ASM_LABEL(label_246)  // Height 6
    ldr x21, [x2, #0x20]
    mov x20, #0xc
    ldr x16, [x2, #0x38]
    ldr x8, [x2, #0x10]
    ldr x17, [x2, #0x18]
    madd x20, x21, x20, x16
    str x20, [x2, #0x38]
KAI_ASM_LABEL(label_247)  // Height 6: Column loop
    cbz x17, label_248
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
    b label_267
KAI_ASM_LABEL(label_248)  // Height 6: no bias
    tbz x3, #0, label_266
    ldr x20, [x2, #0x20]
    cmp x8, #0x20
    add x26, x16, x20, LSL #1
    add x25, x26, x20, LSL #1
    add x24, x25, x20, LSL #1
    add x23, x24, x20, LSL #1
    add x22, x23, x20, LSL #1
    bge label_265
    tbz x8, #4, label_256
    ld1 { v8.8h }, [x16], #0x10
    ld1 { v12.8h }, [x26], #0x10
    ld1 { v16.8h }, [x25], #0x10
    ld1 { v20.8h }, [x24], #0x10
    ld1 { v24.8h }, [x23], #0x10
    ld1 { v28.8h }, [x22], #0x10
    ld1 { v9.8h }, [x16], #0x10
    ld1 { v13.8h }, [x26], #0x10
    ld1 { v17.8h }, [x25], #0x10
    ld1 { v21.8h }, [x24], #0x10
    ld1 { v25.8h }, [x23], #0x10
    ld1 { v29.8h }, [x22], #0x10
    tbz x8, #3, label_252
    ld1 { v10.8h }, [x16], #0x10
    ld1 { v14.8h }, [x26], #0x10
    ld1 { v18.8h }, [x25], #0x10
    ld1 { v22.8h }, [x24], #0x10
    ld1 { v26.8h }, [x23], #0x10
    ld1 { v30.8h }, [x22], #0x10
    tbz x8, #2, label_250
    ldr d11, [x16], #0x8
    ldr d15, [x26], #0x8
    ldr d19, [x25], #0x8
    ldr d23, [x24], #0x8
    ldr d27, [x23], #0x8
    ldr d31, [x22], #0x8
    tbz x8, #1, label_249
    ld1 { v11.s }[2], [x16], #0x4
    mov x20, #0x3c
    ld1 { v15.s }[2], [x26], #0x4
    ld1 { v19.s }[2], [x25], #0x4
    ld1 { v23.s }[2], [x24], #0x4
    ld1 { v27.s }[2], [x23], #0x4
    ld1 { v31.s }[2], [x22], #0x4
    tbz x8, #0, label_264
    ld1 { v11.h }[6], [x16]
    ld1 { v15.h }[6], [x26]
    ld1 { v19.h }[6], [x25]
    ld1 { v23.h }[6], [x24]
    ld1 { v27.h }[6], [x23]
    ld1 { v31.h }[6], [x22]
    b label_264
KAI_ASM_LABEL(label_249)  // Height 6: Partial accumulate: partial_1_28
    mov x20, #0x38
    tbz x8, #0, label_264
    ld1 { v11.h }[4], [x16]
    ld1 { v15.h }[4], [x26]
    ld1 { v19.h }[4], [x25]
    ld1 { v23.h }[4], [x24]
    ld1 { v27.h }[4], [x23]
    ld1 { v31.h }[4], [x22]
    b label_264
KAI_ASM_LABEL(label_250)  // Height 6: Partial accumulate: partial_2_24
    tbz x8, #1, label_251
    ldr s11, [x16], #0x4
    mov x20, #0x34
    ldr s15, [x26], #0x4
    ldr s19, [x25], #0x4
    ldr s23, [x24], #0x4
    ldr s27, [x23], #0x4
    ldr s31, [x22], #0x4
    tbz x8, #0, label_264
    ld1 { v11.h }[2], [x16]
    ld1 { v15.h }[2], [x26]
    ld1 { v19.h }[2], [x25]
    ld1 { v23.h }[2], [x24]
    ld1 { v27.h }[2], [x23]
    ld1 { v31.h }[2], [x22]
    b label_264
KAI_ASM_LABEL(label_251)  // Height 6: Partial accumulate: partial_1_24
    mov x20, #0x30
    tbz x8, #0, label_264
    ldr h11, [x16, #0x0]
    ldr h15, [x26, #0x0]
    ldr h19, [x25, #0x0]
    ldr h23, [x24, #0x0]
    ldr h27, [x23, #0x0]
    ldr h31, [x22, #0x0]
    b label_264
KAI_ASM_LABEL(label_252)  // Height 6: Partial accumulate: partial_4_16
    tbz x8, #2, label_254
    ldr d10, [x16], #0x8
    ldr d14, [x26], #0x8
    ldr d18, [x25], #0x8
    ldr d22, [x24], #0x8
    ldr d26, [x23], #0x8
    ldr d30, [x22], #0x8
    tbz x8, #1, label_253
    ld1 { v10.s }[2], [x16], #0x4
    mov x20, #0x2c
    ld1 { v14.s }[2], [x26], #0x4
    ld1 { v18.s }[2], [x25], #0x4
    ld1 { v22.s }[2], [x24], #0x4
    ld1 { v26.s }[2], [x23], #0x4
    ld1 { v30.s }[2], [x22], #0x4
    tbz x8, #0, label_264
    ld1 { v10.h }[6], [x16]
    ld1 { v14.h }[6], [x26]
    ld1 { v18.h }[6], [x25]
    ld1 { v22.h }[6], [x24]
    ld1 { v26.h }[6], [x23]
    ld1 { v30.h }[6], [x22]
    b label_264
KAI_ASM_LABEL(label_253)  // Height 6: Partial accumulate: partial_1_20
    mov x20, #0x28
    tbz x8, #0, label_264
    ld1 { v10.h }[4], [x16]
    ld1 { v14.h }[4], [x26]
    ld1 { v18.h }[4], [x25]
    ld1 { v22.h }[4], [x24]
    ld1 { v26.h }[4], [x23]
    ld1 { v30.h }[4], [x22]
    b label_264
KAI_ASM_LABEL(label_254)  // Height 6: Partial accumulate: partial_2_16
    tbz x8, #1, label_255
    ldr s10, [x16], #0x4
    mov x20, #0x24
    ldr s14, [x26], #0x4
    ldr s18, [x25], #0x4
    ldr s22, [x24], #0x4
    ldr s26, [x23], #0x4
    ldr s30, [x22], #0x4
    tbz x8, #0, label_264
    ld1 { v10.h }[2], [x16]
    ld1 { v14.h }[2], [x26]
    ld1 { v18.h }[2], [x25]
    ld1 { v22.h }[2], [x24]
    ld1 { v26.h }[2], [x23]
    ld1 { v30.h }[2], [x22]
    b label_264
KAI_ASM_LABEL(label_255)  // Height 6: Partial accumulate: partial_1_16
    mov x20, #0x20
    tbz x8, #0, label_264
    ldr h10, [x16, #0x0]
    ldr h14, [x26, #0x0]
    ldr h18, [x25, #0x0]
    ldr h22, [x24, #0x0]
    ldr h26, [x23, #0x0]
    ldr h30, [x22, #0x0]
    b label_264
KAI_ASM_LABEL(label_256)  // Height 6: Partial accumulate: partial_8_0
    tbz x8, #3, label_260
    ld1 { v8.8h }, [x16], #0x10
    ld1 { v12.8h }, [x26], #0x10
    ld1 { v16.8h }, [x25], #0x10
    ld1 { v20.8h }, [x24], #0x10
    ld1 { v24.8h }, [x23], #0x10
    ld1 { v28.8h }, [x22], #0x10
    tbz x8, #2, label_258
    ldr d9, [x16], #0x8
    ldr d13, [x26], #0x8
    ldr d17, [x25], #0x8
    ldr d21, [x24], #0x8
    ldr d25, [x23], #0x8
    ldr d29, [x22], #0x8
    tbz x8, #1, label_257
    ld1 { v9.s }[2], [x16], #0x4
    mov x20, #0x1c
    ld1 { v13.s }[2], [x26], #0x4
    ld1 { v17.s }[2], [x25], #0x4
    ld1 { v21.s }[2], [x24], #0x4
    ld1 { v25.s }[2], [x23], #0x4
    ld1 { v29.s }[2], [x22], #0x4
    tbz x8, #0, label_264
    ld1 { v9.h }[6], [x16]
    ld1 { v13.h }[6], [x26]
    ld1 { v17.h }[6], [x25]
    ld1 { v21.h }[6], [x24]
    ld1 { v25.h }[6], [x23]
    ld1 { v29.h }[6], [x22]
    b label_264
KAI_ASM_LABEL(label_257)  // Height 6: Partial accumulate: partial_1_12
    mov x20, #0x18
    tbz x8, #0, label_264
    ld1 { v9.h }[4], [x16]
    ld1 { v13.h }[4], [x26]
    ld1 { v17.h }[4], [x25]
    ld1 { v21.h }[4], [x24]
    ld1 { v25.h }[4], [x23]
    ld1 { v29.h }[4], [x22]
    b label_264
KAI_ASM_LABEL(label_258)  // Height 6: Partial accumulate: partial_2_8
    tbz x8, #1, label_259
    ldr s9, [x16], #0x4
    mov x20, #0x14
    ldr s13, [x26], #0x4
    ldr s17, [x25], #0x4
    ldr s21, [x24], #0x4
    ldr s25, [x23], #0x4
    ldr s29, [x22], #0x4
    tbz x8, #0, label_264
    ld1 { v9.h }[2], [x16]
    ld1 { v13.h }[2], [x26]
    ld1 { v17.h }[2], [x25]
    ld1 { v21.h }[2], [x24]
    ld1 { v25.h }[2], [x23]
    ld1 { v29.h }[2], [x22]
    b label_264
KAI_ASM_LABEL(label_259)  // Height 6: Partial accumulate: partial_1_8
    mov x20, #0x10
    tbz x8, #0, label_264
    ldr h9, [x16, #0x0]
    ldr h13, [x26, #0x0]
    ldr h17, [x25, #0x0]
    ldr h21, [x24, #0x0]
    ldr h25, [x23, #0x0]
    ldr h29, [x22, #0x0]
    b label_264
KAI_ASM_LABEL(label_260)  // Height 6: Partial accumulate: partial_4_0
    tbz x8, #2, label_262
    ldr d8, [x16], #0x8
    ldr d12, [x26], #0x8
    ldr d16, [x25], #0x8
    ldr d20, [x24], #0x8
    ldr d24, [x23], #0x8
    ldr d28, [x22], #0x8
    tbz x8, #1, label_261
    ld1 { v8.s }[2], [x16], #0x4
    mov x20, #0xc
    ld1 { v12.s }[2], [x26], #0x4
    ld1 { v16.s }[2], [x25], #0x4
    ld1 { v20.s }[2], [x24], #0x4
    ld1 { v24.s }[2], [x23], #0x4
    ld1 { v28.s }[2], [x22], #0x4
    tbz x8, #0, label_264
    ld1 { v8.h }[6], [x16]
    ld1 { v12.h }[6], [x26]
    ld1 { v16.h }[6], [x25]
    ld1 { v20.h }[6], [x24]
    ld1 { v24.h }[6], [x23]
    ld1 { v28.h }[6], [x22]
    b label_264
KAI_ASM_LABEL(label_261)  // Height 6: Partial accumulate: partial_1_4
    mov x20, #0x8
    tbz x8, #0, label_264
    ld1 { v8.h }[4], [x16]
    ld1 { v12.h }[4], [x26]
    ld1 { v16.h }[4], [x25]
    ld1 { v20.h }[4], [x24]
    ld1 { v24.h }[4], [x23]
    ld1 { v28.h }[4], [x22]
    b label_264
KAI_ASM_LABEL(label_262)  // Height 6: Partial accumulate: partial_2_0
    tbz x8, #1, label_263
    ldr s8, [x16], #0x4
    mov x20, #0x4
    ldr s12, [x26], #0x4
    ldr s16, [x25], #0x4
    ldr s20, [x24], #0x4
    ldr s24, [x23], #0x4
    ldr s28, [x22], #0x4
    tbz x8, #0, label_264
    ld1 { v8.h }[2], [x16]
    ld1 { v12.h }[2], [x26]
    ld1 { v16.h }[2], [x25]
    ld1 { v20.h }[2], [x24]
    ld1 { v24.h }[2], [x23]
    ld1 { v28.h }[2], [x22]
    b label_264
KAI_ASM_LABEL(label_263)  // Height 6: Partial accumulate: partial_1_0
    ldr h8, [x16, #0x0]
    mov x20, #0x0
    ldr h12, [x26, #0x0]
    ldr h16, [x25, #0x0]
    ldr h20, [x24, #0x0]
    ldr h24, [x23, #0x0]
    ldr h28, [x22, #0x0]
KAI_ASM_LABEL(label_264)  // Height 6: Partial accumulate: Done
    sub x16, x16, x20
    b label_267
KAI_ASM_LABEL(label_265)  // Height 6: full accumulate
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
    b label_267
KAI_ASM_LABEL(label_266)  // Height 6: no accumulate
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
KAI_ASM_LABEL(label_267)  // Height 6: setup done
    mov x15, #0x0
KAI_ASM_LABEL(label_268)  // Height 6: String loop
    ldr x20, [x2, #0x8]
    ldr x21, [x2, #0x30]
    ldr w14, [x20, x15, LSL #0x2]
    tbz x3, #3, label_269
    ldr x20, [x0, x15, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x13, [x20, #0x0]
    ldr x12, [x20, #0x8]
    ldr x11, [x20, #0x10]
    ldr x10, [x20, #0x18]
    ldr x9, [x20, #0x20]
    ldr x28, [x20, #0x28]
    cbnz x15, label_270
    ldr x20, [x2, #0x28]
    add x13, x13, x20, LSL #1
    add x12, x12, x20, LSL #1
    add x11, x11, x20, LSL #1
    add x10, x10, x20, LSL #1
    add x9, x9, x20, LSL #1
    add x28, x28, x20, LSL #1
    b label_270
KAI_ASM_LABEL(label_269)  // Height 6: setup direct input
    mov x13, x0
    add x12, x13, x21, LSL #1
    add x11, x12, x21, LSL #1
    add x10, x11, x21, LSL #1
    add x9, x10, x21, LSL #1
    add x28, x9, x21, LSL #1
KAI_ASM_LABEL(label_270)  // Height 6: input setup done
    cmp x14, #0x8
    blt label_273
    ldr q0, [x13, #0x0]
    cmp x14, #0x10
    ldr q1, [x12, #0x0]
    ldr q2, [x11, #0x0]
    ldr q3, [x10, #0x0]
    ldr q4, [x9, #0x0]
    ldr q5, [x28, #0x0]
    ldr q6, [x17, #0x0]
    ldr q7, [x17, #0x10]
    blt label_272
KAI_ASM_LABEL(label_271)  // Height 6: Multiply loop: Main loop head
    fmla v8.8h, v6.8h, v0.h[0]
    ldr x21, [x17, #0x28]
    fmla v12.8h, v6.8h, v1.h[0]
    ldr x20, [x17, #0x38]
    fmla v16.8h, v6.8h, v2.h[0]
    add x13, x13, #0x10
    fmla v20.8h, v6.8h, v3.h[0]
    add x12, x12, #0x10
    fmla v24.8h, v6.8h, v4.h[0]
    add x11, x11, #0x10
    fmla v28.8h, v6.8h, v5.h[0]
    ldr d6, [x17, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    add x10, x10, #0x10
    fmla v13.8h, v7.8h, v1.h[0]
    add x9, x9, #0x10
    fmla v17.8h, v7.8h, v2.h[0]
    mov v6.d[1], x21
    fmla v21.8h, v7.8h, v3.h[0]
    ldr x21, [x17, #0x48]
    fmla v25.8h, v7.8h, v4.h[0]
    add x28, x28, #0x10
    fmla v29.8h, v7.8h, v5.h[0]
    ldr d7, [x17, #0x30]
    fmla v10.8h, v6.8h, v0.h[0]
    ldr x27, [x13, #0x8]
    fmla v14.8h, v6.8h, v1.h[0]
    ldr x26, [x12, #0x8]
    mov v7.d[1], x20
    fmla v18.8h, v6.8h, v2.h[0]
    fmla v22.8h, v6.8h, v3.h[0]
    ldr x20, [x17, #0x58]
    fmla v26.8h, v6.8h, v4.h[0]
    ldr x25, [x11, #0x8]
    fmla v30.8h, v6.8h, v5.h[0]
    ldr d6, [x17, #0x40]
    fmla v11.8h, v7.8h, v0.h[0]
    ldr x24, [x10, #0x8]
    fmla v15.8h, v7.8h, v1.h[0]
    ldr x23, [x9, #0x8]
    fmla v19.8h, v7.8h, v2.h[0]
    mov v6.d[1], x21
    fmla v23.8h, v7.8h, v3.h[0]
    ldr x21, [x17, #0x68]
    fmla v27.8h, v7.8h, v4.h[0]
    ldr x22, [x28, #0x8]
    fmla v31.8h, v7.8h, v5.h[0]
    ldr d7, [x17, #0x50]
    fmla v8.8h, v6.8h, v0.h[1]
    sub x14, x14, #0x8
    fmla v12.8h, v6.8h, v1.h[1]
    cmp x14, #0x10
    mov v7.d[1], x20
    fmla v16.8h, v6.8h, v2.h[1]
    fmla v20.8h, v6.8h, v3.h[1]
    ldr x20, [x17, #0x78]
    fmla v24.8h, v6.8h, v4.h[1]
    prfm pldl1keep, [x13, #0x80]
    fmla v28.8h, v6.8h, v5.h[1]
    ldr d6, [x17, #0x60]
    fmla v9.8h, v7.8h, v0.h[1]
    prfm pldl1keep, [x12, #0x80]
    fmla v13.8h, v7.8h, v1.h[1]
    prfm pldl1keep, [x11, #0x80]
    fmla v17.8h, v7.8h, v2.h[1]
    mov v6.d[1], x21
    fmla v21.8h, v7.8h, v3.h[1]
    ldr x21, [x17, #0x88]
    fmla v25.8h, v7.8h, v4.h[1]
    prfm pldl1keep, [x10, #0x80]
    fmla v29.8h, v7.8h, v5.h[1]
    ldr d7, [x17, #0x70]
    fmla v10.8h, v6.8h, v0.h[1]
    prfm pldl1keep, [x9, #0x80]
    fmla v14.8h, v6.8h, v1.h[1]
    prfm pldl1keep, [x28, #0x80]
    mov v7.d[1], x20
    fmla v18.8h, v6.8h, v2.h[1]
    fmla v22.8h, v6.8h, v3.h[1]
    ldr x20, [x17, #0x98]
    fmla v26.8h, v6.8h, v4.h[1]
    fmla v30.8h, v6.8h, v5.h[1]
    ldr d6, [x17, #0x80]
    fmla v11.8h, v7.8h, v0.h[1]
    fmla v15.8h, v7.8h, v1.h[1]
    fmla v19.8h, v7.8h, v2.h[1]
    mov v6.d[1], x21
    fmla v23.8h, v7.8h, v3.h[1]
    ldr x21, [x17, #0xa8]
    fmla v27.8h, v7.8h, v4.h[1]
    fmla v31.8h, v7.8h, v5.h[1]
    ldr d7, [x17, #0x90]
    fmla v8.8h, v6.8h, v0.h[2]
    fmla v12.8h, v6.8h, v1.h[2]
    mov v7.d[1], x20
    fmla v16.8h, v6.8h, v2.h[2]
    fmla v20.8h, v6.8h, v3.h[2]
    ldr x20, [x17, #0xb8]
    fmla v24.8h, v6.8h, v4.h[2]
    fmla v28.8h, v6.8h, v5.h[2]
    ldr d6, [x17, #0xa0]
    fmla v9.8h, v7.8h, v0.h[2]
    fmla v13.8h, v7.8h, v1.h[2]
    fmla v17.8h, v7.8h, v2.h[2]
    mov v6.d[1], x21
    fmla v21.8h, v7.8h, v3.h[2]
    ldr x21, [x17, #0xc8]
    fmla v25.8h, v7.8h, v4.h[2]
    fmla v29.8h, v7.8h, v5.h[2]
    ldr d7, [x17, #0xb0]
    fmla v10.8h, v6.8h, v0.h[2]
    fmla v14.8h, v6.8h, v1.h[2]
    mov v7.d[1], x20
    fmla v18.8h, v6.8h, v2.h[2]
    fmla v22.8h, v6.8h, v3.h[2]
    ldr x20, [x17, #0xd8]
    fmla v26.8h, v6.8h, v4.h[2]
    fmla v30.8h, v6.8h, v5.h[2]
    ldr d6, [x17, #0xc0]
    fmla v11.8h, v7.8h, v0.h[2]
    fmla v15.8h, v7.8h, v1.h[2]
    fmla v19.8h, v7.8h, v2.h[2]
    mov v6.d[1], x21
    fmla v23.8h, v7.8h, v3.h[2]
    ldr x21, [x17, #0xe8]
    fmla v27.8h, v7.8h, v4.h[2]
    fmla v31.8h, v7.8h, v5.h[2]
    ldr d7, [x17, #0xd0]
    fmla v8.8h, v6.8h, v0.h[3]
    fmla v12.8h, v6.8h, v1.h[3]
    mov v7.d[1], x20
    fmla v16.8h, v6.8h, v2.h[3]
    fmla v20.8h, v6.8h, v3.h[3]
    ldr x20, [x17, #0xf8]
    fmla v24.8h, v6.8h, v4.h[3]
    fmla v28.8h, v6.8h, v5.h[3]
    ldr d6, [x17, #0xe0]
    fmla v9.8h, v7.8h, v0.h[3]
    fmla v13.8h, v7.8h, v1.h[3]
    fmla v17.8h, v7.8h, v2.h[3]
    mov v6.d[1], x21
    fmla v21.8h, v7.8h, v3.h[3]
    ldr x21, [x17, #0x108]
    fmla v25.8h, v7.8h, v4.h[3]
    fmla v29.8h, v7.8h, v5.h[3]
    ldr d7, [x17, #0xf0]
    fmla v10.8h, v6.8h, v0.h[3]
    fmla v14.8h, v6.8h, v1.h[3]
    mov v7.d[1], x20
    fmla v18.8h, v6.8h, v2.h[3]
    fmla v22.8h, v6.8h, v3.h[3]
    ldr x20, [x17, #0x118]
    fmla v26.8h, v6.8h, v4.h[3]
    fmla v30.8h, v6.8h, v5.h[3]
    ldr d6, [x17, #0x100]
    fmla v11.8h, v7.8h, v0.h[3]
    fmla v15.8h, v7.8h, v1.h[3]
    fmla v19.8h, v7.8h, v2.h[3]
    mov v6.d[1], x21
    fmla v23.8h, v7.8h, v3.h[3]
    ldr x21, [x17, #0x128]
    fmla v27.8h, v7.8h, v4.h[3]
    fmla v31.8h, v7.8h, v5.h[3]
    ldr d7, [x17, #0x110]
    fmla v8.8h, v6.8h, v0.h[4]
    fmla v12.8h, v6.8h, v1.h[4]
    mov v7.d[1], x20
    fmla v16.8h, v6.8h, v2.h[4]
    fmla v20.8h, v6.8h, v3.h[4]
    ldr x20, [x17, #0x138]
    fmla v24.8h, v6.8h, v4.h[4]
    fmla v28.8h, v6.8h, v5.h[4]
    ldr d6, [x17, #0x120]
    fmla v9.8h, v7.8h, v0.h[4]
    fmla v13.8h, v7.8h, v1.h[4]
    fmla v17.8h, v7.8h, v2.h[4]
    mov v6.d[1], x21
    fmla v21.8h, v7.8h, v3.h[4]
    ldr x21, [x17, #0x148]
    fmla v25.8h, v7.8h, v4.h[4]
    fmla v29.8h, v7.8h, v5.h[4]
    ldr d7, [x17, #0x130]
    fmla v10.8h, v6.8h, v0.h[4]
    fmla v14.8h, v6.8h, v1.h[4]
    mov v7.d[1], x20
    fmla v18.8h, v6.8h, v2.h[4]
    fmla v22.8h, v6.8h, v3.h[4]
    ldr x20, [x17, #0x158]
    fmla v26.8h, v6.8h, v4.h[4]
    fmla v30.8h, v6.8h, v5.h[4]
    ldr d6, [x17, #0x140]
    fmla v11.8h, v7.8h, v0.h[4]
    fmla v15.8h, v7.8h, v1.h[4]
    fmla v19.8h, v7.8h, v2.h[4]
    mov v6.d[1], x21
    fmla v23.8h, v7.8h, v3.h[4]
    ldr x21, [x17, #0x168]
    fmla v27.8h, v7.8h, v4.h[4]
    fmla v31.8h, v7.8h, v5.h[4]
    ldr d7, [x17, #0x150]
    fmla v8.8h, v6.8h, v0.h[5]
    fmla v12.8h, v6.8h, v1.h[5]
    mov v7.d[1], x20
    fmla v16.8h, v6.8h, v2.h[5]
    fmla v20.8h, v6.8h, v3.h[5]
    ldr x20, [x17, #0x178]
    fmla v24.8h, v6.8h, v4.h[5]
    fmla v28.8h, v6.8h, v5.h[5]
    ldr d6, [x17, #0x160]
    fmla v9.8h, v7.8h, v0.h[5]
    fmla v13.8h, v7.8h, v1.h[5]
    fmla v17.8h, v7.8h, v2.h[5]
    mov v6.d[1], x21
    fmla v21.8h, v7.8h, v3.h[5]
    ldr x21, [x17, #0x188]
    fmla v25.8h, v7.8h, v4.h[5]
    fmla v29.8h, v7.8h, v5.h[5]
    ldr d7, [x17, #0x170]
    fmla v10.8h, v6.8h, v0.h[5]
    fmla v14.8h, v6.8h, v1.h[5]
    mov v7.d[1], x20
    fmla v18.8h, v6.8h, v2.h[5]
    fmla v22.8h, v6.8h, v3.h[5]
    ldr x20, [x17, #0x198]
    fmla v26.8h, v6.8h, v4.h[5]
    fmla v30.8h, v6.8h, v5.h[5]
    ldr d6, [x17, #0x180]
    fmla v11.8h, v7.8h, v0.h[5]
    fmla v15.8h, v7.8h, v1.h[5]
    fmla v19.8h, v7.8h, v2.h[5]
    mov v6.d[1], x21
    fmla v23.8h, v7.8h, v3.h[5]
    ldr x21, [x17, #0x1a8]
    fmla v27.8h, v7.8h, v4.h[5]
    fmla v31.8h, v7.8h, v5.h[5]
    ldr d7, [x17, #0x190]
    fmla v8.8h, v6.8h, v0.h[6]
    fmla v12.8h, v6.8h, v1.h[6]
    mov v7.d[1], x20
    fmla v16.8h, v6.8h, v2.h[6]
    fmla v20.8h, v6.8h, v3.h[6]
    ldr x20, [x17, #0x1b8]
    fmla v24.8h, v6.8h, v4.h[6]
    fmla v28.8h, v6.8h, v5.h[6]
    ldr d6, [x17, #0x1a0]
    fmla v9.8h, v7.8h, v0.h[6]
    fmla v13.8h, v7.8h, v1.h[6]
    fmla v17.8h, v7.8h, v2.h[6]
    mov v6.d[1], x21
    fmla v21.8h, v7.8h, v3.h[6]
    ldr x21, [x17, #0x1c8]
    fmla v25.8h, v7.8h, v4.h[6]
    fmla v29.8h, v7.8h, v5.h[6]
    ldr d7, [x17, #0x1b0]
    fmla v10.8h, v6.8h, v0.h[6]
    fmla v14.8h, v6.8h, v1.h[6]
    mov v7.d[1], x20
    fmla v18.8h, v6.8h, v2.h[6]
    fmla v22.8h, v6.8h, v3.h[6]
    ldr x20, [x17, #0x1d8]
    fmla v26.8h, v6.8h, v4.h[6]
    fmla v30.8h, v6.8h, v5.h[6]
    ldr d6, [x17, #0x1c0]
    fmla v11.8h, v7.8h, v0.h[6]
    fmla v15.8h, v7.8h, v1.h[6]
    fmla v19.8h, v7.8h, v2.h[6]
    mov v6.d[1], x21
    fmla v23.8h, v7.8h, v3.h[6]
    ldr x21, [x17, #0x1e8]
    fmla v27.8h, v7.8h, v4.h[6]
    fmla v31.8h, v7.8h, v5.h[6]
    ldr d7, [x17, #0x1d0]
    fmla v8.8h, v6.8h, v0.h[7]
    fmla v12.8h, v6.8h, v1.h[7]
    mov v7.d[1], x20
    fmla v16.8h, v6.8h, v2.h[7]
    fmla v20.8h, v6.8h, v3.h[7]
    ldr x20, [x17, #0x1f8]
    fmla v24.8h, v6.8h, v4.h[7]
    fmla v28.8h, v6.8h, v5.h[7]
    ldr d6, [x17, #0x1e0]
    fmla v9.8h, v7.8h, v0.h[7]
    fmla v13.8h, v7.8h, v1.h[7]
    fmla v17.8h, v7.8h, v2.h[7]
    mov v6.d[1], x21
    fmla v21.8h, v7.8h, v3.h[7]
    fmla v25.8h, v7.8h, v4.h[7]
    fmla v29.8h, v7.8h, v5.h[7]
    ldr d7, [x17, #0x1f0]
    add x17, x17, #0x200
    fmla v10.8h, v6.8h, v0.h[7]
    fmla v14.8h, v6.8h, v1.h[7]
    ldr x21, [x17, #0x8]
    mov v7.d[1], x20
    fmla v18.8h, v6.8h, v2.h[7]
    fmla v22.8h, v6.8h, v3.h[7]
    ldr x20, [x17, #0x18]
    fmla v26.8h, v6.8h, v4.h[7]
    fmla v30.8h, v6.8h, v5.h[7]
    ldr d6, [x17, #0x0]
    fmla v11.8h, v7.8h, v0.h[7]
    ldr d0, [x13, #0x0]
    fmla v15.8h, v7.8h, v1.h[7]
    ldr d1, [x12, #0x0]
    fmla v19.8h, v7.8h, v2.h[7]
    ldr d2, [x11, #0x0]
    fmla v23.8h, v7.8h, v3.h[7]
    ldr d3, [x10, #0x0]
    fmla v27.8h, v7.8h, v4.h[7]
    ldr d4, [x9, #0x0]
    fmla v31.8h, v7.8h, v5.h[7]
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
    bge label_271
KAI_ASM_LABEL(label_272)  // Height 6: Multiply loop: Single iteration only
    fmla v8.8h, v6.8h, v0.h[0]
    add x13, x13, #0x10
    fmla v12.8h, v6.8h, v1.h[0]
    add x12, x12, #0x10
    fmla v16.8h, v6.8h, v2.h[0]
    add x11, x11, #0x10
    fmla v20.8h, v6.8h, v3.h[0]
    add x10, x10, #0x10
    fmla v24.8h, v6.8h, v4.h[0]
    add x9, x9, #0x10
    fmla v28.8h, v6.8h, v5.h[0]
    ldr q6, [x17, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    add x28, x28, #0x10
    fmla v13.8h, v7.8h, v1.h[0]
    sub x14, x14, #0x8
    fmla v17.8h, v7.8h, v2.h[0]
    prfm pldl1keep, [x13, #0x80]
    fmla v21.8h, v7.8h, v3.h[0]
    prfm pldl1keep, [x12, #0x80]
    fmla v25.8h, v7.8h, v4.h[0]
    prfm pldl1keep, [x11, #0x80]
    fmla v29.8h, v7.8h, v5.h[0]
    ldr q7, [x17, #0x30]
    fmla v10.8h, v6.8h, v0.h[0]
    prfm pldl1keep, [x10, #0x80]
    fmla v14.8h, v6.8h, v1.h[0]
    prfm pldl1keep, [x9, #0x80]
    fmla v18.8h, v6.8h, v2.h[0]
    prfm pldl1keep, [x28, #0x80]
    fmla v22.8h, v6.8h, v3.h[0]
    fmla v26.8h, v6.8h, v4.h[0]
    fmla v30.8h, v6.8h, v5.h[0]
    ldr q6, [x17, #0x40]
    fmla v11.8h, v7.8h, v0.h[0]
    fmla v15.8h, v7.8h, v1.h[0]
    fmla v19.8h, v7.8h, v2.h[0]
    fmla v23.8h, v7.8h, v3.h[0]
    fmla v27.8h, v7.8h, v4.h[0]
    fmla v31.8h, v7.8h, v5.h[0]
    ldr q7, [x17, #0x50]
    fmla v8.8h, v6.8h, v0.h[1]
    fmla v12.8h, v6.8h, v1.h[1]
    fmla v16.8h, v6.8h, v2.h[1]
    fmla v20.8h, v6.8h, v3.h[1]
    fmla v24.8h, v6.8h, v4.h[1]
    fmla v28.8h, v6.8h, v5.h[1]
    ldr q6, [x17, #0x60]
    fmla v9.8h, v7.8h, v0.h[1]
    fmla v13.8h, v7.8h, v1.h[1]
    fmla v17.8h, v7.8h, v2.h[1]
    fmla v21.8h, v7.8h, v3.h[1]
    fmla v25.8h, v7.8h, v4.h[1]
    fmla v29.8h, v7.8h, v5.h[1]
    ldr q7, [x17, #0x70]
    fmla v10.8h, v6.8h, v0.h[1]
    fmla v14.8h, v6.8h, v1.h[1]
    fmla v18.8h, v6.8h, v2.h[1]
    fmla v22.8h, v6.8h, v3.h[1]
    fmla v26.8h, v6.8h, v4.h[1]
    fmla v30.8h, v6.8h, v5.h[1]
    ldr q6, [x17, #0x80]
    fmla v11.8h, v7.8h, v0.h[1]
    fmla v15.8h, v7.8h, v1.h[1]
    fmla v19.8h, v7.8h, v2.h[1]
    fmla v23.8h, v7.8h, v3.h[1]
    fmla v27.8h, v7.8h, v4.h[1]
    fmla v31.8h, v7.8h, v5.h[1]
    ldr q7, [x17, #0x90]
    fmla v8.8h, v6.8h, v0.h[2]
    fmla v12.8h, v6.8h, v1.h[2]
    fmla v16.8h, v6.8h, v2.h[2]
    fmla v20.8h, v6.8h, v3.h[2]
    fmla v24.8h, v6.8h, v4.h[2]
    fmla v28.8h, v6.8h, v5.h[2]
    ldr q6, [x17, #0xa0]
    fmla v9.8h, v7.8h, v0.h[2]
    fmla v13.8h, v7.8h, v1.h[2]
    fmla v17.8h, v7.8h, v2.h[2]
    fmla v21.8h, v7.8h, v3.h[2]
    fmla v25.8h, v7.8h, v4.h[2]
    fmla v29.8h, v7.8h, v5.h[2]
    ldr q7, [x17, #0xb0]
    fmla v10.8h, v6.8h, v0.h[2]
    fmla v14.8h, v6.8h, v1.h[2]
    fmla v18.8h, v6.8h, v2.h[2]
    fmla v22.8h, v6.8h, v3.h[2]
    fmla v26.8h, v6.8h, v4.h[2]
    fmla v30.8h, v6.8h, v5.h[2]
    ldr q6, [x17, #0xc0]
    fmla v11.8h, v7.8h, v0.h[2]
    fmla v15.8h, v7.8h, v1.h[2]
    fmla v19.8h, v7.8h, v2.h[2]
    fmla v23.8h, v7.8h, v3.h[2]
    fmla v27.8h, v7.8h, v4.h[2]
    fmla v31.8h, v7.8h, v5.h[2]
    ldr q7, [x17, #0xd0]
    fmla v8.8h, v6.8h, v0.h[3]
    fmla v12.8h, v6.8h, v1.h[3]
    fmla v16.8h, v6.8h, v2.h[3]
    fmla v20.8h, v6.8h, v3.h[3]
    fmla v24.8h, v6.8h, v4.h[3]
    fmla v28.8h, v6.8h, v5.h[3]
    ldr q6, [x17, #0xe0]
    fmla v9.8h, v7.8h, v0.h[3]
    fmla v13.8h, v7.8h, v1.h[3]
    fmla v17.8h, v7.8h, v2.h[3]
    fmla v21.8h, v7.8h, v3.h[3]
    fmla v25.8h, v7.8h, v4.h[3]
    fmla v29.8h, v7.8h, v5.h[3]
    ldr q7, [x17, #0xf0]
    fmla v10.8h, v6.8h, v0.h[3]
    fmla v14.8h, v6.8h, v1.h[3]
    fmla v18.8h, v6.8h, v2.h[3]
    fmla v22.8h, v6.8h, v3.h[3]
    fmla v26.8h, v6.8h, v4.h[3]
    fmla v30.8h, v6.8h, v5.h[3]
    ldr q6, [x17, #0x100]
    fmla v11.8h, v7.8h, v0.h[3]
    fmla v15.8h, v7.8h, v1.h[3]
    fmla v19.8h, v7.8h, v2.h[3]
    fmla v23.8h, v7.8h, v3.h[3]
    fmla v27.8h, v7.8h, v4.h[3]
    fmla v31.8h, v7.8h, v5.h[3]
    ldr q7, [x17, #0x110]
    fmla v8.8h, v6.8h, v0.h[4]
    fmla v12.8h, v6.8h, v1.h[4]
    fmla v16.8h, v6.8h, v2.h[4]
    fmla v20.8h, v6.8h, v3.h[4]
    fmla v24.8h, v6.8h, v4.h[4]
    fmla v28.8h, v6.8h, v5.h[4]
    ldr q6, [x17, #0x120]
    fmla v9.8h, v7.8h, v0.h[4]
    fmla v13.8h, v7.8h, v1.h[4]
    fmla v17.8h, v7.8h, v2.h[4]
    fmla v21.8h, v7.8h, v3.h[4]
    fmla v25.8h, v7.8h, v4.h[4]
    fmla v29.8h, v7.8h, v5.h[4]
    ldr q7, [x17, #0x130]
    fmla v10.8h, v6.8h, v0.h[4]
    fmla v14.8h, v6.8h, v1.h[4]
    fmla v18.8h, v6.8h, v2.h[4]
    fmla v22.8h, v6.8h, v3.h[4]
    fmla v26.8h, v6.8h, v4.h[4]
    fmla v30.8h, v6.8h, v5.h[4]
    ldr q6, [x17, #0x140]
    fmla v11.8h, v7.8h, v0.h[4]
    fmla v15.8h, v7.8h, v1.h[4]
    fmla v19.8h, v7.8h, v2.h[4]
    fmla v23.8h, v7.8h, v3.h[4]
    fmla v27.8h, v7.8h, v4.h[4]
    fmla v31.8h, v7.8h, v5.h[4]
    ldr q7, [x17, #0x150]
    fmla v8.8h, v6.8h, v0.h[5]
    fmla v12.8h, v6.8h, v1.h[5]
    fmla v16.8h, v6.8h, v2.h[5]
    fmla v20.8h, v6.8h, v3.h[5]
    fmla v24.8h, v6.8h, v4.h[5]
    fmla v28.8h, v6.8h, v5.h[5]
    ldr q6, [x17, #0x160]
    fmla v9.8h, v7.8h, v0.h[5]
    fmla v13.8h, v7.8h, v1.h[5]
    fmla v17.8h, v7.8h, v2.h[5]
    fmla v21.8h, v7.8h, v3.h[5]
    fmla v25.8h, v7.8h, v4.h[5]
    fmla v29.8h, v7.8h, v5.h[5]
    ldr q7, [x17, #0x170]
    fmla v10.8h, v6.8h, v0.h[5]
    fmla v14.8h, v6.8h, v1.h[5]
    fmla v18.8h, v6.8h, v2.h[5]
    fmla v22.8h, v6.8h, v3.h[5]
    fmla v26.8h, v6.8h, v4.h[5]
    fmla v30.8h, v6.8h, v5.h[5]
    ldr q6, [x17, #0x180]
    fmla v11.8h, v7.8h, v0.h[5]
    fmla v15.8h, v7.8h, v1.h[5]
    fmla v19.8h, v7.8h, v2.h[5]
    fmla v23.8h, v7.8h, v3.h[5]
    fmla v27.8h, v7.8h, v4.h[5]
    fmla v31.8h, v7.8h, v5.h[5]
    ldr q7, [x17, #0x190]
    fmla v8.8h, v6.8h, v0.h[6]
    fmla v12.8h, v6.8h, v1.h[6]
    fmla v16.8h, v6.8h, v2.h[6]
    fmla v20.8h, v6.8h, v3.h[6]
    fmla v24.8h, v6.8h, v4.h[6]
    fmla v28.8h, v6.8h, v5.h[6]
    ldr q6, [x17, #0x1a0]
    fmla v9.8h, v7.8h, v0.h[6]
    fmla v13.8h, v7.8h, v1.h[6]
    fmla v17.8h, v7.8h, v2.h[6]
    fmla v21.8h, v7.8h, v3.h[6]
    fmla v25.8h, v7.8h, v4.h[6]
    fmla v29.8h, v7.8h, v5.h[6]
    ldr q7, [x17, #0x1b0]
    fmla v10.8h, v6.8h, v0.h[6]
    fmla v14.8h, v6.8h, v1.h[6]
    fmla v18.8h, v6.8h, v2.h[6]
    fmla v22.8h, v6.8h, v3.h[6]
    fmla v26.8h, v6.8h, v4.h[6]
    fmla v30.8h, v6.8h, v5.h[6]
    ldr q6, [x17, #0x1c0]
    fmla v11.8h, v7.8h, v0.h[6]
    fmla v15.8h, v7.8h, v1.h[6]
    fmla v19.8h, v7.8h, v2.h[6]
    fmla v23.8h, v7.8h, v3.h[6]
    fmla v27.8h, v7.8h, v4.h[6]
    fmla v31.8h, v7.8h, v5.h[6]
    ldr q7, [x17, #0x1d0]
    fmla v8.8h, v6.8h, v0.h[7]
    fmla v12.8h, v6.8h, v1.h[7]
    fmla v16.8h, v6.8h, v2.h[7]
    fmla v20.8h, v6.8h, v3.h[7]
    fmla v24.8h, v6.8h, v4.h[7]
    fmla v28.8h, v6.8h, v5.h[7]
    ldr q6, [x17, #0x1e0]
    fmla v9.8h, v7.8h, v0.h[7]
    fmla v13.8h, v7.8h, v1.h[7]
    fmla v17.8h, v7.8h, v2.h[7]
    fmla v21.8h, v7.8h, v3.h[7]
    fmla v25.8h, v7.8h, v4.h[7]
    fmla v29.8h, v7.8h, v5.h[7]
    ldr q7, [x17, #0x1f0]
    fmla v10.8h, v6.8h, v0.h[7]
    add x17, x17, #0x200
    fmla v14.8h, v6.8h, v1.h[7]
    fmla v18.8h, v6.8h, v2.h[7]
    fmla v22.8h, v6.8h, v3.h[7]
    fmla v26.8h, v6.8h, v4.h[7]
    fmla v30.8h, v6.8h, v5.h[7]
    fmla v11.8h, v7.8h, v0.h[7]
    fmla v15.8h, v7.8h, v1.h[7]
    fmla v19.8h, v7.8h, v2.h[7]
    fmla v23.8h, v7.8h, v3.h[7]
    fmla v27.8h, v7.8h, v4.h[7]
    fmla v31.8h, v7.8h, v5.h[7]
KAI_ASM_LABEL(label_273)  // Height 6: Multiply loop: Main loop skip
    cbz x14, label_275
KAI_ASM_LABEL(label_274)  // Height 6: Multiply loop: Odd block loop
    ldr h0, [x13], #0x2
    sub x14, x14, #0x1
    ldr h1, [x12], #0x2
    ldr h2, [x11], #0x2
    ldr h3, [x10], #0x2
    ldr h4, [x9], #0x2
    ldr h5, [x28], #0x2
    ldr q6, [x17, #0x0]
    ldr q7, [x17, #0x10]
    fmla v8.8h, v6.8h, v0.h[0]
    fmla v12.8h, v6.8h, v1.h[0]
    fmla v16.8h, v6.8h, v2.h[0]
    fmla v20.8h, v6.8h, v3.h[0]
    fmla v24.8h, v6.8h, v4.h[0]
    fmla v28.8h, v6.8h, v5.h[0]
    ldr q6, [x17, #0x20]
    fmla v9.8h, v7.8h, v0.h[0]
    fmla v13.8h, v7.8h, v1.h[0]
    fmla v17.8h, v7.8h, v2.h[0]
    fmla v21.8h, v7.8h, v3.h[0]
    fmla v25.8h, v7.8h, v4.h[0]
    fmla v29.8h, v7.8h, v5.h[0]
    ldr q7, [x17, #0x30]
    fmla v10.8h, v6.8h, v0.h[0]
    add x17, x17, #0x40
    fmla v14.8h, v6.8h, v1.h[0]
    fmla v18.8h, v6.8h, v2.h[0]
    fmla v22.8h, v6.8h, v3.h[0]
    fmla v26.8h, v6.8h, v4.h[0]
    fmla v30.8h, v6.8h, v5.h[0]
    fmla v11.8h, v7.8h, v0.h[0]
    fmla v15.8h, v7.8h, v1.h[0]
    fmla v19.8h, v7.8h, v2.h[0]
    fmla v23.8h, v7.8h, v3.h[0]
    fmla v27.8h, v7.8h, v4.h[0]
    fmla v31.8h, v7.8h, v5.h[0]
    cbnz x14, label_274
KAI_ASM_LABEL(label_275)  // Height 6: Multiply loop: No odd multiplies
    ldr w20, [x2, #0x4]
    add x15, x15, #0x1
    cmp x15, x20
    bne label_268
    ldr x20, [x2, #0x20]
    prfm pstl1keep, [x16, #0x0]
    add x26, x16, x20, LSL #1
    add x25, x26, x20, LSL #1
    prfm pstl1keep, [x26, #0x0]
    add x24, x25, x20, LSL #1
    prfm pstl1keep, [x25, #0x0]
    add x23, x24, x20, LSL #1
    prfm pstl1keep, [x24, #0x0]
    add x22, x23, x20, LSL #1
    prfm pstl1keep, [x23, #0x0]
    prfm pstl1keep, [x22, #0x0]
    tbz x3, #1, label_276
    add x21, x2, #0x0
    add x20, x2, #0x2
    ld1r { v1.8h }, [x21]
    ld1r { v0.8h }, [x20]
    fmin v8.8h, v8.8h, v1.8h
    fmin v9.8h, v9.8h, v1.8h
    fmin v10.8h, v10.8h, v1.8h
    fmin v11.8h, v11.8h, v1.8h
    fmin v12.8h, v12.8h, v1.8h
    fmin v13.8h, v13.8h, v1.8h
    fmin v14.8h, v14.8h, v1.8h
    fmin v15.8h, v15.8h, v1.8h
    fmin v16.8h, v16.8h, v1.8h
    fmin v17.8h, v17.8h, v1.8h
    fmin v18.8h, v18.8h, v1.8h
    fmin v19.8h, v19.8h, v1.8h
    fmin v20.8h, v20.8h, v1.8h
    fmin v21.8h, v21.8h, v1.8h
    fmin v22.8h, v22.8h, v1.8h
    fmin v23.8h, v23.8h, v1.8h
    fmin v24.8h, v24.8h, v1.8h
    fmin v25.8h, v25.8h, v1.8h
    fmin v26.8h, v26.8h, v1.8h
    fmin v27.8h, v27.8h, v1.8h
    fmin v28.8h, v28.8h, v1.8h
    fmin v29.8h, v29.8h, v1.8h
    fmin v30.8h, v30.8h, v1.8h
    fmin v31.8h, v31.8h, v1.8h
    fmax v8.8h, v8.8h, v0.8h
    fmax v9.8h, v9.8h, v0.8h
    fmax v10.8h, v10.8h, v0.8h
    fmax v11.8h, v11.8h, v0.8h
    fmax v12.8h, v12.8h, v0.8h
    fmax v13.8h, v13.8h, v0.8h
    fmax v14.8h, v14.8h, v0.8h
    fmax v15.8h, v15.8h, v0.8h
    fmax v16.8h, v16.8h, v0.8h
    fmax v17.8h, v17.8h, v0.8h
    fmax v18.8h, v18.8h, v0.8h
    fmax v19.8h, v19.8h, v0.8h
    fmax v20.8h, v20.8h, v0.8h
    fmax v21.8h, v21.8h, v0.8h
    fmax v22.8h, v22.8h, v0.8h
    fmax v23.8h, v23.8h, v0.8h
    fmax v24.8h, v24.8h, v0.8h
    fmax v25.8h, v25.8h, v0.8h
    fmax v26.8h, v26.8h, v0.8h
    fmax v27.8h, v27.8h, v0.8h
    fmax v28.8h, v28.8h, v0.8h
    fmax v29.8h, v29.8h, v0.8h
    fmax v30.8h, v30.8h, v0.8h
    fmax v31.8h, v31.8h, v0.8h
KAI_ASM_LABEL(label_276)  // Height 6: No activation
    cmp x8, #0x20
    bge label_293
    tbz x8, #4, label_284
    st1 { v8.8h }, [x16], #0x10
    st1 { v9.8h }, [x16], #0x10
    st1 { v12.8h }, [x26], #0x10
    st1 { v13.8h }, [x26], #0x10
    st1 { v16.8h }, [x25], #0x10
    st1 { v17.8h }, [x25], #0x10
    st1 { v20.8h }, [x24], #0x10
    st1 { v21.8h }, [x24], #0x10
    st1 { v24.8h }, [x23], #0x10
    st1 { v25.8h }, [x23], #0x10
    st1 { v28.8h }, [x22], #0x10
    st1 { v29.8h }, [x22], #0x10
    tbz x8, #3, label_280
    st1 { v10.8h }, [x16], #0x10
    st1 { v14.8h }, [x26], #0x10
    st1 { v18.8h }, [x25], #0x10
    st1 { v22.8h }, [x24], #0x10
    st1 { v26.8h }, [x23], #0x10
    st1 { v30.8h }, [x22], #0x10
    tbz x8, #2, label_278
    str d11, [x16], #0x8
    str d15, [x26], #0x8
    str d19, [x25], #0x8
    str d23, [x24], #0x8
    str d27, [x23], #0x8
    str d31, [x22], #0x8
    tbz x8, #1, label_277
    st1 { v11.s }[2], [x16], #0x4
    st1 { v15.s }[2], [x26], #0x4
    st1 { v19.s }[2], [x25], #0x4
    st1 { v23.s }[2], [x24], #0x4
    st1 { v27.s }[2], [x23], #0x4
    st1 { v31.s }[2], [x22], #0x4
    tbz x8, #0, label_292
    st1 { v11.h }[6], [x16]
    st1 { v15.h }[6], [x26]
    st1 { v19.h }[6], [x25]
    st1 { v23.h }[6], [x24]
    st1 { v27.h }[6], [x23]
    st1 { v31.h }[6], [x22]
    b label_292
KAI_ASM_LABEL(label_277)  // Height 6: Partial direct writeback: partial_1_28
    tbz x8, #0, label_292
    st1 { v11.h }[4], [x16]
    st1 { v15.h }[4], [x26]
    st1 { v19.h }[4], [x25]
    st1 { v23.h }[4], [x24]
    st1 { v27.h }[4], [x23]
    st1 { v31.h }[4], [x22]
    b label_292
KAI_ASM_LABEL(label_278)  // Height 6: Partial direct writeback: partial_2_24
    tbz x8, #1, label_279
    str s11, [x16], #0x4
    str s15, [x26], #0x4
    str s19, [x25], #0x4
    str s23, [x24], #0x4
    str s27, [x23], #0x4
    str s31, [x22], #0x4
    tbz x8, #0, label_292
    st1 { v11.h }[2], [x16]
    st1 { v15.h }[2], [x26]
    st1 { v19.h }[2], [x25]
    st1 { v23.h }[2], [x24]
    st1 { v27.h }[2], [x23]
    st1 { v31.h }[2], [x22]
    b label_292
KAI_ASM_LABEL(label_279)  // Height 6: Partial direct writeback: partial_1_24
    tbz x8, #0, label_292
    str h11, [x16, #0x0]
    str h15, [x26, #0x0]
    str h19, [x25, #0x0]
    str h23, [x24, #0x0]
    str h27, [x23, #0x0]
    str h31, [x22, #0x0]
    b label_292
KAI_ASM_LABEL(label_280)  // Height 6: Partial direct writeback: partial_4_16
    tbz x8, #2, label_282
    str d10, [x16], #0x8
    str d14, [x26], #0x8
    str d18, [x25], #0x8
    str d22, [x24], #0x8
    str d26, [x23], #0x8
    str d30, [x22], #0x8
    tbz x8, #1, label_281
    st1 { v10.s }[2], [x16], #0x4
    st1 { v14.s }[2], [x26], #0x4
    st1 { v18.s }[2], [x25], #0x4
    st1 { v22.s }[2], [x24], #0x4
    st1 { v26.s }[2], [x23], #0x4
    st1 { v30.s }[2], [x22], #0x4
    tbz x8, #0, label_292
    st1 { v10.h }[6], [x16]
    st1 { v14.h }[6], [x26]
    st1 { v18.h }[6], [x25]
    st1 { v22.h }[6], [x24]
    st1 { v26.h }[6], [x23]
    st1 { v30.h }[6], [x22]
    b label_292
KAI_ASM_LABEL(label_281)  // Height 6: Partial direct writeback: partial_1_20
    tbz x8, #0, label_292
    st1 { v10.h }[4], [x16]
    st1 { v14.h }[4], [x26]
    st1 { v18.h }[4], [x25]
    st1 { v22.h }[4], [x24]
    st1 { v26.h }[4], [x23]
    st1 { v30.h }[4], [x22]
    b label_292
KAI_ASM_LABEL(label_282)  // Height 6: Partial direct writeback: partial_2_16
    tbz x8, #1, label_283
    str s10, [x16], #0x4
    str s14, [x26], #0x4
    str s18, [x25], #0x4
    str s22, [x24], #0x4
    str s26, [x23], #0x4
    str s30, [x22], #0x4
    tbz x8, #0, label_292
    st1 { v10.h }[2], [x16]
    st1 { v14.h }[2], [x26]
    st1 { v18.h }[2], [x25]
    st1 { v22.h }[2], [x24]
    st1 { v26.h }[2], [x23]
    st1 { v30.h }[2], [x22]
    b label_292
KAI_ASM_LABEL(label_283)  // Height 6: Partial direct writeback: partial_1_16
    tbz x8, #0, label_292
    str h10, [x16, #0x0]
    str h14, [x26, #0x0]
    str h18, [x25, #0x0]
    str h22, [x24, #0x0]
    str h26, [x23, #0x0]
    str h30, [x22, #0x0]
    b label_292
KAI_ASM_LABEL(label_284)  // Height 6: Partial direct writeback: partial_8_0
    tbz x8, #3, label_288
    st1 { v8.8h }, [x16], #0x10
    st1 { v12.8h }, [x26], #0x10
    st1 { v16.8h }, [x25], #0x10
    st1 { v20.8h }, [x24], #0x10
    st1 { v24.8h }, [x23], #0x10
    st1 { v28.8h }, [x22], #0x10
    tbz x8, #2, label_286
    str d9, [x16], #0x8
    str d13, [x26], #0x8
    str d17, [x25], #0x8
    str d21, [x24], #0x8
    str d25, [x23], #0x8
    str d29, [x22], #0x8
    tbz x8, #1, label_285
    st1 { v9.s }[2], [x16], #0x4
    st1 { v13.s }[2], [x26], #0x4
    st1 { v17.s }[2], [x25], #0x4
    st1 { v21.s }[2], [x24], #0x4
    st1 { v25.s }[2], [x23], #0x4
    st1 { v29.s }[2], [x22], #0x4
    tbz x8, #0, label_292
    st1 { v9.h }[6], [x16]
    st1 { v13.h }[6], [x26]
    st1 { v17.h }[6], [x25]
    st1 { v21.h }[6], [x24]
    st1 { v25.h }[6], [x23]
    st1 { v29.h }[6], [x22]
    b label_292
KAI_ASM_LABEL(label_285)  // Height 6: Partial direct writeback: partial_1_12
    tbz x8, #0, label_292
    st1 { v9.h }[4], [x16]
    st1 { v13.h }[4], [x26]
    st1 { v17.h }[4], [x25]
    st1 { v21.h }[4], [x24]
    st1 { v25.h }[4], [x23]
    st1 { v29.h }[4], [x22]
    b label_292
KAI_ASM_LABEL(label_286)  // Height 6: Partial direct writeback: partial_2_8
    tbz x8, #1, label_287
    str s9, [x16], #0x4
    str s13, [x26], #0x4
    str s17, [x25], #0x4
    str s21, [x24], #0x4
    str s25, [x23], #0x4
    str s29, [x22], #0x4
    tbz x8, #0, label_292
    st1 { v9.h }[2], [x16]
    st1 { v13.h }[2], [x26]
    st1 { v17.h }[2], [x25]
    st1 { v21.h }[2], [x24]
    st1 { v25.h }[2], [x23]
    st1 { v29.h }[2], [x22]
    b label_292
KAI_ASM_LABEL(label_287)  // Height 6: Partial direct writeback: partial_1_8
    tbz x8, #0, label_292
    str h9, [x16, #0x0]
    str h13, [x26, #0x0]
    str h17, [x25, #0x0]
    str h21, [x24, #0x0]
    str h25, [x23, #0x0]
    str h29, [x22, #0x0]
    b label_292
KAI_ASM_LABEL(label_288)  // Height 6: Partial direct writeback: partial_4_0
    tbz x8, #2, label_290
    str d8, [x16], #0x8
    str d12, [x26], #0x8
    str d16, [x25], #0x8
    str d20, [x24], #0x8
    str d24, [x23], #0x8
    str d28, [x22], #0x8
    tbz x8, #1, label_289
    st1 { v8.s }[2], [x16], #0x4
    st1 { v12.s }[2], [x26], #0x4
    st1 { v16.s }[2], [x25], #0x4
    st1 { v20.s }[2], [x24], #0x4
    st1 { v24.s }[2], [x23], #0x4
    st1 { v28.s }[2], [x22], #0x4
    tbz x8, #0, label_292
    st1 { v8.h }[6], [x16]
    st1 { v12.h }[6], [x26]
    st1 { v16.h }[6], [x25]
    st1 { v20.h }[6], [x24]
    st1 { v24.h }[6], [x23]
    st1 { v28.h }[6], [x22]
    b label_292
KAI_ASM_LABEL(label_289)  // Height 6: Partial direct writeback: partial_1_4
    tbz x8, #0, label_292
    st1 { v8.h }[4], [x16]
    st1 { v12.h }[4], [x26]
    st1 { v16.h }[4], [x25]
    st1 { v20.h }[4], [x24]
    st1 { v24.h }[4], [x23]
    st1 { v28.h }[4], [x22]
    b label_292
KAI_ASM_LABEL(label_290)  // Height 6: Partial direct writeback: partial_2_0
    tbz x8, #1, label_291
    str s8, [x16], #0x4
    str s12, [x26], #0x4
    str s16, [x25], #0x4
    str s20, [x24], #0x4
    str s24, [x23], #0x4
    str s28, [x22], #0x4
    tbz x8, #0, label_292
    st1 { v8.h }[2], [x16]
    st1 { v12.h }[2], [x26]
    st1 { v16.h }[2], [x25]
    st1 { v20.h }[2], [x24]
    st1 { v24.h }[2], [x23]
    st1 { v28.h }[2], [x22]
    b label_292
KAI_ASM_LABEL(label_291)  // Height 6: Partial direct writeback: partial_1_0
    str h8, [x16, #0x0]
    str h12, [x26, #0x0]
    str h16, [x25, #0x0]
    str h20, [x24, #0x0]
    str h24, [x23, #0x0]
    str h28, [x22, #0x0]
KAI_ASM_LABEL(label_292)  // Height 6: Partial direct writeback: Done
    b label_294
KAI_ASM_LABEL(label_293)  // Height 6: Full writeback
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
KAI_ASM_LABEL(label_294)  // Height 6: Writeback done
    subs x8, x8, #0x20
    bgt label_247
    subs x1, x1, #0x6
    beq label_296
    ldr x21, [x2, #0x30]
    tbz x3, #3, label_295
    add x21, x21, #0x6
    str x21, [x2, #0x30]
    b label_1
KAI_ASM_LABEL(label_295)  // Update direct input
    mov x20, #0xc
    madd x0, x20, x21, x0
    b label_1
KAI_ASM_LABEL(label_296)  // Exit
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla_cortexa55)

    KAI_ASM_END
