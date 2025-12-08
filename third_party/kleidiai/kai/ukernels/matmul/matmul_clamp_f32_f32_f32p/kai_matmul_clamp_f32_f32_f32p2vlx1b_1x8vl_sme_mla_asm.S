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

    KAI_ASM_CODE(matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    ldr x14, [x0, #0x20]
    cntw x13
    cntw x20, ALL, MUL #2
    ldr x12, [x0, #0x18]
    ptrue p2.b
    ldr x11, [x0, #0x8]
    mov x10, x14
    ldr x9, [x0, #0x10]
    add x28, x12, x13
    lsl x10, x10, #0x2
    ldr x27, [x0, #0x28]
    sub x28, x28, #0x1
    add x10, x10, #0x4
    ldr x26, [x0, #0x30]
    udiv x28, x28, x13
    mul x10, x10, x20
KAI_ASM_LABEL(label_1)  // Column loop
    cmp x28, #0x8
    bge label_36
    cmp x28, #0x6
    bgt label_31
    beq label_26
    cmp x28, #0x4
    bgt label_21
    beq label_16
    cmp x28, #0x2
    bgt label_11
    beq label_6
    mov x25, x14
    whilelt p1.s, XZR, x12
    ld1w { z24.s }, p2/Z, [x9]
    cmp x25, #0x4
    mov x24, x11
    incb x9, ALL, MUL #2
    ble label_3
KAI_ASM_LABEL(label_2)  // Width 1: Multiply loop: Main loop head
    whilelt p0.s, XZR, x25
    ldnt1w { z1.s }, p2/Z, [x9]
    addvl x9, x9, #2
    ld1rqw { z0.s }, p0/Z, [x24]
    sub x25, x25, #0x4
    add x24, x24, #0x10
    ldnt1w { z2.s }, p2/Z, [x9]
    addvl x9, x9, #2
    cmp x25, #0x4
    ldnt1w { z3.s }, p2/Z, [x9]
    addvl x9, x9, #2
    fmla z24.s, z1.s, z0.s[0]
    ldnt1w { z4.s }, p2/Z, [x9]
    addvl x9, x9, #2
    fmla z24.s, z2.s, z0.s[1]
    fmla z24.s, z3.s, z0.s[2]
    fmla z24.s, z4.s, z0.s[3]
    bgt label_2
KAI_ASM_LABEL(label_3)  // Width 1: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x25
    ldnt1w { z5.s }, p2/Z, [x9]
    subs x25, x25, #0x1
    ld1rqw { z0.s }, p0/Z, [x24]
    addvl x9, x9, #2
    fmla z24.s, z5.s, z0.s[0]
    ble label_4
    ldnt1w { z6.s }, p2/Z, [x9]
    subs x25, x25, #0x1
    addvl x9, x9, #2
    fmla z24.s, z6.s, z0.s[1]
    ble label_4
    ldnt1w { z7.s }, p2/Z, [x9]
    subs x25, x25, #0x1
    addvl x9, x9, #2
    fmla z24.s, z7.s, z0.s[2]
    ble label_4
    ldnt1w { z8.s }, p2/Z, [x9]
    fmla z24.s, z8.s, z0.s[3]
KAI_ASM_LABEL(label_4)  // Width 1: Multiply loop: multiply skip
    tbz x26, #1, label_5
    add x21, x0, #0x0
    add x20, x0, #0x4
    KAI_ASM_INST(0x8540cab1)  // ld1rw { z17.s }, p2/Z, [x21]
    KAI_ASM_INST(0x8540ca90)  // ld1rw { z16.s }, p2/Z, [x20]
    fmin z24.s, p2/M, z24.s, z17.s
    fmax z24.s, p2/M, z24.s, z16.s
KAI_ASM_LABEL(label_5)  // Width 1: No activation
    st1w { z24.s }, p1, [x27]
    b label_41
KAI_ASM_LABEL(label_6)  // Width 2
    mov x25, x14
    sub x20, x12, x13
    ld1w { z24.s }, p2/Z, [x9]
    whilelt p1.s, XZR, x20
    cmp x25, #0x4
    ld1w { z25.s }, p2/Z, [x9, #1, MUL VL]
    mov x24, x11
    incb x9, ALL, MUL #2
    ble label_8
KAI_ASM_LABEL(label_7)  // Width 2: Multiply loop: Main loop head
    whilelt p0.s, XZR, x25
    ldnt1w { z1.s }, p2/Z, [x9]
    sub x25, x25, #0x4
    ld1rqw { z0.s }, p0/Z, [x24]
    cmp x25, #0x4
    add x24, x24, #0x10
    ldnt1w { z2.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w { z3.s }, p2/Z, [x9]
    fmla z24.s, z1.s, z0.s[0]
    ldnt1w { z4.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    fmla z25.s, z2.s, z0.s[0]
    ldnt1w { z5.s }, p2/Z, [x9]
    ldnt1w { z6.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w { z7.s }, p2/Z, [x9]
    fmla z24.s, z3.s, z0.s[1]
    ldnt1w { z8.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    fmla z25.s, z4.s, z0.s[1]
    fmla z24.s, z5.s, z0.s[2]
    fmla z25.s, z6.s, z0.s[2]
    fmla z24.s, z7.s, z0.s[3]
    fmla z25.s, z8.s, z0.s[3]
    bgt label_7
KAI_ASM_LABEL(label_8)  // Width 2: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x25
    ldnt1w { z9.s }, p2/Z, [x9]
    subs x25, x25, #0x1
    ld1rqw { z0.s }, p0/Z, [x24]
    ldnt1w { z10.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    fmla z24.s, z9.s, z0.s[0]
    fmla z25.s, z10.s, z0.s[0]
    ble label_9
    ldnt1w { z11.s }, p2/Z, [x9]
    subs x25, x25, #0x1
    ldnt1w { z12.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    fmla z24.s, z11.s, z0.s[1]
    fmla z25.s, z12.s, z0.s[1]
    ble label_9
    ldnt1w { z13.s }, p2/Z, [x9]
    subs x25, x25, #0x1
    ldnt1w { z14.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    fmla z24.s, z13.s, z0.s[2]
    fmla z25.s, z14.s, z0.s[2]
    ble label_9
    ldnt1w { z15.s }, p2/Z, [x9]
    ldnt1w { z16.s }, p2/Z, [x9, #1, MUL VL]
    fmla z24.s, z15.s, z0.s[3]
    fmla z25.s, z16.s, z0.s[3]
KAI_ASM_LABEL(label_9)  // Width 2: Multiply loop: multiply skip
    tbz x26, #1, label_10
    add x21, x0, #0x0
    add x20, x0, #0x4
    KAI_ASM_INST(0x8540cab1)  // ld1rw { z17.s }, p2/Z, [x21]
    KAI_ASM_INST(0x8540ca90)  // ld1rw { z16.s }, p2/Z, [x20]
    fmin z24.s, p2/M, z24.s, z17.s
    fmin z25.s, p2/M, z25.s, z17.s
    fmax z24.s, p2/M, z24.s, z16.s
    fmax z25.s, p2/M, z25.s, z16.s
KAI_ASM_LABEL(label_10)  // Width 2: No activation
    st1w { z24.s }, p2, [x27]
    st1w { z25.s }, p1, [x27, #1, MUL VL]
    b label_41
KAI_ASM_LABEL(label_11)  // Width 3
    mov x20, #0x2
    mov x25, x14
    ld1w { z24.s }, p2/Z, [x9]
    msub x21, x13, x20, x12
    add x20, x9, x10
    ld1w { z25.s }, p2/Z, [x9, #1, MUL VL]
    whilelt p1.s, XZR, x21
    cmp x25, #0x4
    ld1w { z26.s }, p2/Z, [x20]
    mov x24, x11
    incb x9, ALL, MUL #2
    incb x20, ALL, MUL #2
    ble label_13
KAI_ASM_LABEL(label_12)  // Width 3: Multiply loop: Main loop head
    whilelt p0.s, XZR, x25
    ldnt1w { z1.s }, p2/Z, [x9]
    sub x25, x25, #0x4
    ld1rqw { z0.s }, p0/Z, [x24]
    cmp x25, #0x4
    add x24, x24, #0x10
    ldnt1w { z2.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w { z3.s }, p2/Z, [x20]
    addvl x20, x20, #2
    fmla z24.s, z1.s, z0.s[0]
    ldnt1w { z4.s }, p2/Z, [x9]
    fmla z25.s, z2.s, z0.s[0]
    ldnt1w { z5.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    fmla z26.s, z3.s, z0.s[0]
    ldnt1w { z6.s }, p2/Z, [x20]
    addvl x20, x20, #2
    ldnt1w { z7.s }, p2/Z, [x9]
    fmla z24.s, z4.s, z0.s[1]
    ldnt1w { z8.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    fmla z25.s, z5.s, z0.s[1]
    ldnt1w { z9.s }, p2/Z, [x20]
    addvl x20, x20, #2
    fmla z26.s, z6.s, z0.s[1]
    ldnt1w { z10.s }, p2/Z, [x9]
    ldnt1w { z11.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    fmla z24.s, z7.s, z0.s[2]
    ldnt1w { z12.s }, p2/Z, [x20]
    addvl x20, x20, #2
    fmla z25.s, z8.s, z0.s[2]
    fmla z26.s, z9.s, z0.s[2]
    fmla z24.s, z10.s, z0.s[3]
    fmla z25.s, z11.s, z0.s[3]
    fmla z26.s, z12.s, z0.s[3]
    bgt label_12
KAI_ASM_LABEL(label_13)  // Width 3: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x25
    ldnt1w { z13.s }, p2/Z, [x9]
    subs x25, x25, #0x1
    ld1rqw { z0.s }, p0/Z, [x24]
    ldnt1w { z14.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w { z15.s }, p2/Z, [x20]
    addvl x20, x20, #2
    fmla z24.s, z13.s, z0.s[0]
    fmla z25.s, z14.s, z0.s[0]
    fmla z26.s, z15.s, z0.s[0]
    ble label_14
    ldnt1w { z16.s }, p2/Z, [x9]
    subs x25, x25, #0x1
    ldnt1w { z17.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w { z18.s }, p2/Z, [x20]
    addvl x20, x20, #2
    fmla z24.s, z16.s, z0.s[1]
    fmla z25.s, z17.s, z0.s[1]
    fmla z26.s, z18.s, z0.s[1]
    ble label_14
    ldnt1w { z19.s }, p2/Z, [x9]
    subs x25, x25, #0x1
    ldnt1w { z20.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w { z21.s }, p2/Z, [x20]
    addvl x20, x20, #2
    fmla z24.s, z19.s, z0.s[2]
    fmla z25.s, z20.s, z0.s[2]
    fmla z26.s, z21.s, z0.s[2]
    ble label_14
    ldnt1w { z22.s }, p2/Z, [x9]
    ldnt1w { z23.s }, p2/Z, [x9, #1, MUL VL]
    ldnt1w { z1.s }, p2/Z, [x20]
    fmla z24.s, z22.s, z0.s[3]
    fmla z25.s, z23.s, z0.s[3]
    fmla z26.s, z1.s, z0.s[3]
KAI_ASM_LABEL(label_14)  // Width 3: Multiply loop: multiply skip
    tbz x26, #1, label_15
    add x21, x0, #0x0
    add x20, x0, #0x4
    KAI_ASM_INST(0x8540cab1)  // ld1rw { z17.s }, p2/Z, [x21]
    KAI_ASM_INST(0x8540ca90)  // ld1rw { z16.s }, p2/Z, [x20]
    fmin z24.s, p2/M, z24.s, z17.s
    fmin z25.s, p2/M, z25.s, z17.s
    fmin z26.s, p2/M, z26.s, z17.s
    fmax z24.s, p2/M, z24.s, z16.s
    fmax z25.s, p2/M, z25.s, z16.s
    fmax z26.s, p2/M, z26.s, z16.s
KAI_ASM_LABEL(label_15)  // Width 3: No activation
    st1w { z24.s }, p2, [x27]
    st1w { z25.s }, p2, [x27, #1, MUL VL]
    st1w { z26.s }, p1, [x27, #2, MUL VL]
    b label_41
KAI_ASM_LABEL(label_16)  // Width 4
    mov x20, #0x3
    mov x25, x14
    ld1w { z24.s }, p2/Z, [x9]
    msub x21, x13, x20, x12
    add x20, x9, x10
    ld1w { z25.s }, p2/Z, [x9, #1, MUL VL]
    whilelt p1.s, XZR, x21
    cmp x25, #0x4
    ld1w { z26.s }, p2/Z, [x20]
    mov x24, x11
    ld1w { z27.s }, p2/Z, [x20, #1, MUL VL]
    incb x9, ALL, MUL #2
    incb x20, ALL, MUL #2
    ble label_18
KAI_ASM_LABEL(label_17)  // Width 4: Multiply loop: Main loop head
    whilelt p0.s, XZR, x25
    ldnt1w { z1.s }, p2/Z, [x9]
    sub x25, x25, #0x4
    ld1rqw { z0.s }, p0/Z, [x24]
    cmp x25, #0x4
    add x24, x24, #0x10
    ldnt1w { z2.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w { z3.s }, p2/Z, [x20]
    ldnt1w { z4.s }, p2/Z, [x20, #1, MUL VL]
    fmla z24.s, z1.s, z0.s[0]
    addvl x20, x20, #2
    fmla z25.s, z2.s, z0.s[0]
    ldnt1w { z5.s }, p2/Z, [x9]
    fmla z26.s, z3.s, z0.s[0]
    ldnt1w { z6.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    fmla z27.s, z4.s, z0.s[0]
    ldnt1w { z7.s }, p2/Z, [x20]
    ldnt1w { z8.s }, p2/Z, [x20, #1, MUL VL]
    fmla z24.s, z5.s, z0.s[1]
    addvl x20, x20, #2
    fmla z25.s, z6.s, z0.s[1]
    ldnt1w { z9.s }, p2/Z, [x9]
    fmla z26.s, z7.s, z0.s[1]
    ldnt1w { z10.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    fmla z27.s, z8.s, z0.s[1]
    ldnt1w { z11.s }, p2/Z, [x20]
    ldnt1w { z12.s }, p2/Z, [x20, #1, MUL VL]
    fmla z24.s, z9.s, z0.s[2]
    addvl x20, x20, #2
    fmla z25.s, z10.s, z0.s[2]
    ldnt1w { z13.s }, p2/Z, [x9]
    fmla z26.s, z11.s, z0.s[2]
    ldnt1w { z14.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    fmla z27.s, z12.s, z0.s[2]
    ldnt1w { z15.s }, p2/Z, [x20]
    ldnt1w { z16.s }, p2/Z, [x20, #1, MUL VL]
    fmla z24.s, z13.s, z0.s[3]
    addvl x20, x20, #2
    fmla z25.s, z14.s, z0.s[3]
    fmla z26.s, z15.s, z0.s[3]
    fmla z27.s, z16.s, z0.s[3]
    bgt label_17
KAI_ASM_LABEL(label_18)  // Width 4: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x25
    ldnt1w { z17.s }, p2/Z, [x9]
    subs x25, x25, #0x1
    ld1rqw { z0.s }, p0/Z, [x24]
    ldnt1w { z18.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w { z19.s }, p2/Z, [x20]
    ldnt1w { z20.s }, p2/Z, [x20, #1, MUL VL]
    fmla z24.s, z17.s, z0.s[0]
    addvl x20, x20, #2
    fmla z25.s, z18.s, z0.s[0]
    fmla z26.s, z19.s, z0.s[0]
    fmla z27.s, z20.s, z0.s[0]
    ble label_19
    ldnt1w { z21.s }, p2/Z, [x9]
    subs x25, x25, #0x1
    ldnt1w { z22.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w { z23.s }, p2/Z, [x20]
    ldnt1w { z1.s }, p2/Z, [x20, #1, MUL VL]
    fmla z24.s, z21.s, z0.s[1]
    addvl x20, x20, #2
    fmla z25.s, z22.s, z0.s[1]
    fmla z26.s, z23.s, z0.s[1]
    fmla z27.s, z1.s, z0.s[1]
    ble label_19
    ldnt1w { z2.s }, p2/Z, [x9]
    subs x25, x25, #0x1
    ldnt1w { z3.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w { z4.s }, p2/Z, [x20]
    ldnt1w { z5.s }, p2/Z, [x20, #1, MUL VL]
    fmla z24.s, z2.s, z0.s[2]
    addvl x20, x20, #2
    fmla z25.s, z3.s, z0.s[2]
    fmla z26.s, z4.s, z0.s[2]
    fmla z27.s, z5.s, z0.s[2]
    ble label_19
    ldnt1w { z6.s }, p2/Z, [x9]
    ldnt1w { z7.s }, p2/Z, [x9, #1, MUL VL]
    ldnt1w { z8.s }, p2/Z, [x20]
    ldnt1w { z9.s }, p2/Z, [x20, #1, MUL VL]
    fmla z24.s, z6.s, z0.s[3]
    fmla z25.s, z7.s, z0.s[3]
    fmla z26.s, z8.s, z0.s[3]
    fmla z27.s, z9.s, z0.s[3]
KAI_ASM_LABEL(label_19)  // Width 4: Multiply loop: multiply skip
    tbz x26, #1, label_20
    add x21, x0, #0x0
    add x20, x0, #0x4
    KAI_ASM_INST(0x8540cab1)  // ld1rw { z17.s }, p2/Z, [x21]
    KAI_ASM_INST(0x8540ca90)  // ld1rw { z16.s }, p2/Z, [x20]
    fmin z24.s, p2/M, z24.s, z17.s
    fmin z25.s, p2/M, z25.s, z17.s
    fmin z26.s, p2/M, z26.s, z17.s
    fmin z27.s, p2/M, z27.s, z17.s
    fmax z24.s, p2/M, z24.s, z16.s
    fmax z25.s, p2/M, z25.s, z16.s
    fmax z26.s, p2/M, z26.s, z16.s
    fmax z27.s, p2/M, z27.s, z16.s
KAI_ASM_LABEL(label_20)  // Width 4: No activation
    st1w { z24.s }, p2, [x27]
    st1w { z25.s }, p2, [x27, #1, MUL VL]
    st1w { z26.s }, p2, [x27, #2, MUL VL]
    st1w { z27.s }, p1, [x27, #3, MUL VL]
    b label_41
KAI_ASM_LABEL(label_21)  // Width 5
    mov x20, #0x4
    mov x25, x14
    ld1w { z24.s }, p2/Z, [x9]
    msub x22, x13, x20, x12
    add x21, x9, x10
    ld1w { z25.s }, p2/Z, [x9, #1, MUL VL]
    add x20, x9, x10, LSL #1
    whilelt p1.s, XZR, x22
    ld1w { z26.s }, p2/Z, [x21]
    cmp x25, #0x4
    mov x24, x11
    ld1w { z27.s }, p2/Z, [x21, #1, MUL VL]
    ld1w { z28.s }, p2/Z, [x20]
    incb x9, ALL, MUL #2
    incb x21, ALL, MUL #2
    incb x20, ALL, MUL #2
    ble label_23
KAI_ASM_LABEL(label_22)  // Width 5: Multiply loop: Main loop head
    whilelt p0.s, XZR, x25
    ldnt1w { z1.s }, p2/Z, [x9]
    sub x25, x25, #0x4
    ld1rqw { z0.s }, p0/Z, [x24]
    cmp x25, #0x4
    add x24, x24, #0x10
    ldnt1w { z2.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w { z3.s }, p2/Z, [x21]
    ldnt1w { z4.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z1.s, z0.s[0]
    addvl x21, x21, #2
    ldnt1w { z5.s }, p2/Z, [x20]
    fmla z25.s, z2.s, z0.s[0]
    addvl x20, x20, #2
    fmla z26.s, z3.s, z0.s[0]
    ldnt1w { z6.s }, p2/Z, [x9]
    fmla z27.s, z4.s, z0.s[0]
    ldnt1w { z7.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    fmla z28.s, z5.s, z0.s[0]
    ldnt1w { z8.s }, p2/Z, [x21]
    ldnt1w { z9.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z6.s, z0.s[1]
    addvl x21, x21, #2
    ldnt1w { z10.s }, p2/Z, [x20]
    fmla z25.s, z7.s, z0.s[1]
    addvl x20, x20, #2
    fmla z26.s, z8.s, z0.s[1]
    ldnt1w { z11.s }, p2/Z, [x9]
    fmla z27.s, z9.s, z0.s[1]
    ldnt1w { z12.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    fmla z28.s, z10.s, z0.s[1]
    ldnt1w { z13.s }, p2/Z, [x21]
    ldnt1w { z14.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z11.s, z0.s[2]
    addvl x21, x21, #2
    ldnt1w { z15.s }, p2/Z, [x20]
    fmla z25.s, z12.s, z0.s[2]
    addvl x20, x20, #2
    fmla z26.s, z13.s, z0.s[2]
    ldnt1w { z16.s }, p2/Z, [x9]
    fmla z27.s, z14.s, z0.s[2]
    ldnt1w { z17.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    fmla z28.s, z15.s, z0.s[2]
    ldnt1w { z18.s }, p2/Z, [x21]
    ldnt1w { z19.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z16.s, z0.s[3]
    addvl x21, x21, #2
    ldnt1w { z20.s }, p2/Z, [x20]
    fmla z25.s, z17.s, z0.s[3]
    addvl x20, x20, #2
    fmla z26.s, z18.s, z0.s[3]
    fmla z27.s, z19.s, z0.s[3]
    fmla z28.s, z20.s, z0.s[3]
    bgt label_22
KAI_ASM_LABEL(label_23)  // Width 5: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x25
    ldnt1w { z21.s }, p2/Z, [x9]
    subs x25, x25, #0x1
    ld1rqw { z0.s }, p0/Z, [x24]
    ldnt1w { z22.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w { z23.s }, p2/Z, [x21]
    ldnt1w { z1.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z21.s, z0.s[0]
    addvl x21, x21, #2
    ldnt1w { z2.s }, p2/Z, [x20]
    fmla z25.s, z22.s, z0.s[0]
    addvl x20, x20, #2
    fmla z26.s, z23.s, z0.s[0]
    fmla z27.s, z1.s, z0.s[0]
    fmla z28.s, z2.s, z0.s[0]
    ble label_24
    ldnt1w { z3.s }, p2/Z, [x9]
    subs x25, x25, #0x1
    ldnt1w { z4.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w { z5.s }, p2/Z, [x21]
    ldnt1w { z6.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z3.s, z0.s[1]
    addvl x21, x21, #2
    ldnt1w { z7.s }, p2/Z, [x20]
    fmla z25.s, z4.s, z0.s[1]
    addvl x20, x20, #2
    fmla z26.s, z5.s, z0.s[1]
    fmla z27.s, z6.s, z0.s[1]
    fmla z28.s, z7.s, z0.s[1]
    ble label_24
    ldnt1w { z8.s }, p2/Z, [x9]
    subs x25, x25, #0x1
    ldnt1w { z9.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w { z10.s }, p2/Z, [x21]
    ldnt1w { z11.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z8.s, z0.s[2]
    addvl x21, x21, #2
    ldnt1w { z12.s }, p2/Z, [x20]
    fmla z25.s, z9.s, z0.s[2]
    addvl x20, x20, #2
    fmla z26.s, z10.s, z0.s[2]
    fmla z27.s, z11.s, z0.s[2]
    fmla z28.s, z12.s, z0.s[2]
    ble label_24
    ldnt1w { z13.s }, p2/Z, [x9]
    ldnt1w { z14.s }, p2/Z, [x9, #1, MUL VL]
    ldnt1w { z15.s }, p2/Z, [x21]
    ldnt1w { z16.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z13.s, z0.s[3]
    ldnt1w { z17.s }, p2/Z, [x20]
    fmla z25.s, z14.s, z0.s[3]
    fmla z26.s, z15.s, z0.s[3]
    fmla z27.s, z16.s, z0.s[3]
    fmla z28.s, z17.s, z0.s[3]
KAI_ASM_LABEL(label_24)  // Width 5: Multiply loop: multiply skip
    tbz x26, #1, label_25
    add x21, x0, #0x0
    add x20, x0, #0x4
    KAI_ASM_INST(0x8540cab1)  // ld1rw { z17.s }, p2/Z, [x21]
    KAI_ASM_INST(0x8540ca90)  // ld1rw { z16.s }, p2/Z, [x20]
    fmin z24.s, p2/M, z24.s, z17.s
    fmin z25.s, p2/M, z25.s, z17.s
    fmin z26.s, p2/M, z26.s, z17.s
    fmin z27.s, p2/M, z27.s, z17.s
    fmin z28.s, p2/M, z28.s, z17.s
    fmax z24.s, p2/M, z24.s, z16.s
    fmax z25.s, p2/M, z25.s, z16.s
    fmax z26.s, p2/M, z26.s, z16.s
    fmax z27.s, p2/M, z27.s, z16.s
    fmax z28.s, p2/M, z28.s, z16.s
KAI_ASM_LABEL(label_25)  // Width 5: No activation
    st1w { z24.s }, p2, [x27]
    st1w { z25.s }, p2, [x27, #1, MUL VL]
    st1w { z26.s }, p2, [x27, #2, MUL VL]
    st1w { z27.s }, p2, [x27, #3, MUL VL]
    st1w { z28.s }, p1, [x27, #4, MUL VL]
    b label_41
KAI_ASM_LABEL(label_26)  // Width 6
    mov x20, #0x5
    mov x25, x14
    ld1w { z24.s }, p2/Z, [x9]
    msub x22, x13, x20, x12
    add x21, x9, x10
    ld1w { z25.s }, p2/Z, [x9, #1, MUL VL]
    add x20, x9, x10, LSL #1
    whilelt p1.s, XZR, x22
    ld1w { z26.s }, p2/Z, [x21]
    cmp x25, #0x4
    mov x24, x11
    ld1w { z27.s }, p2/Z, [x21, #1, MUL VL]
    ld1w { z28.s }, p2/Z, [x20]
    incb x9, ALL, MUL #2
    incb x21, ALL, MUL #2
    ld1w { z29.s }, p2/Z, [x20, #1, MUL VL]
    incb x20, ALL, MUL #2
    ble label_28
KAI_ASM_LABEL(label_27)  // Width 6: Multiply loop: Main loop head
    whilelt p0.s, XZR, x25
    ldnt1w { z1.s }, p2/Z, [x9]
    sub x25, x25, #0x4
    ld1rqw { z0.s }, p0/Z, [x24]
    cmp x25, #0x4
    add x24, x24, #0x10
    ldnt1w { z2.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w { z3.s }, p2/Z, [x21]
    ldnt1w { z4.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z1.s, z0.s[0]
    addvl x21, x21, #2
    ldnt1w { z5.s }, p2/Z, [x20]
    fmla z25.s, z2.s, z0.s[0]
    ldnt1w { z6.s }, p2/Z, [x20, #1, MUL VL]
    fmla z26.s, z3.s, z0.s[0]
    addvl x20, x20, #2
    fmla z27.s, z4.s, z0.s[0]
    ldnt1w { z7.s }, p2/Z, [x9]
    fmla z28.s, z5.s, z0.s[0]
    ldnt1w { z8.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    fmla z29.s, z6.s, z0.s[0]
    ldnt1w { z9.s }, p2/Z, [x21]
    ldnt1w { z10.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z7.s, z0.s[1]
    addvl x21, x21, #2
    ldnt1w { z11.s }, p2/Z, [x20]
    fmla z25.s, z8.s, z0.s[1]
    ldnt1w { z12.s }, p2/Z, [x20, #1, MUL VL]
    fmla z26.s, z9.s, z0.s[1]
    addvl x20, x20, #2
    fmla z27.s, z10.s, z0.s[1]
    ldnt1w { z13.s }, p2/Z, [x9]
    fmla z28.s, z11.s, z0.s[1]
    ldnt1w { z14.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    fmla z29.s, z12.s, z0.s[1]
    ldnt1w { z15.s }, p2/Z, [x21]
    ldnt1w { z16.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z13.s, z0.s[2]
    addvl x21, x21, #2
    ldnt1w { z17.s }, p2/Z, [x20]
    fmla z25.s, z14.s, z0.s[2]
    ldnt1w { z18.s }, p2/Z, [x20, #1, MUL VL]
    fmla z26.s, z15.s, z0.s[2]
    addvl x20, x20, #2
    fmla z27.s, z16.s, z0.s[2]
    ldnt1w { z19.s }, p2/Z, [x9]
    fmla z28.s, z17.s, z0.s[2]
    ldnt1w { z20.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    fmla z29.s, z18.s, z0.s[2]
    ldnt1w { z21.s }, p2/Z, [x21]
    ldnt1w { z22.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z19.s, z0.s[3]
    addvl x21, x21, #2
    ldnt1w { z23.s }, p2/Z, [x20]
    fmla z25.s, z20.s, z0.s[3]
    ldnt1w { z1.s }, p2/Z, [x20, #1, MUL VL]
    fmla z26.s, z21.s, z0.s[3]
    addvl x20, x20, #2
    fmla z27.s, z22.s, z0.s[3]
    fmla z28.s, z23.s, z0.s[3]
    fmla z29.s, z1.s, z0.s[3]
    bgt label_27
KAI_ASM_LABEL(label_28)  // Width 6: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x25
    ldnt1w { z2.s }, p2/Z, [x9]
    subs x25, x25, #0x1
    ld1rqw { z0.s }, p0/Z, [x24]
    ldnt1w { z3.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w { z4.s }, p2/Z, [x21]
    ldnt1w { z5.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z2.s, z0.s[0]
    addvl x21, x21, #2
    ldnt1w { z6.s }, p2/Z, [x20]
    fmla z25.s, z3.s, z0.s[0]
    ldnt1w { z7.s }, p2/Z, [x20, #1, MUL VL]
    fmla z26.s, z4.s, z0.s[0]
    addvl x20, x20, #2
    fmla z27.s, z5.s, z0.s[0]
    fmla z28.s, z6.s, z0.s[0]
    fmla z29.s, z7.s, z0.s[0]
    ble label_29
    ldnt1w { z8.s }, p2/Z, [x9]
    subs x25, x25, #0x1
    ldnt1w { z9.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w { z10.s }, p2/Z, [x21]
    ldnt1w { z11.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z8.s, z0.s[1]
    addvl x21, x21, #2
    ldnt1w { z12.s }, p2/Z, [x20]
    fmla z25.s, z9.s, z0.s[1]
    ldnt1w { z13.s }, p2/Z, [x20, #1, MUL VL]
    fmla z26.s, z10.s, z0.s[1]
    addvl x20, x20, #2
    fmla z27.s, z11.s, z0.s[1]
    fmla z28.s, z12.s, z0.s[1]
    fmla z29.s, z13.s, z0.s[1]
    ble label_29
    ldnt1w { z14.s }, p2/Z, [x9]
    subs x25, x25, #0x1
    ldnt1w { z15.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w { z16.s }, p2/Z, [x21]
    ldnt1w { z17.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z14.s, z0.s[2]
    addvl x21, x21, #2
    ldnt1w { z18.s }, p2/Z, [x20]
    fmla z25.s, z15.s, z0.s[2]
    ldnt1w { z19.s }, p2/Z, [x20, #1, MUL VL]
    fmla z26.s, z16.s, z0.s[2]
    addvl x20, x20, #2
    fmla z27.s, z17.s, z0.s[2]
    fmla z28.s, z18.s, z0.s[2]
    fmla z29.s, z19.s, z0.s[2]
    ble label_29
    ldnt1w { z20.s }, p2/Z, [x9]
    ldnt1w { z21.s }, p2/Z, [x9, #1, MUL VL]
    ldnt1w { z22.s }, p2/Z, [x21]
    ldnt1w { z23.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z20.s, z0.s[3]
    ldnt1w { z1.s }, p2/Z, [x20]
    fmla z25.s, z21.s, z0.s[3]
    ldnt1w { z2.s }, p2/Z, [x20, #1, MUL VL]
    fmla z26.s, z22.s, z0.s[3]
    fmla z27.s, z23.s, z0.s[3]
    fmla z28.s, z1.s, z0.s[3]
    fmla z29.s, z2.s, z0.s[3]
KAI_ASM_LABEL(label_29)  // Width 6: Multiply loop: multiply skip
    tbz x26, #1, label_30
    add x21, x0, #0x0
    add x20, x0, #0x4
    KAI_ASM_INST(0x8540cab1)  // ld1rw { z17.s }, p2/Z, [x21]
    KAI_ASM_INST(0x8540ca90)  // ld1rw { z16.s }, p2/Z, [x20]
    fmin z24.s, p2/M, z24.s, z17.s
    fmin z25.s, p2/M, z25.s, z17.s
    fmin z26.s, p2/M, z26.s, z17.s
    fmin z27.s, p2/M, z27.s, z17.s
    fmin z28.s, p2/M, z28.s, z17.s
    fmin z29.s, p2/M, z29.s, z17.s
    fmax z24.s, p2/M, z24.s, z16.s
    fmax z25.s, p2/M, z25.s, z16.s
    fmax z26.s, p2/M, z26.s, z16.s
    fmax z27.s, p2/M, z27.s, z16.s
    fmax z28.s, p2/M, z28.s, z16.s
    fmax z29.s, p2/M, z29.s, z16.s
KAI_ASM_LABEL(label_30)  // Width 6: No activation
    st1w { z24.s }, p2, [x27]
    st1w { z25.s }, p2, [x27, #1, MUL VL]
    st1w { z26.s }, p2, [x27, #2, MUL VL]
    st1w { z27.s }, p2, [x27, #3, MUL VL]
    st1w { z28.s }, p2, [x27, #4, MUL VL]
    st1w { z29.s }, p1, [x27, #5, MUL VL]
    b label_41
KAI_ASM_LABEL(label_31)  // Width 7
    mov x20, #0x6
    mov x25, x14
    ld1w { z24.s }, p2/Z, [x9]
    add x23, x9, x10, LSL #1
    msub x22, x13, x20, x12
    ld1w { z25.s }, p2/Z, [x9, #1, MUL VL]
    add x21, x9, x10
    add x20, x23, x10
    ld1w { z28.s }, p2/Z, [x23]
    whilelt p1.s, XZR, x22
    cmp x25, #0x4
    ld1w { z26.s }, p2/Z, [x21]
    mov x24, x11
    ld1w { z27.s }, p2/Z, [x21, #1, MUL VL]
    incb x9, ALL, MUL #2
    ld1w { z29.s }, p2/Z, [x23, #1, MUL VL]
    incb x21, ALL, MUL #2
    incb x23, ALL, MUL #2
    ld1w { z30.s }, p2/Z, [x20]
    incb x20, ALL, MUL #2
    ble label_33
KAI_ASM_LABEL(label_32)  // Width 7: Multiply loop: Main loop head
    whilelt p0.s, XZR, x25
    ldnt1w { z1.s }, p2/Z, [x9]
    sub x25, x25, #0x4
    ld1rqw { z0.s }, p0/Z, [x24]
    cmp x25, #0x4
    add x24, x24, #0x10
    ldnt1w { z2.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w { z3.s }, p2/Z, [x21]
    ldnt1w { z4.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z1.s, z0.s[0]
    addvl x21, x21, #2
    ldnt1w { z5.s }, p2/Z, [x23]
    fmla z25.s, z2.s, z0.s[0]
    ldnt1w { z6.s }, p2/Z, [x23, #1, MUL VL]
    fmla z26.s, z3.s, z0.s[0]
    addvl x23, x23, #2
    ldnt1w { z7.s }, p2/Z, [x20]
    fmla z27.s, z4.s, z0.s[0]
    addvl x20, x20, #2
    fmla z28.s, z5.s, z0.s[0]
    ldnt1w { z8.s }, p2/Z, [x9]
    fmla z29.s, z6.s, z0.s[0]
    ldnt1w { z9.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    fmla z30.s, z7.s, z0.s[0]
    ldnt1w { z10.s }, p2/Z, [x21]
    ldnt1w { z11.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z8.s, z0.s[1]
    addvl x21, x21, #2
    ldnt1w { z12.s }, p2/Z, [x23]
    fmla z25.s, z9.s, z0.s[1]
    ldnt1w { z13.s }, p2/Z, [x23, #1, MUL VL]
    fmla z26.s, z10.s, z0.s[1]
    addvl x23, x23, #2
    ldnt1w { z14.s }, p2/Z, [x20]
    fmla z27.s, z11.s, z0.s[1]
    addvl x20, x20, #2
    fmla z28.s, z12.s, z0.s[1]
    ldnt1w { z15.s }, p2/Z, [x9]
    fmla z29.s, z13.s, z0.s[1]
    ldnt1w { z16.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    fmla z30.s, z14.s, z0.s[1]
    ldnt1w { z17.s }, p2/Z, [x21]
    ldnt1w { z18.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z15.s, z0.s[2]
    addvl x21, x21, #2
    ldnt1w { z19.s }, p2/Z, [x23]
    fmla z25.s, z16.s, z0.s[2]
    ldnt1w { z20.s }, p2/Z, [x23, #1, MUL VL]
    fmla z26.s, z17.s, z0.s[2]
    addvl x23, x23, #2
    ldnt1w { z21.s }, p2/Z, [x20]
    fmla z27.s, z18.s, z0.s[2]
    addvl x20, x20, #2
    fmla z28.s, z19.s, z0.s[2]
    ldnt1w { z22.s }, p2/Z, [x9]
    fmla z29.s, z20.s, z0.s[2]
    ldnt1w { z23.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    fmla z30.s, z21.s, z0.s[2]
    ldnt1w { z1.s }, p2/Z, [x21]
    ldnt1w { z2.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z22.s, z0.s[3]
    addvl x21, x21, #2
    ldnt1w { z3.s }, p2/Z, [x23]
    fmla z25.s, z23.s, z0.s[3]
    ldnt1w { z4.s }, p2/Z, [x23, #1, MUL VL]
    fmla z26.s, z1.s, z0.s[3]
    addvl x23, x23, #2
    ldnt1w { z5.s }, p2/Z, [x20]
    fmla z27.s, z2.s, z0.s[3]
    addvl x20, x20, #2
    fmla z28.s, z3.s, z0.s[3]
    fmla z29.s, z4.s, z0.s[3]
    fmla z30.s, z5.s, z0.s[3]
    bgt label_32
KAI_ASM_LABEL(label_33)  // Width 7: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x25
    ldnt1w { z6.s }, p2/Z, [x9]
    subs x25, x25, #0x1
    ld1rqw { z0.s }, p0/Z, [x24]
    ldnt1w { z7.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w { z8.s }, p2/Z, [x21]
    ldnt1w { z9.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z6.s, z0.s[0]
    addvl x21, x21, #2
    ldnt1w { z10.s }, p2/Z, [x23]
    fmla z25.s, z7.s, z0.s[0]
    ldnt1w { z11.s }, p2/Z, [x23, #1, MUL VL]
    fmla z26.s, z8.s, z0.s[0]
    addvl x23, x23, #2
    ldnt1w { z12.s }, p2/Z, [x20]
    fmla z27.s, z9.s, z0.s[0]
    addvl x20, x20, #2
    fmla z28.s, z10.s, z0.s[0]
    fmla z29.s, z11.s, z0.s[0]
    fmla z30.s, z12.s, z0.s[0]
    ble label_34
    ldnt1w { z13.s }, p2/Z, [x9]
    subs x25, x25, #0x1
    ldnt1w { z14.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w { z15.s }, p2/Z, [x21]
    ldnt1w { z16.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z13.s, z0.s[1]
    addvl x21, x21, #2
    ldnt1w { z17.s }, p2/Z, [x23]
    fmla z25.s, z14.s, z0.s[1]
    ldnt1w { z18.s }, p2/Z, [x23, #1, MUL VL]
    fmla z26.s, z15.s, z0.s[1]
    addvl x23, x23, #2
    ldnt1w { z19.s }, p2/Z, [x20]
    fmla z27.s, z16.s, z0.s[1]
    addvl x20, x20, #2
    fmla z28.s, z17.s, z0.s[1]
    fmla z29.s, z18.s, z0.s[1]
    fmla z30.s, z19.s, z0.s[1]
    ble label_34
    ldnt1w { z20.s }, p2/Z, [x9]
    subs x25, x25, #0x1
    ldnt1w { z21.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w { z22.s }, p2/Z, [x21]
    ldnt1w { z23.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z20.s, z0.s[2]
    addvl x21, x21, #2
    ldnt1w { z1.s }, p2/Z, [x23]
    fmla z25.s, z21.s, z0.s[2]
    ldnt1w { z2.s }, p2/Z, [x23, #1, MUL VL]
    fmla z26.s, z22.s, z0.s[2]
    addvl x23, x23, #2
    ldnt1w { z3.s }, p2/Z, [x20]
    fmla z27.s, z23.s, z0.s[2]
    addvl x20, x20, #2
    fmla z28.s, z1.s, z0.s[2]
    fmla z29.s, z2.s, z0.s[2]
    fmla z30.s, z3.s, z0.s[2]
    ble label_34
    ldnt1w { z4.s }, p2/Z, [x9]
    ldnt1w { z5.s }, p2/Z, [x9, #1, MUL VL]
    ldnt1w { z6.s }, p2/Z, [x21]
    ldnt1w { z7.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z4.s, z0.s[3]
    ldnt1w { z8.s }, p2/Z, [x23]
    fmla z25.s, z5.s, z0.s[3]
    ldnt1w { z9.s }, p2/Z, [x23, #1, MUL VL]
    fmla z26.s, z6.s, z0.s[3]
    ldnt1w { z10.s }, p2/Z, [x20]
    fmla z27.s, z7.s, z0.s[3]
    fmla z28.s, z8.s, z0.s[3]
    fmla z29.s, z9.s, z0.s[3]
    fmla z30.s, z10.s, z0.s[3]
KAI_ASM_LABEL(label_34)  // Width 7: Multiply loop: multiply skip
    tbz x26, #1, label_35
    add x21, x0, #0x0
    add x20, x0, #0x4
    KAI_ASM_INST(0x8540cab1)  // ld1rw { z17.s }, p2/Z, [x21]
    KAI_ASM_INST(0x8540ca90)  // ld1rw { z16.s }, p2/Z, [x20]
    fmin z24.s, p2/M, z24.s, z17.s
    fmin z25.s, p2/M, z25.s, z17.s
    fmin z26.s, p2/M, z26.s, z17.s
    fmin z27.s, p2/M, z27.s, z17.s
    fmin z28.s, p2/M, z28.s, z17.s
    fmin z29.s, p2/M, z29.s, z17.s
    fmin z30.s, p2/M, z30.s, z17.s
    fmax z24.s, p2/M, z24.s, z16.s
    fmax z25.s, p2/M, z25.s, z16.s
    fmax z26.s, p2/M, z26.s, z16.s
    fmax z27.s, p2/M, z27.s, z16.s
    fmax z28.s, p2/M, z28.s, z16.s
    fmax z29.s, p2/M, z29.s, z16.s
    fmax z30.s, p2/M, z30.s, z16.s
KAI_ASM_LABEL(label_35)  // Width 7: No activation
    st1w { z24.s }, p2, [x27]
    st1w { z25.s }, p2, [x27, #1, MUL VL]
    st1w { z26.s }, p2, [x27, #2, MUL VL]
    st1w { z27.s }, p2, [x27, #3, MUL VL]
    st1w { z28.s }, p2, [x27, #4, MUL VL]
    st1w { z29.s }, p2, [x27, #5, MUL VL]
    st1w { z30.s }, p1, [x27, #6, MUL VL]
    b label_41
KAI_ASM_LABEL(label_36)  // Width 8
    mov x20, #0x7
    mov x25, x14
    ld1w { z24.s }, p2/Z, [x9]
    add x23, x9, x10, LSL #1
    msub x22, x13, x20, x12
    ld1w { z25.s }, p2/Z, [x9, #1, MUL VL]
    add x21, x9, x10
    add x20, x23, x10
    ld1w { z28.s }, p2/Z, [x23]
    whilelt p1.s, XZR, x22
    cmp x25, #0x4
    ld1w { z26.s }, p2/Z, [x21]
    mov x24, x11
    add x22, x9, x10, LSL #2
    ld1w { z27.s }, p2/Z, [x21, #1, MUL VL]
    ld1w { z29.s }, p2/Z, [x23, #1, MUL VL]
    incb x9, ALL, MUL #2
    incb x21, ALL, MUL #2
    ld1w { z30.s }, p2/Z, [x20]
    incb x23, ALL, MUL #2
    ld1w { z31.s }, p2/Z, [x20, #1, MUL VL]
    incb x20, ALL, MUL #2
    ble label_38
KAI_ASM_LABEL(label_37)  // Width 8: Multiply loop: Main loop head
    whilelt p0.s, XZR, x25
    ldnt1w { z1.s }, p2/Z, [x9]
    sub x25, x25, #0x4
    ld1rqw { z0.s }, p0/Z, [x24]
    cmp x25, #0x4
    add x24, x24, #0x10
    ldnt1w { z2.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w { z3.s }, p2/Z, [x21]
    ldnt1w { z4.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z1.s, z0.s[0]
    addvl x21, x21, #2
    ldnt1w { z5.s }, p2/Z, [x23]
    fmla z25.s, z2.s, z0.s[0]
    ldnt1w { z6.s }, p2/Z, [x23, #1, MUL VL]
    fmla z26.s, z3.s, z0.s[0]
    addvl x23, x23, #2
    ldnt1w { z7.s }, p2/Z, [x20]
    fmla z27.s, z4.s, z0.s[0]
    ldnt1w { z8.s }, p2/Z, [x20, #1, MUL VL]
    fmla z28.s, z5.s, z0.s[0]
    addvl x20, x20, #2
    fmla z29.s, z6.s, z0.s[0]
    ldnt1w { z9.s }, p2/Z, [x9]
    fmla z30.s, z7.s, z0.s[0]
    ldnt1w { z10.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    fmla z31.s, z8.s, z0.s[0]
    ldnt1w { z11.s }, p2/Z, [x21]
    ldnt1w { z12.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z9.s, z0.s[1]
    addvl x21, x21, #2
    ldnt1w { z13.s }, p2/Z, [x23]
    fmla z25.s, z10.s, z0.s[1]
    ldnt1w { z14.s }, p2/Z, [x23, #1, MUL VL]
    fmla z26.s, z11.s, z0.s[1]
    addvl x23, x23, #2
    ldnt1w { z15.s }, p2/Z, [x20]
    fmla z27.s, z12.s, z0.s[1]
    ldnt1w { z16.s }, p2/Z, [x20, #1, MUL VL]
    fmla z28.s, z13.s, z0.s[1]
    addvl x20, x20, #2
    fmla z29.s, z14.s, z0.s[1]
    ldnt1w { z17.s }, p2/Z, [x9]
    fmla z30.s, z15.s, z0.s[1]
    ldnt1w { z18.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    fmla z31.s, z16.s, z0.s[1]
    ldnt1w { z19.s }, p2/Z, [x21]
    ldnt1w { z20.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z17.s, z0.s[2]
    addvl x21, x21, #2
    ldnt1w { z21.s }, p2/Z, [x23]
    fmla z25.s, z18.s, z0.s[2]
    ldnt1w { z22.s }, p2/Z, [x23, #1, MUL VL]
    fmla z26.s, z19.s, z0.s[2]
    addvl x23, x23, #2
    ldnt1w { z23.s }, p2/Z, [x20]
    fmla z27.s, z20.s, z0.s[2]
    ldnt1w { z1.s }, p2/Z, [x20, #1, MUL VL]
    fmla z28.s, z21.s, z0.s[2]
    addvl x20, x20, #2
    fmla z29.s, z22.s, z0.s[2]
    ldnt1w { z2.s }, p2/Z, [x9]
    fmla z30.s, z23.s, z0.s[2]
    ldnt1w { z3.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    fmla z31.s, z1.s, z0.s[2]
    ldnt1w { z4.s }, p2/Z, [x21]
    ldnt1w { z5.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z2.s, z0.s[3]
    addvl x21, x21, #2
    ldnt1w { z6.s }, p2/Z, [x23]
    fmla z25.s, z3.s, z0.s[3]
    ldnt1w { z7.s }, p2/Z, [x23, #1, MUL VL]
    fmla z26.s, z4.s, z0.s[3]
    addvl x23, x23, #2
    ldnt1w { z8.s }, p2/Z, [x20]
    fmla z27.s, z5.s, z0.s[3]
    ldnt1w { z9.s }, p2/Z, [x20, #1, MUL VL]
    fmla z28.s, z6.s, z0.s[3]
    addvl x20, x20, #2
    fmla z29.s, z7.s, z0.s[3]
    fmla z30.s, z8.s, z0.s[3]
    fmla z31.s, z9.s, z0.s[3]
    bgt label_37
KAI_ASM_LABEL(label_38)  // Width 8: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x25
    ldnt1w { z10.s }, p2/Z, [x9]
    subs x25, x25, #0x1
    ld1rqw { z0.s }, p0/Z, [x24]
    ldnt1w { z11.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w { z12.s }, p2/Z, [x21]
    ldnt1w { z13.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z10.s, z0.s[0]
    addvl x21, x21, #2
    ldnt1w { z14.s }, p2/Z, [x23]
    fmla z25.s, z11.s, z0.s[0]
    ldnt1w { z15.s }, p2/Z, [x23, #1, MUL VL]
    fmla z26.s, z12.s, z0.s[0]
    addvl x23, x23, #2
    ldnt1w { z16.s }, p2/Z, [x20]
    fmla z27.s, z13.s, z0.s[0]
    ldnt1w { z17.s }, p2/Z, [x20, #1, MUL VL]
    fmla z28.s, z14.s, z0.s[0]
    addvl x20, x20, #2
    fmla z29.s, z15.s, z0.s[0]
    fmla z30.s, z16.s, z0.s[0]
    fmla z31.s, z17.s, z0.s[0]
    ble label_39
    ldnt1w { z18.s }, p2/Z, [x9]
    subs x25, x25, #0x1
    ldnt1w { z19.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w { z20.s }, p2/Z, [x21]
    ldnt1w { z21.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z18.s, z0.s[1]
    addvl x21, x21, #2
    ldnt1w { z22.s }, p2/Z, [x23]
    fmla z25.s, z19.s, z0.s[1]
    ldnt1w { z23.s }, p2/Z, [x23, #1, MUL VL]
    fmla z26.s, z20.s, z0.s[1]
    addvl x23, x23, #2
    ldnt1w { z1.s }, p2/Z, [x20]
    fmla z27.s, z21.s, z0.s[1]
    ldnt1w { z2.s }, p2/Z, [x20, #1, MUL VL]
    fmla z28.s, z22.s, z0.s[1]
    addvl x20, x20, #2
    fmla z29.s, z23.s, z0.s[1]
    fmla z30.s, z1.s, z0.s[1]
    fmla z31.s, z2.s, z0.s[1]
    ble label_39
    ldnt1w { z3.s }, p2/Z, [x9]
    subs x25, x25, #0x1
    ldnt1w { z4.s }, p2/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1w { z5.s }, p2/Z, [x21]
    ldnt1w { z6.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z3.s, z0.s[2]
    addvl x21, x21, #2
    ldnt1w { z7.s }, p2/Z, [x23]
    fmla z25.s, z4.s, z0.s[2]
    ldnt1w { z8.s }, p2/Z, [x23, #1, MUL VL]
    fmla z26.s, z5.s, z0.s[2]
    addvl x23, x23, #2
    ldnt1w { z9.s }, p2/Z, [x20]
    fmla z27.s, z6.s, z0.s[2]
    ldnt1w { z10.s }, p2/Z, [x20, #1, MUL VL]
    fmla z28.s, z7.s, z0.s[2]
    addvl x20, x20, #2
    fmla z29.s, z8.s, z0.s[2]
    fmla z30.s, z9.s, z0.s[2]
    fmla z31.s, z10.s, z0.s[2]
    ble label_39
    ldnt1w { z11.s }, p2/Z, [x9]
    ldnt1w { z12.s }, p2/Z, [x9, #1, MUL VL]
    ldnt1w { z13.s }, p2/Z, [x21]
    ldnt1w { z14.s }, p2/Z, [x21, #1, MUL VL]
    fmla z24.s, z11.s, z0.s[3]
    ldnt1w { z15.s }, p2/Z, [x23]
    fmla z25.s, z12.s, z0.s[3]
    ldnt1w { z16.s }, p2/Z, [x23, #1, MUL VL]
    fmla z26.s, z13.s, z0.s[3]
    ldnt1w { z17.s }, p2/Z, [x20]
    fmla z27.s, z14.s, z0.s[3]
    ldnt1w { z18.s }, p2/Z, [x20, #1, MUL VL]
    fmla z28.s, z15.s, z0.s[3]
    fmla z29.s, z16.s, z0.s[3]
    fmla z30.s, z17.s, z0.s[3]
    fmla z31.s, z18.s, z0.s[3]
KAI_ASM_LABEL(label_39)  // Width 8: Multiply loop: multiply skip
    tbz x26, #1, label_40
    add x21, x0, #0x0
    add x20, x0, #0x4
    KAI_ASM_INST(0x8540cab1)  // ld1rw { z17.s }, p2/Z, [x21]
    KAI_ASM_INST(0x8540ca90)  // ld1rw { z16.s }, p2/Z, [x20]
    fmin z24.s, p2/M, z24.s, z17.s
    fmin z25.s, p2/M, z25.s, z17.s
    fmin z26.s, p2/M, z26.s, z17.s
    fmin z27.s, p2/M, z27.s, z17.s
    fmin z28.s, p2/M, z28.s, z17.s
    fmin z29.s, p2/M, z29.s, z17.s
    fmin z30.s, p2/M, z30.s, z17.s
    fmin z31.s, p2/M, z31.s, z17.s
    fmax z24.s, p2/M, z24.s, z16.s
    fmax z25.s, p2/M, z25.s, z16.s
    fmax z26.s, p2/M, z26.s, z16.s
    fmax z27.s, p2/M, z27.s, z16.s
    fmax z28.s, p2/M, z28.s, z16.s
    fmax z29.s, p2/M, z29.s, z16.s
    fmax z30.s, p2/M, z30.s, z16.s
    fmax z31.s, p2/M, z31.s, z16.s
KAI_ASM_LABEL(label_40)  // Width 8: No activation
    subs x28, x28, #0x8
    st1w { z24.s }, p2, [x27]
    mov x9, x22
    st1w { z25.s }, p2, [x27, #1, MUL VL]
    sub x12, x12, x13, LSL #3
    st1w { z26.s }, p2, [x27, #2, MUL VL]
    st1w { z27.s }, p2, [x27, #3, MUL VL]
    st1w { z28.s }, p2, [x27, #4, MUL VL]
    st1w { z29.s }, p2, [x27, #5, MUL VL]
    st1w { z30.s }, p2, [x27, #6, MUL VL]
    st1w { z31.s }, p1, [x27, #7, MUL VL]
    addvl x27, x27, #8
    bgt label_1
KAI_ASM_LABEL(label_41)  // Exit
    KAI_ASM_INST(0xd503467f)  // SMSTOP
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla)

    KAI_ASM_END
