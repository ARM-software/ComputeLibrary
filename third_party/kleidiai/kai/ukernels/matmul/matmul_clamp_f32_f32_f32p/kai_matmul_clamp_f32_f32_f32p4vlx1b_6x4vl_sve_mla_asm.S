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

    KAI_ASM_CODE(matmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    ptrue p5.b
KAI_ASM_LABEL(label_1)  // Row loop
    cmp x1, #0x6
    bge label_66
    cmp x1, #0x4
    bgt label_53
    beq label_40
    cmp x1, #0x2
    bgt label_27
    beq label_14
    ldr x11, [x2, #0x18]
    ldr x10, [x2, #0x20]
    ldr x9, [x2, #0x40]
KAI_ASM_LABEL(label_2)  // Height 1: Column loop
    mov x20, #0x0
    whilelt p4.s, x20, x11
    incw x20
    whilelt p3.s, x20, x11
    incw x20
    whilelt p2.s, x20, x11
    incw x20
    whilelt p1.s, x20, x11
    cbz x10, label_3
    ld1w { z8.s }, p5/Z, [x10]
    ld1w { z9.s }, p5/Z, [x10, #1, MUL VL]
    ld1w { z10.s }, p5/Z, [x10, #2, MUL VL]
    ld1w { z11.s }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    b label_5
KAI_ASM_LABEL(label_3)  // Height 1: no bias
    tbz x3, #0, label_4
    ld1w { z8.s }, p4/Z, [x9]
    ld1w { z9.s }, p3/Z, [x9, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x9, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x9, #3, MUL VL]
    b label_5
KAI_ASM_LABEL(label_4)  // Height 1: no accumulate
    mov z8.b, #0x0
    mov z9.b, #0x0
    mov z10.b, #0x0
    mov z11.b, #0x0
KAI_ASM_LABEL(label_5)  // Height 1: setup done
    mov x28, #0x0
KAI_ASM_LABEL(label_6)  // Height 1: String loop
    ldr x20, [x2, #0x10]
    ldr x21, [x2, #0x38]
    ldr w27, [x20, x28, LSL #0x2]
    tbz x3, #3, label_7
    ldr x20, [x0, x28, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x26, [x20, #0x0]
    cbnz x28, label_8
    ldr x20, [x2, #0x30]
    add x26, x26, x20, LSL #2
    b label_8
KAI_ASM_LABEL(label_7)  // Height 1: setup direct input
    mov x26, x0
KAI_ASM_LABEL(label_8)  // Height 1: input setup done
    cmp x27, #0x4
    ble label_10
KAI_ASM_LABEL(label_9)  // Height 1: Multiply loop: Main loop head
    whilelt p0.s, XZR, x27
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    sub x27, x27, #0x4
    cmp x27, #0x4
    ld1rqw { z0.s }, p0/Z, [x26]
    add x26, x26, #0x10
    fmla z8.s, z6.s, z0.s[0]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[0]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    fmla z10.s, z6.s, z0.s[0]
    ld1w { z6.s }, p5/Z, [x10, #4, MUL VL]
    fmla z11.s, z7.s, z0.s[0]
    ld1w { z7.s }, p5/Z, [x10, #5, MUL VL]
    fmla z8.s, z6.s, z0.s[1]
    ld1w { z6.s }, p5/Z, [x10, #6, MUL VL]
    fmla z9.s, z7.s, z0.s[1]
    ld1w { z7.s }, p5/Z, [x10, #7, MUL VL]
    addvl x10, x10, #16
    fmla z10.s, z6.s, z0.s[1]
    fmla z11.s, z7.s, z0.s[1]
    ld1w { z6.s }, p5/Z, [x10, #-8, MUL VL]
    ld1w { z7.s }, p5/Z, [x10, #-7, MUL VL]
    fmla z8.s, z6.s, z0.s[2]
    ld1w { z6.s }, p5/Z, [x10, #-6, MUL VL]
    fmla z9.s, z7.s, z0.s[2]
    ld1w { z7.s }, p5/Z, [x10, #-5, MUL VL]
    fmla z10.s, z6.s, z0.s[2]
    ld1w { z6.s }, p5/Z, [x10, #-4, MUL VL]
    fmla z11.s, z7.s, z0.s[2]
    ld1w { z7.s }, p5/Z, [x10, #-3, MUL VL]
    fmla z8.s, z6.s, z0.s[3]
    ld1w { z6.s }, p5/Z, [x10, #-2, MUL VL]
    fmla z9.s, z7.s, z0.s[3]
    ld1w { z7.s }, p5/Z, [x10, #-1, MUL VL]
    fmla z10.s, z6.s, z0.s[3]
    fmla z11.s, z7.s, z0.s[3]
    bgt label_9
KAI_ASM_LABEL(label_10)  // Height 1: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x27
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x1
    ld1rqw { z0.s }, p0/Z, [x26]
    fmla z8.s, z6.s, z0.s[0]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[0]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    fmla z10.s, z6.s, z0.s[0]
    fmla z11.s, z7.s, z0.s[0]
    ble label_11
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x1
    fmla z8.s, z6.s, z0.s[1]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[1]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    fmla z10.s, z6.s, z0.s[1]
    fmla z11.s, z7.s, z0.s[1]
    ble label_11
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x1
    fmla z8.s, z6.s, z0.s[2]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[2]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    fmla z10.s, z6.s, z0.s[2]
    fmla z11.s, z7.s, z0.s[2]
    ble label_11
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    fmla z8.s, z6.s, z0.s[3]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[3]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    fmla z10.s, z6.s, z0.s[3]
    fmla z11.s, z7.s, z0.s[3]
KAI_ASM_LABEL(label_11)  // Height 1: Multiply loop: multiply skip
    ldr w20, [x2, #0x8]
    add x28, x28, #0x1
    cmp x28, x20
    bne label_6
    tbz x3, #1, label_12
    add x21, x2, #0x0
    add x20, x2, #0x4
    KAI_ASM_INST(0x8540d6b1)  // ld1rw { z17.s }, p5/Z, [x21]
    KAI_ASM_INST(0x8540d690)  // ld1rw { z16.s }, p5/Z, [x20]
    fmin z8.s, p5/M, z8.s, z17.s
    fmin z9.s, p5/M, z9.s, z17.s
    fmin z10.s, p5/M, z10.s, z17.s
    fmin z11.s, p5/M, z11.s, z17.s
    fmax z8.s, p5/M, z8.s, z16.s
    fmax z9.s, p5/M, z9.s, z16.s
    fmax z10.s, p5/M, z10.s, z16.s
    fmax z11.s, p5/M, z11.s, z16.s
KAI_ASM_LABEL(label_12)  // Height 1: No activation
    st1w { z8.s }, p4, [x9]
    st1w { z9.s }, p3, [x9, #1, MUL VL]
    st1w { z10.s }, p2, [x9, #2, MUL VL]
    st1w { z11.s }, p1, [x9, #3, MUL VL]
    addvl x9, x9, #4
    decw x11, ALL, MUL #4
    cmp x11, XZR
    bgt label_2
    b label_80
KAI_ASM_LABEL(label_14)  // Height 2
    ldr x11, [x2, #0x18]
    ldr x10, [x2, #0x20]
    ldr x9, [x2, #0x40]
KAI_ASM_LABEL(label_15)  // Height 2: Column loop
    mov x20, #0x0
    whilelt p4.s, x20, x11
    incw x20
    whilelt p3.s, x20, x11
    incw x20
    whilelt p2.s, x20, x11
    incw x20
    whilelt p1.s, x20, x11
    cbz x10, label_16
    ld1w { z8.s }, p5/Z, [x10]
    ld1w { z9.s }, p5/Z, [x10, #1, MUL VL]
    ld1w { z10.s }, p5/Z, [x10, #2, MUL VL]
    ld1w { z11.s }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    mov z12.d, z8.d
    mov z13.d, z9.d
    mov z14.d, z10.d
    mov z15.d, z11.d
    b label_18
KAI_ASM_LABEL(label_16)  // Height 2: no bias
    tbz x3, #0, label_17
    ldr x20, [x2, #0x28]
    ld1w { z8.s }, p4/Z, [x9]
    ld1w { z9.s }, p3/Z, [x9, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x9, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x9, #3, MUL VL]
    add x20, x9, x20, LSL #2
    ld1w { z12.s }, p4/Z, [x20]
    ld1w { z13.s }, p3/Z, [x20, #1, MUL VL]
    ld1w { z14.s }, p2/Z, [x20, #2, MUL VL]
    ld1w { z15.s }, p1/Z, [x20, #3, MUL VL]
    b label_18
KAI_ASM_LABEL(label_17)  // Height 2: no accumulate
    mov z8.b, #0x0
    mov z9.b, #0x0
    mov z10.b, #0x0
    mov z11.b, #0x0
    mov z12.b, #0x0
    mov z13.b, #0x0
    mov z14.b, #0x0
    mov z15.b, #0x0
KAI_ASM_LABEL(label_18)  // Height 2: setup done
    mov x28, #0x0
KAI_ASM_LABEL(label_19)  // Height 2: String loop
    ldr x20, [x2, #0x10]
    ldr x21, [x2, #0x38]
    ldr w27, [x20, x28, LSL #0x2]
    tbz x3, #3, label_20
    ldr x20, [x0, x28, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x26, [x20, #0x0]
    ldr x25, [x20, #0x8]
    cbnz x28, label_21
    ldr x20, [x2, #0x30]
    add x26, x26, x20, LSL #2
    add x25, x25, x20, LSL #2
    b label_21
KAI_ASM_LABEL(label_20)  // Height 2: setup direct input
    mov x26, x0
    add x25, x26, x21, LSL #2
KAI_ASM_LABEL(label_21)  // Height 2: input setup done
    cmp x27, #0x4
    ble label_23
KAI_ASM_LABEL(label_22)  // Height 2: Multiply loop: Main loop head
    whilelt p0.s, XZR, x27
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    sub x27, x27, #0x4
    cmp x27, #0x4
    ld1rqw { z0.s }, p0/Z, [x26]
    add x26, x26, #0x10
    ld1rqw { z1.s }, p0/Z, [x25]
    add x25, x25, #0x10
    fmla z8.s, z6.s, z0.s[0]
    fmla z12.s, z6.s, z1.s[0]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[0]
    fmla z13.s, z7.s, z1.s[0]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    fmla z10.s, z6.s, z0.s[0]
    fmla z14.s, z6.s, z1.s[0]
    ld1w { z6.s }, p5/Z, [x10, #4, MUL VL]
    fmla z11.s, z7.s, z0.s[0]
    fmla z15.s, z7.s, z1.s[0]
    ld1w { z7.s }, p5/Z, [x10, #5, MUL VL]
    fmla z8.s, z6.s, z0.s[1]
    fmla z12.s, z6.s, z1.s[1]
    ld1w { z6.s }, p5/Z, [x10, #6, MUL VL]
    fmla z9.s, z7.s, z0.s[1]
    fmla z13.s, z7.s, z1.s[1]
    ld1w { z7.s }, p5/Z, [x10, #7, MUL VL]
    addvl x10, x10, #16
    fmla z10.s, z6.s, z0.s[1]
    fmla z14.s, z6.s, z1.s[1]
    fmla z11.s, z7.s, z0.s[1]
    fmla z15.s, z7.s, z1.s[1]
    ld1w { z6.s }, p5/Z, [x10, #-8, MUL VL]
    ld1w { z7.s }, p5/Z, [x10, #-7, MUL VL]
    fmla z8.s, z6.s, z0.s[2]
    fmla z12.s, z6.s, z1.s[2]
    ld1w { z6.s }, p5/Z, [x10, #-6, MUL VL]
    fmla z9.s, z7.s, z0.s[2]
    fmla z13.s, z7.s, z1.s[2]
    ld1w { z7.s }, p5/Z, [x10, #-5, MUL VL]
    fmla z10.s, z6.s, z0.s[2]
    fmla z14.s, z6.s, z1.s[2]
    ld1w { z6.s }, p5/Z, [x10, #-4, MUL VL]
    fmla z11.s, z7.s, z0.s[2]
    fmla z15.s, z7.s, z1.s[2]
    ld1w { z7.s }, p5/Z, [x10, #-3, MUL VL]
    fmla z8.s, z6.s, z0.s[3]
    fmla z12.s, z6.s, z1.s[3]
    ld1w { z6.s }, p5/Z, [x10, #-2, MUL VL]
    fmla z9.s, z7.s, z0.s[3]
    fmla z13.s, z7.s, z1.s[3]
    ld1w { z7.s }, p5/Z, [x10, #-1, MUL VL]
    fmla z10.s, z6.s, z0.s[3]
    fmla z14.s, z6.s, z1.s[3]
    fmla z11.s, z7.s, z0.s[3]
    fmla z15.s, z7.s, z1.s[3]
    bgt label_22
KAI_ASM_LABEL(label_23)  // Height 2: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x27
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x1
    ld1rqw { z0.s }, p0/Z, [x26]
    ld1rqw { z1.s }, p0/Z, [x25]
    fmla z8.s, z6.s, z0.s[0]
    fmla z12.s, z6.s, z1.s[0]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[0]
    fmla z13.s, z7.s, z1.s[0]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    fmla z10.s, z6.s, z0.s[0]
    fmla z14.s, z6.s, z1.s[0]
    fmla z11.s, z7.s, z0.s[0]
    fmla z15.s, z7.s, z1.s[0]
    ble label_24
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x1
    fmla z8.s, z6.s, z0.s[1]
    fmla z12.s, z6.s, z1.s[1]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[1]
    fmla z13.s, z7.s, z1.s[1]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    fmla z10.s, z6.s, z0.s[1]
    fmla z14.s, z6.s, z1.s[1]
    fmla z11.s, z7.s, z0.s[1]
    fmla z15.s, z7.s, z1.s[1]
    ble label_24
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x1
    fmla z8.s, z6.s, z0.s[2]
    fmla z12.s, z6.s, z1.s[2]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[2]
    fmla z13.s, z7.s, z1.s[2]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    fmla z10.s, z6.s, z0.s[2]
    fmla z14.s, z6.s, z1.s[2]
    fmla z11.s, z7.s, z0.s[2]
    fmla z15.s, z7.s, z1.s[2]
    ble label_24
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    fmla z8.s, z6.s, z0.s[3]
    fmla z12.s, z6.s, z1.s[3]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[3]
    fmla z13.s, z7.s, z1.s[3]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    fmla z10.s, z6.s, z0.s[3]
    fmla z14.s, z6.s, z1.s[3]
    fmla z11.s, z7.s, z0.s[3]
    fmla z15.s, z7.s, z1.s[3]
KAI_ASM_LABEL(label_24)  // Height 2: Multiply loop: multiply skip
    ldr w20, [x2, #0x8]
    add x28, x28, #0x1
    cmp x28, x20
    bne label_19
    ldr x20, [x2, #0x28]
    add x26, x9, x20, LSL #2
    tbz x3, #1, label_25
    add x21, x2, #0x0
    add x20, x2, #0x4
    KAI_ASM_INST(0x8540d6b1)  // ld1rw { z17.s }, p5/Z, [x21]
    KAI_ASM_INST(0x8540d690)  // ld1rw { z16.s }, p5/Z, [x20]
    fmin z8.s, p5/M, z8.s, z17.s
    fmin z9.s, p5/M, z9.s, z17.s
    fmin z10.s, p5/M, z10.s, z17.s
    fmin z11.s, p5/M, z11.s, z17.s
    fmin z12.s, p5/M, z12.s, z17.s
    fmin z13.s, p5/M, z13.s, z17.s
    fmin z14.s, p5/M, z14.s, z17.s
    fmin z15.s, p5/M, z15.s, z17.s
    fmax z8.s, p5/M, z8.s, z16.s
    fmax z9.s, p5/M, z9.s, z16.s
    fmax z10.s, p5/M, z10.s, z16.s
    fmax z11.s, p5/M, z11.s, z16.s
    fmax z12.s, p5/M, z12.s, z16.s
    fmax z13.s, p5/M, z13.s, z16.s
    fmax z14.s, p5/M, z14.s, z16.s
    fmax z15.s, p5/M, z15.s, z16.s
KAI_ASM_LABEL(label_25)  // Height 2: No activation
    st1w { z8.s }, p4, [x9]
    st1w { z9.s }, p3, [x9, #1, MUL VL]
    st1w { z10.s }, p2, [x9, #2, MUL VL]
    st1w { z11.s }, p1, [x9, #3, MUL VL]
    addvl x9, x9, #4
    st1w { z12.s }, p4, [x26]
    st1w { z13.s }, p3, [x26, #1, MUL VL]
    st1w { z14.s }, p2, [x26, #2, MUL VL]
    st1w { z15.s }, p1, [x26, #3, MUL VL]
    decw x11, ALL, MUL #4
    cmp x11, XZR
    bgt label_15
    b label_80
KAI_ASM_LABEL(label_27)  // Height 3
    ldr x11, [x2, #0x18]
    ldr x10, [x2, #0x20]
    ldr x9, [x2, #0x40]
KAI_ASM_LABEL(label_28)  // Height 3: Column loop
    mov x20, #0x0
    whilelt p4.s, x20, x11
    incw x20
    whilelt p3.s, x20, x11
    incw x20
    whilelt p2.s, x20, x11
    incw x20
    whilelt p1.s, x20, x11
    cbz x10, label_29
    ld1w { z8.s }, p5/Z, [x10]
    ld1w { z9.s }, p5/Z, [x10, #1, MUL VL]
    ld1w { z10.s }, p5/Z, [x10, #2, MUL VL]
    ld1w { z11.s }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    mov z12.d, z8.d
    mov z13.d, z9.d
    mov z16.d, z8.d
    mov z17.d, z9.d
    mov z14.d, z10.d
    mov z15.d, z11.d
    mov z18.d, z10.d
    mov z19.d, z11.d
    b label_31
KAI_ASM_LABEL(label_29)  // Height 3: no bias
    tbz x3, #0, label_30
    ldr x20, [x2, #0x28]
    ld1w { z8.s }, p4/Z, [x9]
    ld1w { z9.s }, p3/Z, [x9, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x9, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x9, #3, MUL VL]
    add x21, x9, x20, LSL #2
    add x20, x21, x20, LSL #2
    ld1w { z12.s }, p4/Z, [x21]
    ld1w { z13.s }, p3/Z, [x21, #1, MUL VL]
    ld1w { z14.s }, p2/Z, [x21, #2, MUL VL]
    ld1w { z15.s }, p1/Z, [x21, #3, MUL VL]
    ld1w { z16.s }, p4/Z, [x20]
    ld1w { z17.s }, p3/Z, [x20, #1, MUL VL]
    ld1w { z18.s }, p2/Z, [x20, #2, MUL VL]
    ld1w { z19.s }, p1/Z, [x20, #3, MUL VL]
    b label_31
KAI_ASM_LABEL(label_30)  // Height 3: no accumulate
    mov z8.b, #0x0
    mov z9.b, #0x0
    mov z10.b, #0x0
    mov z11.b, #0x0
    mov z12.b, #0x0
    mov z13.b, #0x0
    mov z14.b, #0x0
    mov z15.b, #0x0
    mov z16.b, #0x0
    mov z17.b, #0x0
    mov z18.b, #0x0
    mov z19.b, #0x0
KAI_ASM_LABEL(label_31)  // Height 3: setup done
    mov x28, #0x0
KAI_ASM_LABEL(label_32)  // Height 3: String loop
    ldr x20, [x2, #0x10]
    ldr x21, [x2, #0x38]
    ldr w27, [x20, x28, LSL #0x2]
    tbz x3, #3, label_33
    ldr x20, [x0, x28, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x26, [x20, #0x0]
    ldr x25, [x20, #0x8]
    ldr x24, [x20, #0x10]
    cbnz x28, label_34
    ldr x20, [x2, #0x30]
    add x26, x26, x20, LSL #2
    add x25, x25, x20, LSL #2
    add x24, x24, x20, LSL #2
    b label_34
KAI_ASM_LABEL(label_33)  // Height 3: setup direct input
    mov x26, x0
    add x25, x26, x21, LSL #2
    add x24, x25, x21, LSL #2
KAI_ASM_LABEL(label_34)  // Height 3: input setup done
    cmp x27, #0x4
    ble label_36
KAI_ASM_LABEL(label_35)  // Height 3: Multiply loop: Main loop head
    whilelt p0.s, XZR, x27
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    sub x27, x27, #0x4
    cmp x27, #0x4
    ld1rqw { z0.s }, p0/Z, [x26]
    add x26, x26, #0x10
    ld1rqw { z1.s }, p0/Z, [x25]
    add x25, x25, #0x10
    ld1rqw { z2.s }, p0/Z, [x24]
    add x24, x24, #0x10
    fmla z8.s, z6.s, z0.s[0]
    fmla z12.s, z6.s, z1.s[0]
    fmla z9.s, z7.s, z0.s[0]
    fmla z13.s, z7.s, z1.s[0]
    fmla z16.s, z6.s, z2.s[0]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z17.s, z7.s, z2.s[0]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    fmla z10.s, z6.s, z0.s[0]
    fmla z14.s, z6.s, z1.s[0]
    fmla z18.s, z6.s, z2.s[0]
    ld1w { z6.s }, p5/Z, [x10, #4, MUL VL]
    fmla z11.s, z7.s, z0.s[0]
    fmla z15.s, z7.s, z1.s[0]
    fmla z19.s, z7.s, z2.s[0]
    ld1w { z7.s }, p5/Z, [x10, #5, MUL VL]
    fmla z8.s, z6.s, z0.s[1]
    fmla z12.s, z6.s, z1.s[1]
    fmla z16.s, z6.s, z2.s[1]
    ld1w { z6.s }, p5/Z, [x10, #6, MUL VL]
    fmla z9.s, z7.s, z0.s[1]
    fmla z13.s, z7.s, z1.s[1]
    fmla z17.s, z7.s, z2.s[1]
    ld1w { z7.s }, p5/Z, [x10, #7, MUL VL]
    addvl x10, x10, #16
    fmla z10.s, z6.s, z0.s[1]
    fmla z14.s, z6.s, z1.s[1]
    fmla z18.s, z6.s, z2.s[1]
    fmla z11.s, z7.s, z0.s[1]
    fmla z15.s, z7.s, z1.s[1]
    fmla z19.s, z7.s, z2.s[1]
    ld1w { z6.s }, p5/Z, [x10, #-8, MUL VL]
    ld1w { z7.s }, p5/Z, [x10, #-7, MUL VL]
    fmla z8.s, z6.s, z0.s[2]
    fmla z12.s, z6.s, z1.s[2]
    fmla z16.s, z6.s, z2.s[2]
    ld1w { z6.s }, p5/Z, [x10, #-6, MUL VL]
    fmla z9.s, z7.s, z0.s[2]
    fmla z13.s, z7.s, z1.s[2]
    fmla z17.s, z7.s, z2.s[2]
    ld1w { z7.s }, p5/Z, [x10, #-5, MUL VL]
    fmla z10.s, z6.s, z0.s[2]
    fmla z14.s, z6.s, z1.s[2]
    fmla z18.s, z6.s, z2.s[2]
    ld1w { z6.s }, p5/Z, [x10, #-4, MUL VL]
    fmla z11.s, z7.s, z0.s[2]
    fmla z15.s, z7.s, z1.s[2]
    fmla z19.s, z7.s, z2.s[2]
    ld1w { z7.s }, p5/Z, [x10, #-3, MUL VL]
    fmla z8.s, z6.s, z0.s[3]
    fmla z12.s, z6.s, z1.s[3]
    fmla z16.s, z6.s, z2.s[3]
    ld1w { z6.s }, p5/Z, [x10, #-2, MUL VL]
    fmla z9.s, z7.s, z0.s[3]
    fmla z13.s, z7.s, z1.s[3]
    fmla z17.s, z7.s, z2.s[3]
    ld1w { z7.s }, p5/Z, [x10, #-1, MUL VL]
    fmla z10.s, z6.s, z0.s[3]
    fmla z14.s, z6.s, z1.s[3]
    fmla z18.s, z6.s, z2.s[3]
    fmla z11.s, z7.s, z0.s[3]
    fmla z15.s, z7.s, z1.s[3]
    fmla z19.s, z7.s, z2.s[3]
    bgt label_35
KAI_ASM_LABEL(label_36)  // Height 3: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x27
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x1
    ld1rqw { z0.s }, p0/Z, [x26]
    ld1rqw { z1.s }, p0/Z, [x25]
    ld1rqw { z2.s }, p0/Z, [x24]
    fmla z8.s, z6.s, z0.s[0]
    fmla z12.s, z6.s, z1.s[0]
    fmla z9.s, z7.s, z0.s[0]
    fmla z13.s, z7.s, z1.s[0]
    fmla z16.s, z6.s, z2.s[0]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z17.s, z7.s, z2.s[0]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    fmla z10.s, z6.s, z0.s[0]
    fmla z14.s, z6.s, z1.s[0]
    fmla z18.s, z6.s, z2.s[0]
    fmla z11.s, z7.s, z0.s[0]
    fmla z15.s, z7.s, z1.s[0]
    fmla z19.s, z7.s, z2.s[0]
    ble label_37
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x1
    fmla z8.s, z6.s, z0.s[1]
    fmla z12.s, z6.s, z1.s[1]
    fmla z16.s, z6.s, z2.s[1]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[1]
    fmla z13.s, z7.s, z1.s[1]
    fmla z17.s, z7.s, z2.s[1]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    fmla z10.s, z6.s, z0.s[1]
    fmla z14.s, z6.s, z1.s[1]
    fmla z18.s, z6.s, z2.s[1]
    fmla z11.s, z7.s, z0.s[1]
    fmla z15.s, z7.s, z1.s[1]
    fmla z19.s, z7.s, z2.s[1]
    ble label_37
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x1
    fmla z8.s, z6.s, z0.s[2]
    fmla z12.s, z6.s, z1.s[2]
    fmla z16.s, z6.s, z2.s[2]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[2]
    fmla z13.s, z7.s, z1.s[2]
    fmla z17.s, z7.s, z2.s[2]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    fmla z10.s, z6.s, z0.s[2]
    fmla z14.s, z6.s, z1.s[2]
    fmla z18.s, z6.s, z2.s[2]
    fmla z11.s, z7.s, z0.s[2]
    fmla z15.s, z7.s, z1.s[2]
    fmla z19.s, z7.s, z2.s[2]
    ble label_37
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    fmla z8.s, z6.s, z0.s[3]
    fmla z12.s, z6.s, z1.s[3]
    fmla z16.s, z6.s, z2.s[3]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[3]
    fmla z13.s, z7.s, z1.s[3]
    fmla z17.s, z7.s, z2.s[3]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    fmla z10.s, z6.s, z0.s[3]
    fmla z14.s, z6.s, z1.s[3]
    fmla z18.s, z6.s, z2.s[3]
    fmla z11.s, z7.s, z0.s[3]
    fmla z15.s, z7.s, z1.s[3]
    fmla z19.s, z7.s, z2.s[3]
KAI_ASM_LABEL(label_37)  // Height 3: Multiply loop: multiply skip
    ldr w20, [x2, #0x8]
    add x28, x28, #0x1
    cmp x28, x20
    bne label_32
    ldr x20, [x2, #0x28]
    add x26, x9, x20, LSL #2
    add x25, x26, x20, LSL #2
    tbz x3, #1, label_38
    add x21, x2, #0x0
    add x20, x2, #0x4
    KAI_ASM_INST(0x8540d6b5)  // ld1rw { z21.s }, p5/Z, [x21]
    KAI_ASM_INST(0x8540d694)  // ld1rw { z20.s }, p5/Z, [x20]
    fmin z8.s, p5/M, z8.s, z21.s
    fmin z9.s, p5/M, z9.s, z21.s
    fmin z10.s, p5/M, z10.s, z21.s
    fmin z11.s, p5/M, z11.s, z21.s
    fmin z12.s, p5/M, z12.s, z21.s
    fmin z13.s, p5/M, z13.s, z21.s
    fmin z14.s, p5/M, z14.s, z21.s
    fmin z15.s, p5/M, z15.s, z21.s
    fmin z16.s, p5/M, z16.s, z21.s
    fmin z17.s, p5/M, z17.s, z21.s
    fmin z18.s, p5/M, z18.s, z21.s
    fmin z19.s, p5/M, z19.s, z21.s
    fmax z8.s, p5/M, z8.s, z20.s
    fmax z9.s, p5/M, z9.s, z20.s
    fmax z10.s, p5/M, z10.s, z20.s
    fmax z11.s, p5/M, z11.s, z20.s
    fmax z12.s, p5/M, z12.s, z20.s
    fmax z13.s, p5/M, z13.s, z20.s
    fmax z14.s, p5/M, z14.s, z20.s
    fmax z15.s, p5/M, z15.s, z20.s
    fmax z16.s, p5/M, z16.s, z20.s
    fmax z17.s, p5/M, z17.s, z20.s
    fmax z18.s, p5/M, z18.s, z20.s
    fmax z19.s, p5/M, z19.s, z20.s
KAI_ASM_LABEL(label_38)  // Height 3: No activation
    st1w { z8.s }, p4, [x9]
    st1w { z9.s }, p3, [x9, #1, MUL VL]
    st1w { z10.s }, p2, [x9, #2, MUL VL]
    st1w { z11.s }, p1, [x9, #3, MUL VL]
    addvl x9, x9, #4
    st1w { z12.s }, p4, [x26]
    st1w { z13.s }, p3, [x26, #1, MUL VL]
    st1w { z14.s }, p2, [x26, #2, MUL VL]
    st1w { z15.s }, p1, [x26, #3, MUL VL]
    st1w { z16.s }, p4, [x25]
    st1w { z17.s }, p3, [x25, #1, MUL VL]
    st1w { z18.s }, p2, [x25, #2, MUL VL]
    st1w { z19.s }, p1, [x25, #3, MUL VL]
    decw x11, ALL, MUL #4
    cmp x11, XZR
    bgt label_28
    b label_80
KAI_ASM_LABEL(label_40)  // Height 4
    ldr x11, [x2, #0x18]
    ldr x10, [x2, #0x20]
    ldr x9, [x2, #0x40]
KAI_ASM_LABEL(label_41)  // Height 4: Column loop
    mov x20, #0x0
    whilelt p4.s, x20, x11
    incw x20
    whilelt p3.s, x20, x11
    incw x20
    whilelt p2.s, x20, x11
    incw x20
    whilelt p1.s, x20, x11
    cbz x10, label_42
    ld1w { z8.s }, p5/Z, [x10]
    ld1w { z9.s }, p5/Z, [x10, #1, MUL VL]
    ld1w { z10.s }, p5/Z, [x10, #2, MUL VL]
    ld1w { z11.s }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    mov z12.d, z8.d
    mov z13.d, z9.d
    mov z16.d, z8.d
    mov z17.d, z9.d
    mov z20.d, z8.d
    mov z14.d, z10.d
    mov z15.d, z11.d
    mov z18.d, z10.d
    mov z19.d, z11.d
    mov z21.d, z9.d
    mov z22.d, z10.d
    mov z23.d, z11.d
    b label_44
KAI_ASM_LABEL(label_42)  // Height 4: no bias
    tbz x3, #0, label_43
    ldr x20, [x2, #0x28]
    ld1w { z8.s }, p4/Z, [x9]
    ld1w { z9.s }, p3/Z, [x9, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x9, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x9, #3, MUL VL]
    add x22, x9, x20, LSL #2
    add x21, x22, x20, LSL #2
    add x20, x21, x20, LSL #2
    ld1w { z12.s }, p4/Z, [x22]
    ld1w { z13.s }, p3/Z, [x22, #1, MUL VL]
    ld1w { z14.s }, p2/Z, [x22, #2, MUL VL]
    ld1w { z15.s }, p1/Z, [x22, #3, MUL VL]
    ld1w { z16.s }, p4/Z, [x21]
    ld1w { z17.s }, p3/Z, [x21, #1, MUL VL]
    ld1w { z18.s }, p2/Z, [x21, #2, MUL VL]
    ld1w { z19.s }, p1/Z, [x21, #3, MUL VL]
    ld1w { z20.s }, p4/Z, [x20]
    ld1w { z21.s }, p3/Z, [x20, #1, MUL VL]
    ld1w { z22.s }, p2/Z, [x20, #2, MUL VL]
    ld1w { z23.s }, p1/Z, [x20, #3, MUL VL]
    b label_44
KAI_ASM_LABEL(label_43)  // Height 4: no accumulate
    mov z8.b, #0x0
    mov z9.b, #0x0
    mov z10.b, #0x0
    mov z11.b, #0x0
    mov z12.b, #0x0
    mov z13.b, #0x0
    mov z14.b, #0x0
    mov z15.b, #0x0
    mov z16.b, #0x0
    mov z17.b, #0x0
    mov z18.b, #0x0
    mov z19.b, #0x0
    mov z20.b, #0x0
    mov z21.b, #0x0
    mov z22.b, #0x0
    mov z23.b, #0x0
KAI_ASM_LABEL(label_44)  // Height 4: setup done
    mov x28, #0x0
KAI_ASM_LABEL(label_45)  // Height 4: String loop
    ldr x20, [x2, #0x10]
    ldr x21, [x2, #0x38]
    ldr w27, [x20, x28, LSL #0x2]
    tbz x3, #3, label_46
    ldr x20, [x0, x28, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x26, [x20, #0x0]
    ldr x25, [x20, #0x8]
    ldr x24, [x20, #0x10]
    ldr x23, [x20, #0x18]
    cbnz x28, label_47
    ldr x20, [x2, #0x30]
    add x26, x26, x20, LSL #2
    add x25, x25, x20, LSL #2
    add x24, x24, x20, LSL #2
    add x23, x23, x20, LSL #2
    b label_47
KAI_ASM_LABEL(label_46)  // Height 4: setup direct input
    mov x26, x0
    add x25, x26, x21, LSL #2
    add x24, x25, x21, LSL #2
    add x23, x24, x21, LSL #2
KAI_ASM_LABEL(label_47)  // Height 4: input setup done
    cmp x27, #0x4
    ble label_49
KAI_ASM_LABEL(label_48)  // Height 4: Multiply loop: Main loop head
    whilelt p0.s, XZR, x27
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    sub x27, x27, #0x4
    cmp x27, #0x4
    ld1rqw { z0.s }, p0/Z, [x26]
    add x26, x26, #0x10
    ld1rqw { z1.s }, p0/Z, [x25]
    add x25, x25, #0x10
    ld1rqw { z2.s }, p0/Z, [x24]
    add x24, x24, #0x10
    ld1rqw { z3.s }, p0/Z, [x23]
    add x23, x23, #0x10
    fmla z8.s, z6.s, z0.s[0]
    fmla z12.s, z6.s, z1.s[0]
    fmla z9.s, z7.s, z0.s[0]
    fmla z13.s, z7.s, z1.s[0]
    fmla z16.s, z6.s, z2.s[0]
    fmla z20.s, z6.s, z3.s[0]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z17.s, z7.s, z2.s[0]
    fmla z21.s, z7.s, z3.s[0]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    fmla z10.s, z6.s, z0.s[0]
    fmla z14.s, z6.s, z1.s[0]
    fmla z18.s, z6.s, z2.s[0]
    fmla z22.s, z6.s, z3.s[0]
    ld1w { z6.s }, p5/Z, [x10, #4, MUL VL]
    fmla z11.s, z7.s, z0.s[0]
    fmla z15.s, z7.s, z1.s[0]
    fmla z19.s, z7.s, z2.s[0]
    fmla z23.s, z7.s, z3.s[0]
    ld1w { z7.s }, p5/Z, [x10, #5, MUL VL]
    fmla z8.s, z6.s, z0.s[1]
    fmla z12.s, z6.s, z1.s[1]
    fmla z16.s, z6.s, z2.s[1]
    fmla z20.s, z6.s, z3.s[1]
    ld1w { z6.s }, p5/Z, [x10, #6, MUL VL]
    fmla z9.s, z7.s, z0.s[1]
    fmla z13.s, z7.s, z1.s[1]
    fmla z17.s, z7.s, z2.s[1]
    fmla z21.s, z7.s, z3.s[1]
    ld1w { z7.s }, p5/Z, [x10, #7, MUL VL]
    addvl x10, x10, #16
    fmla z10.s, z6.s, z0.s[1]
    fmla z14.s, z6.s, z1.s[1]
    fmla z18.s, z6.s, z2.s[1]
    fmla z22.s, z6.s, z3.s[1]
    fmla z11.s, z7.s, z0.s[1]
    fmla z15.s, z7.s, z1.s[1]
    ld1w { z6.s }, p5/Z, [x10, #-8, MUL VL]
    fmla z19.s, z7.s, z2.s[1]
    fmla z23.s, z7.s, z3.s[1]
    ld1w { z7.s }, p5/Z, [x10, #-7, MUL VL]
    fmla z8.s, z6.s, z0.s[2]
    fmla z12.s, z6.s, z1.s[2]
    fmla z16.s, z6.s, z2.s[2]
    fmla z20.s, z6.s, z3.s[2]
    ld1w { z6.s }, p5/Z, [x10, #-6, MUL VL]
    fmla z9.s, z7.s, z0.s[2]
    fmla z13.s, z7.s, z1.s[2]
    fmla z17.s, z7.s, z2.s[2]
    fmla z21.s, z7.s, z3.s[2]
    ld1w { z7.s }, p5/Z, [x10, #-5, MUL VL]
    fmla z10.s, z6.s, z0.s[2]
    fmla z14.s, z6.s, z1.s[2]
    fmla z18.s, z6.s, z2.s[2]
    fmla z22.s, z6.s, z3.s[2]
    ld1w { z6.s }, p5/Z, [x10, #-4, MUL VL]
    fmla z11.s, z7.s, z0.s[2]
    fmla z15.s, z7.s, z1.s[2]
    fmla z19.s, z7.s, z2.s[2]
    fmla z23.s, z7.s, z3.s[2]
    ld1w { z7.s }, p5/Z, [x10, #-3, MUL VL]
    fmla z8.s, z6.s, z0.s[3]
    fmla z12.s, z6.s, z1.s[3]
    fmla z16.s, z6.s, z2.s[3]
    fmla z20.s, z6.s, z3.s[3]
    ld1w { z6.s }, p5/Z, [x10, #-2, MUL VL]
    fmla z9.s, z7.s, z0.s[3]
    fmla z13.s, z7.s, z1.s[3]
    fmla z17.s, z7.s, z2.s[3]
    fmla z21.s, z7.s, z3.s[3]
    ld1w { z7.s }, p5/Z, [x10, #-1, MUL VL]
    fmla z10.s, z6.s, z0.s[3]
    fmla z14.s, z6.s, z1.s[3]
    fmla z18.s, z6.s, z2.s[3]
    fmla z22.s, z6.s, z3.s[3]
    fmla z11.s, z7.s, z0.s[3]
    fmla z15.s, z7.s, z1.s[3]
    fmla z19.s, z7.s, z2.s[3]
    fmla z23.s, z7.s, z3.s[3]
    bgt label_48
KAI_ASM_LABEL(label_49)  // Height 4: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x27
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x1
    ld1rqw { z0.s }, p0/Z, [x26]
    ld1rqw { z1.s }, p0/Z, [x25]
    ld1rqw { z2.s }, p0/Z, [x24]
    ld1rqw { z3.s }, p0/Z, [x23]
    fmla z8.s, z6.s, z0.s[0]
    fmla z12.s, z6.s, z1.s[0]
    fmla z9.s, z7.s, z0.s[0]
    fmla z13.s, z7.s, z1.s[0]
    fmla z16.s, z6.s, z2.s[0]
    fmla z20.s, z6.s, z3.s[0]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z17.s, z7.s, z2.s[0]
    fmla z21.s, z7.s, z3.s[0]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    fmla z10.s, z6.s, z0.s[0]
    fmla z14.s, z6.s, z1.s[0]
    fmla z18.s, z6.s, z2.s[0]
    fmla z22.s, z6.s, z3.s[0]
    fmla z11.s, z7.s, z0.s[0]
    fmla z15.s, z7.s, z1.s[0]
    fmla z19.s, z7.s, z2.s[0]
    fmla z23.s, z7.s, z3.s[0]
    ble label_50
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x1
    fmla z8.s, z6.s, z0.s[1]
    fmla z12.s, z6.s, z1.s[1]
    fmla z16.s, z6.s, z2.s[1]
    fmla z20.s, z6.s, z3.s[1]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[1]
    fmla z13.s, z7.s, z1.s[1]
    fmla z17.s, z7.s, z2.s[1]
    fmla z21.s, z7.s, z3.s[1]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    fmla z10.s, z6.s, z0.s[1]
    fmla z14.s, z6.s, z1.s[1]
    fmla z18.s, z6.s, z2.s[1]
    fmla z22.s, z6.s, z3.s[1]
    fmla z11.s, z7.s, z0.s[1]
    fmla z15.s, z7.s, z1.s[1]
    fmla z19.s, z7.s, z2.s[1]
    fmla z23.s, z7.s, z3.s[1]
    ble label_50
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x1
    fmla z8.s, z6.s, z0.s[2]
    fmla z12.s, z6.s, z1.s[2]
    fmla z16.s, z6.s, z2.s[2]
    fmla z20.s, z6.s, z3.s[2]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[2]
    fmla z13.s, z7.s, z1.s[2]
    fmla z17.s, z7.s, z2.s[2]
    fmla z21.s, z7.s, z3.s[2]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    fmla z10.s, z6.s, z0.s[2]
    fmla z14.s, z6.s, z1.s[2]
    fmla z18.s, z6.s, z2.s[2]
    fmla z22.s, z6.s, z3.s[2]
    fmla z11.s, z7.s, z0.s[2]
    fmla z15.s, z7.s, z1.s[2]
    fmla z19.s, z7.s, z2.s[2]
    fmla z23.s, z7.s, z3.s[2]
    ble label_50
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    fmla z8.s, z6.s, z0.s[3]
    fmla z12.s, z6.s, z1.s[3]
    fmla z16.s, z6.s, z2.s[3]
    fmla z20.s, z6.s, z3.s[3]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[3]
    fmla z13.s, z7.s, z1.s[3]
    fmla z17.s, z7.s, z2.s[3]
    fmla z21.s, z7.s, z3.s[3]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    fmla z10.s, z6.s, z0.s[3]
    fmla z14.s, z6.s, z1.s[3]
    fmla z18.s, z6.s, z2.s[3]
    fmla z22.s, z6.s, z3.s[3]
    fmla z11.s, z7.s, z0.s[3]
    fmla z15.s, z7.s, z1.s[3]
    fmla z19.s, z7.s, z2.s[3]
    fmla z23.s, z7.s, z3.s[3]
KAI_ASM_LABEL(label_50)  // Height 4: Multiply loop: multiply skip
    ldr w20, [x2, #0x8]
    add x28, x28, #0x1
    cmp x28, x20
    bne label_45
    ldr x20, [x2, #0x28]
    add x26, x9, x20, LSL #2
    add x25, x26, x20, LSL #2
    add x24, x25, x20, LSL #2
    tbz x3, #1, label_51
    add x21, x2, #0x0
    add x20, x2, #0x4
    KAI_ASM_INST(0x8540d6b9)  // ld1rw { z25.s }, p5/Z, [x21]
    KAI_ASM_INST(0x8540d698)  // ld1rw { z24.s }, p5/Z, [x20]
    fmin z8.s, p5/M, z8.s, z25.s
    fmin z9.s, p5/M, z9.s, z25.s
    fmin z10.s, p5/M, z10.s, z25.s
    fmin z11.s, p5/M, z11.s, z25.s
    fmin z12.s, p5/M, z12.s, z25.s
    fmin z13.s, p5/M, z13.s, z25.s
    fmin z14.s, p5/M, z14.s, z25.s
    fmin z15.s, p5/M, z15.s, z25.s
    fmin z16.s, p5/M, z16.s, z25.s
    fmin z17.s, p5/M, z17.s, z25.s
    fmin z18.s, p5/M, z18.s, z25.s
    fmin z19.s, p5/M, z19.s, z25.s
    fmin z20.s, p5/M, z20.s, z25.s
    fmin z21.s, p5/M, z21.s, z25.s
    fmin z22.s, p5/M, z22.s, z25.s
    fmin z23.s, p5/M, z23.s, z25.s
    fmax z8.s, p5/M, z8.s, z24.s
    fmax z9.s, p5/M, z9.s, z24.s
    fmax z10.s, p5/M, z10.s, z24.s
    fmax z11.s, p5/M, z11.s, z24.s
    fmax z12.s, p5/M, z12.s, z24.s
    fmax z13.s, p5/M, z13.s, z24.s
    fmax z14.s, p5/M, z14.s, z24.s
    fmax z15.s, p5/M, z15.s, z24.s
    fmax z16.s, p5/M, z16.s, z24.s
    fmax z17.s, p5/M, z17.s, z24.s
    fmax z18.s, p5/M, z18.s, z24.s
    fmax z19.s, p5/M, z19.s, z24.s
    fmax z20.s, p5/M, z20.s, z24.s
    fmax z21.s, p5/M, z21.s, z24.s
    fmax z22.s, p5/M, z22.s, z24.s
    fmax z23.s, p5/M, z23.s, z24.s
KAI_ASM_LABEL(label_51)  // Height 4: No activation
    st1w { z8.s }, p4, [x9]
    st1w { z9.s }, p3, [x9, #1, MUL VL]
    st1w { z10.s }, p2, [x9, #2, MUL VL]
    st1w { z11.s }, p1, [x9, #3, MUL VL]
    addvl x9, x9, #4
    st1w { z12.s }, p4, [x26]
    st1w { z13.s }, p3, [x26, #1, MUL VL]
    st1w { z14.s }, p2, [x26, #2, MUL VL]
    st1w { z15.s }, p1, [x26, #3, MUL VL]
    st1w { z16.s }, p4, [x25]
    st1w { z17.s }, p3, [x25, #1, MUL VL]
    st1w { z18.s }, p2, [x25, #2, MUL VL]
    st1w { z19.s }, p1, [x25, #3, MUL VL]
    st1w { z20.s }, p4, [x24]
    st1w { z21.s }, p3, [x24, #1, MUL VL]
    st1w { z22.s }, p2, [x24, #2, MUL VL]
    st1w { z23.s }, p1, [x24, #3, MUL VL]
    decw x11, ALL, MUL #4
    cmp x11, XZR
    bgt label_41
    b label_80
KAI_ASM_LABEL(label_53)  // Height 5
    ldr x11, [x2, #0x18]
    ldr x10, [x2, #0x20]
    ldr x9, [x2, #0x40]
KAI_ASM_LABEL(label_54)  // Height 5: Column loop
    mov x20, #0x0
    whilelt p4.s, x20, x11
    incw x20
    whilelt p3.s, x20, x11
    incw x20
    whilelt p2.s, x20, x11
    incw x20
    whilelt p1.s, x20, x11
    cbz x10, label_55
    ld1w { z8.s }, p5/Z, [x10]
    ld1w { z9.s }, p5/Z, [x10, #1, MUL VL]
    ld1w { z10.s }, p5/Z, [x10, #2, MUL VL]
    ld1w { z11.s }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    mov z12.d, z8.d
    mov z13.d, z9.d
    mov z16.d, z8.d
    mov z17.d, z9.d
    mov z20.d, z8.d
    mov z14.d, z10.d
    mov z15.d, z11.d
    mov z18.d, z10.d
    mov z19.d, z11.d
    mov z21.d, z9.d
    mov z22.d, z10.d
    mov z23.d, z11.d
    mov z24.d, z8.d
    mov z25.d, z9.d
    mov z26.d, z10.d
    mov z27.d, z11.d
    b label_57
KAI_ASM_LABEL(label_55)  // Height 5: no bias
    tbz x3, #0, label_56
    ldr x20, [x2, #0x28]
    ld1w { z8.s }, p4/Z, [x9]
    ld1w { z9.s }, p3/Z, [x9, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x9, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x9, #3, MUL VL]
    add x23, x9, x20, LSL #2
    add x22, x23, x20, LSL #2
    add x21, x22, x20, LSL #2
    ld1w { z12.s }, p4/Z, [x23]
    ld1w { z13.s }, p3/Z, [x23, #1, MUL VL]
    add x20, x21, x20, LSL #2
    ld1w { z14.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z15.s }, p1/Z, [x23, #3, MUL VL]
    ld1w { z16.s }, p4/Z, [x22]
    ld1w { z17.s }, p3/Z, [x22, #1, MUL VL]
    ld1w { z18.s }, p2/Z, [x22, #2, MUL VL]
    ld1w { z19.s }, p1/Z, [x22, #3, MUL VL]
    ld1w { z20.s }, p4/Z, [x21]
    ld1w { z21.s }, p3/Z, [x21, #1, MUL VL]
    ld1w { z22.s }, p2/Z, [x21, #2, MUL VL]
    ld1w { z23.s }, p1/Z, [x21, #3, MUL VL]
    ld1w { z24.s }, p4/Z, [x20]
    ld1w { z25.s }, p3/Z, [x20, #1, MUL VL]
    ld1w { z26.s }, p2/Z, [x20, #2, MUL VL]
    ld1w { z27.s }, p1/Z, [x20, #3, MUL VL]
    b label_57
KAI_ASM_LABEL(label_56)  // Height 5: no accumulate
    mov z8.b, #0x0
    mov z9.b, #0x0
    mov z10.b, #0x0
    mov z11.b, #0x0
    mov z12.b, #0x0
    mov z13.b, #0x0
    mov z14.b, #0x0
    mov z15.b, #0x0
    mov z16.b, #0x0
    mov z17.b, #0x0
    mov z18.b, #0x0
    mov z19.b, #0x0
    mov z20.b, #0x0
    mov z21.b, #0x0
    mov z22.b, #0x0
    mov z23.b, #0x0
    mov z24.b, #0x0
    mov z25.b, #0x0
    mov z26.b, #0x0
    mov z27.b, #0x0
KAI_ASM_LABEL(label_57)  // Height 5: setup done
    mov x28, #0x0
KAI_ASM_LABEL(label_58)  // Height 5: String loop
    ldr x20, [x2, #0x10]
    ldr x21, [x2, #0x38]
    ldr w27, [x20, x28, LSL #0x2]
    tbz x3, #3, label_59
    ldr x20, [x0, x28, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x26, [x20, #0x0]
    ldr x25, [x20, #0x8]
    ldr x24, [x20, #0x10]
    ldr x23, [x20, #0x18]
    ldr x22, [x20, #0x20]
    cbnz x28, label_60
    ldr x20, [x2, #0x30]
    add x26, x26, x20, LSL #2
    add x25, x25, x20, LSL #2
    add x24, x24, x20, LSL #2
    add x23, x23, x20, LSL #2
    add x22, x22, x20, LSL #2
    b label_60
KAI_ASM_LABEL(label_59)  // Height 5: setup direct input
    mov x26, x0
    add x25, x26, x21, LSL #2
    add x24, x25, x21, LSL #2
    add x23, x24, x21, LSL #2
    add x22, x23, x21, LSL #2
KAI_ASM_LABEL(label_60)  // Height 5: input setup done
    cmp x27, #0x4
    ble label_62
KAI_ASM_LABEL(label_61)  // Height 5: Multiply loop: Main loop head
    whilelt p0.s, XZR, x27
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    sub x27, x27, #0x4
    cmp x27, #0x4
    ld1rqw { z0.s }, p0/Z, [x26]
    add x26, x26, #0x10
    ld1rqw { z1.s }, p0/Z, [x25]
    add x25, x25, #0x10
    ld1rqw { z2.s }, p0/Z, [x24]
    add x24, x24, #0x10
    ld1rqw { z3.s }, p0/Z, [x23]
    ld1rqw { z4.s }, p0/Z, [x22]
    add x23, x23, #0x10
    add x22, x22, #0x10
    fmla z8.s, z6.s, z0.s[0]
    fmla z12.s, z6.s, z1.s[0]
    fmla z9.s, z7.s, z0.s[0]
    fmla z16.s, z6.s, z2.s[0]
    fmla z20.s, z6.s, z3.s[0]
    fmla z24.s, z6.s, z4.s[0]
    fmla z13.s, z7.s, z1.s[0]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z17.s, z7.s, z2.s[0]
    fmla z21.s, z7.s, z3.s[0]
    fmla z25.s, z7.s, z4.s[0]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    fmla z10.s, z6.s, z0.s[0]
    fmla z14.s, z6.s, z1.s[0]
    fmla z18.s, z6.s, z2.s[0]
    fmla z22.s, z6.s, z3.s[0]
    fmla z26.s, z6.s, z4.s[0]
    ld1w { z6.s }, p5/Z, [x10, #4, MUL VL]
    fmla z11.s, z7.s, z0.s[0]
    fmla z15.s, z7.s, z1.s[0]
    fmla z19.s, z7.s, z2.s[0]
    fmla z23.s, z7.s, z3.s[0]
    fmla z27.s, z7.s, z4.s[0]
    ld1w { z7.s }, p5/Z, [x10, #5, MUL VL]
    fmla z8.s, z6.s, z0.s[1]
    fmla z12.s, z6.s, z1.s[1]
    fmla z16.s, z6.s, z2.s[1]
    fmla z20.s, z6.s, z3.s[1]
    fmla z24.s, z6.s, z4.s[1]
    ld1w { z6.s }, p5/Z, [x10, #6, MUL VL]
    fmla z9.s, z7.s, z0.s[1]
    fmla z13.s, z7.s, z1.s[1]
    fmla z17.s, z7.s, z2.s[1]
    fmla z21.s, z7.s, z3.s[1]
    fmla z25.s, z7.s, z4.s[1]
    ld1w { z7.s }, p5/Z, [x10, #7, MUL VL]
    addvl x10, x10, #16
    fmla z10.s, z6.s, z0.s[1]
    fmla z14.s, z6.s, z1.s[1]
    fmla z18.s, z6.s, z2.s[1]
    fmla z22.s, z6.s, z3.s[1]
    fmla z26.s, z6.s, z4.s[1]
    fmla z11.s, z7.s, z0.s[1]
    fmla z15.s, z7.s, z1.s[1]
    ld1w { z6.s }, p5/Z, [x10, #-8, MUL VL]
    fmla z19.s, z7.s, z2.s[1]
    fmla z23.s, z7.s, z3.s[1]
    fmla z27.s, z7.s, z4.s[1]
    ld1w { z7.s }, p5/Z, [x10, #-7, MUL VL]
    fmla z8.s, z6.s, z0.s[2]
    fmla z12.s, z6.s, z1.s[2]
    fmla z16.s, z6.s, z2.s[2]
    fmla z20.s, z6.s, z3.s[2]
    fmla z24.s, z6.s, z4.s[2]
    ld1w { z6.s }, p5/Z, [x10, #-6, MUL VL]
    fmla z9.s, z7.s, z0.s[2]
    fmla z13.s, z7.s, z1.s[2]
    fmla z17.s, z7.s, z2.s[2]
    fmla z21.s, z7.s, z3.s[2]
    fmla z25.s, z7.s, z4.s[2]
    ld1w { z7.s }, p5/Z, [x10, #-5, MUL VL]
    fmla z10.s, z6.s, z0.s[2]
    fmla z14.s, z6.s, z1.s[2]
    fmla z18.s, z6.s, z2.s[2]
    fmla z22.s, z6.s, z3.s[2]
    fmla z26.s, z6.s, z4.s[2]
    ld1w { z6.s }, p5/Z, [x10, #-4, MUL VL]
    fmla z11.s, z7.s, z0.s[2]
    fmla z15.s, z7.s, z1.s[2]
    fmla z19.s, z7.s, z2.s[2]
    fmla z23.s, z7.s, z3.s[2]
    fmla z27.s, z7.s, z4.s[2]
    ld1w { z7.s }, p5/Z, [x10, #-3, MUL VL]
    fmla z8.s, z6.s, z0.s[3]
    fmla z12.s, z6.s, z1.s[3]
    fmla z16.s, z6.s, z2.s[3]
    fmla z20.s, z6.s, z3.s[3]
    fmla z24.s, z6.s, z4.s[3]
    ld1w { z6.s }, p5/Z, [x10, #-2, MUL VL]
    fmla z9.s, z7.s, z0.s[3]
    fmla z13.s, z7.s, z1.s[3]
    fmla z17.s, z7.s, z2.s[3]
    fmla z21.s, z7.s, z3.s[3]
    fmla z25.s, z7.s, z4.s[3]
    ld1w { z7.s }, p5/Z, [x10, #-1, MUL VL]
    fmla z10.s, z6.s, z0.s[3]
    fmla z14.s, z6.s, z1.s[3]
    fmla z18.s, z6.s, z2.s[3]
    fmla z22.s, z6.s, z3.s[3]
    fmla z26.s, z6.s, z4.s[3]
    fmla z11.s, z7.s, z0.s[3]
    fmla z15.s, z7.s, z1.s[3]
    fmla z19.s, z7.s, z2.s[3]
    fmla z23.s, z7.s, z3.s[3]
    fmla z27.s, z7.s, z4.s[3]
    bgt label_61
KAI_ASM_LABEL(label_62)  // Height 5: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x27
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x1
    ld1rqw { z0.s }, p0/Z, [x26]
    ld1rqw { z1.s }, p0/Z, [x25]
    ld1rqw { z2.s }, p0/Z, [x24]
    ld1rqw { z3.s }, p0/Z, [x23]
    ld1rqw { z4.s }, p0/Z, [x22]
    fmla z8.s, z6.s, z0.s[0]
    fmla z12.s, z6.s, z1.s[0]
    fmla z9.s, z7.s, z0.s[0]
    fmla z13.s, z7.s, z1.s[0]
    fmla z16.s, z6.s, z2.s[0]
    fmla z20.s, z6.s, z3.s[0]
    fmla z24.s, z6.s, z4.s[0]
    fmla z17.s, z7.s, z2.s[0]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z21.s, z7.s, z3.s[0]
    fmla z25.s, z7.s, z4.s[0]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    fmla z10.s, z6.s, z0.s[0]
    fmla z14.s, z6.s, z1.s[0]
    fmla z18.s, z6.s, z2.s[0]
    fmla z22.s, z6.s, z3.s[0]
    fmla z26.s, z6.s, z4.s[0]
    fmla z11.s, z7.s, z0.s[0]
    fmla z15.s, z7.s, z1.s[0]
    fmla z19.s, z7.s, z2.s[0]
    fmla z23.s, z7.s, z3.s[0]
    fmla z27.s, z7.s, z4.s[0]
    ble label_63
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x1
    fmla z8.s, z6.s, z0.s[1]
    fmla z12.s, z6.s, z1.s[1]
    fmla z16.s, z6.s, z2.s[1]
    fmla z20.s, z6.s, z3.s[1]
    fmla z24.s, z6.s, z4.s[1]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[1]
    fmla z13.s, z7.s, z1.s[1]
    fmla z17.s, z7.s, z2.s[1]
    fmla z21.s, z7.s, z3.s[1]
    fmla z25.s, z7.s, z4.s[1]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    fmla z10.s, z6.s, z0.s[1]
    addvl x10, x10, #4
    fmla z14.s, z6.s, z1.s[1]
    fmla z18.s, z6.s, z2.s[1]
    fmla z22.s, z6.s, z3.s[1]
    fmla z26.s, z6.s, z4.s[1]
    fmla z11.s, z7.s, z0.s[1]
    fmla z15.s, z7.s, z1.s[1]
    fmla z19.s, z7.s, z2.s[1]
    fmla z23.s, z7.s, z3.s[1]
    fmla z27.s, z7.s, z4.s[1]
    ble label_63
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x1
    fmla z8.s, z6.s, z0.s[2]
    fmla z12.s, z6.s, z1.s[2]
    fmla z16.s, z6.s, z2.s[2]
    fmla z20.s, z6.s, z3.s[2]
    fmla z24.s, z6.s, z4.s[2]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[2]
    fmla z13.s, z7.s, z1.s[2]
    fmla z17.s, z7.s, z2.s[2]
    fmla z21.s, z7.s, z3.s[2]
    fmla z25.s, z7.s, z4.s[2]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    fmla z10.s, z6.s, z0.s[2]
    addvl x10, x10, #4
    fmla z14.s, z6.s, z1.s[2]
    fmla z18.s, z6.s, z2.s[2]
    fmla z22.s, z6.s, z3.s[2]
    fmla z26.s, z6.s, z4.s[2]
    fmla z11.s, z7.s, z0.s[2]
    fmla z15.s, z7.s, z1.s[2]
    fmla z19.s, z7.s, z2.s[2]
    fmla z23.s, z7.s, z3.s[2]
    fmla z27.s, z7.s, z4.s[2]
    ble label_63
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    fmla z8.s, z6.s, z0.s[3]
    fmla z12.s, z6.s, z1.s[3]
    fmla z16.s, z6.s, z2.s[3]
    fmla z20.s, z6.s, z3.s[3]
    fmla z24.s, z6.s, z4.s[3]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[3]
    fmla z13.s, z7.s, z1.s[3]
    fmla z17.s, z7.s, z2.s[3]
    fmla z21.s, z7.s, z3.s[3]
    fmla z25.s, z7.s, z4.s[3]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    fmla z10.s, z6.s, z0.s[3]
    fmla z14.s, z6.s, z1.s[3]
    fmla z18.s, z6.s, z2.s[3]
    fmla z22.s, z6.s, z3.s[3]
    fmla z26.s, z6.s, z4.s[3]
    fmla z11.s, z7.s, z0.s[3]
    fmla z15.s, z7.s, z1.s[3]
    fmla z19.s, z7.s, z2.s[3]
    fmla z23.s, z7.s, z3.s[3]
    fmla z27.s, z7.s, z4.s[3]
KAI_ASM_LABEL(label_63)  // Height 5: Multiply loop: multiply skip
    ldr w20, [x2, #0x8]
    add x28, x28, #0x1
    cmp x28, x20
    bne label_58
    ldr x20, [x2, #0x28]
    add x26, x9, x20, LSL #2
    add x25, x26, x20, LSL #2
    add x24, x25, x20, LSL #2
    add x23, x24, x20, LSL #2
    tbz x3, #1, label_64
    add x21, x2, #0x0
    add x20, x2, #0x4
    KAI_ASM_INST(0x8540d6bd)  // ld1rw { z29.s }, p5/Z, [x21]
    KAI_ASM_INST(0x8540d69c)  // ld1rw { z28.s }, p5/Z, [x20]
    fmin z8.s, p5/M, z8.s, z29.s
    fmin z9.s, p5/M, z9.s, z29.s
    fmin z10.s, p5/M, z10.s, z29.s
    fmin z11.s, p5/M, z11.s, z29.s
    fmin z12.s, p5/M, z12.s, z29.s
    fmin z13.s, p5/M, z13.s, z29.s
    fmin z14.s, p5/M, z14.s, z29.s
    fmin z15.s, p5/M, z15.s, z29.s
    fmin z16.s, p5/M, z16.s, z29.s
    fmin z17.s, p5/M, z17.s, z29.s
    fmin z18.s, p5/M, z18.s, z29.s
    fmin z19.s, p5/M, z19.s, z29.s
    fmin z20.s, p5/M, z20.s, z29.s
    fmin z21.s, p5/M, z21.s, z29.s
    fmin z22.s, p5/M, z22.s, z29.s
    fmin z23.s, p5/M, z23.s, z29.s
    fmin z24.s, p5/M, z24.s, z29.s
    fmin z25.s, p5/M, z25.s, z29.s
    fmin z26.s, p5/M, z26.s, z29.s
    fmin z27.s, p5/M, z27.s, z29.s
    fmax z8.s, p5/M, z8.s, z28.s
    fmax z9.s, p5/M, z9.s, z28.s
    fmax z10.s, p5/M, z10.s, z28.s
    fmax z11.s, p5/M, z11.s, z28.s
    fmax z12.s, p5/M, z12.s, z28.s
    fmax z13.s, p5/M, z13.s, z28.s
    fmax z14.s, p5/M, z14.s, z28.s
    fmax z15.s, p5/M, z15.s, z28.s
    fmax z16.s, p5/M, z16.s, z28.s
    fmax z17.s, p5/M, z17.s, z28.s
    fmax z18.s, p5/M, z18.s, z28.s
    fmax z19.s, p5/M, z19.s, z28.s
    fmax z20.s, p5/M, z20.s, z28.s
    fmax z21.s, p5/M, z21.s, z28.s
    fmax z22.s, p5/M, z22.s, z28.s
    fmax z23.s, p5/M, z23.s, z28.s
    fmax z24.s, p5/M, z24.s, z28.s
    fmax z25.s, p5/M, z25.s, z28.s
    fmax z26.s, p5/M, z26.s, z28.s
    fmax z27.s, p5/M, z27.s, z28.s
KAI_ASM_LABEL(label_64)  // Height 5: No activation
    st1w { z8.s }, p4, [x9]
    st1w { z9.s }, p3, [x9, #1, MUL VL]
    st1w { z10.s }, p2, [x9, #2, MUL VL]
    st1w { z11.s }, p1, [x9, #3, MUL VL]
    addvl x9, x9, #4
    st1w { z12.s }, p4, [x26]
    st1w { z13.s }, p3, [x26, #1, MUL VL]
    st1w { z14.s }, p2, [x26, #2, MUL VL]
    st1w { z15.s }, p1, [x26, #3, MUL VL]
    st1w { z16.s }, p4, [x25]
    st1w { z17.s }, p3, [x25, #1, MUL VL]
    st1w { z18.s }, p2, [x25, #2, MUL VL]
    st1w { z19.s }, p1, [x25, #3, MUL VL]
    st1w { z20.s }, p4, [x24]
    st1w { z21.s }, p3, [x24, #1, MUL VL]
    st1w { z22.s }, p2, [x24, #2, MUL VL]
    st1w { z23.s }, p1, [x24, #3, MUL VL]
    st1w { z24.s }, p4, [x23]
    st1w { z25.s }, p3, [x23, #1, MUL VL]
    st1w { z26.s }, p2, [x23, #2, MUL VL]
    st1w { z27.s }, p1, [x23, #3, MUL VL]
    decw x11, ALL, MUL #4
    cmp x11, XZR
    bgt label_54
    b label_80
KAI_ASM_LABEL(label_66)  // Height 6
    ldr x21, [x2, #0x28]
    ldr x9, [x2, #0x40]
    mov x20, #0x18
    ldr x11, [x2, #0x18]
    ldr x10, [x2, #0x20]
    madd x20, x21, x20, x9
    str x20, [x2, #0x40]
KAI_ASM_LABEL(label_67)  // Height 6: Column loop
    mov x20, #0x0
    whilelt p4.s, x20, x11
    incw x20
    whilelt p3.s, x20, x11
    incw x20
    whilelt p2.s, x20, x11
    incw x20
    whilelt p1.s, x20, x11
    cbz x10, label_68
    ld1w { z8.s }, p5/Z, [x10]
    ld1w { z9.s }, p5/Z, [x10, #1, MUL VL]
    ld1w { z10.s }, p5/Z, [x10, #2, MUL VL]
    ld1w { z11.s }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    mov z12.d, z8.d
    mov z13.d, z9.d
    mov z16.d, z8.d
    mov z17.d, z9.d
    mov z20.d, z8.d
    mov z14.d, z10.d
    mov z15.d, z11.d
    mov z18.d, z10.d
    mov z19.d, z11.d
    mov z21.d, z9.d
    mov z22.d, z10.d
    mov z23.d, z11.d
    mov z24.d, z8.d
    mov z25.d, z9.d
    mov z26.d, z10.d
    mov z27.d, z11.d
    mov z28.d, z8.d
    mov z29.d, z9.d
    mov z30.d, z10.d
    mov z31.d, z11.d
    b label_70
KAI_ASM_LABEL(label_68)  // Height 6: no bias
    tbz x3, #0, label_69
    ldr x20, [x2, #0x28]
    ld1w { z8.s }, p4/Z, [x9]
    ld1w { z9.s }, p3/Z, [x9, #1, MUL VL]
    ld1w { z10.s }, p2/Z, [x9, #2, MUL VL]
    ld1w { z11.s }, p1/Z, [x9, #3, MUL VL]
    add x24, x9, x20, LSL #2
    add x23, x24, x20, LSL #2
    add x22, x23, x20, LSL #2
    ld1w { z12.s }, p4/Z, [x24]
    ld1w { z13.s }, p3/Z, [x24, #1, MUL VL]
    add x21, x22, x20, LSL #2
    add x20, x21, x20, LSL #2
    ld1w { z14.s }, p2/Z, [x24, #2, MUL VL]
    ld1w { z15.s }, p1/Z, [x24, #3, MUL VL]
    ld1w { z16.s }, p4/Z, [x23]
    ld1w { z17.s }, p3/Z, [x23, #1, MUL VL]
    ld1w { z18.s }, p2/Z, [x23, #2, MUL VL]
    ld1w { z19.s }, p1/Z, [x23, #3, MUL VL]
    ld1w { z20.s }, p4/Z, [x22]
    ld1w { z21.s }, p3/Z, [x22, #1, MUL VL]
    ld1w { z22.s }, p2/Z, [x22, #2, MUL VL]
    ld1w { z23.s }, p1/Z, [x22, #3, MUL VL]
    ld1w { z24.s }, p4/Z, [x21]
    ld1w { z25.s }, p3/Z, [x21, #1, MUL VL]
    ld1w { z26.s }, p2/Z, [x21, #2, MUL VL]
    ld1w { z27.s }, p1/Z, [x21, #3, MUL VL]
    ld1w { z28.s }, p4/Z, [x20]
    ld1w { z29.s }, p3/Z, [x20, #1, MUL VL]
    ld1w { z30.s }, p2/Z, [x20, #2, MUL VL]
    ld1w { z31.s }, p1/Z, [x20, #3, MUL VL]
    b label_70
KAI_ASM_LABEL(label_69)  // Height 6: no accumulate
    mov z8.b, #0x0
    mov z9.b, #0x0
    mov z10.b, #0x0
    mov z11.b, #0x0
    mov z12.b, #0x0
    mov z13.b, #0x0
    mov z14.b, #0x0
    mov z15.b, #0x0
    mov z16.b, #0x0
    mov z17.b, #0x0
    mov z18.b, #0x0
    mov z19.b, #0x0
    mov z20.b, #0x0
    mov z21.b, #0x0
    mov z22.b, #0x0
    mov z23.b, #0x0
    mov z24.b, #0x0
    mov z25.b, #0x0
    mov z26.b, #0x0
    mov z27.b, #0x0
    mov z28.b, #0x0
    mov z29.b, #0x0
    mov z30.b, #0x0
    mov z31.b, #0x0
KAI_ASM_LABEL(label_70)  // Height 6: setup done
    mov x28, #0x0
KAI_ASM_LABEL(label_71)  // Height 6: String loop
    ldr x20, [x2, #0x10]
    ldr x21, [x2, #0x38]
    ldr w27, [x20, x28, LSL #0x2]
    tbz x3, #3, label_72
    ldr x20, [x0, x28, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x26, [x20, #0x0]
    ldr x25, [x20, #0x8]
    ldr x24, [x20, #0x10]
    ldr x23, [x20, #0x18]
    ldr x22, [x20, #0x20]
    ldr x21, [x20, #0x28]
    cbnz x28, label_73
    ldr x20, [x2, #0x30]
    add x26, x26, x20, LSL #2
    add x25, x25, x20, LSL #2
    add x24, x24, x20, LSL #2
    add x23, x23, x20, LSL #2
    add x22, x22, x20, LSL #2
    add x21, x21, x20, LSL #2
    b label_73
KAI_ASM_LABEL(label_72)  // Height 6: setup direct input
    mov x26, x0
    add x25, x26, x21, LSL #2
    add x24, x25, x21, LSL #2
    add x23, x24, x21, LSL #2
    add x22, x23, x21, LSL #2
    add x21, x22, x21, LSL #2
KAI_ASM_LABEL(label_73)  // Height 6: input setup done
    cmp x27, #0x4
    ble label_75
KAI_ASM_LABEL(label_74)  // Height 6: Multiply loop: Main loop head
    whilelt p0.s, XZR, x27
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    sub x27, x27, #0x4
    cmp x27, #0x4
    ld1rqw { z0.s }, p0/Z, [x26]
    add x26, x26, #0x10
    ld1rqw { z1.s }, p0/Z, [x25]
    add x25, x25, #0x10
    ld1rqw { z2.s }, p0/Z, [x24]
    add x24, x24, #0x10
    ld1rqw { z3.s }, p0/Z, [x23]
    ld1rqw { z4.s }, p0/Z, [x22]
    ld1rqw { z5.s }, p0/Z, [x21]
    add x23, x23, #0x10
    add x22, x22, #0x10
    fmla z8.s, z6.s, z0.s[0]
    fmla z12.s, z6.s, z1.s[0]
    add x21, x21, #0x10
    fmla z16.s, z6.s, z2.s[0]
    fmla z20.s, z6.s, z3.s[0]
    fmla z24.s, z6.s, z4.s[0]
    fmla z28.s, z6.s, z5.s[0]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[0]
    fmla z13.s, z7.s, z1.s[0]
    fmla z17.s, z7.s, z2.s[0]
    fmla z21.s, z7.s, z3.s[0]
    fmla z25.s, z7.s, z4.s[0]
    fmla z29.s, z7.s, z5.s[0]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    fmla z10.s, z6.s, z0.s[0]
    fmla z14.s, z6.s, z1.s[0]
    fmla z18.s, z6.s, z2.s[0]
    fmla z22.s, z6.s, z3.s[0]
    fmla z26.s, z6.s, z4.s[0]
    fmla z30.s, z6.s, z5.s[0]
    ld1w { z6.s }, p5/Z, [x10, #4, MUL VL]
    fmla z11.s, z7.s, z0.s[0]
    fmla z15.s, z7.s, z1.s[0]
    fmla z19.s, z7.s, z2.s[0]
    fmla z23.s, z7.s, z3.s[0]
    fmla z27.s, z7.s, z4.s[0]
    fmla z31.s, z7.s, z5.s[0]
    ld1w { z7.s }, p5/Z, [x10, #5, MUL VL]
    fmla z8.s, z6.s, z0.s[1]
    fmla z12.s, z6.s, z1.s[1]
    fmla z16.s, z6.s, z2.s[1]
    fmla z20.s, z6.s, z3.s[1]
    fmla z24.s, z6.s, z4.s[1]
    fmla z28.s, z6.s, z5.s[1]
    ld1w { z6.s }, p5/Z, [x10, #6, MUL VL]
    fmla z9.s, z7.s, z0.s[1]
    fmla z13.s, z7.s, z1.s[1]
    fmla z17.s, z7.s, z2.s[1]
    fmla z21.s, z7.s, z3.s[1]
    fmla z25.s, z7.s, z4.s[1]
    fmla z29.s, z7.s, z5.s[1]
    ld1w { z7.s }, p5/Z, [x10, #7, MUL VL]
    addvl x10, x10, #16
    fmla z10.s, z6.s, z0.s[1]
    fmla z14.s, z6.s, z1.s[1]
    fmla z18.s, z6.s, z2.s[1]
    fmla z22.s, z6.s, z3.s[1]
    fmla z26.s, z6.s, z4.s[1]
    fmla z30.s, z6.s, z5.s[1]
    ld1w { z6.s }, p5/Z, [x10, #-8, MUL VL]
    fmla z11.s, z7.s, z0.s[1]
    fmla z15.s, z7.s, z1.s[1]
    fmla z19.s, z7.s, z2.s[1]
    fmla z23.s, z7.s, z3.s[1]
    fmla z27.s, z7.s, z4.s[1]
    fmla z31.s, z7.s, z5.s[1]
    ld1w { z7.s }, p5/Z, [x10, #-7, MUL VL]
    fmla z8.s, z6.s, z0.s[2]
    fmla z12.s, z6.s, z1.s[2]
    fmla z16.s, z6.s, z2.s[2]
    fmla z20.s, z6.s, z3.s[2]
    fmla z24.s, z6.s, z4.s[2]
    fmla z28.s, z6.s, z5.s[2]
    ld1w { z6.s }, p5/Z, [x10, #-6, MUL VL]
    fmla z9.s, z7.s, z0.s[2]
    fmla z13.s, z7.s, z1.s[2]
    fmla z17.s, z7.s, z2.s[2]
    fmla z21.s, z7.s, z3.s[2]
    fmla z25.s, z7.s, z4.s[2]
    fmla z29.s, z7.s, z5.s[2]
    ld1w { z7.s }, p5/Z, [x10, #-5, MUL VL]
    fmla z10.s, z6.s, z0.s[2]
    fmla z14.s, z6.s, z1.s[2]
    fmla z18.s, z6.s, z2.s[2]
    fmla z22.s, z6.s, z3.s[2]
    fmla z26.s, z6.s, z4.s[2]
    fmla z30.s, z6.s, z5.s[2]
    ld1w { z6.s }, p5/Z, [x10, #-4, MUL VL]
    fmla z11.s, z7.s, z0.s[2]
    fmla z15.s, z7.s, z1.s[2]
    fmla z19.s, z7.s, z2.s[2]
    fmla z23.s, z7.s, z3.s[2]
    fmla z27.s, z7.s, z4.s[2]
    fmla z31.s, z7.s, z5.s[2]
    ld1w { z7.s }, p5/Z, [x10, #-3, MUL VL]
    fmla z8.s, z6.s, z0.s[3]
    fmla z12.s, z6.s, z1.s[3]
    fmla z16.s, z6.s, z2.s[3]
    fmla z20.s, z6.s, z3.s[3]
    fmla z24.s, z6.s, z4.s[3]
    fmla z28.s, z6.s, z5.s[3]
    ld1w { z6.s }, p5/Z, [x10, #-2, MUL VL]
    fmla z9.s, z7.s, z0.s[3]
    fmla z13.s, z7.s, z1.s[3]
    fmla z17.s, z7.s, z2.s[3]
    fmla z21.s, z7.s, z3.s[3]
    fmla z25.s, z7.s, z4.s[3]
    fmla z29.s, z7.s, z5.s[3]
    ld1w { z7.s }, p5/Z, [x10, #-1, MUL VL]
    fmla z10.s, z6.s, z0.s[3]
    fmla z14.s, z6.s, z1.s[3]
    fmla z18.s, z6.s, z2.s[3]
    fmla z22.s, z6.s, z3.s[3]
    fmla z26.s, z6.s, z4.s[3]
    fmla z30.s, z6.s, z5.s[3]
    fmla z11.s, z7.s, z0.s[3]
    fmla z15.s, z7.s, z1.s[3]
    fmla z19.s, z7.s, z2.s[3]
    fmla z23.s, z7.s, z3.s[3]
    fmla z27.s, z7.s, z4.s[3]
    fmla z31.s, z7.s, z5.s[3]
    bgt label_74
KAI_ASM_LABEL(label_75)  // Height 6: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x27
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x1
    ld1rqw { z0.s }, p0/Z, [x26]
    ld1rqw { z1.s }, p0/Z, [x25]
    ld1rqw { z2.s }, p0/Z, [x24]
    ld1rqw { z3.s }, p0/Z, [x23]
    ld1rqw { z4.s }, p0/Z, [x22]
    ld1rqw { z5.s }, p0/Z, [x21]
    fmla z8.s, z6.s, z0.s[0]
    fmla z12.s, z6.s, z1.s[0]
    fmla z9.s, z7.s, z0.s[0]
    fmla z13.s, z7.s, z1.s[0]
    fmla z16.s, z6.s, z2.s[0]
    fmla z20.s, z6.s, z3.s[0]
    fmla z24.s, z6.s, z4.s[0]
    fmla z28.s, z6.s, z5.s[0]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z17.s, z7.s, z2.s[0]
    fmla z21.s, z7.s, z3.s[0]
    fmla z25.s, z7.s, z4.s[0]
    fmla z29.s, z7.s, z5.s[0]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    fmla z10.s, z6.s, z0.s[0]
    fmla z14.s, z6.s, z1.s[0]
    fmla z18.s, z6.s, z2.s[0]
    fmla z22.s, z6.s, z3.s[0]
    fmla z26.s, z6.s, z4.s[0]
    fmla z30.s, z6.s, z5.s[0]
    fmla z11.s, z7.s, z0.s[0]
    fmla z15.s, z7.s, z1.s[0]
    fmla z19.s, z7.s, z2.s[0]
    fmla z23.s, z7.s, z3.s[0]
    fmla z27.s, z7.s, z4.s[0]
    fmla z31.s, z7.s, z5.s[0]
    ble label_76
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x1
    fmla z8.s, z6.s, z0.s[1]
    fmla z12.s, z6.s, z1.s[1]
    fmla z16.s, z6.s, z2.s[1]
    fmla z20.s, z6.s, z3.s[1]
    fmla z24.s, z6.s, z4.s[1]
    fmla z28.s, z6.s, z5.s[1]
    fmla z9.s, z7.s, z0.s[1]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z13.s, z7.s, z1.s[1]
    fmla z17.s, z7.s, z2.s[1]
    fmla z21.s, z7.s, z3.s[1]
    fmla z25.s, z7.s, z4.s[1]
    fmla z29.s, z7.s, z5.s[1]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    fmla z10.s, z6.s, z0.s[1]
    addvl x10, x10, #4
    fmla z14.s, z6.s, z1.s[1]
    fmla z18.s, z6.s, z2.s[1]
    fmla z22.s, z6.s, z3.s[1]
    fmla z26.s, z6.s, z4.s[1]
    fmla z30.s, z6.s, z5.s[1]
    fmla z11.s, z7.s, z0.s[1]
    fmla z15.s, z7.s, z1.s[1]
    fmla z19.s, z7.s, z2.s[1]
    fmla z23.s, z7.s, z3.s[1]
    fmla z27.s, z7.s, z4.s[1]
    fmla z31.s, z7.s, z5.s[1]
    ble label_76
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x1
    fmla z8.s, z6.s, z0.s[2]
    fmla z12.s, z6.s, z1.s[2]
    fmla z16.s, z6.s, z2.s[2]
    fmla z20.s, z6.s, z3.s[2]
    fmla z24.s, z6.s, z4.s[2]
    fmla z28.s, z6.s, z5.s[2]
    fmla z9.s, z7.s, z0.s[2]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z13.s, z7.s, z1.s[2]
    fmla z17.s, z7.s, z2.s[2]
    fmla z21.s, z7.s, z3.s[2]
    fmla z25.s, z7.s, z4.s[2]
    fmla z29.s, z7.s, z5.s[2]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    fmla z10.s, z6.s, z0.s[2]
    addvl x10, x10, #4
    fmla z14.s, z6.s, z1.s[2]
    fmla z18.s, z6.s, z2.s[2]
    fmla z22.s, z6.s, z3.s[2]
    fmla z26.s, z6.s, z4.s[2]
    fmla z30.s, z6.s, z5.s[2]
    fmla z11.s, z7.s, z0.s[2]
    fmla z15.s, z7.s, z1.s[2]
    fmla z19.s, z7.s, z2.s[2]
    fmla z23.s, z7.s, z3.s[2]
    fmla z27.s, z7.s, z4.s[2]
    fmla z31.s, z7.s, z5.s[2]
    ble label_76
    ld1w { z6.s }, p5/Z, [x10]
    ld1w { z7.s }, p5/Z, [x10, #1, MUL VL]
    fmla z8.s, z6.s, z0.s[3]
    fmla z12.s, z6.s, z1.s[3]
    fmla z16.s, z6.s, z2.s[3]
    fmla z20.s, z6.s, z3.s[3]
    fmla z24.s, z6.s, z4.s[3]
    fmla z28.s, z6.s, z5.s[3]
    ld1w { z6.s }, p5/Z, [x10, #2, MUL VL]
    fmla z9.s, z7.s, z0.s[3]
    fmla z13.s, z7.s, z1.s[3]
    fmla z17.s, z7.s, z2.s[3]
    fmla z21.s, z7.s, z3.s[3]
    fmla z25.s, z7.s, z4.s[3]
    fmla z29.s, z7.s, z5.s[3]
    ld1w { z7.s }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    fmla z10.s, z6.s, z0.s[3]
    fmla z14.s, z6.s, z1.s[3]
    fmla z18.s, z6.s, z2.s[3]
    fmla z22.s, z6.s, z3.s[3]
    fmla z26.s, z6.s, z4.s[3]
    fmla z30.s, z6.s, z5.s[3]
    fmla z11.s, z7.s, z0.s[3]
    fmla z15.s, z7.s, z1.s[3]
    fmla z19.s, z7.s, z2.s[3]
    fmla z23.s, z7.s, z3.s[3]
    fmla z27.s, z7.s, z4.s[3]
    fmla z31.s, z7.s, z5.s[3]
KAI_ASM_LABEL(label_76)  // Height 6: Multiply loop: multiply skip
    ldr w20, [x2, #0x8]
    add x28, x28, #0x1
    cmp x28, x20
    bne label_71
    ldr x20, [x2, #0x28]
    add x26, x9, x20, LSL #2
    add x25, x26, x20, LSL #2
    add x24, x25, x20, LSL #2
    add x23, x24, x20, LSL #2
    add x22, x23, x20, LSL #2
    tbz x3, #1, label_77
    add x21, x2, #0x0
    add x20, x2, #0x4
    KAI_ASM_INST(0x8540d6a1)  // ld1rw { z1.s }, p5/Z, [x21]
    KAI_ASM_INST(0x8540d680)  // ld1rw { z0.s }, p5/Z, [x20]
    fmin z8.s, p5/M, z8.s, z1.s
    fmin z9.s, p5/M, z9.s, z1.s
    fmin z10.s, p5/M, z10.s, z1.s
    fmin z11.s, p5/M, z11.s, z1.s
    fmin z12.s, p5/M, z12.s, z1.s
    fmin z13.s, p5/M, z13.s, z1.s
    fmin z14.s, p5/M, z14.s, z1.s
    fmin z15.s, p5/M, z15.s, z1.s
    fmin z16.s, p5/M, z16.s, z1.s
    fmin z17.s, p5/M, z17.s, z1.s
    fmin z18.s, p5/M, z18.s, z1.s
    fmin z19.s, p5/M, z19.s, z1.s
    fmin z20.s, p5/M, z20.s, z1.s
    fmin z21.s, p5/M, z21.s, z1.s
    fmin z22.s, p5/M, z22.s, z1.s
    fmin z23.s, p5/M, z23.s, z1.s
    fmin z24.s, p5/M, z24.s, z1.s
    fmin z25.s, p5/M, z25.s, z1.s
    fmin z26.s, p5/M, z26.s, z1.s
    fmin z27.s, p5/M, z27.s, z1.s
    fmin z28.s, p5/M, z28.s, z1.s
    fmin z29.s, p5/M, z29.s, z1.s
    fmin z30.s, p5/M, z30.s, z1.s
    fmin z31.s, p5/M, z31.s, z1.s
    fmax z8.s, p5/M, z8.s, z0.s
    fmax z9.s, p5/M, z9.s, z0.s
    fmax z10.s, p5/M, z10.s, z0.s
    fmax z11.s, p5/M, z11.s, z0.s
    fmax z12.s, p5/M, z12.s, z0.s
    fmax z13.s, p5/M, z13.s, z0.s
    fmax z14.s, p5/M, z14.s, z0.s
    fmax z15.s, p5/M, z15.s, z0.s
    fmax z16.s, p5/M, z16.s, z0.s
    fmax z17.s, p5/M, z17.s, z0.s
    fmax z18.s, p5/M, z18.s, z0.s
    fmax z19.s, p5/M, z19.s, z0.s
    fmax z20.s, p5/M, z20.s, z0.s
    fmax z21.s, p5/M, z21.s, z0.s
    fmax z22.s, p5/M, z22.s, z0.s
    fmax z23.s, p5/M, z23.s, z0.s
    fmax z24.s, p5/M, z24.s, z0.s
    fmax z25.s, p5/M, z25.s, z0.s
    fmax z26.s, p5/M, z26.s, z0.s
    fmax z27.s, p5/M, z27.s, z0.s
    fmax z28.s, p5/M, z28.s, z0.s
    fmax z29.s, p5/M, z29.s, z0.s
    fmax z30.s, p5/M, z30.s, z0.s
    fmax z31.s, p5/M, z31.s, z0.s
KAI_ASM_LABEL(label_77)  // Height 6: No activation
    st1w { z8.s }, p4, [x9]
    st1w { z9.s }, p3, [x9, #1, MUL VL]
    st1w { z10.s }, p2, [x9, #2, MUL VL]
    st1w { z11.s }, p1, [x9, #3, MUL VL]
    addvl x9, x9, #4
    st1w { z12.s }, p4, [x26]
    st1w { z13.s }, p3, [x26, #1, MUL VL]
    st1w { z14.s }, p2, [x26, #2, MUL VL]
    st1w { z15.s }, p1, [x26, #3, MUL VL]
    st1w { z16.s }, p4, [x25]
    st1w { z17.s }, p3, [x25, #1, MUL VL]
    st1w { z18.s }, p2, [x25, #2, MUL VL]
    st1w { z19.s }, p1, [x25, #3, MUL VL]
    st1w { z20.s }, p4, [x24]
    st1w { z21.s }, p3, [x24, #1, MUL VL]
    st1w { z22.s }, p2, [x24, #2, MUL VL]
    st1w { z23.s }, p1, [x24, #3, MUL VL]
    st1w { z24.s }, p4, [x23]
    st1w { z25.s }, p3, [x23, #1, MUL VL]
    st1w { z26.s }, p2, [x23, #2, MUL VL]
    st1w { z27.s }, p1, [x23, #3, MUL VL]
    st1w { z28.s }, p4, [x22]
    st1w { z29.s }, p3, [x22, #1, MUL VL]
    st1w { z30.s }, p2, [x22, #2, MUL VL]
    st1w { z31.s }, p1, [x22, #3, MUL VL]
    decw x11, ALL, MUL #4
    cmp x11, XZR
    bgt label_67
    subs x1, x1, #0x6
    beq label_80
    ldr x21, [x2, #0x38]
    tbz x3, #3, label_79
    add x21, x21, #0x6
    str x21, [x2, #0x38]
    b label_1
KAI_ASM_LABEL(label_79)  // Update direct input
    mov x20, #0x18
    madd x0, x20, x21, x0
    b label_1
KAI_ASM_LABEL(label_80)  // Exit
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla)

    KAI_ASM_END
