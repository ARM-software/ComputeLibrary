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

    KAI_ASM_CODE(matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla)
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
    mov x8, #0x0
    ldr x11, [x0, #0x18]
    cntw x10, ALL, MUL #4
    ptrue p1.b
    ldr x9, [x0, #0x20]
    KAI_ASM_INST(0x25207811)  // ptrue pn9.b
    mov x22, #0x1
    ldr x21, [x0, #0x10]
    add x28, x11, x10
    ldr x20, [x0, #0x28]
    sub x28, x28, #0x1
    ldr x27, [x0, #0x8]
    udiv x28, x28, x10
    ldr x26, [x0, #0x30]
    mov x25, x21
    add x21, x28, #0x3
    mov x24, x20
    and x21, x21, #0xfffffffffffffffc
    mul x21, x21, x10
    mul x21, x21, x9
    lsl x21, x21, #0x2
KAI_ASM_LABEL(label_1)  // RHS size check loop
    cmp x21, #0x200, LSL #12
    blt label_2
    tbnz x21, #0, label_3
    lsr x21, x21, #0x1
    lsl x22, x22, #0x1
    b label_1
KAI_ASM_LABEL(label_2)  // RHS do prefetch
    lsl x20, x21, #0x26
    sub x22, x22, #0x1
    lsl x22, x22, #0x16
    orr x21, x21, x20
    orr x21, x21, x22
    KAI_ASM_INST(0xf8b54b3a)  // rprfm pldonce, x21, [x25]
KAI_ASM_LABEL(label_3)  // RHS prefetch exit
KAI_ASM_LABEL(label_4)  // Column loop
    cmp x28, #0x4
    bge label_22
    cmp x28, #0x2
    bgt label_16
    beq label_10
    KAI_ASM_INST(0xa040c734)  // ld1w { z20.s-z23.s }, pn9.b/Z, [x25]
    mov x23, x9
    mov x21, x11
    mov x22, x27
    lsl x20, x9, #0x2
    KAI_ASM_INST(0x25b567f0)  // whilelt p8.s, XZR, x21, VLx4
    cmp x23, #0x4
    KAI_ASM_INST(0xf8b44ad8)  // rprfm pldmany, x20, [x22]
    KAI_ASM_INST(0xc0040e80)  // mova za.d[x8, #0], { z20.d-z23.d }
    addvl x25, x25, #16
    ble label_6
KAI_ASM_LABEL(label_5)  // Width 1: Multiply loop: Main loop head
    whilelt p0.s, XZR, x23
    KAI_ASM_INST(0xa040c73d)  // ldnt1w { z28.s-z31.s }, pn9.b/Z, [x25]
    addvl x25, x25, #16
    ld1rqw { z2.s }, p0/Z, [x22]
    sub x23, x23, #0x4
    add x22, x22, #0x10
    KAI_ASM_INST(0xa040c731)  // ldnt1w { z16.s-z19.s }, pn9.b/Z, [x25]
    addvl x25, x25, #16
    cmp x23, #0x4
    KAI_ASM_INST(0xa040c739)  // ldnt1w { z24.s-z27.s }, pn9.b/Z, [x25]
    addvl x25, x25, #16
    KAI_ASM_INST(0xc1528380)  // fmla za.s[x8, 0], { z28.s-z31.s }, z2.s[0]
    KAI_ASM_INST(0xa040c72d)  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x25]
    addvl x25, x25, #16
    KAI_ASM_INST(0xc1528600)  // fmla za.s[x8, 0], { z16.s-z19.s }, z2.s[1]
    KAI_ASM_INST(0xc1528b00)  // fmla za.s[x8, 0], { z24.s-z27.s }, z2.s[2]
    KAI_ASM_INST(0xc1528d80)  // fmla za.s[x8, 0], { z12.s-z15.s }, z2.s[3]
    bgt label_5
KAI_ASM_LABEL(label_6)  // Width 1: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x23
    KAI_ASM_INST(0xa040c72d)  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x25]
    subs x23, x23, #0x1
    ld1rqw { z3.s }, p0/Z, [x22]
    addvl x25, x25, #16
    KAI_ASM_INST(0xc1538180)  // fmla za.s[x8, 0], { z12.s-z15.s }, z3.s[0]
    ble label_7
    KAI_ASM_INST(0xa040c725)  // ldnt1w { z4.s-z7.s }, pn9.b/Z, [x25]
    subs x23, x23, #0x1
    addvl x25, x25, #16
    KAI_ASM_INST(0xc1538480)  // fmla za.s[x8, 0], { z4.s-z7.s }, z3.s[1]
    ble label_7
    KAI_ASM_INST(0xa040c72d)  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x25]
    subs x23, x23, #0x1
    addvl x25, x25, #16
    KAI_ASM_INST(0xc1538980)  // fmla za.s[x8, 0], { z12.s-z15.s }, z3.s[2]
    ble label_7
    KAI_ASM_INST(0xa040c729)  // ldnt1w { z8.s-z11.s }, pn9.b/Z, [x25]
    KAI_ASM_INST(0xc1538d00)  // fmla za.s[x8, 0], { z8.s-z11.s }, z3.s[3]
KAI_ASM_LABEL(label_7)  // Width 1: Multiply loop: multiply skip
    tbz x26, #1, label_8
    add x21, x0, #0x4
    add x20, x0, #0x0
    KAI_ASM_INST(0xc0060c00)  // mova { z0.d-z3.d }, za.d[x8, #0]
    KAI_ASM_INST(0x8540c6b7)  // ld1rw { z23.s }, p1/Z, [x21]
    KAI_ASM_INST(0x8540c696)  // ld1rw { z22.s }, p1/Z, [x20]
    KAI_ASM_INST(0xc1b6cae0)  // fclamp { z0.s-z3.s }, z23.s, z22.s
    KAI_ASM_INST(0xa060c300)  // st1w { z0.s-z3.s }, p8, [x24]
    b label_9
KAI_ASM_LABEL(label_8)  // Width 1: No activation
    KAI_ASM_INST(0xc0060c00)  // mova { z0.d-z3.d }, za.d[x8, #0]
    KAI_ASM_INST(0xa060c300)  // st1w { z0.s-z3.s }, p8, [x24]
KAI_ASM_LABEL(label_9)  // Width 1: Output done
    b label_28
KAI_ASM_LABEL(label_10)  // Width 2
    KAI_ASM_INST(0xa040c73c)  // ld1w { z28.s-z31.s }, pn9.b/Z, [x25]
    mov x23, x9
    sub x21, x11, x10
    KAI_ASM_INST(0xa041c724)  // ld1w { z4.s-z7.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    mov x22, x27
    lsl x20, x9, #0x2
    KAI_ASM_INST(0x25b567f0)  // whilelt p8.s, XZR, x21, VLx4
    cmp x23, #0x4
    KAI_ASM_INST(0xf8b44ad8)  // rprfm pldmany, x20, [x22]
    KAI_ASM_INST(0xc0040f80)  // mova za.d[x8, #0], { z28.d-z31.d }
    addvl x25, x25, #16
    KAI_ASM_INST(0xc0040c81)  // mova za.d[x8, #1], { z4.d-z7.d }
    ble label_12
KAI_ASM_LABEL(label_11)  // Width 2: Multiply loop: Main loop head
    whilelt p0.s, XZR, x23
    KAI_ASM_INST(0xa040c73d)  // ldnt1w { z28.s-z31.s }, pn9.b/Z, [x25]
    sub x23, x23, #0x4
    ld1rqw { z1.s }, p0/Z, [x22]
    cmp x23, #0x4
    add x22, x22, #0x10
    KAI_ASM_INST(0xa041c72d)  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    addvl x25, x25, #16
    KAI_ASM_INST(0xa040c731)  // ldnt1w { z16.s-z19.s }, pn9.b/Z, [x25]
    KAI_ASM_INST(0xc1518380)  // fmla za.s[x8, 0], { z28.s-z31.s }, z1.s[0]
    KAI_ASM_INST(0xa041c739)  // ldnt1w { z24.s-z27.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    addvl x25, x25, #16
    KAI_ASM_INST(0xc1518181)  // fmla za.s[x8, 1], { z12.s-z15.s }, z1.s[0]
    KAI_ASM_INST(0xa040c73d)  // ldnt1w { z28.s-z31.s }, pn9.b/Z, [x25]
    KAI_ASM_INST(0xa041c72d)  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    addvl x25, x25, #16
    KAI_ASM_INST(0xa040c729)  // ldnt1w { z8.s-z11.s }, pn9.b/Z, [x25]
    KAI_ASM_INST(0xc1518600)  // fmla za.s[x8, 0], { z16.s-z19.s }, z1.s[1]
    KAI_ASM_INST(0xa041c735)  // ldnt1w { z20.s-z23.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    addvl x25, x25, #16
    KAI_ASM_INST(0xc1518701)  // fmla za.s[x8, 1], { z24.s-z27.s }, z1.s[1]
    KAI_ASM_INST(0xc1518b80)  // fmla za.s[x8, 0], { z28.s-z31.s }, z1.s[2]
    KAI_ASM_INST(0xc1518981)  // fmla za.s[x8, 1], { z12.s-z15.s }, z1.s[2]
    KAI_ASM_INST(0xc1518d00)  // fmla za.s[x8, 0], { z8.s-z11.s }, z1.s[3]
    KAI_ASM_INST(0xc1518e81)  // fmla za.s[x8, 1], { z20.s-z23.s }, z1.s[3]
    bgt label_11
KAI_ASM_LABEL(label_12)  // Width 2: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x23
    KAI_ASM_INST(0xa040c731)  // ldnt1w { z16.s-z19.s }, pn9.b/Z, [x25]
    subs x23, x23, #0x1
    ld1rqw { z3.s }, p0/Z, [x22]
    KAI_ASM_INST(0xa041c73d)  // ldnt1w { z28.s-z31.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    addvl x25, x25, #16
    KAI_ASM_INST(0xc1538200)  // fmla za.s[x8, 0], { z16.s-z19.s }, z3.s[0]
    KAI_ASM_INST(0xc1538381)  // fmla za.s[x8, 1], { z28.s-z31.s }, z3.s[0]
    ble label_13
    KAI_ASM_INST(0xa040c735)  // ldnt1w { z20.s-z23.s }, pn9.b/Z, [x25]
    subs x23, x23, #0x1
    KAI_ASM_INST(0xa041c731)  // ldnt1w { z16.s-z19.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    addvl x25, x25, #16
    KAI_ASM_INST(0xc1538680)  // fmla za.s[x8, 0], { z20.s-z23.s }, z3.s[1]
    KAI_ASM_INST(0xc1538601)  // fmla za.s[x8, 1], { z16.s-z19.s }, z3.s[1]
    ble label_13
    KAI_ASM_INST(0xa040c725)  // ldnt1w { z4.s-z7.s }, pn9.b/Z, [x25]
    subs x23, x23, #0x1
    KAI_ASM_INST(0xa041c731)  // ldnt1w { z16.s-z19.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    addvl x25, x25, #16
    KAI_ASM_INST(0xc1538880)  // fmla za.s[x8, 0], { z4.s-z7.s }, z3.s[2]
    KAI_ASM_INST(0xc1538a01)  // fmla za.s[x8, 1], { z16.s-z19.s }, z3.s[2]
    ble label_13
    KAI_ASM_INST(0xa040c73d)  // ldnt1w { z28.s-z31.s }, pn9.b/Z, [x25]
    KAI_ASM_INST(0xa041c72d)  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    KAI_ASM_INST(0xc1538f80)  // fmla za.s[x8, 0], { z28.s-z31.s }, z3.s[3]
    KAI_ASM_INST(0xc1538d81)  // fmla za.s[x8, 1], { z12.s-z15.s }, z3.s[3]
KAI_ASM_LABEL(label_13)  // Width 2: Multiply loop: multiply skip
    tbz x26, #1, label_14
    add x21, x0, #0x4
    add x20, x0, #0x0
    KAI_ASM_INST(0xc0060c04)  // mova { z4.d-z7.d }, za.d[x8, #0]
    KAI_ASM_INST(0xc0060c28)  // mova { z8.d-z11.d }, za.d[x8, #1]
    KAI_ASM_INST(0x8540c6b1)  // ld1rw { z17.s }, p1/Z, [x21]
    KAI_ASM_INST(0x8540c697)  // ld1rw { z23.s }, p1/Z, [x20]
    KAI_ASM_INST(0xc1b7ca24)  // fclamp { z4.s-z7.s }, z17.s, z23.s
    KAI_ASM_INST(0xc1b7ca28)  // fclamp { z8.s-z11.s }, z17.s, z23.s
    KAI_ASM_INST(0xa060c704)  // st1w { z4.s-z7.s }, pn9.b, [x24]
    KAI_ASM_INST(0xa061c308)  // st1w { z8.s-z11.s }, p8, [x24, #0x4, MUL VL]
    b label_15
KAI_ASM_LABEL(label_14)  // Width 2: No activation
    KAI_ASM_INST(0xc0060c08)  // mova { z8.d-z11.d }, za.d[x8, #0]
    KAI_ASM_INST(0xc0060c30)  // mova { z16.d-z19.d }, za.d[x8, #1]
    KAI_ASM_INST(0xa060c708)  // st1w { z8.s-z11.s }, pn9.b, [x24]
    KAI_ASM_INST(0xa061c310)  // st1w { z16.s-z19.s }, p8, [x24, #0x4, MUL VL]
KAI_ASM_LABEL(label_15)  // Width 2: Output done
    b label_28
KAI_ASM_LABEL(label_16)  // Width 3
    mov x20, #0x2
    KAI_ASM_INST(0xa040c728)  // ld1w { z8.s-z11.s }, pn9.b/Z, [x25]
    mov x23, x9
    KAI_ASM_INST(0xa041c720)  // ld1w { z0.s-z3.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    msub x21, x10, x20, x11
    mov x22, x27
    KAI_ASM_INST(0xa042c724)  // ld1w { z4.s-z7.s }, pn9.b/Z, [x25, #0x8, MUL VL]
    lsl x20, x9, #0x2
    KAI_ASM_INST(0x25b567f0)  // whilelt p8.s, XZR, x21, VLx4
    cmp x23, #0x4
    KAI_ASM_INST(0xf8b44ad8)  // rprfm pldmany, x20, [x22]
    KAI_ASM_INST(0xc0040d00)  // mova za.d[x8, #0], { z8.d-z11.d }
    KAI_ASM_INST(0xc0040c01)  // mova za.d[x8, #1], { z0.d-z3.d }
    addvl x25, x25, #16
    KAI_ASM_INST(0xc0040c82)  // mova za.d[x8, #2], { z4.d-z7.d }
    ble label_18
KAI_ASM_LABEL(label_17)  // Width 3: Multiply loop: Main loop head
    whilelt p0.s, XZR, x23
    KAI_ASM_INST(0xa040c72d)  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x25]
    sub x23, x23, #0x4
    ld1rqw { z3.s }, p0/Z, [x22]
    cmp x23, #0x4
    add x22, x22, #0x10
    KAI_ASM_INST(0xa041c729)  // ldnt1w { z8.s-z11.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c725)  // ldnt1w { z4.s-z7.s }, pn9.b/Z, [x25, #0x8, MUL VL]
    addvl x25, x25, #16
    KAI_ASM_INST(0xc1538180)  // fmla za.s[x8, 0], { z12.s-z15.s }, z3.s[0]
    KAI_ASM_INST(0xa040c731)  // ldnt1w { z16.s-z19.s }, pn9.b/Z, [x25]
    KAI_ASM_INST(0xc1538101)  // fmla za.s[x8, 1], { z8.s-z11.s }, z3.s[0]
    KAI_ASM_INST(0xa041c735)  // ldnt1w { z20.s-z23.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    KAI_ASM_INST(0xc1538082)  // fmla za.s[x8, 2], { z4.s-z7.s }, z3.s[0]
    KAI_ASM_INST(0xa042c739)  // ldnt1w { z24.s-z27.s }, pn9.b/Z, [x25, #0x8, MUL VL]
    addvl x25, x25, #16
    KAI_ASM_INST(0xa040c72d)  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x25]
    KAI_ASM_INST(0xc1538600)  // fmla za.s[x8, 0], { z16.s-z19.s }, z3.s[1]
    KAI_ASM_INST(0xa041c73d)  // ldnt1w { z28.s-z31.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    KAI_ASM_INST(0xc1538681)  // fmla za.s[x8, 1], { z20.s-z23.s }, z3.s[1]
    KAI_ASM_INST(0xa042c729)  // ldnt1w { z8.s-z11.s }, pn9.b/Z, [x25, #0x8, MUL VL]
    addvl x25, x25, #16
    KAI_ASM_INST(0xc1538702)  // fmla za.s[x8, 2], { z24.s-z27.s }, z3.s[1]
    KAI_ASM_INST(0xa040c725)  // ldnt1w { z4.s-z7.s }, pn9.b/Z, [x25]
    KAI_ASM_INST(0xa041c735)  // ldnt1w { z20.s-z23.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    KAI_ASM_INST(0xc1538980)  // fmla za.s[x8, 0], { z12.s-z15.s }, z3.s[2]
    KAI_ASM_INST(0xa042c739)  // ldnt1w { z24.s-z27.s }, pn9.b/Z, [x25, #0x8, MUL VL]
    addvl x25, x25, #16
    KAI_ASM_INST(0xc1538b81)  // fmla za.s[x8, 1], { z28.s-z31.s }, z3.s[2]
    KAI_ASM_INST(0xc1538902)  // fmla za.s[x8, 2], { z8.s-z11.s }, z3.s[2]
    KAI_ASM_INST(0xc1538c80)  // fmla za.s[x8, 0], { z4.s-z7.s }, z3.s[3]
    KAI_ASM_INST(0xc1538e81)  // fmla za.s[x8, 1], { z20.s-z23.s }, z3.s[3]
    KAI_ASM_INST(0xc1538f02)  // fmla za.s[x8, 2], { z24.s-z27.s }, z3.s[3]
    bgt label_17
KAI_ASM_LABEL(label_18)  // Width 3: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x23
    KAI_ASM_INST(0xa040c735)  // ldnt1w { z20.s-z23.s }, pn9.b/Z, [x25]
    subs x23, x23, #0x1
    ld1rqw { z3.s }, p0/Z, [x22]
    KAI_ASM_INST(0xa041c72d)  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c725)  // ldnt1w { z4.s-z7.s }, pn9.b/Z, [x25, #0x8, MUL VL]
    addvl x25, x25, #16
    KAI_ASM_INST(0xc1538280)  // fmla za.s[x8, 0], { z20.s-z23.s }, z3.s[0]
    KAI_ASM_INST(0xc1538181)  // fmla za.s[x8, 1], { z12.s-z15.s }, z3.s[0]
    KAI_ASM_INST(0xc1538082)  // fmla za.s[x8, 2], { z4.s-z7.s }, z3.s[0]
    ble label_19
    KAI_ASM_INST(0xa040c735)  // ldnt1w { z20.s-z23.s }, pn9.b/Z, [x25]
    subs x23, x23, #0x1
    KAI_ASM_INST(0xa041c729)  // ldnt1w { z8.s-z11.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c731)  // ldnt1w { z16.s-z19.s }, pn9.b/Z, [x25, #0x8, MUL VL]
    addvl x25, x25, #16
    KAI_ASM_INST(0xc1538680)  // fmla za.s[x8, 0], { z20.s-z23.s }, z3.s[1]
    KAI_ASM_INST(0xc1538501)  // fmla za.s[x8, 1], { z8.s-z11.s }, z3.s[1]
    KAI_ASM_INST(0xc1538602)  // fmla za.s[x8, 2], { z16.s-z19.s }, z3.s[1]
    ble label_19
    KAI_ASM_INST(0xa040c73d)  // ldnt1w { z28.s-z31.s }, pn9.b/Z, [x25]
    subs x23, x23, #0x1
    KAI_ASM_INST(0xa041c739)  // ldnt1w { z24.s-z27.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c72d)  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x25, #0x8, MUL VL]
    addvl x25, x25, #16
    KAI_ASM_INST(0xc1538b80)  // fmla za.s[x8, 0], { z28.s-z31.s }, z3.s[2]
    KAI_ASM_INST(0xc1538b01)  // fmla za.s[x8, 1], { z24.s-z27.s }, z3.s[2]
    KAI_ASM_INST(0xc1538982)  // fmla za.s[x8, 2], { z12.s-z15.s }, z3.s[2]
    ble label_19
    KAI_ASM_INST(0xa040c729)  // ldnt1w { z8.s-z11.s }, pn9.b/Z, [x25]
    KAI_ASM_INST(0xa041c73d)  // ldnt1w { z28.s-z31.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c72d)  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x25, #0x8, MUL VL]
    KAI_ASM_INST(0xc1538d00)  // fmla za.s[x8, 0], { z8.s-z11.s }, z3.s[3]
    KAI_ASM_INST(0xc1538f81)  // fmla za.s[x8, 1], { z28.s-z31.s }, z3.s[3]
    KAI_ASM_INST(0xc1538d82)  // fmla za.s[x8, 2], { z12.s-z15.s }, z3.s[3]
KAI_ASM_LABEL(label_19)  // Width 3: Multiply loop: multiply skip
    tbz x26, #1, label_20
    add x21, x0, #0x4
    add x20, x0, #0x0
    KAI_ASM_INST(0xc0060c08)  // mova { z8.d-z11.d }, za.d[x8, #0]
    KAI_ASM_INST(0xc0060c2c)  // mova { z12.d-z15.d }, za.d[x8, #1]
    KAI_ASM_INST(0x8540c6b5)  // ld1rw { z21.s }, p1/Z, [x21]
    KAI_ASM_INST(0xc0060c50)  // mova { z16.d-z19.d }, za.d[x8, #2]
    KAI_ASM_INST(0x8540c694)  // ld1rw { z20.s }, p1/Z, [x20]
    KAI_ASM_INST(0xc1b4caa8)  // fclamp { z8.s-z11.s }, z21.s, z20.s
    KAI_ASM_INST(0xc1b4caac)  // fclamp { z12.s-z15.s }, z21.s, z20.s
    KAI_ASM_INST(0xc1b4cab0)  // fclamp { z16.s-z19.s }, z21.s, z20.s
    KAI_ASM_INST(0xa060c708)  // st1w { z8.s-z11.s }, pn9.b, [x24]
    KAI_ASM_INST(0xa061c70c)  // st1w { z12.s-z15.s }, pn9.b, [x24, #0x4, MUL VL]
    KAI_ASM_INST(0xa062c310)  // st1w { z16.s-z19.s }, p8, [x24, #0x8, MUL VL]
    b label_21
KAI_ASM_LABEL(label_20)  // Width 3: No activation
    KAI_ASM_INST(0xc0060c04)  // mova { z4.d-z7.d }, za.d[x8, #0]
    KAI_ASM_INST(0xc0060c2c)  // mova { z12.d-z15.d }, za.d[x8, #1]
    KAI_ASM_INST(0xc0060c5c)  // mova { z28.d-z31.d }, za.d[x8, #2]
    KAI_ASM_INST(0xa060c704)  // st1w { z4.s-z7.s }, pn9.b, [x24]
    KAI_ASM_INST(0xa061c70c)  // st1w { z12.s-z15.s }, pn9.b, [x24, #0x4, MUL VL]
    KAI_ASM_INST(0xa062c31c)  // st1w { z28.s-z31.s }, p8, [x24, #0x8, MUL VL]
KAI_ASM_LABEL(label_21)  // Width 3: Output done
    b label_28
KAI_ASM_LABEL(label_22)  // Width 4
    mov x20, #0x3
    KAI_ASM_INST(0xa040c724)  // ld1w { z4.s-z7.s }, pn9.b/Z, [x25]
    mov x23, x9
    KAI_ASM_INST(0xa041c72c)  // ld1w { z12.s-z15.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    msub x21, x10, x20, x11
    mov x22, x27
    KAI_ASM_INST(0xa042c73c)  // ld1w { z28.s-z31.s }, pn9.b/Z, [x25, #0x8, MUL VL]
    lsl x20, x9, #0x2
    KAI_ASM_INST(0x25b567f0)  // whilelt p8.s, XZR, x21, VLx4
    KAI_ASM_INST(0xa043c730)  // ld1w { z16.s-z19.s }, pn9.b/Z, [x25, #0xc, MUL VL]
    cmp x23, #0x4
    KAI_ASM_INST(0xf8b44ad8)  // rprfm pldmany, x20, [x22]
    KAI_ASM_INST(0xc0040c80)  // mova za.d[x8, #0], { z4.d-z7.d }
    KAI_ASM_INST(0xc0040d81)  // mova za.d[x8, #1], { z12.d-z15.d }
    addvl x25, x25, #16
    KAI_ASM_INST(0xc0040f82)  // mova za.d[x8, #2], { z28.d-z31.d }
    KAI_ASM_INST(0xc0040e03)  // mova za.d[x8, #3], { z16.d-z19.d }
    ble label_24
KAI_ASM_LABEL(label_23)  // Width 4: Multiply loop: Main loop head
    whilelt p0.s, XZR, x23
    KAI_ASM_INST(0xa040c72d)  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x25]
    sub x23, x23, #0x4
    ld1rqw { z3.s }, p0/Z, [x22]
    cmp x23, #0x4
    add x22, x22, #0x10
    KAI_ASM_INST(0xa041c735)  // ldnt1w { z20.s-z23.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c731)  // ldnt1w { z16.s-z19.s }, pn9.b/Z, [x25, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c729)  // ldnt1w { z8.s-z11.s }, pn9.b/Z, [x25, #0xc, MUL VL]
    KAI_ASM_INST(0xc1538180)  // fmla za.s[x8, 0], { z12.s-z15.s }, z3.s[0]
    addvl x25, x25, #16
    KAI_ASM_INST(0xc1538281)  // fmla za.s[x8, 1], { z20.s-z23.s }, z3.s[0]
    KAI_ASM_INST(0xa040c739)  // ldnt1w { z24.s-z27.s }, pn9.b/Z, [x25]
    KAI_ASM_INST(0xc1538202)  // fmla za.s[x8, 2], { z16.s-z19.s }, z3.s[0]
    KAI_ASM_INST(0xa041c72d)  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    KAI_ASM_INST(0xc1538103)  // fmla za.s[x8, 3], { z8.s-z11.s }, z3.s[0]
    KAI_ASM_INST(0xa042c729)  // ldnt1w { z8.s-z11.s }, pn9.b/Z, [x25, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c725)  // ldnt1w { z4.s-z7.s }, pn9.b/Z, [x25, #0xc, MUL VL]
    KAI_ASM_INST(0xc1538700)  // fmla za.s[x8, 0], { z24.s-z27.s }, z3.s[1]
    addvl x25, x25, #16
    KAI_ASM_INST(0xc1538581)  // fmla za.s[x8, 1], { z12.s-z15.s }, z3.s[1]
    KAI_ASM_INST(0xa040c739)  // ldnt1w { z24.s-z27.s }, pn9.b/Z, [x25]
    KAI_ASM_INST(0xc1538502)  // fmla za.s[x8, 2], { z8.s-z11.s }, z3.s[1]
    KAI_ASM_INST(0xa041c729)  // ldnt1w { z8.s-z11.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    KAI_ASM_INST(0xc1538483)  // fmla za.s[x8, 3], { z4.s-z7.s }, z3.s[1]
    KAI_ASM_INST(0xa042c731)  // ldnt1w { z16.s-z19.s }, pn9.b/Z, [x25, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c725)  // ldnt1w { z4.s-z7.s }, pn9.b/Z, [x25, #0xc, MUL VL]
    KAI_ASM_INST(0xc1538b00)  // fmla za.s[x8, 0], { z24.s-z27.s }, z3.s[2]
    addvl x25, x25, #16
    KAI_ASM_INST(0xc1538901)  // fmla za.s[x8, 1], { z8.s-z11.s }, z3.s[2]
    KAI_ASM_INST(0xa040c729)  // ldnt1w { z8.s-z11.s }, pn9.b/Z, [x25]
    KAI_ASM_INST(0xc1538a02)  // fmla za.s[x8, 2], { z16.s-z19.s }, z3.s[2]
    KAI_ASM_INST(0xa041c73d)  // ldnt1w { z28.s-z31.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    KAI_ASM_INST(0xc1538883)  // fmla za.s[x8, 3], { z4.s-z7.s }, z3.s[2]
    KAI_ASM_INST(0xa042c72d)  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x25, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c735)  // ldnt1w { z20.s-z23.s }, pn9.b/Z, [x25, #0xc, MUL VL]
    KAI_ASM_INST(0xc1538d00)  // fmla za.s[x8, 0], { z8.s-z11.s }, z3.s[3]
    addvl x25, x25, #16
    KAI_ASM_INST(0xc1538f81)  // fmla za.s[x8, 1], { z28.s-z31.s }, z3.s[3]
    KAI_ASM_INST(0xc1538d82)  // fmla za.s[x8, 2], { z12.s-z15.s }, z3.s[3]
    KAI_ASM_INST(0xc1538e83)  // fmla za.s[x8, 3], { z20.s-z23.s }, z3.s[3]
    bgt label_23
KAI_ASM_LABEL(label_24)  // Width 4: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x23
    KAI_ASM_INST(0xa040c731)  // ldnt1w { z16.s-z19.s }, pn9.b/Z, [x25]
    subs x23, x23, #0x1
    ld1rqw { z3.s }, p0/Z, [x22]
    KAI_ASM_INST(0xa041c72d)  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c73d)  // ldnt1w { z28.s-z31.s }, pn9.b/Z, [x25, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c735)  // ldnt1w { z20.s-z23.s }, pn9.b/Z, [x25, #0xc, MUL VL]
    KAI_ASM_INST(0xc1538200)  // fmla za.s[x8, 0], { z16.s-z19.s }, z3.s[0]
    addvl x25, x25, #16
    KAI_ASM_INST(0xc1538181)  // fmla za.s[x8, 1], { z12.s-z15.s }, z3.s[0]
    KAI_ASM_INST(0xc1538382)  // fmla za.s[x8, 2], { z28.s-z31.s }, z3.s[0]
    KAI_ASM_INST(0xc1538283)  // fmla za.s[x8, 3], { z20.s-z23.s }, z3.s[0]
    ble label_25
    KAI_ASM_INST(0xa040c72d)  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x25]
    subs x23, x23, #0x1
    KAI_ASM_INST(0xa041c725)  // ldnt1w { z4.s-z7.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c739)  // ldnt1w { z24.s-z27.s }, pn9.b/Z, [x25, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c735)  // ldnt1w { z20.s-z23.s }, pn9.b/Z, [x25, #0xc, MUL VL]
    KAI_ASM_INST(0xc1538580)  // fmla za.s[x8, 0], { z12.s-z15.s }, z3.s[1]
    addvl x25, x25, #16
    KAI_ASM_INST(0xc1538481)  // fmla za.s[x8, 1], { z4.s-z7.s }, z3.s[1]
    KAI_ASM_INST(0xc1538702)  // fmla za.s[x8, 2], { z24.s-z27.s }, z3.s[1]
    KAI_ASM_INST(0xc1538683)  // fmla za.s[x8, 3], { z20.s-z23.s }, z3.s[1]
    ble label_25
    KAI_ASM_INST(0xa040c72d)  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x25]
    subs x23, x23, #0x1
    KAI_ASM_INST(0xa041c729)  // ldnt1w { z8.s-z11.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c735)  // ldnt1w { z20.s-z23.s }, pn9.b/Z, [x25, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c731)  // ldnt1w { z16.s-z19.s }, pn9.b/Z, [x25, #0xc, MUL VL]
    KAI_ASM_INST(0xc1538980)  // fmla za.s[x8, 0], { z12.s-z15.s }, z3.s[2]
    addvl x25, x25, #16
    KAI_ASM_INST(0xc1538901)  // fmla za.s[x8, 1], { z8.s-z11.s }, z3.s[2]
    KAI_ASM_INST(0xc1538a82)  // fmla za.s[x8, 2], { z20.s-z23.s }, z3.s[2]
    KAI_ASM_INST(0xc1538a03)  // fmla za.s[x8, 3], { z16.s-z19.s }, z3.s[2]
    ble label_25
    KAI_ASM_INST(0xa040c72d)  // ldnt1w { z12.s-z15.s }, pn9.b/Z, [x25]
    KAI_ASM_INST(0xa041c729)  // ldnt1w { z8.s-z11.s }, pn9.b/Z, [x25, #0x4, MUL VL]
    KAI_ASM_INST(0xa042c735)  // ldnt1w { z20.s-z23.s }, pn9.b/Z, [x25, #0x8, MUL VL]
    KAI_ASM_INST(0xa043c731)  // ldnt1w { z16.s-z19.s }, pn9.b/Z, [x25, #0xc, MUL VL]
    KAI_ASM_INST(0xc1538d80)  // fmla za.s[x8, 0], { z12.s-z15.s }, z3.s[3]
    addvl x25, x25, #16
    KAI_ASM_INST(0xc1538d01)  // fmla za.s[x8, 1], { z8.s-z11.s }, z3.s[3]
    KAI_ASM_INST(0xc1538e82)  // fmla za.s[x8, 2], { z20.s-z23.s }, z3.s[3]
    KAI_ASM_INST(0xc1538e03)  // fmla za.s[x8, 3], { z16.s-z19.s }, z3.s[3]
KAI_ASM_LABEL(label_25)  // Width 4: Multiply loop: multiply skip
    tbz x26, #1, label_26
    add x21, x0, #0x4
    add x20, x0, #0x0
    KAI_ASM_INST(0xc0060c04)  // mova { z4.d-z7.d }, za.d[x8, #0]
    KAI_ASM_INST(0xc0060c20)  // mova { z0.d-z3.d }, za.d[x8, #1]
    KAI_ASM_INST(0x8540c6b5)  // ld1rw { z21.s }, p1/Z, [x21]
    KAI_ASM_INST(0xc0060c4c)  // mova { z12.d-z15.d }, za.d[x8, #2]
    KAI_ASM_INST(0x8540c694)  // ld1rw { z20.s }, p1/Z, [x20]
    KAI_ASM_INST(0xc0060c70)  // mova { z16.d-z19.d }, za.d[x8, #3]
    KAI_ASM_INST(0xc1b4caa4)  // fclamp { z4.s-z7.s }, z21.s, z20.s
    KAI_ASM_INST(0xc1b4caa0)  // fclamp { z0.s-z3.s }, z21.s, z20.s
    KAI_ASM_INST(0xc1b4caac)  // fclamp { z12.s-z15.s }, z21.s, z20.s
    KAI_ASM_INST(0xc1b4cab0)  // fclamp { z16.s-z19.s }, z21.s, z20.s
    KAI_ASM_INST(0xa060c704)  // st1w { z4.s-z7.s }, pn9.b, [x24]
    KAI_ASM_INST(0xa061c700)  // st1w { z0.s-z3.s }, pn9.b, [x24, #0x4, MUL VL]
    KAI_ASM_INST(0xa062c70c)  // st1w { z12.s-z15.s }, pn9.b, [x24, #0x8, MUL VL]
    KAI_ASM_INST(0xa063c310)  // st1w { z16.s-z19.s }, p8, [x24, #0xc, MUL VL]
    addvl x24, x24, #16
    b label_27
KAI_ASM_LABEL(label_26)  // Width 4: No activation
    KAI_ASM_INST(0xc0060c0c)  // mova { z12.d-z15.d }, za.d[x8, #0]
    KAI_ASM_INST(0xc0060c20)  // mova { z0.d-z3.d }, za.d[x8, #1]
    KAI_ASM_INST(0xc0060c50)  // mova { z16.d-z19.d }, za.d[x8, #2]
    KAI_ASM_INST(0xc0060c64)  // mova { z4.d-z7.d }, za.d[x8, #3]
    KAI_ASM_INST(0xa060c70c)  // st1w { z12.s-z15.s }, pn9.b, [x24]
    KAI_ASM_INST(0xa061c700)  // st1w { z0.s-z3.s }, pn9.b, [x24, #0x4, MUL VL]
    KAI_ASM_INST(0xa062c710)  // st1w { z16.s-z19.s }, pn9.b, [x24, #0x8, MUL VL]
    KAI_ASM_INST(0xa063c304)  // st1w { z4.s-z7.s }, p8, [x24, #0xc, MUL VL]
    addvl x24, x24, #16
KAI_ASM_LABEL(label_27)  // Width 4: Output done
    subs x28, x28, #0x4
    sub x11, x11, x10, LSL #2
    bgt label_4
KAI_ASM_LABEL(label_28)  // Exit
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme2_mla)

    KAI_ASM_END
