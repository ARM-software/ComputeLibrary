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

    KAI_ASM_CODE(matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot)
    KAI_ASM_GLOBAL(kai_f16_from_float_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot)

KAI_ASM_FUNCTION_TYPE(kai_f16_from_float_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot)
KAI_ASM_FUNCTION_LABEL(kai_f16_from_float_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot)
    fcvt h0, s0
    fmov w0, h0
    ret
    KAI_ASM_FUNCTION_END(kai_f16_from_float_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot)
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
    ldr x5, [x0, #0x18]
    cntw x6, ALL, MUL #4
    ptrue p1.b
    ldr x7, [x0, #0x20]
    KAI_ASM_INST(0x25207811)  // ptrue pn9.b
    mov x22, #0x1
    ldr x21, [x0, #0x10]
    add x17, x5, x6
    ldr x20, [x0, #0x28]
    sub x17, x17, #0x1
    ldr x16, [x0, #0x8]
    udiv x17, x17, x6
    ldr x15, [x0, #0x30]
    mov x14, x21
    add x21, x17, #0x3
    mov x13, x20
    and x21, x21, #0xfffffffffffffffc
    mul x21, x21, x6
    mul x21, x21, x7
    lsl x21, x21, #0x1
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
    KAI_ASM_INST(0xf8b549da)  // rprfm pldonce, x21, [x14]
KAI_ASM_LABEL(label_3)  // RHS prefetch exit
    add x12, x7, #0x1
    cntw x20, ALL, MUL #2
    bic x12, x12, #0x1
    lsl x12, x12, #0x1
    add x12, x12, #0x2
    mul x12, x12, x20
KAI_ASM_LABEL(label_4)  // Column loop
    cmp x17, #0x4
    bge label_22
    cmp x17, #0x2
    bgt label_16
    beq label_10
    cntw x20, ALL, MUL #2
    add x22, x14, x12
    ld1h { z8.s }, p1/Z, [x14]
    cmp x5, x20
    ld1h { z9.s }, p1/Z, [x14, #1, MUL VL]
    mov x11, x7
    csel x22, x22, x14, GT
    mov x21, x5
    ld1h { z10.s }, p1/Z, [x22]
    fcvt z8.s, p1/m, z8.h
    mov x10, x16
    lsl x20, x7, #0x1
    ld1h { z11.s }, p1/Z, [x22, #1, MUL VL]
    fcvt z9.s, p1/m, z9.h
    KAI_ASM_INST(0x257547f0)  // whilelt p8.h, XZR, x21, VLx2
    cmp x11, #0x8
    KAI_ASM_INST(0xf8b44958)  // rprfm pldmany, x20, [x10]
    inch x14, ALL, MUL #2
    fcvt z10.s, p1/m, z10.h
    inch x22, ALL, MUL #2
    fcvt z11.s, p1/m, z11.h
    KAI_ASM_INST(0xc0040d00)  // mova za.d[x8, #0], { z8.d-z11.d }
    ble label_6
KAI_ASM_LABEL(label_5)  // Width 1: Multiply loop: Main loop head
    whilelt p0.h, XZR, x11
    KAI_ASM_INST(0xa04025d5)  // ldnt1h { z20.h-z21.h }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    ld1rqh { z4.h }, p0/Z, [x10]
    sub x11, x11, #0x8
    add x10, x10, #0x10
    KAI_ASM_INST(0xa04026d7)  // ldnt1h { z22.h-z23.h }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    cmp x11, #0x8
    KAI_ASM_INST(0xa04025c9)  // ldnt1h { z8.h-z9.h }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04026cb)  // ldnt1h { z10.h-z11.h }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xc1549288)  // fdot za.s[x8, 0], { z20.h-z23.h }, z4.h[0]
    KAI_ASM_INST(0xa04025cd)  // ldnt1h { z12.h-z13.h }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04026cf)  // ldnt1h { z14.h-z15.h }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa04025d5)  // ldnt1h { z20.h-z21.h }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04026d7)  // ldnt1h { z22.h-z23.h }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xc1549508)  // fdot za.s[x8, 0], { z8.h-z11.h }, z4.h[1]
    KAI_ASM_INST(0xc1549988)  // fdot za.s[x8, 0], { z12.h-z15.h }, z4.h[2]
    KAI_ASM_INST(0xc1549e88)  // fdot za.s[x8, 0], { z20.h-z23.h }, z4.h[3]
    bgt label_5
KAI_ASM_LABEL(label_6)  // Width 1: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x11
    KAI_ASM_INST(0xa04025cd)  // ldnt1h { z12.h-z13.h }, pn9.b/Z, [x14]
    subs x11, x11, #0x2
    ld1rqh { z3.h }, p0/Z, [x10]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04026cf)  // ldnt1h { z14.h-z15.h }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xc1539188)  // fdot za.s[x8, 0], { z12.h-z15.h }, z3.h[0]
    ble label_7
    KAI_ASM_INST(0xa04025c5)  // ldnt1h { z4.h-z5.h }, pn9.b/Z, [x14]
    subs x11, x11, #0x2
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04026c7)  // ldnt1h { z6.h-z7.h }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xc1539488)  // fdot za.s[x8, 0], { z4.h-z7.h }, z3.h[1]
    ble label_7
    KAI_ASM_INST(0xa04025cd)  // ldnt1h { z12.h-z13.h }, pn9.b/Z, [x14]
    subs x11, x11, #0x2
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04026cf)  // ldnt1h { z14.h-z15.h }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xc1539988)  // fdot za.s[x8, 0], { z12.h-z15.h }, z3.h[2]
    ble label_7
    KAI_ASM_INST(0xa04025d1)  // ldnt1h { z16.h-z17.h }, pn9.b/Z, [x14]
    KAI_ASM_INST(0xa04026d3)  // ldnt1h { z18.h-z19.h }, pn9.b/Z, [x22]
    KAI_ASM_INST(0xc1539e08)  // fdot za.s[x8, 0], { z16.h-z19.h }, z3.h[3]
KAI_ASM_LABEL(label_7)  // Width 1: Multiply loop: multiply skip
    tbz x15, #1, label_8
    KAI_ASM_INST(0xc0060c04)  // mova { z4.d-z7.d }, za.d[x8, #0]
    add x21, x0, #0x2
    add x20, x0, #0x0
    KAI_ASM_INST(0x84c0a6b3)  // ld1rh { z19.h }, p1/Z, [x21]
    KAI_ASM_INST(0x84c0a696)  // ld1rh { z22.h }, p1/Z, [x20]
    KAI_ASM_INST(0xc120e094)  // fcvt z20.h, { z4.s-z5.s }
    KAI_ASM_INST(0xc120e0d5)  // fcvt z21.h, { z6.s-z7.s }
    KAI_ASM_INST(0xc176c274)  // fclamp { z20.h-z21.h }, z19.h, z22.h
    KAI_ASM_INST(0xa06021b4)  // st1h { z20.h-z21.h }, p8, [x13]
    b label_9
KAI_ASM_LABEL(label_8)  // Width 1: No activation
    KAI_ASM_INST(0xc0060c04)  // mova { z4.d-z7.d }, za.d[x8, #0]
    KAI_ASM_INST(0xc120e084)  // fcvt z4.h, { z4.s-z5.s }
    KAI_ASM_INST(0xc120e0c5)  // fcvt z5.h, { z6.s-z7.s }
    KAI_ASM_INST(0xa06021a4)  // st1h { z4.h-z5.h }, p8, [x13]
KAI_ASM_LABEL(label_9)  // Width 1: Output done
    b label_28
KAI_ASM_LABEL(label_10)  // Width 2
    add x24, x14, x12, LSL #1
    cntw x20, ALL, MUL #6
    ld1h { z24.s }, p1/Z, [x14]
    add x23, x24, x12
    cmp x5, x20
    ld1h { z25.s }, p1/Z, [x14, #1, MUL VL]
    add x22, x14, x12
    csel x23, x23, x14, GT
    ld1h { z0.s }, p1/Z, [x24]
    ld1h { z26.s }, p1/Z, [x22]
    fcvt z24.s, p1/m, z24.h
    mov x11, x7
    sub x21, x5, x6
    ld1h { z27.s }, p1/Z, [x22, #1, MUL VL]
    fcvt z25.s, p1/m, z25.h
    mov x10, x16
    lsl x20, x7, #0x1
    ld1h { z1.s }, p1/Z, [x24, #1, MUL VL]
    fcvt z0.s, p1/m, z0.h
    KAI_ASM_INST(0x257547f0)  // whilelt p8.h, XZR, x21, VLx2
    cmp x11, #0x8
    ld1h { z2.s }, p1/Z, [x23]
    fcvt z26.s, p1/m, z26.h
    KAI_ASM_INST(0xf8b44958)  // rprfm pldmany, x20, [x10]
    inch x14, ALL, MUL #2
    ld1h { z3.s }, p1/Z, [x23, #1, MUL VL]
    fcvt z27.s, p1/m, z27.h
    inch x22, ALL, MUL #2
    inch x24, ALL, MUL #2
    fcvt z1.s, p1/m, z1.h
    inch x23, ALL, MUL #2
    fcvt z2.s, p1/m, z2.h
    fcvt z3.s, p1/m, z3.h
    KAI_ASM_INST(0xc0040f00)  // mova za.d[x8, #0], { z24.d-z27.d }
    KAI_ASM_INST(0xc0040c01)  // mova za.d[x8, #1], { z0.d-z3.d }
    ble label_12
KAI_ASM_LABEL(label_11)  // Width 2: Multiply loop: Main loop head
    whilelt p0.h, XZR, x11
    KAI_ASM_INST(0xa04025d5)  // ldnt1h { z20.h-z21.h }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    ld1rqh { z4.h }, p0/Z, [x10]
    sub x11, x11, #0x8
    add x10, x10, #0x10
    KAI_ASM_INST(0xa04026d7)  // ldnt1h { z22.h-z23.h }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    cmp x11, #0x8
    KAI_ASM_INST(0xa0402709)  // ldnt1h { z8.h-z9.h }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa04026eb)  // ldnt1h { z10.h-z11.h }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc1549288)  // fdot za.s[x8, 0], { z20.h-z23.h }, z4.h[0]
    KAI_ASM_INST(0xa04025d5)  // ldnt1h { z20.h-z21.h }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04026d7)  // ldnt1h { z22.h-z23.h }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xc1549109)  // fdot za.s[x8, 1], { z8.h-z11.h }, z4.h[0]
    KAI_ASM_INST(0xa040270d)  // ldnt1h { z12.h-z13.h }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa04026ef)  // ldnt1h { z14.h-z15.h }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc1549688)  // fdot za.s[x8, 0], { z20.h-z23.h }, z4.h[1]
    KAI_ASM_INST(0xa04025d1)  // ldnt1h { z16.h-z17.h }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04026d3)  // ldnt1h { z18.h-z19.h }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xc1549589)  // fdot za.s[x8, 1], { z12.h-z15.h }, z4.h[1]
    KAI_ASM_INST(0xa0402719)  // ldnt1h { z24.h-z25.h }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa04026fb)  // ldnt1h { z26.h-z27.h }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc1549a08)  // fdot za.s[x8, 0], { z16.h-z19.h }, z4.h[2]
    KAI_ASM_INST(0xa04025d5)  // ldnt1h { z20.h-z21.h }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04026d7)  // ldnt1h { z22.h-z23.h }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xc1549b09)  // fdot za.s[x8, 1], { z24.h-z27.h }, z4.h[2]
    KAI_ASM_INST(0xa0402709)  // ldnt1h { z8.h-z9.h }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa04026eb)  // ldnt1h { z10.h-z11.h }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc1549e88)  // fdot za.s[x8, 0], { z20.h-z23.h }, z4.h[3]
    KAI_ASM_INST(0xc1549d09)  // fdot za.s[x8, 1], { z8.h-z11.h }, z4.h[3]
    bgt label_11
KAI_ASM_LABEL(label_12)  // Width 2: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x11
    KAI_ASM_INST(0xa04025d5)  // ldnt1h { z20.h-z21.h }, pn9.b/Z, [x14]
    subs x11, x11, #0x2
    ld1rqh { z3.h }, p0/Z, [x10]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04026d7)  // ldnt1h { z22.h-z23.h }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa040270d)  // ldnt1h { z12.h-z13.h }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa04026ef)  // ldnt1h { z14.h-z15.h }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc1539288)  // fdot za.s[x8, 0], { z20.h-z23.h }, z3.h[0]
    KAI_ASM_INST(0xc1539189)  // fdot za.s[x8, 1], { z12.h-z15.h }, z3.h[0]
    ble label_13
    KAI_ASM_INST(0xa04025d1)  // ldnt1h { z16.h-z17.h }, pn9.b/Z, [x14]
    subs x11, x11, #0x2
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04026d3)  // ldnt1h { z18.h-z19.h }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0402709)  // ldnt1h { z8.h-z9.h }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa04026eb)  // ldnt1h { z10.h-z11.h }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc1539608)  // fdot za.s[x8, 0], { z16.h-z19.h }, z3.h[1]
    KAI_ASM_INST(0xc1539509)  // fdot za.s[x8, 1], { z8.h-z11.h }, z3.h[1]
    ble label_13
    KAI_ASM_INST(0xa04025d9)  // ldnt1h { z24.h-z25.h }, pn9.b/Z, [x14]
    subs x11, x11, #0x2
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04026db)  // ldnt1h { z26.h-z27.h }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0402711)  // ldnt1h { z16.h-z17.h }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa04026f3)  // ldnt1h { z18.h-z19.h }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc1539b08)  // fdot za.s[x8, 0], { z24.h-z27.h }, z3.h[2]
    KAI_ASM_INST(0xc1539a09)  // fdot za.s[x8, 1], { z16.h-z19.h }, z3.h[2]
    ble label_13
    KAI_ASM_INST(0xa04025c9)  // ldnt1h { z8.h-z9.h }, pn9.b/Z, [x14]
    KAI_ASM_INST(0xa04026cb)  // ldnt1h { z10.h-z11.h }, pn9.b/Z, [x22]
    KAI_ASM_INST(0xa0402705)  // ldnt1h { z4.h-z5.h }, pn9.b/Z, [x24]
    KAI_ASM_INST(0xa04026e7)  // ldnt1h { z6.h-z7.h }, pn9.b/Z, [x23]
    KAI_ASM_INST(0xc1539d08)  // fdot za.s[x8, 0], { z8.h-z11.h }, z3.h[3]
    KAI_ASM_INST(0xc1539c89)  // fdot za.s[x8, 1], { z4.h-z7.h }, z3.h[3]
KAI_ASM_LABEL(label_13)  // Width 2: Multiply loop: multiply skip
    tbz x15, #1, label_14
    KAI_ASM_INST(0xc0060c08)  // mova { z8.d-z11.d }, za.d[x8, #0]
    add x21, x0, #0x2
    add x20, x0, #0x0
    KAI_ASM_INST(0xc0060c2c)  // mova { z12.d-z15.d }, za.d[x8, #1]
    KAI_ASM_INST(0x84c0a6a6)  // ld1rh { z6.h }, p1/Z, [x21]
    KAI_ASM_INST(0x84c0a696)  // ld1rh { z22.h }, p1/Z, [x20]
    KAI_ASM_INST(0xc120e112)  // fcvt z18.h, { z8.s-z9.s }
    KAI_ASM_INST(0xc120e153)  // fcvt z19.h, { z10.s-z11.s }
    KAI_ASM_INST(0xc120e190)  // fcvt z16.h, { z12.s-z13.s }
    KAI_ASM_INST(0xc120e1d1)  // fcvt z17.h, { z14.s-z15.s }
    KAI_ASM_INST(0xc176c0d2)  // fclamp { z18.h-z19.h }, z6.h, z22.h
    KAI_ASM_INST(0xc176c0d0)  // fclamp { z16.h-z17.h }, z6.h, z22.h
    KAI_ASM_INST(0xa06025b2)  // st1h { z18.h-z19.h }, pn9.b, [x13]
    KAI_ASM_INST(0xa06121b0)  // st1h { z16.h-z17.h }, p8, [x13, #0x2, MUL VL]
    b label_15
KAI_ASM_LABEL(label_14)  // Width 2: No activation
    KAI_ASM_INST(0xc0060c1c)  // mova { z28.d-z31.d }, za.d[x8, #0]
    KAI_ASM_INST(0xc0060c34)  // mova { z20.d-z23.d }, za.d[x8, #1]
    KAI_ASM_INST(0xc120e39a)  // fcvt z26.h, { z28.s-z29.s }
    KAI_ASM_INST(0xc120e3db)  // fcvt z27.h, { z30.s-z31.s }
    KAI_ASM_INST(0xa06025ba)  // st1h { z26.h-z27.h }, pn9.b, [x13]
    KAI_ASM_INST(0xc120e291)  // fcvt z17.h, { z20.s-z21.s }
    KAI_ASM_INST(0xc120e2d9)  // fcvt z25.h, { z22.s-z23.s }
    KAI_ASM_INST(0xa16121b1)  // st1h { z17.h, z25.h }, p8, [x13, #0x2, MUL VL]
KAI_ASM_LABEL(label_15)  // Width 2: Output done
    b label_28
KAI_ASM_LABEL(label_16)  // Width 3
    add x26, x14, x12, LSL #2
    cntw x20, ALL, MUL #10
    ld1h { z28.s }, p1/Z, [x14]
    add x25, x14, x12, LSL #1
    add x24, x26, x12
    ld1h { z29.s }, p1/Z, [x14, #1, MUL VL]
    cmp x5, x20
    add x23, x14, x12
    ld1h { z4.s }, p1/Z, [x25]
    add x22, x25, x12
    csel x24, x24, x14, GT
    ld1h { z30.s }, p1/Z, [x23]
    fcvt z28.s, p1/m, z28.h
    ld1h { z31.s }, p1/Z, [x23, #1, MUL VL]
    fcvt z29.s, p1/m, z29.h
    mov x20, #0x2
    mov x11, x7
    ld1h { z5.s }, p1/Z, [x25, #1, MUL VL]
    fcvt z4.s, p1/m, z4.h
    msub x21, x6, x20, x5
    mov x10, x16
    ld1h { z6.s }, p1/Z, [x22]
    fcvt z30.s, p1/m, z30.h
    lsl x20, x7, #0x1
    KAI_ASM_INST(0x257547f0)  // whilelt p8.h, XZR, x21, VLx2
    ld1h { z7.s }, p1/Z, [x22, #1, MUL VL]
    fcvt z31.s, p1/m, z31.h
    cmp x11, #0x8
    KAI_ASM_INST(0xf8b44958)  // rprfm pldmany, x20, [x10]
    ld1h { z8.s }, p1/Z, [x26]
    fcvt z5.s, p1/m, z5.h
    inch x14, ALL, MUL #2
    inch x23, ALL, MUL #2
    ld1h { z9.s }, p1/Z, [x26, #1, MUL VL]
    fcvt z6.s, p1/m, z6.h
    inch x25, ALL, MUL #2
    inch x22, ALL, MUL #2
    ld1h { z10.s }, p1/Z, [x24]
    fcvt z7.s, p1/m, z7.h
    inch x26, ALL, MUL #2
    ld1h { z11.s }, p1/Z, [x24, #1, MUL VL]
    fcvt z8.s, p1/m, z8.h
    inch x24, ALL, MUL #2
    KAI_ASM_INST(0xc0040f80)  // mova za.d[x8, #0], { z28.d-z31.d }
    fcvt z9.s, p1/m, z9.h
    fcvt z10.s, p1/m, z10.h
    fcvt z11.s, p1/m, z11.h
    KAI_ASM_INST(0xc0040c81)  // mova za.d[x8, #1], { z4.d-z7.d }
    KAI_ASM_INST(0xc0040d02)  // mova za.d[x8, #2], { z8.d-z11.d }
    ble label_18
KAI_ASM_LABEL(label_17)  // Width 3: Multiply loop: Main loop head
    whilelt p0.h, XZR, x11
    KAI_ASM_INST(0xa04025cd)  // ldnt1h { z12.h-z13.h }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    ld1rqh { z4.h }, p0/Z, [x10]
    sub x11, x11, #0x8
    add x10, x10, #0x10
    KAI_ASM_INST(0xa04026ef)  // ldnt1h { z14.h-z15.h }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    cmp x11, #0x8
    KAI_ASM_INST(0xa0402731)  // ldnt1h { z16.h-z17.h }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa04026d3)  // ldnt1h { z18.h-z19.h }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0402741)  // ldnt1h { z0.h-z1.h }, pn9.b/Z, [x26]
    KAI_ASM_INST(0xc1549188)  // fdot za.s[x8, 0], { z12.h-z15.h }, z4.h[0]
    addvl x26, x26, #2
    KAI_ASM_INST(0xa0402703)  // ldnt1h { z2.h-z3.h }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xc1549209)  // fdot za.s[x8, 1], { z16.h-z19.h }, z4.h[0]
    KAI_ASM_INST(0xa04025d1)  // ldnt1h { z16.h-z17.h }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04026f3)  // ldnt1h { z18.h-z19.h }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc154900a)  // fdot za.s[x8, 2], { z0.h-z3.h }, z4.h[0]
    KAI_ASM_INST(0xa040272d)  // ldnt1h { z12.h-z13.h }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa04026cf)  // ldnt1h { z14.h-z15.h }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0402755)  // ldnt1h { z20.h-z21.h }, pn9.b/Z, [x26]
    KAI_ASM_INST(0xc1549608)  // fdot za.s[x8, 0], { z16.h-z19.h }, z4.h[1]
    addvl x26, x26, #2
    KAI_ASM_INST(0xa0402717)  // ldnt1h { z22.h-z23.h }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xc1549589)  // fdot za.s[x8, 1], { z12.h-z15.h }, z4.h[1]
    KAI_ASM_INST(0xa04025cd)  // ldnt1h { z12.h-z13.h }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04026ef)  // ldnt1h { z14.h-z15.h }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc154968a)  // fdot za.s[x8, 2], { z20.h-z23.h }, z4.h[1]
    KAI_ASM_INST(0xa0402729)  // ldnt1h { z8.h-z9.h }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa04026cb)  // ldnt1h { z10.h-z11.h }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0402751)  // ldnt1h { z16.h-z17.h }, pn9.b/Z, [x26]
    KAI_ASM_INST(0xc1549988)  // fdot za.s[x8, 0], { z12.h-z15.h }, z4.h[2]
    addvl x26, x26, #2
    KAI_ASM_INST(0xa0402713)  // ldnt1h { z18.h-z19.h }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xc1549909)  // fdot za.s[x8, 1], { z8.h-z11.h }, z4.h[2]
    KAI_ASM_INST(0xa04025d9)  // ldnt1h { z24.h-z25.h }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04026fb)  // ldnt1h { z26.h-z27.h }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc1549a0a)  // fdot za.s[x8, 2], { z16.h-z19.h }, z4.h[2]
    KAI_ASM_INST(0xa0402731)  // ldnt1h { z16.h-z17.h }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa04026d3)  // ldnt1h { z18.h-z19.h }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa040274d)  // ldnt1h { z12.h-z13.h }, pn9.b/Z, [x26]
    KAI_ASM_INST(0xc1549f08)  // fdot za.s[x8, 0], { z24.h-z27.h }, z4.h[3]
    addvl x26, x26, #2
    KAI_ASM_INST(0xa040270f)  // ldnt1h { z14.h-z15.h }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xc1549e09)  // fdot za.s[x8, 1], { z16.h-z19.h }, z4.h[3]
    KAI_ASM_INST(0xc1549d8a)  // fdot za.s[x8, 2], { z12.h-z15.h }, z4.h[3]
    bgt label_17
KAI_ASM_LABEL(label_18)  // Width 3: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x11
    KAI_ASM_INST(0xa04025c5)  // ldnt1h { z4.h-z5.h }, pn9.b/Z, [x14]
    subs x11, x11, #0x2
    ld1rqh { z3.h }, p0/Z, [x10]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04026e7)  // ldnt1h { z6.h-z7.h }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa040272d)  // ldnt1h { z12.h-z13.h }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa04026cf)  // ldnt1h { z14.h-z15.h }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0402759)  // ldnt1h { z24.h-z25.h }, pn9.b/Z, [x26]
    KAI_ASM_INST(0xc1539088)  // fdot za.s[x8, 0], { z4.h-z7.h }, z3.h[0]
    addvl x26, x26, #2
    KAI_ASM_INST(0xa040271b)  // ldnt1h { z26.h-z27.h }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xc1539189)  // fdot za.s[x8, 1], { z12.h-z15.h }, z3.h[0]
    KAI_ASM_INST(0xc153930a)  // fdot za.s[x8, 2], { z24.h-z27.h }, z3.h[0]
    ble label_19
    KAI_ASM_INST(0xa04025d9)  // ldnt1h { z24.h-z25.h }, pn9.b/Z, [x14]
    subs x11, x11, #0x2
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04026fb)  // ldnt1h { z26.h-z27.h }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa0402729)  // ldnt1h { z8.h-z9.h }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa04026cb)  // ldnt1h { z10.h-z11.h }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0402751)  // ldnt1h { z16.h-z17.h }, pn9.b/Z, [x26]
    KAI_ASM_INST(0xc1539708)  // fdot za.s[x8, 0], { z24.h-z27.h }, z3.h[1]
    addvl x26, x26, #2
    KAI_ASM_INST(0xa0402713)  // ldnt1h { z18.h-z19.h }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xc1539509)  // fdot za.s[x8, 1], { z8.h-z11.h }, z3.h[1]
    KAI_ASM_INST(0xc153960a)  // fdot za.s[x8, 2], { z16.h-z19.h }, z3.h[1]
    ble label_19
    KAI_ASM_INST(0xa04025cd)  // ldnt1h { z12.h-z13.h }, pn9.b/Z, [x14]
    subs x11, x11, #0x2
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04026ef)  // ldnt1h { z14.h-z15.h }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa0402729)  // ldnt1h { z8.h-z9.h }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa04026cb)  // ldnt1h { z10.h-z11.h }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0402745)  // ldnt1h { z4.h-z5.h }, pn9.b/Z, [x26]
    KAI_ASM_INST(0xc1539988)  // fdot za.s[x8, 0], { z12.h-z15.h }, z3.h[2]
    addvl x26, x26, #2
    KAI_ASM_INST(0xa0402707)  // ldnt1h { z6.h-z7.h }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xc1539909)  // fdot za.s[x8, 1], { z8.h-z11.h }, z3.h[2]
    KAI_ASM_INST(0xc153988a)  // fdot za.s[x8, 2], { z4.h-z7.h }, z3.h[2]
    ble label_19
    KAI_ASM_INST(0xa04025d9)  // ldnt1h { z24.h-z25.h }, pn9.b/Z, [x14]
    KAI_ASM_INST(0xa04026fb)  // ldnt1h { z26.h-z27.h }, pn9.b/Z, [x23]
    KAI_ASM_INST(0xa040273d)  // ldnt1h { z28.h-z29.h }, pn9.b/Z, [x25]
    KAI_ASM_INST(0xa04026df)  // ldnt1h { z30.h-z31.h }, pn9.b/Z, [x22]
    KAI_ASM_INST(0xa040274d)  // ldnt1h { z12.h-z13.h }, pn9.b/Z, [x26]
    KAI_ASM_INST(0xc1539f08)  // fdot za.s[x8, 0], { z24.h-z27.h }, z3.h[3]
    KAI_ASM_INST(0xa040270f)  // ldnt1h { z14.h-z15.h }, pn9.b/Z, [x24]
    KAI_ASM_INST(0xc1539f89)  // fdot za.s[x8, 1], { z28.h-z31.h }, z3.h[3]
    KAI_ASM_INST(0xc1539d8a)  // fdot za.s[x8, 2], { z12.h-z15.h }, z3.h[3]
KAI_ASM_LABEL(label_19)  // Width 3: Multiply loop: multiply skip
    tbz x15, #1, label_20
    KAI_ASM_INST(0xc0060c18)  // mova { z24.d-z27.d }, za.d[x8, #0]
    add x21, x0, #0x2
    add x20, x0, #0x0
    KAI_ASM_INST(0xc0060c3c)  // mova { z28.d-z31.d }, za.d[x8, #1]
    KAI_ASM_INST(0x84c0a6b3)  // ld1rh { z19.h }, p1/Z, [x21]
    KAI_ASM_INST(0xc0060c40)  // mova { z0.d-z3.d }, za.d[x8, #2]
    KAI_ASM_INST(0x84c0a692)  // ld1rh { z18.h }, p1/Z, [x20]
    KAI_ASM_INST(0xc120e314)  // fcvt z20.h, { z24.s-z25.s }
    KAI_ASM_INST(0xc120e355)  // fcvt z21.h, { z26.s-z27.s }
    KAI_ASM_INST(0xc120e38e)  // fcvt z14.h, { z28.s-z29.s }
    KAI_ASM_INST(0xc120e3cf)  // fcvt z15.h, { z30.s-z31.s }
    KAI_ASM_INST(0xc172c274)  // fclamp { z20.h-z21.h }, z19.h, z18.h
    KAI_ASM_INST(0xc120e010)  // fcvt z16.h, { z0.s-z1.s }
    KAI_ASM_INST(0xc120e051)  // fcvt z17.h, { z2.s-z3.s }
    KAI_ASM_INST(0xc172c26e)  // fclamp { z14.h-z15.h }, z19.h, z18.h
    KAI_ASM_INST(0xc172c270)  // fclamp { z16.h-z17.h }, z19.h, z18.h
    KAI_ASM_INST(0xa06025b4)  // st1h { z20.h-z21.h }, pn9.b, [x13]
    KAI_ASM_INST(0xa06125ae)  // st1h { z14.h-z15.h }, pn9.b, [x13, #0x2, MUL VL]
    KAI_ASM_INST(0xa06221b0)  // st1h { z16.h-z17.h }, p8, [x13, #0x4, MUL VL]
    b label_21
KAI_ASM_LABEL(label_20)  // Width 3: No activation
    KAI_ASM_INST(0xc0060c04)  // mova { z4.d-z7.d }, za.d[x8, #0]
    KAI_ASM_INST(0xc0060c28)  // mova { z8.d-z11.d }, za.d[x8, #1]
    KAI_ASM_INST(0xc0060c4c)  // mova { z12.d-z15.d }, za.d[x8, #2]
    KAI_ASM_INST(0xc120e091)  // fcvt z17.h, { z4.s-z5.s }
    KAI_ASM_INST(0xc120e0d9)  // fcvt z25.h, { z6.s-z7.s }
    KAI_ASM_INST(0xa16025b1)  // st1h { z17.h, z25.h }, pn9.b, [x13]
    KAI_ASM_INST(0xc120e112)  // fcvt z18.h, { z8.s-z9.s }
    KAI_ASM_INST(0xc120e153)  // fcvt z19.h, { z10.s-z11.s }
    KAI_ASM_INST(0xa06125b2)  // st1h { z18.h-z19.h }, pn9.b, [x13, #0x2, MUL VL]
    KAI_ASM_INST(0xc120e191)  // fcvt z17.h, { z12.s-z13.s }
    KAI_ASM_INST(0xc120e1d9)  // fcvt z25.h, { z14.s-z15.s }
    KAI_ASM_INST(0xa16221b1)  // st1h { z17.h, z25.h }, p8, [x13, #0x4, MUL VL]
KAI_ASM_LABEL(label_21)  // Width 3: Output done
    b label_28
KAI_ASM_LABEL(label_22)  // Width 4
    add x9, x14, x12, LSL #2
    cntw x20, ALL, MUL #14
    ld1h { z12.s }, p1/Z, [x14]
    add x28, x9, x12, LSL #1
    add x27, x14, x12, LSL #1
    ld1h { z13.s }, p1/Z, [x14, #1, MUL VL]
    add x26, x28, x12
    cmp x5, x20
    ld1h { z8.s }, p1/Z, [x27]
    add x25, x14, x12
    add x24, x27, x12
    ld1h { z9.s }, p1/Z, [x27, #1, MUL VL]
    fcvt z12.s, p1/m, z12.h
    add x23, x9, x12
    csel x26, x26, x14, GT
    ld1h { z14.s }, p1/Z, [x25]
    fcvt z13.s, p1/m, z13.h
    ld1h { z15.s }, p1/Z, [x25, #1, MUL VL]
    fcvt z8.s, p1/m, z8.h
    mov x20, #0x3
    mov x11, x7
    ld1h { z10.s }, p1/Z, [x24]
    fcvt z9.s, p1/m, z9.h
    msub x21, x6, x20, x5
    mov x10, x16
    ld1h { z11.s }, p1/Z, [x24, #1, MUL VL]
    fcvt z14.s, p1/m, z14.h
    lsl x20, x7, #0x1
    KAI_ASM_INST(0x257547f0)  // whilelt p8.h, XZR, x21, VLx2
    ld1h { z4.s }, p1/Z, [x9]
    fcvt z15.s, p1/m, z15.h
    cmp x11, #0x8
    KAI_ASM_INST(0xf8b44958)  // rprfm pldmany, x20, [x10]
    ld1h { z5.s }, p1/Z, [x9, #1, MUL VL]
    fcvt z10.s, p1/m, z10.h
    add x22, x14, x12, LSL #3
    inch x14, ALL, MUL #2
    ld1h { z6.s }, p1/Z, [x23]
    fcvt z11.s, p1/m, z11.h
    inch x25, ALL, MUL #2
    inch x27, ALL, MUL #2
    ld1h { z7.s }, p1/Z, [x23, #1, MUL VL]
    fcvt z4.s, p1/m, z4.h
    inch x24, ALL, MUL #2
    inch x9, ALL, MUL #2
    ld1h { z0.s }, p1/Z, [x28]
    fcvt z5.s, p1/m, z5.h
    inch x23, ALL, MUL #2
    KAI_ASM_INST(0xc0040d80)  // mova za.d[x8, #0], { z12.d-z15.d }
    ld1h { z1.s }, p1/Z, [x28, #1, MUL VL]
    fcvt z6.s, p1/m, z6.h
    inch x28, ALL, MUL #2
    ld1h { z2.s }, p1/Z, [x26]
    fcvt z7.s, p1/m, z7.h
    KAI_ASM_INST(0xc0040d01)  // mova za.d[x8, #1], { z8.d-z11.d }
    ld1h { z3.s }, p1/Z, [x26, #1, MUL VL]
    fcvt z0.s, p1/m, z0.h
    inch x26, ALL, MUL #2
    fcvt z1.s, p1/m, z1.h
    fcvt z2.s, p1/m, z2.h
    fcvt z3.s, p1/m, z3.h
    KAI_ASM_INST(0xc0040c82)  // mova za.d[x8, #2], { z4.d-z7.d }
    KAI_ASM_INST(0xc0040c03)  // mova za.d[x8, #3], { z0.d-z3.d }
    ble label_24
KAI_ASM_LABEL(label_23)  // Width 4: Multiply loop: Main loop head
    whilelt p0.h, XZR, x11
    KAI_ASM_INST(0xa04025cd)  // ldnt1h { z12.h-z13.h }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    ld1rqh { z0.h }, p0/Z, [x10]
    sub x11, x11, #0x8
    add x10, x10, #0x10
    KAI_ASM_INST(0xa040272f)  // ldnt1h { z14.h-z15.h }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    cmp x11, #0x8
    KAI_ASM_INST(0xa0402765)  // ldnt1h { z4.h-z5.h }, pn9.b/Z, [x27]
    addvl x27, x27, #2
    KAI_ASM_INST(0xa0402707)  // ldnt1h { z6.h-z7.h }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa0402529)  // ldnt1h { z8.h-z9.h }, pn9.b/Z, [x9]
    KAI_ASM_INST(0xc1509188)  // fdot za.s[x8, 0], { z12.h-z15.h }, z0.h[0]
    addvl x9, x9, #2
    KAI_ASM_INST(0xa04026eb)  // ldnt1h { z10.h-z11.h }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa040278d)  // ldnt1h { z12.h-z13.h }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xc1509089)  // fdot za.s[x8, 1], { z4.h-z7.h }, z0.h[0]
    addvl x28, x28, #2
    KAI_ASM_INST(0xa040274f)  // ldnt1h { z14.h-z15.h }, pn9.b/Z, [x26]
    addvl x26, x26, #2
    KAI_ASM_INST(0xc150910a)  // fdot za.s[x8, 2], { z8.h-z11.h }, z0.h[0]
    KAI_ASM_INST(0xa04025c9)  // ldnt1h { z8.h-z9.h }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa040272b)  // ldnt1h { z10.h-z11.h }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xc150918b)  // fdot za.s[x8, 3], { z12.h-z15.h }, z0.h[0]
    KAI_ASM_INST(0xa0402765)  // ldnt1h { z4.h-z5.h }, pn9.b/Z, [x27]
    addvl x27, x27, #2
    KAI_ASM_INST(0xa0402707)  // ldnt1h { z6.h-z7.h }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa040252d)  // ldnt1h { z12.h-z13.h }, pn9.b/Z, [x9]
    KAI_ASM_INST(0xc1509508)  // fdot za.s[x8, 0], { z8.h-z11.h }, z0.h[1]
    addvl x9, x9, #2
    KAI_ASM_INST(0xa04026ef)  // ldnt1h { z14.h-z15.h }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa0402789)  // ldnt1h { z8.h-z9.h }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xc1509489)  // fdot za.s[x8, 1], { z4.h-z7.h }, z0.h[1]
    addvl x28, x28, #2
    KAI_ASM_INST(0xa040274b)  // ldnt1h { z10.h-z11.h }, pn9.b/Z, [x26]
    addvl x26, x26, #2
    KAI_ASM_INST(0xc150958a)  // fdot za.s[x8, 2], { z12.h-z15.h }, z0.h[1]
    KAI_ASM_INST(0xa04025cd)  // ldnt1h { z12.h-z13.h }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa040272f)  // ldnt1h { z14.h-z15.h }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xc150950b)  // fdot za.s[x8, 3], { z8.h-z11.h }, z0.h[1]
    KAI_ASM_INST(0xa0402765)  // ldnt1h { z4.h-z5.h }, pn9.b/Z, [x27]
    addvl x27, x27, #2
    KAI_ASM_INST(0xa0402707)  // ldnt1h { z6.h-z7.h }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa0402529)  // ldnt1h { z8.h-z9.h }, pn9.b/Z, [x9]
    KAI_ASM_INST(0xc1509988)  // fdot za.s[x8, 0], { z12.h-z15.h }, z0.h[2]
    addvl x9, x9, #2
    KAI_ASM_INST(0xa04026eb)  // ldnt1h { z10.h-z11.h }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa040278d)  // ldnt1h { z12.h-z13.h }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xc1509889)  // fdot za.s[x8, 1], { z4.h-z7.h }, z0.h[2]
    addvl x28, x28, #2
    KAI_ASM_INST(0xa040274f)  // ldnt1h { z14.h-z15.h }, pn9.b/Z, [x26]
    addvl x26, x26, #2
    KAI_ASM_INST(0xc150990a)  // fdot za.s[x8, 2], { z8.h-z11.h }, z0.h[2]
    KAI_ASM_INST(0xa04025dd)  // ldnt1h { z28.h-z29.h }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa040273f)  // ldnt1h { z30.h-z31.h }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xc150998b)  // fdot za.s[x8, 3], { z12.h-z15.h }, z0.h[2]
    KAI_ASM_INST(0xa0402769)  // ldnt1h { z8.h-z9.h }, pn9.b/Z, [x27]
    addvl x27, x27, #2
    KAI_ASM_INST(0xa040270b)  // ldnt1h { z10.h-z11.h }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa0402535)  // ldnt1h { z20.h-z21.h }, pn9.b/Z, [x9]
    KAI_ASM_INST(0xc1509f88)  // fdot za.s[x8, 0], { z28.h-z31.h }, z0.h[3]
    addvl x9, x9, #2
    KAI_ASM_INST(0xa04026f7)  // ldnt1h { z22.h-z23.h }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa0402791)  // ldnt1h { z16.h-z17.h }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xc1509d09)  // fdot za.s[x8, 1], { z8.h-z11.h }, z0.h[3]
    addvl x28, x28, #2
    KAI_ASM_INST(0xa0402753)  // ldnt1h { z18.h-z19.h }, pn9.b/Z, [x26]
    addvl x26, x26, #2
    KAI_ASM_INST(0xc1509e8a)  // fdot za.s[x8, 2], { z20.h-z23.h }, z0.h[3]
    KAI_ASM_INST(0xc1509e0b)  // fdot za.s[x8, 3], { z16.h-z19.h }, z0.h[3]
    bgt label_23
KAI_ASM_LABEL(label_24)  // Width 4: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x11
    KAI_ASM_INST(0xa04025d5)  // ldnt1h { z20.h-z21.h }, pn9.b/Z, [x14]
    subs x11, x11, #0x2
    ld1rqh { z3.h }, p0/Z, [x10]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa0402737)  // ldnt1h { z22.h-z23.h }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa040276d)  // ldnt1h { z12.h-z13.h }, pn9.b/Z, [x27]
    addvl x27, x27, #2
    KAI_ASM_INST(0xa040270f)  // ldnt1h { z14.h-z15.h }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa0402531)  // ldnt1h { z16.h-z17.h }, pn9.b/Z, [x9]
    KAI_ASM_INST(0xc1539288)  // fdot za.s[x8, 0], { z20.h-z23.h }, z3.h[0]
    addvl x9, x9, #2
    KAI_ASM_INST(0xa04026f3)  // ldnt1h { z18.h-z19.h }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa040279d)  // ldnt1h { z28.h-z29.h }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xc1539189)  // fdot za.s[x8, 1], { z12.h-z15.h }, z3.h[0]
    addvl x28, x28, #2
    KAI_ASM_INST(0xa040275f)  // ldnt1h { z30.h-z31.h }, pn9.b/Z, [x26]
    addvl x26, x26, #2
    KAI_ASM_INST(0xc153920a)  // fdot za.s[x8, 2], { z16.h-z19.h }, z3.h[0]
    KAI_ASM_INST(0xc153938b)  // fdot za.s[x8, 3], { z28.h-z31.h }, z3.h[0]
    ble label_25
    KAI_ASM_INST(0xa04025c9)  // ldnt1h { z8.h-z9.h }, pn9.b/Z, [x14]
    subs x11, x11, #0x2
    addvl x14, x14, #2
    KAI_ASM_INST(0xa040272b)  // ldnt1h { z10.h-z11.h }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa0402765)  // ldnt1h { z4.h-z5.h }, pn9.b/Z, [x27]
    addvl x27, x27, #2
    KAI_ASM_INST(0xa0402707)  // ldnt1h { z6.h-z7.h }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa040252d)  // ldnt1h { z12.h-z13.h }, pn9.b/Z, [x9]
    KAI_ASM_INST(0xc1539508)  // fdot za.s[x8, 0], { z8.h-z11.h }, z3.h[1]
    addvl x9, x9, #2
    KAI_ASM_INST(0xa04026ef)  // ldnt1h { z14.h-z15.h }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa0402795)  // ldnt1h { z20.h-z21.h }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xc1539489)  // fdot za.s[x8, 1], { z4.h-z7.h }, z3.h[1]
    addvl x28, x28, #2
    KAI_ASM_INST(0xa0402757)  // ldnt1h { z22.h-z23.h }, pn9.b/Z, [x26]
    addvl x26, x26, #2
    KAI_ASM_INST(0xc153958a)  // fdot za.s[x8, 2], { z12.h-z15.h }, z3.h[1]
    KAI_ASM_INST(0xc153968b)  // fdot za.s[x8, 3], { z20.h-z23.h }, z3.h[1]
    ble label_25
    KAI_ASM_INST(0xa04025cd)  // ldnt1h { z12.h-z13.h }, pn9.b/Z, [x14]
    subs x11, x11, #0x2
    addvl x14, x14, #2
    KAI_ASM_INST(0xa040272f)  // ldnt1h { z14.h-z15.h }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa0402769)  // ldnt1h { z8.h-z9.h }, pn9.b/Z, [x27]
    addvl x27, x27, #2
    KAI_ASM_INST(0xa040270b)  // ldnt1h { z10.h-z11.h }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa0402535)  // ldnt1h { z20.h-z21.h }, pn9.b/Z, [x9]
    KAI_ASM_INST(0xc1539988)  // fdot za.s[x8, 0], { z12.h-z15.h }, z3.h[2]
    addvl x9, x9, #2
    KAI_ASM_INST(0xa04026f7)  // ldnt1h { z22.h-z23.h }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa0402791)  // ldnt1h { z16.h-z17.h }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xc1539909)  // fdot za.s[x8, 1], { z8.h-z11.h }, z3.h[2]
    addvl x28, x28, #2
    KAI_ASM_INST(0xa0402753)  // ldnt1h { z18.h-z19.h }, pn9.b/Z, [x26]
    addvl x26, x26, #2
    KAI_ASM_INST(0xc1539a8a)  // fdot za.s[x8, 2], { z20.h-z23.h }, z3.h[2]
    KAI_ASM_INST(0xc1539a0b)  // fdot za.s[x8, 3], { z16.h-z19.h }, z3.h[2]
    ble label_25
    KAI_ASM_INST(0xa04025c5)  // ldnt1h { z4.h-z5.h }, pn9.b/Z, [x14]
    KAI_ASM_INST(0xa0402727)  // ldnt1h { z6.h-z7.h }, pn9.b/Z, [x25]
    KAI_ASM_INST(0xa040276d)  // ldnt1h { z12.h-z13.h }, pn9.b/Z, [x27]
    KAI_ASM_INST(0xa040270f)  // ldnt1h { z14.h-z15.h }, pn9.b/Z, [x24]
    KAI_ASM_INST(0xa0402531)  // ldnt1h { z16.h-z17.h }, pn9.b/Z, [x9]
    KAI_ASM_INST(0xc1539c88)  // fdot za.s[x8, 0], { z4.h-z7.h }, z3.h[3]
    KAI_ASM_INST(0xa04026f3)  // ldnt1h { z18.h-z19.h }, pn9.b/Z, [x23]
    KAI_ASM_INST(0xa0402795)  // ldnt1h { z20.h-z21.h }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xc1539d89)  // fdot za.s[x8, 1], { z12.h-z15.h }, z3.h[3]
    KAI_ASM_INST(0xa0402757)  // ldnt1h { z22.h-z23.h }, pn9.b/Z, [x26]
    KAI_ASM_INST(0xc1539e0a)  // fdot za.s[x8, 2], { z16.h-z19.h }, z3.h[3]
    KAI_ASM_INST(0xc1539e8b)  // fdot za.s[x8, 3], { z20.h-z23.h }, z3.h[3]
KAI_ASM_LABEL(label_25)  // Width 4: Multiply loop: multiply skip
    tbz x15, #1, label_26
    KAI_ASM_INST(0xc0060c1c)  // mova { z28.d-z31.d }, za.d[x8, #0]
    add x21, x0, #0x2
    add x20, x0, #0x0
    KAI_ASM_INST(0xc0060c2c)  // mova { z12.d-z15.d }, za.d[x8, #1]
    KAI_ASM_INST(0x84c0a6b3)  // ld1rh { z19.h }, p1/Z, [x21]
    KAI_ASM_INST(0xc0060c40)  // mova { z0.d-z3.d }, za.d[x8, #2]
    KAI_ASM_INST(0x84c0a692)  // ld1rh { z18.h }, p1/Z, [x20]
    KAI_ASM_INST(0xc0060c64)  // mova { z4.d-z7.d }, za.d[x8, #3]
    KAI_ASM_INST(0xc120e38a)  // fcvt z10.h, { z28.s-z29.s }
    KAI_ASM_INST(0xc120e3cb)  // fcvt z11.h, { z30.s-z31.s }
    KAI_ASM_INST(0xc120e18c)  // fcvt z12.h, { z12.s-z13.s }
    KAI_ASM_INST(0xc120e1cd)  // fcvt z13.h, { z14.s-z15.s }
    KAI_ASM_INST(0xc172c26a)  // fclamp { z10.h-z11.h }, z19.h, z18.h
    KAI_ASM_INST(0xc120e00e)  // fcvt z14.h, { z0.s-z1.s }
    KAI_ASM_INST(0xc120e04f)  // fcvt z15.h, { z2.s-z3.s }
    KAI_ASM_INST(0xc172c26c)  // fclamp { z12.h-z13.h }, z19.h, z18.h
    KAI_ASM_INST(0xc120e090)  // fcvt z16.h, { z4.s-z5.s }
    KAI_ASM_INST(0xc120e0d1)  // fcvt z17.h, { z6.s-z7.s }
    KAI_ASM_INST(0xc172c26e)  // fclamp { z14.h-z15.h }, z19.h, z18.h
    KAI_ASM_INST(0xc172c270)  // fclamp { z16.h-z17.h }, z19.h, z18.h
    KAI_ASM_INST(0xa06025aa)  // st1h { z10.h-z11.h }, pn9.b, [x13]
    KAI_ASM_INST(0xa06125ac)  // st1h { z12.h-z13.h }, pn9.b, [x13, #0x2, MUL VL]
    KAI_ASM_INST(0xa06225ae)  // st1h { z14.h-z15.h }, pn9.b, [x13, #0x4, MUL VL]
    KAI_ASM_INST(0xa06321b0)  // st1h { z16.h-z17.h }, p8, [x13, #0x6, MUL VL]
    addvl x13, x13, #8
    b label_27
KAI_ASM_LABEL(label_26)  // Width 4: No activation
    KAI_ASM_INST(0xc0060c0c)  // mova { z12.d-z15.d }, za.d[x8, #0]
    KAI_ASM_INST(0xc0060c30)  // mova { z16.d-z19.d }, za.d[x8, #1]
    KAI_ASM_INST(0xc0060c5c)  // mova { z28.d-z31.d }, za.d[x8, #2]
    KAI_ASM_INST(0xc0060c68)  // mova { z8.d-z11.d }, za.d[x8, #3]
    KAI_ASM_INST(0xc120e187)  // fcvt z7.h, { z12.s-z13.s }
    KAI_ASM_INST(0xc120e1cf)  // fcvt z15.h, { z14.s-z15.s }
    KAI_ASM_INST(0xa16025a7)  // st1h { z7.h, z15.h }, pn9.b, [x13]
    KAI_ASM_INST(0xc120e207)  // fcvt z7.h, { z16.s-z17.s }
    KAI_ASM_INST(0xc120e24f)  // fcvt z15.h, { z18.s-z19.s }
    KAI_ASM_INST(0xa16125a7)  // st1h { z7.h, z15.h }, pn9.b, [x13, #0x2, MUL VL]
    KAI_ASM_INST(0xc120e38e)  // fcvt z14.h, { z28.s-z29.s }
    KAI_ASM_INST(0xc120e3cf)  // fcvt z15.h, { z30.s-z31.s }
    KAI_ASM_INST(0xa06225ae)  // st1h { z14.h-z15.h }, pn9.b, [x13, #0x4, MUL VL]
    KAI_ASM_INST(0xc120e112)  // fcvt z18.h, { z8.s-z9.s }
    KAI_ASM_INST(0xc120e15a)  // fcvt z26.h, { z10.s-z11.s }
    KAI_ASM_INST(0xa16321b2)  // st1h { z18.h, z26.h }, p8, [x13, #0x6, MUL VL]
    addvl x13, x13, #8
KAI_ASM_LABEL(label_27)  // Width 4: Output done
    subs x17, x17, #0x4
    mov x14, x22
    sub x5, x5, x6, LSL #2
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot)

    KAI_ASM_END
