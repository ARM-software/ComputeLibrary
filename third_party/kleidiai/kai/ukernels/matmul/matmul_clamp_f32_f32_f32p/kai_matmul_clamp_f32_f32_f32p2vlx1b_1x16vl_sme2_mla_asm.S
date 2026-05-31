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

    KAI_ASM_CODE(matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla)
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
    KAI_ASM_INST(0xf8b549da)  // rprfm pldonce, x21, [x14]
KAI_ASM_LABEL(label_3)  // RHS prefetch exit
    mov x12, x7
    cntw x20, ALL, MUL #2
    lsl x12, x12, #0x2
    add x12, x12, #0x4
    mul x12, x12, x20
KAI_ASM_LABEL(label_4)  // Column loop
    cmp x17, #0x4
    bge label_22
    cmp x17, #0x2
    bgt label_16
    beq label_10
    cntw x20, ALL, MUL #2
    add x22, x14, x12
    KAI_ASM_INST(0xa04045d4)  // ld1w { z20.s-z21.s }, pn9.b/Z, [x14]
    cmp x5, x20
    mov x11, x7
    csel x22, x22, x14, GT
    mov x21, x5
    KAI_ASM_INST(0xa04046d6)  // ld1w { z22.s-z23.s }, pn9.b/Z, [x22]
    mov x10, x16
    lsl x20, x7, #0x2
    KAI_ASM_INST(0x25b567f0)  // whilelt p8.s, XZR, x21, VLx4
    cmp x11, #0x4
    KAI_ASM_INST(0xf8b44958)  // rprfm pldmany, x20, [x10]
    addvl x14, x14, #2
    addvl x22, x22, #2
    KAI_ASM_INST(0xc0040e80)  // mova za.d[x8, #0], { z20.d-z23.d }
    ble label_6
KAI_ASM_LABEL(label_5)  // Width 1: Multiply loop: Main loop head
    whilelt p0.s, XZR, x11
    KAI_ASM_INST(0xa04045c5)  // ldnt1w { z4.s-z5.s }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    ld1rqw { z15.s }, p0/Z, [x10]
    sub x11, x11, #0x4
    add x10, x10, #0x10
    KAI_ASM_INST(0xa04046c7)  // ldnt1w { z6.s-z7.s }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    cmp x11, #0x4
    KAI_ASM_INST(0xa04045dd)  // ldnt1w { z28.s-z29.s }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04046df)  // ldnt1w { z30.s-z31.s }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xc15f8080)  // fmla za.s[x8, 0], { z4.s-z7.s }, z15.s[0]
    KAI_ASM_INST(0xa04045c1)  // ldnt1w { z0.s-z1.s }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04046c3)  // ldnt1w { z2.s-z3.s }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa04045d5)  // ldnt1w { z20.s-z21.s }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04046d7)  // ldnt1w { z22.s-z23.s }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xc15f8780)  // fmla za.s[x8, 0], { z28.s-z31.s }, z15.s[1]
    KAI_ASM_INST(0xc15f8800)  // fmla za.s[x8, 0], { z0.s-z3.s }, z15.s[2]
    KAI_ASM_INST(0xc15f8e80)  // fmla za.s[x8, 0], { z20.s-z23.s }, z15.s[3]
    bgt label_5
KAI_ASM_LABEL(label_6)  // Width 1: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x11
    KAI_ASM_INST(0xa04045c1)  // ldnt1w { z0.s-z1.s }, pn9.b/Z, [x14]
    subs x11, x11, #0x1
    ld1rqw { z8.s }, p0/Z, [x10]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04046c3)  // ldnt1w { z2.s-z3.s }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xc1588000)  // fmla za.s[x8, 0], { z0.s-z3.s }, z8.s[0]
    ble label_7
    KAI_ASM_INST(0xa04045d1)  // ldnt1w { z16.s-z17.s }, pn9.b/Z, [x14]
    subs x11, x11, #0x1
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04046d3)  // ldnt1w { z18.s-z19.s }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xc1588600)  // fmla za.s[x8, 0], { z16.s-z19.s }, z8.s[1]
    ble label_7
    KAI_ASM_INST(0xa04045d5)  // ldnt1w { z20.s-z21.s }, pn9.b/Z, [x14]
    subs x11, x11, #0x1
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04046d7)  // ldnt1w { z22.s-z23.s }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xc1588a80)  // fmla za.s[x8, 0], { z20.s-z23.s }, z8.s[2]
    ble label_7
    KAI_ASM_INST(0xa04045cd)  // ldnt1w { z12.s-z13.s }, pn9.b/Z, [x14]
    KAI_ASM_INST(0xa04046cf)  // ldnt1w { z14.s-z15.s }, pn9.b/Z, [x22]
    KAI_ASM_INST(0xc1588d80)  // fmla za.s[x8, 0], { z12.s-z15.s }, z8.s[3]
KAI_ASM_LABEL(label_7)  // Width 1: Multiply loop: multiply skip
    tbz x15, #1, label_8
    add x21, x0, #0x4
    add x20, x0, #0x0
    KAI_ASM_INST(0xc0060c08)  // mova { z8.d-z11.d }, za.d[x8, #0]
    KAI_ASM_INST(0x8540c6b5)  // ld1rw { z21.s }, p1/Z, [x21]
    KAI_ASM_INST(0x8540c69d)  // ld1rw { z29.s }, p1/Z, [x20]
    KAI_ASM_INST(0xc1bdcaa8)  // fclamp { z8.s-z11.s }, z21.s, z29.s
    KAI_ASM_INST(0xa060c1a8)  // st1w { z8.s-z11.s }, p8, [x13]
    b label_9
KAI_ASM_LABEL(label_8)  // Width 1: No activation
    KAI_ASM_INST(0xc0060c08)  // mova { z8.d-z11.d }, za.d[x8, #0]
    KAI_ASM_INST(0xa060c1a8)  // st1w { z8.s-z11.s }, p8, [x13]
KAI_ASM_LABEL(label_9)  // Width 1: Output done
    b label_28
KAI_ASM_LABEL(label_10)  // Width 2
    add x24, x14, x12, LSL #1
    cntw x20, ALL, MUL #6
    KAI_ASM_INST(0xa04045c4)  // ld1w { z4.s-z5.s }, pn9.b/Z, [x14]
    add x23, x24, x12
    cmp x5, x20
    KAI_ASM_INST(0xa0404700)  // ld1w { z0.s-z1.s }, pn9.b/Z, [x24]
    add x22, x14, x12
    csel x23, x23, x14, GT
    KAI_ASM_INST(0xa04046c6)  // ld1w { z6.s-z7.s }, pn9.b/Z, [x22]
    mov x11, x7
    sub x21, x5, x6
    KAI_ASM_INST(0xa04046e2)  // ld1w { z2.s-z3.s }, pn9.b/Z, [x23]
    mov x10, x16
    lsl x20, x7, #0x2
    KAI_ASM_INST(0x25b567f0)  // whilelt p8.s, XZR, x21, VLx4
    cmp x11, #0x4
    KAI_ASM_INST(0xf8b44958)  // rprfm pldmany, x20, [x10]
    addvl x14, x14, #2
    KAI_ASM_INST(0xc0040c80)  // mova za.d[x8, #0], { z4.d-z7.d }
    addvl x22, x22, #2
    addvl x24, x24, #2
    KAI_ASM_INST(0xc0040c01)  // mova za.d[x8, #1], { z0.d-z3.d }
    addvl x23, x23, #2
    ble label_12
KAI_ASM_LABEL(label_11)  // Width 2: Multiply loop: Main loop head
    whilelt p0.s, XZR, x11
    KAI_ASM_INST(0xa04045c5)  // ldnt1w { z4.s-z5.s }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    ld1rqw { z0.s }, p0/Z, [x10]
    sub x11, x11, #0x4
    add x10, x10, #0x10
    KAI_ASM_INST(0xa04046c7)  // ldnt1w { z6.s-z7.s }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    cmp x11, #0x4
    KAI_ASM_INST(0xa0404715)  // ldnt1w { z20.s-z21.s }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa04046f7)  // ldnt1w { z22.s-z23.s }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc1508080)  // fmla za.s[x8, 0], { z4.s-z7.s }, z0.s[0]
    KAI_ASM_INST(0xa04045d9)  // ldnt1w { z24.s-z25.s }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04046db)  // ldnt1w { z26.s-z27.s }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xc1508281)  // fmla za.s[x8, 1], { z20.s-z23.s }, z0.s[0]
    KAI_ASM_INST(0xa0404709)  // ldnt1w { z8.s-z9.s }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa04046eb)  // ldnt1w { z10.s-z11.s }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc1508700)  // fmla za.s[x8, 0], { z24.s-z27.s }, z0.s[1]
    KAI_ASM_INST(0xa04045dd)  // ldnt1w { z28.s-z29.s }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04046df)  // ldnt1w { z30.s-z31.s }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xc1508501)  // fmla za.s[x8, 1], { z8.s-z11.s }, z0.s[1]
    KAI_ASM_INST(0xa0404709)  // ldnt1w { z8.s-z9.s }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa04046eb)  // ldnt1w { z10.s-z11.s }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc1508b80)  // fmla za.s[x8, 0], { z28.s-z31.s }, z0.s[2]
    KAI_ASM_INST(0xa04045d9)  // ldnt1w { z24.s-z25.s }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04046db)  // ldnt1w { z26.s-z27.s }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xc1508901)  // fmla za.s[x8, 1], { z8.s-z11.s }, z0.s[2]
    KAI_ASM_INST(0xa040470d)  // ldnt1w { z12.s-z13.s }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa04046ef)  // ldnt1w { z14.s-z15.s }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc1508f00)  // fmla za.s[x8, 0], { z24.s-z27.s }, z0.s[3]
    KAI_ASM_INST(0xc1508d81)  // fmla za.s[x8, 1], { z12.s-z15.s }, z0.s[3]
    bgt label_11
KAI_ASM_LABEL(label_12)  // Width 2: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x11
    KAI_ASM_INST(0xa04045c5)  // ldnt1w { z4.s-z5.s }, pn9.b/Z, [x14]
    subs x11, x11, #0x1
    ld1rqw { z8.s }, p0/Z, [x10]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04046c7)  // ldnt1w { z6.s-z7.s }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0404715)  // ldnt1w { z20.s-z21.s }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa04046f7)  // ldnt1w { z22.s-z23.s }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc1588080)  // fmla za.s[x8, 0], { z4.s-z7.s }, z8.s[0]
    KAI_ASM_INST(0xc1588281)  // fmla za.s[x8, 1], { z20.s-z23.s }, z8.s[0]
    ble label_13
    KAI_ASM_INST(0xa04045cd)  // ldnt1w { z12.s-z13.s }, pn9.b/Z, [x14]
    subs x11, x11, #0x1
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04046cf)  // ldnt1w { z14.s-z15.s }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa040471d)  // ldnt1w { z28.s-z29.s }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa04046ff)  // ldnt1w { z30.s-z31.s }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc1588580)  // fmla za.s[x8, 0], { z12.s-z15.s }, z8.s[1]
    KAI_ASM_INST(0xc1588781)  // fmla za.s[x8, 1], { z28.s-z31.s }, z8.s[1]
    ble label_13
    KAI_ASM_INST(0xa04045dd)  // ldnt1w { z28.s-z29.s }, pn9.b/Z, [x14]
    subs x11, x11, #0x1
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04046df)  // ldnt1w { z30.s-z31.s }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0404701)  // ldnt1w { z0.s-z1.s }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa04046e3)  // ldnt1w { z2.s-z3.s }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc1588b80)  // fmla za.s[x8, 0], { z28.s-z31.s }, z8.s[2]
    KAI_ASM_INST(0xc1588801)  // fmla za.s[x8, 1], { z0.s-z3.s }, z8.s[2]
    ble label_13
    KAI_ASM_INST(0xa04045d5)  // ldnt1w { z20.s-z21.s }, pn9.b/Z, [x14]
    KAI_ASM_INST(0xa04046d7)  // ldnt1w { z22.s-z23.s }, pn9.b/Z, [x22]
    KAI_ASM_INST(0xa0404701)  // ldnt1w { z0.s-z1.s }, pn9.b/Z, [x24]
    KAI_ASM_INST(0xa04046e3)  // ldnt1w { z2.s-z3.s }, pn9.b/Z, [x23]
    KAI_ASM_INST(0xc1588e80)  // fmla za.s[x8, 0], { z20.s-z23.s }, z8.s[3]
    KAI_ASM_INST(0xc1588c01)  // fmla za.s[x8, 1], { z0.s-z3.s }, z8.s[3]
KAI_ASM_LABEL(label_13)  // Width 2: Multiply loop: multiply skip
    tbz x15, #1, label_14
    add x21, x0, #0x4
    add x20, x0, #0x0
    KAI_ASM_INST(0xc0060c1c)  // mova { z28.d-z31.d }, za.d[x8, #0]
    KAI_ASM_INST(0xc0060c24)  // mova { z4.d-z7.d }, za.d[x8, #1]
    KAI_ASM_INST(0x8540c6b1)  // ld1rw { z17.s }, p1/Z, [x21]
    KAI_ASM_INST(0x8540c689)  // ld1rw { z9.s }, p1/Z, [x20]
    KAI_ASM_INST(0xc1a9ca3c)  // fclamp { z28.s-z31.s }, z17.s, z9.s
    KAI_ASM_INST(0xc1a9ca24)  // fclamp { z4.s-z7.s }, z17.s, z9.s
    KAI_ASM_INST(0xa060c5bc)  // st1w { z28.s-z31.s }, pn9.b, [x13]
    KAI_ASM_INST(0xa061c1a4)  // st1w { z4.s-z7.s }, p8, [x13, #0x4, MUL VL]
    b label_15
KAI_ASM_LABEL(label_14)  // Width 2: No activation
    KAI_ASM_INST(0xc0060c04)  // mova { z4.d-z7.d }, za.d[x8, #0]
    KAI_ASM_INST(0xc0060c3c)  // mova { z28.d-z31.d }, za.d[x8, #1]
    KAI_ASM_INST(0xa060c5a4)  // st1w { z4.s-z7.s }, pn9.b, [x13]
    KAI_ASM_INST(0xa061c1bc)  // st1w { z28.s-z31.s }, p8, [x13, #0x4, MUL VL]
KAI_ASM_LABEL(label_15)  // Width 2: Output done
    b label_28
KAI_ASM_LABEL(label_16)  // Width 3
    add x26, x14, x12, LSL #2
    cntw x20, ALL, MUL #10
    KAI_ASM_INST(0xa04045d4)  // ld1w { z20.s-z21.s }, pn9.b/Z, [x14]
    add x25, x14, x12, LSL #1
    add x24, x26, x12
    KAI_ASM_INST(0xa0404740)  // ld1w { z0.s-z1.s }, pn9.b/Z, [x26]
    cmp x5, x20
    add x23, x14, x12
    KAI_ASM_INST(0xa0404730)  // ld1w { z16.s-z17.s }, pn9.b/Z, [x25]
    add x22, x25, x12
    csel x24, x24, x14, GT
    KAI_ASM_INST(0xa04046f6)  // ld1w { z22.s-z23.s }, pn9.b/Z, [x23]
    mov x20, #0x2
    KAI_ASM_INST(0xa04046d2)  // ld1w { z18.s-z19.s }, pn9.b/Z, [x22]
    mov x11, x7
    KAI_ASM_INST(0xa0404702)  // ld1w { z2.s-z3.s }, pn9.b/Z, [x24]
    msub x21, x6, x20, x5
    mov x10, x16
    lsl x20, x7, #0x2
    KAI_ASM_INST(0x25b567f0)  // whilelt p8.s, XZR, x21, VLx4
    KAI_ASM_INST(0xc0040e80)  // mova za.d[x8, #0], { z20.d-z23.d }
    cmp x11, #0x4
    KAI_ASM_INST(0xf8b44958)  // rprfm pldmany, x20, [x10]
    KAI_ASM_INST(0xc0040e01)  // mova za.d[x8, #1], { z16.d-z19.d }
    addvl x14, x14, #2
    addvl x23, x23, #2
    KAI_ASM_INST(0xc0040c02)  // mova za.d[x8, #2], { z0.d-z3.d }
    addvl x25, x25, #2
    addvl x22, x22, #2
    addvl x26, x26, #2
    addvl x24, x24, #2
    ble label_18
KAI_ASM_LABEL(label_17)  // Width 3: Multiply loop: Main loop head
    whilelt p0.s, XZR, x11
    KAI_ASM_INST(0xa04045cd)  // ldnt1w { z12.s-z13.s }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    ld1rqw { z3.s }, p0/Z, [x10]
    sub x11, x11, #0x4
    add x10, x10, #0x10
    KAI_ASM_INST(0xa04046ef)  // ldnt1w { z14.s-z15.s }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    cmp x11, #0x4
    KAI_ASM_INST(0xa0404729)  // ldnt1w { z8.s-z9.s }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa04046cb)  // ldnt1w { z10.s-z11.s }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0404751)  // ldnt1w { z16.s-z17.s }, pn9.b/Z, [x26]
    KAI_ASM_INST(0xc1538180)  // fmla za.s[x8, 0], { z12.s-z15.s }, z3.s[0]
    addvl x26, x26, #2
    KAI_ASM_INST(0xa0404713)  // ldnt1w { z18.s-z19.s }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xc1538101)  // fmla za.s[x8, 1], { z8.s-z11.s }, z3.s[0]
    KAI_ASM_INST(0xa04045c9)  // ldnt1w { z8.s-z9.s }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04046eb)  // ldnt1w { z10.s-z11.s }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc1538202)  // fmla za.s[x8, 2], { z16.s-z19.s }, z3.s[0]
    KAI_ASM_INST(0xa0404731)  // ldnt1w { z16.s-z17.s }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa04046d3)  // ldnt1w { z18.s-z19.s }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0404745)  // ldnt1w { z4.s-z5.s }, pn9.b/Z, [x26]
    KAI_ASM_INST(0xc1538500)  // fmla za.s[x8, 0], { z8.s-z11.s }, z3.s[1]
    addvl x26, x26, #2
    KAI_ASM_INST(0xa0404707)  // ldnt1w { z6.s-z7.s }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xc1538601)  // fmla za.s[x8, 1], { z16.s-z19.s }, z3.s[1]
    KAI_ASM_INST(0xa04045c9)  // ldnt1w { z8.s-z9.s }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04046eb)  // ldnt1w { z10.s-z11.s }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc1538482)  // fmla za.s[x8, 2], { z4.s-z7.s }, z3.s[1]
    KAI_ASM_INST(0xa0404731)  // ldnt1w { z16.s-z17.s }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa04046d3)  // ldnt1w { z18.s-z19.s }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0404745)  // ldnt1w { z4.s-z5.s }, pn9.b/Z, [x26]
    KAI_ASM_INST(0xc1538900)  // fmla za.s[x8, 0], { z8.s-z11.s }, z3.s[2]
    addvl x26, x26, #2
    KAI_ASM_INST(0xa0404707)  // ldnt1w { z6.s-z7.s }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xc1538a01)  // fmla za.s[x8, 1], { z16.s-z19.s }, z3.s[2]
    KAI_ASM_INST(0xa04045d5)  // ldnt1w { z20.s-z21.s }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04046f7)  // ldnt1w { z22.s-z23.s }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc1538882)  // fmla za.s[x8, 2], { z4.s-z7.s }, z3.s[2]
    KAI_ASM_INST(0xa0404739)  // ldnt1w { z24.s-z25.s }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa04046db)  // ldnt1w { z26.s-z27.s }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0404751)  // ldnt1w { z16.s-z17.s }, pn9.b/Z, [x26]
    KAI_ASM_INST(0xc1538e80)  // fmla za.s[x8, 0], { z20.s-z23.s }, z3.s[3]
    addvl x26, x26, #2
    KAI_ASM_INST(0xa0404713)  // ldnt1w { z18.s-z19.s }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xc1538f01)  // fmla za.s[x8, 1], { z24.s-z27.s }, z3.s[3]
    KAI_ASM_INST(0xc1538e02)  // fmla za.s[x8, 2], { z16.s-z19.s }, z3.s[3]
    bgt label_17
KAI_ASM_LABEL(label_18)  // Width 3: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x11
    KAI_ASM_INST(0xa04045c5)  // ldnt1w { z4.s-z5.s }, pn9.b/Z, [x14]
    subs x11, x11, #0x1
    ld1rqw { z8.s }, p0/Z, [x10]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04046e7)  // ldnt1w { z6.s-z7.s }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa040473d)  // ldnt1w { z28.s-z29.s }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa04046df)  // ldnt1w { z30.s-z31.s }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0404755)  // ldnt1w { z20.s-z21.s }, pn9.b/Z, [x26]
    KAI_ASM_INST(0xc1588080)  // fmla za.s[x8, 0], { z4.s-z7.s }, z8.s[0]
    addvl x26, x26, #2
    KAI_ASM_INST(0xa0404717)  // ldnt1w { z22.s-z23.s }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xc1588381)  // fmla za.s[x8, 1], { z28.s-z31.s }, z8.s[0]
    KAI_ASM_INST(0xc1588282)  // fmla za.s[x8, 2], { z20.s-z23.s }, z8.s[0]
    ble label_19
    KAI_ASM_INST(0xa04045cd)  // ldnt1w { z12.s-z13.s }, pn9.b/Z, [x14]
    subs x11, x11, #0x1
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04046ef)  // ldnt1w { z14.s-z15.s }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa0404725)  // ldnt1w { z4.s-z5.s }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa04046c7)  // ldnt1w { z6.s-z7.s }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0404751)  // ldnt1w { z16.s-z17.s }, pn9.b/Z, [x26]
    KAI_ASM_INST(0xc1588580)  // fmla za.s[x8, 0], { z12.s-z15.s }, z8.s[1]
    addvl x26, x26, #2
    KAI_ASM_INST(0xa0404713)  // ldnt1w { z18.s-z19.s }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xc1588481)  // fmla za.s[x8, 1], { z4.s-z7.s }, z8.s[1]
    KAI_ASM_INST(0xc1588602)  // fmla za.s[x8, 2], { z16.s-z19.s }, z8.s[1]
    ble label_19
    KAI_ASM_INST(0xa04045c1)  // ldnt1w { z0.s-z1.s }, pn9.b/Z, [x14]
    subs x11, x11, #0x1
    addvl x14, x14, #2
    KAI_ASM_INST(0xa04046e3)  // ldnt1w { z2.s-z3.s }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa040472d)  // ldnt1w { z12.s-z13.s }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa04046cf)  // ldnt1w { z14.s-z15.s }, pn9.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0404751)  // ldnt1w { z16.s-z17.s }, pn9.b/Z, [x26]
    KAI_ASM_INST(0xc1588800)  // fmla za.s[x8, 0], { z0.s-z3.s }, z8.s[2]
    addvl x26, x26, #2
    KAI_ASM_INST(0xa0404713)  // ldnt1w { z18.s-z19.s }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xc1588981)  // fmla za.s[x8, 1], { z12.s-z15.s }, z8.s[2]
    KAI_ASM_INST(0xc1588a02)  // fmla za.s[x8, 2], { z16.s-z19.s }, z8.s[2]
    ble label_19
    KAI_ASM_INST(0xa04045c5)  // ldnt1w { z4.s-z5.s }, pn9.b/Z, [x14]
    KAI_ASM_INST(0xa04046e7)  // ldnt1w { z6.s-z7.s }, pn9.b/Z, [x23]
    KAI_ASM_INST(0xa040472d)  // ldnt1w { z12.s-z13.s }, pn9.b/Z, [x25]
    KAI_ASM_INST(0xa04046cf)  // ldnt1w { z14.s-z15.s }, pn9.b/Z, [x22]
    KAI_ASM_INST(0xa0404755)  // ldnt1w { z20.s-z21.s }, pn9.b/Z, [x26]
    KAI_ASM_INST(0xc1588c80)  // fmla za.s[x8, 0], { z4.s-z7.s }, z8.s[3]
    KAI_ASM_INST(0xa0404717)  // ldnt1w { z22.s-z23.s }, pn9.b/Z, [x24]
    KAI_ASM_INST(0xc1588d81)  // fmla za.s[x8, 1], { z12.s-z15.s }, z8.s[3]
    KAI_ASM_INST(0xc1588e82)  // fmla za.s[x8, 2], { z20.s-z23.s }, z8.s[3]
KAI_ASM_LABEL(label_19)  // Width 3: Multiply loop: multiply skip
    tbz x15, #1, label_20
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
    KAI_ASM_INST(0xa060c5a8)  // st1w { z8.s-z11.s }, pn9.b, [x13]
    KAI_ASM_INST(0xa061c5ac)  // st1w { z12.s-z15.s }, pn9.b, [x13, #0x4, MUL VL]
    KAI_ASM_INST(0xa062c1b0)  // st1w { z16.s-z19.s }, p8, [x13, #0x8, MUL VL]
    b label_21
KAI_ASM_LABEL(label_20)  // Width 3: No activation
    KAI_ASM_INST(0xc0060c04)  // mova { z4.d-z7.d }, za.d[x8, #0]
    KAI_ASM_INST(0xc0060c20)  // mova { z0.d-z3.d }, za.d[x8, #1]
    KAI_ASM_INST(0xc0060c50)  // mova { z16.d-z19.d }, za.d[x8, #2]
    KAI_ASM_INST(0xa060c5a4)  // st1w { z4.s-z7.s }, pn9.b, [x13]
    KAI_ASM_INST(0xa061c5a0)  // st1w { z0.s-z3.s }, pn9.b, [x13, #0x4, MUL VL]
    KAI_ASM_INST(0xa062c1b0)  // st1w { z16.s-z19.s }, p8, [x13, #0x8, MUL VL]
KAI_ASM_LABEL(label_21)  // Width 3: Output done
    b label_28
KAI_ASM_LABEL(label_22)  // Width 4
    add x9, x14, x12, LSL #2
    cntw x20, ALL, MUL #14
    KAI_ASM_INST(0xa04045cc)  // ld1w { z12.s-z13.s }, pn9.b/Z, [x14]
    add x28, x9, x12, LSL #1
    add x27, x14, x12, LSL #1
    KAI_ASM_INST(0xa0404528)  // ld1w { z8.s-z9.s }, pn9.b/Z, [x9]
    add x26, x28, x12
    cmp x5, x20
    KAI_ASM_INST(0xa0404760)  // ld1w { z0.s-z1.s }, pn9.b/Z, [x27]
    add x25, x14, x12
    add x24, x27, x12
    KAI_ASM_INST(0xa0404790)  // ld1w { z16.s-z17.s }, pn9.b/Z, [x28]
    add x23, x9, x12
    csel x26, x26, x14, GT
    KAI_ASM_INST(0xa040472e)  // ld1w { z14.s-z15.s }, pn9.b/Z, [x25]
    mov x20, #0x3
    KAI_ASM_INST(0xa0404702)  // ld1w { z2.s-z3.s }, pn9.b/Z, [x24]
    mov x11, x7
    KAI_ASM_INST(0xa04046ea)  // ld1w { z10.s-z11.s }, pn9.b/Z, [x23]
    msub x21, x6, x20, x5
    mov x10, x16
    KAI_ASM_INST(0xa0404752)  // ld1w { z18.s-z19.s }, pn9.b/Z, [x26]
    lsl x20, x7, #0x2
    KAI_ASM_INST(0x25b567f0)  // whilelt p8.s, XZR, x21, VLx4
    KAI_ASM_INST(0xc0040d80)  // mova za.d[x8, #0], { z12.d-z15.d }
    cmp x11, #0x4
    KAI_ASM_INST(0xf8b44958)  // rprfm pldmany, x20, [x10]
    KAI_ASM_INST(0xc0040c01)  // mova za.d[x8, #1], { z0.d-z3.d }
    add x22, x14, x12, LSL #3
    addvl x14, x14, #2
    KAI_ASM_INST(0xc0040d02)  // mova za.d[x8, #2], { z8.d-z11.d }
    addvl x25, x25, #2
    addvl x27, x27, #2
    KAI_ASM_INST(0xc0040e03)  // mova za.d[x8, #3], { z16.d-z19.d }
    addvl x24, x24, #2
    addvl x9, x9, #2
    addvl x23, x23, #2
    addvl x28, x28, #2
    addvl x26, x26, #2
    ble label_24
KAI_ASM_LABEL(label_23)  // Width 4: Multiply loop: Main loop head
    whilelt p0.s, XZR, x11
    KAI_ASM_INST(0xa04045c9)  // ldnt1w { z8.s-z9.s }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    ld1rqw { z13.s }, p0/Z, [x10]
    sub x11, x11, #0x4
    add x10, x10, #0x10
    KAI_ASM_INST(0xa040472b)  // ldnt1w { z10.s-z11.s }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    cmp x11, #0x4
    KAI_ASM_INST(0xa0404765)  // ldnt1w { z4.s-z5.s }, pn9.b/Z, [x27]
    addvl x27, x27, #2
    KAI_ASM_INST(0xa0404707)  // ldnt1w { z6.s-z7.s }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa0404531)  // ldnt1w { z16.s-z17.s }, pn9.b/Z, [x9]
    KAI_ASM_INST(0xc15d8100)  // fmla za.s[x8, 0], { z8.s-z11.s }, z13.s[0]
    addvl x9, x9, #2
    KAI_ASM_INST(0xa04046f3)  // ldnt1w { z18.s-z19.s }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa0404781)  // ldnt1w { z0.s-z1.s }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xc15d8081)  // fmla za.s[x8, 1], { z4.s-z7.s }, z13.s[0]
    addvl x28, x28, #2
    KAI_ASM_INST(0xa0404743)  // ldnt1w { z2.s-z3.s }, pn9.b/Z, [x26]
    addvl x26, x26, #2
    KAI_ASM_INST(0xc15d8202)  // fmla za.s[x8, 2], { z16.s-z19.s }, z13.s[0]
    KAI_ASM_INST(0xa04045dd)  // ldnt1w { z28.s-z29.s }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa040473f)  // ldnt1w { z30.s-z31.s }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xc15d8003)  // fmla za.s[x8, 3], { z0.s-z3.s }, z13.s[0]
    KAI_ASM_INST(0xa0404761)  // ldnt1w { z0.s-z1.s }, pn9.b/Z, [x27]
    addvl x27, x27, #2
    KAI_ASM_INST(0xa0404703)  // ldnt1w { z2.s-z3.s }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa0404529)  // ldnt1w { z8.s-z9.s }, pn9.b/Z, [x9]
    KAI_ASM_INST(0xc15d8780)  // fmla za.s[x8, 0], { z28.s-z31.s }, z13.s[1]
    addvl x9, x9, #2
    KAI_ASM_INST(0xa04046eb)  // ldnt1w { z10.s-z11.s }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa0404791)  // ldnt1w { z16.s-z17.s }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xc15d8401)  // fmla za.s[x8, 1], { z0.s-z3.s }, z13.s[1]
    addvl x28, x28, #2
    KAI_ASM_INST(0xa0404753)  // ldnt1w { z18.s-z19.s }, pn9.b/Z, [x26]
    addvl x26, x26, #2
    KAI_ASM_INST(0xc15d8502)  // fmla za.s[x8, 2], { z8.s-z11.s }, z13.s[1]
    KAI_ASM_INST(0xa04045c5)  // ldnt1w { z4.s-z5.s }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa0404727)  // ldnt1w { z6.s-z7.s }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xc15d8603)  // fmla za.s[x8, 3], { z16.s-z19.s }, z13.s[1]
    KAI_ASM_INST(0xa0404761)  // ldnt1w { z0.s-z1.s }, pn9.b/Z, [x27]
    addvl x27, x27, #2
    KAI_ASM_INST(0xa0404703)  // ldnt1w { z2.s-z3.s }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa0404529)  // ldnt1w { z8.s-z9.s }, pn9.b/Z, [x9]
    KAI_ASM_INST(0xc15d8880)  // fmla za.s[x8, 0], { z4.s-z7.s }, z13.s[2]
    addvl x9, x9, #2
    KAI_ASM_INST(0xa04046eb)  // ldnt1w { z10.s-z11.s }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa0404791)  // ldnt1w { z16.s-z17.s }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xc15d8801)  // fmla za.s[x8, 1], { z0.s-z3.s }, z13.s[2]
    addvl x28, x28, #2
    KAI_ASM_INST(0xa0404753)  // ldnt1w { z18.s-z19.s }, pn9.b/Z, [x26]
    addvl x26, x26, #2
    KAI_ASM_INST(0xc15d8902)  // fmla za.s[x8, 2], { z8.s-z11.s }, z13.s[2]
    KAI_ASM_INST(0xa04045d5)  // ldnt1w { z20.s-z21.s }, pn9.b/Z, [x14]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa0404737)  // ldnt1w { z22.s-z23.s }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xc15d8a03)  // fmla za.s[x8, 3], { z16.s-z19.s }, z13.s[2]
    KAI_ASM_INST(0xa0404779)  // ldnt1w { z24.s-z25.s }, pn9.b/Z, [x27]
    addvl x27, x27, #2
    KAI_ASM_INST(0xa040471b)  // ldnt1w { z26.s-z27.s }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa0404529)  // ldnt1w { z8.s-z9.s }, pn9.b/Z, [x9]
    KAI_ASM_INST(0xc15d8e80)  // fmla za.s[x8, 0], { z20.s-z23.s }, z13.s[3]
    addvl x9, x9, #2
    KAI_ASM_INST(0xa04046eb)  // ldnt1w { z10.s-z11.s }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa0404795)  // ldnt1w { z20.s-z21.s }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xc15d8f01)  // fmla za.s[x8, 1], { z24.s-z27.s }, z13.s[3]
    addvl x28, x28, #2
    KAI_ASM_INST(0xa0404757)  // ldnt1w { z22.s-z23.s }, pn9.b/Z, [x26]
    addvl x26, x26, #2
    KAI_ASM_INST(0xc15d8d02)  // fmla za.s[x8, 2], { z8.s-z11.s }, z13.s[3]
    KAI_ASM_INST(0xc15d8e83)  // fmla za.s[x8, 3], { z20.s-z23.s }, z13.s[3]
    bgt label_23
KAI_ASM_LABEL(label_24)  // Width 4: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x11
    KAI_ASM_INST(0xa04045c5)  // ldnt1w { z4.s-z5.s }, pn9.b/Z, [x14]
    subs x11, x11, #0x1
    ld1rqw { z8.s }, p0/Z, [x10]
    addvl x14, x14, #2
    KAI_ASM_INST(0xa0404727)  // ldnt1w { z6.s-z7.s }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa0404761)  // ldnt1w { z0.s-z1.s }, pn9.b/Z, [x27]
    addvl x27, x27, #2
    KAI_ASM_INST(0xa0404703)  // ldnt1w { z2.s-z3.s }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa040452d)  // ldnt1w { z12.s-z13.s }, pn9.b/Z, [x9]
    KAI_ASM_INST(0xc1588080)  // fmla za.s[x8, 0], { z4.s-z7.s }, z8.s[0]
    addvl x9, x9, #2
    KAI_ASM_INST(0xa04046ef)  // ldnt1w { z14.s-z15.s }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa0404791)  // ldnt1w { z16.s-z17.s }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xc1588001)  // fmla za.s[x8, 1], { z0.s-z3.s }, z8.s[0]
    addvl x28, x28, #2
    KAI_ASM_INST(0xa0404753)  // ldnt1w { z18.s-z19.s }, pn9.b/Z, [x26]
    addvl x26, x26, #2
    KAI_ASM_INST(0xc1588182)  // fmla za.s[x8, 2], { z12.s-z15.s }, z8.s[0]
    KAI_ASM_INST(0xc1588203)  // fmla za.s[x8, 3], { z16.s-z19.s }, z8.s[0]
    ble label_25
    KAI_ASM_INST(0xa04045dd)  // ldnt1w { z28.s-z29.s }, pn9.b/Z, [x14]
    subs x11, x11, #0x1
    addvl x14, x14, #2
    KAI_ASM_INST(0xa040473f)  // ldnt1w { z30.s-z31.s }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa0404761)  // ldnt1w { z0.s-z1.s }, pn9.b/Z, [x27]
    addvl x27, x27, #2
    KAI_ASM_INST(0xa0404703)  // ldnt1w { z2.s-z3.s }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa0404525)  // ldnt1w { z4.s-z5.s }, pn9.b/Z, [x9]
    KAI_ASM_INST(0xc1588780)  // fmla za.s[x8, 0], { z28.s-z31.s }, z8.s[1]
    addvl x9, x9, #2
    KAI_ASM_INST(0xa04046e7)  // ldnt1w { z6.s-z7.s }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa0404791)  // ldnt1w { z16.s-z17.s }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xc1588401)  // fmla za.s[x8, 1], { z0.s-z3.s }, z8.s[1]
    addvl x28, x28, #2
    KAI_ASM_INST(0xa0404753)  // ldnt1w { z18.s-z19.s }, pn9.b/Z, [x26]
    addvl x26, x26, #2
    KAI_ASM_INST(0xc1588482)  // fmla za.s[x8, 2], { z4.s-z7.s }, z8.s[1]
    KAI_ASM_INST(0xc1588603)  // fmla za.s[x8, 3], { z16.s-z19.s }, z8.s[1]
    ble label_25
    KAI_ASM_INST(0xa04045dd)  // ldnt1w { z28.s-z29.s }, pn9.b/Z, [x14]
    subs x11, x11, #0x1
    addvl x14, x14, #2
    KAI_ASM_INST(0xa040473f)  // ldnt1w { z30.s-z31.s }, pn9.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa040476d)  // ldnt1w { z12.s-z13.s }, pn9.b/Z, [x27]
    addvl x27, x27, #2
    KAI_ASM_INST(0xa040470f)  // ldnt1w { z14.s-z15.s }, pn9.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa0404521)  // ldnt1w { z0.s-z1.s }, pn9.b/Z, [x9]
    KAI_ASM_INST(0xc1588b80)  // fmla za.s[x8, 0], { z28.s-z31.s }, z8.s[2]
    addvl x9, x9, #2
    KAI_ASM_INST(0xa04046e3)  // ldnt1w { z2.s-z3.s }, pn9.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa0404791)  // ldnt1w { z16.s-z17.s }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xc1588981)  // fmla za.s[x8, 1], { z12.s-z15.s }, z8.s[2]
    addvl x28, x28, #2
    KAI_ASM_INST(0xa0404753)  // ldnt1w { z18.s-z19.s }, pn9.b/Z, [x26]
    addvl x26, x26, #2
    KAI_ASM_INST(0xc1588802)  // fmla za.s[x8, 2], { z0.s-z3.s }, z8.s[2]
    KAI_ASM_INST(0xc1588a03)  // fmla za.s[x8, 3], { z16.s-z19.s }, z8.s[2]
    ble label_25
    KAI_ASM_INST(0xa04045c5)  // ldnt1w { z4.s-z5.s }, pn9.b/Z, [x14]
    KAI_ASM_INST(0xa0404727)  // ldnt1w { z6.s-z7.s }, pn9.b/Z, [x25]
    KAI_ASM_INST(0xa040476d)  // ldnt1w { z12.s-z13.s }, pn9.b/Z, [x27]
    KAI_ASM_INST(0xa040470f)  // ldnt1w { z14.s-z15.s }, pn9.b/Z, [x24]
    KAI_ASM_INST(0xa0404535)  // ldnt1w { z20.s-z21.s }, pn9.b/Z, [x9]
    KAI_ASM_INST(0xc1588c80)  // fmla za.s[x8, 0], { z4.s-z7.s }, z8.s[3]
    KAI_ASM_INST(0xa04046f7)  // ldnt1w { z22.s-z23.s }, pn9.b/Z, [x23]
    KAI_ASM_INST(0xa0404791)  // ldnt1w { z16.s-z17.s }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xc1588d81)  // fmla za.s[x8, 1], { z12.s-z15.s }, z8.s[3]
    KAI_ASM_INST(0xa0404753)  // ldnt1w { z18.s-z19.s }, pn9.b/Z, [x26]
    KAI_ASM_INST(0xc1588e82)  // fmla za.s[x8, 2], { z20.s-z23.s }, z8.s[3]
    KAI_ASM_INST(0xc1588e03)  // fmla za.s[x8, 3], { z16.s-z19.s }, z8.s[3]
KAI_ASM_LABEL(label_25)  // Width 4: Multiply loop: multiply skip
    tbz x15, #1, label_26
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
    KAI_ASM_INST(0xa060c5a4)  // st1w { z4.s-z7.s }, pn9.b, [x13]
    KAI_ASM_INST(0xa061c5a0)  // st1w { z0.s-z3.s }, pn9.b, [x13, #0x4, MUL VL]
    KAI_ASM_INST(0xa062c5ac)  // st1w { z12.s-z15.s }, pn9.b, [x13, #0x8, MUL VL]
    KAI_ASM_INST(0xa063c1b0)  // st1w { z16.s-z19.s }, p8, [x13, #0xc, MUL VL]
    addvl x13, x13, #16
    b label_27
KAI_ASM_LABEL(label_26)  // Width 4: No activation
    KAI_ASM_INST(0xc0060c0c)  // mova { z12.d-z15.d }, za.d[x8, #0]
    KAI_ASM_INST(0xc0060c20)  // mova { z0.d-z3.d }, za.d[x8, #1]
    KAI_ASM_INST(0xc0060c50)  // mova { z16.d-z19.d }, za.d[x8, #2]
    KAI_ASM_INST(0xc0060c64)  // mova { z4.d-z7.d }, za.d[x8, #3]
    KAI_ASM_INST(0xa060c5ac)  // st1w { z12.s-z15.s }, pn9.b, [x13]
    KAI_ASM_INST(0xa061c5a0)  // st1w { z0.s-z3.s }, pn9.b, [x13, #0x4, MUL VL]
    KAI_ASM_INST(0xa062c5b0)  // st1w { z16.s-z19.s }, pn9.b, [x13, #0x8, MUL VL]
    KAI_ASM_INST(0xa063c1a4)  // st1w { z4.s-z7.s }, p8, [x13, #0xc, MUL VL]
    addvl x13, x13, #16
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla)

    KAI_ASM_END
