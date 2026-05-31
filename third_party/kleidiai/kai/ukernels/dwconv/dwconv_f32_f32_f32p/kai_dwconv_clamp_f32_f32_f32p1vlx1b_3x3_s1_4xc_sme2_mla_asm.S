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

    KAI_ASM_CODE(dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla)

KAI_ASM_FUNCTION_TYPE(kai_kernel_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla)
KAI_ASM_FUNCTION_LABEL(kai_kernel_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    mov x20, #0x6
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    ldr x7, [x0, #0x10]
    ptrue p2.b
    KAI_ASM_INST(0x25207812)  // ptrue pn10.b
    ldr x17, [x0, #0x8]
    KAI_ASM_INST(0x8554c80d)  // ld1rw { z13.s }, p2/Z, [x0, #80]
    ldr x16, [x0, #0x68]
    KAI_ASM_INST(0x8555c818)  // ld1rw { z24.s }, p2/Z, [x0, #84]
    sub x20, x20, x7
    ldr x15, [x0, #0x60]
    whilelt p1.s, XZR, x16
    whilelt p9.b, XZR, x20
    whilelt p8.b, XZR, x17
    eor p8.b, p2/Z, p8.b, p9.b
KAI_ASM_LABEL(label_1)  // Channel loop
    ldr x24, [x0, #0x58]
    mov x20, #0x28
    madd x24, x15, x20, x24
    ld1w { z20.s }, p1/Z, [x24]
    ldr x14, [x0, #0x18]
    addvl x24, x24, #1
    mov x23, #0x6
    mov z21.d, z20.d
    ldr x13, [x0, #0x0]
    KAI_ASM_INST(0xa0404b04)  // ld1w { z4.s-z5.s }, pn10.b/Z, [x24]
    add x22, x17, x7
    lsl x21, x1, #0x2
    ld1w { z3.s }, p2/Z, [x24, #2, MUL VL]
    addvl x24, x24, #3
    mov z22.d, z20.d
    mov z23.d, z20.d
    sub x20, x14, #0x1
    KAI_ASM_INST(0xa0404b0a)  // ld1w { z10.s-z11.s }, pn10.b/Z, [x24]
    mov x8, #0x0
    ldr x11, [x0, #0x20]
    orr x20, x20, x2, LSL #18
    ld1w { z6.s }, p2/Z, [x24, #2, MUL VL]
    addvl x24, x24, #3
    orr x20, x16, x20, LSL #20
    KAI_ASM_INST(0xa0404b08)  // ld1w { z8.s-z9.s }, pn10.b/Z, [x24]
    sub x23, x23, x22
    ld1w { z2.s }, p2/Z, [x24, #2, MUL VL]
    lsl x20, x20, #0x2
    madd x21, x21, x17, x13
KAI_ASM_LABEL(label_3)  // Issue prefetches
    subs x23, x23, #0x1
    KAI_ASM_INST(0xf8b44abc)  // rprfm pldstrm, x20, [x21]
    add x21, x21, x2, LSL #2
    bgt label_3
    ldr x22, [x0, #0x28]
    lsl x21, x1, #0x2
    KAI_ASM_INST(0xc0040e80)  // mova za.d[x8, #0], { z20.d-z23.d }
    mov x10, #0x2
    ldr x20, [x0, #0x30]
    msub x13, x17, x21, x13
    KAI_ASM_INST(0xc0040e81)  // mova za.d[x8, #1], { z20.d-z23.d }
    ldr x21, [x0, #0x48]
    KAI_ASM_INST(0xc0040e82)  // mova za.d[x8, #2], { z20.d-z23.d }
    ldp x9, x28, [x22], #0x10
    ldp x27, x26, [x20], #0x10
    ldp x25, x24, [x22], #0x10
    ldp x23, x22, [x20], #0x10
    cbz x21, label_5
    cmp x21, x10
    csel x20, x21, x10, LT
    sub x21, x21, x20
    sub x10, x10, x20
    cbz x21, label_5
    KAI_ASM_INST(0xc0060c1c)  // mova { z28.d-z31.d }, za.d[x8, #0]
    sub x11, x11, x21
    KAI_ASM_INST(0xc1b8c9bc)  // fclamp { z28.s-z31.s }, z13.s, z24.s
KAI_ASM_LABEL(label_4)  // Left padding
    subs x21, x21, #0x1
    st1w { z28.s }, p1, [x9]
    add x9, x9, x27, LSL #2
    st1w { z29.s }, p1, [x28]
    add x28, x28, x26, LSL #2
    st1w { z30.s }, p1, [x25]
    add x25, x25, x23, LSL #2
    st1w { z31.s }, p1, [x24]
    add x24, x24, x22, LSL #2
    bgt label_4
KAI_ASM_LABEL(label_5)  // Left padding: End
    adds XZR, x17, x7
    bne label_10
    cbz x10, label_8
    cmp x10, #0x1
    sub x14, x14, x10
    beq label_7
    add x20, x13, x1, LSL #2
    ld1w { z14.s }, p1/Z, [x13]
    add x13, x13, x2, LSL #2
    ld1w { z15.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    ld1w { z16.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    ld1w { z17.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    ld1w { z18.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    ld1w { z19.s }, p1/Z, [x20]
    KAI_ASM_INST(0xc13419c0)  // fmla za.s[x8, 0], { z14.s-z17.s }, z4.s
    KAI_ASM_INST(0xc13a19e0)  // fmla za.s[x8, 0], { z15.s-z18.s }, z10.s
    KAI_ASM_INST(0xc1381a00)  // fmla za.s[x8, 0], { z16.s-z19.s }, z8.s
KAI_ASM_LABEL(label_7)  // Unpadded: 1 priming loads
    add x20, x13, x1, LSL #2
    ld1w { z26.s }, p1/Z, [x13]
    add x13, x13, x2, LSL #2
    ld1w { z27.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    ld1w { z28.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    ld1w { z29.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    ld1w { z30.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    ld1w { z31.s }, p1/Z, [x20]
    KAI_ASM_INST(0xc1351b40)  // fmla za.s[x8, 0], { z26.s-z29.s }, z5.s
    KAI_ASM_INST(0xc1341b41)  // fmla za.s[x8, 1], { z26.s-z29.s }, z4.s
    KAI_ASM_INST(0xc13b1b60)  // fmla za.s[x8, 0], { z27.s-z30.s }, z11.s
    KAI_ASM_INST(0xc13a1b61)  // fmla za.s[x8, 1], { z27.s-z30.s }, z10.s
    KAI_ASM_INST(0xc1391b80)  // fmla za.s[x8, 0], { z28.s-z31.s }, z9.s
    KAI_ASM_INST(0xc1381b81)  // fmla za.s[x8, 1], { z28.s-z31.s }, z8.s
KAI_ASM_LABEL(label_8)  // Unpadded: 0 priming loads
    cbz x14, label_16
    add x20, x13, x1, LSL #2
    ld1w { z14.s }, p1/Z, [x13]
    sub x14, x14, #0x1
    ld1w { z15.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    sub x11, x11, #0x1
    ld1w { z16.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    cmp x14, x11
    ld1w { z17.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    csel x21, x14, x11, LT
    ld1w { z18.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    add x13, x13, x2, LSL #2
    ld1w { z19.s }, p1/Z, [x20]
    sub x11, x11, x21
    cbz x21, label_15
KAI_ASM_LABEL(label_9)  // Unpadded: Main loop
    KAI_ASM_INST(0xc13319c0)  // fmla za.s[x8, 0], { z14.s-z17.s }, z3.s
    add x20, x13, x1, LSL #2
    subs x21, x21, #0x1
    KAI_ASM_INST(0xc13519c1)  // fmla za.s[x8, 1], { z14.s-z17.s }, z5.s
    KAI_ASM_INST(0xc13419c2)  // fmla za.s[x8, 2], { z14.s-z17.s }, z4.s
    ld1w { z14.s }, p1/Z, [x13]
    add x13, x13, x2, LSL #2
    KAI_ASM_INST(0xc13619e0)  // fmla za.s[x8, 0], { z15.s-z18.s }, z6.s
    KAI_ASM_INST(0xc13b19e1)  // fmla za.s[x8, 1], { z15.s-z18.s }, z11.s
    KAI_ASM_INST(0xc13a19e2)  // fmla za.s[x8, 2], { z15.s-z18.s }, z10.s
    ld1w { z15.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    KAI_ASM_INST(0xc1321a00)  // fmla za.s[x8, 0], { z16.s-z19.s }, z2.s
    KAI_ASM_INST(0xc1391a01)  // fmla za.s[x8, 1], { z16.s-z19.s }, z9.s
    KAI_ASM_INST(0xc1381a02)  // fmla za.s[x8, 2], { z16.s-z19.s }, z8.s
    ld1w { z16.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    ld1w { z17.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    KAI_ASM_INST(0xc0060c1c)  // mova { z28.d-z31.d }, za.d[x8, #0]
    add x8, x8, #0x1
    ld1w { z18.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    KAI_ASM_INST(0xc0040e82)  // mova za.d[x8, #2], { z20.d-z23.d }
    ld1w { z19.s }, p1/Z, [x20]
    KAI_ASM_INST(0xc1b8c9bc)  // fclamp { z28.s-z31.s }, z13.s, z24.s
    st1w { z28.s }, p1, [x9]
    add x9, x9, x27, LSL #2
    st1w { z29.s }, p1, [x28]
    add x28, x28, x26, LSL #2
    st1w { z30.s }, p1, [x25]
    add x25, x25, x23, LSL #2
    st1w { z31.s }, p1, [x24]
    add x24, x24, x22, LSL #2
    bgt label_9
    b label_15
KAI_ASM_LABEL(label_10)  // Padded
    cbz x10, label_13
    cmp x10, #0x1
    sub x14, x14, x10
    beq label_12
    mov x12, #0x0
    add x20, x13, x1, LSL #2
    KAI_ASM_INST(0x25244500)  // psel p0.b, p1.b/Z, p8.b[w12]
    ld1w { z28.s }, p0/Z, [x13]
    KAI_ASM_INST(0x252c4500)  // psel p0.b, p1.b/Z, p8.b[w12, #1]
    add x13, x13, x2, LSL #2
    ld1w { z29.s }, p0/Z, [x20]
    KAI_ASM_INST(0x25344500)  // psel p0.b, p1.b/Z, p8.b[w12, #2]
    add x20, x20, x1, LSL #2
    ld1w { z30.s }, p0/Z, [x20]
    KAI_ASM_INST(0x253c4500)  // psel p0.b, p1.b/Z, p8.b[w12, #3]
    add x20, x20, x1, LSL #2
    mov x12, #0x4
    ld1w { z31.s }, p0/Z, [x20]
    KAI_ASM_INST(0x25244500)  // psel p0.b, p1.b/Z, p8.b[w12]
    add x20, x20, x1, LSL #2
    KAI_ASM_INST(0xc1341b80)  // fmla za.s[x8, 0], { z28.s-z31.s }, z4.s
    ld1w { z0.s }, p0/Z, [x20]
    KAI_ASM_INST(0x252c4500)  // psel p0.b, p1.b/Z, p8.b[w12, #1]
    add x20, x20, x1, LSL #2
    ld1w { z1.s }, p0/Z, [x20]
    KAI_ASM_INST(0xc13a1ba0)  // fmla za.s[x8, 0], { z29.s-z0.s }, z10.s
    KAI_ASM_INST(0xc1381bc0)  // fmla za.s[x8, 0], { z30.s-z1.s }, z8.s
KAI_ASM_LABEL(label_12)  // Padded: 1 priming loads
    mov x12, #0x0
    add x20, x13, x1, LSL #2
    KAI_ASM_INST(0x25244500)  // psel p0.b, p1.b/Z, p8.b[w12]
    ld1w { z25.s }, p0/Z, [x13]
    KAI_ASM_INST(0x252c4500)  // psel p0.b, p1.b/Z, p8.b[w12, #1]
    add x13, x13, x2, LSL #2
    ld1w { z26.s }, p0/Z, [x20]
    KAI_ASM_INST(0x25344500)  // psel p0.b, p1.b/Z, p8.b[w12, #2]
    add x20, x20, x1, LSL #2
    ld1w { z27.s }, p0/Z, [x20]
    KAI_ASM_INST(0x253c4500)  // psel p0.b, p1.b/Z, p8.b[w12, #3]
    add x20, x20, x1, LSL #2
    mov x12, #0x4
    ld1w { z28.s }, p0/Z, [x20]
    KAI_ASM_INST(0x25244500)  // psel p0.b, p1.b/Z, p8.b[w12]
    add x20, x20, x1, LSL #2
    KAI_ASM_INST(0xc1351b20)  // fmla za.s[x8, 0], { z25.s-z28.s }, z5.s
    ld1w { z29.s }, p0/Z, [x20]
    KAI_ASM_INST(0x252c4500)  // psel p0.b, p1.b/Z, p8.b[w12, #1]
    add x20, x20, x1, LSL #2
    KAI_ASM_INST(0xc1341b21)  // fmla za.s[x8, 1], { z25.s-z28.s }, z4.s
    ld1w { z30.s }, p0/Z, [x20]
    KAI_ASM_INST(0xc13b1b40)  // fmla za.s[x8, 0], { z26.s-z29.s }, z11.s
    KAI_ASM_INST(0xc13a1b41)  // fmla za.s[x8, 1], { z26.s-z29.s }, z10.s
    KAI_ASM_INST(0xc1391b60)  // fmla za.s[x8, 0], { z27.s-z30.s }, z9.s
    KAI_ASM_INST(0xc1381b61)  // fmla za.s[x8, 1], { z27.s-z30.s }, z8.s
KAI_ASM_LABEL(label_13)  // Padded: 0 priming loads
    cbz x14, label_16
    mov x12, #0x0
    add x20, x13, x1, LSL #2
    KAI_ASM_INST(0x25244500)  // psel p0.b, p1.b/Z, p8.b[w12]
    sub x14, x14, #0x1
    sub x11, x11, #0x1
    cmp x14, x11
    ld1w { z14.s }, p0/Z, [x13]
    KAI_ASM_INST(0x252c4500)  // psel p0.b, p1.b/Z, p8.b[w12, #1]
    csel x21, x14, x11, LT
    add x13, x13, x2, LSL #2
    sub x11, x11, x21
    ld1w { z15.s }, p0/Z, [x20]
    KAI_ASM_INST(0x25344500)  // psel p0.b, p1.b/Z, p8.b[w12, #2]
    add x20, x20, x1, LSL #2
    ld1w { z16.s }, p0/Z, [x20]
    KAI_ASM_INST(0x253c4500)  // psel p0.b, p1.b/Z, p8.b[w12, #3]
    add x20, x20, x1, LSL #2
    mov x12, #0x4
    ld1w { z17.s }, p0/Z, [x20]
    KAI_ASM_INST(0x25244500)  // psel p0.b, p1.b/Z, p8.b[w12]
    add x20, x20, x1, LSL #2
    ld1w { z18.s }, p0/Z, [x20]
    KAI_ASM_INST(0x252c4500)  // psel p0.b, p1.b/Z, p8.b[w12, #1]
    add x20, x20, x1, LSL #2
    ld1w { z19.s }, p0/Z, [x20]
    cbz x21, label_15
KAI_ASM_LABEL(label_14)  // Padded: Main loop
    mov x12, #0x0
    KAI_ASM_INST(0xc13319c0)  // fmla za.s[x8, 0], { z14.s-z17.s }, z3.s
    add x20, x13, x1, LSL #2
    KAI_ASM_INST(0x25244500)  // psel p0.b, p1.b/Z, p8.b[w12]
    KAI_ASM_INST(0xc13519c1)  // fmla za.s[x8, 1], { z14.s-z17.s }, z5.s
    subs x21, x21, #0x1
    KAI_ASM_INST(0xc13419c2)  // fmla za.s[x8, 2], { z14.s-z17.s }, z4.s
    ld1w { z14.s }, p0/Z, [x13]
    KAI_ASM_INST(0x252c4500)  // psel p0.b, p1.b/Z, p8.b[w12, #1]
    KAI_ASM_INST(0xc13619e0)  // fmla za.s[x8, 0], { z15.s-z18.s }, z6.s
    add x13, x13, x2, LSL #2
    KAI_ASM_INST(0xc13b19e1)  // fmla za.s[x8, 1], { z15.s-z18.s }, z11.s
    KAI_ASM_INST(0xc13a19e2)  // fmla za.s[x8, 2], { z15.s-z18.s }, z10.s
    ld1w { z15.s }, p0/Z, [x20]
    KAI_ASM_INST(0x25344500)  // psel p0.b, p1.b/Z, p8.b[w12, #2]
    add x20, x20, x1, LSL #2
    KAI_ASM_INST(0xc1321a00)  // fmla za.s[x8, 0], { z16.s-z19.s }, z2.s
    KAI_ASM_INST(0xc1391a01)  // fmla za.s[x8, 1], { z16.s-z19.s }, z9.s
    KAI_ASM_INST(0xc1381a02)  // fmla za.s[x8, 2], { z16.s-z19.s }, z8.s
    ld1w { z16.s }, p0/Z, [x20]
    KAI_ASM_INST(0x253c4500)  // psel p0.b, p1.b/Z, p8.b[w12, #3]
    add x20, x20, x1, LSL #2
    mov x12, #0x4
    KAI_ASM_INST(0xc0060c1c)  // mova { z28.d-z31.d }, za.d[x8, #0]
    add x8, x8, #0x1
    ld1w { z17.s }, p0/Z, [x20]
    KAI_ASM_INST(0x25244500)  // psel p0.b, p1.b/Z, p8.b[w12]
    add x20, x20, x1, LSL #2
    KAI_ASM_INST(0xc0040e82)  // mova za.d[x8, #2], { z20.d-z23.d }
    ld1w { z18.s }, p0/Z, [x20]
    KAI_ASM_INST(0x252c4500)  // psel p0.b, p1.b/Z, p8.b[w12, #1]
    add x20, x20, x1, LSL #2
    KAI_ASM_INST(0xc1b8c9bc)  // fclamp { z28.s-z31.s }, z13.s, z24.s
    ld1w { z19.s }, p0/Z, [x20]
    st1w { z28.s }, p1, [x9]
    add x9, x9, x27, LSL #2
    st1w { z29.s }, p1, [x28]
    add x28, x28, x26, LSL #2
    st1w { z30.s }, p1, [x25]
    add x25, x25, x23, LSL #2
    st1w { z31.s }, p1, [x24]
    add x24, x24, x22, LSL #2
    bgt label_14
KAI_ASM_LABEL(label_15)  // Main loop tail
    KAI_ASM_INST(0xc13319c0)  // fmla za.s[x8, 0], { z14.s-z17.s }, z3.s
    KAI_ASM_INST(0xc13519c1)  // fmla za.s[x8, 1], { z14.s-z17.s }, z5.s
    KAI_ASM_INST(0xc13419c2)  // fmla za.s[x8, 2], { z14.s-z17.s }, z4.s
    KAI_ASM_INST(0xc13619e0)  // fmla za.s[x8, 0], { z15.s-z18.s }, z6.s
    KAI_ASM_INST(0xc13b19e1)  // fmla za.s[x8, 1], { z15.s-z18.s }, z11.s
    KAI_ASM_INST(0xc13a19e2)  // fmla za.s[x8, 2], { z15.s-z18.s }, z10.s
    KAI_ASM_INST(0xc1321a00)  // fmla za.s[x8, 0], { z16.s-z19.s }, z2.s
    KAI_ASM_INST(0xc1391a01)  // fmla za.s[x8, 1], { z16.s-z19.s }, z9.s
    KAI_ASM_INST(0xc1381a02)  // fmla za.s[x8, 2], { z16.s-z19.s }, z8.s
    KAI_ASM_INST(0xc0060c08)  // mova { z8.d-z11.d }, za.d[x8, #0]
    add x8, x8, #0x1
    KAI_ASM_INST(0xc0040e82)  // mova za.d[x8, #2], { z20.d-z23.d }
    KAI_ASM_INST(0xc1b8c9a8)  // fclamp { z8.s-z11.s }, z13.s, z24.s
    st1w { z8.s }, p1, [x9]
    add x9, x9, x27, LSL #2
    st1w { z9.s }, p1, [x28]
    add x28, x28, x26, LSL #2
    st1w { z10.s }, p1, [x25]
    add x25, x25, x23, LSL #2
    st1w { z11.s }, p1, [x24]
    add x24, x24, x22, LSL #2
KAI_ASM_LABEL(label_16)  // Main loop skip tail
    cbz x11, label_18
KAI_ASM_LABEL(label_17)  // Right padding loop
    KAI_ASM_INST(0xc0060c08)  // mova { z8.d-z11.d }, za.d[x8, #0]
    add x8, x8, #0x1
    subs x11, x11, #0x1
    KAI_ASM_INST(0xc0040e82)  // mova za.d[x8, #2], { z20.d-z23.d }
    KAI_ASM_INST(0xc1b8c9a8)  // fclamp { z8.s-z11.s }, z13.s, z24.s
    st1w { z8.s }, p1, [x9]
    add x9, x9, x27, LSL #2
    st1w { z9.s }, p1, [x28]
    add x28, x28, x26, LSL #2
    st1w { z10.s }, p1, [x25]
    add x25, x25, x23, LSL #2
    st1w { z11.s }, p1, [x24]
    add x24, x24, x22, LSL #2
    bgt label_17
KAI_ASM_LABEL(label_18)  // End
    ldr x21, [x0, #0x38]
    incw x15
    ldr x20, [x0, #0x0]
    whilelt p1.s, x15, x16
    add x20, x20, x21, LSL #2
    str x20, [x0, #0x0]
    ldr x25, [x0, #0x28]
    ldr x24, [x0, #0x40]
    ldp x23, x22, [x25, #0x0]
    ldp x21, x20, [x24, #0x0]
    add x23, x23, x21, LSL #2
    add x22, x22, x20, LSL #2
    stp x23, x22, [x25, #0x0]
    ldp x23, x22, [x25, #0x10]
    ldp x21, x20, [x24, #0x10]
    add x23, x23, x21, LSL #2
    add x22, x22, x20, LSL #2
    stp x23, x22, [x25, #0x10]
    b.ne label_1
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
    KAI_ASM_FUNCTION_END(kai_kernel_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla)

    KAI_ASM_END
