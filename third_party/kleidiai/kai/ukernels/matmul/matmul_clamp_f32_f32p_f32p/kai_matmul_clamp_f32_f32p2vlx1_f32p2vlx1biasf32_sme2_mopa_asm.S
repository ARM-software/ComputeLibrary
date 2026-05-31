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

    KAI_ASM_CODE(matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa)
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
    mov x15, #0x0
    ptrue p0.b
    KAI_ASM_INST(0x25207811)  // ptrue pn9.b
    ldr x14, [x0, #0x30]
    ldr w13, [x0, #0x20]
    mov x11, #0x0
    ldr w10, [x0, #0x28]
    ldr x9, [x0, #0x0]
KAI_ASM_LABEL(label_1)  // M loop
    ldr x28, [x0, #0x8]
KAI_ASM_LABEL(label_2)  // N loop
    fmov z22.s, #1.0
    KAI_ASM_INST(0xa040478c)  // ld1w { z12.s-z13.s }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    mov x27, x9
    KAI_ASM_INST(0x25aa4570)  // whilelt pn8.s, x11, x10, VLx2
    addvl x28, x28, #2
    KAI_ASM_INST(0x808c02c0)  // fmopa za0.s, p0/M, p0/M, z22.s, z12.s
    KAI_ASM_INST(0x808d02c1)  // fmopa za1.s, p0/M, p0/M, z22.s, z13.s
    KAI_ASM_INST(0x808c02c2)  // fmopa za2.s, p0/M, p0/M, z22.s, z12.s
    KAI_ASM_INST(0x808d02c3)  // fmopa za3.s, p0/M, p0/M, z22.s, z13.s
    lsr x21, x14, #0x2
    and x20, x14, #0x3
    cbz x21, label_6
    subs x21, x21, #0x1
    KAI_ASM_INST(0xa040c764)  // ld1w { z4.s-z7.s }, pn9.b/Z, [x27]
    KAI_ASM_INST(0xa041c768)  // ld1w { z8.s-z11.s }, pn9.b/Z, [x27, #0x4, MUL VL]
    addvl x27, x27, #8
    KAI_ASM_INST(0xa040c794)  // ld1w { z20.s-z23.s }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xa041c78c)  // ld1w { z12.s-z15.s }, pn9.b/Z, [x28, #0x4, MUL VL]
    addvl x28, x28, #8
    ble label_5
KAI_ASM_LABEL(label_4)  // K loop
    KAI_ASM_INST(0x80940080)  // fmopa za0.s, p0/M, p0/M, z4.s, z20.s
    subs x21, x21, #0x1
    KAI_ASM_INST(0x80950081)  // fmopa za1.s, p0/M, p0/M, z4.s, z21.s
    KAI_ASM_INST(0x809400a2)  // fmopa za2.s, p0/M, p0/M, z5.s, z20.s
    KAI_ASM_INST(0x809500a3)  // fmopa za3.s, p0/M, p0/M, z5.s, z21.s
    KAI_ASM_INST(0x809600c0)  // fmopa za0.s, p0/M, p0/M, z6.s, z22.s
    KAI_ASM_INST(0x809700c1)  // fmopa za1.s, p0/M, p0/M, z6.s, z23.s
    KAI_ASM_INST(0x809600e2)  // fmopa za2.s, p0/M, p0/M, z7.s, z22.s
    KAI_ASM_INST(0x809700e3)  // fmopa za3.s, p0/M, p0/M, z7.s, z23.s
    KAI_ASM_INST(0xa040c764)  // ld1w { z4.s-z7.s }, pn9.b/Z, [x27]
    KAI_ASM_INST(0x808c0100)  // fmopa za0.s, p0/M, p0/M, z8.s, z12.s
    KAI_ASM_INST(0xa040c794)  // ld1w { z20.s-z23.s }, pn9.b/Z, [x28]
    KAI_ASM_INST(0x808d0101)  // fmopa za1.s, p0/M, p0/M, z8.s, z13.s
    KAI_ASM_INST(0x808c0122)  // fmopa za2.s, p0/M, p0/M, z9.s, z12.s
    KAI_ASM_INST(0x808d0123)  // fmopa za3.s, p0/M, p0/M, z9.s, z13.s
    KAI_ASM_INST(0x808e0140)  // fmopa za0.s, p0/M, p0/M, z10.s, z14.s
    KAI_ASM_INST(0x808f0141)  // fmopa za1.s, p0/M, p0/M, z10.s, z15.s
    KAI_ASM_INST(0x808e0162)  // fmopa za2.s, p0/M, p0/M, z11.s, z14.s
    KAI_ASM_INST(0x808f0163)  // fmopa za3.s, p0/M, p0/M, z11.s, z15.s
    KAI_ASM_INST(0xa041c768)  // ld1w { z8.s-z11.s }, pn9.b/Z, [x27, #0x4, MUL VL]
    addvl x27, x27, #8
    KAI_ASM_INST(0xa041c78c)  // ld1w { z12.s-z15.s }, pn9.b/Z, [x28, #0x4, MUL VL]
    addvl x28, x28, #8
    bgt label_4
KAI_ASM_LABEL(label_5)  // K loop tail
    KAI_ASM_INST(0x80940080)  // fmopa za0.s, p0/M, p0/M, z4.s, z20.s
    KAI_ASM_INST(0x80950081)  // fmopa za1.s, p0/M, p0/M, z4.s, z21.s
    KAI_ASM_INST(0x809400a2)  // fmopa za2.s, p0/M, p0/M, z5.s, z20.s
    KAI_ASM_INST(0x809500a3)  // fmopa za3.s, p0/M, p0/M, z5.s, z21.s
    KAI_ASM_INST(0x809600c0)  // fmopa za0.s, p0/M, p0/M, z6.s, z22.s
    KAI_ASM_INST(0x809700c1)  // fmopa za1.s, p0/M, p0/M, z6.s, z23.s
    KAI_ASM_INST(0x809600e2)  // fmopa za2.s, p0/M, p0/M, z7.s, z22.s
    KAI_ASM_INST(0x809700e3)  // fmopa za3.s, p0/M, p0/M, z7.s, z23.s
    KAI_ASM_INST(0x808c0100)  // fmopa za0.s, p0/M, p0/M, z8.s, z12.s
    KAI_ASM_INST(0x808d0101)  // fmopa za1.s, p0/M, p0/M, z8.s, z13.s
    KAI_ASM_INST(0x808c0122)  // fmopa za2.s, p0/M, p0/M, z9.s, z12.s
    KAI_ASM_INST(0x808d0123)  // fmopa za3.s, p0/M, p0/M, z9.s, z13.s
    KAI_ASM_INST(0x808e0140)  // fmopa za0.s, p0/M, p0/M, z10.s, z14.s
    KAI_ASM_INST(0x808f0141)  // fmopa za1.s, p0/M, p0/M, z10.s, z15.s
    KAI_ASM_INST(0x808e0162)  // fmopa za2.s, p0/M, p0/M, z11.s, z14.s
    KAI_ASM_INST(0x808f0163)  // fmopa za3.s, p0/M, p0/M, z11.s, z15.s
KAI_ASM_LABEL(label_6)  // K oddments
    cbz x20, label_8
KAI_ASM_LABEL(label_7)  // K oddments: Loop
    KAI_ASM_INST(0xa040477c)  // ld1w { z28.s-z29.s }, pn9.b/Z, [x27]
    subs x20, x20, #0x1
    addvl x27, x27, #2
    KAI_ASM_INST(0xa1404787)  // ld1w { z7.s, z15.s }, pn9.b/Z, [x28]
    addvl x28, x28, #2
    KAI_ASM_INST(0x80870380)  // fmopa za0.s, p0/M, p0/M, z28.s, z7.s
    KAI_ASM_INST(0x808f0381)  // fmopa za1.s, p0/M, p0/M, z28.s, z15.s
    KAI_ASM_INST(0x808703a2)  // fmopa za2.s, p0/M, p0/M, z29.s, z7.s
    KAI_ASM_INST(0x808f03a3)  // fmopa za3.s, p0/M, p0/M, z29.s, z15.s
    bgt label_7
KAI_ASM_LABEL(label_8)  // K oddments: End
    ldr x26, [x0, #0x10]
    sub x25, x13, x15
    cntw x24
    KAI_ASM_INST(0x854ec01a)  // ld1rw { z26.s }, p0/Z, [x0, #56]
    ldr x23, [x0, #0x18]
    cmp x25, x24
    KAI_ASM_INST(0x854fc018)  // ld1rw { z24.s }, p0/Z, [x0, #60]
    mov x12, #0x0
    csel x22, x25, x24, LT
    add x26, x26, x11, LSL #2  // C += n
    lsr x21, x22, #0x2
    madd x26, x15, x23, x26  // C += m * ldc
    and x20, x22, #0x3
    cbz x21, label_11
KAI_ASM_LABEL(label_10)  // Store to output array: Accumulator row 0 loop
    KAI_ASM_INST(0xc0860404)  // mova { z4.s-z7.s }, za0h.s[x12]
    KAI_ASM_INST(0xc086042c)  // mova { z12.s-z15.s }, za1h.s[x12]
    KAI_ASM_INST(0xc1b8cb44)  // fclamp { z4.s-z7.s }, z26.s, z24.s
    KAI_ASM_INST(0xc1b8cb4c)  // fclamp { z12.s-z15.s }, z26.s, z24.s
    add x12, x12, #0x4
    cmp x12, x21, LSL #2
    KAI_ASM_INST(0xa1604344)  // st1w { z4.s, z12.s }, p8, [x26]
    add x26, x26, x23
    KAI_ASM_INST(0xa1604345)  // st1w { z5.s, z13.s }, p8, [x26]
    add x26, x26, x23
    KAI_ASM_INST(0xa1604346)  // st1w { z6.s, z14.s }, p8, [x26]
    add x26, x26, x23
    KAI_ASM_INST(0xa1604347)  // st1w { z7.s, z15.s }, p8, [x26]
    add x26, x26, x23
    blt label_10
KAI_ASM_LABEL(label_11)  // Store to output array: Accumulator row 0 oddments
    cbz x20, label_12
    KAI_ASM_INST(0xc0860400)  // mova { z0.s-z3.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860428)  // mova { z8.s-z11.s }, za1h.s[x12]
    subs x20, x20, #0x1
    KAI_ASM_INST(0xc1b8cb40)  // fclamp { z0.s-z3.s }, z26.s, z24.s
    KAI_ASM_INST(0xc1b8cb48)  // fclamp { z8.s-z11.s }, z26.s, z24.s
    KAI_ASM_INST(0xa1604340)  // st1w { z0.s, z8.s }, p8, [x26]
    add x26, x26, x23
    beq label_12
    subs x20, x20, #0x1
    KAI_ASM_INST(0xa1604341)  // st1w { z1.s, z9.s }, p8, [x26]
    add x26, x26, x23
    beq label_12
    KAI_ASM_INST(0xa1604342)  // st1w { z2.s, z10.s }, p8, [x26]
    add x26, x26, x23
KAI_ASM_LABEL(label_12)  // Store to output array: Accumulator row 0 oddments: End
    subs x25, x25, x22
    beq label_16
    cmp x25, x24
    mov x12, #0x0
    csel x20, x25, x24, LT
    lsr x21, x20, #0x2
    and x20, x20, #0x3
    cbz x21, label_14
KAI_ASM_LABEL(label_13)  // Store to output array: Accumulator row 1 loop
    KAI_ASM_INST(0xc0860454)  // mova { z20.s-z23.s }, za2h.s[x12]
    KAI_ASM_INST(0xc086047c)  // mova { z28.s-z31.s }, za3h.s[x12]
    KAI_ASM_INST(0xc1b8cb54)  // fclamp { z20.s-z23.s }, z26.s, z24.s
    KAI_ASM_INST(0xc1b8cb5c)  // fclamp { z28.s-z31.s }, z26.s, z24.s
    add x12, x12, #0x4
    cmp x12, x21, LSL #2
    KAI_ASM_INST(0xa1604354)  // st1w { z20.s, z28.s }, p8, [x26]
    add x26, x26, x23
    KAI_ASM_INST(0xa1604355)  // st1w { z21.s, z29.s }, p8, [x26]
    add x26, x26, x23
    KAI_ASM_INST(0xa1604356)  // st1w { z22.s, z30.s }, p8, [x26]
    add x26, x26, x23
    KAI_ASM_INST(0xa1604357)  // st1w { z23.s, z31.s }, p8, [x26]
    add x26, x26, x23
    blt label_13
KAI_ASM_LABEL(label_14)  // Store to output array: Accumulator row 1 oddments
    cbz x20, label_15
    KAI_ASM_INST(0xc0860444)  // mova { z4.s-z7.s }, za2h.s[x12]
    KAI_ASM_INST(0xc086046c)  // mova { z12.s-z15.s }, za3h.s[x12]
    subs x20, x20, #0x1
    KAI_ASM_INST(0xc1b8cb44)  // fclamp { z4.s-z7.s }, z26.s, z24.s
    KAI_ASM_INST(0xc1b8cb4c)  // fclamp { z12.s-z15.s }, z26.s, z24.s
    KAI_ASM_INST(0xa1604344)  // st1w { z4.s, z12.s }, p8, [x26]
    add x26, x26, x23
    beq label_15
    subs x20, x20, #0x1
    KAI_ASM_INST(0xa1604345)  // st1w { z5.s, z13.s }, p8, [x26]
    add x26, x26, x23
    beq label_15
    KAI_ASM_INST(0xa1604346)  // st1w { z6.s, z14.s }, p8, [x26]
KAI_ASM_LABEL(label_15)  // Store to output array: Accumulator row 1 oddments: End
KAI_ASM_LABEL(label_16)  // Store to output array: End
    incw x11, ALL, MUL #2
    cmp x11, x10
    blt label_2
    incw x15, ALL, MUL #2
    mov x11, #0x0
    cmp x15, x13
    mov x9, x27
    blt label_1
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa)

    KAI_ASM_END
