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

    KAI_ASM_CODE(imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa)
    KAI_ASM_GLOBAL(kai_f16_from_float_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_f16_from_float_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_f16_from_float_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa)
    fcvt h0, s0
    fmov w0, h0
    ret
    KAI_ASM_FUNCTION_END(kai_f16_from_float_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa)
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
    mov x14, #0x0
    ldr x13, [x0, #0x30]
    ptrue p1.b
    KAI_ASM_INST(0x25207810)  // ptrue pn8.b
    ldr w11, [x0, #0x20]
    mov x10, #0x0
    ldr w9, [x0, #0x28]
    add x13, x13, #0x1
    ldr x28, [x0, #0x0]
    lsr x13, x13, #0x1
KAI_ASM_LABEL(label_1)  // M loop
    ldr x27, [x0, #0x8]
KAI_ASM_LABEL(label_2)  // N loop
    fmov z23.h, #0.0
    ld1h { z18.h }, p1/Z, [x27]
    fmov z2.h, #1.0
    mov x26, x28
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    inch x27, ALL, MUL #2
    zip1 z14.h, z18.h, z23.h
    zip2 z3.h, z18.h, z23.h
    KAI_ASM_INST(0x81ae2440)  // fmopa za0.s, p1/M, p1/M, z2.h, z14.h
    KAI_ASM_INST(0x81a32441)  // fmopa za1.s, p1/M, p1/M, z2.h, z3.h
    KAI_ASM_INST(0x81ae2442)  // fmopa za2.s, p1/M, p1/M, z2.h, z14.h
    KAI_ASM_INST(0x81a32443)  // fmopa za3.s, p1/M, p1/M, z2.h, z3.h
    lsr x21, x13, #0x2
    and x20, x13, #0x3
    cbz x21, label_6
    subs x21, x21, #0x1
    KAI_ASM_INST(0xa040a350)  // ld1h { z16.h-z19.h }, pn8.b/Z, [x26]
    KAI_ASM_INST(0xa041a35c)  // ld1h { z28.h-z31.h }, pn8.b/Z, [x26, #0x4, MUL VL]
    addvl x26, x26, #8
    KAI_ASM_INST(0xa040a360)  // ld1h { z0.h-z3.h }, pn8.b/Z, [x27]
    KAI_ASM_INST(0xa041a368)  // ld1h { z8.h-z11.h }, pn8.b/Z, [x27, #0x4, MUL VL]
    addvl x27, x27, #8
    ble label_5
KAI_ASM_LABEL(label_4)  // K loop
    KAI_ASM_INST(0x81a02600)  // fmopa za0.s, p1/M, p1/M, z16.h, z0.h
    subs x21, x21, #0x1
    KAI_ASM_INST(0x81a12601)  // fmopa za1.s, p1/M, p1/M, z16.h, z1.h
    KAI_ASM_INST(0x81a02622)  // fmopa za2.s, p1/M, p1/M, z17.h, z0.h
    KAI_ASM_INST(0x81a12623)  // fmopa za3.s, p1/M, p1/M, z17.h, z1.h
    KAI_ASM_INST(0x81a22640)  // fmopa za0.s, p1/M, p1/M, z18.h, z2.h
    KAI_ASM_INST(0x81a32641)  // fmopa za1.s, p1/M, p1/M, z18.h, z3.h
    KAI_ASM_INST(0x81a22662)  // fmopa za2.s, p1/M, p1/M, z19.h, z2.h
    KAI_ASM_INST(0x81a32663)  // fmopa za3.s, p1/M, p1/M, z19.h, z3.h
    KAI_ASM_INST(0xa040a350)  // ld1h { z16.h-z19.h }, pn8.b/Z, [x26]
    KAI_ASM_INST(0x81a82780)  // fmopa za0.s, p1/M, p1/M, z28.h, z8.h
    KAI_ASM_INST(0xa040a360)  // ld1h { z0.h-z3.h }, pn8.b/Z, [x27]
    KAI_ASM_INST(0x81a92781)  // fmopa za1.s, p1/M, p1/M, z28.h, z9.h
    KAI_ASM_INST(0x81a827a2)  // fmopa za2.s, p1/M, p1/M, z29.h, z8.h
    KAI_ASM_INST(0x81a927a3)  // fmopa za3.s, p1/M, p1/M, z29.h, z9.h
    KAI_ASM_INST(0x81aa27c0)  // fmopa za0.s, p1/M, p1/M, z30.h, z10.h
    KAI_ASM_INST(0x81ab27c1)  // fmopa za1.s, p1/M, p1/M, z30.h, z11.h
    KAI_ASM_INST(0x81aa27e2)  // fmopa za2.s, p1/M, p1/M, z31.h, z10.h
    KAI_ASM_INST(0x81ab27e3)  // fmopa za3.s, p1/M, p1/M, z31.h, z11.h
    KAI_ASM_INST(0xa041a35c)  // ld1h { z28.h-z31.h }, pn8.b/Z, [x26, #0x4, MUL VL]
    addvl x26, x26, #8
    KAI_ASM_INST(0xa041a368)  // ld1h { z8.h-z11.h }, pn8.b/Z, [x27, #0x4, MUL VL]
    addvl x27, x27, #8
    bgt label_4
KAI_ASM_LABEL(label_5)  // K loop tail
    KAI_ASM_INST(0x81a02600)  // fmopa za0.s, p1/M, p1/M, z16.h, z0.h
    KAI_ASM_INST(0x81a12601)  // fmopa za1.s, p1/M, p1/M, z16.h, z1.h
    KAI_ASM_INST(0x81a02622)  // fmopa za2.s, p1/M, p1/M, z17.h, z0.h
    KAI_ASM_INST(0x81a12623)  // fmopa za3.s, p1/M, p1/M, z17.h, z1.h
    KAI_ASM_INST(0x81a22640)  // fmopa za0.s, p1/M, p1/M, z18.h, z2.h
    KAI_ASM_INST(0x81a32641)  // fmopa za1.s, p1/M, p1/M, z18.h, z3.h
    KAI_ASM_INST(0x81a22662)  // fmopa za2.s, p1/M, p1/M, z19.h, z2.h
    KAI_ASM_INST(0x81a32663)  // fmopa za3.s, p1/M, p1/M, z19.h, z3.h
    KAI_ASM_INST(0x81a82780)  // fmopa za0.s, p1/M, p1/M, z28.h, z8.h
    KAI_ASM_INST(0x81a92781)  // fmopa za1.s, p1/M, p1/M, z28.h, z9.h
    KAI_ASM_INST(0x81a827a2)  // fmopa za2.s, p1/M, p1/M, z29.h, z8.h
    KAI_ASM_INST(0x81a927a3)  // fmopa za3.s, p1/M, p1/M, z29.h, z9.h
    KAI_ASM_INST(0x81aa27c0)  // fmopa za0.s, p1/M, p1/M, z30.h, z10.h
    KAI_ASM_INST(0x81ab27c1)  // fmopa za1.s, p1/M, p1/M, z30.h, z11.h
    KAI_ASM_INST(0x81aa27e2)  // fmopa za2.s, p1/M, p1/M, z31.h, z10.h
    KAI_ASM_INST(0x81ab27e3)  // fmopa za3.s, p1/M, p1/M, z31.h, z11.h
KAI_ASM_LABEL(label_6)  // K oddments
    cbz x20, label_8
KAI_ASM_LABEL(label_7)  // K oddments: Loop
    KAI_ASM_INST(0xa1402345)  // ld1h { z5.h, z13.h }, pn8.b/Z, [x26]
    subs x20, x20, #0x1
    addvl x26, x26, #2
    KAI_ASM_INST(0xa040236e)  // ld1h { z14.h-z15.h }, pn8.b/Z, [x27]
    addvl x27, x27, #2
    KAI_ASM_INST(0x81ae24a0)  // fmopa za0.s, p1/M, p1/M, z5.h, z14.h
    KAI_ASM_INST(0x81af24a1)  // fmopa za1.s, p1/M, p1/M, z5.h, z15.h
    KAI_ASM_INST(0x81ae25a2)  // fmopa za2.s, p1/M, p1/M, z13.h, z14.h
    KAI_ASM_INST(0x81af25a3)  // fmopa za3.s, p1/M, p1/M, z13.h, z15.h
    bgt label_7
KAI_ASM_LABEL(label_8)  // K oddments: End
    ldr x25, [x0, #0x10]
    sub x24, x11, x14
    cntw x23, ALL, MUL #2
    KAI_ASM_INST(0x84dca411)  // ld1rh { z17.h }, p1/Z, [x0, #56]
    ldr x22, [x0, #0x18]
    whilelt p0.h, x10, x9
    cmp x24, x23
    KAI_ASM_INST(0x84dda410)  // ld1rh { z16.h }, p1/Z, [x0, #58]
    mov x12, #0x0
    mov x21, #0x0
    add x25, x25, x10, LSL #1  // C += n
    mov x20, #0x2
    madd x25, x14, x22, x25  // C += m * ldc
    csel x24, x24, x23, LT
KAI_ASM_LABEL(label_10)  // Store to output array: Accumulator loop
    KAI_ASM_INST(0xc006000e)  // mova { z14.b-z15.b }, za0h.b[x12, 0:1]
    add x12, x12, #0x4
    cmp x12, x23, LSL #1
    add x21, x21, #0x1
    KAI_ASM_INST(0xc120e1c4)  // fcvt z4.h, { z14.s-z15.s }
    csel x12, x12, x20, LT
    cmp x21, x24
    KAI_ASM_INST(0x64702624)  // fclamp z4.h, z17.h, z16.h
    st1h { z4.h }, p0, [x25]
    add x25, x25, x22
    blt label_10
    incw x10, ALL, MUL #2
    cmp x10, x9
    blt label_2
    incw x14, ALL, MUL #2
    mov x10, #0x0
    cmp x14, x11
    mov x28, x26
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
    KAI_ASM_FUNCTION_END(kai_kernel_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa)

    KAI_ASM_END
