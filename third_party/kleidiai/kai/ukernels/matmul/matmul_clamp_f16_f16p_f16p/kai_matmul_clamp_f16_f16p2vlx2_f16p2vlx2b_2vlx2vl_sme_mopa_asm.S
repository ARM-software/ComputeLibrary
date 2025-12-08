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

    KAI_ASM_CODE(matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa)
    KAI_ASM_GLOBAL(kai_f16_from_float_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa)

KAI_ASM_FUNCTION_TYPE(kai_f16_from_float_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa)
KAI_ASM_FUNCTION_LABEL(kai_f16_from_float_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa)
    fcvt h0, s0
    fmov w0, h0
    ret
    KAI_ASM_FUNCTION_END(kai_f16_from_float_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa)
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
    mov x11, #0x0
    ldr w10, [x0, #0x20]
    ldr w9, [x0, #0x28]
    add x13, x13, #0x1
    ldr x28, [x0, #0x0]
    lsr x13, x13, #0x1
KAI_ASM_LABEL(label_1)  // M loop
    ldr x27, [x0, #0x8]
KAI_ASM_LABEL(label_2)  // N loop
    fmov z19.h, #0.0
    ld1h { z16.h }, p1/Z, [x27]
    fmov z18.h, #1.0
    mov x20, x11
    whilelt p8.s, x20, x9
    incw x20
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    mov x26, x28
    whilelt p8.s, x20, x9
    inch x27, ALL, MUL #2
    zip1 z17.h, z16.h, z19.h
    zip2 z16.h, z16.h, z19.h
    KAI_ASM_INST(0x81b12640)  // fmopa za0.s, p1/M, p1/M, z18.h, z17.h
    KAI_ASM_INST(0x81b02641)  // fmopa za1.s, p1/M, p1/M, z18.h, z16.h
    KAI_ASM_INST(0x81b12642)  // fmopa za2.s, p1/M, p1/M, z18.h, z17.h
    KAI_ASM_INST(0x81b02643)  // fmopa za3.s, p1/M, p1/M, z18.h, z16.h
    lsr x21, x13, #0x2
    and x20, x13, #0x3
    cbz x21, label_6
    subs x21, x21, #0x1
    ld1h { z31.h }, p1/Z, [x26]
    ld1h { z30.h }, p1/Z, [x26, #1, MUL VL]
    ld1h { z29.h }, p1/Z, [x26, #2, MUL VL]
    ld1h { z28.h }, p1/Z, [x26, #3, MUL VL]
    ld1h { z27.h }, p1/Z, [x26, #4, MUL VL]
    ld1h { z26.h }, p1/Z, [x26, #5, MUL VL]
    ld1h { z25.h }, p1/Z, [x26, #6, MUL VL]
    ld1h { z24.h }, p1/Z, [x26, #7, MUL VL]
    addvl x26, x26, #8
    ld1h { z23.h }, p1/Z, [x27]
    ld1h { z22.h }, p1/Z, [x27, #1, MUL VL]
    ld1h { z21.h }, p1/Z, [x27, #2, MUL VL]
    ld1h { z20.h }, p1/Z, [x27, #3, MUL VL]
    ld1h { z19.h }, p1/Z, [x27, #4, MUL VL]
    ld1h { z18.h }, p1/Z, [x27, #5, MUL VL]
    ld1h { z17.h }, p1/Z, [x27, #6, MUL VL]
    ld1h { z16.h }, p1/Z, [x27, #7, MUL VL]
    addvl x27, x27, #8
    ble label_5
KAI_ASM_LABEL(label_4)  // K loop
    KAI_ASM_INST(0x81b727e0)  // fmopa za0.s, p1/M, p1/M, z31.h, z23.h
    subs x21, x21, #0x1
    KAI_ASM_INST(0x81b627e1)  // fmopa za1.s, p1/M, p1/M, z31.h, z22.h
    ld1h { z31.h }, p1/Z, [x26]
    KAI_ASM_INST(0x81b727c2)  // fmopa za2.s, p1/M, p1/M, z30.h, z23.h
    ld1h { z23.h }, p1/Z, [x27]
    KAI_ASM_INST(0x81b627c3)  // fmopa za3.s, p1/M, p1/M, z30.h, z22.h
    ld1h { z30.h }, p1/Z, [x26, #1, MUL VL]
    KAI_ASM_INST(0x81b527a0)  // fmopa za0.s, p1/M, p1/M, z29.h, z21.h
    ld1h { z22.h }, p1/Z, [x27, #1, MUL VL]
    KAI_ASM_INST(0x81b427a1)  // fmopa za1.s, p1/M, p1/M, z29.h, z20.h
    ld1h { z29.h }, p1/Z, [x26, #2, MUL VL]
    KAI_ASM_INST(0x81b52782)  // fmopa za2.s, p1/M, p1/M, z28.h, z21.h
    ld1h { z21.h }, p1/Z, [x27, #2, MUL VL]
    KAI_ASM_INST(0x81b42783)  // fmopa za3.s, p1/M, p1/M, z28.h, z20.h
    ld1h { z28.h }, p1/Z, [x26, #3, MUL VL]
    KAI_ASM_INST(0x81b32760)  // fmopa za0.s, p1/M, p1/M, z27.h, z19.h
    ld1h { z20.h }, p1/Z, [x27, #3, MUL VL]
    KAI_ASM_INST(0x81b22761)  // fmopa za1.s, p1/M, p1/M, z27.h, z18.h
    ld1h { z27.h }, p1/Z, [x26, #4, MUL VL]
    KAI_ASM_INST(0x81b32742)  // fmopa za2.s, p1/M, p1/M, z26.h, z19.h
    ld1h { z19.h }, p1/Z, [x27, #4, MUL VL]
    KAI_ASM_INST(0x81b22743)  // fmopa za3.s, p1/M, p1/M, z26.h, z18.h
    ld1h { z26.h }, p1/Z, [x26, #5, MUL VL]
    KAI_ASM_INST(0x81b12720)  // fmopa za0.s, p1/M, p1/M, z25.h, z17.h
    ld1h { z18.h }, p1/Z, [x27, #5, MUL VL]
    KAI_ASM_INST(0x81b02721)  // fmopa za1.s, p1/M, p1/M, z25.h, z16.h
    ld1h { z25.h }, p1/Z, [x26, #6, MUL VL]
    KAI_ASM_INST(0x81b12702)  // fmopa za2.s, p1/M, p1/M, z24.h, z17.h
    ld1h { z17.h }, p1/Z, [x27, #6, MUL VL]
    KAI_ASM_INST(0x81b02703)  // fmopa za3.s, p1/M, p1/M, z24.h, z16.h
    ld1h { z24.h }, p1/Z, [x26, #7, MUL VL]
    addvl x26, x26, #8
    ld1h { z16.h }, p1/Z, [x27, #7, MUL VL]
    addvl x27, x27, #8
    bgt label_4
KAI_ASM_LABEL(label_5)  // K loop tail
    KAI_ASM_INST(0x81b727e0)  // fmopa za0.s, p1/M, p1/M, z31.h, z23.h
    KAI_ASM_INST(0x81b627e1)  // fmopa za1.s, p1/M, p1/M, z31.h, z22.h
    KAI_ASM_INST(0x81b727c2)  // fmopa za2.s, p1/M, p1/M, z30.h, z23.h
    KAI_ASM_INST(0x81b627c3)  // fmopa za3.s, p1/M, p1/M, z30.h, z22.h
    KAI_ASM_INST(0x81b527a0)  // fmopa za0.s, p1/M, p1/M, z29.h, z21.h
    KAI_ASM_INST(0x81b427a1)  // fmopa za1.s, p1/M, p1/M, z29.h, z20.h
    KAI_ASM_INST(0x81b52782)  // fmopa za2.s, p1/M, p1/M, z28.h, z21.h
    KAI_ASM_INST(0x81b42783)  // fmopa za3.s, p1/M, p1/M, z28.h, z20.h
    KAI_ASM_INST(0x81b32760)  // fmopa za0.s, p1/M, p1/M, z27.h, z19.h
    KAI_ASM_INST(0x81b22761)  // fmopa za1.s, p1/M, p1/M, z27.h, z18.h
    KAI_ASM_INST(0x81b32742)  // fmopa za2.s, p1/M, p1/M, z26.h, z19.h
    KAI_ASM_INST(0x81b22743)  // fmopa za3.s, p1/M, p1/M, z26.h, z18.h
    KAI_ASM_INST(0x81b12720)  // fmopa za0.s, p1/M, p1/M, z25.h, z17.h
    KAI_ASM_INST(0x81b02721)  // fmopa za1.s, p1/M, p1/M, z25.h, z16.h
    KAI_ASM_INST(0x81b12702)  // fmopa za2.s, p1/M, p1/M, z24.h, z17.h
    KAI_ASM_INST(0x81b02703)  // fmopa za3.s, p1/M, p1/M, z24.h, z16.h
KAI_ASM_LABEL(label_6)  // K oddments
    cbz x20, label_8
KAI_ASM_LABEL(label_7)  // K oddments: Loop
    ld1h { z19.h }, p1/Z, [x26]
    subs x20, x20, #0x1
    ld1h { z18.h }, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    ld1h { z17.h }, p1/Z, [x27]
    ld1h { z16.h }, p1/Z, [x27, #1, MUL VL]
    addvl x27, x27, #2
    KAI_ASM_INST(0x81b12660)  // fmopa za0.s, p1/M, p1/M, z19.h, z17.h
    KAI_ASM_INST(0x81b02661)  // fmopa za1.s, p1/M, p1/M, z19.h, z16.h
    KAI_ASM_INST(0x81b12642)  // fmopa za2.s, p1/M, p1/M, z18.h, z17.h
    KAI_ASM_INST(0x81b02643)  // fmopa za3.s, p1/M, p1/M, z18.h, z16.h
    bgt label_7
KAI_ASM_LABEL(label_8)  // K oddments: End
    ldr x25, [x0, #0x10]
    sub x24, x10, x14
    cntw x23, ALL, MUL #2
    KAI_ASM_INST(0x84dca413)  // ld1rh { z19.h }, p1/Z, [x0, #56]
    ldr x22, [x0, #0x18]
    whilelt p0.h, x11, x9
    cmp x24, x23
    KAI_ASM_INST(0x84dda412)  // ld1rh { z18.h }, p1/Z, [x0, #58]
    mov x12, #0x0
    mov x21, #0x0
    add x25, x25, x11, LSL #1  // C += n
    mov x20, #0x2
    madd x25, x14, x22, x25  // C += m * ldc
    csel x24, x24, x23, LT
KAI_ASM_LABEL(label_10)  // Store to output array: Accumulator loop
    KAI_ASM_INST(0xc0020411)  // mova z17.b, p1/M, za0h.b[x12]
    add x21, x21, #0x1
    KAI_ASM_INST(0xc0020430)  // mova z16.b, p1/M, za0h.b[x12, #1]
    fcvt z17.h, p1/m, z17.s
    add x12, x12, #0x4
    fcvt z16.h, p1/m, z16.s
    cmp x12, x23, LSL #1
    csel x12, x12, x20, LT
    cmp x21, x24
    uzp1 z16.h, z17.h, z16.h
    fmin z16.h, p1/M, z16.h, z18.h
    fmax z16.h, p1/M, z16.h, z19.h
    st1h { z16.h }, p0, [x25]
    add x25, x25, x22
    blt label_10
    incw x11, ALL, MUL #2
    cmp x11, x9
    blt label_2
    incw x14, ALL, MUL #2
    mov x11, #0x0
    cmp x14, x10
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_f16p2vlx2_f16p2vlx2b_2vlx2vl_sme_mopa)

    KAI_ASM_END
