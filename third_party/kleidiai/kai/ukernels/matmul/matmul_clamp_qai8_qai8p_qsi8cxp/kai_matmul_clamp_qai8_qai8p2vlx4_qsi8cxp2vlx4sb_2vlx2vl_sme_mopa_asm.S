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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa)
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
    ldr x14, [x0, #0x30]
    ptrue p1.b
    mov x13, #0x0
    ldr w11, [x0, #0x20]
    ldr w10, [x0, #0x28]
    add x14, x14, #0x3
    ldr x9, [x0, #0x0]
    lsr x14, x14, #0x2
KAI_ASM_LABEL(label_1)  // M loop
    ldr x28, [x0, #0x8]
KAI_ASM_LABEL(label_2)  // N loop
    ld1w { z17.s }, p1/Z, [x28]
    mov x20, x13
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    mov x27, x9
    ld1w { z16.s }, p1/Z, [x28, #1, MUL VL]
    whilelt p8.s, x20, x10
    incw x20
    whilelt p8.s, x20, x10
    addvl x28, x28, #2
    KAI_ASM_INST(0xc0902620)  // addha za0.s, p1/M, p1/M, z17.s
    KAI_ASM_INST(0xc0902601)  // addha za1.s, p1/M, p1/M, z16.s
    KAI_ASM_INST(0xc0902622)  // addha za2.s, p1/M, p1/M, z17.s
    KAI_ASM_INST(0xc0902603)  // addha za3.s, p1/M, p1/M, z16.s
    lsr x21, x14, #0x2
    and x20, x14, #0x3
    cbz x21, label_6
    subs x21, x21, #0x1
    ld1b { z31.b }, p1/Z, [x27]
    ld1b { z30.b }, p1/Z, [x27, #1, MUL VL]
    ld1b { z29.b }, p1/Z, [x27, #2, MUL VL]
    ld1b { z28.b }, p1/Z, [x27, #3, MUL VL]
    ld1b { z27.b }, p1/Z, [x27, #4, MUL VL]
    ld1b { z26.b }, p1/Z, [x27, #5, MUL VL]
    ld1b { z25.b }, p1/Z, [x27, #6, MUL VL]
    ld1b { z24.b }, p1/Z, [x27, #7, MUL VL]
    addvl x27, x27, #8
    ld1b { z23.b }, p1/Z, [x28]
    ld1b { z22.b }, p1/Z, [x28, #1, MUL VL]
    ld1b { z21.b }, p1/Z, [x28, #2, MUL VL]
    ld1b { z20.b }, p1/Z, [x28, #3, MUL VL]
    ld1b { z19.b }, p1/Z, [x28, #4, MUL VL]
    ld1b { z18.b }, p1/Z, [x28, #5, MUL VL]
    ld1b { z17.b }, p1/Z, [x28, #6, MUL VL]
    ld1b { z16.b }, p1/Z, [x28, #7, MUL VL]
    addvl x28, x28, #8
    ble label_5
KAI_ASM_LABEL(label_4)  // K loop
    KAI_ASM_INST(0xa09727e0)  // smopa za0.s, p1/M, p1/M, z31.b, z23.b
    subs x21, x21, #0x1
    KAI_ASM_INST(0xa09627e1)  // smopa za1.s, p1/M, p1/M, z31.b, z22.b
    ld1b { z31.b }, p1/Z, [x27]
    KAI_ASM_INST(0xa09727c2)  // smopa za2.s, p1/M, p1/M, z30.b, z23.b
    ld1b { z23.b }, p1/Z, [x28]
    KAI_ASM_INST(0xa09627c3)  // smopa za3.s, p1/M, p1/M, z30.b, z22.b
    ld1b { z30.b }, p1/Z, [x27, #1, MUL VL]
    KAI_ASM_INST(0xa09527a0)  // smopa za0.s, p1/M, p1/M, z29.b, z21.b
    ld1b { z22.b }, p1/Z, [x28, #1, MUL VL]
    KAI_ASM_INST(0xa09427a1)  // smopa za1.s, p1/M, p1/M, z29.b, z20.b
    ld1b { z29.b }, p1/Z, [x27, #2, MUL VL]
    KAI_ASM_INST(0xa0952782)  // smopa za2.s, p1/M, p1/M, z28.b, z21.b
    ld1b { z21.b }, p1/Z, [x28, #2, MUL VL]
    KAI_ASM_INST(0xa0942783)  // smopa za3.s, p1/M, p1/M, z28.b, z20.b
    ld1b { z28.b }, p1/Z, [x27, #3, MUL VL]
    KAI_ASM_INST(0xa0932760)  // smopa za0.s, p1/M, p1/M, z27.b, z19.b
    ld1b { z20.b }, p1/Z, [x28, #3, MUL VL]
    KAI_ASM_INST(0xa0922761)  // smopa za1.s, p1/M, p1/M, z27.b, z18.b
    ld1b { z27.b }, p1/Z, [x27, #4, MUL VL]
    KAI_ASM_INST(0xa0932742)  // smopa za2.s, p1/M, p1/M, z26.b, z19.b
    ld1b { z19.b }, p1/Z, [x28, #4, MUL VL]
    KAI_ASM_INST(0xa0922743)  // smopa za3.s, p1/M, p1/M, z26.b, z18.b
    ld1b { z26.b }, p1/Z, [x27, #5, MUL VL]
    KAI_ASM_INST(0xa0912720)  // smopa za0.s, p1/M, p1/M, z25.b, z17.b
    ld1b { z18.b }, p1/Z, [x28, #5, MUL VL]
    KAI_ASM_INST(0xa0902721)  // smopa za1.s, p1/M, p1/M, z25.b, z16.b
    ld1b { z25.b }, p1/Z, [x27, #6, MUL VL]
    KAI_ASM_INST(0xa0912702)  // smopa za2.s, p1/M, p1/M, z24.b, z17.b
    ld1b { z17.b }, p1/Z, [x28, #6, MUL VL]
    KAI_ASM_INST(0xa0902703)  // smopa za3.s, p1/M, p1/M, z24.b, z16.b
    ld1b { z24.b }, p1/Z, [x27, #7, MUL VL]
    addvl x27, x27, #8
    ld1b { z16.b }, p1/Z, [x28, #7, MUL VL]
    addvl x28, x28, #8
    bgt label_4
KAI_ASM_LABEL(label_5)  // K loop tail
    KAI_ASM_INST(0xa09727e0)  // smopa za0.s, p1/M, p1/M, z31.b, z23.b
    KAI_ASM_INST(0xa09627e1)  // smopa za1.s, p1/M, p1/M, z31.b, z22.b
    KAI_ASM_INST(0xa09727c2)  // smopa za2.s, p1/M, p1/M, z30.b, z23.b
    KAI_ASM_INST(0xa09627c3)  // smopa za3.s, p1/M, p1/M, z30.b, z22.b
    KAI_ASM_INST(0xa09527a0)  // smopa za0.s, p1/M, p1/M, z29.b, z21.b
    KAI_ASM_INST(0xa09427a1)  // smopa za1.s, p1/M, p1/M, z29.b, z20.b
    KAI_ASM_INST(0xa0952782)  // smopa za2.s, p1/M, p1/M, z28.b, z21.b
    KAI_ASM_INST(0xa0942783)  // smopa za3.s, p1/M, p1/M, z28.b, z20.b
    KAI_ASM_INST(0xa0932760)  // smopa za0.s, p1/M, p1/M, z27.b, z19.b
    KAI_ASM_INST(0xa0922761)  // smopa za1.s, p1/M, p1/M, z27.b, z18.b
    KAI_ASM_INST(0xa0932742)  // smopa za2.s, p1/M, p1/M, z26.b, z19.b
    KAI_ASM_INST(0xa0922743)  // smopa za3.s, p1/M, p1/M, z26.b, z18.b
    KAI_ASM_INST(0xa0912720)  // smopa za0.s, p1/M, p1/M, z25.b, z17.b
    KAI_ASM_INST(0xa0902721)  // smopa za1.s, p1/M, p1/M, z25.b, z16.b
    KAI_ASM_INST(0xa0912702)  // smopa za2.s, p1/M, p1/M, z24.b, z17.b
    KAI_ASM_INST(0xa0902703)  // smopa za3.s, p1/M, p1/M, z24.b, z16.b
KAI_ASM_LABEL(label_6)  // K oddments
    cbz x20, label_8
KAI_ASM_LABEL(label_7)  // K oddments: Loop
    ld1b { z19.b }, p1/Z, [x27]
    subs x20, x20, #0x1
    ld1b { z18.b }, p1/Z, [x27, #1, MUL VL]
    addvl x27, x27, #2
    ld1b { z17.b }, p1/Z, [x28]
    ld1b { z16.b }, p1/Z, [x28, #1, MUL VL]
    addvl x28, x28, #2
    KAI_ASM_INST(0xa0912660)  // smopa za0.s, p1/M, p1/M, z19.b, z17.b
    KAI_ASM_INST(0xa0902661)  // smopa za1.s, p1/M, p1/M, z19.b, z16.b
    KAI_ASM_INST(0xa0912642)  // smopa za2.s, p1/M, p1/M, z18.b, z17.b
    KAI_ASM_INST(0xa0902643)  // smopa za3.s, p1/M, p1/M, z18.b, z16.b
    bgt label_7
KAI_ASM_LABEL(label_8)  // K oddments: End
    ldr x26, [x0, #0x10]
    sub x25, x11, x15
    cntw x24
    KAI_ASM_INST(0x854ec41c)  // ld1rw { z28.s }, p1/Z, [x0, #56]
    ldr x23, [x0, #0x18]
    whilelt p0.h, x13, x10
    cmp x25, x24
    KAI_ASM_INST(0x854fc41b)  // ld1rw { z27.s }, p1/Z, [x0, #60]
    csel x22, x25, x24, LT
    KAI_ASM_INST(0x8550c41a)  // ld1rw { z26.s }, p1/Z, [x0, #64]
    mov x12, #0x0
    add x26, x26, x13  // C += n
    lsr x21, x22, #0x2
    ld1w { z25.s }, p1/Z, [x28]
    madd x26, x15, x23, x26  // C += m * ldc
    ld1w { z24.s }, p1/Z, [x28, #1, MUL VL]
    addvl x28, x28, #2
    and x20, x22, #0x3
    cbz x21, label_11
KAI_ASM_LABEL(label_10)  // Store to output array: Accumulator row 0 loop
    KAI_ASM_INST(0xc0820417)  // mova z23.s, p1/M, za0h.s[x12]
    KAI_ASM_INST(0xc0820491)  // mova z17.s, p1/M, za1h.s[x12]
    scvtf z23.s, p1/m, z23.s
    KAI_ASM_INST(0xc0820436)  // mova z22.s, p1/M, za0h.s[x12, #1]
    scvtf z17.s, p1/m, z17.s
    KAI_ASM_INST(0xc08204b0)  // mova z16.s, p1/M, za1h.s[x12, #1]
    scvtf z22.s, p1/m, z22.s
    KAI_ASM_INST(0xc0820455)  // mova z21.s, p1/M, za0h.s[x12, #2]
    scvtf z16.s, p1/m, z16.s
    KAI_ASM_INST(0xc08204d4)  // mova z20.s, p1/M, za1h.s[x12, #2]
    scvtf z21.s, p1/m, z21.s
    fmul z23.s, z23.s, z25.s
    KAI_ASM_INST(0xc0820473)  // mova z19.s, p1/M, za0h.s[x12, #3]
    scvtf z20.s, p1/m, z20.s
    fmul z17.s, z17.s, z24.s
    KAI_ASM_INST(0xc08204f2)  // mova z18.s, p1/M, za1h.s[x12, #3]
    scvtf z19.s, p1/m, z19.s
    fmul z22.s, z22.s, z25.s
    add x12, x12, #0x4
    scvtf z18.s, p1/m, z18.s
    fmul z16.s, z16.s, z24.s
    cmp x12, x21, LSL #2
    fmul z21.s, z21.s, z25.s
    frintn z23.s, p1/m, z23.s
    fmul z20.s, z20.s, z24.s
    frintn z17.s, p1/m, z17.s
    fmul z19.s, z19.s, z25.s
    frintn z22.s, p1/m, z22.s
    fmul z18.s, z18.s, z24.s
    frintn z16.s, p1/m, z16.s
    frintn z21.s, p1/m, z21.s
    fcvtzs z23.s, p1/m, z23.s
    frintn z20.s, p1/m, z20.s
    fcvtzs z17.s, p1/m, z17.s
    frintn z19.s, p1/m, z19.s
    fcvtzs z22.s, p1/m, z22.s
    frintn z18.s, p1/m, z18.s
    fcvtzs z16.s, p1/m, z16.s
    fcvtzs z21.s, p1/m, z21.s
    add z23.s, z23.s, z26.s
    fcvtzs z20.s, p1/m, z20.s
    add z17.s, z17.s, z26.s
    fcvtzs z19.s, p1/m, z19.s
    add z22.s, z22.s, z26.s
    fcvtzs z18.s, p1/m, z18.s
    add z16.s, z16.s, z26.s
    add z21.s, z21.s, z26.s
    smin z23.s, p1/M, z23.s, z27.s
    add z20.s, z20.s, z26.s
    smin z17.s, p1/M, z17.s, z27.s
    add z19.s, z19.s, z26.s
    smin z22.s, p1/M, z22.s, z27.s
    add z18.s, z18.s, z26.s
    smin z16.s, p1/M, z16.s, z27.s
    smin z21.s, p1/M, z21.s, z27.s
    smax z23.s, p1/M, z23.s, z28.s
    smin z20.s, p1/M, z20.s, z27.s
    smax z17.s, p1/M, z17.s, z28.s
    smin z19.s, p1/M, z19.s, z27.s
    smax z22.s, p1/M, z22.s, z28.s
    smin z18.s, p1/M, z18.s, z27.s
    smax z16.s, p1/M, z16.s, z28.s
    smax z21.s, p1/M, z21.s, z28.s
    smax z20.s, p1/M, z20.s, z28.s
    uzp1 z17.h, z23.h, z17.h
    smax z19.s, p1/M, z19.s, z28.s
    smax z18.s, p1/M, z18.s, z28.s
    uzp1 z16.h, z22.h, z16.h
    st1b { z17.h }, p0, [x26]
    add x26, x26, x23
    uzp1 z17.h, z21.h, z20.h
    st1b { z16.h }, p0, [x26]
    add x26, x26, x23
    uzp1 z16.h, z19.h, z18.h
    st1b { z17.h }, p0, [x26]
    add x26, x26, x23
    st1b { z16.h }, p0, [x26]
    add x26, x26, x23
    blt label_10
KAI_ASM_LABEL(label_11)  // Store to output array: Accumulator row 0 oddments
    cbz x20, label_12
    KAI_ASM_INST(0xc0820415)  // mova z21.s, p1/M, za0h.s[x12]
    KAI_ASM_INST(0xc0820490)  // mova z16.s, p1/M, za1h.s[x12]
    scvtf z21.s, p1/m, z21.s
    scvtf z16.s, p1/m, z16.s
    KAI_ASM_INST(0xc0820434)  // mova z20.s, p1/M, za0h.s[x12, #1]
    subs x20, x20, #0x1
    KAI_ASM_INST(0xc0820453)  // mova z19.s, p1/M, za0h.s[x12, #2]
    scvtf z20.s, p1/m, z20.s
    KAI_ASM_INST(0xc08204b2)  // mova z18.s, p1/M, za1h.s[x12, #1]
    scvtf z19.s, p1/m, z19.s
    KAI_ASM_INST(0xc08204d1)  // mova z17.s, p1/M, za1h.s[x12, #2]
    scvtf z18.s, p1/m, z18.s
    fmul z21.s, z21.s, z25.s
    scvtf z17.s, p1/m, z17.s
    fmul z16.s, z16.s, z24.s
    fmul z20.s, z20.s, z25.s
    fmul z19.s, z19.s, z25.s
    fmul z18.s, z18.s, z24.s
    frintn z21.s, p1/m, z21.s
    fmul z17.s, z17.s, z24.s
    frintn z16.s, p1/m, z16.s
    frintn z20.s, p1/m, z20.s
    frintn z19.s, p1/m, z19.s
    fcvtzs z21.s, p1/m, z21.s
    frintn z18.s, p1/m, z18.s
    frintn z17.s, p1/m, z17.s
    fcvtzs z16.s, p1/m, z16.s
    fcvtzs z20.s, p1/m, z20.s
    fcvtzs z19.s, p1/m, z19.s
    fcvtzs z18.s, p1/m, z18.s
    add z21.s, z21.s, z26.s
    fcvtzs z17.s, p1/m, z17.s
    add z16.s, z16.s, z26.s
    add z20.s, z20.s, z26.s
    add z19.s, z19.s, z26.s
    add z18.s, z18.s, z26.s
    smin z21.s, p1/M, z21.s, z27.s
    add z17.s, z17.s, z26.s
    smin z16.s, p1/M, z16.s, z27.s
    smin z20.s, p1/M, z20.s, z27.s
    smin z19.s, p1/M, z19.s, z27.s
    smax z21.s, p1/M, z21.s, z28.s
    smin z18.s, p1/M, z18.s, z27.s
    smax z16.s, p1/M, z16.s, z28.s
    smin z17.s, p1/M, z17.s, z27.s
    smax z20.s, p1/M, z20.s, z28.s
    smax z19.s, p1/M, z19.s, z28.s
    smax z18.s, p1/M, z18.s, z28.s
    smax z17.s, p1/M, z17.s, z28.s
    uzp1 z16.h, z21.h, z16.h
    st1b { z16.h }, p0, [x26]
    add x26, x26, x23
    beq label_12
    subs x20, x20, #0x1
    uzp1 z16.h, z20.h, z18.h
    st1b { z16.h }, p0, [x26]
    add x26, x26, x23
    beq label_12
    uzp1 z16.h, z19.h, z17.h
    st1b { z16.h }, p0, [x26]
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
    KAI_ASM_INST(0xc0820517)  // mova z23.s, p1/M, za2h.s[x12]
    KAI_ASM_INST(0xc0820591)  // mova z17.s, p1/M, za3h.s[x12]
    scvtf z23.s, p1/m, z23.s
    KAI_ASM_INST(0xc0820536)  // mova z22.s, p1/M, za2h.s[x12, #1]
    scvtf z17.s, p1/m, z17.s
    KAI_ASM_INST(0xc08205b0)  // mova z16.s, p1/M, za3h.s[x12, #1]
    scvtf z22.s, p1/m, z22.s
    KAI_ASM_INST(0xc0820555)  // mova z21.s, p1/M, za2h.s[x12, #2]
    scvtf z16.s, p1/m, z16.s
    KAI_ASM_INST(0xc08205d4)  // mova z20.s, p1/M, za3h.s[x12, #2]
    scvtf z21.s, p1/m, z21.s
    fmul z23.s, z23.s, z25.s
    KAI_ASM_INST(0xc0820573)  // mova z19.s, p1/M, za2h.s[x12, #3]
    scvtf z20.s, p1/m, z20.s
    fmul z17.s, z17.s, z24.s
    KAI_ASM_INST(0xc08205f2)  // mova z18.s, p1/M, za3h.s[x12, #3]
    scvtf z19.s, p1/m, z19.s
    fmul z22.s, z22.s, z25.s
    add x12, x12, #0x4
    scvtf z18.s, p1/m, z18.s
    fmul z16.s, z16.s, z24.s
    cmp x12, x21, LSL #2
    fmul z21.s, z21.s, z25.s
    frintn z23.s, p1/m, z23.s
    fmul z20.s, z20.s, z24.s
    frintn z17.s, p1/m, z17.s
    fmul z19.s, z19.s, z25.s
    frintn z22.s, p1/m, z22.s
    fmul z18.s, z18.s, z24.s
    frintn z16.s, p1/m, z16.s
    frintn z21.s, p1/m, z21.s
    fcvtzs z23.s, p1/m, z23.s
    frintn z20.s, p1/m, z20.s
    fcvtzs z17.s, p1/m, z17.s
    frintn z19.s, p1/m, z19.s
    fcvtzs z22.s, p1/m, z22.s
    frintn z18.s, p1/m, z18.s
    fcvtzs z16.s, p1/m, z16.s
    fcvtzs z21.s, p1/m, z21.s
    add z23.s, z23.s, z26.s
    fcvtzs z20.s, p1/m, z20.s
    add z17.s, z17.s, z26.s
    fcvtzs z19.s, p1/m, z19.s
    add z22.s, z22.s, z26.s
    fcvtzs z18.s, p1/m, z18.s
    add z16.s, z16.s, z26.s
    add z21.s, z21.s, z26.s
    smin z23.s, p1/M, z23.s, z27.s
    add z20.s, z20.s, z26.s
    smin z17.s, p1/M, z17.s, z27.s
    add z19.s, z19.s, z26.s
    smin z22.s, p1/M, z22.s, z27.s
    add z18.s, z18.s, z26.s
    smin z16.s, p1/M, z16.s, z27.s
    smin z21.s, p1/M, z21.s, z27.s
    smax z23.s, p1/M, z23.s, z28.s
    smin z20.s, p1/M, z20.s, z27.s
    smax z17.s, p1/M, z17.s, z28.s
    smin z19.s, p1/M, z19.s, z27.s
    smax z22.s, p1/M, z22.s, z28.s
    smin z18.s, p1/M, z18.s, z27.s
    smax z16.s, p1/M, z16.s, z28.s
    smax z21.s, p1/M, z21.s, z28.s
    smax z20.s, p1/M, z20.s, z28.s
    uzp1 z17.h, z23.h, z17.h
    smax z19.s, p1/M, z19.s, z28.s
    smax z18.s, p1/M, z18.s, z28.s
    uzp1 z16.h, z22.h, z16.h
    st1b { z17.h }, p0, [x26]
    add x26, x26, x23
    uzp1 z17.h, z21.h, z20.h
    st1b { z16.h }, p0, [x26]
    add x26, x26, x23
    uzp1 z16.h, z19.h, z18.h
    st1b { z17.h }, p0, [x26]
    add x26, x26, x23
    st1b { z16.h }, p0, [x26]
    add x26, x26, x23
    blt label_13
KAI_ASM_LABEL(label_14)  // Store to output array: Accumulator row 1 oddments
    cbz x20, label_15
    KAI_ASM_INST(0xc0820515)  // mova z21.s, p1/M, za2h.s[x12]
    KAI_ASM_INST(0xc0820590)  // mova z16.s, p1/M, za3h.s[x12]
    scvtf z21.s, p1/m, z21.s
    scvtf z16.s, p1/m, z16.s
    KAI_ASM_INST(0xc0820534)  // mova z20.s, p1/M, za2h.s[x12, #1]
    subs x20, x20, #0x1
    KAI_ASM_INST(0xc0820553)  // mova z19.s, p1/M, za2h.s[x12, #2]
    scvtf z20.s, p1/m, z20.s
    KAI_ASM_INST(0xc08205b2)  // mova z18.s, p1/M, za3h.s[x12, #1]
    scvtf z19.s, p1/m, z19.s
    KAI_ASM_INST(0xc08205d1)  // mova z17.s, p1/M, za3h.s[x12, #2]
    scvtf z18.s, p1/m, z18.s
    fmul z21.s, z21.s, z25.s
    scvtf z17.s, p1/m, z17.s
    fmul z16.s, z16.s, z24.s
    fmul z20.s, z20.s, z25.s
    fmul z19.s, z19.s, z25.s
    fmul z18.s, z18.s, z24.s
    frintn z21.s, p1/m, z21.s
    fmul z17.s, z17.s, z24.s
    frintn z16.s, p1/m, z16.s
    frintn z20.s, p1/m, z20.s
    frintn z19.s, p1/m, z19.s
    fcvtzs z21.s, p1/m, z21.s
    frintn z18.s, p1/m, z18.s
    frintn z17.s, p1/m, z17.s
    fcvtzs z16.s, p1/m, z16.s
    fcvtzs z20.s, p1/m, z20.s
    fcvtzs z19.s, p1/m, z19.s
    fcvtzs z18.s, p1/m, z18.s
    add z21.s, z21.s, z26.s
    fcvtzs z17.s, p1/m, z17.s
    add z16.s, z16.s, z26.s
    add z20.s, z20.s, z26.s
    add z19.s, z19.s, z26.s
    add z18.s, z18.s, z26.s
    smin z21.s, p1/M, z21.s, z27.s
    add z17.s, z17.s, z26.s
    smin z16.s, p1/M, z16.s, z27.s
    smin z20.s, p1/M, z20.s, z27.s
    smin z19.s, p1/M, z19.s, z27.s
    smax z21.s, p1/M, z21.s, z28.s
    smin z18.s, p1/M, z18.s, z27.s
    smax z16.s, p1/M, z16.s, z28.s
    smin z17.s, p1/M, z17.s, z27.s
    smax z20.s, p1/M, z20.s, z28.s
    smax z19.s, p1/M, z19.s, z28.s
    smax z18.s, p1/M, z18.s, z28.s
    smax z17.s, p1/M, z17.s, z28.s
    uzp1 z16.h, z21.h, z16.h
    st1b { z16.h }, p0, [x26]
    add x26, x26, x23
    beq label_15
    subs x20, x20, #0x1
    uzp1 z16.h, z20.h, z18.h
    st1b { z16.h }, p0, [x26]
    add x26, x26, x23
    beq label_15
    uzp1 z16.h, z19.h, z17.h
    st1b { z16.h }, p0, [x26]
KAI_ASM_LABEL(label_15)  // Store to output array: Accumulator row 1 oddments: End
KAI_ASM_LABEL(label_16)  // Store to output array: End
    incw x13, ALL, MUL #2
    cmp x13, x10
    blt label_2
    incw x15, ALL, MUL #2
    mov x13, #0x0
    cmp x15, x11
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa)

    KAI_ASM_END
