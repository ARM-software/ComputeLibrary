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

    KAI_ASM_CODE(imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa)
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
    ptrue p2.b
    ldr x14, [x0, #0x30]
    mov x13, #0x0
    ldr w11, [x0, #0x20]
    ldr w10, [x0, #0x28]
    ldr x9, [x0, #0x0]
KAI_ASM_LABEL(label_1)  // M loop
    ldr x28, [x0, #0x8]
KAI_ASM_LABEL(label_2)  // N loop
    fmov z18.s, #1.0
    ld1w { z17.s }, p2/Z, [x28]
    mov x20, x13
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    ld1w { z16.s }, p2/Z, [x28, #1, MUL VL]
    whilelt p1.s, x20, x10
    incw x20
    mov x27, x9
    whilelt p0.s, x20, x10
    addvl x28, x28, #2
    KAI_ASM_INST(0x80914a40)  // fmopa za0.s, p2/M, p2/M, z18.s, z17.s
    KAI_ASM_INST(0x80904a41)  // fmopa za1.s, p2/M, p2/M, z18.s, z16.s
    KAI_ASM_INST(0x80914a42)  // fmopa za2.s, p2/M, p2/M, z18.s, z17.s
    KAI_ASM_INST(0x80904a43)  // fmopa za3.s, p2/M, p2/M, z18.s, z16.s
    lsr x21, x14, #0x2
    and x20, x14, #0x3
    cbz x21, label_6
    subs x21, x21, #0x1
    ld1w { z31.s }, p2/Z, [x27]
    ld1w { z30.s }, p2/Z, [x27, #1, MUL VL]
    ld1w { z29.s }, p2/Z, [x27, #2, MUL VL]
    ld1w { z28.s }, p2/Z, [x27, #3, MUL VL]
    ld1w { z27.s }, p2/Z, [x27, #4, MUL VL]
    ld1w { z26.s }, p2/Z, [x27, #5, MUL VL]
    ld1w { z25.s }, p2/Z, [x27, #6, MUL VL]
    ld1w { z24.s }, p2/Z, [x27, #7, MUL VL]
    addvl x27, x27, #8
    ld1w { z23.s }, p2/Z, [x28]
    ld1w { z22.s }, p2/Z, [x28, #1, MUL VL]
    ld1w { z21.s }, p2/Z, [x28, #2, MUL VL]
    ld1w { z20.s }, p2/Z, [x28, #3, MUL VL]
    ld1w { z19.s }, p2/Z, [x28, #4, MUL VL]
    ld1w { z18.s }, p2/Z, [x28, #5, MUL VL]
    ld1w { z17.s }, p2/Z, [x28, #6, MUL VL]
    ld1w { z16.s }, p2/Z, [x28, #7, MUL VL]
    addvl x28, x28, #8
    ble label_5
KAI_ASM_LABEL(label_4)  // K loop
    KAI_ASM_INST(0x80974be0)  // fmopa za0.s, p2/M, p2/M, z31.s, z23.s
    subs x21, x21, #0x1
    KAI_ASM_INST(0x80964be1)  // fmopa za1.s, p2/M, p2/M, z31.s, z22.s
    ld1w { z31.s }, p2/Z, [x27]
    KAI_ASM_INST(0x80974bc2)  // fmopa za2.s, p2/M, p2/M, z30.s, z23.s
    ld1w { z23.s }, p2/Z, [x28]
    KAI_ASM_INST(0x80964bc3)  // fmopa za3.s, p2/M, p2/M, z30.s, z22.s
    ld1w { z30.s }, p2/Z, [x27, #1, MUL VL]
    KAI_ASM_INST(0x80954ba0)  // fmopa za0.s, p2/M, p2/M, z29.s, z21.s
    ld1w { z22.s }, p2/Z, [x28, #1, MUL VL]
    KAI_ASM_INST(0x80944ba1)  // fmopa za1.s, p2/M, p2/M, z29.s, z20.s
    ld1w { z29.s }, p2/Z, [x27, #2, MUL VL]
    KAI_ASM_INST(0x80954b82)  // fmopa za2.s, p2/M, p2/M, z28.s, z21.s
    ld1w { z21.s }, p2/Z, [x28, #2, MUL VL]
    KAI_ASM_INST(0x80944b83)  // fmopa za3.s, p2/M, p2/M, z28.s, z20.s
    ld1w { z28.s }, p2/Z, [x27, #3, MUL VL]
    KAI_ASM_INST(0x80934b60)  // fmopa za0.s, p2/M, p2/M, z27.s, z19.s
    ld1w { z20.s }, p2/Z, [x28, #3, MUL VL]
    KAI_ASM_INST(0x80924b61)  // fmopa za1.s, p2/M, p2/M, z27.s, z18.s
    ld1w { z27.s }, p2/Z, [x27, #4, MUL VL]
    KAI_ASM_INST(0x80934b42)  // fmopa za2.s, p2/M, p2/M, z26.s, z19.s
    ld1w { z19.s }, p2/Z, [x28, #4, MUL VL]
    KAI_ASM_INST(0x80924b43)  // fmopa za3.s, p2/M, p2/M, z26.s, z18.s
    ld1w { z26.s }, p2/Z, [x27, #5, MUL VL]
    KAI_ASM_INST(0x80914b20)  // fmopa za0.s, p2/M, p2/M, z25.s, z17.s
    ld1w { z18.s }, p2/Z, [x28, #5, MUL VL]
    KAI_ASM_INST(0x80904b21)  // fmopa za1.s, p2/M, p2/M, z25.s, z16.s
    ld1w { z25.s }, p2/Z, [x27, #6, MUL VL]
    KAI_ASM_INST(0x80914b02)  // fmopa za2.s, p2/M, p2/M, z24.s, z17.s
    ld1w { z17.s }, p2/Z, [x28, #6, MUL VL]
    KAI_ASM_INST(0x80904b03)  // fmopa za3.s, p2/M, p2/M, z24.s, z16.s
    ld1w { z24.s }, p2/Z, [x27, #7, MUL VL]
    addvl x27, x27, #8
    ld1w { z16.s }, p2/Z, [x28, #7, MUL VL]
    addvl x28, x28, #8
    bgt label_4
KAI_ASM_LABEL(label_5)  // K loop tail
    KAI_ASM_INST(0x80974be0)  // fmopa za0.s, p2/M, p2/M, z31.s, z23.s
    KAI_ASM_INST(0x80964be1)  // fmopa za1.s, p2/M, p2/M, z31.s, z22.s
    KAI_ASM_INST(0x80974bc2)  // fmopa za2.s, p2/M, p2/M, z30.s, z23.s
    KAI_ASM_INST(0x80964bc3)  // fmopa za3.s, p2/M, p2/M, z30.s, z22.s
    KAI_ASM_INST(0x80954ba0)  // fmopa za0.s, p2/M, p2/M, z29.s, z21.s
    KAI_ASM_INST(0x80944ba1)  // fmopa za1.s, p2/M, p2/M, z29.s, z20.s
    KAI_ASM_INST(0x80954b82)  // fmopa za2.s, p2/M, p2/M, z28.s, z21.s
    KAI_ASM_INST(0x80944b83)  // fmopa za3.s, p2/M, p2/M, z28.s, z20.s
    KAI_ASM_INST(0x80934b60)  // fmopa za0.s, p2/M, p2/M, z27.s, z19.s
    KAI_ASM_INST(0x80924b61)  // fmopa za1.s, p2/M, p2/M, z27.s, z18.s
    KAI_ASM_INST(0x80934b42)  // fmopa za2.s, p2/M, p2/M, z26.s, z19.s
    KAI_ASM_INST(0x80924b43)  // fmopa za3.s, p2/M, p2/M, z26.s, z18.s
    KAI_ASM_INST(0x80914b20)  // fmopa za0.s, p2/M, p2/M, z25.s, z17.s
    KAI_ASM_INST(0x80904b21)  // fmopa za1.s, p2/M, p2/M, z25.s, z16.s
    KAI_ASM_INST(0x80914b02)  // fmopa za2.s, p2/M, p2/M, z24.s, z17.s
    KAI_ASM_INST(0x80904b03)  // fmopa za3.s, p2/M, p2/M, z24.s, z16.s
KAI_ASM_LABEL(label_6)  // K oddments
    cbz x20, label_8
KAI_ASM_LABEL(label_7)  // K oddments: Loop
    ld1w { z19.s }, p2/Z, [x27]
    subs x20, x20, #0x1
    ld1w { z18.s }, p2/Z, [x27, #1, MUL VL]
    addvl x27, x27, #2
    ld1w { z17.s }, p2/Z, [x28]
    ld1w { z16.s }, p2/Z, [x28, #1, MUL VL]
    addvl x28, x28, #2
    KAI_ASM_INST(0x80914a60)  // fmopa za0.s, p2/M, p2/M, z19.s, z17.s
    KAI_ASM_INST(0x80904a61)  // fmopa za1.s, p2/M, p2/M, z19.s, z16.s
    KAI_ASM_INST(0x80914a42)  // fmopa za2.s, p2/M, p2/M, z18.s, z17.s
    KAI_ASM_INST(0x80904a43)  // fmopa za3.s, p2/M, p2/M, z18.s, z16.s
    bgt label_7
KAI_ASM_LABEL(label_8)  // K oddments: End
    ldr x26, [x0, #0x10]
    sub x25, x11, x15
    cntw x24
    KAI_ASM_INST(0x854ec819)  // ld1rw { z25.s }, p2/Z, [x0, #56]
    ldr x23, [x0, #0x18]
    cmp x25, x24
    KAI_ASM_INST(0x854fc818)  // ld1rw { z24.s }, p2/Z, [x0, #60]
    mov x12, #0x0
    csel x22, x25, x24, LT
    add x26, x26, x13, LSL #2  // C += n
    lsr x21, x22, #0x2
    madd x26, x15, x23, x26  // C += m * ldc
    and x20, x22, #0x3
    cbz x21, label_11
KAI_ASM_LABEL(label_10)  // Store to output array: Accumulator row 0 loop
    KAI_ASM_INST(0xc0820817)  // mova z23.s, p2/M, za0h.s[x12]
    KAI_ASM_INST(0xc0820896)  // mova z22.s, p2/M, za1h.s[x12]
    fmin z23.s, p2/M, z23.s, z24.s
    KAI_ASM_INST(0xc0820835)  // mova z21.s, p2/M, za0h.s[x12, #1]
    fmin z22.s, p2/M, z22.s, z24.s
    KAI_ASM_INST(0xc08208b4)  // mova z20.s, p2/M, za1h.s[x12, #1]
    fmin z21.s, p2/M, z21.s, z24.s
    KAI_ASM_INST(0xc0820853)  // mova z19.s, p2/M, za0h.s[x12, #2]
    fmin z20.s, p2/M, z20.s, z24.s
    KAI_ASM_INST(0xc08208d2)  // mova z18.s, p2/M, za1h.s[x12, #2]
    fmin z19.s, p2/M, z19.s, z24.s
    fmax z23.s, p2/M, z23.s, z25.s
    KAI_ASM_INST(0xc0820871)  // mova z17.s, p2/M, za0h.s[x12, #3]
    fmin z18.s, p2/M, z18.s, z24.s
    fmax z22.s, p2/M, z22.s, z25.s
    KAI_ASM_INST(0xc08208f0)  // mova z16.s, p2/M, za1h.s[x12, #3]
    fmin z17.s, p2/M, z17.s, z24.s
    fmax z21.s, p2/M, z21.s, z25.s
    add x12, x12, #0x4
    fmin z16.s, p2/M, z16.s, z24.s
    fmax z20.s, p2/M, z20.s, z25.s
    cmp x12, x21, LSL #2
    st1w { z23.s }, p1, [x26]
    fmax z19.s, p2/M, z19.s, z25.s
    st1w { z22.s }, p0, [x26, #1, MUL VL]
    add x26, x26, x23
    fmax z18.s, p2/M, z18.s, z25.s
    st1w { z21.s }, p1, [x26]
    fmax z17.s, p2/M, z17.s, z25.s
    st1w { z20.s }, p0, [x26, #1, MUL VL]
    add x26, x26, x23
    fmax z16.s, p2/M, z16.s, z25.s
    st1w { z19.s }, p1, [x26]
    st1w { z18.s }, p0, [x26, #1, MUL VL]
    add x26, x26, x23
    st1w { z17.s }, p1, [x26]
    st1w { z16.s }, p0, [x26, #1, MUL VL]
    add x26, x26, x23
    blt label_10
KAI_ASM_LABEL(label_11)  // Store to output array: Accumulator row 0 oddments
    cbz x20, label_12
    KAI_ASM_INST(0xc0820815)  // mova z21.s, p2/M, za0h.s[x12]
    KAI_ASM_INST(0xc0820834)  // mova z20.s, p2/M, za0h.s[x12, #1]
    fmin z21.s, p2/M, z21.s, z24.s
    KAI_ASM_INST(0xc0820853)  // mova z19.s, p2/M, za0h.s[x12, #2]
    fmin z20.s, p2/M, z20.s, z24.s
    subs x20, x20, #0x1
    KAI_ASM_INST(0xc0820892)  // mova z18.s, p2/M, za1h.s[x12]
    fmin z19.s, p2/M, z19.s, z24.s
    KAI_ASM_INST(0xc08208b1)  // mova z17.s, p2/M, za1h.s[x12, #1]
    fmin z18.s, p2/M, z18.s, z24.s
    KAI_ASM_INST(0xc08208d0)  // mova z16.s, p2/M, za1h.s[x12, #2]
    fmin z17.s, p2/M, z17.s, z24.s
    fmax z21.s, p2/M, z21.s, z25.s
    fmin z16.s, p2/M, z16.s, z24.s
    fmax z20.s, p2/M, z20.s, z25.s
    fmax z19.s, p2/M, z19.s, z25.s
    fmax z18.s, p2/M, z18.s, z25.s
    fmax z17.s, p2/M, z17.s, z25.s
    st1w { z21.s }, p1, [x26]
    fmax z16.s, p2/M, z16.s, z25.s
    st1w { z18.s }, p0, [x26, #1, MUL VL]
    add x26, x26, x23
    beq label_12
    subs x20, x20, #0x1
    st1w { z20.s }, p1, [x26]
    st1w { z17.s }, p0, [x26, #1, MUL VL]
    add x26, x26, x23
    beq label_12
    st1w { z19.s }, p1, [x26]
    st1w { z16.s }, p0, [x26, #1, MUL VL]
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
    KAI_ASM_INST(0xc0820917)  // mova z23.s, p2/M, za2h.s[x12]
    KAI_ASM_INST(0xc0820996)  // mova z22.s, p2/M, za3h.s[x12]
    fmin z23.s, p2/M, z23.s, z24.s
    KAI_ASM_INST(0xc0820935)  // mova z21.s, p2/M, za2h.s[x12, #1]
    fmin z22.s, p2/M, z22.s, z24.s
    KAI_ASM_INST(0xc08209b4)  // mova z20.s, p2/M, za3h.s[x12, #1]
    fmin z21.s, p2/M, z21.s, z24.s
    KAI_ASM_INST(0xc0820953)  // mova z19.s, p2/M, za2h.s[x12, #2]
    fmin z20.s, p2/M, z20.s, z24.s
    KAI_ASM_INST(0xc08209d2)  // mova z18.s, p2/M, za3h.s[x12, #2]
    fmin z19.s, p2/M, z19.s, z24.s
    fmax z23.s, p2/M, z23.s, z25.s
    KAI_ASM_INST(0xc0820971)  // mova z17.s, p2/M, za2h.s[x12, #3]
    fmin z18.s, p2/M, z18.s, z24.s
    fmax z22.s, p2/M, z22.s, z25.s
    KAI_ASM_INST(0xc08209f0)  // mova z16.s, p2/M, za3h.s[x12, #3]
    fmin z17.s, p2/M, z17.s, z24.s
    fmax z21.s, p2/M, z21.s, z25.s
    add x12, x12, #0x4
    fmin z16.s, p2/M, z16.s, z24.s
    fmax z20.s, p2/M, z20.s, z25.s
    cmp x12, x21, LSL #2
    st1w { z23.s }, p1, [x26]
    fmax z19.s, p2/M, z19.s, z25.s
    st1w { z22.s }, p0, [x26, #1, MUL VL]
    add x26, x26, x23
    fmax z18.s, p2/M, z18.s, z25.s
    st1w { z21.s }, p1, [x26]
    fmax z17.s, p2/M, z17.s, z25.s
    st1w { z20.s }, p0, [x26, #1, MUL VL]
    add x26, x26, x23
    fmax z16.s, p2/M, z16.s, z25.s
    st1w { z19.s }, p1, [x26]
    st1w { z18.s }, p0, [x26, #1, MUL VL]
    add x26, x26, x23
    st1w { z17.s }, p1, [x26]
    st1w { z16.s }, p0, [x26, #1, MUL VL]
    add x26, x26, x23
    blt label_13
KAI_ASM_LABEL(label_14)  // Store to output array: Accumulator row 1 oddments
    cbz x20, label_15
    KAI_ASM_INST(0xc0820915)  // mova z21.s, p2/M, za2h.s[x12]
    KAI_ASM_INST(0xc0820934)  // mova z20.s, p2/M, za2h.s[x12, #1]
    fmin z21.s, p2/M, z21.s, z24.s
    KAI_ASM_INST(0xc0820953)  // mova z19.s, p2/M, za2h.s[x12, #2]
    fmin z20.s, p2/M, z20.s, z24.s
    subs x20, x20, #0x1
    KAI_ASM_INST(0xc0820992)  // mova z18.s, p2/M, za3h.s[x12]
    fmin z19.s, p2/M, z19.s, z24.s
    KAI_ASM_INST(0xc08209b1)  // mova z17.s, p2/M, za3h.s[x12, #1]
    fmin z18.s, p2/M, z18.s, z24.s
    KAI_ASM_INST(0xc08209d0)  // mova z16.s, p2/M, za3h.s[x12, #2]
    fmin z17.s, p2/M, z17.s, z24.s
    fmax z21.s, p2/M, z21.s, z25.s
    fmin z16.s, p2/M, z16.s, z24.s
    fmax z20.s, p2/M, z20.s, z25.s
    fmax z19.s, p2/M, z19.s, z25.s
    fmax z18.s, p2/M, z18.s, z25.s
    fmax z17.s, p2/M, z17.s, z25.s
    st1w { z21.s }, p1, [x26]
    fmax z16.s, p2/M, z16.s, z25.s
    st1w { z18.s }, p0, [x26, #1, MUL VL]
    add x26, x26, x23
    beq label_15
    subs x20, x20, #0x1
    st1w { z20.s }, p1, [x26]
    st1w { z17.s }, p0, [x26, #1, MUL VL]
    add x26, x26, x23
    beq label_15
    st1w { z19.s }, p1, [x26]
    st1w { z16.s }, p0, [x26, #1, MUL VL]
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
    KAI_ASM_FUNCTION_END(kai_kernel_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa)

    KAI_ASM_END
