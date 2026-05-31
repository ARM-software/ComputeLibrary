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

    KAI_ASM_CODE(imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa)
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
    KAI_ASM_INST(0x25207810)  // ptrue pn8.b
    ldr w13, [x0, #0x20]
    mov x11, #0x0
    ldr w10, [x0, #0x28]
    add x14, x14, #0x3
    ldr x9, [x0, #0x0]
    lsr x14, x14, #0x2
KAI_ASM_LABEL(label_1)  // M loop
    ldr x28, [x0, #0x8]
KAI_ASM_LABEL(label_2)  // N loop
    KAI_ASM_INST(0xa0404382)  // ld1w { z2.s-z3.s }, pn8.b/Z, [x28]
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    mov x27, x9
    addvl x28, x28, #2
    KAI_ASM_INST(0xc0902440)  // addha za0.s, p1/M, p1/M, z2.s
    KAI_ASM_INST(0xc0902461)  // addha za1.s, p1/M, p1/M, z3.s
    KAI_ASM_INST(0xc0902442)  // addha za2.s, p1/M, p1/M, z2.s
    KAI_ASM_INST(0xc0902463)  // addha za3.s, p1/M, p1/M, z3.s
    lsr x21, x14, #0x2
    and x20, x14, #0x3
    cbz x21, label_6
    subs x21, x21, #0x1
    KAI_ASM_INST(0xa1408362)  // ld1b { z2.b, z6.b, z10.b, z14.b }, pn8.b/Z, [x27]
    KAI_ASM_INST(0xa1418360)  // ld1b { z0.b, z4.b, z8.b, z12.b }, pn8.b/Z, [x27, #0x4, MUL VL]
    addvl x27, x27, #8
    KAI_ASM_INST(0xa0408390)  // ld1b { z16.b-z19.b }, pn8.b/Z, [x28]
    KAI_ASM_INST(0xa041839c)  // ld1b { z28.b-z31.b }, pn8.b/Z, [x28, #0x4, MUL VL]
    addvl x28, x28, #8
    ble label_5
KAI_ASM_LABEL(label_4)  // K loop
    KAI_ASM_INST(0xa0902440)  // smopa za0.s, p1/M, p1/M, z2.b, z16.b
    subs x21, x21, #0x1
    KAI_ASM_INST(0xa0912441)  // smopa za1.s, p1/M, p1/M, z2.b, z17.b
    KAI_ASM_INST(0xa09024c2)  // smopa za2.s, p1/M, p1/M, z6.b, z16.b
    KAI_ASM_INST(0xa09124c3)  // smopa za3.s, p1/M, p1/M, z6.b, z17.b
    KAI_ASM_INST(0xa0922540)  // smopa za0.s, p1/M, p1/M, z10.b, z18.b
    KAI_ASM_INST(0xa0932541)  // smopa za1.s, p1/M, p1/M, z10.b, z19.b
    KAI_ASM_INST(0xa09225c2)  // smopa za2.s, p1/M, p1/M, z14.b, z18.b
    KAI_ASM_INST(0xa09325c3)  // smopa za3.s, p1/M, p1/M, z14.b, z19.b
    KAI_ASM_INST(0xa1408362)  // ld1b { z2.b, z6.b, z10.b, z14.b }, pn8.b/Z, [x27]
    KAI_ASM_INST(0xa09c2400)  // smopa za0.s, p1/M, p1/M, z0.b, z28.b
    KAI_ASM_INST(0xa0408390)  // ld1b { z16.b-z19.b }, pn8.b/Z, [x28]
    KAI_ASM_INST(0xa09d2401)  // smopa za1.s, p1/M, p1/M, z0.b, z29.b
    KAI_ASM_INST(0xa09c2482)  // smopa za2.s, p1/M, p1/M, z4.b, z28.b
    KAI_ASM_INST(0xa09d2483)  // smopa za3.s, p1/M, p1/M, z4.b, z29.b
    KAI_ASM_INST(0xa09e2500)  // smopa za0.s, p1/M, p1/M, z8.b, z30.b
    KAI_ASM_INST(0xa09f2501)  // smopa za1.s, p1/M, p1/M, z8.b, z31.b
    KAI_ASM_INST(0xa09e2582)  // smopa za2.s, p1/M, p1/M, z12.b, z30.b
    KAI_ASM_INST(0xa09f2583)  // smopa za3.s, p1/M, p1/M, z12.b, z31.b
    KAI_ASM_INST(0xa1418360)  // ld1b { z0.b, z4.b, z8.b, z12.b }, pn8.b/Z, [x27, #0x4, MUL VL]
    addvl x27, x27, #8
    KAI_ASM_INST(0xa041839c)  // ld1b { z28.b-z31.b }, pn8.b/Z, [x28, #0x4, MUL VL]
    addvl x28, x28, #8
    bgt label_4
KAI_ASM_LABEL(label_5)  // K loop tail
    KAI_ASM_INST(0xa0902440)  // smopa za0.s, p1/M, p1/M, z2.b, z16.b
    KAI_ASM_INST(0xa0912441)  // smopa za1.s, p1/M, p1/M, z2.b, z17.b
    KAI_ASM_INST(0xa09024c2)  // smopa za2.s, p1/M, p1/M, z6.b, z16.b
    KAI_ASM_INST(0xa09124c3)  // smopa za3.s, p1/M, p1/M, z6.b, z17.b
    KAI_ASM_INST(0xa0922540)  // smopa za0.s, p1/M, p1/M, z10.b, z18.b
    KAI_ASM_INST(0xa0932541)  // smopa za1.s, p1/M, p1/M, z10.b, z19.b
    KAI_ASM_INST(0xa09225c2)  // smopa za2.s, p1/M, p1/M, z14.b, z18.b
    KAI_ASM_INST(0xa09325c3)  // smopa za3.s, p1/M, p1/M, z14.b, z19.b
    KAI_ASM_INST(0xa09c2400)  // smopa za0.s, p1/M, p1/M, z0.b, z28.b
    KAI_ASM_INST(0xa09d2401)  // smopa za1.s, p1/M, p1/M, z0.b, z29.b
    KAI_ASM_INST(0xa09c2482)  // smopa za2.s, p1/M, p1/M, z4.b, z28.b
    KAI_ASM_INST(0xa09d2483)  // smopa za3.s, p1/M, p1/M, z4.b, z29.b
    KAI_ASM_INST(0xa09e2500)  // smopa za0.s, p1/M, p1/M, z8.b, z30.b
    KAI_ASM_INST(0xa09f2501)  // smopa za1.s, p1/M, p1/M, z8.b, z31.b
    KAI_ASM_INST(0xa09e2582)  // smopa za2.s, p1/M, p1/M, z12.b, z30.b
    KAI_ASM_INST(0xa09f2583)  // smopa za3.s, p1/M, p1/M, z12.b, z31.b
KAI_ASM_LABEL(label_6)  // K oddments
    cbz x20, label_8
KAI_ASM_LABEL(label_7)  // K oddments: Loop
    KAI_ASM_INST(0xa0400370)  // ld1b { z16.b-z17.b }, pn8.b/Z, [x27]
    subs x20, x20, #0x1
    addvl x27, x27, #2
    KAI_ASM_INST(0xa1400385)  // ld1b { z5.b, z13.b }, pn8.b/Z, [x28]
    addvl x28, x28, #2
    KAI_ASM_INST(0xa0852600)  // smopa za0.s, p1/M, p1/M, z16.b, z5.b
    KAI_ASM_INST(0xa08d2601)  // smopa za1.s, p1/M, p1/M, z16.b, z13.b
    KAI_ASM_INST(0xa0852622)  // smopa za2.s, p1/M, p1/M, z17.b, z5.b
    KAI_ASM_INST(0xa08d2623)  // smopa za3.s, p1/M, p1/M, z17.b, z13.b
    bgt label_7
KAI_ASM_LABEL(label_8)  // K oddments: End
    ldr x26, [x0, #0x10]
    sub x25, x13, x15
    cntw x24
    KAI_ASM_INST(0x854ec41a)  // ld1rw { z26.s }, p1/Z, [x0, #56]
    ldr x23, [x0, #0x18]
    whilelt p0.h, x11, x10
    cmp x25, x24
    KAI_ASM_INST(0x854fc417)  // ld1rw { z23.s }, p1/Z, [x0, #60]
    csel x22, x25, x24, LT
    KAI_ASM_INST(0x8550c400)  // ld1rw { z0.s }, p1/Z, [x0, #64]
    mov x12, #0x0
    add x26, x26, x11  // C += n
    lsr x21, x22, #0x2
    KAI_ASM_INST(0xa0404382)  // ld1w { z2.s-z3.s }, pn8.b/Z, [x28]
    madd x26, x15, x23, x26  // C += m * ldc
    addvl x28, x28, #2
    and x20, x22, #0x3
    cbz x21, label_11
KAI_ASM_LABEL(label_10)  // Store to output array: Accumulator row 0 loop
    KAI_ASM_INST(0xc0860408)  // mova { z8.s-z11.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860430)  // mova { z16.s-z19.s }, za1h.s[x12]
    KAI_ASM_INST(0xc132e108)  // scvtf { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc132e210)  // scvtf { z16.s-z19.s }, { z16.s-z19.s }
    fmul z8.s, z8.s, z2.s
    fmul z9.s, z9.s, z2.s
    add x12, x12, #0x4
    fmul z10.s, z10.s, z2.s
    fmul z11.s, z11.s, z2.s
    cmp x12, x21, LSL #2
    fmul z16.s, z16.s, z3.s
    fmul z17.s, z17.s, z3.s
    fmul z18.s, z18.s, z3.s
    fmul z19.s, z19.s, z3.s
    KAI_ASM_INST(0xc1b8e108)  // frintn { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc131e108)  // fcvtzs { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc1b8e210)  // frintn { z16.s-z19.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc1a0ab08)  // add { z8.s-z11.s }, { z8.s-z11.s }, z0.s
    KAI_ASM_INST(0xc131e210)  // fcvtzs { z16.s-z19.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc1a0ab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z0.s
    KAI_ASM_INST(0xc1b7cf48)  // sclamp { z8.s-z11.s }, z26.s, z23.s
    KAI_ASM_INST(0xc1b7cf50)  // sclamp { z16.s-z19.s }, z26.s, z23.s
    uzp1 z5.h, z8.h, z16.h
    uzp1 z14.h, z9.h, z17.h
    uzp1 z17.h, z10.h, z18.h
    uzp1 z16.h, z11.h, z19.h
    st1b { z5.h }, p0, [x26]
    add x26, x26, x23
    st1b { z14.h }, p0, [x26]
    add x26, x26, x23
    st1b { z17.h }, p0, [x26]
    add x26, x26, x23
    st1b { z16.h }, p0, [x26]
    add x26, x26, x23
    blt label_10
KAI_ASM_LABEL(label_11)  // Store to output array: Accumulator row 0 oddments
    cbz x20, label_12
    KAI_ASM_INST(0xc0860408)  // mova { z8.s-z11.s }, za0h.s[x12]
    KAI_ASM_INST(0xc086042c)  // mova { z12.s-z15.s }, za1h.s[x12]
    KAI_ASM_INST(0xc132e108)  // scvtf { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc132e18c)  // scvtf { z12.s-z15.s }, { z12.s-z15.s }
    fmul z8.s, z8.s, z2.s
    fmul z9.s, z9.s, z2.s
    subs x20, x20, #0x1
    fmul z10.s, z10.s, z2.s
    fmul z11.s, z11.s, z2.s
    fmul z12.s, z12.s, z3.s
    fmul z13.s, z13.s, z3.s
    fmul z14.s, z14.s, z3.s
    fmul z15.s, z15.s, z3.s
    KAI_ASM_INST(0xc1b8e108)  // frintn { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc131e108)  // fcvtzs { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc1b8e18c)  // frintn { z12.s-z15.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc1a0ab08)  // add { z8.s-z11.s }, { z8.s-z11.s }, z0.s
    KAI_ASM_INST(0xc131e18c)  // fcvtzs { z12.s-z15.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc1a0ab0c)  // add { z12.s-z15.s }, { z12.s-z15.s }, z0.s
    KAI_ASM_INST(0xc1b7cf48)  // sclamp { z8.s-z11.s }, z26.s, z23.s
    KAI_ASM_INST(0xc1b7cf4c)  // sclamp { z12.s-z15.s }, z26.s, z23.s
    uzp1 z16.h, z8.h, z12.h
    st1b { z16.h }, p0, [x26]
    add x26, x26, x23
    beq label_12
    subs x20, x20, #0x1
    uzp1 z16.h, z9.h, z13.h
    st1b { z16.h }, p0, [x26]
    add x26, x26, x23
    beq label_12
    uzp1 z16.h, z10.h, z14.h
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
    KAI_ASM_INST(0xc0860448)  // mova { z8.s-z11.s }, za2h.s[x12]
    KAI_ASM_INST(0xc0860470)  // mova { z16.s-z19.s }, za3h.s[x12]
    KAI_ASM_INST(0xc132e108)  // scvtf { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc132e210)  // scvtf { z16.s-z19.s }, { z16.s-z19.s }
    fmul z8.s, z8.s, z2.s
    fmul z9.s, z9.s, z2.s
    add x12, x12, #0x4
    fmul z10.s, z10.s, z2.s
    fmul z11.s, z11.s, z2.s
    cmp x12, x21, LSL #2
    fmul z16.s, z16.s, z3.s
    fmul z17.s, z17.s, z3.s
    fmul z18.s, z18.s, z3.s
    fmul z19.s, z19.s, z3.s
    KAI_ASM_INST(0xc1b8e108)  // frintn { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc131e108)  // fcvtzs { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc1b8e210)  // frintn { z16.s-z19.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc1a0ab08)  // add { z8.s-z11.s }, { z8.s-z11.s }, z0.s
    KAI_ASM_INST(0xc131e210)  // fcvtzs { z16.s-z19.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc1a0ab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z0.s
    KAI_ASM_INST(0xc1b7cf48)  // sclamp { z8.s-z11.s }, z26.s, z23.s
    KAI_ASM_INST(0xc1b7cf50)  // sclamp { z16.s-z19.s }, z26.s, z23.s
    uzp1 z21.h, z8.h, z16.h
    uzp1 z20.h, z9.h, z17.h
    uzp1 z17.h, z10.h, z18.h
    uzp1 z16.h, z11.h, z19.h
    st1b { z21.h }, p0, [x26]
    add x26, x26, x23
    st1b { z20.h }, p0, [x26]
    add x26, x26, x23
    st1b { z17.h }, p0, [x26]
    add x26, x26, x23
    st1b { z16.h }, p0, [x26]
    add x26, x26, x23
    blt label_13
KAI_ASM_LABEL(label_14)  // Store to output array: Accumulator row 1 oddments
    cbz x20, label_15
    KAI_ASM_INST(0xc086044c)  // mova { z12.s-z15.s }, za2h.s[x12]
    KAI_ASM_INST(0xc0860464)  // mova { z4.s-z7.s }, za3h.s[x12]
    KAI_ASM_INST(0xc132e18c)  // scvtf { z12.s-z15.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc132e084)  // scvtf { z4.s-z7.s }, { z4.s-z7.s }
    fmul z12.s, z12.s, z2.s
    fmul z13.s, z13.s, z2.s
    subs x20, x20, #0x1
    fmul z14.s, z14.s, z2.s
    fmul z15.s, z15.s, z2.s
    fmul z4.s, z4.s, z3.s
    fmul z5.s, z5.s, z3.s
    fmul z6.s, z6.s, z3.s
    fmul z7.s, z7.s, z3.s
    KAI_ASM_INST(0xc1b8e18c)  // frintn { z12.s-z15.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc131e18c)  // fcvtzs { z12.s-z15.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc1b8e084)  // frintn { z4.s-z7.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc1a0ab0c)  // add { z12.s-z15.s }, { z12.s-z15.s }, z0.s
    KAI_ASM_INST(0xc131e084)  // fcvtzs { z4.s-z7.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc1a0ab04)  // add { z4.s-z7.s }, { z4.s-z7.s }, z0.s
    KAI_ASM_INST(0xc1b7cf4c)  // sclamp { z12.s-z15.s }, z26.s, z23.s
    KAI_ASM_INST(0xc1b7cf44)  // sclamp { z4.s-z7.s }, z26.s, z23.s
    uzp1 z16.h, z12.h, z4.h
    st1b { z16.h }, p0, [x26]
    add x26, x26, x23
    beq label_15
    subs x20, x20, #0x1
    uzp1 z16.h, z13.h, z5.h
    st1b { z16.h }, p0, [x26]
    add x26, x26, x23
    beq label_15
    uzp1 z16.h, z14.h, z6.h
    st1b { z16.h }, p0, [x26]
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
    KAI_ASM_FUNCTION_END(kai_kernel_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa)

    KAI_ASM_END
