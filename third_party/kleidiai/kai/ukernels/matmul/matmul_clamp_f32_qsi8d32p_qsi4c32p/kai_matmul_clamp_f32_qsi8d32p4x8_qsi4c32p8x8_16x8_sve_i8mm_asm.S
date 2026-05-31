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

    KAI_ASM_CODE(matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    mov x21, #0x4
    mov z10.s, #-0x4
    mov x17, #0x88
    ptrue p5.b
    ldr x20, [x0, #0x28]
    ldr x16, [x0, #0x38]
    whilelt p4.s, XZR, x21
    ldr x15, [x0, #0x8]
    ldr x14, [x0, #0x10]
    ldr x13, [x0, #0x30]
    ldr x12, [x0, #0x0]
    mov x11, x20
    ldr x10, [x0, #0x20]
    mul x17, x16, x17
    ldr x9, [x0, #0x18]
    cmp x11, #0x10
    blt label_9
KAI_ASM_LABEL(label_1)  // Row loop
    mov x28, x14
    mov x27, x13
    add x25, x12, x10, LSL #4
KAI_ASM_LABEL(label_2)  // Column loop
    mov x26, x15
    mov z31.b, #0x0
    mov z30.b, #0x0
    mov x23, x16
    mov z29.b, #0x0
    mov z28.b, #0x0
    mov z27.b, #0x0
    mov z26.b, #0x0
    add x22, x26, x17
    add x21, x22, x17
    mov z25.b, #0x0
    mov z24.b, #0x0
    add x20, x21, x17
    mov z23.b, #0x0
    mov z22.b, #0x0
    mov z21.b, #0x0
    mov z20.b, #0x0
    mov z19.b, #0x0
    mov z18.b, #0x0
    mov z17.b, #0x0
    mov z16.b, #0x0
KAI_ASM_LABEL(label_3)  // Block loop
    ld1h { z15.s }, p5/Z, [x28]
    add x28, x28, #0x10
    ld1h { z4.s }, p4/Z, [x26]
    add x26, x26, #0x8
    ld1b { z13.b }, p5/Z, [x28]
    ld1b { z7.b }, p5/Z, [x28, #1, MUL VL]
    mov z8.s, #0x0
    mov z0.s, #0x0
    ld1rqb { z11.b }, p5/Z, [x26]
    ld1rqb { z5.b }, p5/Z, [x26, #16]
    mov z9.s, #0x0
    mov z6.s, #0x0
    ld1b { z2.b }, p5/Z, [x28, #2, MUL VL]
    ld1b { z12.b }, p5/Z, [x28, #3, MUL VL]
    fcvt z15.s, p5/m, z15.h
    fcvt z4.s, p5/m, z4.h
    lsl z14.b, z13.b, #0x4
    lsl z3.b, z7.b, #0x4
    ld1rqb { z1.b }, p5/Z, [x26, #32]
    add x28, x28, #0x80
    and z13.b, z13.b, #0xf0
    and z7.b, z7.b, #0xf0
    fscale z15.s, p5/m, z15.s, z10.s
    mov z4.q, z4.q[0]
    KAI_ASM_INST(0x450e9968)  // smmla z8.s, z11.b, z14.b
    KAI_ASM_INST(0x45039960)  // smmla z0.s, z11.b, z3.b
    KAI_ASM_INST(0x450e98a9)  // smmla z9.s, z5.b, z14.b
    lsl z11.b, z2.b, #0x4
    KAI_ASM_INST(0x450398a6)  // smmla z6.s, z5.b, z3.b
    lsl z5.b, z12.b, #0x4
    and z2.b, z2.b, #0xf0
    and z12.b, z12.b, #0xf0
    KAI_ASM_INST(0x450b9828)  // smmla z8.s, z1.b, z11.b
    KAI_ASM_INST(0x45059820)  // smmla z0.s, z1.b, z5.b
    ld1rqb { z1.b }, p5/Z, [x26, #48]
    KAI_ASM_INST(0x450b9829)  // smmla z9.s, z1.b, z11.b
    KAI_ASM_INST(0x45059826)  // smmla z6.s, z1.b, z5.b
    ld1rqb { z1.b }, p5/Z, [x26, #64]
    KAI_ASM_INST(0x450d9828)  // smmla z8.s, z1.b, z13.b
    KAI_ASM_INST(0x45079820)  // smmla z0.s, z1.b, z7.b
    ld1rqb { z1.b }, p5/Z, [x26, #80]
    KAI_ASM_INST(0x450d9829)  // smmla z9.s, z1.b, z13.b
    KAI_ASM_INST(0x45079826)  // smmla z6.s, z1.b, z7.b
    ld1rqb { z1.b }, p5/Z, [x26, #96]
    KAI_ASM_INST(0x45029828)  // smmla z8.s, z1.b, z2.b
    KAI_ASM_INST(0x450c9820)  // smmla z0.s, z1.b, z12.b
    ld1rqb { z1.b }, p5/Z, [x26, #112]
    add x26, x26, #0x80
    KAI_ASM_INST(0x45029829)  // smmla z9.s, z1.b, z2.b
    KAI_ASM_INST(0x450c9826)  // smmla z6.s, z1.b, z12.b
    uzp1 z1.d, z8.d, z0.d
    uzp2 z0.d, z8.d, z0.d
    KAI_ASM_INST(0x64a421e8)  // fmul z8.s, z15.s, z4.s[0]
    scvtf z1.s, p5/m, z1.s
    scvtf z0.s, p5/m, z0.s
    fmla z31.s, p5/M, z1.s, z8.s
    KAI_ASM_INST(0x64ac21e1)  // fmul z1.s, z15.s, z4.s[1]
    KAI_ASM_INST(0x64b421e8)  // fmul z8.s, z15.s, z4.s[2]
    KAI_ASM_INST(0x64bc21e4)  // fmul z4.s, z15.s, z4.s[3]
    fmla z30.s, p5/M, z0.s, z1.s
    uzp1 z1.d, z9.d, z6.d
    uzp2 z6.d, z9.d, z6.d
    scvtf z1.s, p5/m, z1.s
    scvtf z6.s, p5/m, z6.s
    fmla z29.s, p5/M, z1.s, z8.s
    fmla z28.s, p5/M, z6.s, z4.s
    ld1h { z0.s }, p4/Z, [x22]
    add x22, x22, #0x8
    mov z4.s, #0x0
    mov z8.s, #0x0
    ld1rqb { z9.b }, p5/Z, [x22]
    ld1rqb { z6.b }, p5/Z, [x22, #16]
    mov z1.s, #0x0
    fcvt z0.s, p5/m, z0.h
    KAI_ASM_INST(0x450e9924)  // smmla z4.s, z9.b, z14.b
    KAI_ASM_INST(0x45039928)  // smmla z8.s, z9.b, z3.b
    mov z9.s, #0x0
    KAI_ASM_INST(0x450e98c1)  // smmla z1.s, z6.b, z14.b
    KAI_ASM_INST(0x450398c9)  // smmla z9.s, z6.b, z3.b
    ld1rqb { z6.b }, p5/Z, [x22, #32]
    mov z0.q, z0.q[0]
    KAI_ASM_INST(0x450b98c4)  // smmla z4.s, z6.b, z11.b
    KAI_ASM_INST(0x450598c8)  // smmla z8.s, z6.b, z5.b
    ld1rqb { z6.b }, p5/Z, [x22, #48]
    KAI_ASM_INST(0x450b98c1)  // smmla z1.s, z6.b, z11.b
    KAI_ASM_INST(0x450598c9)  // smmla z9.s, z6.b, z5.b
    ld1rqb { z6.b }, p5/Z, [x22, #64]
    KAI_ASM_INST(0x450d98c4)  // smmla z4.s, z6.b, z13.b
    KAI_ASM_INST(0x450798c8)  // smmla z8.s, z6.b, z7.b
    ld1rqb { z6.b }, p5/Z, [x22, #80]
    KAI_ASM_INST(0x450d98c1)  // smmla z1.s, z6.b, z13.b
    KAI_ASM_INST(0x450798c9)  // smmla z9.s, z6.b, z7.b
    ld1rqb { z6.b }, p5/Z, [x22, #96]
    KAI_ASM_INST(0x450298c4)  // smmla z4.s, z6.b, z2.b
    KAI_ASM_INST(0x450c98c8)  // smmla z8.s, z6.b, z12.b
    ld1rqb { z6.b }, p5/Z, [x22, #112]
    add x22, x22, #0x80
    KAI_ASM_INST(0x450298c1)  // smmla z1.s, z6.b, z2.b
    KAI_ASM_INST(0x450c98c9)  // smmla z9.s, z6.b, z12.b
    uzp1 z6.d, z4.d, z8.d
    uzp2 z8.d, z4.d, z8.d
    KAI_ASM_INST(0x64a021e4)  // fmul z4.s, z15.s, z0.s[0]
    scvtf z6.s, p5/m, z6.s
    scvtf z8.s, p5/m, z8.s
    fmla z27.s, p5/M, z6.s, z4.s
    KAI_ASM_INST(0x64a821e6)  // fmul z6.s, z15.s, z0.s[1]
    KAI_ASM_INST(0x64b021e4)  // fmul z4.s, z15.s, z0.s[2]
    KAI_ASM_INST(0x64b821e0)  // fmul z0.s, z15.s, z0.s[3]
    fmla z26.s, p5/M, z8.s, z6.s
    uzp1 z8.d, z1.d, z9.d
    uzp2 z9.d, z1.d, z9.d
    scvtf z8.s, p5/m, z8.s
    scvtf z9.s, p5/m, z9.s
    fmla z25.s, p5/M, z8.s, z4.s
    fmla z24.s, p5/M, z9.s, z0.s
    ld1h { z0.s }, p4/Z, [x21]
    add x21, x21, #0x8
    mov z1.s, #0x0
    mov z8.s, #0x0
    ld1rqb { z9.b }, p5/Z, [x21]
    ld1rqb { z6.b }, p5/Z, [x21, #16]
    mov z4.s, #0x0
    fcvt z0.s, p5/m, z0.h
    KAI_ASM_INST(0x450e9921)  // smmla z1.s, z9.b, z14.b
    KAI_ASM_INST(0x45039928)  // smmla z8.s, z9.b, z3.b
    mov z9.s, #0x0
    KAI_ASM_INST(0x450e98c4)  // smmla z4.s, z6.b, z14.b
    KAI_ASM_INST(0x450398c9)  // smmla z9.s, z6.b, z3.b
    ld1rqb { z6.b }, p5/Z, [x21, #32]
    mov z0.q, z0.q[0]
    KAI_ASM_INST(0x450b98c1)  // smmla z1.s, z6.b, z11.b
    KAI_ASM_INST(0x450598c8)  // smmla z8.s, z6.b, z5.b
    ld1rqb { z6.b }, p5/Z, [x21, #48]
    KAI_ASM_INST(0x450b98c4)  // smmla z4.s, z6.b, z11.b
    KAI_ASM_INST(0x450598c9)  // smmla z9.s, z6.b, z5.b
    ld1rqb { z6.b }, p5/Z, [x21, #64]
    KAI_ASM_INST(0x450d98c1)  // smmla z1.s, z6.b, z13.b
    KAI_ASM_INST(0x450798c8)  // smmla z8.s, z6.b, z7.b
    ld1rqb { z6.b }, p5/Z, [x21, #80]
    KAI_ASM_INST(0x450d98c4)  // smmla z4.s, z6.b, z13.b
    KAI_ASM_INST(0x450798c9)  // smmla z9.s, z6.b, z7.b
    ld1rqb { z6.b }, p5/Z, [x21, #96]
    KAI_ASM_INST(0x450298c1)  // smmla z1.s, z6.b, z2.b
    KAI_ASM_INST(0x450c98c8)  // smmla z8.s, z6.b, z12.b
    ld1rqb { z6.b }, p5/Z, [x21, #112]
    add x21, x21, #0x80
    KAI_ASM_INST(0x450298c4)  // smmla z4.s, z6.b, z2.b
    KAI_ASM_INST(0x450c98c9)  // smmla z9.s, z6.b, z12.b
    uzp1 z6.d, z1.d, z8.d
    uzp2 z8.d, z1.d, z8.d
    KAI_ASM_INST(0x64a021e1)  // fmul z1.s, z15.s, z0.s[0]
    scvtf z6.s, p5/m, z6.s
    scvtf z8.s, p5/m, z8.s
    fmla z23.s, p5/M, z6.s, z1.s
    KAI_ASM_INST(0x64a821e1)  // fmul z1.s, z15.s, z0.s[1]
    KAI_ASM_INST(0x64b021e6)  // fmul z6.s, z15.s, z0.s[2]
    KAI_ASM_INST(0x64b821e0)  // fmul z0.s, z15.s, z0.s[3]
    fmla z22.s, p5/M, z8.s, z1.s
    uzp1 z8.d, z4.d, z9.d
    uzp2 z1.d, z4.d, z9.d
    scvtf z8.s, p5/m, z8.s
    scvtf z1.s, p5/m, z1.s
    fmla z21.s, p5/M, z8.s, z6.s
    fmla z20.s, p5/M, z1.s, z0.s
    ld1h { z6.s }, p4/Z, [x20]
    add x20, x20, #0x8
    mov z4.s, #0x0
    mov z0.s, #0x0
    ld1rqb { z9.b }, p5/Z, [x20]
    ld1rqb { z8.b }, p5/Z, [x20, #16]
    mov z1.s, #0x0
    fcvt z6.s, p5/m, z6.h
    KAI_ASM_INST(0x450e9924)  // smmla z4.s, z9.b, z14.b
    KAI_ASM_INST(0x45039920)  // smmla z0.s, z9.b, z3.b
    mov z9.s, #0x0
    KAI_ASM_INST(0x450e9901)  // smmla z1.s, z8.b, z14.b
    ld1rqb { z14.b }, p5/Z, [x20, #32]
    KAI_ASM_INST(0x45039909)  // smmla z9.s, z8.b, z3.b
    ld1rqb { z3.b }, p5/Z, [x20, #48]
    ld1rqb { z8.b }, p5/Z, [x20, #64]
    mov z6.q, z6.q[0]
    KAI_ASM_INST(0x450b99c4)  // smmla z4.s, z14.b, z11.b
    KAI_ASM_INST(0x450599c0)  // smmla z0.s, z14.b, z5.b
    ld1rqb { z14.b }, p5/Z, [x20, #80]
    KAI_ASM_INST(0x450b9861)  // smmla z1.s, z3.b, z11.b
    ld1rqb { z11.b }, p5/Z, [x20, #96]
    KAI_ASM_INST(0x45059869)  // smmla z9.s, z3.b, z5.b
    ld1rqb { z3.b }, p5/Z, [x20, #112]
    KAI_ASM_INST(0x64a621e5)  // fmul z5.s, z15.s, z6.s[0]
    add x20, x20, #0x80
    KAI_ASM_INST(0x450d9904)  // smmla z4.s, z8.b, z13.b
    KAI_ASM_INST(0x45079900)  // smmla z0.s, z8.b, z7.b
    KAI_ASM_INST(0x64ae21e8)  // fmul z8.s, z15.s, z6.s[1]
    KAI_ASM_INST(0x450d99c1)  // smmla z1.s, z14.b, z13.b
    KAI_ASM_INST(0x64b621ed)  // fmul z13.s, z15.s, z6.s[2]
    KAI_ASM_INST(0x64be21e6)  // fmul z6.s, z15.s, z6.s[3]
    KAI_ASM_INST(0x450799c9)  // smmla z9.s, z14.b, z7.b
    KAI_ASM_INST(0x45029964)  // smmla z4.s, z11.b, z2.b
    KAI_ASM_INST(0x450c9960)  // smmla z0.s, z11.b, z12.b
    KAI_ASM_INST(0x45029861)  // smmla z1.s, z3.b, z2.b
    KAI_ASM_INST(0x450c9869)  // smmla z9.s, z3.b, z12.b
    uzp1 z11.d, z4.d, z0.d
    uzp2 z15.d, z4.d, z0.d
    scvtf z11.s, p5/m, z11.s
    uzp1 z14.d, z1.d, z9.d
    uzp2 z0.d, z1.d, z9.d
    scvtf z15.s, p5/m, z15.s
    fmla z19.s, p5/M, z11.s, z5.s
    scvtf z14.s, p5/m, z14.s
    scvtf z0.s, p5/m, z0.s
    fmla z18.s, p5/M, z15.s, z8.s
    fmla z17.s, p5/M, z14.s, z13.s
    fmla z16.s, p5/M, z0.s, z6.s
    subs x23, x23, #0x1
    bgt label_3
    KAI_ASM_INST(0x8540d521)  // ld1rw { z1.s }, p5/Z, [x9]
    mov x24, #0x0
    mov x22, #0x0
    add x20, x9, #0x4
    mov x21, #0x0
    KAI_ASM_INST(0x8540d680)  // ld1rw { z0.s }, p5/Z, [x20]
    mov x23, x12
    mov x20, #0x0
    fmax z31.s, p5/M, z31.s, z1.s
    whilelt p3.s, x24, x27
    fmax z30.s, p5/M, z30.s, z1.s
    whilelt p2.s, x22, x27
    mov x22, #0x0
    fmax z29.s, p5/M, z29.s, z1.s
    whilelt p1.s, x21, x27
    mov x21, #0x0
    fmin z31.s, p5/M, z31.s, z0.s
    fmax z28.s, p5/M, z28.s, z1.s
    whilelt p0.s, x20, x27
    mov x20, #0x0
    fmin z30.s, p5/M, z30.s, z0.s
    fmax z27.s, p5/M, z27.s, z1.s
    st1w { z31.s }, p3, [x23]
    add x23, x23, x10
    whilelt p3.s, x22, x27
    mov x22, #0x0
    fmin z29.s, p5/M, z29.s, z0.s
    st1w { z30.s }, p2, [x23]
    add x23, x23, x10
    fmax z26.s, p5/M, z26.s, z1.s
    whilelt p2.s, x21, x27
    mov x21, #0x0
    st1w { z29.s }, p1, [x23]
    add x23, x23, x10
    fmin z28.s, p5/M, z28.s, z0.s
    fmax z25.s, p5/M, z25.s, z1.s
    whilelt p1.s, x20, x27
    mov x20, #0x0
    fmin z27.s, p5/M, z27.s, z0.s
    st1w { z28.s }, p0, [x23]
    add x23, x23, x10
    fmax z24.s, p5/M, z24.s, z1.s
    whilelt p0.s, x22, x27
    mov x22, #0x0
    st1w { z27.s }, p3, [x23]
    add x23, x23, x10
    fmin z26.s, p5/M, z26.s, z0.s
    fmax z23.s, p5/M, z23.s, z1.s
    whilelt p3.s, x21, x27
    mov x21, #0x0
    fmin z25.s, p5/M, z25.s, z0.s
    st1w { z26.s }, p2, [x23]
    add x23, x23, x10
    fmax z22.s, p5/M, z22.s, z1.s
    whilelt p2.s, x20, x27
    mov x20, #0x0
    st1w { z25.s }, p1, [x23]
    add x23, x23, x10
    fmin z24.s, p5/M, z24.s, z0.s
    fmax z21.s, p5/M, z21.s, z1.s
    whilelt p1.s, x22, x27
    mov x22, #0x0
    fmin z23.s, p5/M, z23.s, z0.s
    st1w { z24.s }, p0, [x23]
    add x23, x23, x10
    fmax z20.s, p5/M, z20.s, z1.s
    whilelt p0.s, x21, x27
    mov x21, #0x0
    st1w { z23.s }, p3, [x23]
    add x23, x23, x10
    fmin z22.s, p5/M, z22.s, z0.s
    fmax z19.s, p5/M, z19.s, z1.s
    whilelt p3.s, x20, x27
    mov x20, #0x0
    fmin z21.s, p5/M, z21.s, z0.s
    st1w { z22.s }, p2, [x23]
    add x23, x23, x10
    fmax z18.s, p5/M, z18.s, z1.s
    whilelt p2.s, x22, x27
    fmin z20.s, p5/M, z20.s, z0.s
    st1w { z21.s }, p1, [x23]
    add x23, x23, x10
    fmax z17.s, p5/M, z17.s, z1.s
    whilelt p1.s, x21, x27
    fmin z19.s, p5/M, z19.s, z0.s
    st1w { z20.s }, p0, [x23]
    add x23, x23, x10
    fmax z16.s, p5/M, z16.s, z1.s
    whilelt p0.s, x20, x27
    fmin z18.s, p5/M, z18.s, z0.s
    st1w { z19.s }, p3, [x23]
    add x23, x23, x10
    fmin z17.s, p5/M, z17.s, z0.s
    fmin z16.s, p5/M, z16.s, z0.s
    st1w { z18.s }, p2, [x23]
    add x23, x23, x10
    st1w { z17.s }, p1, [x23]
    add x23, x23, x10
    st1w { z16.s }, p0, [x23]
    subs x27, x27, #0x8
    add x12, x12, #0x20
    bgt label_2
    mov x20, #0x4
    sub x11, x11, #0x10
    cmp x11, #0x10
    mov x12, x25
    madd x15, x20, x17, x15
    bge label_1
KAI_ASM_LABEL(label_9)  // Row loop skip
    cbz x11, label_15
KAI_ASM_LABEL(label_10)  // Row tail: Row loop
    mov x25, x14
    mov x24, x13
    add x23, x12, x10, LSL #2
KAI_ASM_LABEL(label_11)  // Row tail: Column loop
    mov z31.b, #0x0
    mov z30.b, #0x0
    mov x26, x15
    mov x20, x16
    mov z29.b, #0x0
    mov z28.b, #0x0
KAI_ASM_LABEL(label_12)  // Row tail: Block loop
    ld1h { z9.s }, p5/Z, [x25]
    add x25, x25, #0x10
    ld1h { z7.s }, p4/Z, [x26]
    add x26, x26, #0x8
    ld1b { z8.b }, p5/Z, [x25]
    ld1b { z6.b }, p5/Z, [x25, #1, MUL VL]
    mov z14.s, #0x0
    mov z4.s, #0x0
    ld1rqb { z5.b }, p5/Z, [x26]
    ld1rqb { z15.b }, p5/Z, [x26, #16]
    mov z1.s, #0x0
    mov z0.s, #0x0
    ld1b { z27.b }, p5/Z, [x25, #2, MUL VL]
    ld1b { z26.b }, p5/Z, [x25, #3, MUL VL]
    fcvt z9.s, p5/m, z9.h
    fcvt z7.s, p5/m, z7.h
    lsl z23.b, z8.b, #0x4
    lsl z22.b, z6.b, #0x4
    ld1rqb { z21.b }, p5/Z, [x26, #32]
    ld1rqb { z20.b }, p5/Z, [x26, #48]
    and z8.b, z8.b, #0xf0
    and z6.b, z6.b, #0xf0
    ld1rqb { z25.b }, p5/Z, [x26, #64]
    ld1rqb { z24.b }, p5/Z, [x26, #80]
    lsl z19.b, z27.b, #0x4
    lsl z18.b, z26.b, #0x4
    ld1rqb { z17.b }, p5/Z, [x26, #96]
    ld1rqb { z16.b }, p5/Z, [x26, #112]
    KAI_ASM_INST(0x451798ae)  // smmla z14.s, z5.b, z23.b
    KAI_ASM_INST(0x451698a4)  // smmla z4.s, z5.b, z22.b
    and z27.b, z27.b, #0xf0
    add x25, x25, #0x80
    KAI_ASM_INST(0x451799e1)  // smmla z1.s, z15.b, z23.b
    KAI_ASM_INST(0x451699e0)  // smmla z0.s, z15.b, z22.b
    and z26.b, z26.b, #0xf0
    add x26, x26, #0x80
    fscale z9.s, p5/m, z9.s, z10.s
    mov z7.q, z7.q[0]
    KAI_ASM_INST(0x45139aae)  // smmla z14.s, z21.b, z19.b
    KAI_ASM_INST(0x45129aa4)  // smmla z4.s, z21.b, z18.b
    KAI_ASM_INST(0x45139a81)  // smmla z1.s, z20.b, z19.b
    KAI_ASM_INST(0x45129a80)  // smmla z0.s, z20.b, z18.b
    KAI_ASM_INST(0x64a72137)  // fmul z23.s, z9.s, z7.s[0]
    KAI_ASM_INST(0x64af2136)  // fmul z22.s, z9.s, z7.s[1]
    KAI_ASM_INST(0x64b72135)  // fmul z21.s, z9.s, z7.s[2]
    KAI_ASM_INST(0x64bf2134)  // fmul z20.s, z9.s, z7.s[3]
    KAI_ASM_INST(0x45089b2e)  // smmla z14.s, z25.b, z8.b
    KAI_ASM_INST(0x45069b24)  // smmla z4.s, z25.b, z6.b
    KAI_ASM_INST(0x45089b01)  // smmla z1.s, z24.b, z8.b
    KAI_ASM_INST(0x45069b00)  // smmla z0.s, z24.b, z6.b
    KAI_ASM_INST(0x451b9a2e)  // smmla z14.s, z17.b, z27.b
    KAI_ASM_INST(0x451a9a24)  // smmla z4.s, z17.b, z26.b
    KAI_ASM_INST(0x451b9a01)  // smmla z1.s, z16.b, z27.b
    KAI_ASM_INST(0x451a9a00)  // smmla z0.s, z16.b, z26.b
    uzp1 z19.d, z14.d, z4.d
    uzp2 z18.d, z14.d, z4.d
    scvtf z19.s, p5/m, z19.s
    uzp1 z17.d, z1.d, z0.d
    uzp2 z16.d, z1.d, z0.d
    scvtf z18.s, p5/m, z18.s
    fmla z31.s, p5/M, z19.s, z23.s
    scvtf z17.s, p5/m, z17.s
    scvtf z16.s, p5/m, z16.s
    fmla z30.s, p5/M, z18.s, z22.s
    fmla z29.s, p5/M, z17.s, z21.s
    fmla z28.s, p5/M, z16.s, z20.s
    subs x20, x20, #0x1
    bgt label_12
    KAI_ASM_INST(0x8540d531)  // ld1rw { z17.s }, p5/Z, [x9]
    mov x22, #0x0
    add x20, x9, #0x4
    KAI_ASM_INST(0x8540d690)  // ld1rw { z16.s }, p5/Z, [x20]
    mov x21, x12
    fmax z31.s, p5/M, z31.s, z17.s
    fmax z30.s, p5/M, z30.s, z17.s
    fmax z29.s, p5/M, z29.s, z17.s
    fmax z28.s, p5/M, z28.s, z17.s
    whilelt p0.s, x22, x24
    cmp x11, #0x1
    fmin z31.s, p5/M, z31.s, z16.s
    fmin z30.s, p5/M, z30.s, z16.s
    fmin z29.s, p5/M, z29.s, z16.s
    fmin z28.s, p5/M, z28.s, z16.s
    st1w { z31.s }, p0, [x21]
    add x21, x21, x10
    ble label_14
    mov x20, #0x0
    whilelt p0.s, x20, x24
    cmp x11, #0x2
    st1w { z30.s }, p0, [x21]
    add x21, x21, x10
    ble label_14
    mov x20, #0x0
    whilelt p0.s, x20, x24
    cmp x11, #0x3
    st1w { z29.s }, p0, [x21]
    add x21, x21, x10
    ble label_14
    mov x20, #0x0
    whilelt p0.s, x20, x24
    st1w { z28.s }, p0, [x21]
KAI_ASM_LABEL(label_14)  // Row tail: Output stage exit
    subs x24, x24, #0x8
    add x12, x12, #0x20
    bgt label_11
    subs x11, x11, #0x4
    add x15, x15, x17
    mov x12, x23
    bgt label_10
KAI_ASM_LABEL(label_15)  // Row tail: Row loop skip
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm)

    KAI_ASM_END
