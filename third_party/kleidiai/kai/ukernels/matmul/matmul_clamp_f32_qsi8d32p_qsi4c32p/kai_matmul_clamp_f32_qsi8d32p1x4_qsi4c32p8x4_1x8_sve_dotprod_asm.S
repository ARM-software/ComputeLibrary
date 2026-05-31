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

    KAI_ASM_CODE(matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod)
    stp x20, x21, [sp, -80]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    mov x13, #0x22
    mov z28.s, #-0x4
    ptrue p1.b
    ldr x12, [x0, #0x38]
    ldr x20, [x0, #0x28]
    ldr x11, [x0, #0x8]
    ldr x10, [x0, #0x10]
    ldr x9, [x0, #0x30]
    ldr x28, [x0, #0x0]
    ldr x27, [x0, #0x20]
    mul x13, x12, x13
    ldr x26, [x0, #0x18]
    mov x25, x20
KAI_ASM_LABEL(label_1)  // Row loop
    mov x24, x10
    mov x23, x9
    add x22, x28, x27
KAI_ASM_LABEL(label_2)  // Column loop
    mov x21, x11
    mov z27.b, #0x0
    mov x20, x12
KAI_ASM_LABEL(label_3)  // Block loop
    ld1h { z26.s }, p1/Z, [x24]
    add x24, x24, #0x10
    KAI_ASM_INST(0x84c0a6b9)  // ld1rh { z25.h }, p1/Z, [x21]
    add x21, x21, #0x2
    ld1b { z24.b }, p1/Z, [x24]
    ld1rqb { z1.b }, p1/Z, [x21]
    mov z23.s, #0x0
    sub x20, x20, #0x1
    ld1b { z22.b }, p1/Z, [x24, #1, MUL VL]
    ld1b { z21.b }, p1/Z, [x24, #2, MUL VL]
    ld1b { z20.b }, p1/Z, [x24, #3, MUL VL]
    ld1rqb { z0.b }, p1/Z, [x21, #16]
    fcvt z26.s, p1/m, z26.h
    fcvt z25.s, p1/m, z25.h
    lsl z19.b, z24.b, #0x4
    and z24.b, z24.b, #0xf0
    add x24, x24, #0x80
    add x21, x21, #0x20
    lsl z18.b, z22.b, #0x4
    lsl z17.b, z21.b, #0x4
    lsl z16.b, z20.b, #0x4
    and z22.b, z22.b, #0xf0
    sdot z23.s, z19.b, z1.b[0]
    and z21.b, z21.b, #0xf0
    and z20.b, z20.b, #0xf0
    fscale z26.s, p1/m, z26.s, z28.s
    sdot z23.s, z18.b, z1.b[1]
    fmul z26.s, z26.s, z25.s
    sdot z23.s, z17.b, z1.b[2]
    sdot z23.s, z16.b, z1.b[3]
    sdot z23.s, z24.b, z0.b[0]
    sdot z23.s, z22.b, z0.b[1]
    sdot z23.s, z21.b, z0.b[2]
    sdot z23.s, z20.b, z0.b[3]
    scvtf z23.s, p1/m, z23.s
    fmla z27.s, p1/M, z23.s, z26.s
    cbnz x20, label_3
    KAI_ASM_INST(0x8540c751)  // ld1rw { z17.s }, p1/Z, [x26]
    mov x21, #0x0
    add x20, x26, #0x4
    KAI_ASM_INST(0x8540c690)  // ld1rw { z16.s }, p1/Z, [x20]
    fmax z27.s, p1/M, z27.s, z17.s
    whilelt p0.s, x21, x23
    fmin z27.s, p1/M, z27.s, z16.s
    st1w { z27.s }, p0, [x28]
    subs x23, x23, #0x8
    add x28, x28, #0x20
    bgt label_2
    subs x25, x25, #0x1
    add x11, x11, x13
    mov x28, x22
    bgt label_1
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp x20, x21, [sp], 80
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod)

    KAI_ASM_END
