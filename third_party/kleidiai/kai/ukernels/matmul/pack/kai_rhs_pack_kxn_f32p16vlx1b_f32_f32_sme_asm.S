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

    KAI_ASM_CODE(rhs_pack_kxn_f32p16vlx1b_f32_f32_sme)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_rhs_pack_kxn_f32p16vlx1b_f32_f32_sme)

KAI_ASM_FUNCTION_TYPE(kai_kernel_rhs_pack_kxn_f32p16vlx1b_f32_f32_sme)
KAI_ASM_FUNCTION_LABEL(kai_kernel_rhs_pack_kxn_f32p16vlx1b_f32_f32_sme)
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
    ldr x11, [x0, #0x8]
    ptrue p7.b
    ldr x10, [x0, #0x30]
    ldr x25, [x0, #0x0]
    ldr x9, [x0, #0x10]
    mov x24, x11
    ldr x28, [x0, #0x18]
    mov x23, x10
    ldr x27, [x0, #0x20]
    ldr x26, [x0, #0x28]
KAI_ASM_LABEL(label_1)  // Bias: Full loop
    mov x22, x24
    mov x21, x25
    whilelt p0.s, XZR, x22
    decw x22
    whilelt p1.s, XZR, x22
    decw x22
    ld1w { z31.s }, p0/Z, [x21]
    whilelt p0.s, XZR, x22
    decw x22
    ld1w { z30.s }, p1/Z, [x21, #1, MUL VL]
    whilelt p1.s, XZR, x22
    decw x22
    ld1w { z29.s }, p0/Z, [x21, #2, MUL VL]
    whilelt p0.s, XZR, x22
    decw x22
    ld1w { z28.s }, p1/Z, [x21, #3, MUL VL]
    whilelt p1.s, XZR, x22
    decw x22
    ld1w { z27.s }, p0/Z, [x21, #4, MUL VL]
    whilelt p0.s, XZR, x22
    decw x22
    ld1w { z26.s }, p1/Z, [x21, #5, MUL VL]
    whilelt p1.s, XZR, x22
    decw x22
    ld1w { z25.s }, p0/Z, [x21, #6, MUL VL]
    whilelt p0.s, XZR, x22
    decw x22
    ld1w { z24.s }, p1/Z, [x21, #7, MUL VL]
    whilelt p6.s, XZR, x22
    decw x22
    whilelt p5.s, XZR, x22
    decw x22
    whilelt p4.s, XZR, x22
    decw x22
    whilelt p3.s, XZR, x22
    decw x22
    whilelt p2.s, XZR, x22
    decw x22
    whilelt p1.s, XZR, x22
    decw x22
    addvl x21, x21, #16
    mov x20, x23
    decw x24, ALL, MUL #16
    ld1w { z23.s }, p0/Z, [x21, #-8, MUL VL]
    whilelt p0.s, XZR, x22
    ld1w { z22.s }, p6/Z, [x21, #-7, MUL VL]
    cmp x24, #0x0
    incb x25, ALL, MUL #16
    ld1w { z21.s }, p5/Z, [x21, #-6, MUL VL]
    add x23, x23, x27
    ld1w { z20.s }, p4/Z, [x21, #-5, MUL VL]
    ld1w { z19.s }, p3/Z, [x21, #-4, MUL VL]
    ld1w { z18.s }, p2/Z, [x21, #-3, MUL VL]
    ld1w { z17.s }, p1/Z, [x21, #-2, MUL VL]
    ld1w { z16.s }, p0/Z, [x21, #-1, MUL VL]
    st1w { z31.s }, p7, [x20]
    st1w { z30.s }, p7, [x20, #1, MUL VL]
    st1w { z29.s }, p7, [x20, #2, MUL VL]
    st1w { z28.s }, p7, [x20, #3, MUL VL]
    st1w { z27.s }, p7, [x20, #4, MUL VL]
    st1w { z26.s }, p7, [x20, #5, MUL VL]
    st1w { z25.s }, p7, [x20, #6, MUL VL]
    st1w { z24.s }, p7, [x20, #7, MUL VL]
    addvl x20, x20, #16
    st1w { z23.s }, p7, [x20, #-8, MUL VL]
    st1w { z22.s }, p7, [x20, #-7, MUL VL]
    st1w { z21.s }, p7, [x20, #-6, MUL VL]
    st1w { z20.s }, p7, [x20, #-5, MUL VL]
    st1w { z19.s }, p7, [x20, #-4, MUL VL]
    st1w { z18.s }, p7, [x20, #-3, MUL VL]
    st1w { z17.s }, p7, [x20, #-2, MUL VL]
    st1w { z16.s }, p7, [x20, #-1, MUL VL]
    bgt label_1
    incb x10, ALL, MUL #16
    mov x25, x9
    cbz x9, label_5
KAI_ASM_LABEL(label_2)  // Main row loop: Head
    mov x24, x26
    mov x23, x10
    add x26, x24, x28
    sub x25, x25, #0x1
    mov x22, x11
KAI_ASM_LABEL(label_3)  // Main row loop: Column loop
    mov x21, x22
    mov x20, x23
    whilelt p0.s, XZR, x21
    decw x21
    whilelt p1.s, XZR, x21
    decw x21
    ld1w { z31.s }, p0/Z, [x24]
    whilelt p0.s, XZR, x21
    decw x21
    ld1w { z30.s }, p1/Z, [x24, #1, MUL VL]
    whilelt p1.s, XZR, x21
    decw x21
    ld1w { z29.s }, p0/Z, [x24, #2, MUL VL]
    whilelt p0.s, XZR, x21
    decw x21
    ld1w { z28.s }, p1/Z, [x24, #3, MUL VL]
    whilelt p1.s, XZR, x21
    decw x21
    ld1w { z27.s }, p0/Z, [x24, #4, MUL VL]
    whilelt p0.s, XZR, x21
    decw x21
    ld1w { z26.s }, p1/Z, [x24, #5, MUL VL]
    whilelt p1.s, XZR, x21
    decw x21
    ld1w { z25.s }, p0/Z, [x24, #6, MUL VL]
    whilelt p0.s, XZR, x21
    decw x21
    ld1w { z24.s }, p1/Z, [x24, #7, MUL VL]
    whilelt p6.s, XZR, x21
    decw x21
    whilelt p5.s, XZR, x21
    decw x21
    whilelt p4.s, XZR, x21
    decw x21
    whilelt p3.s, XZR, x21
    decw x21
    whilelt p2.s, XZR, x21
    decw x21
    whilelt p1.s, XZR, x21
    decw x21
    addvl x24, x24, #16
    decw x22, ALL, MUL #16
    ld1w { z23.s }, p0/Z, [x24, #-8, MUL VL]
    whilelt p0.s, XZR, x21
    cmp x22, #0x0
    ld1w { z22.s }, p6/Z, [x24, #-7, MUL VL]
    add x23, x23, x27
    ld1w { z21.s }, p5/Z, [x24, #-6, MUL VL]
    ld1w { z20.s }, p4/Z, [x24, #-5, MUL VL]
    ld1w { z19.s }, p3/Z, [x24, #-4, MUL VL]
    ld1w { z18.s }, p2/Z, [x24, #-3, MUL VL]
    ld1w { z17.s }, p1/Z, [x24, #-2, MUL VL]
    ld1w { z16.s }, p0/Z, [x24, #-1, MUL VL]
    st1w { z31.s }, p7, [x20]
    st1w { z30.s }, p7, [x20, #1, MUL VL]
    st1w { z29.s }, p7, [x20, #2, MUL VL]
    st1w { z28.s }, p7, [x20, #3, MUL VL]
    st1w { z27.s }, p7, [x20, #4, MUL VL]
    st1w { z26.s }, p7, [x20, #5, MUL VL]
    st1w { z25.s }, p7, [x20, #6, MUL VL]
    st1w { z24.s }, p7, [x20, #7, MUL VL]
    addvl x20, x20, #16
    st1w { z23.s }, p7, [x20, #-8, MUL VL]
    st1w { z22.s }, p7, [x20, #-7, MUL VL]
    st1w { z21.s }, p7, [x20, #-6, MUL VL]
    st1w { z20.s }, p7, [x20, #-5, MUL VL]
    st1w { z19.s }, p7, [x20, #-4, MUL VL]
    st1w { z18.s }, p7, [x20, #-3, MUL VL]
    st1w { z17.s }, p7, [x20, #-2, MUL VL]
    st1w { z16.s }, p7, [x20, #-1, MUL VL]
    bgt label_3
    cmp x25, #0x1
    addvl x10, x10, #16
    bge label_2
KAI_ASM_LABEL(label_5)  // Done
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
    KAI_ASM_FUNCTION_END(kai_kernel_rhs_pack_kxn_f32p16vlx1b_f32_f32_sme)

    KAI_ASM_END
