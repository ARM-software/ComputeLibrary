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

    KAI_ASM_CODE(rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme)

KAI_ASM_FUNCTION_TYPE(kai_kernel_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme)
KAI_ASM_FUNCTION_LABEL(kai_kernel_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme)
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
    ldr x14, [x0, #0x8]
    ptrue p2.b
    ldr x13, [x0, #0x38]
    ldr x24, [x0, #0x0]
    ldr x12, [x0, #0x10]
    mov x23, x14
    ldr x22, [x0, #0x18]
    mov x21, x13
    ldr x11, [x0, #0x20]
    ldr x10, [x0, #0x28]
    ldr x9, [x0, #0x30]
KAI_ASM_LABEL(label_1)  // Bias: Full loop
    mov x20, x23
    decw x23, ALL, MUL #2
    whilelt p1.s, XZR, x20
    decw x20
    whilelt p0.s, XZR, x20
    cmp x23, #0x0
    ld1w { z17.s }, p1/Z, [x24]
    ld1w { z16.s }, p0/Z, [x24, #1, MUL VL]
    incb x24, ALL, MUL #2
    st1w { z17.s }, p2, [x21]
    st1w { z16.s }, p2, [x21, #1, MUL VL]
    add x21, x21, x10
    bgt label_1
    incb x13, ALL, MUL #2
    mov x28, x22
KAI_ASM_LABEL(label_2)  // Chunk Loop
    mov x27, x12
    cmp x27, #0x4
    blt label_6
KAI_ASM_LABEL(label_3)  // Main row loop: Head
    mov x26, x9
    mov x25, x13
    add x24, x26, x11
    sub x27, x27, #0x4
    add x23, x24, x11
    mov x22, x14
    add x21, x23, x11
    add x9, x21, x11
KAI_ASM_LABEL(label_4)  // Main row loop: Column loop
    mov x20, x22
    decw x22, ALL, MUL #2
    whilelt p1.s, XZR, x20
    decw x20
    whilelt p0.s, XZR, x20
    cmp x22, #0x0
    ld1w { z23.s }, p1/Z, [x26]
    ld1w { z22.s }, p0/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    ld1w { z21.s }, p1/Z, [x24]
    ld1w { z20.s }, p0/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    ld1w { z19.s }, p1/Z, [x23]
    ld1w { z18.s }, p0/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ld1w { z17.s }, p1/Z, [x21]
    ld1w { z16.s }, p0/Z, [x21, #1, MUL VL]
    addvl x21, x21, #2
    st1w { z23.s }, p2, [x25]
    st1w { z22.s }, p2, [x25, #1, MUL VL]
    st1w { z21.s }, p2, [x25, #2, MUL VL]
    st1w { z20.s }, p2, [x25, #3, MUL VL]
    st1w { z19.s }, p2, [x25, #4, MUL VL]
    st1w { z18.s }, p2, [x25, #5, MUL VL]
    st1w { z17.s }, p2, [x25, #6, MUL VL]
    st1w { z16.s }, p2, [x25, #7, MUL VL]
    add x25, x25, x10
    bgt label_4
    cmp x27, #0x4
    addvl x13, x13, #8
    bge label_3
    cbz x27, label_10
KAI_ASM_LABEL(label_6)  // Main loop skip
KAI_ASM_LABEL(label_7)  // Tail row loop: Head
    mov x26, x9
    cntw x22, ALL, MUL #8
    add x9, x26, x11
    mov x25, x13
    sub x27, x27, #0x1
    mov x21, x14
KAI_ASM_LABEL(label_8)  // Tail row loop: Column loop
    mov x20, x21
    decw x21, ALL, MUL #2
    whilelt p1.s, XZR, x20
    decw x20
    whilelt p0.s, XZR, x20
    cmp x21, #0x0
    ld1w { z17.s }, p1/Z, [x26]
    ld1w { z16.s }, p0/Z, [x26, #1, MUL VL]
    add x26, x26, x22
    st1w { z17.s }, p2, [x25]
    st1w { z16.s }, p2, [x25, #1, MUL VL]
    add x25, x25, x10
    bgt label_8
    cmp x27, #0x1
    addvl x13, x13, #2
    bge label_7
KAI_ASM_LABEL(label_10)  // Done
    sub x28, x28, #0x1
    cbnz x28, label_2
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
    KAI_ASM_FUNCTION_END(kai_kernel_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme)

    KAI_ASM_END
