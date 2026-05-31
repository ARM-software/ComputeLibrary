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

    KAI_ASM_CODE(rhs_pack_kxn_x16p2vlx2b_x16_x16_sme)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme)

KAI_ASM_FUNCTION_TYPE(kai_kernel_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme)
KAI_ASM_FUNCTION_LABEL(kai_kernel_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme)
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
    ldr x16, [x0, #0x8]
    ptrue p1.b
    ldr x15, [x0, #0x30]
    ldr x23, [x0, #0x0]
    ldr x22, [x0, #0x10]
    mov x21, x16
    ldr x14, [x0, #0x18]
    mov x20, x15
    ldr x13, [x0, #0x20]
    ldr x12, [x0, #0x28]
    ldr x11, [x0, #0x38]
KAI_ASM_LABEL(label_1)  // Bias: Full loop
    whilelt p0.h, XZR, x21
    dech x21
    cmp x21, #0x0
    ld1h { z16.h }, p0/Z, [x23]
    incb x23
    st1h { z16.h }, p1, [x20]
    add x20, x20, x13
    bgt label_1
    mov x10, x22
    incb x15
    cmp x10, #0x8
    blt label_5
KAI_ASM_LABEL(label_2)  // Main row loop: Head
    mov x9, x12
    mov x28, x15
    add x27, x9, x14
    sub x10, x10, #0x8
    add x26, x27, x14
    mov x25, x16
    add x24, x26, x14
    add x23, x24, x14
    add x22, x23, x14
    add x21, x22, x14
    add x20, x21, x14
    add x12, x20, x14
KAI_ASM_LABEL(label_3)  // Main row loop: Column loop
    whilelt p0.h, XZR, x25
    decw x25, ALL, MUL #2
    ld1h { z20.h }, p0/Z, [x9]
    cmp x25, #0x0
    addvl x9, x9, #1
    ld1h { z17.h }, p0/Z, [x27]
    addvl x27, x27, #1
    ld1h { z19.h }, p0/Z, [x26]
    addvl x26, x26, #1
    ld1h { z16.h }, p0/Z, [x24]
    addvl x24, x24, #1
    ld1h { z18.h }, p0/Z, [x23]
    addvl x23, x23, #1
    zip1 z24.h, z20.h, z17.h
    zip2 z23.h, z20.h, z17.h
    ld1h { z17.h }, p0/Z, [x22]
    addvl x22, x22, #1
    ld1h { z22.h }, p0/Z, [x21]
    addvl x21, x21, #1
    zip1 z21.h, z19.h, z16.h
    zip2 z20.h, z19.h, z16.h
    ld1h { z16.h }, p0/Z, [x20]
    addvl x20, x20, #1
    zip1 z19.h, z18.h, z17.h
    zip2 z18.h, z18.h, z17.h
    st1h { z24.h }, p1, [x28]
    st1h { z23.h }, p1, [x28, #1, MUL VL]
    zip1 z17.h, z22.h, z16.h
    zip2 z16.h, z22.h, z16.h
    st1h { z21.h }, p1, [x28, #2, MUL VL]
    st1h { z20.h }, p1, [x28, #3, MUL VL]
    st1h { z19.h }, p1, [x28, #4, MUL VL]
    st1h { z18.h }, p1, [x28, #5, MUL VL]
    st1h { z17.h }, p1, [x28, #6, MUL VL]
    st1h { z16.h }, p1, [x28, #7, MUL VL]
    add x28, x28, x13
    bgt label_3
    cmp x10, #0x8
    addvl x15, x15, #8
    bge label_2
    cbz x10, label_9
KAI_ASM_LABEL(label_5)  // Main loop skip
KAI_ASM_LABEL(label_6)  // Tail row loop: Head
    mov x9, x12
    cntw x22, ALL, MUL #4
    add x27, x9, x14
    cmp x10, #0x1
    add x12, x27, x14
    mov x28, x15
    csel x12, x12, x27, GT
    csel x27, x27, x11, GT
    csel x21, x22, XZR, GT
    sub x10, x10, #0x2
    mov x20, x16
KAI_ASM_LABEL(label_7)  // Tail row loop: Column loop
    whilelt p0.h, XZR, x20
    decw x20, ALL, MUL #2
    ld1h { z18.h }, p0/Z, [x9]
    cmp x20, #0x0
    add x9, x9, x22
    ld1h { z16.h }, p0/Z, [x27]
    add x27, x27, x21
    zip1 z17.h, z18.h, z16.h
    zip2 z16.h, z18.h, z16.h
    st1h { z17.h }, p1, [x28]
    st1h { z16.h }, p1, [x28, #1, MUL VL]
    add x28, x28, x13
    bgt label_7
    cmp x10, #0x1
    addvl x15, x15, #2
    bge label_6
KAI_ASM_LABEL(label_9)  // Done
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
    KAI_ASM_FUNCTION_END(kai_kernel_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme)

    KAI_ASM_END
