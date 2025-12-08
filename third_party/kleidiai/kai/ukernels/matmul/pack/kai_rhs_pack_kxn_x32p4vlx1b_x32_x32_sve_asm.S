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

    KAI_ASM_CODE(rhs_pack_kxn_x32p4vlx1b_x32_x32_sve)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_rhs_pack_kxn_x32p4vlx1b_x32_x32_sve)

KAI_ASM_FUNCTION_TYPE(kai_kernel_rhs_pack_kxn_x32p4vlx1b_x32_x32_sve)
KAI_ASM_FUNCTION_LABEL(kai_kernel_rhs_pack_kxn_x32p4vlx1b_x32_x32_sve)
    stp x20, x21, [sp, -80]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    ptrue p2.b
    ldr x13, [x0, #0x8]
    ldr x12, [x0, #0x30]
    ldr x24, [x0, #0x0]
    ldr x23, [x0, #0x10]
    ldr x11, [x0, #0x18]
    ldr x10, [x0, #0x20]
    ldr x9, [x0, #0x28]
    mov x22, x12
    mov x21, x13
KAI_ASM_LABEL(label_1)  // Bias: Full loop
    mov x20, x21
    decw x21, ALL, MUL #4
    whilelt p1.s, XZR, x20
    decw x20
    whilelt p0.s, XZR, x20
    decw x20
    ld1w { z19.s }, p1/Z, [x24]
    whilelt p1.s, XZR, x20
    decw x20
    ld1w { z18.s }, p0/Z, [x24, #1, MUL VL]
    whilelt p0.s, XZR, x20
    cmp x21, #0x0
    ld1w { z17.s }, p1/Z, [x24, #2, MUL VL]
    ld1w { z16.s }, p0/Z, [x24, #3, MUL VL]
    st1w { z19.s }, p2, [x22]
    incb x24, ALL, MUL #4
    st1w { z18.s }, p2, [x22, #1, MUL VL]
    st1w { z17.s }, p2, [x22, #2, MUL VL]
    st1w { z16.s }, p2, [x22, #3, MUL VL]
    add x22, x22, x10
    bgt label_1
    mov x28, x23
    incb x12, ALL, MUL #4
    cmp x28, #0x4
    blt label_5
KAI_ASM_LABEL(label_2)  // Main row loop: Head
    mov x27, x9
    mov x26, x12
    sub x28, x28, #0x4
    mov x25, x13
    add x24, x27, x11
    add x23, x24, x11
    add x22, x23, x11
    add x9, x22, x11
KAI_ASM_LABEL(label_3)  // Main row loop: Column loop
    mov x21, x25
    mov x20, x26
    decw x25, ALL, MUL #4
    add x26, x26, x10
    whilelt p1.s, XZR, x21
    decw x21
    whilelt p0.s, XZR, x21
    decw x21
    ld1w { z31.s }, p1/Z, [x27]
    ld1w { z30.s }, p1/Z, [x24]
    ld1w { z29.s }, p1/Z, [x23]
    ld1w { z28.s }, p1/Z, [x22]
    whilelt p1.s, XZR, x21
    decw x21
    ld1w { z27.s }, p0/Z, [x27, #1, MUL VL]
    ld1w { z26.s }, p0/Z, [x24, #1, MUL VL]
    ld1w { z25.s }, p0/Z, [x23, #1, MUL VL]
    ld1w { z24.s }, p0/Z, [x22, #1, MUL VL]
    whilelt p0.s, XZR, x21
    cmp x25, #0x0
    ld1w { z23.s }, p1/Z, [x27, #2, MUL VL]
    ld1w { z22.s }, p1/Z, [x24, #2, MUL VL]
    ld1w { z21.s }, p1/Z, [x23, #2, MUL VL]
    ld1w { z20.s }, p1/Z, [x22, #2, MUL VL]
    ld1w { z19.s }, p0/Z, [x27, #3, MUL VL]
    ld1w { z18.s }, p0/Z, [x24, #3, MUL VL]
    addvl x27, x27, #4
    addvl x24, x24, #4
    ld1w { z17.s }, p0/Z, [x23, #3, MUL VL]
    ld1w { z16.s }, p0/Z, [x22, #3, MUL VL]
    st1w { z31.s }, p2, [x20]
    addvl x23, x23, #4
    st1w { z27.s }, p2, [x20, #1, MUL VL]
    addvl x22, x22, #4
    st1w { z23.s }, p2, [x20, #2, MUL VL]
    st1w { z19.s }, p2, [x20, #3, MUL VL]
    st1w { z30.s }, p2, [x20, #4, MUL VL]
    st1w { z26.s }, p2, [x20, #5, MUL VL]
    st1w { z22.s }, p2, [x20, #6, MUL VL]
    st1w { z18.s }, p2, [x20, #7, MUL VL]
    addvl x20, x20, #16
    st1w { z29.s }, p2, [x20, #-8, MUL VL]
    st1w { z25.s }, p2, [x20, #-7, MUL VL]
    st1w { z21.s }, p2, [x20, #-6, MUL VL]
    st1w { z17.s }, p2, [x20, #-5, MUL VL]
    st1w { z28.s }, p2, [x20, #-4, MUL VL]
    st1w { z24.s }, p2, [x20, #-3, MUL VL]
    st1w { z20.s }, p2, [x20, #-2, MUL VL]
    st1w { z16.s }, p2, [x20, #-1, MUL VL]
    bgt label_3
    cmp x28, #0x4
    addvl x12, x12, #16
    bge label_2
    cbz x28, label_9
KAI_ASM_LABEL(label_5)  // Main loop skip
KAI_ASM_LABEL(label_6)  // Tail row loop: Head
    mov x27, x9
    mov x26, x12
    sub x28, x28, #0x1
    mov x21, x13
    add x9, x27, x11
KAI_ASM_LABEL(label_7)  // Tail row loop: Column loop
    mov x20, x21
    decw x21, ALL, MUL #4
    whilelt p1.s, XZR, x20
    decw x20
    whilelt p0.s, XZR, x20
    decw x20
    ld1w { z19.s }, p1/Z, [x27]
    whilelt p1.s, XZR, x20
    decw x20
    ld1w { z18.s }, p0/Z, [x27, #1, MUL VL]
    whilelt p0.s, XZR, x20
    cmp x21, #0x0
    ld1w { z17.s }, p1/Z, [x27, #2, MUL VL]
    ld1w { z16.s }, p0/Z, [x27, #3, MUL VL]
    addvl x27, x27, #4
    st1w { z19.s }, p2, [x26]
    st1w { z18.s }, p2, [x26, #1, MUL VL]
    st1w { z17.s }, p2, [x26, #2, MUL VL]
    st1w { z16.s }, p2, [x26, #3, MUL VL]
    add x26, x26, x10
    bgt label_7
    cmp x28, #0x1
    addvl x12, x12, #4
    bge label_6
KAI_ASM_LABEL(label_9)  // Done
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp x20, x21, [sp], 80
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_rhs_pack_kxn_x32p4vlx1b_x32_x32_sve)

    KAI_ASM_END
