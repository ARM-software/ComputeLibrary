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

    KAI_ASM_CODE(rhs_pack_kxn_x16p32x1b_x16_x16_neon)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_rhs_pack_kxn_x16p32x1b_x16_x16_neon)

KAI_ASM_FUNCTION_TYPE(kai_kernel_rhs_pack_kxn_x16p32x1b_x16_x16_neon)
KAI_ASM_FUNCTION_LABEL(kai_kernel_rhs_pack_kxn_x16p32x1b_x16_x16_neon)
    stp x20, x21, [sp, -80]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    ldr x11, [x0, #0x8]
    mov x24, x11
    ldr x10, [x0, #0x38]
    cmp x24, #0x20
    ldr x23, [x0, #0x0]
    ldr x22, [x0, #0x10]
    ldr x9, [x0, #0x18]
    ldr x28, [x0, #0x20]
    ldr x20, [x0, #0x28]
    ldr x27, [x0, #0x30]
    mov x21, x10
    blt label_2
KAI_ASM_LABEL(label_1)  // Bias: Full loop
    ldr q19, [x23, #0x0]
    ldr q18, [x23, #0x10]
    sub x24, x24, #0x20
    ldr q17, [x23, #0x20]
    ldr q16, [x23, #0x30]
    cmp x24, #0x20
    add x23, x23, x20
    str q19, [x21, #0x0]
    str q18, [x21, #0x10]
    str q17, [x21, #0x20]
    str q16, [x21, #0x30]
    add x21, x21, x28
    bge label_1
    cbz x24, label_3
KAI_ASM_LABEL(label_2)  // Bias: Tail loop
    ldr h20, [x23, #0x0]
    sub x24, x24, #0x1
    add x23, x23, #0x2
    cmp x24, #0x0
    str h20, [x21]
    add x21, x21, #0x2
    bgt label_2
KAI_ASM_LABEL(label_3)  // Bias: Done
    mov x26, x22
    add x10, x10, #0x40
    cmp x26, #0x4
    blt label_14
KAI_ASM_LABEL(label_4)  // Main row loop: Head
    mov x25, x27
    mov x24, x11
    mov x23, x10
    sub x26, x26, #0x4
    add x22, x25, x9
    add x21, x22, x9
    add x20, x21, x9
    cmp x24, #0x20
    add x27, x20, x9
    blt label_6
KAI_ASM_LABEL(label_5)  // Main row loop: Column loop
    ldr q31, [x25], #0x10
    ldr q30, [x22], #0x10
    sub x24, x24, #0x20
    ldr q29, [x21], #0x10
    ldr q28, [x20], #0x10
    cmp x24, #0x20
    ldr q27, [x25], #0x10
    ldr q26, [x22], #0x10
    ldr q25, [x21], #0x10
    ldr q24, [x20], #0x10
    ldr q23, [x25], #0x10
    ldr q22, [x22], #0x10
    ldr q21, [x21], #0x10
    ldr q20, [x20], #0x10
    ldr q19, [x25], #0x10
    ldr q18, [x22], #0x10
    ldr q17, [x21], #0x10
    ldr q16, [x20], #0x10
    str q31, [x23, #0x0]
    str q27, [x23, #0x10]
    str q23, [x23, #0x20]
    str q19, [x23, #0x30]
    str q30, [x23, #0x40]
    str q26, [x23, #0x50]
    str q22, [x23, #0x60]
    str q18, [x23, #0x70]
    str q29, [x23, #0x80]
    str q25, [x23, #0x90]
    str q21, [x23, #0xa0]
    str q17, [x23, #0xb0]
    str q28, [x23, #0xc0]
    str q24, [x23, #0xd0]
    str q20, [x23, #0xe0]
    str q16, [x23, #0xf0]
    add x23, x23, x28
    bge label_5
KAI_ASM_LABEL(label_6)  // Main row loop: Column loop skip
    cbz x24, label_13
    cmp x24, #0x10
    movi v16.8h, #0x0
    str q16, [x23, #0x0]
    str q16, [x23, #0x10]
    str q16, [x23, #0x20]
    str q16, [x23, #0x30]
    str q16, [x23, #0x40]
    str q16, [x23, #0x50]
    str q16, [x23, #0x60]
    str q16, [x23, #0x70]
    str q16, [x23, #0x80]
    str q16, [x23, #0x90]
    str q16, [x23, #0xa0]
    str q16, [x23, #0xb0]
    str q16, [x23, #0xc0]
    str q16, [x23, #0xd0]
    str q16, [x23, #0xe0]
    str q16, [x23, #0xf0]
    blt label_8
KAI_ASM_LABEL(label_7)  // Main row loop: width 16 loop: loop
    ldr q23, [x25], #0x10
    ldr q22, [x22], #0x10
    sub x24, x24, #0x10
    ldr q21, [x21], #0x10
    ldr q20, [x20], #0x10
    cmp x24, #0x10
    ldr q19, [x25], #0x10
    ldr q18, [x22], #0x10
    ldr q17, [x21], #0x10
    ldr q16, [x20], #0x10
    str q23, [x23, #0x0]
    str q19, [x23, #0x10]
    str q22, [x23, #0x40]
    str q18, [x23, #0x50]
    str q21, [x23, #0x80]
    str q17, [x23, #0x90]
    str q20, [x23, #0xc0]
    str q16, [x23, #0xd0]
    add x23, x23, #0x20
    bge label_7
KAI_ASM_LABEL(label_8)  // Main row loop: width 16 loop: skip
    cmp x24, #0x4
    blt label_10
KAI_ASM_LABEL(label_9)  // Main row loop: width 4 loop: loop
    ldr d19, [x25], #0x8
    ldr d18, [x22], #0x8
    sub x24, x24, #0x4
    ldr d17, [x21], #0x8
    ldr d16, [x20], #0x8
    cmp x24, #0x4
    str d19, [x23, #0x0]
    str d18, [x23, #0x40]
    str d17, [x23, #0x80]
    str d16, [x23, #0xc0]
    add x23, x23, #0x8
    bge label_9
KAI_ASM_LABEL(label_10)  // Main row loop: width 4 loop: skip
    cmp x24, #0x1
    blt label_12
KAI_ASM_LABEL(label_11)  // Main row loop: width 1 loop: loop
    ldr h19, [x25], #0x2
    ldr h18, [x22], #0x2
    sub x24, x24, #0x1
    ldr h17, [x21], #0x2
    ldr h16, [x20], #0x2
    cmp x24, #0x1
    str h19, [x23, #0x0]
    str h18, [x23, #0x40]
    str h17, [x23, #0x80]
    str h16, [x23, #0xc0]
    add x23, x23, #0x2
    bge label_11
KAI_ASM_LABEL(label_12)  // Main row loop: width 1 loop: skip
KAI_ASM_LABEL(label_13)  // Main row loop: odd col skip
    cmp x26, #0x4
    add x10, x10, #0x100
    bge label_4
    cbz x26, label_25
KAI_ASM_LABEL(label_14)  // Main loop skip
KAI_ASM_LABEL(label_15)  // Tail row loop: Head
    mov x20, x11
    mov x25, x27
    mov x23, x10
    sub x26, x26, #0x1
    cmp x20, #0x20
    add x27, x25, x9
    blt label_17
KAI_ASM_LABEL(label_16)  // Tail row loop: Column loop
    ldr q19, [x25], #0x10
    sub x20, x20, #0x20
    ldr q18, [x25], #0x10
    ldr q17, [x25], #0x10
    cmp x20, #0x20
    ldr q16, [x25], #0x10
    str q19, [x23, #0x0]
    str q18, [x23, #0x10]
    str q17, [x23, #0x20]
    str q16, [x23, #0x30]
    add x23, x23, x28
    bge label_16
KAI_ASM_LABEL(label_17)  // Tail row loop: Column loop skip
    cbz x20, label_24
    cmp x20, #0x10
    movi v16.8h, #0x0
    str q16, [x23, #0x0]
    str q16, [x23, #0x10]
    str q16, [x23, #0x20]
    str q16, [x23, #0x30]
    blt label_19
KAI_ASM_LABEL(label_18)  // Tail row loop: width 16 loop: loop
    ldr q17, [x25], #0x10
    sub x20, x20, #0x10
    ldr q16, [x25], #0x10
    cmp x20, #0x10
    str q17, [x23, #0x0]
    str q16, [x23, #0x10]
    add x23, x23, #0x20
    bge label_18
KAI_ASM_LABEL(label_19)  // Tail row loop: width 16 loop: skip
    cmp x20, #0x4
    blt label_21
KAI_ASM_LABEL(label_20)  // Tail row loop: width 4 loop: loop
    ldr d16, [x25], #0x8
    sub x20, x20, #0x4
    cmp x20, #0x4
    str d16, [x23, #0x0]
    add x23, x23, #0x8
    bge label_20
KAI_ASM_LABEL(label_21)  // Tail row loop: width 4 loop: skip
    cmp x20, #0x1
    blt label_23
KAI_ASM_LABEL(label_22)  // Tail row loop: width 1 loop: loop
    ldr h16, [x25], #0x2
    sub x20, x20, #0x1
    cmp x20, #0x1
    str h16, [x23, #0x0]
    add x23, x23, #0x2
    bge label_22
KAI_ASM_LABEL(label_23)  // Tail row loop: width 1 loop: skip
KAI_ASM_LABEL(label_24)  // Tail row loop: odd col skip
    cmp x26, #0x1
    add x10, x10, #0x40
    bge label_15
KAI_ASM_LABEL(label_25)  // Done
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp x20, x21, [sp], 80
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_rhs_pack_kxn_x16p32x1b_x16_x16_neon)

    KAI_ASM_END
