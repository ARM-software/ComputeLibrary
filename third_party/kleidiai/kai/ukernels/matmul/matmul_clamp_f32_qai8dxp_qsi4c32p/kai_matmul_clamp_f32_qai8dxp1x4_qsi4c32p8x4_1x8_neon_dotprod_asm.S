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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp1x4_qsi4c32p8x4_1x8_neon_dotprod)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p8x4_1x8_neon_dotprod)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p8x4_1x8_neon_dotprod)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p8x4_1x8_neon_dotprod)
    stp x20, x21, [sp, -80]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    mov x15, #0x20
    movi v4.16b, #0xf0
    mov x21, #0x8
    ldr x14, [x0, #0x40]
    ldr x13, [x0, #0x38]
    ldr x20, [x0, #0x28]
    ldr x12, [x0, #0x8]
    ldr x11, [x0, #0x10]
    ldr x10, [x0, #0x30]
    mul x15, x14, x15
    ldr x9, [x0, #0x0]
    ldr x28, [x0, #0x20]
    ldr x27, [x0, #0x18]
    mov x26, x20
    madd x15, x13, x15, x21
KAI_ASM_LABEL(label_1)  // Row loop
    mov x25, x11
    mov x24, x10
    add x23, x9, x28
KAI_ASM_LABEL(label_2)  // Column loop
    mov x22, x12
    movi v3.16b, #0x0
    movi v2.16b, #0x0
    mov x21, x13
KAI_ASM_LABEL(label_3)  // Block loop
    movi v1.4s, #0x0
    movi v0.4s, #0x0
    mov x20, x14
KAI_ASM_LABEL(label_4)  // Sub block loop
    ldr q31, [x25, #0x0]
    ldr q30, [x25, #0x10]
    subs x20, x20, #0x1
    ldr q29, [x22, #0x0]
    ldr q28, [x25, #0x20]
    ldr q27, [x25, #0x30]
    ldr q26, [x25, #0x40]
    ldr q25, [x25, #0x50]
    ldr q24, [x25, #0x60]
    shl v17.16b, v31.16b, #0x4
    shl v16.16b, v30.16b, #0x4
    ldr q23, [x25, #0x70]
    ldr q22, [x22, #0x10]
    shl v21.16b, v28.16b, #0x4
    and v31.16b, v31.16b, v4.16b
    shl v20.16b, v27.16b, #0x4
    shl v19.16b, v26.16b, #0x4
    add x25, x25, #0x80
    add x22, x22, #0x20
    KAI_ASM_INST(0x4f9de221)  // sdot v1.4s, v17.16b, v29.4b[0]
    KAI_ASM_INST(0x4f9de200)  // sdot v0.4s, v16.16b, v29.4b[0]
    shl v18.16b, v25.16b, #0x4
    shl v17.16b, v24.16b, #0x4
    shl v16.16b, v23.16b, #0x4
    and v30.16b, v30.16b, v4.16b
    and v28.16b, v28.16b, v4.16b
    and v27.16b, v27.16b, v4.16b
    and v26.16b, v26.16b, v4.16b
    KAI_ASM_INST(0x4fbde2a1)  // sdot v1.4s, v21.16b, v29.4b[1]
    KAI_ASM_INST(0x4fbde280)  // sdot v0.4s, v20.16b, v29.4b[1]
    and v25.16b, v25.16b, v4.16b
    and v24.16b, v24.16b, v4.16b
    and v23.16b, v23.16b, v4.16b
    KAI_ASM_INST(0x4f9dea61)  // sdot v1.4s, v19.16b, v29.4b[2]
    KAI_ASM_INST(0x4f9dea40)  // sdot v0.4s, v18.16b, v29.4b[2]
    KAI_ASM_INST(0x4fbdea21)  // sdot v1.4s, v17.16b, v29.4b[3]
    KAI_ASM_INST(0x4fbdea00)  // sdot v0.4s, v16.16b, v29.4b[3]
    KAI_ASM_INST(0x4f96e3e1)  // sdot v1.4s, v31.16b, v22.4b[0]
    KAI_ASM_INST(0x4f96e3c0)  // sdot v0.4s, v30.16b, v22.4b[0]
    KAI_ASM_INST(0x4fb6e381)  // sdot v1.4s, v28.16b, v22.4b[1]
    KAI_ASM_INST(0x4fb6e360)  // sdot v0.4s, v27.16b, v22.4b[1]
    KAI_ASM_INST(0x4f96eb41)  // sdot v1.4s, v26.16b, v22.4b[2]
    KAI_ASM_INST(0x4f96eb20)  // sdot v0.4s, v25.16b, v22.4b[2]
    KAI_ASM_INST(0x4fb6eb01)  // sdot v1.4s, v24.16b, v22.4b[3]
    KAI_ASM_INST(0x4fb6eae0)  // sdot v0.4s, v23.16b, v22.4b[3]
    bgt label_4
    ldr q16, [x25, #0x0]
    scvtf v1.4s, v1.4s, #0x4
    scvtf v0.4s, v0.4s, #0x4
    sub x21, x21, #0x1
    add x25, x25, #0x10
    shll v17.4s, v16.4h, #0x10
    shll2 v16.4s, v16.8h, #0x10
    fmla v3.4s, v1.4s, v17.4s
    fmla v2.4s, v0.4s, v16.4s
    cbnz x21, label_3
    ld1r { v23.4s }, [x22]
    ldr q22, [x25, #0x0]
    add x22, x22, #0x4
    add x20, x27, #0x4
    ldr q21, [x25, #0x10]
    ld1r { v20.4s }, [x22]
    cmp x24, #0x8
    ldr q19, [x25, #0x20]
    ldr q18, [x25, #0x30]
    add x25, x25, #0x40
    ld1r { v17.4s }, [x27]
    ld1r { v16.4s }, [x20]
    scvtf v23.4s, v23.4s
    fmla v3.4s, v22.4s, v23.s[0]
    fmla v2.4s, v21.4s, v23.s[0]
    fmul v3.4s, v3.4s, v20.4s
    fadd v3.4s, v3.4s, v19.4s
    fmul v2.4s, v2.4s, v20.4s
    fadd v2.4s, v2.4s, v18.4s
    fmax v3.4s, v3.4s, v17.4s
    fmax v2.4s, v2.4s, v17.4s
    fmin v3.4s, v3.4s, v16.4s
    fmin v2.4s, v2.4s, v16.4s
    blt label_5
    str q3, [x9, #0x0]
    str q2, [x9, #0x10]
    b label_10
KAI_ASM_LABEL(label_5)  // Partial output
    mov x20, x9
    tbz x24, #2, label_7
    st1 { v3.4s }, [x20], #0x10
    tbz x24, #1, label_6
    st1 { v2.d }[0], [x20], #0x8
    tbz x24, #0, label_9
    st1 { v2.s }[2], [x20]
    b label_9
KAI_ASM_LABEL(label_6)  // Output block 0: partial_1_4
    tbz x24, #0, label_9
    st1 { v2.s }[0], [x20]
    b label_9
KAI_ASM_LABEL(label_7)  // Output block 0: partial_2_0
    tbz x24, #1, label_8
    st1 { v3.d }[0], [x20], #0x8
    tbz x24, #0, label_9
    st1 { v3.s }[2], [x20]
    b label_9
KAI_ASM_LABEL(label_8)  // Output block 0: partial_1_0
    st1 { v3.s }[0], [x20]
KAI_ASM_LABEL(label_9)  // Output block 0: Done
KAI_ASM_LABEL(label_10)  // Stores done
    subs x24, x24, #0x8
    add x9, x9, #0x20
    bgt label_2
    subs x26, x26, #0x1
    add x12, x12, x15
    mov x9, x23
    bgt label_1
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp x20, x21, [sp], 80
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p8x4_1x8_neon_dotprod)

    KAI_ASM_END
