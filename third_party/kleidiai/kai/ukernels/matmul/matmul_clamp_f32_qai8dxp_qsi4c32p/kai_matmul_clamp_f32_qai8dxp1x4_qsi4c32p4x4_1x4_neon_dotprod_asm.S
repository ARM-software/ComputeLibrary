//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod)
    stp x20, x21, [sp, -80]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    mov x15, #0x20
    movi v28.16b, #0xf0
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
    movi v27.16b, #0x0
    mov x21, x13
KAI_ASM_LABEL(label_3)  // Block loop
    movi v26.4s, #0x0
    mov x20, x14
KAI_ASM_LABEL(label_4)  // Sub block loop
    ldr q25, [x25, #0x0]
    ldr q24, [x22, #0x0]
    subs x20, x20, #0x1
    ldr q23, [x25, #0x10]
    ldr q22, [x25, #0x20]
    ldr q21, [x25, #0x30]
    ldr q20, [x22, #0x10]
    add x25, x25, #0x40
    add x22, x22, #0x20
    shl v19.16b, v25.16b, #0x4
    and v25.16b, v25.16b, v28.16b
    shl v18.16b, v23.16b, #0x4
    shl v17.16b, v22.16b, #0x4
    shl v16.16b, v21.16b, #0x4
    and v23.16b, v23.16b, v28.16b
    KAI_ASM_INST(0x4f98e27a)  // sdot v26.4s, v19.16b, v24.4b[0]
    and v22.16b, v22.16b, v28.16b
    and v21.16b, v21.16b, v28.16b
    KAI_ASM_INST(0x4fb8e25a)  // sdot v26.4s, v18.16b, v24.4b[1]
    KAI_ASM_INST(0x4f98ea3a)  // sdot v26.4s, v17.16b, v24.4b[2]
    KAI_ASM_INST(0x4fb8ea1a)  // sdot v26.4s, v16.16b, v24.4b[3]
    KAI_ASM_INST(0x4f94e33a)  // sdot v26.4s, v25.16b, v20.4b[0]
    KAI_ASM_INST(0x4fb4e2fa)  // sdot v26.4s, v23.16b, v20.4b[1]
    KAI_ASM_INST(0x4f94eada)  // sdot v26.4s, v22.16b, v20.4b[2]
    KAI_ASM_INST(0x4fb4eaba)  // sdot v26.4s, v21.16b, v20.4b[3]
    bgt label_4
    ldr d16, [x25, #0x0]
    scvtf v26.4s, v26.4s, #0x4
    sub x21, x21, #0x1
    add x25, x25, #0x8
    shll v16.4s, v16.4h, #0x10
    fmla v27.4s, v26.4s, v16.4s
    cbnz x21, label_3
    ld1r { v21.4s }, [x22]
    ldr q20, [x25, #0x0]
    add x22, x22, #0x4
    add x20, x27, #0x4
    ld1r { v19.4s }, [x22]
    ldr q18, [x25, #0x10]
    cmp x24, #0x4
    add x25, x25, #0x20
    ld1r { v17.4s }, [x27]
    ld1r { v16.4s }, [x20]
    scvtf v21.4s, v21.4s
    fmla v27.4s, v20.4s, v21.s[0]
    fmul v27.4s, v27.4s, v19.4s
    fadd v27.4s, v27.4s, v18.4s
    fmax v27.4s, v27.4s, v17.4s
    fmin v27.4s, v27.4s, v16.4s
    blt label_5
    str q27, [x9, #0x0]
    b label_8
KAI_ASM_LABEL(label_5)  // Partial output
    mov x20, x9
    tbz x24, #1, label_6
    st1 { v27.d }[0], [x20], #0x8
    tbz x24, #0, label_7
    st1 { v27.s }[2], [x20]
    b label_7
KAI_ASM_LABEL(label_6)  // Output block 0: partial_1_0
    st1 { v27.s }[0], [x20]
KAI_ASM_LABEL(label_7)  // Output block 0: Done
KAI_ASM_LABEL(label_8)  // Stores done
    subs x24, x24, #0x4
    add x9, x9, #0x10
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod)

    KAI_ASM_END
