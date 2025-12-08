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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d10, d11, [sp, 72]
    stp d12, d13, [sp, 88]
    stp d14, d15, [sp, 104]
    stp d8, d9, [sp, 120]
    mov x16, #0x80
    movi v9.16b, #0xf0
    mov x21, #0x20
    ldr x15, [x0, #0x40]
    ldr x20, [x0, #0x28]
    ldr x14, [x0, #0x38]
    ldr x13, [x0, #0x8]
    ldr x12, [x0, #0x10]
    ldr x11, [x0, #0x30]
    mul x16, x15, x16
    mov x10, x20
    ldr x9, [x0, #0x0]
    ldr x28, [x0, #0x20]
    ldr x27, [x0, #0x18]
    madd x16, x14, x16, x21
    cbz x10, label_12
KAI_ASM_LABEL(label_1)  // Row loop
    mov x26, x12
    mov x25, x11
    add x24, x9, x28, LSL #2
KAI_ASM_LABEL(label_2)  // Column loop
    movi v12.16b, #0x0
    movi v5.16b, #0x0
    mov x22, x13
    mov x21, x14
    movi v11.16b, #0x0
    movi v13.16b, #0x0
    movi v21.16b, #0x0
    movi v27.16b, #0x0
    movi v7.16b, #0x0
    movi v4.16b, #0x0
KAI_ASM_LABEL(label_3)  // Block loop
    movi v10.4s, #0x0
    movi v26.4s, #0x0
    mov x20, x15
    movi v18.4s, #0x0
    movi v2.4s, #0x0
    movi v3.4s, #0x0
    movi v16.4s, #0x0
    movi v8.4s, #0x0
    movi v19.4s, #0x0
KAI_ASM_LABEL(label_4)  // Sub block loop
    ldr q14, [x26, #0x0]
    ldr q28, [x26, #0x10]
    subs x20, x20, #0x1
    ldr q23, [x26, #0x20]
    ldr q1, [x26, #0x30]
    ldr q22, [x22, #0x0]
    ldr q15, [x22, #0x10]
    ldr q25, [x26, #0x40]
    ldr q17, [x26, #0x50]
    shl v31.16b, v14.16b, #0x4
    shl v0.16b, v28.16b, #0x4
    ldr q6, [x26, #0x60]
    ldr q24, [x26, #0x70]
    shl v20.16b, v23.16b, #0x4
    shl v29.16b, v1.16b, #0x4
    ldr q30, [x22, #0x20]
    and v14.16b, v14.16b, v9.16b
    and v28.16b, v28.16b, v9.16b
    add x26, x26, #0x80
    KAI_ASM_INST(0x4e9fa6ca)  // smmla v10.4s, v22.16b, v31.16b
    KAI_ASM_INST(0x4e80a6d2)  // smmla v18.4s, v22.16b, v0.16b
    and v23.16b, v23.16b, v9.16b
    KAI_ASM_INST(0x4e94a6da)  // smmla v26.4s, v22.16b, v20.16b
    KAI_ASM_INST(0x4e9da6c2)  // smmla v2.4s, v22.16b, v29.16b
    ldr q22, [x22, #0x30]
    and v1.16b, v1.16b, v9.16b
    KAI_ASM_INST(0x4e9fa5e3)  // smmla v3.4s, v15.16b, v31.16b
    ldr q31, [x22, #0x40]
    KAI_ASM_INST(0x4e80a5e8)  // smmla v8.4s, v15.16b, v0.16b
    ldr q0, [x22, #0x50]
    KAI_ASM_INST(0x4e94a5f0)  // smmla v16.4s, v15.16b, v20.16b
    ldr q20, [x22, #0x60]
    KAI_ASM_INST(0x4e9da5f3)  // smmla v19.4s, v15.16b, v29.16b
    ldr q29, [x22, #0x70]
    shl v15.16b, v25.16b, #0x4
    and v25.16b, v25.16b, v9.16b
    add x22, x22, #0x80
    KAI_ASM_INST(0x4e8fa7ca)  // smmla v10.4s, v30.16b, v15.16b
    KAI_ASM_INST(0x4e8fa6c3)  // smmla v3.4s, v22.16b, v15.16b
    shl v15.16b, v17.16b, #0x4
    and v17.16b, v17.16b, v9.16b
    KAI_ASM_INST(0x4e8fa7d2)  // smmla v18.4s, v30.16b, v15.16b
    KAI_ASM_INST(0x4e8fa6c8)  // smmla v8.4s, v22.16b, v15.16b
    shl v15.16b, v6.16b, #0x4
    and v6.16b, v6.16b, v9.16b
    KAI_ASM_INST(0x4e8ea7ea)  // smmla v10.4s, v31.16b, v14.16b
    KAI_ASM_INST(0x4e8ea403)  // smmla v3.4s, v0.16b, v14.16b
    shl v14.16b, v24.16b, #0x4
    and v24.16b, v24.16b, v9.16b
    KAI_ASM_INST(0x4e8fa7da)  // smmla v26.4s, v30.16b, v15.16b
    KAI_ASM_INST(0x4e8fa6d0)  // smmla v16.4s, v22.16b, v15.16b
    KAI_ASM_INST(0x4e9ca7f2)  // smmla v18.4s, v31.16b, v28.16b
    KAI_ASM_INST(0x4e9ca408)  // smmla v8.4s, v0.16b, v28.16b
    KAI_ASM_INST(0x4e8ea7c2)  // smmla v2.4s, v30.16b, v14.16b
    KAI_ASM_INST(0x4e8ea6d3)  // smmla v19.4s, v22.16b, v14.16b
    KAI_ASM_INST(0x4e99a68a)  // smmla v10.4s, v20.16b, v25.16b
    KAI_ASM_INST(0x4e99a7a3)  // smmla v3.4s, v29.16b, v25.16b
    KAI_ASM_INST(0x4e97a7fa)  // smmla v26.4s, v31.16b, v23.16b
    KAI_ASM_INST(0x4e97a410)  // smmla v16.4s, v0.16b, v23.16b
    KAI_ASM_INST(0x4e91a692)  // smmla v18.4s, v20.16b, v17.16b
    KAI_ASM_INST(0x4e91a7a8)  // smmla v8.4s, v29.16b, v17.16b
    KAI_ASM_INST(0x4e81a7e2)  // smmla v2.4s, v31.16b, v1.16b
    KAI_ASM_INST(0x4e81a413)  // smmla v19.4s, v0.16b, v1.16b
    KAI_ASM_INST(0x4e86a69a)  // smmla v26.4s, v20.16b, v6.16b
    KAI_ASM_INST(0x4e86a7b0)  // smmla v16.4s, v29.16b, v6.16b
    KAI_ASM_INST(0x4e98a682)  // smmla v2.4s, v20.16b, v24.16b
    KAI_ASM_INST(0x4e98a7b3)  // smmla v19.4s, v29.16b, v24.16b
    bgt label_4
    ldr q0, [x26, #0x0]
    uzp1 v14.2d, v10.2d, v18.2d
    uzp2 v30.2d, v10.2d, v18.2d
    add x26, x26, #0x10
    uzp1 v25.2d, v26.2d, v2.2d
    uzp2 v22.2d, v26.2d, v2.2d
    uzp1 v24.2d, v3.2d, v8.2d
    uzp2 v20.2d, v3.2d, v8.2d
    uzp1 v28.2d, v16.2d, v19.2d
    uzp2 v18.2d, v16.2d, v19.2d
    shll v17.4s, v0.4h, #0x10
    shll2 v16.4s, v0.8h, #0x10
    scvtf v14.4s, v14.4s, #0x4
    scvtf v25.4s, v25.4s, #0x4
    scvtf v30.4s, v30.4s, #0x4
    scvtf v22.4s, v22.4s, #0x4
    scvtf v24.4s, v24.4s, #0x4
    scvtf v28.4s, v28.4s, #0x4
    scvtf v20.4s, v20.4s, #0x4
    scvtf v18.4s, v18.4s, #0x4
    fmla v12.4s, v14.4s, v17.4s
    fmla v5.4s, v25.4s, v16.4s
    fmla v11.4s, v30.4s, v17.4s
    fmla v13.4s, v22.4s, v16.4s
    fmla v21.4s, v24.4s, v17.4s
    fmla v27.4s, v28.4s, v16.4s
    fmla v7.4s, v20.4s, v17.4s
    fmla v4.4s, v18.4s, v16.4s
    subs x21, x21, #0x1
    bgt label_3
    ld1 { v23.4s }, [x22]
    ldr q22, [x26, #0x0]
    add x22, x22, #0x10
    add x20, x27, #0x4
    ldr q6, [x26, #0x10]
    ldr q20, [x22, #0x0]
    cmp x25, #0x8
    ldr q19, [x26, #0x20]
    ldr q18, [x26, #0x30]
    add x26, x26, #0x40
    ld1r { v17.4s }, [x27]
    ld1r { v16.4s }, [x20]
    scvtf v23.4s, v23.4s
    fmla v12.4s, v22.4s, v23.s[0]
    fmla v5.4s, v6.4s, v23.s[0]
    fmla v11.4s, v22.4s, v23.s[1]
    fmla v13.4s, v6.4s, v23.s[1]
    fmla v21.4s, v22.4s, v23.s[2]
    fmla v27.4s, v6.4s, v23.s[2]
    fmla v7.4s, v22.4s, v23.s[3]
    fmla v4.4s, v6.4s, v23.s[3]
    fmul v12.4s, v12.4s, v20.s[0]
    fmul v5.4s, v5.4s, v20.s[0]
    fmul v11.4s, v11.4s, v20.s[1]
    fmul v13.4s, v13.4s, v20.s[1]
    fmul v21.4s, v21.4s, v20.s[2]
    fmul v27.4s, v27.4s, v20.s[2]
    fmul v7.4s, v7.4s, v20.s[3]
    fmul v4.4s, v4.4s, v20.s[3]
    fadd v12.4s, v12.4s, v19.4s
    fadd v5.4s, v5.4s, v18.4s
    fadd v11.4s, v11.4s, v19.4s
    fadd v13.4s, v13.4s, v18.4s
    fadd v21.4s, v21.4s, v19.4s
    fadd v27.4s, v27.4s, v18.4s
    fadd v7.4s, v7.4s, v19.4s
    fadd v4.4s, v4.4s, v18.4s
    fmax v12.4s, v12.4s, v17.4s
    fmax v5.4s, v5.4s, v17.4s
    fmax v11.4s, v11.4s, v17.4s
    fmax v13.4s, v13.4s, v17.4s
    fmax v21.4s, v21.4s, v17.4s
    fmax v27.4s, v27.4s, v17.4s
    fmax v7.4s, v7.4s, v17.4s
    fmax v4.4s, v4.4s, v17.4s
    fmin v12.4s, v12.4s, v16.4s
    fmin v5.4s, v5.4s, v16.4s
    fmin v11.4s, v11.4s, v16.4s
    fmin v13.4s, v13.4s, v16.4s
    fmin v21.4s, v21.4s, v16.4s
    fmin v27.4s, v27.4s, v16.4s
    fmin v7.4s, v7.4s, v16.4s
    fmin v4.4s, v4.4s, v16.4s
    blt label_6
    mov x20, x9
    cmp x10, #0x1
    str q12, [x20, #0x0]
    str q5, [x20, #0x10]
    add x20, x20, x28
    ble label_11
    cmp x10, #0x2
    str q11, [x20, #0x0]
    str q13, [x20, #0x10]
    add x20, x20, x28
    ble label_11
    cmp x10, #0x3
    str q21, [x20, #0x0]
    str q27, [x20, #0x10]
    add x20, x20, x28
    ble label_11
    str q7, [x20, #0x0]
    str q4, [x20, #0x10]
    b label_11
KAI_ASM_LABEL(label_6)  // Partial output
    mov x23, x9
    cmp x10, #0x1
    add x22, x23, x28
    csel x22, x22, x23, GT
    cmp x10, #0x2
    add x21, x23, x28, LSL #1
    csel x21, x21, x22, GT
    cmp x10, #0x3
    add x20, x21, x28
    csel x20, x20, x21, GT
    tbz x25, #2, label_8
    st1 { v7.4s }, [x20], #0x10
    st1 { v21.4s }, [x21], #0x10
    st1 { v11.4s }, [x22], #0x10
    st1 { v12.4s }, [x23], #0x10
    tbz x25, #1, label_7
    st1 { v4.d }[0], [x20], #0x8
    st1 { v27.d }[0], [x21], #0x8
    st1 { v13.d }[0], [x22], #0x8
    st1 { v5.d }[0], [x23], #0x8
    tbz x25, #0, label_10
    st1 { v4.s }[2], [x20]
    st1 { v27.s }[2], [x21]
    st1 { v13.s }[2], [x22]
    st1 { v5.s }[2], [x23]
    b label_10
KAI_ASM_LABEL(label_7)  // Output block 0: partial_1_4
    tbz x25, #0, label_10
    st1 { v4.s }[0], [x20]
    st1 { v27.s }[0], [x21]
    st1 { v13.s }[0], [x22]
    st1 { v5.s }[0], [x23]
    b label_10
KAI_ASM_LABEL(label_8)  // Output block 0: partial_2_0
    tbz x25, #1, label_9
    st1 { v7.d }[0], [x20], #0x8
    st1 { v21.d }[0], [x21], #0x8
    st1 { v11.d }[0], [x22], #0x8
    st1 { v12.d }[0], [x23], #0x8
    tbz x25, #0, label_10
    st1 { v7.s }[2], [x20]
    st1 { v21.s }[2], [x21]
    st1 { v11.s }[2], [x22]
    st1 { v12.s }[2], [x23]
    b label_10
KAI_ASM_LABEL(label_9)  // Output block 0: partial_1_0
    st1 { v7.s }[0], [x20]
    st1 { v21.s }[0], [x21]
    st1 { v11.s }[0], [x22]
    st1 { v12.s }[0], [x23]
KAI_ASM_LABEL(label_10)  // Output block 0: Done
KAI_ASM_LABEL(label_11)  // Output stage exit
    subs x25, x25, #0x8
    add x9, x9, #0x20
    bgt label_2
    subs x10, x10, #0x4
    add x13, x13, x16
    mov x9, x24
    bgt label_1
KAI_ASM_LABEL(label_12)  // Row loop skip
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp d10, d11, [sp, 72]
    ldp d12, d13, [sp, 88]
    ldp d14, d15, [sp, 104]
    ldp d8, d9, [sp, 120]
    ldp x20, x21, [sp], 144
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm)

    KAI_ASM_END
