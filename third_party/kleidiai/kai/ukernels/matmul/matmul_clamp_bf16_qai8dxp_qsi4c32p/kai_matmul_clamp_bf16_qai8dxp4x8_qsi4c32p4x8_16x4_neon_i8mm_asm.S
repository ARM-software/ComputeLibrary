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

    KAI_ASM_CODE(matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d10, d11, [sp, 72]
    stp d12, d13, [sp, 88]
    stp d14, d15, [sp, 104]
    stp d8, d9, [sp, 120]
    mov x5, #0x80
    mov x21, #0x20
    sub SP, SP, #0x100
    ldr x20, [x0, #0x28]
    ldr x6, [x0, #0x40]
    ldr x7, [x0, #0x38]
    ldr x8, [x0, #0x8]
    ldr x17, [x0, #0x10]
    ldr x16, [x0, #0x30]
    mov x15, x20
    mul x5, x6, x5
    ldr x14, [x0, #0x0]
    ldr x13, [x0, #0x20]
    ldr x12, [x0, #0x18]
    cmp x15, #0x10
    madd x5, x7, x5, x21
    blt label_15
KAI_ASM_LABEL(label_1)  // Row loop
    mov x11, x17
    mov x10, x16
    add x9, x14, x13, LSL #4
KAI_ASM_LABEL(label_2)  // Column loop
    mov x27, x8
    movi v6.4s, #0x0
    mov x24, x7
    str q6, [SP, #0x0]
    str q6, [SP, #0x10]
    str q6, [SP, #0x20]
    add x23, x27, x5
    add x22, x23, x5
    str q6, [SP, #0x30]
    add x21, x22, x5
    str q6, [SP, #0x40]
    str q6, [SP, #0x50]
    str q6, [SP, #0x60]
    str q6, [SP, #0x70]
    str q6, [SP, #0x80]
    str q6, [SP, #0x90]
    str q6, [SP, #0xa0]
    str q6, [SP, #0xb0]
    str q6, [SP, #0xc0]
    str q6, [SP, #0xd0]
    str q6, [SP, #0xe0]
    str q6, [SP, #0xf0]
KAI_ASM_LABEL(label_3)  // Block loop
    movi v14.4s, #0x0
    movi v24.4s, #0x0
    mov x20, x6
    movi v13.4s, #0x0
    movi v11.4s, #0x0
    movi v15.4s, #0x0
    movi v22.4s, #0x0
    movi v12.4s, #0x0
    movi v4.4s, #0x0
    movi v25.4s, #0x0
    movi v17.4s, #0x0
    movi v5.4s, #0x0
    movi v10.4s, #0x0
    movi v20.4s, #0x0
    movi v9.4s, #0x0
    movi v27.4s, #0x0
    movi v18.4s, #0x0
KAI_ASM_LABEL(label_4)  // Sub block loop
    ldr q8, [x11, #0x0]
    ldr q3, [x11, #0x10]
    movi v16.16b, #0xf0
    subs x20, x20, #0x1
    ldr q31, [x27, #0x0]
    ldr q2, [x27, #0x10]
    ldr q28, [x23, #0x0]
    ldr q23, [x23, #0x10]
    ldr q26, [x22, #0x0]
    ldr q30, [x22, #0x10]
    shl v21.16b, v8.16b, #0x4
    shl v1.16b, v3.16b, #0x4
    ldr q7, [x21, #0x0]
    ldr q19, [x21, #0x10]
    and v8.16b, v8.16b, v16.16b
    and v3.16b, v3.16b, v16.16b
    ldr q0, [x11, #0x20]
    ldr q29, [x11, #0x30]
    add x11, x11, #0x40
    ldr q6, [x27, #0x20]
    KAI_ASM_INST(0x4e95a7ee)  // smmla v14.4s, v31.16b, v21.16b
    KAI_ASM_INST(0x4e81a7f8)  // smmla v24.4s, v31.16b, v1.16b
    ldr q31, [x27, #0x30]
    KAI_ASM_INST(0x4e95a44d)  // smmla v13.4s, v2.16b, v21.16b
    KAI_ASM_INST(0x4e81a44b)  // smmla v11.4s, v2.16b, v1.16b
    ldr q2, [x23, #0x20]
    KAI_ASM_INST(0x4e95a78f)  // smmla v15.4s, v28.16b, v21.16b
    KAI_ASM_INST(0x4e81a796)  // smmla v22.4s, v28.16b, v1.16b
    ldr q28, [x23, #0x30]
    KAI_ASM_INST(0x4e95a6ec)  // smmla v12.4s, v23.16b, v21.16b
    KAI_ASM_INST(0x4e81a6e4)  // smmla v4.4s, v23.16b, v1.16b
    ldr q23, [x22, #0x20]
    KAI_ASM_INST(0x4e95a759)  // smmla v25.4s, v26.16b, v21.16b
    KAI_ASM_INST(0x4e81a751)  // smmla v17.4s, v26.16b, v1.16b
    ldr q26, [x22, #0x30]
    KAI_ASM_INST(0x4e95a7c5)  // smmla v5.4s, v30.16b, v21.16b
    KAI_ASM_INST(0x4e81a7ca)  // smmla v10.4s, v30.16b, v1.16b
    ldr q30, [x21, #0x20]
    KAI_ASM_INST(0x4e95a4f4)  // smmla v20.4s, v7.16b, v21.16b
    KAI_ASM_INST(0x4e81a4e9)  // smmla v9.4s, v7.16b, v1.16b
    ldr q7, [x21, #0x30]
    KAI_ASM_INST(0x4e95a67b)  // smmla v27.4s, v19.16b, v21.16b
    ldr q21, [x27, #0x40]
    KAI_ASM_INST(0x4e81a672)  // smmla v18.4s, v19.16b, v1.16b
    ldr q19, [x27, #0x50]
    shl v1.16b, v0.16b, #0x4
    and v0.16b, v0.16b, v16.16b
    KAI_ASM_INST(0x4e81a4ce)  // smmla v14.4s, v6.16b, v1.16b
    KAI_ASM_INST(0x4e81a7ed)  // smmla v13.4s, v31.16b, v1.16b
    KAI_ASM_INST(0x4e81a44f)  // smmla v15.4s, v2.16b, v1.16b
    KAI_ASM_INST(0x4e81a78c)  // smmla v12.4s, v28.16b, v1.16b
    KAI_ASM_INST(0x4e81a6f9)  // smmla v25.4s, v23.16b, v1.16b
    KAI_ASM_INST(0x4e81a745)  // smmla v5.4s, v26.16b, v1.16b
    KAI_ASM_INST(0x4e81a7d4)  // smmla v20.4s, v30.16b, v1.16b
    KAI_ASM_INST(0x4e81a4fb)  // smmla v27.4s, v7.16b, v1.16b
    shl v1.16b, v29.16b, #0x4
    KAI_ASM_INST(0x4e88a6ae)  // smmla v14.4s, v21.16b, v8.16b
    KAI_ASM_INST(0x4e88a66d)  // smmla v13.4s, v19.16b, v8.16b
    and v29.16b, v29.16b, v16.16b
    ldr q16, [x23, #0x40]
    KAI_ASM_INST(0x4e81a4d8)  // smmla v24.4s, v6.16b, v1.16b
    ldr q6, [x23, #0x50]
    KAI_ASM_INST(0x4e81a7eb)  // smmla v11.4s, v31.16b, v1.16b
    ldr q31, [x22, #0x40]
    KAI_ASM_INST(0x4e81a456)  // smmla v22.4s, v2.16b, v1.16b
    ldr q2, [x22, #0x50]
    KAI_ASM_INST(0x4e81a784)  // smmla v4.4s, v28.16b, v1.16b
    ldr q28, [x21, #0x40]
    KAI_ASM_INST(0x4e81a6f1)  // smmla v17.4s, v23.16b, v1.16b
    ldr q23, [x21, #0x50]
    KAI_ASM_INST(0x4e81a74a)  // smmla v10.4s, v26.16b, v1.16b
    ldr q26, [x27, #0x60]
    KAI_ASM_INST(0x4e81a7c9)  // smmla v9.4s, v30.16b, v1.16b
    ldr q30, [x27, #0x70]
    KAI_ASM_INST(0x4e81a4f2)  // smmla v18.4s, v7.16b, v1.16b
    ldr q1, [x23, #0x60]
    KAI_ASM_INST(0x4e83a6b8)  // smmla v24.4s, v21.16b, v3.16b
    ldr q21, [x23, #0x70]
    ldr q7, [x22, #0x60]
    KAI_ASM_INST(0x4e83a66b)  // smmla v11.4s, v19.16b, v3.16b
    KAI_ASM_INST(0x4e88a60f)  // smmla v15.4s, v16.16b, v8.16b
    ldr q19, [x22, #0x70]
    KAI_ASM_INST(0x4e83a616)  // smmla v22.4s, v16.16b, v3.16b
    ldr q16, [x21, #0x60]
    KAI_ASM_INST(0x4e88a4cc)  // smmla v12.4s, v6.16b, v8.16b
    KAI_ASM_INST(0x4e83a4c4)  // smmla v4.4s, v6.16b, v3.16b
    ldr q6, [x21, #0x70]
    KAI_ASM_INST(0x4e88a7f9)  // smmla v25.4s, v31.16b, v8.16b
    add x27, x27, #0x80
    KAI_ASM_INST(0x4e83a7f1)  // smmla v17.4s, v31.16b, v3.16b
    KAI_ASM_INST(0x4e88a445)  // smmla v5.4s, v2.16b, v8.16b
    add x23, x23, #0x80
    add x22, x22, #0x80
    KAI_ASM_INST(0x4e83a44a)  // smmla v10.4s, v2.16b, v3.16b
    KAI_ASM_INST(0x4e88a794)  // smmla v20.4s, v28.16b, v8.16b
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e83a789)  // smmla v9.4s, v28.16b, v3.16b
    KAI_ASM_INST(0x4e88a6fb)  // smmla v27.4s, v23.16b, v8.16b
    KAI_ASM_INST(0x4e83a6f2)  // smmla v18.4s, v23.16b, v3.16b
    KAI_ASM_INST(0x4e80a74e)  // smmla v14.4s, v26.16b, v0.16b
    KAI_ASM_INST(0x4e9da758)  // smmla v24.4s, v26.16b, v29.16b
    KAI_ASM_INST(0x4e80a7cd)  // smmla v13.4s, v30.16b, v0.16b
    KAI_ASM_INST(0x4e9da7cb)  // smmla v11.4s, v30.16b, v29.16b
    KAI_ASM_INST(0x4e80a42f)  // smmla v15.4s, v1.16b, v0.16b
    KAI_ASM_INST(0x4e9da436)  // smmla v22.4s, v1.16b, v29.16b
    KAI_ASM_INST(0x4e80a6ac)  // smmla v12.4s, v21.16b, v0.16b
    KAI_ASM_INST(0x4e9da6a4)  // smmla v4.4s, v21.16b, v29.16b
    KAI_ASM_INST(0x4e80a4f9)  // smmla v25.4s, v7.16b, v0.16b
    KAI_ASM_INST(0x4e9da4f1)  // smmla v17.4s, v7.16b, v29.16b
    KAI_ASM_INST(0x4e80a665)  // smmla v5.4s, v19.16b, v0.16b
    KAI_ASM_INST(0x4e9da66a)  // smmla v10.4s, v19.16b, v29.16b
    KAI_ASM_INST(0x4e80a614)  // smmla v20.4s, v16.16b, v0.16b
    KAI_ASM_INST(0x4e9da609)  // smmla v9.4s, v16.16b, v29.16b
    KAI_ASM_INST(0x4e80a4db)  // smmla v27.4s, v6.16b, v0.16b
    KAI_ASM_INST(0x4e9da4d2)  // smmla v18.4s, v6.16b, v29.16b
    bgt label_4
    ldr d19, [x11, #0x0]
    ldr q28, [SP, #0x0]
    uzp1 v1.2d, v14.2d, v24.2d
    uzp2 v23.2d, v14.2d, v24.2d
    ldr q21, [SP, #0x10]
    ldr q6, [SP, #0x20]
    uzp1 v31.2d, v13.2d, v11.2d
    uzp2 v24.2d, v13.2d, v11.2d
    ldr q16, [SP, #0x30]
    add x11, x11, #0x8
    shll v26.4s, v19.4h, #0x10
    scvtf v1.4s, v1.4s, #0x4
    scvtf v23.4s, v23.4s, #0x4
    scvtf v31.4s, v31.4s, #0x4
    scvtf v24.4s, v24.4s, #0x4
    fmla v28.4s, v1.4s, v26.4s
    fmla v21.4s, v23.4s, v26.4s
    fmla v6.4s, v31.4s, v26.4s
    fmla v16.4s, v24.4s, v26.4s
    str q28, [SP, #0x0]
    str q21, [SP, #0x10]
    str q6, [SP, #0x20]
    str q16, [SP, #0x30]
    ldr q11, [SP, #0x40]
    ldr q8, [SP, #0x50]
    uzp1 v23.2d, v15.2d, v22.2d
    uzp2 v21.2d, v15.2d, v22.2d
    ldr q19, [SP, #0x60]
    ldr q2, [SP, #0x70]
    uzp1 v3.2d, v12.2d, v4.2d
    uzp2 v16.2d, v12.2d, v4.2d
    scvtf v23.4s, v23.4s, #0x4
    scvtf v21.4s, v21.4s, #0x4
    scvtf v3.4s, v3.4s, #0x4
    scvtf v16.4s, v16.4s, #0x4
    fmla v11.4s, v23.4s, v26.4s
    fmla v8.4s, v21.4s, v26.4s
    fmla v19.4s, v3.4s, v26.4s
    fmla v2.4s, v16.4s, v26.4s
    str q11, [SP, #0x40]
    str q8, [SP, #0x50]
    str q19, [SP, #0x60]
    str q2, [SP, #0x70]
    ldr q8, [SP, #0x80]
    ldr q23, [SP, #0x90]
    uzp1 v3.2d, v25.2d, v17.2d
    uzp2 v21.2d, v25.2d, v17.2d
    ldr q19, [SP, #0xa0]
    ldr q4, [SP, #0xb0]
    uzp1 v24.2d, v5.2d, v10.2d
    uzp2 v16.2d, v5.2d, v10.2d
    scvtf v3.4s, v3.4s, #0x4
    scvtf v21.4s, v21.4s, #0x4
    scvtf v24.4s, v24.4s, #0x4
    scvtf v16.4s, v16.4s, #0x4
    fmla v8.4s, v3.4s, v26.4s
    fmla v23.4s, v21.4s, v26.4s
    fmla v19.4s, v24.4s, v26.4s
    fmla v4.4s, v16.4s, v26.4s
    str q8, [SP, #0x80]
    str q23, [SP, #0x90]
    str q19, [SP, #0xa0]
    str q4, [SP, #0xb0]
    ldr q23, [SP, #0xc0]
    ldr q22, [SP, #0xd0]
    uzp1 v21.2d, v20.2d, v9.2d
    uzp2 v20.2d, v20.2d, v9.2d
    ldr q19, [SP, #0xe0]
    ldr q8, [SP, #0xf0]
    uzp1 v4.2d, v27.2d, v18.2d
    uzp2 v16.2d, v27.2d, v18.2d
    scvtf v21.4s, v21.4s, #0x4
    scvtf v20.4s, v20.4s, #0x4
    scvtf v4.4s, v4.4s, #0x4
    scvtf v16.4s, v16.4s, #0x4
    fmla v23.4s, v21.4s, v26.4s
    fmla v22.4s, v20.4s, v26.4s
    fmla v19.4s, v4.4s, v26.4s
    fmla v8.4s, v16.4s, v26.4s
    str q23, [SP, #0xc0]
    str q22, [SP, #0xd0]
    str q19, [SP, #0xe0]
    str q8, [SP, #0xf0]
    subs x24, x24, #0x1
    bgt label_3
    ld1 { v11.4s }, [x27]
    ld1 { v10.4s }, [x23]
    add x27, x27, #0x10
    add x23, x23, #0x10
    ld1 { v9.4s }, [x22]
    ld1 { v8.4s }, [x21]
    add x22, x22, #0x10
    add x21, x21, #0x10
    ldr q31, [SP, #0x0]
    ldr q30, [SP, #0x10]
    add x20, x12, #0x4
    cmp x10, #0x4
    ldr q29, [SP, #0x20]
    ldr q28, [SP, #0x30]
    scvtf v11.4s, v11.4s
    scvtf v10.4s, v10.4s
    ldr q27, [SP, #0x40]
    ldr q26, [SP, #0x50]
    scvtf v9.4s, v9.4s
    scvtf v8.4s, v8.4s
    ldr q25, [SP, #0x60]
    ldr q24, [SP, #0x70]
    ldr q23, [SP, #0x80]
    ldr q22, [SP, #0x90]
    ldr q21, [SP, #0xa0]
    ldr q20, [SP, #0xb0]
    ldr q19, [SP, #0xc0]
    ldr q18, [SP, #0xd0]
    ldr q17, [SP, #0xe0]
    ldr q16, [SP, #0xf0]
    ldr q7, [x11, #0x0]
    ldr q6, [x27, #0x0]
    ldr q5, [x23, #0x0]
    ldr q4, [x22, #0x0]
    ldr q3, [x21, #0x0]
    ldr q2, [x11, #0x10]
    add x11, x11, #0x20
    ld1r { v1.4s }, [x12]
    ld1r { v0.4s }, [x20]
    fmla v31.4s, v7.4s, v11.s[0]
    fmla v30.4s, v7.4s, v11.s[1]
    fmla v29.4s, v7.4s, v11.s[2]
    fmla v28.4s, v7.4s, v11.s[3]
    fmla v27.4s, v7.4s, v10.s[0]
    fmla v26.4s, v7.4s, v10.s[1]
    fmla v25.4s, v7.4s, v10.s[2]
    fmla v24.4s, v7.4s, v10.s[3]
    fmla v23.4s, v7.4s, v9.s[0]
    fmla v22.4s, v7.4s, v9.s[1]
    fmul v31.4s, v31.4s, v6.s[0]
    fmla v21.4s, v7.4s, v9.s[2]
    fmla v20.4s, v7.4s, v9.s[3]
    fmul v30.4s, v30.4s, v6.s[1]
    fmla v19.4s, v7.4s, v8.s[0]
    fmla v18.4s, v7.4s, v8.s[1]
    fmul v29.4s, v29.4s, v6.s[2]
    fmla v17.4s, v7.4s, v8.s[2]
    fmla v16.4s, v7.4s, v8.s[3]
    fmul v28.4s, v28.4s, v6.s[3]
    fmul v27.4s, v27.4s, v5.s[0]
    fmul v26.4s, v26.4s, v5.s[1]
    fmul v25.4s, v25.4s, v5.s[2]
    fmul v24.4s, v24.4s, v5.s[3]
    fmul v23.4s, v23.4s, v4.s[0]
    fmul v22.4s, v22.4s, v4.s[1]
    fmul v21.4s, v21.4s, v4.s[2]
    fmul v20.4s, v20.4s, v4.s[3]
    fmul v19.4s, v19.4s, v3.s[0]
    fmul v18.4s, v18.4s, v3.s[1]
    fmul v17.4s, v17.4s, v3.s[2]
    fmul v16.4s, v16.4s, v3.s[3]
    fadd v31.4s, v31.4s, v2.4s
    fadd v30.4s, v30.4s, v2.4s
    fadd v29.4s, v29.4s, v2.4s
    fadd v28.4s, v28.4s, v2.4s
    fadd v27.4s, v27.4s, v2.4s
    fadd v26.4s, v26.4s, v2.4s
    fadd v25.4s, v25.4s, v2.4s
    fadd v24.4s, v24.4s, v2.4s
    fadd v23.4s, v23.4s, v2.4s
    fadd v22.4s, v22.4s, v2.4s
    fadd v21.4s, v21.4s, v2.4s
    fadd v20.4s, v20.4s, v2.4s
    fadd v19.4s, v19.4s, v2.4s
    fadd v18.4s, v18.4s, v2.4s
    fadd v17.4s, v17.4s, v2.4s
    fadd v16.4s, v16.4s, v2.4s
    fmax v31.4s, v31.4s, v1.4s
    fmax v30.4s, v30.4s, v1.4s
    fmax v29.4s, v29.4s, v1.4s
    fmax v28.4s, v28.4s, v1.4s
    fmax v27.4s, v27.4s, v1.4s
    fmax v26.4s, v26.4s, v1.4s
    fmax v25.4s, v25.4s, v1.4s
    fmax v24.4s, v24.4s, v1.4s
    fmax v23.4s, v23.4s, v1.4s
    fmax v22.4s, v22.4s, v1.4s
    fmax v21.4s, v21.4s, v1.4s
    fmax v20.4s, v20.4s, v1.4s
    fmax v19.4s, v19.4s, v1.4s
    fmax v18.4s, v18.4s, v1.4s
    fmax v17.4s, v17.4s, v1.4s
    fmax v16.4s, v16.4s, v1.4s
    fmin v31.4s, v31.4s, v0.4s
    fmin v30.4s, v30.4s, v0.4s
    fmin v29.4s, v29.4s, v0.4s
    fmin v28.4s, v28.4s, v0.4s
    fmin v27.4s, v27.4s, v0.4s
    fmin v26.4s, v26.4s, v0.4s
    fmin v25.4s, v25.4s, v0.4s
    fmin v24.4s, v24.4s, v0.4s
    fmin v23.4s, v23.4s, v0.4s
    fmin v22.4s, v22.4s, v0.4s
    fmin v21.4s, v21.4s, v0.4s
    fmin v20.4s, v20.4s, v0.4s
    fmin v19.4s, v19.4s, v0.4s
    fmin v18.4s, v18.4s, v0.4s
    fmin v17.4s, v17.4s, v0.4s
    fmin v16.4s, v16.4s, v0.4s
    KAI_ASM_INST(0x0ea16bff)  // bfcvtn v31.4h, v31.4s
    KAI_ASM_INST(0x0ea16bde)  // bfcvtn v30.4h, v30.4s
    KAI_ASM_INST(0x0ea16bbd)  // bfcvtn v29.4h, v29.4s
    KAI_ASM_INST(0x0ea16b9c)  // bfcvtn v28.4h, v28.4s
    KAI_ASM_INST(0x0ea16b7b)  // bfcvtn v27.4h, v27.4s
    KAI_ASM_INST(0x0ea16b5a)  // bfcvtn v26.4h, v26.4s
    KAI_ASM_INST(0x0ea16b39)  // bfcvtn v25.4h, v25.4s
    KAI_ASM_INST(0x0ea16b18)  // bfcvtn v24.4h, v24.4s
    KAI_ASM_INST(0x0ea16af7)  // bfcvtn v23.4h, v23.4s
    KAI_ASM_INST(0x0ea16ad6)  // bfcvtn v22.4h, v22.4s
    KAI_ASM_INST(0x0ea16ab5)  // bfcvtn v21.4h, v21.4s
    KAI_ASM_INST(0x0ea16a94)  // bfcvtn v20.4h, v20.4s
    KAI_ASM_INST(0x0ea16a73)  // bfcvtn v19.4h, v19.4s
    KAI_ASM_INST(0x0ea16a52)  // bfcvtn v18.4h, v18.4s
    KAI_ASM_INST(0x0ea16a31)  // bfcvtn v17.4h, v17.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    blt label_9
    mov x20, x14
    str d31, [x20, #0x0]
    add x20, x20, x13
    str d30, [x20, #0x0]
    add x20, x20, x13
    str d29, [x20, #0x0]
    add x20, x20, x13
    str d28, [x20, #0x0]
    add x20, x20, x13
    str d27, [x20, #0x0]
    add x20, x20, x13
    str d26, [x20, #0x0]
    add x20, x20, x13
    str d25, [x20, #0x0]
    add x20, x20, x13
    str d24, [x20, #0x0]
    add x20, x20, x13
    str d23, [x20, #0x0]
    add x20, x20, x13
    str d22, [x20, #0x0]
    add x20, x20, x13
    str d21, [x20, #0x0]
    add x20, x20, x13
    str d20, [x20, #0x0]
    add x20, x20, x13
    str d19, [x20, #0x0]
    add x20, x20, x13
    str d18, [x20, #0x0]
    add x20, x20, x13
    str d17, [x20, #0x0]
    add x20, x20, x13
    str d16, [x20, #0x0]
    b label_14
KAI_ASM_LABEL(label_9)  // Partial output
    mov x28, x14
    add x26, x28, x13, LSL #2
    add x25, x26, x13, LSL #1
    add x24, x26, x13
    add x23, x25, x13
    add x22, x28, x13, LSL #1
    add x21, x28, x13
    add x20, x22, x13
    add x27, x23, x13
    tbz x10, #1, label_10
    st1 { v24.s }[0], [x23], #0x4
    st1 { v25.s }[0], [x25], #0x4
    st1 { v26.s }[0], [x24], #0x4
    st1 { v27.s }[0], [x26], #0x4
    st1 { v28.s }[0], [x20], #0x4
    st1 { v29.s }[0], [x22], #0x4
    st1 { v30.s }[0], [x21], #0x4
    st1 { v31.s }[0], [x28], #0x4
    tbz x10, #0, label_11
    st1 { v24.h }[2], [x23]
    st1 { v25.h }[2], [x25]
    st1 { v26.h }[2], [x24]
    st1 { v27.h }[2], [x26]
    st1 { v28.h }[2], [x20]
    st1 { v29.h }[2], [x22]
    st1 { v30.h }[2], [x21]
    st1 { v31.h }[2], [x28]
    b label_11
KAI_ASM_LABEL(label_10)  // Output block 0: partial_1_0
    st1 { v24.h }[0], [x23]
    st1 { v25.h }[0], [x25]
    st1 { v26.h }[0], [x24]
    st1 { v27.h }[0], [x26]
    st1 { v28.h }[0], [x20]
    st1 { v29.h }[0], [x22]
    st1 { v30.h }[0], [x21]
    st1 { v31.h }[0], [x28]
KAI_ASM_LABEL(label_11)  // Output block 0: Done
    add x26, x27, x13, LSL #2
    add x25, x27, x13, LSL #1
    add x24, x26, x13, LSL #1
    add x23, x27, x13
    add x22, x25, x13
    add x21, x26, x13
    add x20, x24, x13
    tbz x10, #1, label_12
    st1 { v16.s }[0], [x20], #0x4
    st1 { v17.s }[0], [x24], #0x4
    st1 { v18.s }[0], [x21], #0x4
    st1 { v19.s }[0], [x26], #0x4
    st1 { v20.s }[0], [x22], #0x4
    st1 { v21.s }[0], [x25], #0x4
    st1 { v22.s }[0], [x23], #0x4
    st1 { v23.s }[0], [x27], #0x4
    tbz x10, #0, label_13
    st1 { v16.h }[2], [x20]
    st1 { v17.h }[2], [x24]
    st1 { v18.h }[2], [x21]
    st1 { v19.h }[2], [x26]
    st1 { v20.h }[2], [x22]
    st1 { v21.h }[2], [x25]
    st1 { v22.h }[2], [x23]
    st1 { v23.h }[2], [x27]
    b label_13
KAI_ASM_LABEL(label_12)  // Output block 1: partial_1_0
    st1 { v16.h }[0], [x20]
    st1 { v17.h }[0], [x24]
    st1 { v18.h }[0], [x21]
    st1 { v19.h }[0], [x26]
    st1 { v20.h }[0], [x22]
    st1 { v21.h }[0], [x25]
    st1 { v22.h }[0], [x23]
    st1 { v23.h }[0], [x27]
KAI_ASM_LABEL(label_13)  // Output block 1: Done
KAI_ASM_LABEL(label_14)  // Output stage exit
    subs x10, x10, #0x4
    add x14, x14, #0x8
    bgt label_2
    mov x20, #0x4
    sub x15, x15, #0x10
    cmp x15, #0x10
    mov x14, x9
    madd x8, x20, x5, x8
    bge label_1
KAI_ASM_LABEL(label_15)  // Row loop skip
    cbz x15, label_25
KAI_ASM_LABEL(label_16)  // Row tail: Row loop
    mov x26, x17
    mov x25, x16
    add x24, x14, x13, LSL #2
KAI_ASM_LABEL(label_17)  // Row tail: Column loop
    movi v16.4s, #0x0
    mov x27, x8
    mov x21, x7
    str q16, [SP, #0x0]
    str q16, [SP, #0x10]
    str q16, [SP, #0x20]
    str q16, [SP, #0x30]
KAI_ASM_LABEL(label_18)  // Row tail: Block loop
    movi v14.4s, #0x0
    movi v24.4s, #0x0
    mov x20, x6
    movi v13.4s, #0x0
    movi v11.4s, #0x0
KAI_ASM_LABEL(label_19)  // Row tail: Sub block loop
    ldr q2, [x26, #0x0]
    ldr q1, [x26, #0x10]
    movi v0.16b, #0xf0
    subs x20, x20, #0x1
    ldr q31, [x27, #0x0]
    ldr q30, [x27, #0x10]
    ldr q15, [x26, #0x20]
    ldr q28, [x26, #0x30]
    add x26, x26, #0x40
    ldr q27, [x27, #0x20]
    ldr q26, [x27, #0x30]
    shl v29.16b, v2.16b, #0x4
    shl v25.16b, v1.16b, #0x4
    ldr q23, [x27, #0x40]
    ldr q22, [x27, #0x50]
    and v2.16b, v2.16b, v0.16b
    and v1.16b, v1.16b, v0.16b
    ldr q21, [x27, #0x60]
    ldr q20, [x27, #0x70]
    shl v19.16b, v15.16b, #0x4
    shl v18.16b, v28.16b, #0x4
    KAI_ASM_INST(0x4e9da7ee)  // smmla v14.4s, v31.16b, v29.16b
    KAI_ASM_INST(0x4e99a7f8)  // smmla v24.4s, v31.16b, v25.16b
    and v15.16b, v15.16b, v0.16b
    add x27, x27, #0x80
    KAI_ASM_INST(0x4e9da7cd)  // smmla v13.4s, v30.16b, v29.16b
    KAI_ASM_INST(0x4e99a7cb)  // smmla v11.4s, v30.16b, v25.16b
    and v28.16b, v28.16b, v0.16b
    KAI_ASM_INST(0x4e93a76e)  // smmla v14.4s, v27.16b, v19.16b
    KAI_ASM_INST(0x4e92a778)  // smmla v24.4s, v27.16b, v18.16b
    KAI_ASM_INST(0x4e93a74d)  // smmla v13.4s, v26.16b, v19.16b
    KAI_ASM_INST(0x4e92a74b)  // smmla v11.4s, v26.16b, v18.16b
    KAI_ASM_INST(0x4e82a6ee)  // smmla v14.4s, v23.16b, v2.16b
    KAI_ASM_INST(0x4e81a6f8)  // smmla v24.4s, v23.16b, v1.16b
    KAI_ASM_INST(0x4e82a6cd)  // smmla v13.4s, v22.16b, v2.16b
    KAI_ASM_INST(0x4e81a6cb)  // smmla v11.4s, v22.16b, v1.16b
    KAI_ASM_INST(0x4e8fa6ae)  // smmla v14.4s, v21.16b, v15.16b
    KAI_ASM_INST(0x4e9ca6b8)  // smmla v24.4s, v21.16b, v28.16b
    KAI_ASM_INST(0x4e8fa68d)  // smmla v13.4s, v20.16b, v15.16b
    KAI_ASM_INST(0x4e9ca68b)  // smmla v11.4s, v20.16b, v28.16b
    bgt label_19
    ldr d26, [x26, #0x0]
    ldr q25, [SP, #0x0]
    uzp1 v23.2d, v14.2d, v24.2d
    uzp2 v22.2d, v14.2d, v24.2d
    ldr q21, [SP, #0x10]
    ldr q20, [SP, #0x20]
    uzp1 v19.2d, v13.2d, v11.2d
    uzp2 v18.2d, v13.2d, v11.2d
    ldr q17, [SP, #0x30]
    add x26, x26, #0x8
    shll v16.4s, v26.4h, #0x10
    scvtf v23.4s, v23.4s, #0x4
    scvtf v22.4s, v22.4s, #0x4
    scvtf v19.4s, v19.4s, #0x4
    scvtf v18.4s, v18.4s, #0x4
    fmla v25.4s, v23.4s, v16.4s
    fmla v21.4s, v22.4s, v16.4s
    fmla v20.4s, v19.4s, v16.4s
    fmla v17.4s, v18.4s, v16.4s
    str q25, [SP, #0x0]
    str q21, [SP, #0x10]
    str q20, [SP, #0x20]
    str q17, [SP, #0x30]
    subs x21, x21, #0x1
    bgt label_18
    ld1 { v21.4s }, [x27]
    ldr q31, [SP, #0x0]
    add x27, x27, #0x10
    add x20, x12, #0x4
    ldr q30, [SP, #0x10]
    ldr q29, [SP, #0x20]
    cmp x25, #0x4
    ldr q28, [SP, #0x30]
    ldr q20, [x26, #0x0]
    ldr q19, [x27, #0x0]
    ldr q18, [x26, #0x10]
    scvtf v21.4s, v21.4s
    add x26, x26, #0x20
    ld1r { v17.4s }, [x12]
    ld1r { v16.4s }, [x20]
    fmla v31.4s, v20.4s, v21.s[0]
    fmla v30.4s, v20.4s, v21.s[1]
    fmla v29.4s, v20.4s, v21.s[2]
    fmla v28.4s, v20.4s, v21.s[3]
    fmul v31.4s, v31.4s, v19.s[0]
    fmul v30.4s, v30.4s, v19.s[1]
    fadd v31.4s, v31.4s, v18.4s
    fmul v29.4s, v29.4s, v19.s[2]
    fmul v28.4s, v28.4s, v19.s[3]
    fadd v30.4s, v30.4s, v18.4s
    fmax v31.4s, v31.4s, v17.4s
    fadd v29.4s, v29.4s, v18.4s
    fadd v28.4s, v28.4s, v18.4s
    fmax v30.4s, v30.4s, v17.4s
    fmin v31.4s, v31.4s, v16.4s
    fmax v29.4s, v29.4s, v17.4s
    fmax v28.4s, v28.4s, v17.4s
    fmin v30.4s, v30.4s, v16.4s
    KAI_ASM_INST(0x0ea16bf3)  // bfcvtn v19.4h, v31.4s
    fmin v29.4s, v29.4s, v16.4s
    fmin v28.4s, v28.4s, v16.4s
    KAI_ASM_INST(0x0ea16bd2)  // bfcvtn v18.4h, v30.4s
    KAI_ASM_INST(0x0ea16bb1)  // bfcvtn v17.4h, v29.4s
    KAI_ASM_INST(0x0ea16b90)  // bfcvtn v16.4h, v28.4s
    blt label_21
    mov x20, x14
    cmp x15, #0x1
    str d19, [x20, #0x0]
    add x20, x20, x13
    ble label_24
    cmp x15, #0x2
    str d18, [x20, #0x0]
    add x20, x20, x13
    ble label_24
    cmp x15, #0x3
    str d17, [x20, #0x0]
    add x20, x20, x13
    ble label_24
    str d16, [x20, #0x0]
    b label_24
KAI_ASM_LABEL(label_21)  // Row tail: Partial output
    mov x23, x14
    cmp x15, #0x1
    add x22, x23, x13
    csel x22, x22, x23, GT
    cmp x15, #0x2
    add x21, x23, x13, LSL #1
    csel x21, x21, x22, GT
    cmp x15, #0x3
    add x20, x21, x13
    csel x20, x20, x21, GT
    tbz x25, #1, label_22
    st1 { v16.s }[0], [x20], #0x4
    st1 { v17.s }[0], [x21], #0x4
    st1 { v18.s }[0], [x22], #0x4
    st1 { v19.s }[0], [x23], #0x4
    tbz x25, #0, label_23
    st1 { v16.h }[2], [x20]
    st1 { v17.h }[2], [x21]
    st1 { v18.h }[2], [x22]
    st1 { v19.h }[2], [x23]
    b label_23
KAI_ASM_LABEL(label_22)  // Row tail: Output block 0: partial_1_0
    st1 { v16.h }[0], [x20]
    st1 { v17.h }[0], [x21]
    st1 { v18.h }[0], [x22]
    st1 { v19.h }[0], [x23]
KAI_ASM_LABEL(label_23)  // Row tail: Output block 0: Done
KAI_ASM_LABEL(label_24)  // Row tail: Output stage exit
    subs x25, x25, #0x4
    add x14, x14, #0x8
    bgt label_17
    subs x15, x15, #0x4
    add x8, x8, x5
    mov x14, x24
    bgt label_16
KAI_ASM_LABEL(label_25)  // Row tail: Row loop skip
    add SP, SP, #0x100
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm)

    KAI_ASM_END
