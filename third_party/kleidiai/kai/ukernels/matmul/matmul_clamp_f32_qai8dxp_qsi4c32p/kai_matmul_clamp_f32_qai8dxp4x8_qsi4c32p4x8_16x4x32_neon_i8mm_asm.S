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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm)
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
    movi v29.4s, #0x0
    mov x24, x7
    str q29, [SP, #0x0]
    str q29, [SP, #0x10]
    str q29, [SP, #0x20]
    add x23, x27, x5
    add x22, x23, x5
    str q29, [SP, #0x30]
    add x21, x22, x5
    str q29, [SP, #0x40]
    str q29, [SP, #0x50]
    str q29, [SP, #0x60]
    str q29, [SP, #0x70]
    str q29, [SP, #0x80]
    str q29, [SP, #0x90]
    str q29, [SP, #0xa0]
    str q29, [SP, #0xb0]
    str q29, [SP, #0xc0]
    str q29, [SP, #0xd0]
    str q29, [SP, #0xe0]
    str q29, [SP, #0xf0]
KAI_ASM_LABEL(label_3)  // Block loop
    movi v7.4s, #0x0
    movi v13.4s, #0x0
    mov x20, x6
    movi v29.4s, #0x0
    movi v12.4s, #0x0
    movi v28.4s, #0x0
    movi v15.4s, #0x0
    movi v2.4s, #0x0
    movi v22.4s, #0x0
    movi v30.4s, #0x0
    movi v26.4s, #0x0
    movi v6.4s, #0x0
    movi v10.4s, #0x0
    movi v9.4s, #0x0
    movi v18.4s, #0x0
    movi v0.4s, #0x0
    movi v14.4s, #0x0
KAI_ASM_LABEL(label_4)  // Sub block loop
    ldr q4, [x11, #0x0]
    ldr q3, [x11, #0x10]
    movi v31.16b, #0xf0
    subs x20, x20, #0x1
    ldr q27, [x27, #0x0]
    ldr q1, [x27, #0x10]
    ldr q19, [x23, #0x0]
    ldr q17, [x23, #0x10]
    ldr q21, [x22, #0x0]
    ldr q23, [x22, #0x10]
    shl v25.16b, v4.16b, #0x4
    shl v20.16b, v3.16b, #0x4
    ldr q5, [x21, #0x0]
    ldr q16, [x21, #0x10]
    and v4.16b, v4.16b, v31.16b
    and v3.16b, v3.16b, v31.16b
    ldr q8, [x11, #0x20]
    ldr q11, [x11, #0x30]
    add x11, x11, #0x40
    ldr q24, [x27, #0x20]
    KAI_ASM_INST(0x4e99a767)  // smmla v7.4s, v27.16b, v25.16b
    KAI_ASM_INST(0x4e94a76d)  // smmla v13.4s, v27.16b, v20.16b
    ldr q27, [x27, #0x30]
    KAI_ASM_INST(0x4e99a43d)  // smmla v29.4s, v1.16b, v25.16b
    KAI_ASM_INST(0x4e94a42c)  // smmla v12.4s, v1.16b, v20.16b
    ldr q1, [x23, #0x20]
    KAI_ASM_INST(0x4e99a67c)  // smmla v28.4s, v19.16b, v25.16b
    KAI_ASM_INST(0x4e94a66f)  // smmla v15.4s, v19.16b, v20.16b
    ldr q19, [x23, #0x30]
    KAI_ASM_INST(0x4e99a622)  // smmla v2.4s, v17.16b, v25.16b
    KAI_ASM_INST(0x4e94a636)  // smmla v22.4s, v17.16b, v20.16b
    ldr q17, [x22, #0x20]
    KAI_ASM_INST(0x4e99a6be)  // smmla v30.4s, v21.16b, v25.16b
    KAI_ASM_INST(0x4e94a6ba)  // smmla v26.4s, v21.16b, v20.16b
    ldr q21, [x22, #0x30]
    KAI_ASM_INST(0x4e99a6e6)  // smmla v6.4s, v23.16b, v25.16b
    KAI_ASM_INST(0x4e94a6ea)  // smmla v10.4s, v23.16b, v20.16b
    ldr q23, [x21, #0x20]
    KAI_ASM_INST(0x4e99a4a9)  // smmla v9.4s, v5.16b, v25.16b
    KAI_ASM_INST(0x4e94a4b2)  // smmla v18.4s, v5.16b, v20.16b
    ldr q5, [x21, #0x30]
    KAI_ASM_INST(0x4e99a600)  // smmla v0.4s, v16.16b, v25.16b
    ldr q25, [x27, #0x40]
    KAI_ASM_INST(0x4e94a60e)  // smmla v14.4s, v16.16b, v20.16b
    ldr q16, [x27, #0x50]
    shl v20.16b, v8.16b, #0x4
    and v8.16b, v8.16b, v31.16b
    KAI_ASM_INST(0x4e94a707)  // smmla v7.4s, v24.16b, v20.16b
    KAI_ASM_INST(0x4e94a77d)  // smmla v29.4s, v27.16b, v20.16b
    KAI_ASM_INST(0x4e94a43c)  // smmla v28.4s, v1.16b, v20.16b
    KAI_ASM_INST(0x4e94a662)  // smmla v2.4s, v19.16b, v20.16b
    KAI_ASM_INST(0x4e94a63e)  // smmla v30.4s, v17.16b, v20.16b
    KAI_ASM_INST(0x4e94a6a6)  // smmla v6.4s, v21.16b, v20.16b
    KAI_ASM_INST(0x4e94a6e9)  // smmla v9.4s, v23.16b, v20.16b
    KAI_ASM_INST(0x4e94a4a0)  // smmla v0.4s, v5.16b, v20.16b
    shl v20.16b, v11.16b, #0x4
    KAI_ASM_INST(0x4e84a727)  // smmla v7.4s, v25.16b, v4.16b
    KAI_ASM_INST(0x4e84a61d)  // smmla v29.4s, v16.16b, v4.16b
    and v11.16b, v11.16b, v31.16b
    ldr q31, [x23, #0x40]
    KAI_ASM_INST(0x4e94a70d)  // smmla v13.4s, v24.16b, v20.16b
    ldr q24, [x23, #0x50]
    KAI_ASM_INST(0x4e94a76c)  // smmla v12.4s, v27.16b, v20.16b
    ldr q27, [x22, #0x40]
    KAI_ASM_INST(0x4e94a42f)  // smmla v15.4s, v1.16b, v20.16b
    ldr q1, [x22, #0x50]
    KAI_ASM_INST(0x4e94a676)  // smmla v22.4s, v19.16b, v20.16b
    ldr q19, [x21, #0x40]
    KAI_ASM_INST(0x4e94a63a)  // smmla v26.4s, v17.16b, v20.16b
    ldr q17, [x21, #0x50]
    KAI_ASM_INST(0x4e94a6aa)  // smmla v10.4s, v21.16b, v20.16b
    ldr q21, [x27, #0x60]
    KAI_ASM_INST(0x4e94a6f2)  // smmla v18.4s, v23.16b, v20.16b
    ldr q23, [x27, #0x70]
    KAI_ASM_INST(0x4e94a4ae)  // smmla v14.4s, v5.16b, v20.16b
    ldr q20, [x23, #0x60]
    KAI_ASM_INST(0x4e83a72d)  // smmla v13.4s, v25.16b, v3.16b
    ldr q5, [x23, #0x70]
    ldr q25, [x22, #0x60]
    KAI_ASM_INST(0x4e83a60c)  // smmla v12.4s, v16.16b, v3.16b
    KAI_ASM_INST(0x4e84a7fc)  // smmla v28.4s, v31.16b, v4.16b
    ldr q16, [x22, #0x70]
    KAI_ASM_INST(0x4e83a7ef)  // smmla v15.4s, v31.16b, v3.16b
    ldr q31, [x21, #0x60]
    KAI_ASM_INST(0x4e84a702)  // smmla v2.4s, v24.16b, v4.16b
    KAI_ASM_INST(0x4e83a716)  // smmla v22.4s, v24.16b, v3.16b
    ldr q24, [x21, #0x70]
    KAI_ASM_INST(0x4e84a77e)  // smmla v30.4s, v27.16b, v4.16b
    add x27, x27, #0x80
    KAI_ASM_INST(0x4e83a77a)  // smmla v26.4s, v27.16b, v3.16b
    KAI_ASM_INST(0x4e84a426)  // smmla v6.4s, v1.16b, v4.16b
    add x23, x23, #0x80
    add x22, x22, #0x80
    KAI_ASM_INST(0x4e83a42a)  // smmla v10.4s, v1.16b, v3.16b
    KAI_ASM_INST(0x4e84a669)  // smmla v9.4s, v19.16b, v4.16b
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e83a672)  // smmla v18.4s, v19.16b, v3.16b
    KAI_ASM_INST(0x4e84a620)  // smmla v0.4s, v17.16b, v4.16b
    KAI_ASM_INST(0x4e83a62e)  // smmla v14.4s, v17.16b, v3.16b
    KAI_ASM_INST(0x4e88a6a7)  // smmla v7.4s, v21.16b, v8.16b
    KAI_ASM_INST(0x4e8ba6ad)  // smmla v13.4s, v21.16b, v11.16b
    KAI_ASM_INST(0x4e88a6fd)  // smmla v29.4s, v23.16b, v8.16b
    KAI_ASM_INST(0x4e8ba6ec)  // smmla v12.4s, v23.16b, v11.16b
    KAI_ASM_INST(0x4e88a69c)  // smmla v28.4s, v20.16b, v8.16b
    KAI_ASM_INST(0x4e8ba68f)  // smmla v15.4s, v20.16b, v11.16b
    KAI_ASM_INST(0x4e88a4a2)  // smmla v2.4s, v5.16b, v8.16b
    KAI_ASM_INST(0x4e8ba4b6)  // smmla v22.4s, v5.16b, v11.16b
    KAI_ASM_INST(0x4e88a73e)  // smmla v30.4s, v25.16b, v8.16b
    KAI_ASM_INST(0x4e8ba73a)  // smmla v26.4s, v25.16b, v11.16b
    KAI_ASM_INST(0x4e88a606)  // smmla v6.4s, v16.16b, v8.16b
    KAI_ASM_INST(0x4e8ba60a)  // smmla v10.4s, v16.16b, v11.16b
    KAI_ASM_INST(0x4e88a7e9)  // smmla v9.4s, v31.16b, v8.16b
    KAI_ASM_INST(0x4e8ba7f2)  // smmla v18.4s, v31.16b, v11.16b
    KAI_ASM_INST(0x4e88a700)  // smmla v0.4s, v24.16b, v8.16b
    KAI_ASM_INST(0x4e8ba70e)  // smmla v14.4s, v24.16b, v11.16b
    bgt label_4
    ldr d4, [x11, #0x0]
    ldr q23, [SP, #0x0]
    uzp1 v16.2d, v7.2d, v13.2d
    uzp2 v19.2d, v7.2d, v13.2d
    uzp1 v20.2d, v29.2d, v12.2d
    uzp2 v17.2d, v29.2d, v12.2d
    add x11, x11, #0x8
    shll v24.4s, v4.4h, #0x10
    scvtf v16.4s, v16.4s, #0x4
    scvtf v19.4s, v19.4s, #0x4
    scvtf v20.4s, v20.4s, #0x4
    scvtf v17.4s, v17.4s, #0x4
    fmla v23.4s, v16.4s, v24.4s
    str q23, [SP, #0x0]
    ldr q16, [SP, #0x10]
    fmla v16.4s, v19.4s, v24.4s
    str q16, [SP, #0x10]
    ldr q16, [SP, #0x20]
    fmla v16.4s, v20.4s, v24.4s
    str q16, [SP, #0x20]
    ldr q16, [SP, #0x30]
    fmla v16.4s, v17.4s, v24.4s
    str q16, [SP, #0x30]
    ldr q1, [SP, #0x40]
    uzp1 v16.2d, v28.2d, v15.2d
    uzp2 v19.2d, v28.2d, v15.2d
    uzp1 v5.2d, v2.2d, v22.2d
    uzp2 v17.2d, v2.2d, v22.2d
    scvtf v16.4s, v16.4s, #0x4
    scvtf v19.4s, v19.4s, #0x4
    scvtf v5.4s, v5.4s, #0x4
    scvtf v17.4s, v17.4s, #0x4
    fmla v1.4s, v16.4s, v24.4s
    str q1, [SP, #0x40]
    ldr q16, [SP, #0x50]
    fmla v16.4s, v19.4s, v24.4s
    str q16, [SP, #0x50]
    ldr q16, [SP, #0x60]
    fmla v16.4s, v5.4s, v24.4s
    str q16, [SP, #0x60]
    ldr q16, [SP, #0x70]
    fmla v16.4s, v17.4s, v24.4s
    str q16, [SP, #0x70]
    ldr q1, [SP, #0x80]
    uzp1 v16.2d, v30.2d, v26.2d
    uzp2 v19.2d, v30.2d, v26.2d
    uzp1 v30.2d, v6.2d, v10.2d
    uzp2 v17.2d, v6.2d, v10.2d
    scvtf v16.4s, v16.4s, #0x4
    scvtf v19.4s, v19.4s, #0x4
    scvtf v30.4s, v30.4s, #0x4
    scvtf v17.4s, v17.4s, #0x4
    fmla v1.4s, v16.4s, v24.4s
    str q1, [SP, #0x80]
    ldr q16, [SP, #0x90]
    fmla v16.4s, v19.4s, v24.4s
    str q16, [SP, #0x90]
    ldr q16, [SP, #0xa0]
    fmla v16.4s, v30.4s, v24.4s
    str q16, [SP, #0xa0]
    ldr q16, [SP, #0xb0]
    fmla v16.4s, v17.4s, v24.4s
    str q16, [SP, #0xb0]
    ldr q31, [SP, #0xc0]
    uzp1 v16.2d, v9.2d, v18.2d
    uzp2 v19.2d, v9.2d, v18.2d
    uzp1 v21.2d, v0.2d, v14.2d
    uzp2 v17.2d, v0.2d, v14.2d
    scvtf v16.4s, v16.4s, #0x4
    scvtf v19.4s, v19.4s, #0x4
    scvtf v21.4s, v21.4s, #0x4
    scvtf v17.4s, v17.4s, #0x4
    fmla v31.4s, v16.4s, v24.4s
    str q31, [SP, #0xc0]
    ldr q16, [SP, #0xd0]
    fmla v16.4s, v19.4s, v24.4s
    str q16, [SP, #0xd0]
    ldr q16, [SP, #0xe0]
    fmla v16.4s, v21.4s, v24.4s
    str q16, [SP, #0xe0]
    ldr q16, [SP, #0xf0]
    fmla v16.4s, v17.4s, v24.4s
    str q16, [SP, #0xf0]
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
    blt label_9
    mov x20, x14
    str q31, [x20, #0x0]
    add x20, x20, x13
    str q30, [x20, #0x0]
    add x20, x20, x13
    str q29, [x20, #0x0]
    add x20, x20, x13
    str q28, [x20, #0x0]
    add x20, x20, x13
    str q27, [x20, #0x0]
    add x20, x20, x13
    str q26, [x20, #0x0]
    add x20, x20, x13
    str q25, [x20, #0x0]
    add x20, x20, x13
    str q24, [x20, #0x0]
    add x20, x20, x13
    str q23, [x20, #0x0]
    add x20, x20, x13
    str q22, [x20, #0x0]
    add x20, x20, x13
    str q21, [x20, #0x0]
    add x20, x20, x13
    str q20, [x20, #0x0]
    add x20, x20, x13
    str q19, [x20, #0x0]
    add x20, x20, x13
    str q18, [x20, #0x0]
    add x20, x20, x13
    str q17, [x20, #0x0]
    add x20, x20, x13
    str q16, [x20, #0x0]
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
    st1 { v24.d }[0], [x23], #0x8
    st1 { v25.d }[0], [x25], #0x8
    st1 { v26.d }[0], [x24], #0x8
    st1 { v27.d }[0], [x26], #0x8
    st1 { v28.d }[0], [x20], #0x8
    st1 { v29.d }[0], [x22], #0x8
    st1 { v30.d }[0], [x21], #0x8
    st1 { v31.d }[0], [x28], #0x8
    tbz x10, #0, label_11
    st1 { v24.s }[2], [x23]
    st1 { v25.s }[2], [x25]
    st1 { v26.s }[2], [x24]
    st1 { v27.s }[2], [x26]
    st1 { v28.s }[2], [x20]
    st1 { v29.s }[2], [x22]
    st1 { v30.s }[2], [x21]
    st1 { v31.s }[2], [x28]
    b label_11
KAI_ASM_LABEL(label_10)  // Output block 0: partial_1_0
    st1 { v24.s }[0], [x23]
    st1 { v25.s }[0], [x25]
    st1 { v26.s }[0], [x24]
    st1 { v27.s }[0], [x26]
    st1 { v28.s }[0], [x20]
    st1 { v29.s }[0], [x22]
    st1 { v30.s }[0], [x21]
    st1 { v31.s }[0], [x28]
KAI_ASM_LABEL(label_11)  // Output block 0: Done
    add x26, x27, x13, LSL #2
    add x25, x27, x13, LSL #1
    add x24, x26, x13, LSL #1
    add x23, x27, x13
    add x22, x25, x13
    add x21, x26, x13
    add x20, x24, x13
    tbz x10, #1, label_12
    st1 { v16.d }[0], [x20], #0x8
    st1 { v17.d }[0], [x24], #0x8
    st1 { v18.d }[0], [x21], #0x8
    st1 { v19.d }[0], [x26], #0x8
    st1 { v20.d }[0], [x22], #0x8
    st1 { v21.d }[0], [x25], #0x8
    st1 { v22.d }[0], [x23], #0x8
    st1 { v23.d }[0], [x27], #0x8
    tbz x10, #0, label_13
    st1 { v16.s }[2], [x20]
    st1 { v17.s }[2], [x24]
    st1 { v18.s }[2], [x21]
    st1 { v19.s }[2], [x26]
    st1 { v20.s }[2], [x22]
    st1 { v21.s }[2], [x25]
    st1 { v22.s }[2], [x23]
    st1 { v23.s }[2], [x27]
    b label_13
KAI_ASM_LABEL(label_12)  // Output block 1: partial_1_0
    st1 { v16.s }[0], [x20]
    st1 { v17.s }[0], [x24]
    st1 { v18.s }[0], [x21]
    st1 { v19.s }[0], [x26]
    st1 { v20.s }[0], [x22]
    st1 { v21.s }[0], [x25]
    st1 { v22.s }[0], [x23]
    st1 { v23.s }[0], [x27]
KAI_ASM_LABEL(label_13)  // Output block 1: Done
KAI_ASM_LABEL(label_14)  // Output stage exit
    subs x10, x10, #0x4
    add x14, x14, #0x10
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
    movi v7.4s, #0x0
    movi v13.4s, #0x0
    mov x20, x6
    movi v29.4s, #0x0
    movi v12.4s, #0x0
KAI_ASM_LABEL(label_19)  // Row tail: Sub block loop
    ldr q0, [x26, #0x0]
    ldr q31, [x26, #0x10]
    movi v30.16b, #0xf0
    subs x20, x20, #0x1
    ldr q18, [x27, #0x0]
    ldr q28, [x27, #0x10]
    ldr q27, [x26, #0x20]
    ldr q26, [x26, #0x30]
    add x26, x26, #0x40
    ldr q25, [x27, #0x20]
    ldr q24, [x27, #0x30]
    shl v23.16b, v0.16b, #0x4
    shl v22.16b, v31.16b, #0x4
    ldr q21, [x27, #0x40]
    ldr q20, [x27, #0x50]
    and v0.16b, v0.16b, v30.16b
    and v31.16b, v31.16b, v30.16b
    ldr q19, [x27, #0x60]
    ldr q14, [x27, #0x70]
    shl v17.16b, v27.16b, #0x4
    shl v16.16b, v26.16b, #0x4
    KAI_ASM_INST(0x4e97a647)  // smmla v7.4s, v18.16b, v23.16b
    KAI_ASM_INST(0x4e96a64d)  // smmla v13.4s, v18.16b, v22.16b
    and v27.16b, v27.16b, v30.16b
    add x27, x27, #0x80
    KAI_ASM_INST(0x4e97a79d)  // smmla v29.4s, v28.16b, v23.16b
    KAI_ASM_INST(0x4e96a78c)  // smmla v12.4s, v28.16b, v22.16b
    and v26.16b, v26.16b, v30.16b
    KAI_ASM_INST(0x4e91a727)  // smmla v7.4s, v25.16b, v17.16b
    KAI_ASM_INST(0x4e90a72d)  // smmla v13.4s, v25.16b, v16.16b
    KAI_ASM_INST(0x4e91a71d)  // smmla v29.4s, v24.16b, v17.16b
    KAI_ASM_INST(0x4e90a70c)  // smmla v12.4s, v24.16b, v16.16b
    KAI_ASM_INST(0x4e80a6a7)  // smmla v7.4s, v21.16b, v0.16b
    KAI_ASM_INST(0x4e9fa6ad)  // smmla v13.4s, v21.16b, v31.16b
    KAI_ASM_INST(0x4e80a69d)  // smmla v29.4s, v20.16b, v0.16b
    KAI_ASM_INST(0x4e9fa68c)  // smmla v12.4s, v20.16b, v31.16b
    KAI_ASM_INST(0x4e9ba667)  // smmla v7.4s, v19.16b, v27.16b
    KAI_ASM_INST(0x4e9aa66d)  // smmla v13.4s, v19.16b, v26.16b
    KAI_ASM_INST(0x4e9ba5dd)  // smmla v29.4s, v14.16b, v27.16b
    KAI_ASM_INST(0x4e9aa5cc)  // smmla v12.4s, v14.16b, v26.16b
    bgt label_19
    ldr d17, [x26, #0x0]
    ldr q21, [SP, #0x0]
    uzp1 v16.2d, v7.2d, v13.2d
    uzp2 v20.2d, v7.2d, v13.2d
    uzp1 v19.2d, v29.2d, v12.2d
    uzp2 v18.2d, v29.2d, v12.2d
    add x26, x26, #0x8
    shll v17.4s, v17.4h, #0x10
    scvtf v16.4s, v16.4s, #0x4
    scvtf v20.4s, v20.4s, #0x4
    scvtf v19.4s, v19.4s, #0x4
    scvtf v18.4s, v18.4s, #0x4
    fmla v21.4s, v16.4s, v17.4s
    str q21, [SP, #0x0]
    ldr q16, [SP, #0x10]
    fmla v16.4s, v20.4s, v17.4s
    str q16, [SP, #0x10]
    ldr q16, [SP, #0x20]
    fmla v16.4s, v19.4s, v17.4s
    str q16, [SP, #0x20]
    ldr q16, [SP, #0x30]
    fmla v16.4s, v18.4s, v17.4s
    str q16, [SP, #0x30]
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
    fmin v29.4s, v29.4s, v16.4s
    fmin v28.4s, v28.4s, v16.4s
    blt label_21
    mov x20, x14
    cmp x15, #0x1
    str q31, [x20, #0x0]
    add x20, x20, x13
    ble label_24
    cmp x15, #0x2
    str q30, [x20, #0x0]
    add x20, x20, x13
    ble label_24
    cmp x15, #0x3
    str q29, [x20, #0x0]
    add x20, x20, x13
    ble label_24
    str q28, [x20, #0x0]
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
    st1 { v28.d }[0], [x20], #0x8
    st1 { v29.d }[0], [x21], #0x8
    st1 { v30.d }[0], [x22], #0x8
    st1 { v31.d }[0], [x23], #0x8
    tbz x25, #0, label_23
    st1 { v28.s }[2], [x20]
    st1 { v29.s }[2], [x21]
    st1 { v30.s }[2], [x22]
    st1 { v31.s }[2], [x23]
    b label_23
KAI_ASM_LABEL(label_22)  // Row tail: Output block 0: partial_1_0
    st1 { v28.s }[0], [x20]
    st1 { v29.s }[0], [x21]
    st1 { v30.s }[0], [x22]
    st1 { v31.s }[0], [x23]
KAI_ASM_LABEL(label_23)  // Row tail: Output block 0: Done
KAI_ASM_LABEL(label_24)  // Row tail: Output stage exit
    subs x25, x25, #0x4
    add x14, x14, #0x10
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm)

// Optimized kernel for bl = 32

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_opt32_neon_i8mm)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_opt32_neon_i8mm)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_opt32_neon_i8mm)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_opt32_neon_i8mm)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d10, d11, [sp, 72]
    stp d12, d13, [sp, 88]
    stp d14, d15, [sp, 104]
    stp d8, d9, [sp, 120]
    mov x6, #0x80
    mov x21, #0x20
    ldr x20, [x0, #0x28]
    ldr x7, [x0, #0x38]
    ldr x8, [x0, #0x8]
    ldr x17, [x0, #0x10]
    ldr x16, [x0, #0x30]
    ldr x15, [x0, #0x0]
    mov x14, x20
    ldr x13, [x0, #0x20]
    madd x6, x7, x6, x21
    ldr x12, [x0, #0x18]
    cmp x14, #0x10
    blt label_opt_14
KAI_ASM_LABEL(label_opt_1)  // Row loop
    mov x11, x17
    mov x10, x16
    add x9, x15, x13, LSL #4
KAI_ASM_LABEL(label_opt_2)  // Column loop
    mov x27, x8
    movi v31.16b, #0x0
    movi v30.16b, #0x0
    mov x20, x7
    movi v29.16b, #0x0
    movi v28.16b, #0x0
    movi v27.16b, #0x0
    movi v26.16b, #0x0
    add x23, x27, x6
    add x22, x23, x6
    movi v25.16b, #0x0
    movi v24.16b, #0x0
    add x21, x22, x6
    movi v23.16b, #0x0
    movi v22.16b, #0x0
    movi v21.16b, #0x0
    movi v20.16b, #0x0
    movi v19.16b, #0x0
    movi v18.16b, #0x0
    movi v17.16b, #0x0
    movi v16.16b, #0x0
KAI_ASM_LABEL(label_opt_3)  // Block loop
    ldr q11, [x11, #0x0]
    ldr q4, [x11, #0x10]
    movi v2.4s, #0x0
    movi v9.4s, #0x0
    ldr q12, [x27, #0x0]
    ldr q0, [x27, #0x10]
    movi v7.4s, #0x0
    movi v5.4s, #0x0
    ldr q15, [x11, #0x20]
    ldr q13, [x11, #0x30]
    movi v10.16b, #0xf0
    add x11, x11, #0x40
    ldr q8, [x27, #0x20]
    ldr q6, [x27, #0x30]
    shl v14.16b, v11.16b, #0x4
    shl v3.16b, v4.16b, #0x4
    ldr q1, [x27, #0x40]
    and v11.16b, v11.16b, v10.16b
    and v4.16b, v4.16b, v10.16b
    KAI_ASM_INST(0x4e8ea582)  // smmla v2.4s, v12.16b, v14.16b
    KAI_ASM_INST(0x4e83a589)  // smmla v9.4s, v12.16b, v3.16b
    shl v12.16b, v15.16b, #0x4
    KAI_ASM_INST(0x4e8ea407)  // smmla v7.4s, v0.16b, v14.16b
    KAI_ASM_INST(0x4e83a405)  // smmla v5.4s, v0.16b, v3.16b
    shl v0.16b, v13.16b, #0x4
    and v15.16b, v15.16b, v10.16b
    and v13.16b, v13.16b, v10.16b
    ldr q10, [x27, #0x50]
    KAI_ASM_INST(0x4e8ca502)  // smmla v2.4s, v8.16b, v12.16b
    KAI_ASM_INST(0x4e80a509)  // smmla v9.4s, v8.16b, v0.16b
    ldr q8, [x27, #0x60]
    KAI_ASM_INST(0x4e8ca4c7)  // smmla v7.4s, v6.16b, v12.16b
    KAI_ASM_INST(0x4e80a4c5)  // smmla v5.4s, v6.16b, v0.16b
    ldr q6, [x27, #0x70]
    add x27, x27, #0x80
    KAI_ASM_INST(0x4e8ba422)  // smmla v2.4s, v1.16b, v11.16b
    KAI_ASM_INST(0x4e84a429)  // smmla v9.4s, v1.16b, v4.16b
    ldr d1, [x11, #0x0]
    add x11, x11, #0x8
    KAI_ASM_INST(0x4e8ba547)  // smmla v7.4s, v10.16b, v11.16b
    KAI_ASM_INST(0x4e84a545)  // smmla v5.4s, v10.16b, v4.16b
    KAI_ASM_INST(0x4e8fa502)  // smmla v2.4s, v8.16b, v15.16b
    shll v1.4s, v1.4h, #0x10
    KAI_ASM_INST(0x4e8da509)  // smmla v9.4s, v8.16b, v13.16b
    KAI_ASM_INST(0x4e8fa4c7)  // smmla v7.4s, v6.16b, v15.16b
    KAI_ASM_INST(0x4e8da4c5)  // smmla v5.4s, v6.16b, v13.16b
    uzp1 v6.2d, v2.2d, v9.2d
    uzp2 v8.2d, v2.2d, v9.2d
    scvtf v6.4s, v6.4s, #0x4
    uzp1 v9.2d, v7.2d, v5.2d
    uzp2 v2.2d, v7.2d, v5.2d
    scvtf v8.4s, v8.4s, #0x4
    fmla v31.4s, v6.4s, v1.4s
    scvtf v9.4s, v9.4s, #0x4
    scvtf v2.4s, v2.4s, #0x4
    fmla v30.4s, v8.4s, v1.4s
    fmla v29.4s, v9.4s, v1.4s
    fmla v28.4s, v2.4s, v1.4s
    ldr q9, [x23, #0x0]
    ldr q7, [x23, #0x10]
    movi v8.4s, #0x0
    movi v2.4s, #0x0
    ldr q5, [x23, #0x20]
    ldr q10, [x23, #0x30]
    movi v6.4s, #0x0
    KAI_ASM_INST(0x4e8ea528)  // smmla v8.4s, v9.16b, v14.16b
    KAI_ASM_INST(0x4e83a522)  // smmla v2.4s, v9.16b, v3.16b
    ldr q9, [x23, #0x40]
    KAI_ASM_INST(0x4e8ea4e6)  // smmla v6.4s, v7.16b, v14.16b
    KAI_ASM_INST(0x4e8ca4a8)  // smmla v8.4s, v5.16b, v12.16b
    KAI_ASM_INST(0x4e80a4a2)  // smmla v2.4s, v5.16b, v0.16b
    ldr q5, [x23, #0x50]
    KAI_ASM_INST(0x4e8ca546)  // smmla v6.4s, v10.16b, v12.16b
    KAI_ASM_INST(0x4e8ba528)  // smmla v8.4s, v9.16b, v11.16b
    KAI_ASM_INST(0x4e84a522)  // smmla v2.4s, v9.16b, v4.16b
    ldr q9, [x23, #0x60]
    KAI_ASM_INST(0x4e8ba4a6)  // smmla v6.4s, v5.16b, v11.16b
    KAI_ASM_INST(0x4e8fa528)  // smmla v8.4s, v9.16b, v15.16b
    KAI_ASM_INST(0x4e8da522)  // smmla v2.4s, v9.16b, v13.16b
    movi v9.4s, #0x0
    KAI_ASM_INST(0x4e83a4e9)  // smmla v9.4s, v7.16b, v3.16b
    ldr q7, [x23, #0x70]
    add x23, x23, #0x80
    KAI_ASM_INST(0x4e8fa4e6)  // smmla v6.4s, v7.16b, v15.16b
    KAI_ASM_INST(0x4e80a549)  // smmla v9.4s, v10.16b, v0.16b
    uzp1 v10.2d, v8.2d, v2.2d
    uzp2 v2.2d, v8.2d, v2.2d
    scvtf v10.4s, v10.4s, #0x4
    KAI_ASM_INST(0x4e84a4a9)  // smmla v9.4s, v5.16b, v4.16b
    scvtf v2.4s, v2.4s, #0x4
    fmla v27.4s, v10.4s, v1.4s
    KAI_ASM_INST(0x4e8da4e9)  // smmla v9.4s, v7.16b, v13.16b
    fmla v26.4s, v2.4s, v1.4s
    uzp1 v2.2d, v6.2d, v9.2d
    uzp2 v10.2d, v6.2d, v9.2d
    scvtf v2.4s, v2.4s, #0x4
    scvtf v10.4s, v10.4s, #0x4
    fmla v25.4s, v2.4s, v1.4s
    fmla v24.4s, v10.4s, v1.4s
    ldr q8, [x22, #0x0]
    ldr q7, [x22, #0x10]
    movi v9.4s, #0x0
    movi v6.4s, #0x0
    ldr q2, [x22, #0x20]
    ldr q5, [x22, #0x30]
    movi v10.4s, #0x0
    KAI_ASM_INST(0x4e8ea509)  // smmla v9.4s, v8.16b, v14.16b
    KAI_ASM_INST(0x4e83a506)  // smmla v6.4s, v8.16b, v3.16b
    ldr q8, [x22, #0x40]
    KAI_ASM_INST(0x4e8ea4ea)  // smmla v10.4s, v7.16b, v14.16b
    KAI_ASM_INST(0x4e8ca449)  // smmla v9.4s, v2.16b, v12.16b
    KAI_ASM_INST(0x4e80a446)  // smmla v6.4s, v2.16b, v0.16b
    ldr q2, [x22, #0x50]
    KAI_ASM_INST(0x4e8ca4aa)  // smmla v10.4s, v5.16b, v12.16b
    KAI_ASM_INST(0x4e8ba509)  // smmla v9.4s, v8.16b, v11.16b
    KAI_ASM_INST(0x4e84a506)  // smmla v6.4s, v8.16b, v4.16b
    ldr q8, [x22, #0x60]
    KAI_ASM_INST(0x4e8ba44a)  // smmla v10.4s, v2.16b, v11.16b
    KAI_ASM_INST(0x4e8fa509)  // smmla v9.4s, v8.16b, v15.16b
    KAI_ASM_INST(0x4e8da506)  // smmla v6.4s, v8.16b, v13.16b
    movi v8.4s, #0x0
    KAI_ASM_INST(0x4e83a4e8)  // smmla v8.4s, v7.16b, v3.16b
    ldr q7, [x22, #0x70]
    add x22, x22, #0x80
    KAI_ASM_INST(0x4e8fa4ea)  // smmla v10.4s, v7.16b, v15.16b
    KAI_ASM_INST(0x4e80a4a8)  // smmla v8.4s, v5.16b, v0.16b
    uzp1 v5.2d, v9.2d, v6.2d
    uzp2 v9.2d, v9.2d, v6.2d
    scvtf v5.4s, v5.4s, #0x4
    KAI_ASM_INST(0x4e84a448)  // smmla v8.4s, v2.16b, v4.16b
    scvtf v9.4s, v9.4s, #0x4
    fmla v23.4s, v5.4s, v1.4s
    KAI_ASM_INST(0x4e8da4e8)  // smmla v8.4s, v7.16b, v13.16b
    fmla v22.4s, v9.4s, v1.4s
    uzp1 v2.2d, v10.2d, v8.2d
    uzp2 v10.2d, v10.2d, v8.2d
    scvtf v2.4s, v2.4s, #0x4
    scvtf v10.4s, v10.4s, #0x4
    fmla v21.4s, v2.4s, v1.4s
    fmla v20.4s, v10.4s, v1.4s
    ldr q2, [x21, #0x0]
    ldr q10, [x21, #0x10]
    movi v6.4s, #0x0
    movi v9.4s, #0x0
    ldr q5, [x21, #0x20]
    ldr q8, [x21, #0x30]
    movi v7.4s, #0x0
    KAI_ASM_INST(0x4e8ea446)  // smmla v6.4s, v2.16b, v14.16b
    KAI_ASM_INST(0x4e83a449)  // smmla v9.4s, v2.16b, v3.16b
    ldr q2, [x21, #0x40]
    KAI_ASM_INST(0x4e8ea547)  // smmla v7.4s, v10.16b, v14.16b
    ldr q14, [x21, #0x50]
    KAI_ASM_INST(0x4e8ca4a6)  // smmla v6.4s, v5.16b, v12.16b
    KAI_ASM_INST(0x4e80a4a9)  // smmla v9.4s, v5.16b, v0.16b
    ldr q5, [x21, #0x60]
    KAI_ASM_INST(0x4e8ca507)  // smmla v7.4s, v8.16b, v12.16b
    ldr q12, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e8ba446)  // smmla v6.4s, v2.16b, v11.16b
    KAI_ASM_INST(0x4e84a449)  // smmla v9.4s, v2.16b, v4.16b
    movi v2.4s, #0x0
    KAI_ASM_INST(0x4e83a542)  // smmla v2.4s, v10.16b, v3.16b
    KAI_ASM_INST(0x4e8ba5c7)  // smmla v7.4s, v14.16b, v11.16b
    KAI_ASM_INST(0x4e8fa4a6)  // smmla v6.4s, v5.16b, v15.16b
    KAI_ASM_INST(0x4e80a502)  // smmla v2.4s, v8.16b, v0.16b
    KAI_ASM_INST(0x4e8da4a9)  // smmla v9.4s, v5.16b, v13.16b
    KAI_ASM_INST(0x4e8fa587)  // smmla v7.4s, v12.16b, v15.16b
    KAI_ASM_INST(0x4e84a5c2)  // smmla v2.4s, v14.16b, v4.16b
    uzp1 v11.2d, v6.2d, v9.2d
    uzp2 v14.2d, v6.2d, v9.2d
    scvtf v11.4s, v11.4s, #0x4
    KAI_ASM_INST(0x4e8da582)  // smmla v2.4s, v12.16b, v13.16b
    scvtf v14.4s, v14.4s, #0x4
    fmla v19.4s, v11.4s, v1.4s
    uzp1 v9.2d, v7.2d, v2.2d
    uzp2 v0.2d, v7.2d, v2.2d
    fmla v18.4s, v14.4s, v1.4s
    scvtf v9.4s, v9.4s, #0x4
    scvtf v0.4s, v0.4s, #0x4
    fmla v17.4s, v9.4s, v1.4s
    fmla v16.4s, v0.4s, v1.4s
    subs x20, x20, #0x1
    bgt label_opt_3
    ld1 { v11.4s }, [x27]
    ld1 { v10.4s }, [x23]
    add x27, x27, #0x10
    add x23, x23, #0x10
    ld1 { v9.4s }, [x22]
    ld1 { v8.4s }, [x21]
    add x22, x22, #0x10
    add x21, x21, #0x10
    ldr q7, [x11, #0x0]
    ldr q6, [x27, #0x0]
    add x20, x12, #0x4
    cmp x10, #0x4
    ldr q5, [x23, #0x0]
    ldr q4, [x22, #0x0]
    scvtf v11.4s, v11.4s
    scvtf v10.4s, v10.4s
    ldr q3, [x21, #0x0]
    ldr q2, [x11, #0x10]
    scvtf v9.4s, v9.4s
    scvtf v8.4s, v8.4s
    ld1r { v1.4s }, [x12]
    ld1r { v0.4s }, [x20]
    add x11, x11, #0x20
    fmla v31.4s, v7.4s, v11.s[0]
    fmla v30.4s, v7.4s, v11.s[1]
    fmla v29.4s, v7.4s, v11.s[2]
    fmla v28.4s, v7.4s, v11.s[3]
    fmla v27.4s, v7.4s, v10.s[0]
    fmla v26.4s, v7.4s, v10.s[1]
    fmla v25.4s, v7.4s, v10.s[2]
    fmla v24.4s, v7.4s, v10.s[3]
    fmla v23.4s, v7.4s, v9.s[0]
    fmul v31.4s, v31.4s, v6.s[0]
    fmla v22.4s, v7.4s, v9.s[1]
    fmla v21.4s, v7.4s, v9.s[2]
    fmul v30.4s, v30.4s, v6.s[1]
    fmla v20.4s, v7.4s, v9.s[3]
    fmla v19.4s, v7.4s, v8.s[0]
    fmul v29.4s, v29.4s, v6.s[2]
    fmla v18.4s, v7.4s, v8.s[1]
    fmla v17.4s, v7.4s, v8.s[2]
    fmul v28.4s, v28.4s, v6.s[3]
    fmla v16.4s, v7.4s, v8.s[3]
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
    blt label_opt_8
    mov x20, x15
    str q31, [x20, #0x0]
    add x20, x20, x13
    str q30, [x20, #0x0]
    add x20, x20, x13
    str q29, [x20, #0x0]
    add x20, x20, x13
    str q28, [x20, #0x0]
    add x20, x20, x13
    str q27, [x20, #0x0]
    add x20, x20, x13
    str q26, [x20, #0x0]
    add x20, x20, x13
    str q25, [x20, #0x0]
    add x20, x20, x13
    str q24, [x20, #0x0]
    add x20, x20, x13
    str q23, [x20, #0x0]
    add x20, x20, x13
    str q22, [x20, #0x0]
    add x20, x20, x13
    str q21, [x20, #0x0]
    add x20, x20, x13
    str q20, [x20, #0x0]
    add x20, x20, x13
    str q19, [x20, #0x0]
    add x20, x20, x13
    str q18, [x20, #0x0]
    add x20, x20, x13
    str q17, [x20, #0x0]
    add x20, x20, x13
    str q16, [x20, #0x0]
    b label_opt_13
KAI_ASM_LABEL(label_opt_8)  // Partial output
    mov x28, x15
    add x26, x28, x13, LSL #2
    add x25, x26, x13, LSL #1
    add x24, x26, x13
    add x23, x25, x13
    add x22, x28, x13, LSL #1
    add x21, x28, x13
    add x20, x22, x13
    add x27, x23, x13
    tbz x10, #1, label_opt_9
    st1 { v24.d }[0], [x23], #0x8
    st1 { v25.d }[0], [x25], #0x8
    st1 { v26.d }[0], [x24], #0x8
    st1 { v27.d }[0], [x26], #0x8
    st1 { v28.d }[0], [x20], #0x8
    st1 { v29.d }[0], [x22], #0x8
    st1 { v30.d }[0], [x21], #0x8
    st1 { v31.d }[0], [x28], #0x8
    tbz x10, #0, label_opt_10
    st1 { v24.s }[2], [x23]
    st1 { v25.s }[2], [x25]
    st1 { v26.s }[2], [x24]
    st1 { v27.s }[2], [x26]
    st1 { v28.s }[2], [x20]
    st1 { v29.s }[2], [x22]
    st1 { v30.s }[2], [x21]
    st1 { v31.s }[2], [x28]
    b label_opt_10
KAI_ASM_LABEL(label_opt_9)  // Output block 0: partial_1_0
    st1 { v24.s }[0], [x23]
    st1 { v25.s }[0], [x25]
    st1 { v26.s }[0], [x24]
    st1 { v27.s }[0], [x26]
    st1 { v28.s }[0], [x20]
    st1 { v29.s }[0], [x22]
    st1 { v30.s }[0], [x21]
    st1 { v31.s }[0], [x28]
KAI_ASM_LABEL(label_opt_10)  // Output block 0: Done
    add x26, x27, x13, LSL #2
    add x25, x27, x13, LSL #1
    add x24, x26, x13, LSL #1
    add x23, x27, x13
    add x22, x25, x13
    add x21, x26, x13
    add x20, x24, x13
    tbz x10, #1, label_opt_11
    st1 { v16.d }[0], [x20], #0x8
    st1 { v17.d }[0], [x24], #0x8
    st1 { v18.d }[0], [x21], #0x8
    st1 { v19.d }[0], [x26], #0x8
    st1 { v20.d }[0], [x22], #0x8
    st1 { v21.d }[0], [x25], #0x8
    st1 { v22.d }[0], [x23], #0x8
    st1 { v23.d }[0], [x27], #0x8
    tbz x10, #0, label_opt_12
    st1 { v16.s }[2], [x20]
    st1 { v17.s }[2], [x24]
    st1 { v18.s }[2], [x21]
    st1 { v19.s }[2], [x26]
    st1 { v20.s }[2], [x22]
    st1 { v21.s }[2], [x25]
    st1 { v22.s }[2], [x23]
    st1 { v23.s }[2], [x27]
    b label_opt_12
KAI_ASM_LABEL(label_opt_11)  // Output block 1: partial_1_0
    st1 { v16.s }[0], [x20]
    st1 { v17.s }[0], [x24]
    st1 { v18.s }[0], [x21]
    st1 { v19.s }[0], [x26]
    st1 { v20.s }[0], [x22]
    st1 { v21.s }[0], [x25]
    st1 { v22.s }[0], [x23]
    st1 { v23.s }[0], [x27]
KAI_ASM_LABEL(label_opt_12)  // Output block 1: Done
KAI_ASM_LABEL(label_opt_13)  // Output stage exit
    subs x10, x10, #0x4
    add x15, x15, #0x10
    bgt label_opt_2
    mov x20, #0x4
    sub x14, x14, #0x10
    cmp x14, #0x10
    mov x15, x9
    madd x8, x20, x6, x8
    bge label_opt_1
KAI_ASM_LABEL(label_opt_14)  // Row loop skip
    cbz x14, label_opt_23
KAI_ASM_LABEL(label_opt_15)  // Row tail: Row loop
    mov x26, x17
    mov x25, x16
    add x24, x15, x13, LSL #2
KAI_ASM_LABEL(label_opt_16)  // Row tail: Column loop
    movi v31.16b, #0x0
    movi v30.16b, #0x0
    mov x27, x8
    mov x20, x7
    movi v29.16b, #0x0
    movi v28.16b, #0x0
KAI_ASM_LABEL(label_opt_17)  // Row tail: Block loop
    ldr q9, [x26, #0x0]
    ldr q8, [x26, #0x10]
    movi v7.4s, #0x0
    movi v6.4s, #0x0
    ldr q5, [x27, #0x0]
    ldr q4, [x27, #0x10]
    movi v3.4s, #0x0
    movi v2.4s, #0x0
    ldr q1, [x26, #0x20]
    ldr q0, [x26, #0x30]
    movi v27.16b, #0xf0
    add x26, x26, #0x40
    ldr q26, [x27, #0x20]
    ldr q25, [x27, #0x30]
    shl v24.16b, v9.16b, #0x4
    shl v20.16b, v8.16b, #0x4
    ldr q23, [x27, #0x40]
    ldr q22, [x27, #0x50]
    and v9.16b, v9.16b, v27.16b
    and v8.16b, v8.16b, v27.16b
    ldr q21, [x27, #0x60]
    ldr q19, [x27, #0x70]
    shl v18.16b, v1.16b, #0x4
    shl v17.16b, v0.16b, #0x4
    ldr d16, [x26, #0x0]
    KAI_ASM_INST(0x4e98a4a7)  // smmla v7.4s, v5.16b, v24.16b
    KAI_ASM_INST(0x4e94a4a6)  // smmla v6.4s, v5.16b, v20.16b
    and v1.16b, v1.16b, v27.16b
    KAI_ASM_INST(0x4e98a483)  // smmla v3.4s, v4.16b, v24.16b
    KAI_ASM_INST(0x4e94a482)  // smmla v2.4s, v4.16b, v20.16b
    and v0.16b, v0.16b, v27.16b
    add x26, x26, #0x8
    add x27, x27, #0x80
    shll v20.4s, v16.4h, #0x10
    KAI_ASM_INST(0x4e92a747)  // smmla v7.4s, v26.16b, v18.16b
    KAI_ASM_INST(0x4e91a746)  // smmla v6.4s, v26.16b, v17.16b
    KAI_ASM_INST(0x4e92a723)  // smmla v3.4s, v25.16b, v18.16b
    KAI_ASM_INST(0x4e91a722)  // smmla v2.4s, v25.16b, v17.16b
    KAI_ASM_INST(0x4e89a6e7)  // smmla v7.4s, v23.16b, v9.16b
    KAI_ASM_INST(0x4e88a6e6)  // smmla v6.4s, v23.16b, v8.16b
    KAI_ASM_INST(0x4e89a6c3)  // smmla v3.4s, v22.16b, v9.16b
    KAI_ASM_INST(0x4e88a6c2)  // smmla v2.4s, v22.16b, v8.16b
    KAI_ASM_INST(0x4e81a6a7)  // smmla v7.4s, v21.16b, v1.16b
    KAI_ASM_INST(0x4e80a6a6)  // smmla v6.4s, v21.16b, v0.16b
    KAI_ASM_INST(0x4e81a663)  // smmla v3.4s, v19.16b, v1.16b
    KAI_ASM_INST(0x4e80a662)  // smmla v2.4s, v19.16b, v0.16b
    uzp1 v19.2d, v7.2d, v6.2d
    uzp2 v18.2d, v7.2d, v6.2d
    scvtf v19.4s, v19.4s, #0x4
    uzp1 v17.2d, v3.2d, v2.2d
    uzp2 v16.2d, v3.2d, v2.2d
    scvtf v18.4s, v18.4s, #0x4
    fmla v31.4s, v19.4s, v20.4s
    scvtf v17.4s, v17.4s, #0x4
    scvtf v16.4s, v16.4s, #0x4
    fmla v30.4s, v18.4s, v20.4s
    fmla v29.4s, v17.4s, v20.4s
    fmla v28.4s, v16.4s, v20.4s
    subs x20, x20, #0x1
    bgt label_opt_17
    ld1 { v21.4s }, [x27]
    ldr q20, [x26, #0x0]
    add x27, x27, #0x10
    add x20, x12, #0x4
    ldr q19, [x27, #0x0]
    ldr q18, [x26, #0x10]
    cmp x25, #0x4
    add x26, x26, #0x20
    ld1r { v17.4s }, [x12]
    ld1r { v16.4s }, [x20]
    scvtf v21.4s, v21.4s
    fmla v31.4s, v20.4s, v21.s[0]
    fmla v30.4s, v20.4s, v21.s[1]
    fmla v29.4s, v20.4s, v21.s[2]
    fmla v28.4s, v20.4s, v21.s[3]
    fmul v31.4s, v31.4s, v19.s[0]
    fmul v30.4s, v30.4s, v19.s[1]
    fmul v29.4s, v29.4s, v19.s[2]
    fadd v31.4s, v31.4s, v18.4s
    fmul v28.4s, v28.4s, v19.s[3]
    fadd v30.4s, v30.4s, v18.4s
    fadd v29.4s, v29.4s, v18.4s
    fadd v28.4s, v28.4s, v18.4s
    fmax v31.4s, v31.4s, v17.4s
    fmax v30.4s, v30.4s, v17.4s
    fmax v29.4s, v29.4s, v17.4s
    fmax v28.4s, v28.4s, v17.4s
    fmin v31.4s, v31.4s, v16.4s
    fmin v30.4s, v30.4s, v16.4s
    fmin v29.4s, v29.4s, v16.4s
    fmin v28.4s, v28.4s, v16.4s
    blt label_opt_19
    mov x20, x15
    cmp x14, #0x1
    str q31, [x20, #0x0]
    add x20, x20, x13
    ble label_opt_22
    cmp x14, #0x2
    str q30, [x20, #0x0]
    add x20, x20, x13
    ble label_opt_22
    cmp x14, #0x3
    str q29, [x20, #0x0]
    add x20, x20, x13
    ble label_opt_22
    str q28, [x20, #0x0]
    b label_opt_22
KAI_ASM_LABEL(label_opt_19)  // Row tail: Partial output
    mov x23, x15
    cmp x14, #0x1
    add x22, x23, x13
    csel x22, x22, x23, GT
    cmp x14, #0x2
    add x21, x23, x13, LSL #1
    csel x21, x21, x22, GT
    cmp x14, #0x3
    add x20, x21, x13
    csel x20, x20, x21, GT
    tbz x25, #1, label_opt_20
    st1 { v28.d }[0], [x20], #0x8
    st1 { v29.d }[0], [x21], #0x8
    st1 { v30.d }[0], [x22], #0x8
    st1 { v31.d }[0], [x23], #0x8
    tbz x25, #0, label_opt_21
    st1 { v28.s }[2], [x20]
    st1 { v29.s }[2], [x21]
    st1 { v30.s }[2], [x22]
    st1 { v31.s }[2], [x23]
    b label_opt_21
KAI_ASM_LABEL(label_opt_20)  // Row tail: Output block 0: partial_1_0
    st1 { v28.s }[0], [x20]
    st1 { v29.s }[0], [x21]
    st1 { v30.s }[0], [x22]
    st1 { v31.s }[0], [x23]
KAI_ASM_LABEL(label_opt_21)  // Row tail: Output block 0: Done
KAI_ASM_LABEL(label_opt_22)  // Row tail: Output stage exit
    subs x25, x25, #0x4
    add x15, x15, #0x10
    bgt label_opt_16
    subs x14, x14, #0x4
    add x8, x8, x6
    mov x15, x24
    bgt label_opt_15
KAI_ASM_LABEL(label_opt_23)  // Row tail: Row loop skip
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_opt32_neon_i8mm)

    KAI_ASM_END
