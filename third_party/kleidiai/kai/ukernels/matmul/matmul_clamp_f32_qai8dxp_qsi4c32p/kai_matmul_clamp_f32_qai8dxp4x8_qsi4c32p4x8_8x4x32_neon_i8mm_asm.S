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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm)
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
    movi v15.16b, #0xf0
    mov x21, #0x20
    ldr x20, [x0, #0x28]
    ldr x7, [x0, #0x40]
    ldr x8, [x0, #0x38]
    ldr x17, [x0, #0x8]
    ldr x16, [x0, #0x10]
    ldr x15, [x0, #0x30]
    mov x14, x20
    mul x6, x7, x6
    ldr x13, [x0, #0x0]
    ldr x12, [x0, #0x20]
    ldr x11, [x0, #0x18]
    cmp x14, #0x8
    madd x6, x8, x6, x21
    blt label_11
KAI_ASM_LABEL(label_1)  // Row loop
    mov x10, x16
    mov x9, x15
    add x28, x13, x12, LSL #3
KAI_ASM_LABEL(label_2)  // Column loop
    mov x23, x17
    movi v23.16b, #0x0
    movi v16.16b, #0x0
    mov x22, x8
    movi v13.16b, #0x0
    movi v11.16b, #0x0
    movi v14.16b, #0x0
    movi v5.16b, #0x0
    movi v10.16b, #0x0
    movi v26.16b, #0x0
    add x21, x23, x6
KAI_ASM_LABEL(label_3)  // Block loop
    movi v8.4s, #0x0
    movi v19.4s, #0x0
    mov x20, x7
    movi v6.4s, #0x0
    movi v7.4s, #0x0
    movi v2.4s, #0x0
    movi v4.4s, #0x0
    movi v3.4s, #0x0
    movi v0.4s, #0x0
KAI_ASM_LABEL(label_4)  // Sub block loop
    ldr q1, [x10, #0x0]
    ldr q21, [x10, #0x10]
    subs x20, x20, #0x1
    ldr q9, [x23, #0x0]
    ldr q28, [x23, #0x10]
    ldr q20, [x21, #0x0]
    ldr q31, [x21, #0x10]
    ldr q30, [x10, #0x20]
    ldr q24, [x10, #0x30]
    shl v27.16b, v1.16b, #0x4
    shl v17.16b, v21.16b, #0x4
    ldr q22, [x23, #0x20]
    ldr q12, [x23, #0x30]
    and v1.16b, v1.16b, v15.16b
    and v21.16b, v21.16b, v15.16b
    ldr q18, [x21, #0x20]
    ldr q25, [x21, #0x30]
    add x10, x10, #0x40
    ldr q29, [x23, #0x40]
    KAI_ASM_INST(0x4e9ba528)  // smmla v8.4s, v9.16b, v27.16b
    KAI_ASM_INST(0x4e91a533)  // smmla v19.4s, v9.16b, v17.16b
    ldr q9, [x23, #0x50]
    KAI_ASM_INST(0x4e9ba786)  // smmla v6.4s, v28.16b, v27.16b
    KAI_ASM_INST(0x4e91a787)  // smmla v7.4s, v28.16b, v17.16b
    ldr q28, [x21, #0x40]
    KAI_ASM_INST(0x4e9ba682)  // smmla v2.4s, v20.16b, v27.16b
    KAI_ASM_INST(0x4e91a684)  // smmla v4.4s, v20.16b, v17.16b
    ldr q20, [x21, #0x50]
    KAI_ASM_INST(0x4e9ba7e3)  // smmla v3.4s, v31.16b, v27.16b
    ldr q27, [x23, #0x60]
    KAI_ASM_INST(0x4e91a7e0)  // smmla v0.4s, v31.16b, v17.16b
    ldr q17, [x23, #0x70]
    shl v31.16b, v30.16b, #0x4
    and v30.16b, v30.16b, v15.16b
    add x23, x23, #0x80
    KAI_ASM_INST(0x4e9fa6c8)  // smmla v8.4s, v22.16b, v31.16b
    KAI_ASM_INST(0x4e9fa586)  // smmla v6.4s, v12.16b, v31.16b
    KAI_ASM_INST(0x4e9fa642)  // smmla v2.4s, v18.16b, v31.16b
    KAI_ASM_INST(0x4e9fa723)  // smmla v3.4s, v25.16b, v31.16b
    ldr q31, [x21, #0x60]
    KAI_ASM_INST(0x4e81a7a8)  // smmla v8.4s, v29.16b, v1.16b
    KAI_ASM_INST(0x4e81a526)  // smmla v6.4s, v9.16b, v1.16b
    KAI_ASM_INST(0x4e81a782)  // smmla v2.4s, v28.16b, v1.16b
    KAI_ASM_INST(0x4e81a683)  // smmla v3.4s, v20.16b, v1.16b
    ldr q1, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e9ea768)  // smmla v8.4s, v27.16b, v30.16b
    KAI_ASM_INST(0x4e9ea626)  // smmla v6.4s, v17.16b, v30.16b
    KAI_ASM_INST(0x4e9ea7e2)  // smmla v2.4s, v31.16b, v30.16b
    KAI_ASM_INST(0x4e9ea423)  // smmla v3.4s, v1.16b, v30.16b
    shl v30.16b, v24.16b, #0x4
    and v24.16b, v24.16b, v15.16b
    KAI_ASM_INST(0x4e9ea6d3)  // smmla v19.4s, v22.16b, v30.16b
    KAI_ASM_INST(0x4e9ea587)  // smmla v7.4s, v12.16b, v30.16b
    KAI_ASM_INST(0x4e9ea644)  // smmla v4.4s, v18.16b, v30.16b
    KAI_ASM_INST(0x4e9ea720)  // smmla v0.4s, v25.16b, v30.16b
    KAI_ASM_INST(0x4e95a7b3)  // smmla v19.4s, v29.16b, v21.16b
    KAI_ASM_INST(0x4e95a527)  // smmla v7.4s, v9.16b, v21.16b
    KAI_ASM_INST(0x4e95a784)  // smmla v4.4s, v28.16b, v21.16b
    KAI_ASM_INST(0x4e95a680)  // smmla v0.4s, v20.16b, v21.16b
    KAI_ASM_INST(0x4e98a773)  // smmla v19.4s, v27.16b, v24.16b
    KAI_ASM_INST(0x4e98a627)  // smmla v7.4s, v17.16b, v24.16b
    KAI_ASM_INST(0x4e98a7e4)  // smmla v4.4s, v31.16b, v24.16b
    KAI_ASM_INST(0x4e98a420)  // smmla v0.4s, v1.16b, v24.16b
    bgt label_4
    ldr d20, [x10, #0x0]
    uzp1 v18.2d, v8.2d, v19.2d
    uzp2 v9.2d, v8.2d, v19.2d
    add x10, x10, #0x8
    uzp1 v17.2d, v6.2d, v7.2d
    uzp2 v12.2d, v6.2d, v7.2d
    shll v20.4s, v20.4h, #0x10
    scvtf v18.4s, v18.4s, #0x4
    scvtf v9.4s, v9.4s, #0x4
    scvtf v17.4s, v17.4s, #0x4
    scvtf v12.4s, v12.4s, #0x4
    fmla v23.4s, v18.4s, v20.4s
    fmla v16.4s, v9.4s, v20.4s
    fmla v13.4s, v17.4s, v20.4s
    fmla v11.4s, v12.4s, v20.4s
    uzp1 v19.2d, v2.2d, v4.2d
    uzp2 v18.2d, v2.2d, v4.2d
    uzp1 v17.2d, v3.2d, v0.2d
    uzp2 v25.2d, v3.2d, v0.2d
    scvtf v19.4s, v19.4s, #0x4
    scvtf v18.4s, v18.4s, #0x4
    scvtf v17.4s, v17.4s, #0x4
    scvtf v25.4s, v25.4s, #0x4
    fmla v14.4s, v19.4s, v20.4s
    fmla v5.4s, v18.4s, v20.4s
    fmla v10.4s, v17.4s, v20.4s
    fmla v26.4s, v25.4s, v20.4s
    subs x22, x22, #0x1
    bgt label_3
    ld1 { v24.4s }, [x23]
    ld1 { v22.4s }, [x21]
    add x23, x23, #0x10
    add x21, x21, #0x10
    ldr q21, [x10, #0x0]
    ldr q20, [x23, #0x0]
    add x20, x11, #0x4
    cmp x9, #0x4
    ldr q19, [x21, #0x0]
    ldr q18, [x10, #0x10]
    add x10, x10, #0x20
    ld1r { v17.4s }, [x11]
    ld1r { v9.4s }, [x20]
    scvtf v24.4s, v24.4s
    scvtf v22.4s, v22.4s
    fmla v23.4s, v21.4s, v24.s[0]
    fmla v16.4s, v21.4s, v24.s[1]
    fmla v13.4s, v21.4s, v24.s[2]
    fmla v11.4s, v21.4s, v24.s[3]
    fmla v14.4s, v21.4s, v22.s[0]
    fmla v5.4s, v21.4s, v22.s[1]
    fmla v10.4s, v21.4s, v22.s[2]
    fmla v26.4s, v21.4s, v22.s[3]
    fmul v23.4s, v23.4s, v20.s[0]
    fmul v16.4s, v16.4s, v20.s[1]
    fmul v13.4s, v13.4s, v20.s[2]
    fmul v11.4s, v11.4s, v20.s[3]
    fmul v14.4s, v14.4s, v19.s[0]
    fmul v5.4s, v5.4s, v19.s[1]
    fadd v23.4s, v23.4s, v18.4s
    fmul v10.4s, v10.4s, v19.s[2]
    fmul v26.4s, v26.4s, v19.s[3]
    fadd v16.4s, v16.4s, v18.4s
    fadd v13.4s, v13.4s, v18.4s
    fadd v11.4s, v11.4s, v18.4s
    fadd v14.4s, v14.4s, v18.4s
    fadd v5.4s, v5.4s, v18.4s
    fadd v10.4s, v10.4s, v18.4s
    fadd v26.4s, v26.4s, v18.4s
    fmax v23.4s, v23.4s, v17.4s
    fmax v16.4s, v16.4s, v17.4s
    fmax v13.4s, v13.4s, v17.4s
    fmax v11.4s, v11.4s, v17.4s
    fmax v14.4s, v14.4s, v17.4s
    fmax v5.4s, v5.4s, v17.4s
    fmax v10.4s, v10.4s, v17.4s
    fmax v26.4s, v26.4s, v17.4s
    fmin v23.4s, v23.4s, v9.4s
    fmin v16.4s, v16.4s, v9.4s
    fmin v13.4s, v13.4s, v9.4s
    fmin v11.4s, v11.4s, v9.4s
    fmin v14.4s, v14.4s, v9.4s
    fmin v5.4s, v5.4s, v9.4s
    fmin v10.4s, v10.4s, v9.4s
    fmin v26.4s, v26.4s, v9.4s
    blt label_7
    mov x20, x13
    str q23, [x20, #0x0]
    add x20, x20, x12
    str q16, [x20, #0x0]
    add x20, x20, x12
    str q13, [x20, #0x0]
    add x20, x20, x12
    str q11, [x20, #0x0]
    add x20, x20, x12
    str q14, [x20, #0x0]
    add x20, x20, x12
    str q5, [x20, #0x0]
    add x20, x20, x12
    str q10, [x20, #0x0]
    add x20, x20, x12
    str q26, [x20, #0x0]
    b label_10
KAI_ASM_LABEL(label_7)  // Partial output
    mov x27, x13
    add x26, x27, x12, LSL #2
    add x25, x26, x12, LSL #1
    add x24, x26, x12
    add x23, x25, x12
    add x22, x27, x12, LSL #1
    add x21, x27, x12
    add x20, x22, x12
    tbz x9, #1, label_8
    st1 { v26.d }[0], [x23], #0x8
    st1 { v10.d }[0], [x25], #0x8
    st1 { v5.d }[0], [x24], #0x8
    st1 { v14.d }[0], [x26], #0x8
    st1 { v11.d }[0], [x20], #0x8
    st1 { v13.d }[0], [x22], #0x8
    st1 { v16.d }[0], [x21], #0x8
    st1 { v23.d }[0], [x27], #0x8
    tbz x9, #0, label_9
    st1 { v26.s }[2], [x23]
    st1 { v10.s }[2], [x25]
    st1 { v5.s }[2], [x24]
    st1 { v14.s }[2], [x26]
    st1 { v11.s }[2], [x20]
    st1 { v13.s }[2], [x22]
    st1 { v16.s }[2], [x21]
    st1 { v23.s }[2], [x27]
    b label_9
KAI_ASM_LABEL(label_8)  // Output block 0: partial_1_0
    st1 { v26.s }[0], [x23]
    st1 { v10.s }[0], [x25]
    st1 { v5.s }[0], [x24]
    st1 { v14.s }[0], [x26]
    st1 { v11.s }[0], [x20]
    st1 { v13.s }[0], [x22]
    st1 { v16.s }[0], [x21]
    st1 { v23.s }[0], [x27]
KAI_ASM_LABEL(label_9)  // Output block 0: Done
KAI_ASM_LABEL(label_10)  // Output stage exit
    subs x9, x9, #0x4
    add x13, x13, #0x10
    bgt label_2
    mov x20, #0x2
    sub x14, x14, #0x8
    cmp x14, #0x8
    mov x13, x28
    madd x17, x20, x6, x17
    bge label_1
KAI_ASM_LABEL(label_11)  // Row loop skip
    cbz x14, label_21
KAI_ASM_LABEL(label_12)  // Row tail: Row loop
    mov x26, x16
    mov x25, x15
    add x24, x13, x12, LSL #2
KAI_ASM_LABEL(label_13)  // Row tail: Column loop
    movi v23.16b, #0x0
    movi v16.16b, #0x0
    mov x23, x17
    mov x21, x8
    movi v13.16b, #0x0
    movi v11.16b, #0x0
KAI_ASM_LABEL(label_14)  // Row tail: Block loop
    movi v8.4s, #0x0
    movi v19.4s, #0x0
    mov x20, x7
    movi v6.4s, #0x0
    movi v7.4s, #0x0
KAI_ASM_LABEL(label_15)  // Row tail: Sub block loop
    ldr q0, [x26, #0x0]
    ldr q31, [x26, #0x10]
    subs x20, x20, #0x1
    ldr q30, [x23, #0x0]
    ldr q29, [x23, #0x10]
    ldr q28, [x26, #0x20]
    ldr q27, [x26, #0x30]
    add x26, x26, #0x40
    ldr q26, [x23, #0x20]
    ldr q25, [x23, #0x30]
    shl v24.16b, v0.16b, #0x4
    shl v22.16b, v31.16b, #0x4
    ldr q21, [x23, #0x40]
    ldr q20, [x23, #0x50]
    and v0.16b, v0.16b, v15.16b
    and v31.16b, v31.16b, v15.16b
    ldr q3, [x23, #0x60]
    ldr q18, [x23, #0x70]
    shl v17.16b, v28.16b, #0x4
    shl v12.16b, v27.16b, #0x4
    KAI_ASM_INST(0x4e98a7c8)  // smmla v8.4s, v30.16b, v24.16b
    KAI_ASM_INST(0x4e96a7d3)  // smmla v19.4s, v30.16b, v22.16b
    and v28.16b, v28.16b, v15.16b
    add x23, x23, #0x80
    KAI_ASM_INST(0x4e98a7a6)  // smmla v6.4s, v29.16b, v24.16b
    KAI_ASM_INST(0x4e96a7a7)  // smmla v7.4s, v29.16b, v22.16b
    and v27.16b, v27.16b, v15.16b
    KAI_ASM_INST(0x4e91a748)  // smmla v8.4s, v26.16b, v17.16b
    KAI_ASM_INST(0x4e8ca753)  // smmla v19.4s, v26.16b, v12.16b
    KAI_ASM_INST(0x4e91a726)  // smmla v6.4s, v25.16b, v17.16b
    KAI_ASM_INST(0x4e8ca727)  // smmla v7.4s, v25.16b, v12.16b
    KAI_ASM_INST(0x4e80a6a8)  // smmla v8.4s, v21.16b, v0.16b
    KAI_ASM_INST(0x4e9fa6b3)  // smmla v19.4s, v21.16b, v31.16b
    KAI_ASM_INST(0x4e80a686)  // smmla v6.4s, v20.16b, v0.16b
    KAI_ASM_INST(0x4e9fa687)  // smmla v7.4s, v20.16b, v31.16b
    KAI_ASM_INST(0x4e9ca468)  // smmla v8.4s, v3.16b, v28.16b
    KAI_ASM_INST(0x4e9ba473)  // smmla v19.4s, v3.16b, v27.16b
    KAI_ASM_INST(0x4e9ca646)  // smmla v6.4s, v18.16b, v28.16b
    KAI_ASM_INST(0x4e9ba647)  // smmla v7.4s, v18.16b, v27.16b
    bgt label_15
    ldr d12, [x26, #0x0]
    uzp1 v20.2d, v8.2d, v19.2d
    uzp2 v19.2d, v8.2d, v19.2d
    add x26, x26, #0x8
    uzp1 v18.2d, v6.2d, v7.2d
    uzp2 v17.2d, v6.2d, v7.2d
    shll v12.4s, v12.4h, #0x10
    scvtf v20.4s, v20.4s, #0x4
    scvtf v19.4s, v19.4s, #0x4
    scvtf v18.4s, v18.4s, #0x4
    scvtf v17.4s, v17.4s, #0x4
    fmla v23.4s, v20.4s, v12.4s
    fmla v16.4s, v19.4s, v12.4s
    fmla v13.4s, v18.4s, v12.4s
    fmla v11.4s, v17.4s, v12.4s
    subs x21, x21, #0x1
    bgt label_14
    ld1 { v21.4s }, [x23]
    ldr q20, [x26, #0x0]
    add x23, x23, #0x10
    add x20, x11, #0x4
    ldr q19, [x23, #0x0]
    ldr q18, [x26, #0x10]
    cmp x25, #0x4
    add x26, x26, #0x20
    ld1r { v17.4s }, [x11]
    ld1r { v29.4s }, [x20]
    scvtf v21.4s, v21.4s
    fmla v23.4s, v20.4s, v21.s[0]
    fmla v16.4s, v20.4s, v21.s[1]
    fmla v13.4s, v20.4s, v21.s[2]
    fmla v11.4s, v20.4s, v21.s[3]
    fmul v23.4s, v23.4s, v19.s[0]
    fmul v16.4s, v16.4s, v19.s[1]
    fmul v13.4s, v13.4s, v19.s[2]
    fadd v23.4s, v23.4s, v18.4s
    fmul v11.4s, v11.4s, v19.s[3]
    fadd v16.4s, v16.4s, v18.4s
    fadd v13.4s, v13.4s, v18.4s
    fadd v11.4s, v11.4s, v18.4s
    fmax v23.4s, v23.4s, v17.4s
    fmax v16.4s, v16.4s, v17.4s
    fmax v13.4s, v13.4s, v17.4s
    fmax v11.4s, v11.4s, v17.4s
    fmin v23.4s, v23.4s, v29.4s
    fmin v16.4s, v16.4s, v29.4s
    fmin v13.4s, v13.4s, v29.4s
    fmin v11.4s, v11.4s, v29.4s
    blt label_17
    mov x20, x13
    cmp x14, #0x1
    str q23, [x20, #0x0]
    add x20, x20, x12
    ble label_20
    cmp x14, #0x2
    str q16, [x20, #0x0]
    add x20, x20, x12
    ble label_20
    cmp x14, #0x3
    str q13, [x20, #0x0]
    add x20, x20, x12
    ble label_20
    str q11, [x20, #0x0]
    b label_20
KAI_ASM_LABEL(label_17)  // Row tail: Partial output
    mov x23, x13
    cmp x14, #0x1
    add x22, x23, x12
    csel x22, x22, x23, GT
    cmp x14, #0x2
    add x21, x23, x12, LSL #1
    csel x21, x21, x22, GT
    cmp x14, #0x3
    add x20, x21, x12
    csel x20, x20, x21, GT
    tbz x25, #1, label_18
    st1 { v11.d }[0], [x20], #0x8
    st1 { v13.d }[0], [x21], #0x8
    st1 { v16.d }[0], [x22], #0x8
    st1 { v23.d }[0], [x23], #0x8
    tbz x25, #0, label_19
    st1 { v11.s }[2], [x20]
    st1 { v13.s }[2], [x21]
    st1 { v16.s }[2], [x22]
    st1 { v23.s }[2], [x23]
    b label_19
KAI_ASM_LABEL(label_18)  // Row tail: Output block 0: partial_1_0
    st1 { v11.s }[0], [x20]
    st1 { v13.s }[0], [x21]
    st1 { v16.s }[0], [x22]
    st1 { v23.s }[0], [x23]
KAI_ASM_LABEL(label_19)  // Row tail: Output block 0: Done
KAI_ASM_LABEL(label_20)  // Row tail: Output stage exit
    subs x25, x25, #0x4
    add x13, x13, #0x10
    bgt label_13
    subs x14, x14, #0x4
    add x17, x17, x6
    mov x13, x24
    bgt label_12
KAI_ASM_LABEL(label_21)  // Row tail: Row loop skip
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm)

    KAI_ASM_END
