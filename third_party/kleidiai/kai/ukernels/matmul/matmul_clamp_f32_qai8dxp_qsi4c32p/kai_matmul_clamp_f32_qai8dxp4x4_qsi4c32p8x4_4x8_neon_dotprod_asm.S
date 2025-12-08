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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod)
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
    movi v11.16b, #0xf0
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
    movi v0.16b, #0x0
    movi v12.16b, #0x0
    mov x22, x13
    mov x21, x14
    movi v14.16b, #0x0
    movi v13.16b, #0x0
    movi v15.16b, #0x0
    movi v7.16b, #0x0
    movi v8.16b, #0x0
    movi v9.16b, #0x0
KAI_ASM_LABEL(label_3)  // Block loop
    movi v5.4s, #0x0
    movi v6.4s, #0x0
    mov x20, x15
    movi v4.4s, #0x0
    movi v1.4s, #0x0
    movi v3.4s, #0x0
    movi v21.4s, #0x0
    movi v10.4s, #0x0
    movi v2.4s, #0x0
KAI_ASM_LABEL(label_4)  // Sub block loop
    ldr q28, [x26, #0x0]
    ldr q26, [x26, #0x10]
    subs x20, x20, #0x1
    ldr q19, [x22, #0x0]
    ldr q23, [x26, #0x20]
    ldr q31, [x26, #0x30]
    ldr q18, [x22, #0x10]
    ldr q27, [x26, #0x40]
    ldr q25, [x26, #0x50]
    shl v30.16b, v28.16b, #0x4
    shl v20.16b, v26.16b, #0x4
    ldr q17, [x22, #0x20]
    ldr q24, [x26, #0x60]
    shl v16.16b, v23.16b, #0x4
    and v28.16b, v28.16b, v11.16b
    ldr q29, [x26, #0x70]
    ldr q22, [x22, #0x30]
    and v26.16b, v26.16b, v11.16b
    and v23.16b, v23.16b, v11.16b
    KAI_ASM_INST(0x4f93e3c5)  // sdot v5.4s, v30.16b, v19.4b[0]
    KAI_ASM_INST(0x4f93e286)  // sdot v6.4s, v20.16b, v19.4b[0]
    add x26, x26, #0x80
    KAI_ASM_INST(0x4fb3e3c4)  // sdot v4.4s, v30.16b, v19.4b[1]
    KAI_ASM_INST(0x4fb3e281)  // sdot v1.4s, v20.16b, v19.4b[1]
    KAI_ASM_INST(0x4f93ebc3)  // sdot v3.4s, v30.16b, v19.4b[2]
    KAI_ASM_INST(0x4f93ea95)  // sdot v21.4s, v20.16b, v19.4b[2]
    KAI_ASM_INST(0x4fb3ebca)  // sdot v10.4s, v30.16b, v19.4b[3]
    ldr q30, [x22, #0x40]
    KAI_ASM_INST(0x4fb3ea82)  // sdot v2.4s, v20.16b, v19.4b[3]
    ldr q20, [x22, #0x50]
    ldr q19, [x22, #0x60]
    KAI_ASM_INST(0x4f92e205)  // sdot v5.4s, v16.16b, v18.4b[0]
    KAI_ASM_INST(0x4fb2e204)  // sdot v4.4s, v16.16b, v18.4b[1]
    KAI_ASM_INST(0x4f92ea03)  // sdot v3.4s, v16.16b, v18.4b[2]
    KAI_ASM_INST(0x4fb2ea0a)  // sdot v10.4s, v16.16b, v18.4b[3]
    shl v16.16b, v31.16b, #0x4
    and v31.16b, v31.16b, v11.16b
    KAI_ASM_INST(0x4f92e206)  // sdot v6.4s, v16.16b, v18.4b[0]
    KAI_ASM_INST(0x4fb2e201)  // sdot v1.4s, v16.16b, v18.4b[1]
    KAI_ASM_INST(0x4f92ea15)  // sdot v21.4s, v16.16b, v18.4b[2]
    KAI_ASM_INST(0x4fb2ea02)  // sdot v2.4s, v16.16b, v18.4b[3]
    ldr q18, [x22, #0x70]
    shl v16.16b, v27.16b, #0x4
    and v27.16b, v27.16b, v11.16b
    add x22, x22, #0x80
    KAI_ASM_INST(0x4f91e205)  // sdot v5.4s, v16.16b, v17.4b[0]
    KAI_ASM_INST(0x4fb1e204)  // sdot v4.4s, v16.16b, v17.4b[1]
    KAI_ASM_INST(0x4f91ea03)  // sdot v3.4s, v16.16b, v17.4b[2]
    KAI_ASM_INST(0x4fb1ea0a)  // sdot v10.4s, v16.16b, v17.4b[3]
    shl v16.16b, v25.16b, #0x4
    and v25.16b, v25.16b, v11.16b
    KAI_ASM_INST(0x4f91e206)  // sdot v6.4s, v16.16b, v17.4b[0]
    KAI_ASM_INST(0x4fb1e201)  // sdot v1.4s, v16.16b, v17.4b[1]
    KAI_ASM_INST(0x4f91ea15)  // sdot v21.4s, v16.16b, v17.4b[2]
    KAI_ASM_INST(0x4fb1ea02)  // sdot v2.4s, v16.16b, v17.4b[3]
    shl v17.16b, v24.16b, #0x4
    shl v16.16b, v29.16b, #0x4
    and v24.16b, v24.16b, v11.16b
    and v29.16b, v29.16b, v11.16b
    KAI_ASM_INST(0x4f96e225)  // sdot v5.4s, v17.16b, v22.4b[0]
    KAI_ASM_INST(0x4fb6e224)  // sdot v4.4s, v17.16b, v22.4b[1]
    KAI_ASM_INST(0x4f96ea23)  // sdot v3.4s, v17.16b, v22.4b[2]
    KAI_ASM_INST(0x4fb6ea2a)  // sdot v10.4s, v17.16b, v22.4b[3]
    KAI_ASM_INST(0x4f96e206)  // sdot v6.4s, v16.16b, v22.4b[0]
    KAI_ASM_INST(0x4fb6e201)  // sdot v1.4s, v16.16b, v22.4b[1]
    KAI_ASM_INST(0x4f96ea15)  // sdot v21.4s, v16.16b, v22.4b[2]
    KAI_ASM_INST(0x4fb6ea02)  // sdot v2.4s, v16.16b, v22.4b[3]
    KAI_ASM_INST(0x4f9ee385)  // sdot v5.4s, v28.16b, v30.4b[0]
    KAI_ASM_INST(0x4fbee384)  // sdot v4.4s, v28.16b, v30.4b[1]
    KAI_ASM_INST(0x4f9eeb83)  // sdot v3.4s, v28.16b, v30.4b[2]
    KAI_ASM_INST(0x4fbeeb8a)  // sdot v10.4s, v28.16b, v30.4b[3]
    KAI_ASM_INST(0x4f9ee346)  // sdot v6.4s, v26.16b, v30.4b[0]
    KAI_ASM_INST(0x4fbee341)  // sdot v1.4s, v26.16b, v30.4b[1]
    KAI_ASM_INST(0x4f9eeb55)  // sdot v21.4s, v26.16b, v30.4b[2]
    KAI_ASM_INST(0x4fbeeb42)  // sdot v2.4s, v26.16b, v30.4b[3]
    KAI_ASM_INST(0x4f94e2e5)  // sdot v5.4s, v23.16b, v20.4b[0]
    KAI_ASM_INST(0x4fb4e2e4)  // sdot v4.4s, v23.16b, v20.4b[1]
    KAI_ASM_INST(0x4f94eae3)  // sdot v3.4s, v23.16b, v20.4b[2]
    KAI_ASM_INST(0x4fb4eaea)  // sdot v10.4s, v23.16b, v20.4b[3]
    KAI_ASM_INST(0x4f94e3e6)  // sdot v6.4s, v31.16b, v20.4b[0]
    KAI_ASM_INST(0x4fb4e3e1)  // sdot v1.4s, v31.16b, v20.4b[1]
    KAI_ASM_INST(0x4f94ebf5)  // sdot v21.4s, v31.16b, v20.4b[2]
    KAI_ASM_INST(0x4fb4ebe2)  // sdot v2.4s, v31.16b, v20.4b[3]
    KAI_ASM_INST(0x4f93e365)  // sdot v5.4s, v27.16b, v19.4b[0]
    KAI_ASM_INST(0x4fb3e364)  // sdot v4.4s, v27.16b, v19.4b[1]
    KAI_ASM_INST(0x4f93eb63)  // sdot v3.4s, v27.16b, v19.4b[2]
    KAI_ASM_INST(0x4fb3eb6a)  // sdot v10.4s, v27.16b, v19.4b[3]
    KAI_ASM_INST(0x4f93e326)  // sdot v6.4s, v25.16b, v19.4b[0]
    KAI_ASM_INST(0x4fb3e321)  // sdot v1.4s, v25.16b, v19.4b[1]
    KAI_ASM_INST(0x4f93eb35)  // sdot v21.4s, v25.16b, v19.4b[2]
    KAI_ASM_INST(0x4fb3eb22)  // sdot v2.4s, v25.16b, v19.4b[3]
    KAI_ASM_INST(0x4f92e305)  // sdot v5.4s, v24.16b, v18.4b[0]
    KAI_ASM_INST(0x4fb2e304)  // sdot v4.4s, v24.16b, v18.4b[1]
    KAI_ASM_INST(0x4f92eb03)  // sdot v3.4s, v24.16b, v18.4b[2]
    KAI_ASM_INST(0x4fb2eb0a)  // sdot v10.4s, v24.16b, v18.4b[3]
    KAI_ASM_INST(0x4f92e3a6)  // sdot v6.4s, v29.16b, v18.4b[0]
    KAI_ASM_INST(0x4fb2e3a1)  // sdot v1.4s, v29.16b, v18.4b[1]
    KAI_ASM_INST(0x4f92ebb5)  // sdot v21.4s, v29.16b, v18.4b[2]
    KAI_ASM_INST(0x4fb2eba2)  // sdot v2.4s, v29.16b, v18.4b[3]
    bgt label_4
    ldr q16, [x26, #0x0]
    scvtf v5.4s, v5.4s, #0x4
    scvtf v6.4s, v6.4s, #0x4
    add x26, x26, #0x10
    scvtf v4.4s, v4.4s, #0x4
    scvtf v1.4s, v1.4s, #0x4
    scvtf v3.4s, v3.4s, #0x4
    scvtf v21.4s, v21.4s, #0x4
    shll v17.4s, v16.4h, #0x10
    shll2 v16.4s, v16.8h, #0x10
    scvtf v10.4s, v10.4s, #0x4
    scvtf v2.4s, v2.4s, #0x4
    fmla v0.4s, v5.4s, v17.4s
    fmla v12.4s, v6.4s, v16.4s
    fmla v14.4s, v4.4s, v17.4s
    fmla v13.4s, v1.4s, v16.4s
    fmla v15.4s, v3.4s, v17.4s
    fmla v7.4s, v21.4s, v16.4s
    fmla v8.4s, v10.4s, v17.4s
    fmla v9.4s, v2.4s, v16.4s
    subs x21, x21, #0x1
    bgt label_3
    ld1 { v23.4s }, [x22]
    ldr q22, [x26, #0x0]
    add x22, x22, #0x10
    add x20, x27, #0x4
    ldr q21, [x26, #0x10]
    ldr q20, [x22, #0x0]
    cmp x25, #0x8
    ldr q19, [x26, #0x20]
    ldr q18, [x26, #0x30]
    add x26, x26, #0x40
    ld1r { v17.4s }, [x27]
    ld1r { v16.4s }, [x20]
    scvtf v23.4s, v23.4s
    fmla v0.4s, v22.4s, v23.s[0]
    fmla v12.4s, v21.4s, v23.s[0]
    fmla v14.4s, v22.4s, v23.s[1]
    fmla v13.4s, v21.4s, v23.s[1]
    fmla v15.4s, v22.4s, v23.s[2]
    fmla v7.4s, v21.4s, v23.s[2]
    fmla v8.4s, v22.4s, v23.s[3]
    fmla v9.4s, v21.4s, v23.s[3]
    fmul v0.4s, v0.4s, v20.s[0]
    fmul v12.4s, v12.4s, v20.s[0]
    fmul v14.4s, v14.4s, v20.s[1]
    fmul v13.4s, v13.4s, v20.s[1]
    fmul v15.4s, v15.4s, v20.s[2]
    fmul v7.4s, v7.4s, v20.s[2]
    fmul v8.4s, v8.4s, v20.s[3]
    fmul v9.4s, v9.4s, v20.s[3]
    fadd v0.4s, v0.4s, v19.4s
    fadd v12.4s, v12.4s, v18.4s
    fadd v14.4s, v14.4s, v19.4s
    fadd v13.4s, v13.4s, v18.4s
    fadd v15.4s, v15.4s, v19.4s
    fadd v7.4s, v7.4s, v18.4s
    fadd v8.4s, v8.4s, v19.4s
    fadd v9.4s, v9.4s, v18.4s
    fmax v0.4s, v0.4s, v17.4s
    fmax v12.4s, v12.4s, v17.4s
    fmax v14.4s, v14.4s, v17.4s
    fmax v13.4s, v13.4s, v17.4s
    fmax v15.4s, v15.4s, v17.4s
    fmax v7.4s, v7.4s, v17.4s
    fmax v8.4s, v8.4s, v17.4s
    fmax v9.4s, v9.4s, v17.4s
    fmin v0.4s, v0.4s, v16.4s
    fmin v12.4s, v12.4s, v16.4s
    fmin v14.4s, v14.4s, v16.4s
    fmin v13.4s, v13.4s, v16.4s
    fmin v15.4s, v15.4s, v16.4s
    fmin v7.4s, v7.4s, v16.4s
    fmin v8.4s, v8.4s, v16.4s
    fmin v9.4s, v9.4s, v16.4s
    blt label_6
    mov x20, x9
    cmp x10, #0x1
    str q0, [x20, #0x0]
    str q12, [x20, #0x10]
    add x20, x20, x28
    ble label_11
    cmp x10, #0x2
    str q14, [x20, #0x0]
    str q13, [x20, #0x10]
    add x20, x20, x28
    ble label_11
    cmp x10, #0x3
    str q15, [x20, #0x0]
    str q7, [x20, #0x10]
    add x20, x20, x28
    ble label_11
    str q8, [x20, #0x0]
    str q9, [x20, #0x10]
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
    st1 { v8.4s }, [x20], #0x10
    st1 { v15.4s }, [x21], #0x10
    st1 { v14.4s }, [x22], #0x10
    st1 { v0.4s }, [x23], #0x10
    tbz x25, #1, label_7
    st1 { v9.d }[0], [x20], #0x8
    st1 { v7.d }[0], [x21], #0x8
    st1 { v13.d }[0], [x22], #0x8
    st1 { v12.d }[0], [x23], #0x8
    tbz x25, #0, label_10
    st1 { v9.s }[2], [x20]
    st1 { v7.s }[2], [x21]
    st1 { v13.s }[2], [x22]
    st1 { v12.s }[2], [x23]
    b label_10
KAI_ASM_LABEL(label_7)  // Output block 0: partial_1_4
    tbz x25, #0, label_10
    st1 { v9.s }[0], [x20]
    st1 { v7.s }[0], [x21]
    st1 { v13.s }[0], [x22]
    st1 { v12.s }[0], [x23]
    b label_10
KAI_ASM_LABEL(label_8)  // Output block 0: partial_2_0
    tbz x25, #1, label_9
    st1 { v8.d }[0], [x20], #0x8
    st1 { v15.d }[0], [x21], #0x8
    st1 { v14.d }[0], [x22], #0x8
    st1 { v0.d }[0], [x23], #0x8
    tbz x25, #0, label_10
    st1 { v8.s }[2], [x20]
    st1 { v15.s }[2], [x21]
    st1 { v14.s }[2], [x22]
    st1 { v0.s }[2], [x23]
    b label_10
KAI_ASM_LABEL(label_9)  // Output block 0: partial_1_0
    st1 { v8.s }[0], [x20]
    st1 { v15.s }[0], [x21]
    st1 { v14.s }[0], [x22]
    st1 { v0.s }[0], [x23]
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod)

    KAI_ASM_END
