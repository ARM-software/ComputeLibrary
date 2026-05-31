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

    KAI_ASM_CODE(matmul_clamp_f32_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d10, d11, [sp, 72]
    stp d12, d13, [sp, 88]
    stp d14, d15, [sp, 104]
    stp d8, d9, [sp, 120]
    mov x21, #0x20
    movi v13.16b, #0xf0
    mov x6, #0x80
    ldr x20, [x0, #0x28]
    ldr x7, [x0, #0x40]
    ldr x8, [x0, #0x38]
    ldr x17, [x0, #0x8]
    ldr x16, [x0, #0x10]
    ldr x15, [x0, #0x30]
    mov x14, x20
    madd x6, x7, x6, x21
    ldr x13, [x0, #0x0]
    ldr x12, [x0, #0x20]
    ldr x11, [x0, #0x18]
    cmp x14, #0x8
    mul x6, x8, x6
    blt label_11
KAI_ASM_LABEL(label_1)  // Row loop
    mov x10, x16
    mov x9, x15
    add x28, x13, x12, LSL #3
KAI_ASM_LABEL(label_2)  // Column loop
    mov x23, x17
    movi v24.16b, #0x0
    movi v11.16b, #0x0
    mov x22, x8
    movi v23.16b, #0x0
    movi v14.16b, #0x0
    movi v12.16b, #0x0
    movi v31.16b, #0x0
    movi v7.16b, #0x0
    movi v25.16b, #0x0
    add x21, x23, x6
KAI_ASM_LABEL(label_3)  // Block loop
    movi v6.4s, #0x0
    movi v2.4s, #0x0
    mov x20, x7
    movi v22.4s, #0x0
    movi v3.4s, #0x0
    movi v9.4s, #0x0
    movi v20.4s, #0x0
    movi v5.4s, #0x0
    movi v0.4s, #0x0
KAI_ASM_LABEL(label_4)  // Sub block loop
    ldr q30, [x10, #0x0]
    ldr q15, [x23, #0x0]
    subs x20, x20, #0x1
    ldr q18, [x21, #0x0]
    ldr q4, [x10, #0x10]
    ldr q29, [x23, #0x10]
    ldr q10, [x21, #0x10]
    ldr q26, [x10, #0x20]
    ldr q28, [x23, #0x20]
    shl v19.16b, v30.16b, #0x4
    and v30.16b, v30.16b, v13.16b
    ldr q17, [x21, #0x20]
    ldr q16, [x10, #0x30]
    shl v27.16b, v4.16b, #0x4
    and v4.16b, v4.16b, v13.16b
    ldr q8, [x23, #0x30]
    ldr q21, [x21, #0x30]
    add x10, x10, #0x40
    ldr q1, [x23, #0x40]
    KAI_ASM_INST(0x4f8fe266)  // sdot v6.4s, v19.16b, v15.4b[0]
    KAI_ASM_INST(0x4fafe262)  // sdot v2.4s, v19.16b, v15.4b[1]
    KAI_ASM_INST(0x4f8fea76)  // sdot v22.4s, v19.16b, v15.4b[2]
    KAI_ASM_INST(0x4fafea63)  // sdot v3.4s, v19.16b, v15.4b[3]
    ldr q15, [x21, #0x40]
    KAI_ASM_INST(0x4f92e269)  // sdot v9.4s, v19.16b, v18.4b[0]
    KAI_ASM_INST(0x4fb2e274)  // sdot v20.4s, v19.16b, v18.4b[1]
    KAI_ASM_INST(0x4f92ea65)  // sdot v5.4s, v19.16b, v18.4b[2]
    KAI_ASM_INST(0x4fb2ea60)  // sdot v0.4s, v19.16b, v18.4b[3]
    ldr q18, [x23, #0x50]
    ldr q19, [x21, #0x50]
    KAI_ASM_INST(0x4f9de366)  // sdot v6.4s, v27.16b, v29.4b[0]
    KAI_ASM_INST(0x4fbde362)  // sdot v2.4s, v27.16b, v29.4b[1]
    KAI_ASM_INST(0x4f9deb76)  // sdot v22.4s, v27.16b, v29.4b[2]
    KAI_ASM_INST(0x4fbdeb63)  // sdot v3.4s, v27.16b, v29.4b[3]
    ldr q29, [x23, #0x60]
    KAI_ASM_INST(0x4f8ae369)  // sdot v9.4s, v27.16b, v10.4b[0]
    KAI_ASM_INST(0x4faae374)  // sdot v20.4s, v27.16b, v10.4b[1]
    KAI_ASM_INST(0x4f8aeb65)  // sdot v5.4s, v27.16b, v10.4b[2]
    KAI_ASM_INST(0x4faaeb60)  // sdot v0.4s, v27.16b, v10.4b[3]
    ldr q10, [x21, #0x60]
    shl v27.16b, v26.16b, #0x4
    and v26.16b, v26.16b, v13.16b
    KAI_ASM_INST(0x4f9ce366)  // sdot v6.4s, v27.16b, v28.4b[0]
    KAI_ASM_INST(0x4fbce362)  // sdot v2.4s, v27.16b, v28.4b[1]
    KAI_ASM_INST(0x4f9ceb76)  // sdot v22.4s, v27.16b, v28.4b[2]
    KAI_ASM_INST(0x4fbceb63)  // sdot v3.4s, v27.16b, v28.4b[3]
    ldr q28, [x23, #0x70]
    add x23, x23, #0x80
    KAI_ASM_INST(0x4f91e369)  // sdot v9.4s, v27.16b, v17.4b[0]
    KAI_ASM_INST(0x4fb1e374)  // sdot v20.4s, v27.16b, v17.4b[1]
    KAI_ASM_INST(0x4f91eb65)  // sdot v5.4s, v27.16b, v17.4b[2]
    KAI_ASM_INST(0x4fb1eb60)  // sdot v0.4s, v27.16b, v17.4b[3]
    ldr q27, [x21, #0x70]
    shl v17.16b, v16.16b, #0x4
    and v16.16b, v16.16b, v13.16b
    add x21, x21, #0x80
    KAI_ASM_INST(0x4f88e226)  // sdot v6.4s, v17.16b, v8.4b[0]
    KAI_ASM_INST(0x4fa8e222)  // sdot v2.4s, v17.16b, v8.4b[1]
    KAI_ASM_INST(0x4f88ea36)  // sdot v22.4s, v17.16b, v8.4b[2]
    KAI_ASM_INST(0x4fa8ea23)  // sdot v3.4s, v17.16b, v8.4b[3]
    KAI_ASM_INST(0x4f95e229)  // sdot v9.4s, v17.16b, v21.4b[0]
    KAI_ASM_INST(0x4fb5e234)  // sdot v20.4s, v17.16b, v21.4b[1]
    KAI_ASM_INST(0x4f95ea25)  // sdot v5.4s, v17.16b, v21.4b[2]
    KAI_ASM_INST(0x4fb5ea20)  // sdot v0.4s, v17.16b, v21.4b[3]
    KAI_ASM_INST(0x4f81e3c6)  // sdot v6.4s, v30.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e3c2)  // sdot v2.4s, v30.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81ebd6)  // sdot v22.4s, v30.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1ebc3)  // sdot v3.4s, v30.16b, v1.4b[3]
    KAI_ASM_INST(0x4f8fe3c9)  // sdot v9.4s, v30.16b, v15.4b[0]
    KAI_ASM_INST(0x4fafe3d4)  // sdot v20.4s, v30.16b, v15.4b[1]
    KAI_ASM_INST(0x4f8febc5)  // sdot v5.4s, v30.16b, v15.4b[2]
    KAI_ASM_INST(0x4fafebc0)  // sdot v0.4s, v30.16b, v15.4b[3]
    KAI_ASM_INST(0x4f92e086)  // sdot v6.4s, v4.16b, v18.4b[0]
    KAI_ASM_INST(0x4fb2e082)  // sdot v2.4s, v4.16b, v18.4b[1]
    KAI_ASM_INST(0x4f92e896)  // sdot v22.4s, v4.16b, v18.4b[2]
    KAI_ASM_INST(0x4fb2e883)  // sdot v3.4s, v4.16b, v18.4b[3]
    KAI_ASM_INST(0x4f93e089)  // sdot v9.4s, v4.16b, v19.4b[0]
    KAI_ASM_INST(0x4fb3e094)  // sdot v20.4s, v4.16b, v19.4b[1]
    KAI_ASM_INST(0x4f93e885)  // sdot v5.4s, v4.16b, v19.4b[2]
    KAI_ASM_INST(0x4fb3e880)  // sdot v0.4s, v4.16b, v19.4b[3]
    KAI_ASM_INST(0x4f9de346)  // sdot v6.4s, v26.16b, v29.4b[0]
    KAI_ASM_INST(0x4fbde342)  // sdot v2.4s, v26.16b, v29.4b[1]
    KAI_ASM_INST(0x4f9deb56)  // sdot v22.4s, v26.16b, v29.4b[2]
    KAI_ASM_INST(0x4fbdeb43)  // sdot v3.4s, v26.16b, v29.4b[3]
    KAI_ASM_INST(0x4f8ae349)  // sdot v9.4s, v26.16b, v10.4b[0]
    KAI_ASM_INST(0x4faae354)  // sdot v20.4s, v26.16b, v10.4b[1]
    KAI_ASM_INST(0x4f8aeb45)  // sdot v5.4s, v26.16b, v10.4b[2]
    KAI_ASM_INST(0x4faaeb40)  // sdot v0.4s, v26.16b, v10.4b[3]
    KAI_ASM_INST(0x4f9ce206)  // sdot v6.4s, v16.16b, v28.4b[0]
    KAI_ASM_INST(0x4fbce202)  // sdot v2.4s, v16.16b, v28.4b[1]
    KAI_ASM_INST(0x4f9cea16)  // sdot v22.4s, v16.16b, v28.4b[2]
    KAI_ASM_INST(0x4fbcea03)  // sdot v3.4s, v16.16b, v28.4b[3]
    KAI_ASM_INST(0x4f9be209)  // sdot v9.4s, v16.16b, v27.4b[0]
    KAI_ASM_INST(0x4fbbe214)  // sdot v20.4s, v16.16b, v27.4b[1]
    KAI_ASM_INST(0x4f9bea05)  // sdot v5.4s, v16.16b, v27.4b[2]
    KAI_ASM_INST(0x4fbbea00)  // sdot v0.4s, v16.16b, v27.4b[3]
    bgt label_4
    ldr q1, [x10, #0x0]
    ld1 { v17.4s }, [x23]
    add x23, x23, #0x10
    scvtf v6.4s, v6.4s
    ldr q29, [x10, #0x10]
    ldr q16, [x23, #0x0]
    scvtf v2.4s, v2.4s
    scvtf v22.4s, v22.4s
    scvtf v3.4s, v3.4s
    add x10, x10, #0x20
    add x23, x23, #0x10
    fmla v24.4s, v1.4s, v17.s[0]
    fmla v11.4s, v1.4s, v17.s[1]
    fmla v23.4s, v1.4s, v17.s[2]
    fmla v14.4s, v1.4s, v17.s[3]
    fmul v8.4s, v29.4s, v16.s[0]
    fmul v18.4s, v29.4s, v16.s[1]
    fmul v17.4s, v29.4s, v16.s[2]
    fmul v16.4s, v29.4s, v16.s[3]
    fmla v24.4s, v6.4s, v8.4s
    fmla v11.4s, v2.4s, v18.4s
    fmla v23.4s, v22.4s, v17.4s
    fmla v14.4s, v3.4s, v16.4s
    ld1 { v17.4s }, [x21]
    add x21, x21, #0x10
    scvtf v9.4s, v9.4s
    scvtf v20.4s, v20.4s
    ldr q16, [x21, #0x0]
    scvtf v5.4s, v5.4s
    scvtf v0.4s, v0.4s
    add x21, x21, #0x10
    fmla v12.4s, v1.4s, v17.s[0]
    fmla v31.4s, v1.4s, v17.s[1]
    fmla v7.4s, v1.4s, v17.s[2]
    fmla v25.4s, v1.4s, v17.s[3]
    fmul v19.4s, v29.4s, v16.s[0]
    fmul v18.4s, v29.4s, v16.s[1]
    fmul v17.4s, v29.4s, v16.s[2]
    fmul v16.4s, v29.4s, v16.s[3]
    fmla v12.4s, v9.4s, v19.4s
    fmla v31.4s, v20.4s, v18.4s
    fmla v7.4s, v5.4s, v17.4s
    fmla v25.4s, v0.4s, v16.4s
    subs x22, x22, #0x1
    bgt label_3
    ldr q18, [x10, #0x0]
    ld1r { v17.4s }, [x11]
    add x20, x11, #0x4
    cmp x9, #0x4
    ld1r { v16.4s }, [x20]
    add x10, x10, #0x10
    fadd v24.4s, v24.4s, v18.4s
    fadd v11.4s, v11.4s, v18.4s
    fadd v23.4s, v23.4s, v18.4s
    fadd v14.4s, v14.4s, v18.4s
    fadd v12.4s, v12.4s, v18.4s
    fadd v31.4s, v31.4s, v18.4s
    fadd v7.4s, v7.4s, v18.4s
    fadd v25.4s, v25.4s, v18.4s
    fmax v24.4s, v24.4s, v17.4s
    fmax v11.4s, v11.4s, v17.4s
    fmax v23.4s, v23.4s, v17.4s
    fmax v14.4s, v14.4s, v17.4s
    fmax v12.4s, v12.4s, v17.4s
    fmax v31.4s, v31.4s, v17.4s
    fmax v7.4s, v7.4s, v17.4s
    fmax v25.4s, v25.4s, v17.4s
    fmin v24.4s, v24.4s, v16.4s
    fmin v11.4s, v11.4s, v16.4s
    fmin v23.4s, v23.4s, v16.4s
    fmin v14.4s, v14.4s, v16.4s
    fmin v12.4s, v12.4s, v16.4s
    fmin v31.4s, v31.4s, v16.4s
    fmin v7.4s, v7.4s, v16.4s
    fmin v25.4s, v25.4s, v16.4s
    blt label_7
    mov x20, x13
    str q24, [x20, #0x0]
    add x20, x20, x12
    str q11, [x20, #0x0]
    add x20, x20, x12
    str q23, [x20, #0x0]
    add x20, x20, x12
    str q14, [x20, #0x0]
    add x20, x20, x12
    str q12, [x20, #0x0]
    add x20, x20, x12
    str q31, [x20, #0x0]
    add x20, x20, x12
    str q7, [x20, #0x0]
    add x20, x20, x12
    str q25, [x20, #0x0]
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
    st1 { v25.d }[0], [x23], #0x8
    st1 { v7.d }[0], [x25], #0x8
    st1 { v31.d }[0], [x24], #0x8
    st1 { v12.d }[0], [x26], #0x8
    st1 { v14.d }[0], [x20], #0x8
    st1 { v23.d }[0], [x22], #0x8
    st1 { v11.d }[0], [x21], #0x8
    st1 { v24.d }[0], [x27], #0x8
    tbz x9, #0, label_9
    st1 { v25.s }[2], [x23]
    st1 { v7.s }[2], [x25]
    st1 { v31.s }[2], [x24]
    st1 { v12.s }[2], [x26]
    st1 { v14.s }[2], [x20]
    st1 { v23.s }[2], [x22]
    st1 { v11.s }[2], [x21]
    st1 { v24.s }[2], [x27]
    b label_9
KAI_ASM_LABEL(label_8)  // Output block 0: partial_1_0
    st1 { v25.s }[0], [x23]
    st1 { v7.s }[0], [x25]
    st1 { v31.s }[0], [x24]
    st1 { v12.s }[0], [x26]
    st1 { v14.s }[0], [x20]
    st1 { v23.s }[0], [x22]
    st1 { v11.s }[0], [x21]
    st1 { v24.s }[0], [x27]
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
    movi v24.16b, #0x0
    movi v11.16b, #0x0
    mov x23, x17
    mov x21, x8
    movi v23.16b, #0x0
    movi v14.16b, #0x0
KAI_ASM_LABEL(label_14)  // Row tail: Block loop
    movi v6.4s, #0x0
    movi v2.4s, #0x0
    mov x20, x7
    movi v22.4s, #0x0
    movi v3.4s, #0x0
KAI_ASM_LABEL(label_15)  // Row tail: Sub block loop
    ldr q31, [x26, #0x0]
    ldr q30, [x23, #0x0]
    subs x20, x20, #0x1
    ldr q29, [x26, #0x10]
    ldr q28, [x23, #0x10]
    ldr q5, [x26, #0x20]
    ldr q27, [x23, #0x20]
    ldr q26, [x26, #0x30]
    ldr q25, [x23, #0x30]
    shl v7.16b, v31.16b, #0x4
    and v31.16b, v31.16b, v13.16b
    ldr q10, [x23, #0x40]
    ldr q8, [x23, #0x50]
    shl v21.16b, v29.16b, #0x4
    and v29.16b, v29.16b, v13.16b
    ldr q20, [x23, #0x60]
    ldr q18, [x23, #0x70]
    shl v17.16b, v5.16b, #0x4
    and v5.16b, v5.16b, v13.16b
    KAI_ASM_INST(0x4f9ee0e6)  // sdot v6.4s, v7.16b, v30.4b[0]
    KAI_ASM_INST(0x4fbee0e2)  // sdot v2.4s, v7.16b, v30.4b[1]
    shl v16.16b, v26.16b, #0x4
    add x26, x26, #0x40
    KAI_ASM_INST(0x4f9ee8f6)  // sdot v22.4s, v7.16b, v30.4b[2]
    KAI_ASM_INST(0x4fbee8e3)  // sdot v3.4s, v7.16b, v30.4b[3]
    and v26.16b, v26.16b, v13.16b
    add x23, x23, #0x80
    KAI_ASM_INST(0x4f9ce2a6)  // sdot v6.4s, v21.16b, v28.4b[0]
    KAI_ASM_INST(0x4fbce2a2)  // sdot v2.4s, v21.16b, v28.4b[1]
    KAI_ASM_INST(0x4f9ceab6)  // sdot v22.4s, v21.16b, v28.4b[2]
    KAI_ASM_INST(0x4fbceaa3)  // sdot v3.4s, v21.16b, v28.4b[3]
    KAI_ASM_INST(0x4f9be226)  // sdot v6.4s, v17.16b, v27.4b[0]
    KAI_ASM_INST(0x4fbbe222)  // sdot v2.4s, v17.16b, v27.4b[1]
    KAI_ASM_INST(0x4f9bea36)  // sdot v22.4s, v17.16b, v27.4b[2]
    KAI_ASM_INST(0x4fbbea23)  // sdot v3.4s, v17.16b, v27.4b[3]
    KAI_ASM_INST(0x4f99e206)  // sdot v6.4s, v16.16b, v25.4b[0]
    KAI_ASM_INST(0x4fb9e202)  // sdot v2.4s, v16.16b, v25.4b[1]
    KAI_ASM_INST(0x4f99ea16)  // sdot v22.4s, v16.16b, v25.4b[2]
    KAI_ASM_INST(0x4fb9ea03)  // sdot v3.4s, v16.16b, v25.4b[3]
    KAI_ASM_INST(0x4f8ae3e6)  // sdot v6.4s, v31.16b, v10.4b[0]
    KAI_ASM_INST(0x4faae3e2)  // sdot v2.4s, v31.16b, v10.4b[1]
    KAI_ASM_INST(0x4f8aebf6)  // sdot v22.4s, v31.16b, v10.4b[2]
    KAI_ASM_INST(0x4faaebe3)  // sdot v3.4s, v31.16b, v10.4b[3]
    KAI_ASM_INST(0x4f88e3a6)  // sdot v6.4s, v29.16b, v8.4b[0]
    KAI_ASM_INST(0x4fa8e3a2)  // sdot v2.4s, v29.16b, v8.4b[1]
    KAI_ASM_INST(0x4f88ebb6)  // sdot v22.4s, v29.16b, v8.4b[2]
    KAI_ASM_INST(0x4fa8eba3)  // sdot v3.4s, v29.16b, v8.4b[3]
    KAI_ASM_INST(0x4f94e0a6)  // sdot v6.4s, v5.16b, v20.4b[0]
    KAI_ASM_INST(0x4fb4e0a2)  // sdot v2.4s, v5.16b, v20.4b[1]
    KAI_ASM_INST(0x4f94e8b6)  // sdot v22.4s, v5.16b, v20.4b[2]
    KAI_ASM_INST(0x4fb4e8a3)  // sdot v3.4s, v5.16b, v20.4b[3]
    KAI_ASM_INST(0x4f92e346)  // sdot v6.4s, v26.16b, v18.4b[0]
    KAI_ASM_INST(0x4fb2e342)  // sdot v2.4s, v26.16b, v18.4b[1]
    KAI_ASM_INST(0x4f92eb56)  // sdot v22.4s, v26.16b, v18.4b[2]
    KAI_ASM_INST(0x4fb2eb43)  // sdot v3.4s, v26.16b, v18.4b[3]
    bgt label_15
    ldr q18, [x26, #0x0]
    ld1 { v17.4s }, [x23]
    add x23, x23, #0x10
    scvtf v6.4s, v6.4s
    ldr q20, [x26, #0x10]
    ldr q16, [x23, #0x0]
    scvtf v2.4s, v2.4s
    scvtf v22.4s, v22.4s
    scvtf v3.4s, v3.4s
    add x26, x26, #0x20
    add x23, x23, #0x10
    fmla v24.4s, v18.4s, v17.s[0]
    fmla v11.4s, v18.4s, v17.s[1]
    fmla v23.4s, v18.4s, v17.s[2]
    fmla v14.4s, v18.4s, v17.s[3]
    fmul v17.4s, v20.4s, v16.s[0]
    fmul v18.4s, v20.4s, v16.s[1]
    fmul v8.4s, v20.4s, v16.s[2]
    fmul v16.4s, v20.4s, v16.s[3]
    fmla v24.4s, v6.4s, v17.4s
    fmla v11.4s, v2.4s, v18.4s
    fmla v23.4s, v22.4s, v8.4s
    fmla v14.4s, v3.4s, v16.4s
    subs x21, x21, #0x1
    bgt label_14
    ldr q18, [x26, #0x0]
    ld1r { v17.4s }, [x11]
    add x20, x11, #0x4
    cmp x25, #0x4
    ld1r { v16.4s }, [x20]
    add x26, x26, #0x10
    fadd v24.4s, v24.4s, v18.4s
    fadd v11.4s, v11.4s, v18.4s
    fadd v23.4s, v23.4s, v18.4s
    fadd v14.4s, v14.4s, v18.4s
    fmax v24.4s, v24.4s, v17.4s
    fmax v11.4s, v11.4s, v17.4s
    fmax v23.4s, v23.4s, v17.4s
    fmax v14.4s, v14.4s, v17.4s
    fmin v24.4s, v24.4s, v16.4s
    fmin v11.4s, v11.4s, v16.4s
    fmin v23.4s, v23.4s, v16.4s
    fmin v14.4s, v14.4s, v16.4s
    blt label_17
    mov x20, x13
    cmp x14, #0x1
    str q24, [x20, #0x0]
    add x20, x20, x12
    ble label_20
    cmp x14, #0x2
    str q11, [x20, #0x0]
    add x20, x20, x12
    ble label_20
    cmp x14, #0x3
    str q23, [x20, #0x0]
    add x20, x20, x12
    ble label_20
    str q14, [x20, #0x0]
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
    st1 { v14.d }[0], [x20], #0x8
    st1 { v23.d }[0], [x21], #0x8
    st1 { v11.d }[0], [x22], #0x8
    st1 { v24.d }[0], [x23], #0x8
    tbz x25, #0, label_19
    st1 { v14.s }[2], [x20]
    st1 { v23.s }[2], [x21]
    st1 { v11.s }[2], [x22]
    st1 { v24.s }[2], [x23]
    b label_19
KAI_ASM_LABEL(label_18)  // Row tail: Output block 0: partial_1_0
    st1 { v14.s }[0], [x20]
    st1 { v23.s }[0], [x21]
    st1 { v11.s }[0], [x22]
    st1 { v24.s }[0], [x23]
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod)

    KAI_ASM_END
