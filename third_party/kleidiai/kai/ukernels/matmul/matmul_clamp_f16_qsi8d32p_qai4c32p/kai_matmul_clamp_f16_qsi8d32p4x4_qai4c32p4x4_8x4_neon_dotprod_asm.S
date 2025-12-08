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

    KAI_ASM_CODE(matmul_clamp_f16_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod)
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
    movi v11.16b, #0xf0
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
    movi v0.16b, #0x0
    movi v12.16b, #0x0
    mov x22, x8
    movi v14.16b, #0x0
    movi v13.16b, #0x0
    movi v15.16b, #0x0
    movi v9.16b, #0x0
    movi v5.16b, #0x0
    movi v8.16b, #0x0
    add x21, x23, x6
KAI_ASM_LABEL(label_3)  // Block loop
    movi v1.4s, #0x0
    movi v7.4s, #0x0
    mov x20, x7
    movi v3.4s, #0x0
    movi v4.4s, #0x0
    movi v23.4s, #0x0
    movi v6.4s, #0x0
    movi v10.4s, #0x0
    movi v2.4s, #0x0
KAI_ASM_LABEL(label_4)  // Sub block loop
    ldr q31, [x10, #0x0]
    ldr q22, [x23, #0x0]
    subs x20, x20, #0x1
    ldr q30, [x21, #0x0]
    ldr q21, [x10, #0x10]
    ldr q20, [x23, #0x10]
    ldr q19, [x21, #0x10]
    ldr q29, [x10, #0x20]
    ldr q18, [x23, #0x20]
    shl v28.16b, v31.16b, #0x4
    and v31.16b, v31.16b, v11.16b
    ldr q17, [x21, #0x20]
    ldr q27, [x10, #0x30]
    shl v16.16b, v21.16b, #0x4
    and v21.16b, v21.16b, v11.16b
    ldr q26, [x23, #0x30]
    ldr q25, [x21, #0x30]
    add x10, x10, #0x40
    ldr q24, [x23, #0x40]
    KAI_ASM_INST(0x4f96e381)  // sdot v1.4s, v28.16b, v22.4b[0]
    KAI_ASM_INST(0x4fb6e387)  // sdot v7.4s, v28.16b, v22.4b[1]
    KAI_ASM_INST(0x4f96eb83)  // sdot v3.4s, v28.16b, v22.4b[2]
    KAI_ASM_INST(0x4fb6eb84)  // sdot v4.4s, v28.16b, v22.4b[3]
    ldr q22, [x21, #0x40]
    KAI_ASM_INST(0x4f9ee397)  // sdot v23.4s, v28.16b, v30.4b[0]
    KAI_ASM_INST(0x4fbee386)  // sdot v6.4s, v28.16b, v30.4b[1]
    KAI_ASM_INST(0x4f9eeb8a)  // sdot v10.4s, v28.16b, v30.4b[2]
    KAI_ASM_INST(0x4fbeeb82)  // sdot v2.4s, v28.16b, v30.4b[3]
    ldr q28, [x23, #0x50]
    ldr q30, [x21, #0x50]
    KAI_ASM_INST(0x4f94e201)  // sdot v1.4s, v16.16b, v20.4b[0]
    KAI_ASM_INST(0x4fb4e207)  // sdot v7.4s, v16.16b, v20.4b[1]
    KAI_ASM_INST(0x4f94ea03)  // sdot v3.4s, v16.16b, v20.4b[2]
    KAI_ASM_INST(0x4fb4ea04)  // sdot v4.4s, v16.16b, v20.4b[3]
    ldr q20, [x23, #0x60]
    KAI_ASM_INST(0x4f93e217)  // sdot v23.4s, v16.16b, v19.4b[0]
    KAI_ASM_INST(0x4fb3e206)  // sdot v6.4s, v16.16b, v19.4b[1]
    KAI_ASM_INST(0x4f93ea0a)  // sdot v10.4s, v16.16b, v19.4b[2]
    KAI_ASM_INST(0x4fb3ea02)  // sdot v2.4s, v16.16b, v19.4b[3]
    ldr q19, [x21, #0x60]
    shl v16.16b, v29.16b, #0x4
    and v29.16b, v29.16b, v11.16b
    KAI_ASM_INST(0x4f92e201)  // sdot v1.4s, v16.16b, v18.4b[0]
    KAI_ASM_INST(0x4fb2e207)  // sdot v7.4s, v16.16b, v18.4b[1]
    KAI_ASM_INST(0x4f92ea03)  // sdot v3.4s, v16.16b, v18.4b[2]
    KAI_ASM_INST(0x4fb2ea04)  // sdot v4.4s, v16.16b, v18.4b[3]
    ldr q18, [x23, #0x70]
    add x23, x23, #0x80
    KAI_ASM_INST(0x4f91e217)  // sdot v23.4s, v16.16b, v17.4b[0]
    KAI_ASM_INST(0x4fb1e206)  // sdot v6.4s, v16.16b, v17.4b[1]
    KAI_ASM_INST(0x4f91ea0a)  // sdot v10.4s, v16.16b, v17.4b[2]
    KAI_ASM_INST(0x4fb1ea02)  // sdot v2.4s, v16.16b, v17.4b[3]
    ldr q17, [x21, #0x70]
    shl v16.16b, v27.16b, #0x4
    and v27.16b, v27.16b, v11.16b
    add x21, x21, #0x80
    KAI_ASM_INST(0x4f9ae201)  // sdot v1.4s, v16.16b, v26.4b[0]
    KAI_ASM_INST(0x4fbae207)  // sdot v7.4s, v16.16b, v26.4b[1]
    KAI_ASM_INST(0x4f9aea03)  // sdot v3.4s, v16.16b, v26.4b[2]
    KAI_ASM_INST(0x4fbaea04)  // sdot v4.4s, v16.16b, v26.4b[3]
    KAI_ASM_INST(0x4f99e217)  // sdot v23.4s, v16.16b, v25.4b[0]
    KAI_ASM_INST(0x4fb9e206)  // sdot v6.4s, v16.16b, v25.4b[1]
    KAI_ASM_INST(0x4f99ea0a)  // sdot v10.4s, v16.16b, v25.4b[2]
    KAI_ASM_INST(0x4fb9ea02)  // sdot v2.4s, v16.16b, v25.4b[3]
    KAI_ASM_INST(0x4f98e3e1)  // sdot v1.4s, v31.16b, v24.4b[0]
    KAI_ASM_INST(0x4fb8e3e7)  // sdot v7.4s, v31.16b, v24.4b[1]
    KAI_ASM_INST(0x4f98ebe3)  // sdot v3.4s, v31.16b, v24.4b[2]
    KAI_ASM_INST(0x4fb8ebe4)  // sdot v4.4s, v31.16b, v24.4b[3]
    KAI_ASM_INST(0x4f96e3f7)  // sdot v23.4s, v31.16b, v22.4b[0]
    KAI_ASM_INST(0x4fb6e3e6)  // sdot v6.4s, v31.16b, v22.4b[1]
    KAI_ASM_INST(0x4f96ebea)  // sdot v10.4s, v31.16b, v22.4b[2]
    KAI_ASM_INST(0x4fb6ebe2)  // sdot v2.4s, v31.16b, v22.4b[3]
    KAI_ASM_INST(0x4f9ce2a1)  // sdot v1.4s, v21.16b, v28.4b[0]
    KAI_ASM_INST(0x4fbce2a7)  // sdot v7.4s, v21.16b, v28.4b[1]
    KAI_ASM_INST(0x4f9ceaa3)  // sdot v3.4s, v21.16b, v28.4b[2]
    KAI_ASM_INST(0x4fbceaa4)  // sdot v4.4s, v21.16b, v28.4b[3]
    KAI_ASM_INST(0x4f9ee2b7)  // sdot v23.4s, v21.16b, v30.4b[0]
    KAI_ASM_INST(0x4fbee2a6)  // sdot v6.4s, v21.16b, v30.4b[1]
    KAI_ASM_INST(0x4f9eeaaa)  // sdot v10.4s, v21.16b, v30.4b[2]
    KAI_ASM_INST(0x4fbeeaa2)  // sdot v2.4s, v21.16b, v30.4b[3]
    KAI_ASM_INST(0x4f94e3a1)  // sdot v1.4s, v29.16b, v20.4b[0]
    KAI_ASM_INST(0x4fb4e3a7)  // sdot v7.4s, v29.16b, v20.4b[1]
    KAI_ASM_INST(0x4f94eba3)  // sdot v3.4s, v29.16b, v20.4b[2]
    KAI_ASM_INST(0x4fb4eba4)  // sdot v4.4s, v29.16b, v20.4b[3]
    KAI_ASM_INST(0x4f93e3b7)  // sdot v23.4s, v29.16b, v19.4b[0]
    KAI_ASM_INST(0x4fb3e3a6)  // sdot v6.4s, v29.16b, v19.4b[1]
    KAI_ASM_INST(0x4f93ebaa)  // sdot v10.4s, v29.16b, v19.4b[2]
    KAI_ASM_INST(0x4fb3eba2)  // sdot v2.4s, v29.16b, v19.4b[3]
    KAI_ASM_INST(0x4f92e361)  // sdot v1.4s, v27.16b, v18.4b[0]
    KAI_ASM_INST(0x4fb2e367)  // sdot v7.4s, v27.16b, v18.4b[1]
    KAI_ASM_INST(0x4f92eb63)  // sdot v3.4s, v27.16b, v18.4b[2]
    KAI_ASM_INST(0x4fb2eb64)  // sdot v4.4s, v27.16b, v18.4b[3]
    KAI_ASM_INST(0x4f91e377)  // sdot v23.4s, v27.16b, v17.4b[0]
    KAI_ASM_INST(0x4fb1e366)  // sdot v6.4s, v27.16b, v17.4b[1]
    KAI_ASM_INST(0x4f91eb6a)  // sdot v10.4s, v27.16b, v17.4b[2]
    KAI_ASM_INST(0x4fb1eb62)  // sdot v2.4s, v27.16b, v17.4b[3]
    bgt label_4
    ldr q21, [x10, #0x0]
    ld1 { v17.4s }, [x23]
    add x23, x23, #0x10
    scvtf v1.4s, v1.4s
    ldr q20, [x10, #0x10]
    ldr q16, [x23, #0x0]
    scvtf v7.4s, v7.4s
    scvtf v3.4s, v3.4s
    scvtf v4.4s, v4.4s
    add x10, x10, #0x20
    add x23, x23, #0x10
    fmla v0.4s, v21.4s, v17.s[0]
    fmla v12.4s, v21.4s, v17.s[1]
    fmla v14.4s, v21.4s, v17.s[2]
    fmla v13.4s, v21.4s, v17.s[3]
    fmul v19.4s, v20.4s, v16.s[0]
    fmul v18.4s, v20.4s, v16.s[1]
    fmul v17.4s, v20.4s, v16.s[2]
    fmul v16.4s, v20.4s, v16.s[3]
    fmla v0.4s, v1.4s, v19.4s
    fmla v12.4s, v7.4s, v18.4s
    fmla v14.4s, v3.4s, v17.4s
    fmla v13.4s, v4.4s, v16.4s
    ld1 { v17.4s }, [x21]
    add x21, x21, #0x10
    scvtf v23.4s, v23.4s
    scvtf v6.4s, v6.4s
    ldr q16, [x21, #0x0]
    scvtf v10.4s, v10.4s
    scvtf v2.4s, v2.4s
    add x21, x21, #0x10
    fmla v15.4s, v21.4s, v17.s[0]
    fmla v9.4s, v21.4s, v17.s[1]
    fmla v5.4s, v21.4s, v17.s[2]
    fmla v8.4s, v21.4s, v17.s[3]
    fmul v19.4s, v20.4s, v16.s[0]
    fmul v18.4s, v20.4s, v16.s[1]
    fmul v17.4s, v20.4s, v16.s[2]
    fmul v16.4s, v20.4s, v16.s[3]
    fmla v15.4s, v23.4s, v19.4s
    fmla v9.4s, v6.4s, v18.4s
    fmla v5.4s, v10.4s, v17.4s
    fmla v8.4s, v2.4s, v16.4s
    subs x22, x22, #0x1
    bgt label_3
    ldr q18, [x10, #0x0]
    ld1r { v17.4s }, [x11]
    add x20, x11, #0x4
    cmp x9, #0x4
    ld1r { v16.4s }, [x20]
    add x10, x10, #0x10
    fadd v0.4s, v0.4s, v18.4s
    fadd v12.4s, v12.4s, v18.4s
    fadd v14.4s, v14.4s, v18.4s
    fadd v13.4s, v13.4s, v18.4s
    fadd v15.4s, v15.4s, v18.4s
    fadd v9.4s, v9.4s, v18.4s
    fadd v5.4s, v5.4s, v18.4s
    fadd v8.4s, v8.4s, v18.4s
    fmax v0.4s, v0.4s, v17.4s
    fmax v12.4s, v12.4s, v17.4s
    fmax v14.4s, v14.4s, v17.4s
    fmax v13.4s, v13.4s, v17.4s
    fmax v15.4s, v15.4s, v17.4s
    fmax v9.4s, v9.4s, v17.4s
    fmax v5.4s, v5.4s, v17.4s
    fmax v8.4s, v8.4s, v17.4s
    fmin v0.4s, v0.4s, v16.4s
    fmin v12.4s, v12.4s, v16.4s
    fmin v14.4s, v14.4s, v16.4s
    fmin v13.4s, v13.4s, v16.4s
    fmin v15.4s, v15.4s, v16.4s
    fmin v9.4s, v9.4s, v16.4s
    fmin v5.4s, v5.4s, v16.4s
    fmin v8.4s, v8.4s, v16.4s
    fcvtn v23.4h, v0.4s
    fcvtn v22.4h, v12.4s
    fcvtn v21.4h, v14.4s
    fcvtn v20.4h, v13.4s
    fcvtn v19.4h, v15.4s
    fcvtn v18.4h, v9.4s
    fcvtn v17.4h, v5.4s
    fcvtn v16.4h, v8.4s
    blt label_7
    mov x20, x13
    str d23, [x20, #0x0]
    add x20, x20, x12
    str d22, [x20, #0x0]
    add x20, x20, x12
    str d21, [x20, #0x0]
    add x20, x20, x12
    str d20, [x20, #0x0]
    add x20, x20, x12
    str d19, [x20, #0x0]
    add x20, x20, x12
    str d18, [x20, #0x0]
    add x20, x20, x12
    str d17, [x20, #0x0]
    add x20, x20, x12
    str d16, [x20, #0x0]
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
    st1 { v16.s }[0], [x23], #0x4
    st1 { v17.s }[0], [x25], #0x4
    st1 { v18.s }[0], [x24], #0x4
    st1 { v19.s }[0], [x26], #0x4
    st1 { v20.s }[0], [x20], #0x4
    st1 { v21.s }[0], [x22], #0x4
    st1 { v22.s }[0], [x21], #0x4
    st1 { v23.s }[0], [x27], #0x4
    tbz x9, #0, label_9
    st1 { v16.h }[2], [x23]
    st1 { v17.h }[2], [x25]
    st1 { v18.h }[2], [x24]
    st1 { v19.h }[2], [x26]
    st1 { v20.h }[2], [x20]
    st1 { v21.h }[2], [x22]
    st1 { v22.h }[2], [x21]
    st1 { v23.h }[2], [x27]
    b label_9
KAI_ASM_LABEL(label_8)  // Output block 0: partial_1_0
    st1 { v16.h }[0], [x23]
    st1 { v17.h }[0], [x25]
    st1 { v18.h }[0], [x24]
    st1 { v19.h }[0], [x26]
    st1 { v20.h }[0], [x20]
    st1 { v21.h }[0], [x22]
    st1 { v22.h }[0], [x21]
    st1 { v23.h }[0], [x27]
KAI_ASM_LABEL(label_9)  // Output block 0: Done
KAI_ASM_LABEL(label_10)  // Output stage exit
    subs x9, x9, #0x4
    add x13, x13, #0x8
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
    movi v0.16b, #0x0
    movi v12.16b, #0x0
    mov x23, x17
    mov x21, x8
    movi v14.16b, #0x0
    movi v13.16b, #0x0
KAI_ASM_LABEL(label_14)  // Row tail: Block loop
    movi v1.4s, #0x0
    movi v7.4s, #0x0
    mov x20, x7
    movi v3.4s, #0x0
    movi v4.4s, #0x0
KAI_ASM_LABEL(label_15)  // Row tail: Sub block loop
    ldr q31, [x26, #0x0]
    ldr q30, [x23, #0x0]
    subs x20, x20, #0x1
    ldr q29, [x26, #0x10]
    ldr q28, [x23, #0x10]
    ldr q27, [x26, #0x20]
    ldr q26, [x23, #0x20]
    ldr q25, [x26, #0x30]
    ldr q24, [x23, #0x30]
    shl v23.16b, v31.16b, #0x4
    and v31.16b, v31.16b, v11.16b
    ldr q22, [x23, #0x40]
    ldr q21, [x23, #0x50]
    shl v20.16b, v29.16b, #0x4
    and v29.16b, v29.16b, v11.16b
    ldr q19, [x23, #0x60]
    ldr q18, [x23, #0x70]
    shl v17.16b, v27.16b, #0x4
    and v27.16b, v27.16b, v11.16b
    KAI_ASM_INST(0x4f9ee2e1)  // sdot v1.4s, v23.16b, v30.4b[0]
    KAI_ASM_INST(0x4fbee2e7)  // sdot v7.4s, v23.16b, v30.4b[1]
    shl v16.16b, v25.16b, #0x4
    add x26, x26, #0x40
    KAI_ASM_INST(0x4f9eeae3)  // sdot v3.4s, v23.16b, v30.4b[2]
    KAI_ASM_INST(0x4fbeeae4)  // sdot v4.4s, v23.16b, v30.4b[3]
    and v25.16b, v25.16b, v11.16b
    add x23, x23, #0x80
    KAI_ASM_INST(0x4f9ce281)  // sdot v1.4s, v20.16b, v28.4b[0]
    KAI_ASM_INST(0x4fbce287)  // sdot v7.4s, v20.16b, v28.4b[1]
    KAI_ASM_INST(0x4f9cea83)  // sdot v3.4s, v20.16b, v28.4b[2]
    KAI_ASM_INST(0x4fbcea84)  // sdot v4.4s, v20.16b, v28.4b[3]
    KAI_ASM_INST(0x4f9ae221)  // sdot v1.4s, v17.16b, v26.4b[0]
    KAI_ASM_INST(0x4fbae227)  // sdot v7.4s, v17.16b, v26.4b[1]
    KAI_ASM_INST(0x4f9aea23)  // sdot v3.4s, v17.16b, v26.4b[2]
    KAI_ASM_INST(0x4fbaea24)  // sdot v4.4s, v17.16b, v26.4b[3]
    KAI_ASM_INST(0x4f98e201)  // sdot v1.4s, v16.16b, v24.4b[0]
    KAI_ASM_INST(0x4fb8e207)  // sdot v7.4s, v16.16b, v24.4b[1]
    KAI_ASM_INST(0x4f98ea03)  // sdot v3.4s, v16.16b, v24.4b[2]
    KAI_ASM_INST(0x4fb8ea04)  // sdot v4.4s, v16.16b, v24.4b[3]
    KAI_ASM_INST(0x4f96e3e1)  // sdot v1.4s, v31.16b, v22.4b[0]
    KAI_ASM_INST(0x4fb6e3e7)  // sdot v7.4s, v31.16b, v22.4b[1]
    KAI_ASM_INST(0x4f96ebe3)  // sdot v3.4s, v31.16b, v22.4b[2]
    KAI_ASM_INST(0x4fb6ebe4)  // sdot v4.4s, v31.16b, v22.4b[3]
    KAI_ASM_INST(0x4f95e3a1)  // sdot v1.4s, v29.16b, v21.4b[0]
    KAI_ASM_INST(0x4fb5e3a7)  // sdot v7.4s, v29.16b, v21.4b[1]
    KAI_ASM_INST(0x4f95eba3)  // sdot v3.4s, v29.16b, v21.4b[2]
    KAI_ASM_INST(0x4fb5eba4)  // sdot v4.4s, v29.16b, v21.4b[3]
    KAI_ASM_INST(0x4f93e361)  // sdot v1.4s, v27.16b, v19.4b[0]
    KAI_ASM_INST(0x4fb3e367)  // sdot v7.4s, v27.16b, v19.4b[1]
    KAI_ASM_INST(0x4f93eb63)  // sdot v3.4s, v27.16b, v19.4b[2]
    KAI_ASM_INST(0x4fb3eb64)  // sdot v4.4s, v27.16b, v19.4b[3]
    KAI_ASM_INST(0x4f92e321)  // sdot v1.4s, v25.16b, v18.4b[0]
    KAI_ASM_INST(0x4fb2e327)  // sdot v7.4s, v25.16b, v18.4b[1]
    KAI_ASM_INST(0x4f92eb23)  // sdot v3.4s, v25.16b, v18.4b[2]
    KAI_ASM_INST(0x4fb2eb24)  // sdot v4.4s, v25.16b, v18.4b[3]
    bgt label_15
    ldr q18, [x26, #0x0]
    ld1 { v17.4s }, [x23]
    add x23, x23, #0x10
    scvtf v1.4s, v1.4s
    ldr q20, [x26, #0x10]
    ldr q16, [x23, #0x0]
    scvtf v7.4s, v7.4s
    scvtf v3.4s, v3.4s
    scvtf v4.4s, v4.4s
    add x26, x26, #0x20
    add x23, x23, #0x10
    fmla v0.4s, v18.4s, v17.s[0]
    fmla v12.4s, v18.4s, v17.s[1]
    fmla v14.4s, v18.4s, v17.s[2]
    fmla v13.4s, v18.4s, v17.s[3]
    fmul v19.4s, v20.4s, v16.s[0]
    fmul v18.4s, v20.4s, v16.s[1]
    fmul v17.4s, v20.4s, v16.s[2]
    fmul v16.4s, v20.4s, v16.s[3]
    fmla v0.4s, v1.4s, v19.4s
    fmla v12.4s, v7.4s, v18.4s
    fmla v14.4s, v3.4s, v17.4s
    fmla v13.4s, v4.4s, v16.4s
    subs x21, x21, #0x1
    bgt label_14
    ldr q18, [x26, #0x0]
    ld1r { v17.4s }, [x11]
    add x20, x11, #0x4
    cmp x25, #0x4
    ld1r { v16.4s }, [x20]
    add x26, x26, #0x10
    fadd v0.4s, v0.4s, v18.4s
    fadd v12.4s, v12.4s, v18.4s
    fadd v14.4s, v14.4s, v18.4s
    fadd v13.4s, v13.4s, v18.4s
    fmax v0.4s, v0.4s, v17.4s
    fmax v12.4s, v12.4s, v17.4s
    fmax v14.4s, v14.4s, v17.4s
    fmax v13.4s, v13.4s, v17.4s
    fmin v0.4s, v0.4s, v16.4s
    fmin v12.4s, v12.4s, v16.4s
    fmin v14.4s, v14.4s, v16.4s
    fmin v13.4s, v13.4s, v16.4s
    fcvtn v19.4h, v0.4s
    fcvtn v18.4h, v12.4s
    fcvtn v17.4h, v14.4s
    fcvtn v16.4h, v13.4s
    blt label_17
    mov x20, x13
    cmp x14, #0x1
    str d19, [x20, #0x0]
    add x20, x20, x12
    ble label_20
    cmp x14, #0x2
    str d18, [x20, #0x0]
    add x20, x20, x12
    ble label_20
    cmp x14, #0x3
    str d17, [x20, #0x0]
    add x20, x20, x12
    ble label_20
    str d16, [x20, #0x0]
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
    st1 { v16.s }[0], [x20], #0x4
    st1 { v17.s }[0], [x21], #0x4
    st1 { v18.s }[0], [x22], #0x4
    st1 { v19.s }[0], [x23], #0x4
    tbz x25, #0, label_19
    st1 { v16.h }[2], [x20]
    st1 { v17.h }[2], [x21]
    st1 { v18.h }[2], [x22]
    st1 { v19.h }[2], [x23]
    b label_19
KAI_ASM_LABEL(label_18)  // Row tail: Output block 0: partial_1_0
    st1 { v16.h }[0], [x20]
    st1 { v17.h }[0], [x21]
    st1 { v18.h }[0], [x22]
    st1 { v19.h }[0], [x23]
KAI_ASM_LABEL(label_19)  // Row tail: Output block 0: Done
KAI_ASM_LABEL(label_20)  // Row tail: Output stage exit
    subs x25, x25, #0x4
    add x13, x13, #0x8
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod)

    KAI_ASM_END
