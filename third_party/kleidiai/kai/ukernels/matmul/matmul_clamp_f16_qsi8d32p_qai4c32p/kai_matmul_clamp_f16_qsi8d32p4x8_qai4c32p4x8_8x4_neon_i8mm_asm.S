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

    KAI_ASM_CODE(matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm)
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
    movi v29.16b, #0x0
    movi v14.16b, #0x0
    mov x22, x8
    movi v6.16b, #0x0
    movi v12.16b, #0x0
    movi v15.16b, #0x0
    movi v1.16b, #0x0
    movi v7.16b, #0x0
    movi v4.16b, #0x0
    add x21, x23, x6
KAI_ASM_LABEL(label_3)  // Block loop
    movi v22.4s, #0x0
    movi v9.4s, #0x0
    mov x20, x7
    movi v8.4s, #0x0
    movi v5.4s, #0x0
    movi v16.4s, #0x0
    movi v10.4s, #0x0
    movi v26.4s, #0x0
    movi v25.4s, #0x0
KAI_ASM_LABEL(label_4)  // Sub block loop
    ldr q17, [x10, #0x0]
    ldr q19, [x10, #0x10]
    subs x20, x20, #0x1
    ldr q28, [x23, #0x0]
    ldr q23, [x23, #0x10]
    ldr q30, [x21, #0x0]
    ldr q18, [x21, #0x10]
    ldr q0, [x10, #0x20]
    ldr q21, [x10, #0x30]
    shl v31.16b, v17.16b, #0x4
    shl v24.16b, v19.16b, #0x4
    ldr q3, [x23, #0x20]
    ldr q2, [x23, #0x30]
    and v17.16b, v17.16b, v11.16b
    and v19.16b, v19.16b, v11.16b
    ldr q27, [x21, #0x20]
    ldr q20, [x21, #0x30]
    add x10, x10, #0x40
    ldr q13, [x23, #0x40]
    KAI_ASM_INST(0x4e9fa796)  // smmla v22.4s, v28.16b, v31.16b
    KAI_ASM_INST(0x4e98a789)  // smmla v9.4s, v28.16b, v24.16b
    ldr q28, [x23, #0x50]
    KAI_ASM_INST(0x4e9fa6e8)  // smmla v8.4s, v23.16b, v31.16b
    KAI_ASM_INST(0x4e98a6e5)  // smmla v5.4s, v23.16b, v24.16b
    ldr q23, [x21, #0x40]
    KAI_ASM_INST(0x4e9fa7d0)  // smmla v16.4s, v30.16b, v31.16b
    KAI_ASM_INST(0x4e98a7ca)  // smmla v10.4s, v30.16b, v24.16b
    ldr q30, [x21, #0x50]
    KAI_ASM_INST(0x4e9fa65a)  // smmla v26.4s, v18.16b, v31.16b
    ldr q31, [x23, #0x60]
    KAI_ASM_INST(0x4e98a659)  // smmla v25.4s, v18.16b, v24.16b
    ldr q24, [x23, #0x70]
    shl v18.16b, v0.16b, #0x4
    and v0.16b, v0.16b, v11.16b
    add x23, x23, #0x80
    KAI_ASM_INST(0x4e92a476)  // smmla v22.4s, v3.16b, v18.16b
    KAI_ASM_INST(0x4e92a448)  // smmla v8.4s, v2.16b, v18.16b
    KAI_ASM_INST(0x4e92a770)  // smmla v16.4s, v27.16b, v18.16b
    KAI_ASM_INST(0x4e92a69a)  // smmla v26.4s, v20.16b, v18.16b
    ldr q18, [x21, #0x60]
    KAI_ASM_INST(0x4e91a5b6)  // smmla v22.4s, v13.16b, v17.16b
    KAI_ASM_INST(0x4e91a788)  // smmla v8.4s, v28.16b, v17.16b
    KAI_ASM_INST(0x4e91a6f0)  // smmla v16.4s, v23.16b, v17.16b
    KAI_ASM_INST(0x4e91a7da)  // smmla v26.4s, v30.16b, v17.16b
    ldr q17, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e80a7f6)  // smmla v22.4s, v31.16b, v0.16b
    KAI_ASM_INST(0x4e80a708)  // smmla v8.4s, v24.16b, v0.16b
    KAI_ASM_INST(0x4e80a650)  // smmla v16.4s, v18.16b, v0.16b
    KAI_ASM_INST(0x4e80a63a)  // smmla v26.4s, v17.16b, v0.16b
    shl v0.16b, v21.16b, #0x4
    and v21.16b, v21.16b, v11.16b
    KAI_ASM_INST(0x4e80a469)  // smmla v9.4s, v3.16b, v0.16b
    KAI_ASM_INST(0x4e80a445)  // smmla v5.4s, v2.16b, v0.16b
    KAI_ASM_INST(0x4e80a76a)  // smmla v10.4s, v27.16b, v0.16b
    KAI_ASM_INST(0x4e80a699)  // smmla v25.4s, v20.16b, v0.16b
    KAI_ASM_INST(0x4e93a5a9)  // smmla v9.4s, v13.16b, v19.16b
    KAI_ASM_INST(0x4e93a785)  // smmla v5.4s, v28.16b, v19.16b
    KAI_ASM_INST(0x4e93a6ea)  // smmla v10.4s, v23.16b, v19.16b
    KAI_ASM_INST(0x4e93a7d9)  // smmla v25.4s, v30.16b, v19.16b
    KAI_ASM_INST(0x4e95a7e9)  // smmla v9.4s, v31.16b, v21.16b
    KAI_ASM_INST(0x4e95a705)  // smmla v5.4s, v24.16b, v21.16b
    KAI_ASM_INST(0x4e95a64a)  // smmla v10.4s, v18.16b, v21.16b
    KAI_ASM_INST(0x4e95a639)  // smmla v25.4s, v17.16b, v21.16b
    bgt label_4
    ldr q30, [x10, #0x0]
    ld1 { v17.4s }, [x23]
    add x23, x23, #0x10
    uzp1 v0.2d, v22.2d, v9.2d
    ldr q24, [x10, #0x10]
    ldr q21, [x23, #0x0]
    uzp2 v28.2d, v22.2d, v9.2d
    uzp1 v19.2d, v8.2d, v5.2d
    uzp2 v31.2d, v8.2d, v5.2d
    add x10, x10, #0x20
    add x23, x23, #0x10
    fmla v29.4s, v30.4s, v17.s[0]
    fmla v14.4s, v30.4s, v17.s[1]
    fmla v6.4s, v30.4s, v17.s[2]
    scvtf v0.4s, v0.4s
    fmla v12.4s, v30.4s, v17.s[3]
    fmul v3.4s, v24.4s, v21.s[0]
    fmul v18.4s, v24.4s, v21.s[1]
    scvtf v28.4s, v28.4s
    fmul v17.4s, v24.4s, v21.s[2]
    scvtf v19.4s, v19.4s
    fmul v8.4s, v24.4s, v21.s[3]
    scvtf v31.4s, v31.4s
    fmla v29.4s, v0.4s, v3.4s
    fmla v14.4s, v28.4s, v18.4s
    fmla v6.4s, v19.4s, v17.4s
    fmla v12.4s, v31.4s, v8.4s
    ld1 { v17.4s }, [x21]
    add x21, x21, #0x10
    uzp1 v22.2d, v16.2d, v10.2d
    uzp2 v3.2d, v16.2d, v10.2d
    ldr q16, [x21, #0x0]
    uzp1 v21.2d, v26.2d, v25.2d
    uzp2 v25.2d, v26.2d, v25.2d
    add x21, x21, #0x10
    fmla v15.4s, v30.4s, v17.s[0]
    fmla v1.4s, v30.4s, v17.s[1]
    fmla v7.4s, v30.4s, v17.s[2]
    fmla v4.4s, v30.4s, v17.s[3]
    scvtf v22.4s, v22.4s
    fmul v19.4s, v24.4s, v16.s[0]
    fmul v18.4s, v24.4s, v16.s[1]
    scvtf v3.4s, v3.4s
    fmul v17.4s, v24.4s, v16.s[2]
    scvtf v21.4s, v21.4s
    fmul v16.4s, v24.4s, v16.s[3]
    scvtf v25.4s, v25.4s
    fmla v15.4s, v22.4s, v19.4s
    fmla v1.4s, v3.4s, v18.4s
    fmla v7.4s, v21.4s, v17.4s
    fmla v4.4s, v25.4s, v16.4s
    subs x22, x22, #0x1
    bgt label_3
    ldr q18, [x10, #0x0]
    ld1r { v17.4s }, [x11]
    add x20, x11, #0x4
    cmp x9, #0x4
    ld1r { v16.4s }, [x20]
    add x10, x10, #0x10
    fadd v29.4s, v29.4s, v18.4s
    fadd v14.4s, v14.4s, v18.4s
    fadd v6.4s, v6.4s, v18.4s
    fadd v12.4s, v12.4s, v18.4s
    fadd v15.4s, v15.4s, v18.4s
    fadd v1.4s, v1.4s, v18.4s
    fadd v7.4s, v7.4s, v18.4s
    fadd v4.4s, v4.4s, v18.4s
    fmax v29.4s, v29.4s, v17.4s
    fmax v14.4s, v14.4s, v17.4s
    fmax v6.4s, v6.4s, v17.4s
    fmax v12.4s, v12.4s, v17.4s
    fmax v15.4s, v15.4s, v17.4s
    fmax v1.4s, v1.4s, v17.4s
    fmax v7.4s, v7.4s, v17.4s
    fmax v4.4s, v4.4s, v17.4s
    fmin v29.4s, v29.4s, v16.4s
    fmin v14.4s, v14.4s, v16.4s
    fmin v6.4s, v6.4s, v16.4s
    fmin v12.4s, v12.4s, v16.4s
    fmin v15.4s, v15.4s, v16.4s
    fmin v1.4s, v1.4s, v16.4s
    fmin v7.4s, v7.4s, v16.4s
    fmin v4.4s, v4.4s, v16.4s
    fcvtn v24.4h, v29.4s
    fcvtn v23.4h, v14.4s
    fcvtn v2.4h, v6.4s
    fcvtn v21.4h, v12.4s
    fcvtn v19.4h, v15.4s
    fcvtn v18.4h, v1.4s
    fcvtn v17.4h, v7.4s
    fcvtn v16.4h, v4.4s
    blt label_7
    mov x20, x13
    str d24, [x20, #0x0]
    add x20, x20, x12
    str d23, [x20, #0x0]
    add x20, x20, x12
    str d2, [x20, #0x0]
    add x20, x20, x12
    str d21, [x20, #0x0]
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
    st1 { v21.s }[0], [x20], #0x4
    st1 { v2.s }[0], [x22], #0x4
    st1 { v23.s }[0], [x21], #0x4
    st1 { v24.s }[0], [x27], #0x4
    tbz x9, #0, label_9
    st1 { v16.h }[2], [x23]
    st1 { v17.h }[2], [x25]
    st1 { v18.h }[2], [x24]
    st1 { v19.h }[2], [x26]
    st1 { v21.h }[2], [x20]
    st1 { v2.h }[2], [x22]
    st1 { v23.h }[2], [x21]
    st1 { v24.h }[2], [x27]
    b label_9
KAI_ASM_LABEL(label_8)  // Output block 0: partial_1_0
    st1 { v16.h }[0], [x23]
    st1 { v17.h }[0], [x25]
    st1 { v18.h }[0], [x24]
    st1 { v19.h }[0], [x26]
    st1 { v21.h }[0], [x20]
    st1 { v2.h }[0], [x22]
    st1 { v23.h }[0], [x21]
    st1 { v24.h }[0], [x27]
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
    movi v29.16b, #0x0
    movi v14.16b, #0x0
    mov x23, x17
    mov x21, x8
    movi v6.16b, #0x0
    movi v12.16b, #0x0
KAI_ASM_LABEL(label_14)  // Row tail: Block loop
    movi v22.4s, #0x0
    movi v9.4s, #0x0
    mov x20, x7
    movi v8.4s, #0x0
    movi v5.4s, #0x0
KAI_ASM_LABEL(label_15)  // Row tail: Sub block loop
    ldr q7, [x26, #0x0]
    ldr q31, [x26, #0x10]
    subs x20, x20, #0x1
    ldr q30, [x23, #0x0]
    ldr q15, [x23, #0x10]
    ldr q28, [x26, #0x20]
    ldr q25, [x26, #0x30]
    add x26, x26, #0x40
    ldr q20, [x23, #0x20]
    ldr q3, [x23, #0x30]
    shl v24.16b, v7.16b, #0x4
    shl v23.16b, v31.16b, #0x4
    ldr q21, [x23, #0x40]
    ldr q26, [x23, #0x50]
    and v7.16b, v7.16b, v11.16b
    and v31.16b, v31.16b, v11.16b
    ldr q19, [x23, #0x60]
    ldr q18, [x23, #0x70]
    shl v17.16b, v28.16b, #0x4
    shl v16.16b, v25.16b, #0x4
    KAI_ASM_INST(0x4e98a7d6)  // smmla v22.4s, v30.16b, v24.16b
    KAI_ASM_INST(0x4e97a7c9)  // smmla v9.4s, v30.16b, v23.16b
    and v28.16b, v28.16b, v11.16b
    add x23, x23, #0x80
    KAI_ASM_INST(0x4e98a5e8)  // smmla v8.4s, v15.16b, v24.16b
    KAI_ASM_INST(0x4e97a5e5)  // smmla v5.4s, v15.16b, v23.16b
    and v25.16b, v25.16b, v11.16b
    KAI_ASM_INST(0x4e91a696)  // smmla v22.4s, v20.16b, v17.16b
    KAI_ASM_INST(0x4e90a689)  // smmla v9.4s, v20.16b, v16.16b
    KAI_ASM_INST(0x4e91a468)  // smmla v8.4s, v3.16b, v17.16b
    KAI_ASM_INST(0x4e90a465)  // smmla v5.4s, v3.16b, v16.16b
    KAI_ASM_INST(0x4e87a6b6)  // smmla v22.4s, v21.16b, v7.16b
    KAI_ASM_INST(0x4e9fa6a9)  // smmla v9.4s, v21.16b, v31.16b
    KAI_ASM_INST(0x4e87a748)  // smmla v8.4s, v26.16b, v7.16b
    KAI_ASM_INST(0x4e9fa745)  // smmla v5.4s, v26.16b, v31.16b
    KAI_ASM_INST(0x4e9ca676)  // smmla v22.4s, v19.16b, v28.16b
    KAI_ASM_INST(0x4e99a669)  // smmla v9.4s, v19.16b, v25.16b
    KAI_ASM_INST(0x4e9ca648)  // smmla v8.4s, v18.16b, v28.16b
    KAI_ASM_INST(0x4e99a645)  // smmla v5.4s, v18.16b, v25.16b
    bgt label_15
    ldr q18, [x26, #0x0]
    ld1 { v17.4s }, [x23]
    add x23, x23, #0x10
    uzp1 v24.2d, v22.2d, v9.2d
    ldr q23, [x26, #0x10]
    ldr q16, [x23, #0x0]
    uzp2 v10.2d, v22.2d, v9.2d
    uzp1 v21.2d, v8.2d, v5.2d
    uzp2 v28.2d, v8.2d, v5.2d
    add x26, x26, #0x20
    add x23, x23, #0x10
    fmla v29.4s, v18.4s, v17.s[0]
    fmla v14.4s, v18.4s, v17.s[1]
    fmla v6.4s, v18.4s, v17.s[2]
    scvtf v24.4s, v24.4s
    fmla v12.4s, v18.4s, v17.s[3]
    fmul v19.4s, v23.4s, v16.s[0]
    fmul v18.4s, v23.4s, v16.s[1]
    scvtf v10.4s, v10.4s
    fmul v17.4s, v23.4s, v16.s[2]
    scvtf v21.4s, v21.4s
    fmul v16.4s, v23.4s, v16.s[3]
    scvtf v28.4s, v28.4s
    fmla v29.4s, v24.4s, v19.4s
    fmla v14.4s, v10.4s, v18.4s
    fmla v6.4s, v21.4s, v17.4s
    fmla v12.4s, v28.4s, v16.4s
    subs x21, x21, #0x1
    bgt label_14
    ldr q18, [x26, #0x0]
    ld1r { v17.4s }, [x11]
    add x20, x11, #0x4
    cmp x25, #0x4
    ld1r { v16.4s }, [x20]
    add x26, x26, #0x10
    fadd v29.4s, v29.4s, v18.4s
    fadd v14.4s, v14.4s, v18.4s
    fadd v6.4s, v6.4s, v18.4s
    fadd v12.4s, v12.4s, v18.4s
    fmax v29.4s, v29.4s, v17.4s
    fmax v14.4s, v14.4s, v17.4s
    fmax v6.4s, v6.4s, v17.4s
    fmax v12.4s, v12.4s, v17.4s
    fmin v29.4s, v29.4s, v16.4s
    fmin v14.4s, v14.4s, v16.4s
    fmin v6.4s, v6.4s, v16.4s
    fmin v12.4s, v12.4s, v16.4s
    fcvtn v19.4h, v29.4s
    fcvtn v18.4h, v14.4s
    fcvtn v17.4h, v6.4s
    fcvtn v16.4h, v12.4s
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm)

    KAI_ASM_END
