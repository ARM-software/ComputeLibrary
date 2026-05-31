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

    KAI_ASM_CODE(matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d10, d11, [sp, 72]
    stp d12, d13, [sp, 88]
    stp d14, d15, [sp, 104]
    stp d8, d9, [sp, 120]
    mov x7, #0x80
    movi v1.16b, #0xf0
    mov x21, #0x20
    ldr x20, [x0, #0x28]
    ldr x8, [x0, #0x38]
    ldr x17, [x0, #0x8]
    ldr x16, [x0, #0x10]
    ldr x15, [x0, #0x30]
    ldr x14, [x0, #0x0]
    mov x13, x20
    ldr x12, [x0, #0x20]
    madd x7, x8, x7, x21
    ldr x11, [x0, #0x18]
    cmp x13, #0x8
    blt label_12
KAI_ASM_LABEL(label_1)  // Row loop
    mov x10, x16
    mov x9, x15
    add x28, x14, x12, LSL #3
KAI_ASM_LABEL(label_2)  // Column loop
    mov x22, x17
    movi v12.4s, #0x0
    movi v10.4s, #0x0
    mov x21, x8
    movi v13.4s, #0x0
    movi v0.4s, #0x0
    movi v15.4s, #0x0
    movi v31.4s, #0x0
    add x20, x22, x7
    movi v14.4s, #0x0
    movi v8.4s, #0x0
    movi v9.4s, #0x0
    movi v7.4s, #0x0
    movi v16.4s, #0x0
    movi v2.4s, #0x0
    movi v20.4s, #0x0
    movi v30.4s, #0x0
    movi v5.4s, #0x0
    movi v17.4s, #0x0
KAI_ASM_LABEL(label_3)  // Sub block loop
    ldr q21, [x10, #0x0]
    ldr q3, [x10, #0x10]
    subs x21, x21, #0x1
    ldr q26, [x10, #0x20]
    ldr q4, [x10, #0x30]
    ldr q6, [x22, #0x0]
    ldr q29, [x22, #0x10]
    ldr q24, [x20, #0x0]
    ldr q27, [x20, #0x10]
    shl v23.16b, v21.16b, #0x4
    shl v19.16b, v3.16b, #0x4
    ldr q18, [x10, #0x40]
    ldr q28, [x10, #0x50]
    shl v22.16b, v26.16b, #0x4
    shl v11.16b, v4.16b, #0x4
    ldr q25, [x10, #0x60]
    and v21.16b, v21.16b, v1.16b
    and v3.16b, v3.16b, v1.16b
    KAI_ASM_INST(0x4e97a4cc)  // smmla v12.4s, v6.16b, v23.16b
    KAI_ASM_INST(0x4e93a4cd)  // smmla v13.4s, v6.16b, v19.16b
    KAI_ASM_INST(0x4e97a7af)  // smmla v15.4s, v29.16b, v23.16b
    and v26.16b, v26.16b, v1.16b
    KAI_ASM_INST(0x4e96a4ca)  // smmla v10.4s, v6.16b, v22.16b
    KAI_ASM_INST(0x4e8ba4c0)  // smmla v0.4s, v6.16b, v11.16b
    ldr q6, [x10, #0x70]
    and v4.16b, v4.16b, v1.16b
    KAI_ASM_INST(0x4e93a7ae)  // smmla v14.4s, v29.16b, v19.16b
    KAI_ASM_INST(0x4e96a7bf)  // smmla v31.4s, v29.16b, v22.16b
    add x10, x10, #0x80
    KAI_ASM_INST(0x4e8ba7a8)  // smmla v8.4s, v29.16b, v11.16b
    ldr q29, [x22, #0x20]
    KAI_ASM_INST(0x4e97a709)  // smmla v9.4s, v24.16b, v23.16b
    KAI_ASM_INST(0x4e93a710)  // smmla v16.4s, v24.16b, v19.16b
    KAI_ASM_INST(0x4e96a707)  // smmla v7.4s, v24.16b, v22.16b
    KAI_ASM_INST(0x4e8ba702)  // smmla v2.4s, v24.16b, v11.16b
    ldr q24, [x22, #0x30]
    KAI_ASM_INST(0x4e97a774)  // smmla v20.4s, v27.16b, v23.16b
    ldr q23, [x20, #0x20]
    KAI_ASM_INST(0x4e93a765)  // smmla v5.4s, v27.16b, v19.16b
    ldr q19, [x20, #0x30]
    KAI_ASM_INST(0x4e96a77e)  // smmla v30.4s, v27.16b, v22.16b
    ldr q22, [x22, #0x40]
    KAI_ASM_INST(0x4e8ba771)  // smmla v17.4s, v27.16b, v11.16b
    ldr q11, [x22, #0x50]
    shl v27.16b, v18.16b, #0x4
    and v18.16b, v18.16b, v1.16b
    KAI_ASM_INST(0x4e9ba7ac)  // smmla v12.4s, v29.16b, v27.16b
    KAI_ASM_INST(0x4e9ba70f)  // smmla v15.4s, v24.16b, v27.16b
    KAI_ASM_INST(0x4e9ba6e9)  // smmla v9.4s, v23.16b, v27.16b
    KAI_ASM_INST(0x4e9ba674)  // smmla v20.4s, v19.16b, v27.16b
    shl v27.16b, v28.16b, #0x4
    and v28.16b, v28.16b, v1.16b
    KAI_ASM_INST(0x4e9ba7ad)  // smmla v13.4s, v29.16b, v27.16b
    KAI_ASM_INST(0x4e9ba70e)  // smmla v14.4s, v24.16b, v27.16b
    KAI_ASM_INST(0x4e9ba6f0)  // smmla v16.4s, v23.16b, v27.16b
    KAI_ASM_INST(0x4e9ba665)  // smmla v5.4s, v19.16b, v27.16b
    shl v27.16b, v25.16b, #0x4
    KAI_ASM_INST(0x4e95a6cc)  // smmla v12.4s, v22.16b, v21.16b
    KAI_ASM_INST(0x4e95a56f)  // smmla v15.4s, v11.16b, v21.16b
    and v25.16b, v25.16b, v1.16b
    KAI_ASM_INST(0x4e9ba7aa)  // smmla v10.4s, v29.16b, v27.16b
    KAI_ASM_INST(0x4e9ba71f)  // smmla v31.4s, v24.16b, v27.16b
    KAI_ASM_INST(0x4e9ba6e7)  // smmla v7.4s, v23.16b, v27.16b
    KAI_ASM_INST(0x4e9ba67e)  // smmla v30.4s, v19.16b, v27.16b
    shl v27.16b, v6.16b, #0x4
    KAI_ASM_INST(0x4e83a6cd)  // smmla v13.4s, v22.16b, v3.16b
    KAI_ASM_INST(0x4e83a56e)  // smmla v14.4s, v11.16b, v3.16b
    and v6.16b, v6.16b, v1.16b
    KAI_ASM_INST(0x4e9ba7a0)  // smmla v0.4s, v29.16b, v27.16b
    ldr q29, [x20, #0x40]
    KAI_ASM_INST(0x4e9ba708)  // smmla v8.4s, v24.16b, v27.16b
    ldr q24, [x20, #0x50]
    KAI_ASM_INST(0x4e9ba6e2)  // smmla v2.4s, v23.16b, v27.16b
    ldr q23, [x22, #0x60]
    KAI_ASM_INST(0x4e9ba671)  // smmla v17.4s, v19.16b, v27.16b
    ldr q19, [x22, #0x70]
    ldr q27, [x20, #0x60]
    KAI_ASM_INST(0x4e9aa6ca)  // smmla v10.4s, v22.16b, v26.16b
    KAI_ASM_INST(0x4e9aa57f)  // smmla v31.4s, v11.16b, v26.16b
    add x22, x22, #0x80
    KAI_ASM_INST(0x4e95a7a9)  // smmla v9.4s, v29.16b, v21.16b
    KAI_ASM_INST(0x4e83a7b0)  // smmla v16.4s, v29.16b, v3.16b
    KAI_ASM_INST(0x4e84a6c0)  // smmla v0.4s, v22.16b, v4.16b
    ldr q22, [x20, #0x70]
    KAI_ASM_INST(0x4e84a568)  // smmla v8.4s, v11.16b, v4.16b
    add x20, x20, #0x80
    KAI_ASM_INST(0x4e9aa7a7)  // smmla v7.4s, v29.16b, v26.16b
    KAI_ASM_INST(0x4e84a7a2)  // smmla v2.4s, v29.16b, v4.16b
    KAI_ASM_INST(0x4e95a714)  // smmla v20.4s, v24.16b, v21.16b
    KAI_ASM_INST(0x4e83a705)  // smmla v5.4s, v24.16b, v3.16b
    KAI_ASM_INST(0x4e9aa71e)  // smmla v30.4s, v24.16b, v26.16b
    KAI_ASM_INST(0x4e84a711)  // smmla v17.4s, v24.16b, v4.16b
    KAI_ASM_INST(0x4e92a6ec)  // smmla v12.4s, v23.16b, v18.16b
    KAI_ASM_INST(0x4e9ca6ed)  // smmla v13.4s, v23.16b, v28.16b
    KAI_ASM_INST(0x4e99a6ea)  // smmla v10.4s, v23.16b, v25.16b
    KAI_ASM_INST(0x4e86a6e0)  // smmla v0.4s, v23.16b, v6.16b
    KAI_ASM_INST(0x4e92a66f)  // smmla v15.4s, v19.16b, v18.16b
    KAI_ASM_INST(0x4e9ca66e)  // smmla v14.4s, v19.16b, v28.16b
    KAI_ASM_INST(0x4e99a67f)  // smmla v31.4s, v19.16b, v25.16b
    KAI_ASM_INST(0x4e86a668)  // smmla v8.4s, v19.16b, v6.16b
    KAI_ASM_INST(0x4e92a769)  // smmla v9.4s, v27.16b, v18.16b
    KAI_ASM_INST(0x4e9ca770)  // smmla v16.4s, v27.16b, v28.16b
    KAI_ASM_INST(0x4e99a767)  // smmla v7.4s, v27.16b, v25.16b
    KAI_ASM_INST(0x4e86a762)  // smmla v2.4s, v27.16b, v6.16b
    KAI_ASM_INST(0x4e92a6d4)  // smmla v20.4s, v22.16b, v18.16b
    KAI_ASM_INST(0x4e9ca6c5)  // smmla v5.4s, v22.16b, v28.16b
    KAI_ASM_INST(0x4e99a6de)  // smmla v30.4s, v22.16b, v25.16b
    KAI_ASM_INST(0x4e86a6d1)  // smmla v17.4s, v22.16b, v6.16b
    bgt label_3
    ldr q22, [x10, #0x0]
    ldr q4, [x10, #0x10]
    uzp1 v3.2d, v12.2d, v13.2d
    uzp2 v27.2d, v12.2d, v13.2d
    ld1 { v18.4s }, [x22]
    ldr q6, [x10, #0x20]
    uzp1 v26.2d, v10.2d, v0.2d
    uzp2 v24.2d, v10.2d, v0.2d
    ldr q11, [x10, #0x30]
    uzp1 v25.2d, v15.2d, v14.2d
    uzp2 v29.2d, v15.2d, v14.2d
    add x22, x22, #0x10
    ldr q13, [x22, #0x0]
    uzp1 v28.2d, v31.2d, v8.2d
    uzp2 v23.2d, v31.2d, v8.2d
    add x10, x10, #0x40
    mla v3.4s, v22.4s, v18.s[0]
    mla v26.4s, v4.4s, v18.s[0]
    mla v27.4s, v22.4s, v18.s[1]
    mla v24.4s, v4.4s, v18.s[1]
    mla v25.4s, v22.4s, v18.s[2]
    mla v28.4s, v4.4s, v18.s[2]
    fmul v10.4s, v6.4s, v13.s[0]
    mla v29.4s, v22.4s, v18.s[3]
    mla v23.4s, v4.4s, v18.s[3]
    fmul v0.4s, v11.4s, v13.s[0]
    scvtf v3.4s, v3.4s
    scvtf v26.4s, v26.4s
    fmul v15.4s, v6.4s, v13.s[1]
    scvtf v27.4s, v27.4s
    fmul v21.4s, v11.4s, v13.s[1]
    scvtf v24.4s, v24.4s
    fmul v14.4s, v6.4s, v13.s[2]
    scvtf v25.4s, v25.4s
    fmul v31.4s, v11.4s, v13.s[2]
    scvtf v28.4s, v28.4s
    fmul v18.4s, v6.4s, v13.s[3]
    scvtf v29.4s, v29.4s
    fmul v19.4s, v11.4s, v13.s[3]
    scvtf v23.4s, v23.4s
    fmul v12.4s, v3.4s, v10.4s
    fmul v10.4s, v26.4s, v0.4s
    fmul v13.4s, v27.4s, v15.4s
    fmul v0.4s, v24.4s, v21.4s
    fmul v15.4s, v25.4s, v14.4s
    fmul v31.4s, v28.4s, v31.4s
    fmul v14.4s, v29.4s, v18.4s
    fmul v8.4s, v23.4s, v19.4s
    ld1 { v21.4s }, [x20]
    uzp1 v18.2d, v9.2d, v16.2d
    uzp2 v3.2d, v9.2d, v16.2d
    add x20, x20, #0x10
    ldr q16, [x20, #0x0]
    uzp1 v26.2d, v7.2d, v2.2d
    uzp2 v25.2d, v7.2d, v2.2d
    uzp1 v24.2d, v20.2d, v5.2d
    uzp2 v29.2d, v20.2d, v5.2d
    uzp1 v28.2d, v30.2d, v17.2d
    uzp2 v27.2d, v30.2d, v17.2d
    mla v18.4s, v22.4s, v21.s[0]
    mla v26.4s, v4.4s, v21.s[0]
    mla v3.4s, v22.4s, v21.s[1]
    fmul v23.4s, v6.4s, v16.s[0]
    mla v25.4s, v4.4s, v21.s[1]
    mla v24.4s, v22.4s, v21.s[2]
    fmul v7.4s, v11.4s, v16.s[0]
    mla v28.4s, v4.4s, v21.s[2]
    mla v29.4s, v22.4s, v21.s[3]
    fmul v22.4s, v6.4s, v16.s[1]
    mla v27.4s, v4.4s, v21.s[3]
    scvtf v18.4s, v18.4s
    scvtf v26.4s, v26.4s
    scvtf v3.4s, v3.4s
    fmul v2.4s, v11.4s, v16.s[1]
    scvtf v25.4s, v25.4s
    fmul v19.4s, v6.4s, v16.s[2]
    scvtf v24.4s, v24.4s
    fmul v30.4s, v11.4s, v16.s[2]
    scvtf v28.4s, v28.4s
    fmul v5.4s, v6.4s, v16.s[3]
    scvtf v29.4s, v29.4s
    fmul v6.4s, v11.4s, v16.s[3]
    scvtf v27.4s, v27.4s
    fmul v9.4s, v18.4s, v23.4s
    fmul v7.4s, v26.4s, v7.4s
    fmul v16.4s, v3.4s, v22.4s
    fmul v2.4s, v25.4s, v2.4s
    fmul v20.4s, v24.4s, v19.4s
    fmul v30.4s, v28.4s, v30.4s
    fmul v5.4s, v29.4s, v5.4s
    fmul v17.4s, v27.4s, v6.4s
    ldr q25, [x10, #0x0]
    ldr q18, [x10, #0x10]
    add x20, x11, #0x4
    cmp x9, #0x8
    ld1r { v24.4s }, [x11]
    ld1r { v27.4s }, [x20]
    add x10, x10, #0x20
    fadd v12.4s, v12.4s, v25.4s
    fadd v13.4s, v13.4s, v25.4s
    fadd v15.4s, v15.4s, v25.4s
    fadd v14.4s, v14.4s, v25.4s
    fadd v9.4s, v9.4s, v25.4s
    fadd v16.4s, v16.4s, v25.4s
    fadd v20.4s, v20.4s, v25.4s
    fadd v5.4s, v5.4s, v25.4s
    fadd v10.4s, v10.4s, v18.4s
    fadd v0.4s, v0.4s, v18.4s
    fadd v31.4s, v31.4s, v18.4s
    fadd v8.4s, v8.4s, v18.4s
    fadd v7.4s, v7.4s, v18.4s
    fadd v2.4s, v2.4s, v18.4s
    fadd v30.4s, v30.4s, v18.4s
    fadd v17.4s, v17.4s, v18.4s
    fmax v12.4s, v12.4s, v24.4s
    fmax v13.4s, v13.4s, v24.4s
    fmax v15.4s, v15.4s, v24.4s
    fmax v14.4s, v14.4s, v24.4s
    fmax v9.4s, v9.4s, v24.4s
    fmax v16.4s, v16.4s, v24.4s
    fmax v20.4s, v20.4s, v24.4s
    fmax v5.4s, v5.4s, v24.4s
    fmin v12.4s, v12.4s, v27.4s
    fmax v10.4s, v10.4s, v24.4s
    fmin v13.4s, v13.4s, v27.4s
    fmax v0.4s, v0.4s, v24.4s
    fmin v15.4s, v15.4s, v27.4s
    fmax v31.4s, v31.4s, v24.4s
    fmin v14.4s, v14.4s, v27.4s
    fmax v8.4s, v8.4s, v24.4s
    fmin v9.4s, v9.4s, v27.4s
    fmax v7.4s, v7.4s, v24.4s
    fmin v16.4s, v16.4s, v27.4s
    fmax v2.4s, v2.4s, v24.4s
    fmin v20.4s, v20.4s, v27.4s
    fmax v30.4s, v30.4s, v24.4s
    fmin v5.4s, v5.4s, v27.4s
    fmax v17.4s, v17.4s, v24.4s
    fmin v10.4s, v10.4s, v27.4s
    fmin v0.4s, v0.4s, v27.4s
    fmin v31.4s, v31.4s, v27.4s
    fmin v8.4s, v8.4s, v27.4s
    fmin v7.4s, v7.4s, v27.4s
    fmin v2.4s, v2.4s, v27.4s
    fmin v30.4s, v30.4s, v27.4s
    fmin v17.4s, v17.4s, v27.4s
    KAI_ASM_INST(0x0ea16997)  // bfcvtn v23.4h, v12.4s
    KAI_ASM_INST(0x0ea169b6)  // bfcvtn v22.4h, v13.4s
    KAI_ASM_INST(0x0ea169e6)  // bfcvtn v6.4h, v15.4s
    KAI_ASM_INST(0x0ea169cc)  // bfcvtn v12.4h, v14.4s
    KAI_ASM_INST(0x0ea16939)  // bfcvtn v25.4h, v9.4s
    KAI_ASM_INST(0x0ea16a12)  // bfcvtn v18.4h, v16.4s
    KAI_ASM_INST(0x0ea16a8e)  // bfcvtn v14.4h, v20.4s
    KAI_ASM_INST(0x0ea168a9)  // bfcvtn v9.4h, v5.4s
    KAI_ASM_INST(0x4ea16957)  // bfcvtn2 v23.8h, v10.4s
    KAI_ASM_INST(0x4ea16816)  // bfcvtn2 v22.8h, v0.4s
    KAI_ASM_INST(0x4ea16be6)  // bfcvtn2 v6.8h, v31.4s
    KAI_ASM_INST(0x4ea1690c)  // bfcvtn2 v12.8h, v8.4s
    KAI_ASM_INST(0x4ea168f9)  // bfcvtn2 v25.8h, v7.4s
    KAI_ASM_INST(0x4ea16852)  // bfcvtn2 v18.8h, v2.4s
    KAI_ASM_INST(0x4ea16bce)  // bfcvtn2 v14.8h, v30.4s
    KAI_ASM_INST(0x4ea16a29)  // bfcvtn2 v9.8h, v17.4s
    blt label_6
    mov x20, x14
    str q23, [x20, #0x0]
    add x20, x20, x12
    str q22, [x20, #0x0]
    add x20, x20, x12
    str q6, [x20, #0x0]
    add x20, x20, x12
    str q12, [x20, #0x0]
    add x20, x20, x12
    str q25, [x20, #0x0]
    add x20, x20, x12
    str q18, [x20, #0x0]
    add x20, x20, x12
    str q14, [x20, #0x0]
    add x20, x20, x12
    str q9, [x20, #0x0]
    b label_11
KAI_ASM_LABEL(label_6)  // Partial output
    mov x27, x14
    add x26, x27, x12, LSL #2
    add x25, x26, x12, LSL #1
    add x24, x26, x12
    add x23, x25, x12
    add x22, x27, x12, LSL #1
    add x21, x27, x12
    add x20, x22, x12
    tbz x9, #2, label_8
    st1 { v9.d }[0], [x23], #0x8
    st1 { v14.d }[0], [x25], #0x8
    st1 { v18.d }[0], [x24], #0x8
    st1 { v25.d }[0], [x26], #0x8
    st1 { v12.d }[0], [x20], #0x8
    st1 { v6.d }[0], [x22], #0x8
    st1 { v22.d }[0], [x21], #0x8
    st1 { v23.d }[0], [x27], #0x8
    tbz x9, #1, label_7
    st1 { v9.s }[2], [x23], #0x4
    st1 { v14.s }[2], [x25], #0x4
    st1 { v18.s }[2], [x24], #0x4
    st1 { v25.s }[2], [x26], #0x4
    st1 { v12.s }[2], [x20], #0x4
    st1 { v6.s }[2], [x22], #0x4
    st1 { v22.s }[2], [x21], #0x4
    st1 { v23.s }[2], [x27], #0x4
    tbz x9, #0, label_10
    st1 { v9.h }[6], [x23]
    st1 { v14.h }[6], [x25]
    st1 { v18.h }[6], [x24]
    st1 { v25.h }[6], [x26]
    st1 { v12.h }[6], [x20]
    st1 { v6.h }[6], [x22]
    st1 { v22.h }[6], [x21]
    st1 { v23.h }[6], [x27]
    b label_10
KAI_ASM_LABEL(label_7)  // Output block 0: partial_1_4
    tbz x9, #0, label_10
    st1 { v9.h }[4], [x23]
    st1 { v14.h }[4], [x25]
    st1 { v18.h }[4], [x24]
    st1 { v25.h }[4], [x26]
    st1 { v12.h }[4], [x20]
    st1 { v6.h }[4], [x22]
    st1 { v22.h }[4], [x21]
    st1 { v23.h }[4], [x27]
    b label_10
KAI_ASM_LABEL(label_8)  // Output block 0: partial_2_0
    tbz x9, #1, label_9
    st1 { v9.s }[0], [x23], #0x4
    st1 { v14.s }[0], [x25], #0x4
    st1 { v18.s }[0], [x24], #0x4
    st1 { v25.s }[0], [x26], #0x4
    st1 { v12.s }[0], [x20], #0x4
    st1 { v6.s }[0], [x22], #0x4
    st1 { v22.s }[0], [x21], #0x4
    st1 { v23.s }[0], [x27], #0x4
    tbz x9, #0, label_10
    st1 { v9.h }[2], [x23]
    st1 { v14.h }[2], [x25]
    st1 { v18.h }[2], [x24]
    st1 { v25.h }[2], [x26]
    st1 { v12.h }[2], [x20]
    st1 { v6.h }[2], [x22]
    st1 { v22.h }[2], [x21]
    st1 { v23.h }[2], [x27]
    b label_10
KAI_ASM_LABEL(label_9)  // Output block 0: partial_1_0
    st1 { v9.h }[0], [x23]
    st1 { v14.h }[0], [x25]
    st1 { v18.h }[0], [x24]
    st1 { v25.h }[0], [x26]
    st1 { v12.h }[0], [x20]
    st1 { v6.h }[0], [x22]
    st1 { v22.h }[0], [x21]
    st1 { v23.h }[0], [x27]
KAI_ASM_LABEL(label_10)  // Output block 0: Done
KAI_ASM_LABEL(label_11)  // Output stage exit
    subs x9, x9, #0x8
    add x14, x14, #0x10
    bgt label_2
    mov x20, #0x2
    sub x13, x13, #0x8
    cmp x13, #0x8
    mov x14, x28
    madd x17, x20, x7, x17
    bge label_1
KAI_ASM_LABEL(label_12)  // Row loop skip
    cbz x13, label_23
KAI_ASM_LABEL(label_13)  // Row tail: Row loop
    mov x26, x16
    mov x25, x15
    add x24, x14, x12, LSL #2
KAI_ASM_LABEL(label_14)  // Row tail: Column loop
    mov x22, x17
    movi v12.4s, #0x0
    movi v10.4s, #0x0
    mov x20, x8
    movi v13.4s, #0x0
    movi v0.4s, #0x0
    movi v15.4s, #0x0
    movi v31.4s, #0x0
    movi v14.4s, #0x0
    movi v8.4s, #0x0
KAI_ASM_LABEL(label_15)  // Row tail: Sub block loop
    ldr q7, [x26, #0x0]
    ldr q6, [x26, #0x10]
    subs x20, x20, #0x1
    ldr q5, [x26, #0x20]
    ldr q4, [x26, #0x30]
    ldr q3, [x22, #0x0]
    ldr q2, [x22, #0x10]
    ldr q21, [x26, #0x40]
    ldr q9, [x26, #0x50]
    shl v19.16b, v7.16b, #0x4
    shl v18.16b, v6.16b, #0x4
    ldr q28, [x26, #0x60]
    ldr q26, [x26, #0x70]
    shl v29.16b, v5.16b, #0x4
    shl v30.16b, v4.16b, #0x4
    ldr q25, [x22, #0x20]
    ldr q24, [x22, #0x30]
    and v7.16b, v7.16b, v1.16b
    and v6.16b, v6.16b, v1.16b
    ldr q23, [x22, #0x40]
    ldr q22, [x22, #0x50]
    KAI_ASM_INST(0x4e93a46c)  // smmla v12.4s, v3.16b, v19.16b
    KAI_ASM_INST(0x4e92a46d)  // smmla v13.4s, v3.16b, v18.16b
    ldr q11, [x22, #0x60]
    ldr q20, [x22, #0x70]
    KAI_ASM_INST(0x4e9da46a)  // smmla v10.4s, v3.16b, v29.16b
    KAI_ASM_INST(0x4e9ea460)  // smmla v0.4s, v3.16b, v30.16b
    KAI_ASM_INST(0x4e93a44f)  // smmla v15.4s, v2.16b, v19.16b
    KAI_ASM_INST(0x4e92a44e)  // smmla v14.4s, v2.16b, v18.16b
    shl v19.16b, v21.16b, #0x4
    add x26, x26, #0x80
    KAI_ASM_INST(0x4e9da45f)  // smmla v31.4s, v2.16b, v29.16b
    KAI_ASM_INST(0x4e9ea448)  // smmla v8.4s, v2.16b, v30.16b
    shl v18.16b, v9.16b, #0x4
    add x22, x22, #0x80
    shl v27.16b, v28.16b, #0x4
    shl v29.16b, v26.16b, #0x4
    KAI_ASM_INST(0x4e93a72c)  // smmla v12.4s, v25.16b, v19.16b
    and v5.16b, v5.16b, v1.16b
    and v4.16b, v4.16b, v1.16b
    KAI_ASM_INST(0x4e92a72d)  // smmla v13.4s, v25.16b, v18.16b
    KAI_ASM_INST(0x4e93a70f)  // smmla v15.4s, v24.16b, v19.16b
    KAI_ASM_INST(0x4e92a70e)  // smmla v14.4s, v24.16b, v18.16b
    and v21.16b, v21.16b, v1.16b
    KAI_ASM_INST(0x4e9ba72a)  // smmla v10.4s, v25.16b, v27.16b
    KAI_ASM_INST(0x4e9da720)  // smmla v0.4s, v25.16b, v29.16b
    and v9.16b, v9.16b, v1.16b
    KAI_ASM_INST(0x4e9ba71f)  // smmla v31.4s, v24.16b, v27.16b
    KAI_ASM_INST(0x4e9da708)  // smmla v8.4s, v24.16b, v29.16b
    and v28.16b, v28.16b, v1.16b
    KAI_ASM_INST(0x4e87a6ec)  // smmla v12.4s, v23.16b, v7.16b
    KAI_ASM_INST(0x4e86a6ed)  // smmla v13.4s, v23.16b, v6.16b
    and v26.16b, v26.16b, v1.16b
    KAI_ASM_INST(0x4e87a6cf)  // smmla v15.4s, v22.16b, v7.16b
    KAI_ASM_INST(0x4e86a6ce)  // smmla v14.4s, v22.16b, v6.16b
    KAI_ASM_INST(0x4e85a6ea)  // smmla v10.4s, v23.16b, v5.16b
    KAI_ASM_INST(0x4e84a6e0)  // smmla v0.4s, v23.16b, v4.16b
    KAI_ASM_INST(0x4e85a6df)  // smmla v31.4s, v22.16b, v5.16b
    KAI_ASM_INST(0x4e84a6c8)  // smmla v8.4s, v22.16b, v4.16b
    KAI_ASM_INST(0x4e95a56c)  // smmla v12.4s, v11.16b, v21.16b
    KAI_ASM_INST(0x4e89a56d)  // smmla v13.4s, v11.16b, v9.16b
    KAI_ASM_INST(0x4e95a68f)  // smmla v15.4s, v20.16b, v21.16b
    KAI_ASM_INST(0x4e89a68e)  // smmla v14.4s, v20.16b, v9.16b
    KAI_ASM_INST(0x4e9ca56a)  // smmla v10.4s, v11.16b, v28.16b
    KAI_ASM_INST(0x4e9aa560)  // smmla v0.4s, v11.16b, v26.16b
    KAI_ASM_INST(0x4e9ca69f)  // smmla v31.4s, v20.16b, v28.16b
    KAI_ASM_INST(0x4e9aa688)  // smmla v8.4s, v20.16b, v26.16b
    bgt label_15
    ldr q20, [x26, #0x0]
    ldr q19, [x26, #0x10]
    uzp1 v3.2d, v12.2d, v13.2d
    uzp2 v2.2d, v12.2d, v13.2d
    ld1 { v18.4s }, [x22]
    ldr q29, [x26, #0x20]
    uzp1 v13.2d, v10.2d, v0.2d
    uzp2 v4.2d, v10.2d, v0.2d
    ldr q21, [x26, #0x30]
    uzp1 v28.2d, v15.2d, v14.2d
    uzp2 v7.2d, v15.2d, v14.2d
    add x22, x22, #0x10
    ldr q9, [x22, #0x0]
    uzp1 v25.2d, v31.2d, v8.2d
    uzp2 v24.2d, v31.2d, v8.2d
    add x26, x26, #0x40
    mla v3.4s, v20.4s, v18.s[0]
    mla v13.4s, v19.4s, v18.s[0]
    mla v2.4s, v20.4s, v18.s[1]
    mla v4.4s, v19.4s, v18.s[1]
    mla v28.4s, v20.4s, v18.s[2]
    mla v25.4s, v19.4s, v18.s[2]
    fmul v23.4s, v29.4s, v9.s[0]
    mla v7.4s, v20.4s, v18.s[3]
    mla v24.4s, v19.4s, v18.s[3]
    fmul v22.4s, v21.4s, v9.s[0]
    scvtf v3.4s, v3.4s
    scvtf v13.4s, v13.4s
    fmul v30.4s, v29.4s, v9.s[1]
    scvtf v2.4s, v2.4s
    fmul v20.4s, v21.4s, v9.s[1]
    scvtf v4.4s, v4.4s
    fmul v19.4s, v29.4s, v9.s[2]
    scvtf v28.4s, v28.4s
    fmul v18.4s, v21.4s, v9.s[2]
    scvtf v25.4s, v25.4s
    fmul v27.4s, v29.4s, v9.s[3]
    scvtf v7.4s, v7.4s
    fmul v26.4s, v21.4s, v9.s[3]
    scvtf v24.4s, v24.4s
    fmul v12.4s, v3.4s, v23.4s
    fmul v10.4s, v13.4s, v22.4s
    fmul v13.4s, v2.4s, v30.4s
    fmul v0.4s, v4.4s, v20.4s
    fmul v15.4s, v28.4s, v19.4s
    fmul v31.4s, v25.4s, v18.4s
    fmul v14.4s, v7.4s, v27.4s
    fmul v8.4s, v24.4s, v26.4s
    ldr q19, [x26, #0x0]
    ldr q18, [x26, #0x10]
    add x20, x11, #0x4
    cmp x25, #0x8
    ld1r { v22.4s }, [x11]
    ld1r { v30.4s }, [x20]
    add x26, x26, #0x20
    fadd v12.4s, v12.4s, v19.4s
    fadd v13.4s, v13.4s, v19.4s
    fadd v15.4s, v15.4s, v19.4s
    fadd v14.4s, v14.4s, v19.4s
    fadd v10.4s, v10.4s, v18.4s
    fadd v0.4s, v0.4s, v18.4s
    fadd v31.4s, v31.4s, v18.4s
    fadd v8.4s, v8.4s, v18.4s
    fmax v12.4s, v12.4s, v22.4s
    fmax v13.4s, v13.4s, v22.4s
    fmax v15.4s, v15.4s, v22.4s
    fmax v14.4s, v14.4s, v22.4s
    fmax v10.4s, v10.4s, v22.4s
    fmin v12.4s, v12.4s, v30.4s
    fmax v0.4s, v0.4s, v22.4s
    fmin v13.4s, v13.4s, v30.4s
    fmin v15.4s, v15.4s, v30.4s
    fmax v31.4s, v31.4s, v22.4s
    fmin v14.4s, v14.4s, v30.4s
    fmax v8.4s, v8.4s, v22.4s
    fmin v10.4s, v10.4s, v30.4s
    fmin v0.4s, v0.4s, v30.4s
    KAI_ASM_INST(0x0ea16993)  // bfcvtn v19.4h, v12.4s
    fmin v31.4s, v31.4s, v30.4s
    KAI_ASM_INST(0x0ea169b2)  // bfcvtn v18.4h, v13.4s
    fmin v8.4s, v8.4s, v30.4s
    KAI_ASM_INST(0x0ea169e5)  // bfcvtn v5.4h, v15.4s
    KAI_ASM_INST(0x0ea169c3)  // bfcvtn v3.4h, v14.4s
    KAI_ASM_INST(0x4ea16953)  // bfcvtn2 v19.8h, v10.4s
    KAI_ASM_INST(0x4ea16812)  // bfcvtn2 v18.8h, v0.4s
    KAI_ASM_INST(0x4ea16be5)  // bfcvtn2 v5.8h, v31.4s
    KAI_ASM_INST(0x4ea16903)  // bfcvtn2 v3.8h, v8.4s
    blt label_17
    mov x20, x14
    cmp x13, #0x1
    str q19, [x20, #0x0]
    add x20, x20, x12
    ble label_22
    cmp x13, #0x2
    str q18, [x20, #0x0]
    add x20, x20, x12
    ble label_22
    cmp x13, #0x3
    str q5, [x20, #0x0]
    add x20, x20, x12
    ble label_22
    str q3, [x20, #0x0]
    b label_22
KAI_ASM_LABEL(label_17)  // Row tail: Partial output
    mov x23, x14
    cmp x13, #0x1
    add x22, x23, x12
    csel x22, x22, x23, GT
    cmp x13, #0x2
    add x21, x23, x12, LSL #1
    csel x21, x21, x22, GT
    cmp x13, #0x3
    add x20, x21, x12
    csel x20, x20, x21, GT
    tbz x25, #2, label_19
    st1 { v3.d }[0], [x20], #0x8
    st1 { v5.d }[0], [x21], #0x8
    st1 { v18.d }[0], [x22], #0x8
    st1 { v19.d }[0], [x23], #0x8
    tbz x25, #1, label_18
    st1 { v3.s }[2], [x20], #0x4
    st1 { v5.s }[2], [x21], #0x4
    st1 { v18.s }[2], [x22], #0x4
    st1 { v19.s }[2], [x23], #0x4
    tbz x25, #0, label_21
    st1 { v3.h }[6], [x20]
    st1 { v5.h }[6], [x21]
    st1 { v18.h }[6], [x22]
    st1 { v19.h }[6], [x23]
    b label_21
KAI_ASM_LABEL(label_18)  // Row tail: Output block 0: partial_1_4
    tbz x25, #0, label_21
    st1 { v3.h }[4], [x20]
    st1 { v5.h }[4], [x21]
    st1 { v18.h }[4], [x22]
    st1 { v19.h }[4], [x23]
    b label_21
KAI_ASM_LABEL(label_19)  // Row tail: Output block 0: partial_2_0
    tbz x25, #1, label_20
    st1 { v3.s }[0], [x20], #0x4
    st1 { v5.s }[0], [x21], #0x4
    st1 { v18.s }[0], [x22], #0x4
    st1 { v19.s }[0], [x23], #0x4
    tbz x25, #0, label_21
    st1 { v3.h }[2], [x20]
    st1 { v5.h }[2], [x21]
    st1 { v18.h }[2], [x22]
    st1 { v19.h }[2], [x23]
    b label_21
KAI_ASM_LABEL(label_20)  // Row tail: Output block 0: partial_1_0
    st1 { v3.h }[0], [x20]
    st1 { v5.h }[0], [x21]
    st1 { v18.h }[0], [x22]
    st1 { v19.h }[0], [x23]
KAI_ASM_LABEL(label_21)  // Row tail: Output block 0: Done
KAI_ASM_LABEL(label_22)  // Row tail: Output stage exit
    subs x25, x25, #0x8
    add x14, x14, #0x10
    bgt label_14
    subs x13, x13, #0x4
    add x17, x17, x7
    mov x14, x24
    bgt label_13
KAI_ASM_LABEL(label_23)  // Row tail: Row loop skip
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_bf16_qai8dxp4x8_qsi4cxp8x8_8x8_neon_i8mm)

    KAI_ASM_END
