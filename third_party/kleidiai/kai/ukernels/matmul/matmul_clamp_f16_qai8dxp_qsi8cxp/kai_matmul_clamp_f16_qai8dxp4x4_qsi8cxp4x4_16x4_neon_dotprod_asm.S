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

    KAI_ASM_CODE(matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod)
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
    blt label_14
KAI_ASM_LABEL(label_1)  // Row loop
    mov x11, x17
    mov x10, x16
    add x9, x15, x13, LSL #4
KAI_ASM_LABEL(label_2)  // Column loop
    mov x27, x8
    movi v31.4s, #0x0
    movi v30.4s, #0x0
    mov x23, x7
    movi v29.4s, #0x0
    movi v28.4s, #0x0
    movi v27.4s, #0x0
    movi v26.4s, #0x0
    add x22, x27, x6
    add x21, x22, x6
    add x20, x21, x6
    movi v25.4s, #0x0
    movi v24.4s, #0x0
    movi v23.4s, #0x0
    movi v22.4s, #0x0
    movi v21.4s, #0x0
    movi v20.4s, #0x0
    movi v19.4s, #0x0
    movi v18.4s, #0x0
    movi v17.4s, #0x0
    movi v16.4s, #0x0
KAI_ASM_LABEL(label_3)  // Sub block loop
    ldr q15, [x11, #0x0]
    ldr q7, [x27, #0x0]
    subs x23, x23, #0x1
    ldr q5, [x22, #0x0]
    ldr q6, [x21, #0x0]
    ldr q4, [x20, #0x0]
    ldr q14, [x11, #0x10]
    ldr q3, [x27, #0x10]
    ldr q2, [x22, #0x10]
    KAI_ASM_INST(0x4f87e1ff)  // sdot v31.4s, v15.16b, v7.4b[0]
    KAI_ASM_INST(0x4fa7e1fe)  // sdot v30.4s, v15.16b, v7.4b[1]
    ldr q1, [x21, #0x10]
    ldr q0, [x20, #0x10]
    KAI_ASM_INST(0x4f87e9fd)  // sdot v29.4s, v15.16b, v7.4b[2]
    KAI_ASM_INST(0x4fa7e9fc)  // sdot v28.4s, v15.16b, v7.4b[3]
    ldr q10, [x11, #0x20]
    ldr q13, [x27, #0x20]
    KAI_ASM_INST(0x4f85e1fb)  // sdot v27.4s, v15.16b, v5.4b[0]
    KAI_ASM_INST(0x4fa5e1fa)  // sdot v26.4s, v15.16b, v5.4b[1]
    ldr q12, [x22, #0x20]
    ldr q11, [x21, #0x20]
    KAI_ASM_INST(0x4f85e9f9)  // sdot v25.4s, v15.16b, v5.4b[2]
    KAI_ASM_INST(0x4fa5e9f8)  // sdot v24.4s, v15.16b, v5.4b[3]
    ldr q9, [x20, #0x20]
    ldr q5, [x11, #0x30]
    KAI_ASM_INST(0x4f86e1f7)  // sdot v23.4s, v15.16b, v6.4b[0]
    KAI_ASM_INST(0x4fa6e1f6)  // sdot v22.4s, v15.16b, v6.4b[1]
    ldr q8, [x27, #0x30]
    ldr q7, [x22, #0x30]
    KAI_ASM_INST(0x4f86e9f5)  // sdot v21.4s, v15.16b, v6.4b[2]
    KAI_ASM_INST(0x4fa6e9f4)  // sdot v20.4s, v15.16b, v6.4b[3]
    ldr q6, [x21, #0x30]
    KAI_ASM_INST(0x4f84e1f3)  // sdot v19.4s, v15.16b, v4.4b[0]
    KAI_ASM_INST(0x4fa4e1f2)  // sdot v18.4s, v15.16b, v4.4b[1]
    KAI_ASM_INST(0x4f84e9f1)  // sdot v17.4s, v15.16b, v4.4b[2]
    KAI_ASM_INST(0x4fa4e9f0)  // sdot v16.4s, v15.16b, v4.4b[3]
    ldr q4, [x20, #0x30]
    ldr q15, [x11, #0x40]
    KAI_ASM_INST(0x4f83e1df)  // sdot v31.4s, v14.16b, v3.4b[0]
    KAI_ASM_INST(0x4fa3e1de)  // sdot v30.4s, v14.16b, v3.4b[1]
    KAI_ASM_INST(0x4f83e9dd)  // sdot v29.4s, v14.16b, v3.4b[2]
    KAI_ASM_INST(0x4fa3e9dc)  // sdot v28.4s, v14.16b, v3.4b[3]
    ldr q3, [x27, #0x40]
    KAI_ASM_INST(0x4f82e1db)  // sdot v27.4s, v14.16b, v2.4b[0]
    KAI_ASM_INST(0x4fa2e1da)  // sdot v26.4s, v14.16b, v2.4b[1]
    KAI_ASM_INST(0x4f82e9d9)  // sdot v25.4s, v14.16b, v2.4b[2]
    KAI_ASM_INST(0x4fa2e9d8)  // sdot v24.4s, v14.16b, v2.4b[3]
    ldr q2, [x22, #0x40]
    KAI_ASM_INST(0x4f81e1d7)  // sdot v23.4s, v14.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e1d6)  // sdot v22.4s, v14.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e9d5)  // sdot v21.4s, v14.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e9d4)  // sdot v20.4s, v14.16b, v1.4b[3]
    ldr q1, [x21, #0x40]
    KAI_ASM_INST(0x4f80e1d3)  // sdot v19.4s, v14.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e1d2)  // sdot v18.4s, v14.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e9d1)  // sdot v17.4s, v14.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e9d0)  // sdot v16.4s, v14.16b, v0.4b[3]
    ldr q0, [x20, #0x40]
    ldr q14, [x11, #0x50]
    KAI_ASM_INST(0x4f8de15f)  // sdot v31.4s, v10.16b, v13.4b[0]
    KAI_ASM_INST(0x4fade15e)  // sdot v30.4s, v10.16b, v13.4b[1]
    KAI_ASM_INST(0x4f8de95d)  // sdot v29.4s, v10.16b, v13.4b[2]
    KAI_ASM_INST(0x4fade95c)  // sdot v28.4s, v10.16b, v13.4b[3]
    ldr q13, [x27, #0x50]
    KAI_ASM_INST(0x4f8ce15b)  // sdot v27.4s, v10.16b, v12.4b[0]
    KAI_ASM_INST(0x4face15a)  // sdot v26.4s, v10.16b, v12.4b[1]
    KAI_ASM_INST(0x4f8ce959)  // sdot v25.4s, v10.16b, v12.4b[2]
    KAI_ASM_INST(0x4face958)  // sdot v24.4s, v10.16b, v12.4b[3]
    ldr q12, [x22, #0x50]
    KAI_ASM_INST(0x4f8be157)  // sdot v23.4s, v10.16b, v11.4b[0]
    KAI_ASM_INST(0x4fabe156)  // sdot v22.4s, v10.16b, v11.4b[1]
    KAI_ASM_INST(0x4f8be955)  // sdot v21.4s, v10.16b, v11.4b[2]
    KAI_ASM_INST(0x4fabe954)  // sdot v20.4s, v10.16b, v11.4b[3]
    ldr q11, [x21, #0x50]
    KAI_ASM_INST(0x4f89e153)  // sdot v19.4s, v10.16b, v9.4b[0]
    KAI_ASM_INST(0x4fa9e152)  // sdot v18.4s, v10.16b, v9.4b[1]
    KAI_ASM_INST(0x4f89e951)  // sdot v17.4s, v10.16b, v9.4b[2]
    KAI_ASM_INST(0x4fa9e950)  // sdot v16.4s, v10.16b, v9.4b[3]
    ldr q10, [x20, #0x50]
    ldr q9, [x11, #0x60]
    KAI_ASM_INST(0x4f88e0bf)  // sdot v31.4s, v5.16b, v8.4b[0]
    KAI_ASM_INST(0x4fa8e0be)  // sdot v30.4s, v5.16b, v8.4b[1]
    KAI_ASM_INST(0x4f88e8bd)  // sdot v29.4s, v5.16b, v8.4b[2]
    KAI_ASM_INST(0x4fa8e8bc)  // sdot v28.4s, v5.16b, v8.4b[3]
    ldr q8, [x27, #0x60]
    KAI_ASM_INST(0x4f87e0bb)  // sdot v27.4s, v5.16b, v7.4b[0]
    KAI_ASM_INST(0x4fa7e0ba)  // sdot v26.4s, v5.16b, v7.4b[1]
    KAI_ASM_INST(0x4f87e8b9)  // sdot v25.4s, v5.16b, v7.4b[2]
    KAI_ASM_INST(0x4fa7e8b8)  // sdot v24.4s, v5.16b, v7.4b[3]
    ldr q7, [x22, #0x60]
    KAI_ASM_INST(0x4f86e0b7)  // sdot v23.4s, v5.16b, v6.4b[0]
    KAI_ASM_INST(0x4fa6e0b6)  // sdot v22.4s, v5.16b, v6.4b[1]
    KAI_ASM_INST(0x4f86e8b5)  // sdot v21.4s, v5.16b, v6.4b[2]
    KAI_ASM_INST(0x4fa6e8b4)  // sdot v20.4s, v5.16b, v6.4b[3]
    ldr q6, [x21, #0x60]
    KAI_ASM_INST(0x4f84e0b3)  // sdot v19.4s, v5.16b, v4.4b[0]
    KAI_ASM_INST(0x4fa4e0b2)  // sdot v18.4s, v5.16b, v4.4b[1]
    KAI_ASM_INST(0x4f84e8b1)  // sdot v17.4s, v5.16b, v4.4b[2]
    KAI_ASM_INST(0x4fa4e8b0)  // sdot v16.4s, v5.16b, v4.4b[3]
    ldr q5, [x20, #0x60]
    ldr q4, [x11, #0x70]
    KAI_ASM_INST(0x4f83e1ff)  // sdot v31.4s, v15.16b, v3.4b[0]
    KAI_ASM_INST(0x4fa3e1fe)  // sdot v30.4s, v15.16b, v3.4b[1]
    add x11, x11, #0x80
    KAI_ASM_INST(0x4f83e9fd)  // sdot v29.4s, v15.16b, v3.4b[2]
    KAI_ASM_INST(0x4fa3e9fc)  // sdot v28.4s, v15.16b, v3.4b[3]
    ldr q3, [x27, #0x70]
    add x27, x27, #0x80
    KAI_ASM_INST(0x4f82e1fb)  // sdot v27.4s, v15.16b, v2.4b[0]
    KAI_ASM_INST(0x4fa2e1fa)  // sdot v26.4s, v15.16b, v2.4b[1]
    KAI_ASM_INST(0x4f82e9f9)  // sdot v25.4s, v15.16b, v2.4b[2]
    KAI_ASM_INST(0x4fa2e9f8)  // sdot v24.4s, v15.16b, v2.4b[3]
    ldr q2, [x22, #0x70]
    add x22, x22, #0x80
    KAI_ASM_INST(0x4f81e1f7)  // sdot v23.4s, v15.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e1f6)  // sdot v22.4s, v15.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e9f5)  // sdot v21.4s, v15.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e9f4)  // sdot v20.4s, v15.16b, v1.4b[3]
    ldr q1, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4f80e1f3)  // sdot v19.4s, v15.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e1f2)  // sdot v18.4s, v15.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e9f1)  // sdot v17.4s, v15.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e9f0)  // sdot v16.4s, v15.16b, v0.4b[3]
    ldr q0, [x20, #0x70]
    add x20, x20, #0x80
    KAI_ASM_INST(0x4f8de1df)  // sdot v31.4s, v14.16b, v13.4b[0]
    KAI_ASM_INST(0x4fade1de)  // sdot v30.4s, v14.16b, v13.4b[1]
    KAI_ASM_INST(0x4f8de9dd)  // sdot v29.4s, v14.16b, v13.4b[2]
    KAI_ASM_INST(0x4fade9dc)  // sdot v28.4s, v14.16b, v13.4b[3]
    KAI_ASM_INST(0x4f8ce1db)  // sdot v27.4s, v14.16b, v12.4b[0]
    KAI_ASM_INST(0x4face1da)  // sdot v26.4s, v14.16b, v12.4b[1]
    KAI_ASM_INST(0x4f8ce9d9)  // sdot v25.4s, v14.16b, v12.4b[2]
    KAI_ASM_INST(0x4face9d8)  // sdot v24.4s, v14.16b, v12.4b[3]
    KAI_ASM_INST(0x4f8be1d7)  // sdot v23.4s, v14.16b, v11.4b[0]
    KAI_ASM_INST(0x4fabe1d6)  // sdot v22.4s, v14.16b, v11.4b[1]
    KAI_ASM_INST(0x4f8be9d5)  // sdot v21.4s, v14.16b, v11.4b[2]
    KAI_ASM_INST(0x4fabe9d4)  // sdot v20.4s, v14.16b, v11.4b[3]
    KAI_ASM_INST(0x4f8ae1d3)  // sdot v19.4s, v14.16b, v10.4b[0]
    KAI_ASM_INST(0x4faae1d2)  // sdot v18.4s, v14.16b, v10.4b[1]
    KAI_ASM_INST(0x4f8ae9d1)  // sdot v17.4s, v14.16b, v10.4b[2]
    KAI_ASM_INST(0x4faae9d0)  // sdot v16.4s, v14.16b, v10.4b[3]
    KAI_ASM_INST(0x4f88e13f)  // sdot v31.4s, v9.16b, v8.4b[0]
    KAI_ASM_INST(0x4fa8e13e)  // sdot v30.4s, v9.16b, v8.4b[1]
    KAI_ASM_INST(0x4f88e93d)  // sdot v29.4s, v9.16b, v8.4b[2]
    KAI_ASM_INST(0x4fa8e93c)  // sdot v28.4s, v9.16b, v8.4b[3]
    KAI_ASM_INST(0x4f87e13b)  // sdot v27.4s, v9.16b, v7.4b[0]
    KAI_ASM_INST(0x4fa7e13a)  // sdot v26.4s, v9.16b, v7.4b[1]
    KAI_ASM_INST(0x4f87e939)  // sdot v25.4s, v9.16b, v7.4b[2]
    KAI_ASM_INST(0x4fa7e938)  // sdot v24.4s, v9.16b, v7.4b[3]
    KAI_ASM_INST(0x4f86e137)  // sdot v23.4s, v9.16b, v6.4b[0]
    KAI_ASM_INST(0x4fa6e136)  // sdot v22.4s, v9.16b, v6.4b[1]
    KAI_ASM_INST(0x4f86e935)  // sdot v21.4s, v9.16b, v6.4b[2]
    KAI_ASM_INST(0x4fa6e934)  // sdot v20.4s, v9.16b, v6.4b[3]
    KAI_ASM_INST(0x4f85e133)  // sdot v19.4s, v9.16b, v5.4b[0]
    KAI_ASM_INST(0x4fa5e132)  // sdot v18.4s, v9.16b, v5.4b[1]
    KAI_ASM_INST(0x4f85e931)  // sdot v17.4s, v9.16b, v5.4b[2]
    KAI_ASM_INST(0x4fa5e930)  // sdot v16.4s, v9.16b, v5.4b[3]
    KAI_ASM_INST(0x4f83e09f)  // sdot v31.4s, v4.16b, v3.4b[0]
    KAI_ASM_INST(0x4fa3e09e)  // sdot v30.4s, v4.16b, v3.4b[1]
    KAI_ASM_INST(0x4f83e89d)  // sdot v29.4s, v4.16b, v3.4b[2]
    KAI_ASM_INST(0x4fa3e89c)  // sdot v28.4s, v4.16b, v3.4b[3]
    KAI_ASM_INST(0x4f82e09b)  // sdot v27.4s, v4.16b, v2.4b[0]
    KAI_ASM_INST(0x4fa2e09a)  // sdot v26.4s, v4.16b, v2.4b[1]
    KAI_ASM_INST(0x4f82e899)  // sdot v25.4s, v4.16b, v2.4b[2]
    KAI_ASM_INST(0x4fa2e898)  // sdot v24.4s, v4.16b, v2.4b[3]
    KAI_ASM_INST(0x4f81e097)  // sdot v23.4s, v4.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e096)  // sdot v22.4s, v4.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e895)  // sdot v21.4s, v4.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e894)  // sdot v20.4s, v4.16b, v1.4b[3]
    KAI_ASM_INST(0x4f80e093)  // sdot v19.4s, v4.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e092)  // sdot v18.4s, v4.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e891)  // sdot v17.4s, v4.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e890)  // sdot v16.4s, v4.16b, v0.4b[3]
    bgt label_3
    ldr q5, [x11, #0x0]
    ld1 { v1.4s }, [x27]
    add x27, x27, #0x10
    ldr q4, [x11, #0x10]
    ldr q0, [x27, #0x0]
    add x11, x11, #0x20
    mla v31.4s, v5.4s, v1.s[0]
    mla v30.4s, v5.4s, v1.s[1]
    mla v29.4s, v5.4s, v1.s[2]
    mla v28.4s, v5.4s, v1.s[3]
    fmul v3.4s, v4.4s, v0.s[0]
    fmul v2.4s, v4.4s, v0.s[1]
    fmul v1.4s, v4.4s, v0.s[2]
    scvtf v31.4s, v31.4s
    fmul v0.4s, v4.4s, v0.s[3]
    scvtf v30.4s, v30.4s
    scvtf v29.4s, v29.4s
    scvtf v28.4s, v28.4s
    fmul v31.4s, v31.4s, v3.4s
    fmul v30.4s, v30.4s, v2.4s
    fmul v29.4s, v29.4s, v1.4s
    fmul v28.4s, v28.4s, v0.4s
    ld1 { v1.4s }, [x22]
    add x22, x22, #0x10
    ldr q0, [x22, #0x0]
    mla v27.4s, v5.4s, v1.s[0]
    mla v26.4s, v5.4s, v1.s[1]
    mla v25.4s, v5.4s, v1.s[2]
    mla v24.4s, v5.4s, v1.s[3]
    fmul v3.4s, v4.4s, v0.s[0]
    fmul v2.4s, v4.4s, v0.s[1]
    fmul v1.4s, v4.4s, v0.s[2]
    scvtf v27.4s, v27.4s
    fmul v0.4s, v4.4s, v0.s[3]
    scvtf v26.4s, v26.4s
    scvtf v25.4s, v25.4s
    scvtf v24.4s, v24.4s
    fmul v27.4s, v27.4s, v3.4s
    fmul v26.4s, v26.4s, v2.4s
    fmul v25.4s, v25.4s, v1.4s
    fmul v24.4s, v24.4s, v0.4s
    ld1 { v1.4s }, [x21]
    add x21, x21, #0x10
    ldr q0, [x21, #0x0]
    mla v23.4s, v5.4s, v1.s[0]
    mla v22.4s, v5.4s, v1.s[1]
    mla v21.4s, v5.4s, v1.s[2]
    mla v20.4s, v5.4s, v1.s[3]
    fmul v3.4s, v4.4s, v0.s[0]
    fmul v2.4s, v4.4s, v0.s[1]
    fmul v1.4s, v4.4s, v0.s[2]
    scvtf v23.4s, v23.4s
    fmul v0.4s, v4.4s, v0.s[3]
    scvtf v22.4s, v22.4s
    scvtf v21.4s, v21.4s
    scvtf v20.4s, v20.4s
    fmul v23.4s, v23.4s, v3.4s
    fmul v22.4s, v22.4s, v2.4s
    fmul v21.4s, v21.4s, v1.4s
    fmul v20.4s, v20.4s, v0.4s
    ld1 { v1.4s }, [x20]
    add x20, x20, #0x10
    ldr q0, [x20, #0x0]
    mla v19.4s, v5.4s, v1.s[0]
    mla v18.4s, v5.4s, v1.s[1]
    mla v17.4s, v5.4s, v1.s[2]
    mla v16.4s, v5.4s, v1.s[3]
    fmul v3.4s, v4.4s, v0.s[0]
    fmul v2.4s, v4.4s, v0.s[1]
    fmul v1.4s, v4.4s, v0.s[2]
    scvtf v19.4s, v19.4s
    fmul v0.4s, v4.4s, v0.s[3]
    scvtf v18.4s, v18.4s
    scvtf v17.4s, v17.4s
    scvtf v16.4s, v16.4s
    fmul v19.4s, v19.4s, v3.4s
    fmul v18.4s, v18.4s, v2.4s
    fmul v17.4s, v17.4s, v1.4s
    fmul v16.4s, v16.4s, v0.4s
    ldr q2, [x11, #0x0]
    ld1r { v1.4s }, [x12]
    add x20, x12, #0x4
    cmp x10, #0x4
    ld1r { v0.4s }, [x20]
    add x11, x11, #0x10
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
    fcvtn v31.4h, v31.4s
    fcvtn v30.4h, v30.4s
    fcvtn v29.4h, v29.4s
    fcvtn v28.4h, v28.4s
    fcvtn v27.4h, v27.4s
    fcvtn v26.4h, v26.4s
    fcvtn v25.4h, v25.4s
    fcvtn v24.4h, v24.4s
    fcvtn v23.4h, v23.4s
    fcvtn v22.4h, v22.4s
    fcvtn v21.4h, v21.4s
    fcvtn v20.4h, v20.4s
    fcvtn v19.4h, v19.4s
    fcvtn v18.4h, v18.4s
    fcvtn v17.4h, v17.4s
    fcvtn v16.4h, v16.4s
    blt label_8
    mov x20, x15
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
    b label_13
KAI_ASM_LABEL(label_8)  // Partial output
    mov x28, x15
    add x26, x28, x13, LSL #2
    add x25, x26, x13, LSL #1
    add x24, x26, x13
    add x23, x25, x13
    add x22, x28, x13, LSL #1
    add x21, x28, x13
    add x20, x22, x13
    add x27, x23, x13
    tbz x10, #1, label_9
    st1 { v24.s }[0], [x23], #0x4
    st1 { v25.s }[0], [x25], #0x4
    st1 { v26.s }[0], [x24], #0x4
    st1 { v27.s }[0], [x26], #0x4
    st1 { v28.s }[0], [x20], #0x4
    st1 { v29.s }[0], [x22], #0x4
    st1 { v30.s }[0], [x21], #0x4
    st1 { v31.s }[0], [x28], #0x4
    tbz x10, #0, label_10
    st1 { v24.h }[2], [x23]
    st1 { v25.h }[2], [x25]
    st1 { v26.h }[2], [x24]
    st1 { v27.h }[2], [x26]
    st1 { v28.h }[2], [x20]
    st1 { v29.h }[2], [x22]
    st1 { v30.h }[2], [x21]
    st1 { v31.h }[2], [x28]
    b label_10
KAI_ASM_LABEL(label_9)  // Output block 0: partial_1_0
    st1 { v24.h }[0], [x23]
    st1 { v25.h }[0], [x25]
    st1 { v26.h }[0], [x24]
    st1 { v27.h }[0], [x26]
    st1 { v28.h }[0], [x20]
    st1 { v29.h }[0], [x22]
    st1 { v30.h }[0], [x21]
    st1 { v31.h }[0], [x28]
KAI_ASM_LABEL(label_10)  // Output block 0: Done
    add x26, x27, x13, LSL #2
    add x25, x27, x13, LSL #1
    add x24, x26, x13, LSL #1
    add x23, x27, x13
    add x22, x25, x13
    add x21, x26, x13
    add x20, x24, x13
    tbz x10, #1, label_11
    st1 { v16.s }[0], [x20], #0x4
    st1 { v17.s }[0], [x24], #0x4
    st1 { v18.s }[0], [x21], #0x4
    st1 { v19.s }[0], [x26], #0x4
    st1 { v20.s }[0], [x22], #0x4
    st1 { v21.s }[0], [x25], #0x4
    st1 { v22.s }[0], [x23], #0x4
    st1 { v23.s }[0], [x27], #0x4
    tbz x10, #0, label_12
    st1 { v16.h }[2], [x20]
    st1 { v17.h }[2], [x24]
    st1 { v18.h }[2], [x21]
    st1 { v19.h }[2], [x26]
    st1 { v20.h }[2], [x22]
    st1 { v21.h }[2], [x25]
    st1 { v22.h }[2], [x23]
    st1 { v23.h }[2], [x27]
    b label_12
KAI_ASM_LABEL(label_11)  // Output block 1: partial_1_0
    st1 { v16.h }[0], [x20]
    st1 { v17.h }[0], [x24]
    st1 { v18.h }[0], [x21]
    st1 { v19.h }[0], [x26]
    st1 { v20.h }[0], [x22]
    st1 { v21.h }[0], [x25]
    st1 { v22.h }[0], [x23]
    st1 { v23.h }[0], [x27]
KAI_ASM_LABEL(label_12)  // Output block 1: Done
KAI_ASM_LABEL(label_13)  // Output stage exit
    subs x10, x10, #0x4
    add x15, x15, #0x8
    bgt label_2
    mov x20, #0x4
    sub x14, x14, #0x10
    cmp x14, #0x10
    mov x15, x9
    madd x8, x20, x6, x8
    bge label_1
KAI_ASM_LABEL(label_14)  // Row loop skip
    cbz x14, label_23
KAI_ASM_LABEL(label_15)  // Row tail: Row loop
    mov x26, x17
    mov x25, x16
    add x24, x15, x13, LSL #2
KAI_ASM_LABEL(label_16)  // Row tail: Column loop
    mov x27, x8
    movi v31.4s, #0x0
    movi v30.4s, #0x0
    mov x20, x7
    movi v29.4s, #0x0
    movi v28.4s, #0x0
KAI_ASM_LABEL(label_17)  // Row tail: Sub block loop
    ldr q17, [x26, #0x0]
    ldr q16, [x27, #0x0]
    subs x20, x20, #0x1
    ldr q1, [x26, #0x10]
    ldr q0, [x27, #0x10]
    ldr q27, [x26, #0x20]
    ldr q26, [x27, #0x20]
    ldr q25, [x26, #0x30]
    ldr q24, [x27, #0x30]
    KAI_ASM_INST(0x4f90e23f)  // sdot v31.4s, v17.16b, v16.4b[0]
    KAI_ASM_INST(0x4fb0e23e)  // sdot v30.4s, v17.16b, v16.4b[1]
    ldr q23, [x26, #0x40]
    ldr q22, [x27, #0x40]
    KAI_ASM_INST(0x4f90ea3d)  // sdot v29.4s, v17.16b, v16.4b[2]
    KAI_ASM_INST(0x4fb0ea3c)  // sdot v28.4s, v17.16b, v16.4b[3]
    ldr q21, [x26, #0x50]
    ldr q20, [x27, #0x50]
    ldr q19, [x26, #0x60]
    ldr q18, [x27, #0x60]
    ldr q17, [x26, #0x70]
    ldr q16, [x27, #0x70]
    KAI_ASM_INST(0x4f80e03f)  // sdot v31.4s, v1.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e03e)  // sdot v30.4s, v1.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e83d)  // sdot v29.4s, v1.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e83c)  // sdot v28.4s, v1.16b, v0.4b[3]
    add x27, x27, #0x80
    add x26, x26, #0x80
    KAI_ASM_INST(0x4f9ae37f)  // sdot v31.4s, v27.16b, v26.4b[0]
    KAI_ASM_INST(0x4fbae37e)  // sdot v30.4s, v27.16b, v26.4b[1]
    KAI_ASM_INST(0x4f9aeb7d)  // sdot v29.4s, v27.16b, v26.4b[2]
    KAI_ASM_INST(0x4fbaeb7c)  // sdot v28.4s, v27.16b, v26.4b[3]
    KAI_ASM_INST(0x4f98e33f)  // sdot v31.4s, v25.16b, v24.4b[0]
    KAI_ASM_INST(0x4fb8e33e)  // sdot v30.4s, v25.16b, v24.4b[1]
    KAI_ASM_INST(0x4f98eb3d)  // sdot v29.4s, v25.16b, v24.4b[2]
    KAI_ASM_INST(0x4fb8eb3c)  // sdot v28.4s, v25.16b, v24.4b[3]
    KAI_ASM_INST(0x4f96e2ff)  // sdot v31.4s, v23.16b, v22.4b[0]
    KAI_ASM_INST(0x4fb6e2fe)  // sdot v30.4s, v23.16b, v22.4b[1]
    KAI_ASM_INST(0x4f96eafd)  // sdot v29.4s, v23.16b, v22.4b[2]
    KAI_ASM_INST(0x4fb6eafc)  // sdot v28.4s, v23.16b, v22.4b[3]
    KAI_ASM_INST(0x4f94e2bf)  // sdot v31.4s, v21.16b, v20.4b[0]
    KAI_ASM_INST(0x4fb4e2be)  // sdot v30.4s, v21.16b, v20.4b[1]
    KAI_ASM_INST(0x4f94eabd)  // sdot v29.4s, v21.16b, v20.4b[2]
    KAI_ASM_INST(0x4fb4eabc)  // sdot v28.4s, v21.16b, v20.4b[3]
    KAI_ASM_INST(0x4f92e27f)  // sdot v31.4s, v19.16b, v18.4b[0]
    KAI_ASM_INST(0x4fb2e27e)  // sdot v30.4s, v19.16b, v18.4b[1]
    KAI_ASM_INST(0x4f92ea7d)  // sdot v29.4s, v19.16b, v18.4b[2]
    KAI_ASM_INST(0x4fb2ea7c)  // sdot v28.4s, v19.16b, v18.4b[3]
    KAI_ASM_INST(0x4f90e23f)  // sdot v31.4s, v17.16b, v16.4b[0]
    KAI_ASM_INST(0x4fb0e23e)  // sdot v30.4s, v17.16b, v16.4b[1]
    KAI_ASM_INST(0x4f90ea3d)  // sdot v29.4s, v17.16b, v16.4b[2]
    KAI_ASM_INST(0x4fb0ea3c)  // sdot v28.4s, v17.16b, v16.4b[3]
    bgt label_17
    ldr q18, [x26, #0x0]
    ld1 { v17.4s }, [x27]
    add x27, x27, #0x10
    ldr q20, [x26, #0x10]
    ldr q16, [x27, #0x0]
    add x26, x26, #0x20
    mla v31.4s, v18.4s, v17.s[0]
    mla v30.4s, v18.4s, v17.s[1]
    mla v29.4s, v18.4s, v17.s[2]
    mla v28.4s, v18.4s, v17.s[3]
    fmul v19.4s, v20.4s, v16.s[0]
    fmul v18.4s, v20.4s, v16.s[1]
    fmul v17.4s, v20.4s, v16.s[2]
    scvtf v31.4s, v31.4s
    fmul v16.4s, v20.4s, v16.s[3]
    scvtf v30.4s, v30.4s
    scvtf v29.4s, v29.4s
    scvtf v28.4s, v28.4s
    fmul v31.4s, v31.4s, v19.4s
    fmul v30.4s, v30.4s, v18.4s
    fmul v29.4s, v29.4s, v17.4s
    fmul v28.4s, v28.4s, v16.4s
    ldr q18, [x26, #0x0]
    ld1r { v17.4s }, [x12]
    add x20, x12, #0x4
    cmp x25, #0x4
    ld1r { v16.4s }, [x20]
    add x26, x26, #0x10
    fadd v31.4s, v31.4s, v18.4s
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
    fcvtn v19.4h, v31.4s
    fcvtn v18.4h, v30.4s
    fcvtn v17.4h, v29.4s
    fcvtn v16.4h, v28.4s
    blt label_19
    mov x20, x15
    cmp x14, #0x1
    str d19, [x20, #0x0]
    add x20, x20, x13
    ble label_22
    cmp x14, #0x2
    str d18, [x20, #0x0]
    add x20, x20, x13
    ble label_22
    cmp x14, #0x3
    str d17, [x20, #0x0]
    add x20, x20, x13
    ble label_22
    str d16, [x20, #0x0]
    b label_22
KAI_ASM_LABEL(label_19)  // Row tail: Partial output
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
    tbz x25, #1, label_20
    st1 { v16.s }[0], [x20], #0x4
    st1 { v17.s }[0], [x21], #0x4
    st1 { v18.s }[0], [x22], #0x4
    st1 { v19.s }[0], [x23], #0x4
    tbz x25, #0, label_21
    st1 { v16.h }[2], [x20]
    st1 { v17.h }[2], [x21]
    st1 { v18.h }[2], [x22]
    st1 { v19.h }[2], [x23]
    b label_21
KAI_ASM_LABEL(label_20)  // Row tail: Output block 0: partial_1_0
    st1 { v16.h }[0], [x20]
    st1 { v17.h }[0], [x21]
    st1 { v18.h }[0], [x22]
    st1 { v19.h }[0], [x23]
KAI_ASM_LABEL(label_21)  // Row tail: Output block 0: Done
KAI_ASM_LABEL(label_22)  // Row tail: Output stage exit
    subs x25, x25, #0x4
    add x15, x15, #0x8
    bgt label_16
    subs x14, x14, #0x4
    add x8, x8, x6
    mov x15, x24
    bgt label_15
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod)

    KAI_ASM_END
