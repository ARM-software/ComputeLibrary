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

    KAI_ASM_CODE(matmul_clamp_f16_qai8dxp4x4_qsi4cxp4x4_16x4_neon_dotprod)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_qai8dxp4x4_qsi4cxp4x4_16x4_neon_dotprod)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_qai8dxp4x4_qsi4cxp4x4_16x4_neon_dotprod)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_qai8dxp4x4_qsi4cxp4x4_16x4_neon_dotprod)
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
    ldr q11, [x11, #0x0]
    ldr q8, [x27, #0x0]
    movi v6.16b, #0xf0
    subs x23, x23, #0x1
    ldr q1, [x22, #0x0]
    ldr q15, [x21, #0x0]
    ldr q3, [x20, #0x0]
    ldr q13, [x11, #0x10]
    ldr q10, [x27, #0x10]
    ldr q7, [x22, #0x10]
    shl v9.16b, v11.16b, #0x4
    and v11.16b, v11.16b, v6.16b
    ldr q4, [x21, #0x10]
    ldr q0, [x20, #0x10]
    ldr q5, [x11, #0x20]
    ldr q2, [x27, #0x20]
    shl v12.16b, v13.16b, #0x4
    and v13.16b, v13.16b, v6.16b
    ldr q14, [x22, #0x20]
    KAI_ASM_INST(0x4f88e13f)  // sdot v31.4s, v9.16b, v8.4b[0]
    KAI_ASM_INST(0x4fa8e13e)  // sdot v30.4s, v9.16b, v8.4b[1]
    KAI_ASM_INST(0x4f88e93d)  // sdot v29.4s, v9.16b, v8.4b[2]
    KAI_ASM_INST(0x4fa8e93c)  // sdot v28.4s, v9.16b, v8.4b[3]
    ldr q8, [x21, #0x20]
    KAI_ASM_INST(0x4f81e13b)  // sdot v27.4s, v9.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e13a)  // sdot v26.4s, v9.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e939)  // sdot v25.4s, v9.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e938)  // sdot v24.4s, v9.16b, v1.4b[3]
    ldr q1, [x20, #0x20]
    KAI_ASM_INST(0x4f8fe137)  // sdot v23.4s, v9.16b, v15.4b[0]
    KAI_ASM_INST(0x4fafe136)  // sdot v22.4s, v9.16b, v15.4b[1]
    KAI_ASM_INST(0x4f8fe935)  // sdot v21.4s, v9.16b, v15.4b[2]
    KAI_ASM_INST(0x4fafe934)  // sdot v20.4s, v9.16b, v15.4b[3]
    ldr q15, [x11, #0x30]
    add x11, x11, #0x40
    KAI_ASM_INST(0x4f83e133)  // sdot v19.4s, v9.16b, v3.4b[0]
    KAI_ASM_INST(0x4fa3e132)  // sdot v18.4s, v9.16b, v3.4b[1]
    KAI_ASM_INST(0x4f83e931)  // sdot v17.4s, v9.16b, v3.4b[2]
    KAI_ASM_INST(0x4fa3e930)  // sdot v16.4s, v9.16b, v3.4b[3]
    ldr q3, [x27, #0x30]
    ldr q9, [x22, #0x30]
    KAI_ASM_INST(0x4f8ae19f)  // sdot v31.4s, v12.16b, v10.4b[0]
    KAI_ASM_INST(0x4faae19e)  // sdot v30.4s, v12.16b, v10.4b[1]
    KAI_ASM_INST(0x4f8ae99d)  // sdot v29.4s, v12.16b, v10.4b[2]
    KAI_ASM_INST(0x4faae99c)  // sdot v28.4s, v12.16b, v10.4b[3]
    ldr q10, [x21, #0x30]
    KAI_ASM_INST(0x4f87e19b)  // sdot v27.4s, v12.16b, v7.4b[0]
    KAI_ASM_INST(0x4fa7e19a)  // sdot v26.4s, v12.16b, v7.4b[1]
    KAI_ASM_INST(0x4f87e999)  // sdot v25.4s, v12.16b, v7.4b[2]
    KAI_ASM_INST(0x4fa7e998)  // sdot v24.4s, v12.16b, v7.4b[3]
    ldr q7, [x20, #0x30]
    KAI_ASM_INST(0x4f84e197)  // sdot v23.4s, v12.16b, v4.4b[0]
    KAI_ASM_INST(0x4fa4e196)  // sdot v22.4s, v12.16b, v4.4b[1]
    KAI_ASM_INST(0x4f84e995)  // sdot v21.4s, v12.16b, v4.4b[2]
    KAI_ASM_INST(0x4fa4e994)  // sdot v20.4s, v12.16b, v4.4b[3]
    ldr q4, [x27, #0x40]
    KAI_ASM_INST(0x4f80e193)  // sdot v19.4s, v12.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e192)  // sdot v18.4s, v12.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e991)  // sdot v17.4s, v12.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e990)  // sdot v16.4s, v12.16b, v0.4b[3]
    ldr q0, [x22, #0x40]
    shl v12.16b, v5.16b, #0x4
    and v5.16b, v5.16b, v6.16b
    KAI_ASM_INST(0x4f82e19f)  // sdot v31.4s, v12.16b, v2.4b[0]
    KAI_ASM_INST(0x4fa2e19e)  // sdot v30.4s, v12.16b, v2.4b[1]
    KAI_ASM_INST(0x4f82e99d)  // sdot v29.4s, v12.16b, v2.4b[2]
    KAI_ASM_INST(0x4fa2e99c)  // sdot v28.4s, v12.16b, v2.4b[3]
    ldr q2, [x21, #0x40]
    KAI_ASM_INST(0x4f8ee19b)  // sdot v27.4s, v12.16b, v14.4b[0]
    KAI_ASM_INST(0x4faee19a)  // sdot v26.4s, v12.16b, v14.4b[1]
    KAI_ASM_INST(0x4f8ee999)  // sdot v25.4s, v12.16b, v14.4b[2]
    KAI_ASM_INST(0x4faee998)  // sdot v24.4s, v12.16b, v14.4b[3]
    ldr q14, [x20, #0x40]
    KAI_ASM_INST(0x4f88e197)  // sdot v23.4s, v12.16b, v8.4b[0]
    KAI_ASM_INST(0x4fa8e196)  // sdot v22.4s, v12.16b, v8.4b[1]
    KAI_ASM_INST(0x4f88e995)  // sdot v21.4s, v12.16b, v8.4b[2]
    KAI_ASM_INST(0x4fa8e994)  // sdot v20.4s, v12.16b, v8.4b[3]
    ldr q8, [x27, #0x50]
    KAI_ASM_INST(0x4f81e193)  // sdot v19.4s, v12.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e192)  // sdot v18.4s, v12.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e991)  // sdot v17.4s, v12.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e990)  // sdot v16.4s, v12.16b, v1.4b[3]
    ldr q12, [x22, #0x50]
    shl v1.16b, v15.16b, #0x4
    and v15.16b, v15.16b, v6.16b
    ldr q6, [x21, #0x50]
    KAI_ASM_INST(0x4f83e03f)  // sdot v31.4s, v1.16b, v3.4b[0]
    KAI_ASM_INST(0x4fa3e03e)  // sdot v30.4s, v1.16b, v3.4b[1]
    KAI_ASM_INST(0x4f83e83d)  // sdot v29.4s, v1.16b, v3.4b[2]
    KAI_ASM_INST(0x4fa3e83c)  // sdot v28.4s, v1.16b, v3.4b[3]
    ldr q3, [x20, #0x50]
    KAI_ASM_INST(0x4f89e03b)  // sdot v27.4s, v1.16b, v9.4b[0]
    KAI_ASM_INST(0x4fa9e03a)  // sdot v26.4s, v1.16b, v9.4b[1]
    KAI_ASM_INST(0x4f89e839)  // sdot v25.4s, v1.16b, v9.4b[2]
    KAI_ASM_INST(0x4fa9e838)  // sdot v24.4s, v1.16b, v9.4b[3]
    ldr q9, [x27, #0x60]
    KAI_ASM_INST(0x4f8ae037)  // sdot v23.4s, v1.16b, v10.4b[0]
    KAI_ASM_INST(0x4faae036)  // sdot v22.4s, v1.16b, v10.4b[1]
    KAI_ASM_INST(0x4f8ae835)  // sdot v21.4s, v1.16b, v10.4b[2]
    KAI_ASM_INST(0x4faae834)  // sdot v20.4s, v1.16b, v10.4b[3]
    ldr q10, [x22, #0x60]
    KAI_ASM_INST(0x4f87e033)  // sdot v19.4s, v1.16b, v7.4b[0]
    KAI_ASM_INST(0x4fa7e032)  // sdot v18.4s, v1.16b, v7.4b[1]
    KAI_ASM_INST(0x4f87e831)  // sdot v17.4s, v1.16b, v7.4b[2]
    KAI_ASM_INST(0x4fa7e830)  // sdot v16.4s, v1.16b, v7.4b[3]
    ldr q7, [x21, #0x60]
    ldr q1, [x20, #0x60]
    KAI_ASM_INST(0x4f84e17f)  // sdot v31.4s, v11.16b, v4.4b[0]
    KAI_ASM_INST(0x4fa4e17e)  // sdot v30.4s, v11.16b, v4.4b[1]
    KAI_ASM_INST(0x4f84e97d)  // sdot v29.4s, v11.16b, v4.4b[2]
    KAI_ASM_INST(0x4fa4e97c)  // sdot v28.4s, v11.16b, v4.4b[3]
    ldr q4, [x27, #0x70]
    add x27, x27, #0x80
    KAI_ASM_INST(0x4f80e17b)  // sdot v27.4s, v11.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e17a)  // sdot v26.4s, v11.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e979)  // sdot v25.4s, v11.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e978)  // sdot v24.4s, v11.16b, v0.4b[3]
    ldr q0, [x22, #0x70]
    add x22, x22, #0x80
    KAI_ASM_INST(0x4f82e177)  // sdot v23.4s, v11.16b, v2.4b[0]
    KAI_ASM_INST(0x4fa2e176)  // sdot v22.4s, v11.16b, v2.4b[1]
    KAI_ASM_INST(0x4f82e975)  // sdot v21.4s, v11.16b, v2.4b[2]
    KAI_ASM_INST(0x4fa2e974)  // sdot v20.4s, v11.16b, v2.4b[3]
    ldr q2, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4f8ee173)  // sdot v19.4s, v11.16b, v14.4b[0]
    KAI_ASM_INST(0x4faee172)  // sdot v18.4s, v11.16b, v14.4b[1]
    KAI_ASM_INST(0x4f8ee971)  // sdot v17.4s, v11.16b, v14.4b[2]
    KAI_ASM_INST(0x4faee970)  // sdot v16.4s, v11.16b, v14.4b[3]
    ldr q11, [x20, #0x70]
    add x20, x20, #0x80
    KAI_ASM_INST(0x4f88e1bf)  // sdot v31.4s, v13.16b, v8.4b[0]
    KAI_ASM_INST(0x4fa8e1be)  // sdot v30.4s, v13.16b, v8.4b[1]
    KAI_ASM_INST(0x4f88e9bd)  // sdot v29.4s, v13.16b, v8.4b[2]
    KAI_ASM_INST(0x4fa8e9bc)  // sdot v28.4s, v13.16b, v8.4b[3]
    KAI_ASM_INST(0x4f8ce1bb)  // sdot v27.4s, v13.16b, v12.4b[0]
    KAI_ASM_INST(0x4face1ba)  // sdot v26.4s, v13.16b, v12.4b[1]
    KAI_ASM_INST(0x4f8ce9b9)  // sdot v25.4s, v13.16b, v12.4b[2]
    KAI_ASM_INST(0x4face9b8)  // sdot v24.4s, v13.16b, v12.4b[3]
    KAI_ASM_INST(0x4f86e1b7)  // sdot v23.4s, v13.16b, v6.4b[0]
    KAI_ASM_INST(0x4fa6e1b6)  // sdot v22.4s, v13.16b, v6.4b[1]
    KAI_ASM_INST(0x4f86e9b5)  // sdot v21.4s, v13.16b, v6.4b[2]
    KAI_ASM_INST(0x4fa6e9b4)  // sdot v20.4s, v13.16b, v6.4b[3]
    KAI_ASM_INST(0x4f83e1b3)  // sdot v19.4s, v13.16b, v3.4b[0]
    KAI_ASM_INST(0x4fa3e1b2)  // sdot v18.4s, v13.16b, v3.4b[1]
    KAI_ASM_INST(0x4f83e9b1)  // sdot v17.4s, v13.16b, v3.4b[2]
    KAI_ASM_INST(0x4fa3e9b0)  // sdot v16.4s, v13.16b, v3.4b[3]
    KAI_ASM_INST(0x4f89e0bf)  // sdot v31.4s, v5.16b, v9.4b[0]
    KAI_ASM_INST(0x4fa9e0be)  // sdot v30.4s, v5.16b, v9.4b[1]
    KAI_ASM_INST(0x4f89e8bd)  // sdot v29.4s, v5.16b, v9.4b[2]
    KAI_ASM_INST(0x4fa9e8bc)  // sdot v28.4s, v5.16b, v9.4b[3]
    KAI_ASM_INST(0x4f8ae0bb)  // sdot v27.4s, v5.16b, v10.4b[0]
    KAI_ASM_INST(0x4faae0ba)  // sdot v26.4s, v5.16b, v10.4b[1]
    KAI_ASM_INST(0x4f8ae8b9)  // sdot v25.4s, v5.16b, v10.4b[2]
    KAI_ASM_INST(0x4faae8b8)  // sdot v24.4s, v5.16b, v10.4b[3]
    KAI_ASM_INST(0x4f87e0b7)  // sdot v23.4s, v5.16b, v7.4b[0]
    KAI_ASM_INST(0x4fa7e0b6)  // sdot v22.4s, v5.16b, v7.4b[1]
    KAI_ASM_INST(0x4f87e8b5)  // sdot v21.4s, v5.16b, v7.4b[2]
    KAI_ASM_INST(0x4fa7e8b4)  // sdot v20.4s, v5.16b, v7.4b[3]
    KAI_ASM_INST(0x4f81e0b3)  // sdot v19.4s, v5.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e0b2)  // sdot v18.4s, v5.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e8b1)  // sdot v17.4s, v5.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e8b0)  // sdot v16.4s, v5.16b, v1.4b[3]
    KAI_ASM_INST(0x4f84e1ff)  // sdot v31.4s, v15.16b, v4.4b[0]
    KAI_ASM_INST(0x4fa4e1fe)  // sdot v30.4s, v15.16b, v4.4b[1]
    KAI_ASM_INST(0x4f84e9fd)  // sdot v29.4s, v15.16b, v4.4b[2]
    KAI_ASM_INST(0x4fa4e9fc)  // sdot v28.4s, v15.16b, v4.4b[3]
    KAI_ASM_INST(0x4f80e1fb)  // sdot v27.4s, v15.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e1fa)  // sdot v26.4s, v15.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e9f9)  // sdot v25.4s, v15.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e9f8)  // sdot v24.4s, v15.16b, v0.4b[3]
    KAI_ASM_INST(0x4f82e1f7)  // sdot v23.4s, v15.16b, v2.4b[0]
    KAI_ASM_INST(0x4fa2e1f6)  // sdot v22.4s, v15.16b, v2.4b[1]
    KAI_ASM_INST(0x4f82e9f5)  // sdot v21.4s, v15.16b, v2.4b[2]
    KAI_ASM_INST(0x4fa2e9f4)  // sdot v20.4s, v15.16b, v2.4b[3]
    KAI_ASM_INST(0x4f8be1f3)  // sdot v19.4s, v15.16b, v11.4b[0]
    KAI_ASM_INST(0x4fabe1f2)  // sdot v18.4s, v15.16b, v11.4b[1]
    KAI_ASM_INST(0x4f8be9f1)  // sdot v17.4s, v15.16b, v11.4b[2]
    KAI_ASM_INST(0x4fabe9f0)  // sdot v16.4s, v15.16b, v11.4b[3]
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
    ldr q4, [x26, #0x0]
    ldr q3, [x27, #0x0]
    movi v2.16b, #0xf0
    subs x20, x20, #0x1
    ldr q1, [x26, #0x10]
    ldr q0, [x27, #0x10]
    ldr q27, [x26, #0x20]
    ldr q26, [x27, #0x20]
    ldr q25, [x26, #0x30]
    ldr q24, [x27, #0x30]
    shl v23.16b, v4.16b, #0x4
    and v4.16b, v4.16b, v2.16b
    ldr q22, [x27, #0x40]
    ldr q21, [x27, #0x50]
    shl v20.16b, v1.16b, #0x4
    and v1.16b, v1.16b, v2.16b
    ldr q19, [x27, #0x60]
    ldr q18, [x27, #0x70]
    shl v17.16b, v27.16b, #0x4
    and v27.16b, v27.16b, v2.16b
    KAI_ASM_INST(0x4f83e2ff)  // sdot v31.4s, v23.16b, v3.4b[0]
    KAI_ASM_INST(0x4fa3e2fe)  // sdot v30.4s, v23.16b, v3.4b[1]
    shl v16.16b, v25.16b, #0x4
    add x26, x26, #0x40
    KAI_ASM_INST(0x4f83eafd)  // sdot v29.4s, v23.16b, v3.4b[2]
    KAI_ASM_INST(0x4fa3eafc)  // sdot v28.4s, v23.16b, v3.4b[3]
    and v25.16b, v25.16b, v2.16b
    add x27, x27, #0x80
    KAI_ASM_INST(0x4f80e29f)  // sdot v31.4s, v20.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e29e)  // sdot v30.4s, v20.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80ea9d)  // sdot v29.4s, v20.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0ea9c)  // sdot v28.4s, v20.16b, v0.4b[3]
    KAI_ASM_INST(0x4f9ae23f)  // sdot v31.4s, v17.16b, v26.4b[0]
    KAI_ASM_INST(0x4fbae23e)  // sdot v30.4s, v17.16b, v26.4b[1]
    KAI_ASM_INST(0x4f9aea3d)  // sdot v29.4s, v17.16b, v26.4b[2]
    KAI_ASM_INST(0x4fbaea3c)  // sdot v28.4s, v17.16b, v26.4b[3]
    KAI_ASM_INST(0x4f98e21f)  // sdot v31.4s, v16.16b, v24.4b[0]
    KAI_ASM_INST(0x4fb8e21e)  // sdot v30.4s, v16.16b, v24.4b[1]
    KAI_ASM_INST(0x4f98ea1d)  // sdot v29.4s, v16.16b, v24.4b[2]
    KAI_ASM_INST(0x4fb8ea1c)  // sdot v28.4s, v16.16b, v24.4b[3]
    KAI_ASM_INST(0x4f96e09f)  // sdot v31.4s, v4.16b, v22.4b[0]
    KAI_ASM_INST(0x4fb6e09e)  // sdot v30.4s, v4.16b, v22.4b[1]
    KAI_ASM_INST(0x4f96e89d)  // sdot v29.4s, v4.16b, v22.4b[2]
    KAI_ASM_INST(0x4fb6e89c)  // sdot v28.4s, v4.16b, v22.4b[3]
    KAI_ASM_INST(0x4f95e03f)  // sdot v31.4s, v1.16b, v21.4b[0]
    KAI_ASM_INST(0x4fb5e03e)  // sdot v30.4s, v1.16b, v21.4b[1]
    KAI_ASM_INST(0x4f95e83d)  // sdot v29.4s, v1.16b, v21.4b[2]
    KAI_ASM_INST(0x4fb5e83c)  // sdot v28.4s, v1.16b, v21.4b[3]
    KAI_ASM_INST(0x4f93e37f)  // sdot v31.4s, v27.16b, v19.4b[0]
    KAI_ASM_INST(0x4fb3e37e)  // sdot v30.4s, v27.16b, v19.4b[1]
    KAI_ASM_INST(0x4f93eb7d)  // sdot v29.4s, v27.16b, v19.4b[2]
    KAI_ASM_INST(0x4fb3eb7c)  // sdot v28.4s, v27.16b, v19.4b[3]
    KAI_ASM_INST(0x4f92e33f)  // sdot v31.4s, v25.16b, v18.4b[0]
    KAI_ASM_INST(0x4fb2e33e)  // sdot v30.4s, v25.16b, v18.4b[1]
    KAI_ASM_INST(0x4f92eb3d)  // sdot v29.4s, v25.16b, v18.4b[2]
    KAI_ASM_INST(0x4fb2eb3c)  // sdot v28.4s, v25.16b, v18.4b[3]
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_qai8dxp4x4_qsi4cxp4x4_16x4_neon_dotprod)

    KAI_ASM_END
