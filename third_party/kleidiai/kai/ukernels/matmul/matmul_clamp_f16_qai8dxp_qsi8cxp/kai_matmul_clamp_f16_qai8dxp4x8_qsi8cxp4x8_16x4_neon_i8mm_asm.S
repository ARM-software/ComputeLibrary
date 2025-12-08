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

    KAI_ASM_CODE(matmul_clamp_f16_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm)
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
    ldr q2, [x11, #0x0]
    ldr q1, [x11, #0x10]
    subs x23, x23, #0x1
    ldr q5, [x27, #0x0]
    ldr q9, [x27, #0x10]
    ldr q8, [x22, #0x0]
    ldr q7, [x22, #0x10]
    ldr q4, [x21, #0x0]
    ldr q14, [x21, #0x10]
    ldr q3, [x20, #0x0]
    ldr q0, [x20, #0x10]
    KAI_ASM_INST(0x4e82a4bf)  // smmla v31.4s, v5.16b, v2.16b
    KAI_ASM_INST(0x4e81a4be)  // smmla v30.4s, v5.16b, v1.16b
    ldr q6, [x11, #0x20]
    ldr q5, [x11, #0x30]
    KAI_ASM_INST(0x4e82a53d)  // smmla v29.4s, v9.16b, v2.16b
    KAI_ASM_INST(0x4e81a53c)  // smmla v28.4s, v9.16b, v1.16b
    ldr q13, [x27, #0x20]
    ldr q12, [x27, #0x30]
    KAI_ASM_INST(0x4e82a51b)  // smmla v27.4s, v8.16b, v2.16b
    KAI_ASM_INST(0x4e81a51a)  // smmla v26.4s, v8.16b, v1.16b
    ldr q11, [x22, #0x20]
    ldr q10, [x22, #0x30]
    KAI_ASM_INST(0x4e82a4f9)  // smmla v25.4s, v7.16b, v2.16b
    KAI_ASM_INST(0x4e81a4f8)  // smmla v24.4s, v7.16b, v1.16b
    ldr q9, [x21, #0x20]
    ldr q8, [x21, #0x30]
    KAI_ASM_INST(0x4e82a497)  // smmla v23.4s, v4.16b, v2.16b
    KAI_ASM_INST(0x4e81a496)  // smmla v22.4s, v4.16b, v1.16b
    ldr q7, [x20, #0x20]
    ldr q4, [x20, #0x30]
    KAI_ASM_INST(0x4e82a5d5)  // smmla v21.4s, v14.16b, v2.16b
    KAI_ASM_INST(0x4e81a5d4)  // smmla v20.4s, v14.16b, v1.16b
    ldr q15, [x11, #0x40]
    ldr q14, [x11, #0x50]
    KAI_ASM_INST(0x4e82a473)  // smmla v19.4s, v3.16b, v2.16b
    KAI_ASM_INST(0x4e81a472)  // smmla v18.4s, v3.16b, v1.16b
    ldr q3, [x27, #0x40]
    KAI_ASM_INST(0x4e82a411)  // smmla v17.4s, v0.16b, v2.16b
    ldr q2, [x27, #0x50]
    KAI_ASM_INST(0x4e81a410)  // smmla v16.4s, v0.16b, v1.16b
    ldr q1, [x22, #0x40]
    ldr q0, [x22, #0x50]
    KAI_ASM_INST(0x4e86a5bf)  // smmla v31.4s, v13.16b, v6.16b
    KAI_ASM_INST(0x4e85a5be)  // smmla v30.4s, v13.16b, v5.16b
    ldr q13, [x21, #0x40]
    KAI_ASM_INST(0x4e86a59d)  // smmla v29.4s, v12.16b, v6.16b
    KAI_ASM_INST(0x4e85a59c)  // smmla v28.4s, v12.16b, v5.16b
    ldr q12, [x21, #0x50]
    KAI_ASM_INST(0x4e86a57b)  // smmla v27.4s, v11.16b, v6.16b
    KAI_ASM_INST(0x4e85a57a)  // smmla v26.4s, v11.16b, v5.16b
    ldr q11, [x20, #0x40]
    KAI_ASM_INST(0x4e86a559)  // smmla v25.4s, v10.16b, v6.16b
    KAI_ASM_INST(0x4e85a558)  // smmla v24.4s, v10.16b, v5.16b
    ldr q10, [x20, #0x50]
    KAI_ASM_INST(0x4e86a537)  // smmla v23.4s, v9.16b, v6.16b
    KAI_ASM_INST(0x4e85a536)  // smmla v22.4s, v9.16b, v5.16b
    ldr q9, [x11, #0x60]
    KAI_ASM_INST(0x4e86a515)  // smmla v21.4s, v8.16b, v6.16b
    KAI_ASM_INST(0x4e85a514)  // smmla v20.4s, v8.16b, v5.16b
    ldr q8, [x11, #0x70]
    add x11, x11, #0x80
    KAI_ASM_INST(0x4e86a4f3)  // smmla v19.4s, v7.16b, v6.16b
    KAI_ASM_INST(0x4e85a4f2)  // smmla v18.4s, v7.16b, v5.16b
    ldr q7, [x27, #0x60]
    KAI_ASM_INST(0x4e86a491)  // smmla v17.4s, v4.16b, v6.16b
    ldr q6, [x27, #0x70]
    KAI_ASM_INST(0x4e85a490)  // smmla v16.4s, v4.16b, v5.16b
    ldr q5, [x22, #0x60]
    ldr q4, [x22, #0x70]
    KAI_ASM_INST(0x4e8fa47f)  // smmla v31.4s, v3.16b, v15.16b
    KAI_ASM_INST(0x4e8ea47e)  // smmla v30.4s, v3.16b, v14.16b
    ldr q3, [x21, #0x60]
    KAI_ASM_INST(0x4e8fa45d)  // smmla v29.4s, v2.16b, v15.16b
    KAI_ASM_INST(0x4e8ea45c)  // smmla v28.4s, v2.16b, v14.16b
    ldr q2, [x21, #0x70]
    add x27, x27, #0x80
    KAI_ASM_INST(0x4e8fa43b)  // smmla v27.4s, v1.16b, v15.16b
    KAI_ASM_INST(0x4e8ea43a)  // smmla v26.4s, v1.16b, v14.16b
    ldr q1, [x20, #0x60]
    add x22, x22, #0x80
    KAI_ASM_INST(0x4e8fa419)  // smmla v25.4s, v0.16b, v15.16b
    KAI_ASM_INST(0x4e8ea418)  // smmla v24.4s, v0.16b, v14.16b
    ldr q0, [x20, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4e8fa5b7)  // smmla v23.4s, v13.16b, v15.16b
    KAI_ASM_INST(0x4e8ea5b6)  // smmla v22.4s, v13.16b, v14.16b
    add x20, x20, #0x80
    KAI_ASM_INST(0x4e8fa595)  // smmla v21.4s, v12.16b, v15.16b
    KAI_ASM_INST(0x4e8ea594)  // smmla v20.4s, v12.16b, v14.16b
    KAI_ASM_INST(0x4e8fa573)  // smmla v19.4s, v11.16b, v15.16b
    KAI_ASM_INST(0x4e8ea572)  // smmla v18.4s, v11.16b, v14.16b
    KAI_ASM_INST(0x4e8fa551)  // smmla v17.4s, v10.16b, v15.16b
    KAI_ASM_INST(0x4e8ea550)  // smmla v16.4s, v10.16b, v14.16b
    KAI_ASM_INST(0x4e89a4ff)  // smmla v31.4s, v7.16b, v9.16b
    KAI_ASM_INST(0x4e88a4fe)  // smmla v30.4s, v7.16b, v8.16b
    KAI_ASM_INST(0x4e89a4dd)  // smmla v29.4s, v6.16b, v9.16b
    KAI_ASM_INST(0x4e88a4dc)  // smmla v28.4s, v6.16b, v8.16b
    KAI_ASM_INST(0x4e89a4bb)  // smmla v27.4s, v5.16b, v9.16b
    KAI_ASM_INST(0x4e88a4ba)  // smmla v26.4s, v5.16b, v8.16b
    KAI_ASM_INST(0x4e89a499)  // smmla v25.4s, v4.16b, v9.16b
    KAI_ASM_INST(0x4e88a498)  // smmla v24.4s, v4.16b, v8.16b
    KAI_ASM_INST(0x4e89a477)  // smmla v23.4s, v3.16b, v9.16b
    KAI_ASM_INST(0x4e88a476)  // smmla v22.4s, v3.16b, v8.16b
    KAI_ASM_INST(0x4e89a455)  // smmla v21.4s, v2.16b, v9.16b
    KAI_ASM_INST(0x4e88a454)  // smmla v20.4s, v2.16b, v8.16b
    KAI_ASM_INST(0x4e89a433)  // smmla v19.4s, v1.16b, v9.16b
    KAI_ASM_INST(0x4e88a432)  // smmla v18.4s, v1.16b, v8.16b
    KAI_ASM_INST(0x4e89a411)  // smmla v17.4s, v0.16b, v9.16b
    KAI_ASM_INST(0x4e88a410)  // smmla v16.4s, v0.16b, v8.16b
    bgt label_3
    ldr q7, [x11, #0x0]
    ld1 { v4.4s }, [x27]
    uzp1 v3.2d, v31.2d, v30.2d
    uzp2 v2.2d, v31.2d, v30.2d
    ldr q6, [x11, #0x10]
    uzp1 v1.2d, v29.2d, v28.2d
    uzp2 v0.2d, v29.2d, v28.2d
    add x27, x27, #0x10
    ldr q28, [x27, #0x0]
    add x11, x11, #0x20
    mla v3.4s, v7.4s, v4.s[0]
    mla v2.4s, v7.4s, v4.s[1]
    mla v1.4s, v7.4s, v4.s[2]
    mla v0.4s, v7.4s, v4.s[3]
    fmul v31.4s, v6.4s, v28.s[0]
    fmul v30.4s, v6.4s, v28.s[1]
    fmul v29.4s, v6.4s, v28.s[2]
    fmul v28.4s, v6.4s, v28.s[3]
    scvtf v3.4s, v3.4s
    scvtf v2.4s, v2.4s
    scvtf v1.4s, v1.4s
    scvtf v0.4s, v0.4s
    fmul v31.4s, v3.4s, v31.4s
    fmul v30.4s, v2.4s, v30.4s
    fmul v29.4s, v1.4s, v29.4s
    fmul v28.4s, v0.4s, v28.4s
    ld1 { v5.4s }, [x22]
    uzp1 v4.2d, v27.2d, v26.2d
    uzp2 v3.2d, v27.2d, v26.2d
    add x22, x22, #0x10
    ldr q2, [x22, #0x0]
    uzp1 v1.2d, v25.2d, v24.2d
    uzp2 v0.2d, v25.2d, v24.2d
    mla v4.4s, v7.4s, v5.s[0]
    mla v3.4s, v7.4s, v5.s[1]
    mla v1.4s, v7.4s, v5.s[2]
    mla v0.4s, v7.4s, v5.s[3]
    fmul v27.4s, v6.4s, v2.s[0]
    fmul v26.4s, v6.4s, v2.s[1]
    fmul v25.4s, v6.4s, v2.s[2]
    scvtf v4.4s, v4.4s
    fmul v24.4s, v6.4s, v2.s[3]
    scvtf v3.4s, v3.4s
    scvtf v1.4s, v1.4s
    scvtf v0.4s, v0.4s
    fmul v27.4s, v4.4s, v27.4s
    fmul v26.4s, v3.4s, v26.4s
    fmul v25.4s, v1.4s, v25.4s
    fmul v24.4s, v0.4s, v24.4s
    ld1 { v5.4s }, [x21]
    uzp1 v4.2d, v23.2d, v22.2d
    uzp2 v3.2d, v23.2d, v22.2d
    add x21, x21, #0x10
    ldr q2, [x21, #0x0]
    uzp1 v1.2d, v21.2d, v20.2d
    uzp2 v0.2d, v21.2d, v20.2d
    mla v4.4s, v7.4s, v5.s[0]
    mla v3.4s, v7.4s, v5.s[1]
    mla v1.4s, v7.4s, v5.s[2]
    mla v0.4s, v7.4s, v5.s[3]
    fmul v23.4s, v6.4s, v2.s[0]
    fmul v22.4s, v6.4s, v2.s[1]
    fmul v21.4s, v6.4s, v2.s[2]
    scvtf v4.4s, v4.4s
    fmul v20.4s, v6.4s, v2.s[3]
    scvtf v3.4s, v3.4s
    scvtf v1.4s, v1.4s
    scvtf v0.4s, v0.4s
    fmul v23.4s, v4.4s, v23.4s
    fmul v22.4s, v3.4s, v22.4s
    fmul v21.4s, v1.4s, v21.4s
    fmul v20.4s, v0.4s, v20.4s
    ld1 { v5.4s }, [x20]
    uzp1 v4.2d, v19.2d, v18.2d
    uzp2 v3.2d, v19.2d, v18.2d
    add x20, x20, #0x10
    ldr q2, [x20, #0x0]
    uzp1 v1.2d, v17.2d, v16.2d
    uzp2 v0.2d, v17.2d, v16.2d
    mla v4.4s, v7.4s, v5.s[0]
    mla v3.4s, v7.4s, v5.s[1]
    mla v1.4s, v7.4s, v5.s[2]
    mla v0.4s, v7.4s, v5.s[3]
    fmul v19.4s, v6.4s, v2.s[0]
    fmul v18.4s, v6.4s, v2.s[1]
    fmul v17.4s, v6.4s, v2.s[2]
    scvtf v4.4s, v4.4s
    fmul v16.4s, v6.4s, v2.s[3]
    scvtf v3.4s, v3.4s
    scvtf v1.4s, v1.4s
    scvtf v0.4s, v0.4s
    fmul v19.4s, v4.4s, v19.4s
    fmul v18.4s, v3.4s, v18.4s
    fmul v17.4s, v1.4s, v17.4s
    fmul v16.4s, v0.4s, v16.4s
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
    ldr q19, [x26, #0x0]
    ldr q18, [x26, #0x10]
    subs x20, x20, #0x1
    ldr q17, [x27, #0x0]
    ldr q16, [x27, #0x10]
    ldr q27, [x26, #0x20]
    ldr q26, [x26, #0x30]
    ldr q25, [x27, #0x20]
    ldr q24, [x27, #0x30]
    ldr q23, [x26, #0x40]
    ldr q22, [x26, #0x50]
    KAI_ASM_INST(0x4e93a63f)  // smmla v31.4s, v17.16b, v19.16b
    KAI_ASM_INST(0x4e92a63e)  // smmla v30.4s, v17.16b, v18.16b
    ldr q21, [x27, #0x40]
    ldr q20, [x27, #0x50]
    KAI_ASM_INST(0x4e93a61d)  // smmla v29.4s, v16.16b, v19.16b
    KAI_ASM_INST(0x4e92a61c)  // smmla v28.4s, v16.16b, v18.16b
    ldr q19, [x26, #0x60]
    ldr q18, [x26, #0x70]
    add x26, x26, #0x80
    ldr q17, [x27, #0x60]
    ldr q16, [x27, #0x70]
    add x27, x27, #0x80
    KAI_ASM_INST(0x4e9ba73f)  // smmla v31.4s, v25.16b, v27.16b
    KAI_ASM_INST(0x4e9aa73e)  // smmla v30.4s, v25.16b, v26.16b
    KAI_ASM_INST(0x4e9ba71d)  // smmla v29.4s, v24.16b, v27.16b
    KAI_ASM_INST(0x4e9aa71c)  // smmla v28.4s, v24.16b, v26.16b
    KAI_ASM_INST(0x4e97a6bf)  // smmla v31.4s, v21.16b, v23.16b
    KAI_ASM_INST(0x4e96a6be)  // smmla v30.4s, v21.16b, v22.16b
    KAI_ASM_INST(0x4e97a69d)  // smmla v29.4s, v20.16b, v23.16b
    KAI_ASM_INST(0x4e96a69c)  // smmla v28.4s, v20.16b, v22.16b
    KAI_ASM_INST(0x4e93a63f)  // smmla v31.4s, v17.16b, v19.16b
    KAI_ASM_INST(0x4e92a63e)  // smmla v30.4s, v17.16b, v18.16b
    KAI_ASM_INST(0x4e93a61d)  // smmla v29.4s, v16.16b, v19.16b
    KAI_ASM_INST(0x4e92a61c)  // smmla v28.4s, v16.16b, v18.16b
    bgt label_17
    ldr q18, [x26, #0x0]
    ld1 { v17.4s }, [x27]
    uzp1 v24.2d, v31.2d, v30.2d
    uzp2 v23.2d, v31.2d, v30.2d
    ldr q22, [x26, #0x10]
    uzp1 v21.2d, v29.2d, v28.2d
    uzp2 v20.2d, v29.2d, v28.2d
    add x27, x27, #0x10
    ldr q16, [x27, #0x0]
    add x26, x26, #0x20
    mla v24.4s, v18.4s, v17.s[0]
    mla v23.4s, v18.4s, v17.s[1]
    mla v21.4s, v18.4s, v17.s[2]
    mla v20.4s, v18.4s, v17.s[3]
    fmul v19.4s, v22.4s, v16.s[0]
    fmul v18.4s, v22.4s, v16.s[1]
    fmul v17.4s, v22.4s, v16.s[2]
    fmul v16.4s, v22.4s, v16.s[3]
    scvtf v24.4s, v24.4s
    scvtf v23.4s, v23.4s
    scvtf v21.4s, v21.4s
    scvtf v20.4s, v20.4s
    fmul v31.4s, v24.4s, v19.4s
    fmul v30.4s, v23.4s, v18.4s
    fmul v29.4s, v21.4s, v17.4s
    fmul v28.4s, v20.4s, v16.4s
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm)

    KAI_ASM_END
