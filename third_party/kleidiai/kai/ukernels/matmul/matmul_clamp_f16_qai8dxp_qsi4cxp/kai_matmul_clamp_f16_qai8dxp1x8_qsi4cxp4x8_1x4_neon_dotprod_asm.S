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

    KAI_ASM_CODE(matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod)
    stp x20, x21, [sp, -80]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    mov x13, #0x20
    movi v30.16b, #0xf0
    mov x21, #0x8
    ldr x12, [x0, #0x38]
    ldr x20, [x0, #0x28]
    ldr x11, [x0, #0x8]
    ldr x10, [x0, #0x10]
    ldr x9, [x0, #0x30]
    ldr x28, [x0, #0x0]
    ldr x27, [x0, #0x20]
    madd x13, x12, x13, x21
    ldr x26, [x0, #0x18]
    mov x25, x20
KAI_ASM_LABEL(label_1)  // Row loop
    mov x24, x10
    mov x23, x9
    add x22, x28, x27
KAI_ASM_LABEL(label_2)  // Column loop
    mov x21, x11
    movi v29.4s, #0x0
    movi v28.4s, #0x0
    mov x20, x12
KAI_ASM_LABEL(label_3)  // Sub block loop
    ldr q27, [x24, #0x0]
    ldr q26, [x24, #0x10]
    subs x20, x20, #0x1
    ld1r { v25.2d }, [x21], #0x8
    ldr q24, [x24, #0x20]
    ldr q23, [x24, #0x30]
    add x24, x24, #0x40
    ld1r { v22.2d }, [x21], #0x8
    ld1r { v21.2d }, [x21], #0x8
    shl v20.16b, v27.16b, #0x4
    shl v19.16b, v26.16b, #0x4
    ld1r { v18.2d }, [x21], #0x8
    shl v17.16b, v24.16b, #0x4
    and v27.16b, v27.16b, v30.16b
    shl v16.16b, v23.16b, #0x4
    and v26.16b, v26.16b, v30.16b
    KAI_ASM_INST(0x4e99969d)  // sdot v29.4s, v20.16b, v25.16b
    KAI_ASM_INST(0x4e99967c)  // sdot v28.4s, v19.16b, v25.16b
    and v24.16b, v24.16b, v30.16b
    and v23.16b, v23.16b, v30.16b
    KAI_ASM_INST(0x4e96963d)  // sdot v29.4s, v17.16b, v22.16b
    KAI_ASM_INST(0x4e96961c)  // sdot v28.4s, v16.16b, v22.16b
    KAI_ASM_INST(0x4e95977d)  // sdot v29.4s, v27.16b, v21.16b
    KAI_ASM_INST(0x4e95975c)  // sdot v28.4s, v26.16b, v21.16b
    KAI_ASM_INST(0x4e92971d)  // sdot v29.4s, v24.16b, v18.16b
    KAI_ASM_INST(0x4e9296fc)  // sdot v28.4s, v23.16b, v18.16b
    bgt label_3
    ldr q22, [x24, #0x0]
    ld1r { v21.4s }, [x21]
    addp v29.4s, v29.4s, v28.4s
    add x21, x21, #0x4
    ld1r { v20.4s }, [x21]
    ldr q16, [x24, #0x10]
    add x20, x26, #0x4
    cmp x23, #0x4
    ldr q19, [x24, #0x20]
    ld1r { v18.4s }, [x26]
    add x24, x24, #0x30
    ld1r { v17.4s }, [x20]
    mla v29.4s, v22.4s, v21.s[0]
    fmul v16.4s, v16.4s, v20.4s
    scvtf v29.4s, v29.4s
    fmul v16.4s, v29.4s, v16.4s
    fadd v16.4s, v16.4s, v19.4s
    fmax v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v17.4s
    fcvtn v16.4h, v16.4s
    blt label_4
    str d16, [x28, #0x0]
    b label_7
KAI_ASM_LABEL(label_4)  // Partial output
    mov x20, x28
    tbz x23, #1, label_5
    st1 { v16.s }[0], [x20], #0x4
    tbz x23, #0, label_6
    st1 { v16.h }[2], [x20]
    b label_6
KAI_ASM_LABEL(label_5)  // Output block 0: partial_1_0
    st1 { v16.h }[0], [x20]
KAI_ASM_LABEL(label_6)  // Output block 0: Done
KAI_ASM_LABEL(label_7)  // Stores done
    subs x23, x23, #0x4
    add x28, x28, #0x8
    bgt label_2
    subs x25, x25, #0x1
    add x11, x11, x13
    mov x28, x22
    bgt label_1
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp x20, x21, [sp], 80
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_qai8dxp1x8_qsi4cxp4x8_1x4_neon_dotprod)

    KAI_ASM_END
