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

    KAI_ASM_CODE(matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod)
    stp x20, x21, [sp, -80]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    mov x21, #0x8
    movi v31.16b, #0xf0
    mov x15, #0x20
    ldr x14, [x0, #0x40]
    ldr x13, [x0, #0x38]
    ldr x20, [x0, #0x28]
    ldr x12, [x0, #0x8]
    ldr x11, [x0, #0x10]
    ldr x10, [x0, #0x30]
    madd x15, x14, x15, x21
    ldr x9, [x0, #0x0]
    ldr x28, [x0, #0x20]
    ldr x27, [x0, #0x18]
    mov x26, x20
    mul x15, x13, x15
KAI_ASM_LABEL(label_1)  // Row loop
    mov x25, x11
    mov x24, x10
    add x23, x9, x28
KAI_ASM_LABEL(label_2)  // Column loop
    mov x22, x12
    movi v30.16b, #0x0
    mov x21, x13
KAI_ASM_LABEL(label_3)  // Block loop
    movi v29.4s, #0x0
    movi v28.4s, #0x0
    mov x20, x14
KAI_ASM_LABEL(label_4)  // Sub block loop
    ldr q27, [x25, #0x0]
    ldr q26, [x25, #0x10]
    subs x20, x20, #0x1
    ld1r { v25.2d }, [x22], #0x8
    ldr q24, [x25, #0x20]
    ldr q23, [x25, #0x30]
    add x25, x25, #0x40
    ld1r { v22.2d }, [x22], #0x8
    ld1r { v21.2d }, [x22], #0x8
    shl v20.16b, v27.16b, #0x4
    shl v19.16b, v26.16b, #0x4
    ld1r { v18.2d }, [x22], #0x8
    shl v17.16b, v24.16b, #0x4
    and v27.16b, v27.16b, v31.16b
    shl v16.16b, v23.16b, #0x4
    and v26.16b, v26.16b, v31.16b
    KAI_ASM_INST(0x4e99969d)  // sdot v29.4s, v20.16b, v25.16b
    KAI_ASM_INST(0x4e99967c)  // sdot v28.4s, v19.16b, v25.16b
    and v24.16b, v24.16b, v31.16b
    and v23.16b, v23.16b, v31.16b
    KAI_ASM_INST(0x4e96963d)  // sdot v29.4s, v17.16b, v22.16b
    KAI_ASM_INST(0x4e96961c)  // sdot v28.4s, v16.16b, v22.16b
    KAI_ASM_INST(0x4e95977d)  // sdot v29.4s, v27.16b, v21.16b
    KAI_ASM_INST(0x4e95975c)  // sdot v28.4s, v26.16b, v21.16b
    KAI_ASM_INST(0x4e92971d)  // sdot v29.4s, v24.16b, v18.16b
    KAI_ASM_INST(0x4e9296fc)  // sdot v28.4s, v23.16b, v18.16b
    bgt label_4
    ldr q19, [x25, #0x0]
    ld1r { v18.4s }, [x22]
    add x22, x22, #0x4
    addp v29.4s, v29.4s, v28.4s
    ld1r { v17.4s }, [x22]
    ldr q16, [x25, #0x10]
    sub x21, x21, #0x1
    add x22, x22, #0x4
    add x25, x25, #0x20
    fmla v30.4s, v19.4s, v18.s[0]
    scvtf v29.4s, v29.4s
    fmul v16.4s, v16.4s, v17.4s
    fmla v30.4s, v29.4s, v16.4s
    cbnz x21, label_3
    ldr q18, [x25, #0x0]
    ld1r { v17.4s }, [x27]
    add x20, x27, #0x4
    cmp x24, #0x4
    ld1r { v16.4s }, [x20]
    add x25, x25, #0x10
    fadd v30.4s, v30.4s, v18.4s
    fmax v30.4s, v30.4s, v17.4s
    fmin v30.4s, v30.4s, v16.4s
    blt label_5
    str q30, [x9, #0x0]
    b label_8
KAI_ASM_LABEL(label_5)  // Partial output
    mov x20, x9
    tbz x24, #1, label_6
    st1 { v30.d }[0], [x20], #0x8
    tbz x24, #0, label_7
    st1 { v30.s }[2], [x20]
    b label_7
KAI_ASM_LABEL(label_6)  // Output block 0: partial_1_0
    st1 { v30.s }[0], [x20]
KAI_ASM_LABEL(label_7)  // Output block 0: Done
KAI_ASM_LABEL(label_8)  // Stores done
    subs x24, x24, #0x4
    add x9, x9, #0x10
    bgt label_2
    subs x26, x26, #0x1
    add x12, x12, x15
    mov x9, x23
    bgt label_1
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp x20, x21, [sp], 80
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod)

    KAI_ASM_END
