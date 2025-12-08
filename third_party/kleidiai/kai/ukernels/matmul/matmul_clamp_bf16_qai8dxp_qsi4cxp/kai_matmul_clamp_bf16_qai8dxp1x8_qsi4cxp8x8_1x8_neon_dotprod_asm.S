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

    KAI_ASM_CODE(matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod)
    stp x20, x21, [sp, -80]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    mov x13, #0x20
    movi v5.16b, #0xf0
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
    movi v4.4s, #0x0
    movi v3.4s, #0x0
    mov x20, x12
    movi v2.4s, #0x0
    movi v1.4s, #0x0
KAI_ASM_LABEL(label_3)  // Sub block loop
    ldr q0, [x24, #0x0]
    ldr q31, [x24, #0x10]
    subs x20, x20, #0x1
    ldr q30, [x24, #0x20]
    ldr q29, [x24, #0x30]
    ld1r { v28.2d }, [x21], #0x8
    ldr q27, [x24, #0x40]
    ldr q26, [x24, #0x50]
    ldr q25, [x24, #0x60]
    shl v24.16b, v0.16b, #0x4
    shl v18.16b, v31.16b, #0x4
    ldr q23, [x24, #0x70]
    shl v17.16b, v30.16b, #0x4
    shl v16.16b, v29.16b, #0x4
    add x24, x24, #0x80
    ld1r { v22.2d }, [x21], #0x8
    shl v21.16b, v27.16b, #0x4
    and v0.16b, v0.16b, v5.16b
    ld1r { v20.2d }, [x21], #0x8
    ld1r { v19.2d }, [x21], #0x8
    KAI_ASM_INST(0x4e9c9704)  // sdot v4.4s, v24.16b, v28.16b
    KAI_ASM_INST(0x4e9c9643)  // sdot v3.4s, v18.16b, v28.16b
    shl v18.16b, v26.16b, #0x4
    KAI_ASM_INST(0x4e9c9622)  // sdot v2.4s, v17.16b, v28.16b
    KAI_ASM_INST(0x4e9c9601)  // sdot v1.4s, v16.16b, v28.16b
    shl v17.16b, v25.16b, #0x4
    shl v16.16b, v23.16b, #0x4
    and v31.16b, v31.16b, v5.16b
    and v30.16b, v30.16b, v5.16b
    and v29.16b, v29.16b, v5.16b
    KAI_ASM_INST(0x4e9696a4)  // sdot v4.4s, v21.16b, v22.16b
    KAI_ASM_INST(0x4e969643)  // sdot v3.4s, v18.16b, v22.16b
    and v27.16b, v27.16b, v5.16b
    KAI_ASM_INST(0x4e969622)  // sdot v2.4s, v17.16b, v22.16b
    KAI_ASM_INST(0x4e969601)  // sdot v1.4s, v16.16b, v22.16b
    and v26.16b, v26.16b, v5.16b
    and v25.16b, v25.16b, v5.16b
    and v23.16b, v23.16b, v5.16b
    KAI_ASM_INST(0x4e949404)  // sdot v4.4s, v0.16b, v20.16b
    KAI_ASM_INST(0x4e9497e3)  // sdot v3.4s, v31.16b, v20.16b
    KAI_ASM_INST(0x4e9497c2)  // sdot v2.4s, v30.16b, v20.16b
    KAI_ASM_INST(0x4e9497a1)  // sdot v1.4s, v29.16b, v20.16b
    KAI_ASM_INST(0x4e939764)  // sdot v4.4s, v27.16b, v19.16b
    KAI_ASM_INST(0x4e939743)  // sdot v3.4s, v26.16b, v19.16b
    KAI_ASM_INST(0x4e939722)  // sdot v2.4s, v25.16b, v19.16b
    KAI_ASM_INST(0x4e9396e1)  // sdot v1.4s, v23.16b, v19.16b
    bgt label_3
    ldr q18, [x24, #0x0]
    ld1r { v24.4s }, [x21]
    addp v4.4s, v4.4s, v3.4s
    addp v2.4s, v2.4s, v1.4s
    ldr q23, [x24, #0x10]
    ldr q22, [x24, #0x20]
    add x21, x21, #0x4
    add x20, x26, #0x4
    ld1r { v16.4s }, [x21]
    ldr q17, [x24, #0x30]
    cmp x23, #0x8
    ldr q21, [x24, #0x40]
    ldr q20, [x24, #0x50]
    mla v4.4s, v18.4s, v24.s[0]
    add x24, x24, #0x60
    ld1r { v19.4s }, [x26]
    ld1r { v18.4s }, [x20]
    mla v2.4s, v23.4s, v24.s[0]
    fmul v22.4s, v22.4s, v16.4s
    fmul v17.4s, v17.4s, v16.4s
    scvtf v4.4s, v4.4s
    fmul v16.4s, v4.4s, v22.4s
    scvtf v2.4s, v2.4s
    fmul v17.4s, v2.4s, v17.4s
    fadd v16.4s, v16.4s, v21.4s
    fadd v17.4s, v17.4s, v20.4s
    fmax v16.4s, v16.4s, v19.4s
    fmin v16.4s, v16.4s, v18.4s
    fmax v17.4s, v17.4s, v19.4s
    fmin v17.4s, v17.4s, v18.4s
    KAI_ASM_INST(0x0ea16a10)  // bfcvtn v16.4h, v16.4s
    KAI_ASM_INST(0x4ea16a30)  // bfcvtn2 v16.8h, v17.4s
    blt label_4
    str q16, [x28, #0x0]
    b label_9
KAI_ASM_LABEL(label_4)  // Partial output
    mov x20, x28
    tbz x23, #2, label_6
    st1 { v16.d }[0], [x20], #0x8
    tbz x23, #1, label_5
    st1 { v16.s }[2], [x20], #0x4
    tbz x23, #0, label_8
    st1 { v16.h }[6], [x20]
    b label_8
KAI_ASM_LABEL(label_5)  // Output block 0: partial_1_4
    tbz x23, #0, label_8
    st1 { v16.h }[4], [x20]
    b label_8
KAI_ASM_LABEL(label_6)  // Output block 0: partial_2_0
    tbz x23, #1, label_7
    st1 { v16.s }[0], [x20], #0x4
    tbz x23, #0, label_8
    st1 { v16.h }[2], [x20]
    b label_8
KAI_ASM_LABEL(label_7)  // Output block 0: partial_1_0
    st1 { v16.h }[0], [x20]
KAI_ASM_LABEL(label_8)  // Output block 0: Done
KAI_ASM_LABEL(label_9)  // Stores done
    subs x23, x23, #0x8
    add x28, x28, #0x10
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_bf16_qai8dxp1x8_qsi4cxp8x8_1x8_neon_dotprod)

    KAI_ASM_END
