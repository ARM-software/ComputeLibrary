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
    #define KAI_ASM_FUNCTION_USE(name) name

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
        #define KAI_ASM_FUNCTION_USE(name) _##name
    #else
        #define KAI_ASM_GLOBAL(name) .global name
        #define KAI_ASM_FUNCTION_TYPE(name) .type name, %function
        #define KAI_ASM_FUNCTION_LABEL(name) name:
        #define KAI_ASM_FUNCTION_END(name) .size name, .-name
        #define KAI_ASM_FUNCTION_USE(name) name
    #endif

    #define KAI_ASM_CODE(name) .text
    #define KAI_ASM_ALIGN .p2align 4,,11
    #define KAI_ASM_LABEL(name) name:
    #define KAI_ASM_INST(hex) .inst hex
    #define KAI_ASM_END
#endif

    KAI_ASM_CODE(kai_common)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_get_sme_vector_length_u8)
    KAI_ASM_GLOBAL(kai_commit_za)

KAI_ASM_FUNCTION_TYPE(kai_commit_za)
KAI_ASM_FUNCTION_LABEL(kai_commit_za)
#if defined(__ARM_FEATURE_SME)
    stp fp, lr, [sp, #-16]!
    mov fp, sp
    mrs x9, TPIDR2_EL0
    cbz x9, leave
    bl KAI_ASM_FUNCTION_USE(__arm_tpidr2_save)
    msr TPIDR2_EL0, xzr
leave:
    ldp fp, lr, [sp], #16
#endif
    ret
    KAI_ASM_FUNCTION_END(kai_commit_za)

KAI_ASM_FUNCTION_TYPE(kai_get_sme_vector_length_u8)
KAI_ASM_FUNCTION_LABEL(kai_get_sme_vector_length_u8)
    KAI_ASM_INST(0x04bf5820) // rdsvl x0, #1
    ret
    KAI_ASM_FUNCTION_END(kai_get_sme_vector_length_u8)

KAI_ASM_END
