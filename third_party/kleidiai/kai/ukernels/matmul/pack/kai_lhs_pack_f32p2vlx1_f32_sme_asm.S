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

    KAI_ASM_CODE(lhs_pack_f32p2vlx1_f32_sme)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_lhs_pack_f32p2vlx1_f32_sme)

KAI_ASM_FUNCTION_TYPE(kai_kernel_lhs_pack_f32p2vlx1_f32_sme)
KAI_ASM_FUNCTION_LABEL(kai_kernel_lhs_pack_f32p2vlx1_f32_sme)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x22, #0x0
    mov x21, x1
    cntw x16
    incw x21
    mov x20, x1
    sub x21, x21, #0x1
    sub x15, x16, #0x1
    udiv x21, x21, x16  // n_passes = ceildiv(width, VL<T>)
    ands x15, x20, x15
    sub x20, x21, #0x1
    sub x14, x16, #0x2
    mov x11, #0x0
    mov x10, x2
    add x9, x2, x16, LSL #3
    cntw x28, ALL, MUL #2
    ldr x27, [x10, #0x0]
    cntw x26, ALL, MUL #3
    lsr x20, x20, #0x1  // n_loops = (n_passes - 1) / 2
    ldr x25, [x9, #0x0]
    and x24, x21, #0x1  // odd_tail = bool(n_passes & 0x1)
    csel x15, x15, x16, NE
    ldr x23, [x10, #0x8]
    ptrue p12.s
    whilelt p11.s, XZR, x0
    ldr x21, [x9, #0x8]
    whilelt p10.s, x16, x0
    mov x22, x22
    whilelt p9.s, x11, x1
    whilelt p8.s, x11, x1
    add x10, x10, #0x10
    add x9, x9, #0x10
    mov x12, #0x0
    cbz x14, label_2
KAI_ASM_LABEL(label_1)  // K loop: Charge: Loop
    KAI_ASM_INST(0x25306163)  // psel p3.s, p8.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25306142)  // psel p2.s, p8.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25706161)  // psel p1.s, p8.s/Z, p11.s[w12, #1]
    KAI_ASM_INST(0x25706140)  // psel p0.s, p8.s/Z, p10.s[w12, #1]
    KAI_ASM_INST(0xe0960f60)  // ld1w { za0h.s[x12] }, p3/Z, [x27, x22, LSL #2]
    ldr x27, [x10, #0x0]
    KAI_ASM_INST(0xe0960b24)  // ld1w { za1h.s[x12] }, p2/Z, [x25, x22, LSL #2]
    ldr x25, [x9, #0x0]
    KAI_ASM_INST(0xe09606e1)  // ld1w { za0h.s[x12, #1] }, p1/Z, [x23, x22, LSL #2]
    ldr x23, [x10, #0x8]
    add x10, x10, #0x10
    KAI_ASM_INST(0xe09602a5)  // ld1w { za1h.s[x12, #1] }, p0/Z, [x21, x22, LSL #2]
    add x12, x12, #0x2
    ldr x21, [x9, #0x8]
    add x9, x9, #0x10
    cmp x12, x14
    blt label_1
KAI_ASM_LABEL(label_2)  // K loop: Charge: End
    KAI_ASM_INST(0x25306163)  // psel p3.s, p8.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25306142)  // psel p2.s, p8.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25706161)  // psel p1.s, p8.s/Z, p11.s[w12, #1]
    KAI_ASM_INST(0x25706140)  // psel p0.s, p8.s/Z, p10.s[w12, #1]
    mov x10, x2
    add x9, x2, x16, LSL #3
    KAI_ASM_INST(0xe0960f60)  // ld1w { za0h.s[x12] }, p3/Z, [x27, x22, LSL #2]
    ldr x27, [x10, #0x0]
    incw x11
    KAI_ASM_INST(0xe0960b24)  // ld1w { za1h.s[x12] }, p2/Z, [x25, x22, LSL #2]
    ldr x25, [x9, #0x0]
    KAI_ASM_INST(0xe09606e1)  // ld1w { za0h.s[x12, #1] }, p1/Z, [x23, x22, LSL #2]
    ldr x23, [x10, #0x8]
    add x10, x10, #0x10
    KAI_ASM_INST(0xe09602a5)  // ld1w { za1h.s[x12, #1] }, p0/Z, [x21, x22, LSL #2]
    ldr x21, [x9, #0x8]
    add x9, x9, #0x10
    incw x22
    cbz x20, label_8
    mov x20, x20
KAI_ASM_LABEL(label_3)  // K loop: Main loop
    whilelt p8.s, x11, x1
    mov x13, #0x0
    cbz x14, label_5
KAI_ASM_LABEL(label_4)  // K loop: Main loop: First: Loop
    KAI_ASM_INST(0x25316160)  // psel p0.s, p8.s/Z, p11.s[w13]
    KAI_ASM_INST(0x25316142)  // psel p2.s, p8.s/Z, p10.s[w13]
    KAI_ASM_INST(0x25716161)  // psel p1.s, p8.s/Z, p11.s[w13, #1]
    KAI_ASM_INST(0x25716143)  // psel p3.s, p8.s/Z, p10.s[w13, #1]
    KAI_ASM_INST(0xe0962368)  // ld1w { za2h.s[x13] }, p0/Z, [x27, x22, LSL #2]
    KAI_ASM_INST(0x25317120)  // psel p0.s, p12.s/Z, p9.s[w13]
    ldr x27, [x10, #0x0]
    KAI_ASM_INST(0xe0962b2c)  // ld1w { za3h.s[x13] }, p2/Z, [x25, x22, LSL #2]
    KAI_ASM_INST(0x25317122)  // psel p2.s, p12.s/Z, p9.s[w13]
    ldr x25, [x9, #0x0]
    KAI_ASM_INST(0xe09626e9)  // ld1w { za2h.s[x13, #1] }, p1/Z, [x23, x22, LSL #2]
    KAI_ASM_INST(0x25717121)  // psel p1.s, p12.s/Z, p9.s[w13, #1]
    ldr x23, [x10, #0x8]
    add x10, x10, #0x10
    KAI_ASM_INST(0xe0962ead)  // ld1w { za3h.s[x13, #1] }, p3/Z, [x21, x22, LSL #2]
    ldr x21, [x9, #0x8]
    KAI_ASM_INST(0xe0bfa060)  // st1w { za0v.s[x13] }, p0/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0x25717120)  // psel p0.s, p12.s/Z, p9.s[w13, #1]
    KAI_ASM_INST(0xe0b0a864)  // st1w { za1v.s[x13] }, p2/Z, [x3, x16, LSL #2]
    add x9, x9, #0x10
    KAI_ASM_INST(0xe0bca461)  // st1w { za0v.s[x13, #1] }, p1/Z, [x3, x28, LSL #2]
    KAI_ASM_INST(0xe0baa065)  // st1w { za1v.s[x13, #1] }, p0/Z, [x3, x26, LSL #2]
    add x13, x13, #0x2
    addvl x3, x3, #4
    cmp x13, x14
    blt label_4
KAI_ASM_LABEL(label_5)  // K loop: Main loop: First: Tail
    KAI_ASM_INST(0x25316160)  // psel p0.s, p8.s/Z, p11.s[w13]
    KAI_ASM_INST(0x25316142)  // psel p2.s, p8.s/Z, p10.s[w13]
    KAI_ASM_INST(0x25716161)  // psel p1.s, p8.s/Z, p11.s[w13, #1]
    KAI_ASM_INST(0x25716143)  // psel p3.s, p8.s/Z, p10.s[w13, #1]
    mov x10, x2
    add x9, x2, x16, LSL #3
    KAI_ASM_INST(0xe0962368)  // ld1w { za2h.s[x13] }, p0/Z, [x27, x22, LSL #2]
    KAI_ASM_INST(0x25317120)  // psel p0.s, p12.s/Z, p9.s[w13]
    ldr x27, [x10, #0x0]
    mov x12, #0x0
    KAI_ASM_INST(0xe0962b2c)  // ld1w { za3h.s[x13] }, p2/Z, [x25, x22, LSL #2]
    KAI_ASM_INST(0x25317122)  // psel p2.s, p12.s/Z, p9.s[w13]
    ldr x25, [x9, #0x0]
    KAI_ASM_INST(0xe09626e9)  // ld1w { za2h.s[x13, #1] }, p1/Z, [x23, x22, LSL #2]
    KAI_ASM_INST(0x25717121)  // psel p1.s, p12.s/Z, p9.s[w13, #1]
    ldr x23, [x10, #0x8]
    add x10, x10, #0x10
    KAI_ASM_INST(0xe0962ead)  // ld1w { za3h.s[x13, #1] }, p3/Z, [x21, x22, LSL #2]
    ldr x21, [x9, #0x8]
    KAI_ASM_INST(0xe0bfa060)  // st1w { za0v.s[x13] }, p0/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0x25717120)  // psel p0.s, p12.s/Z, p9.s[w13, #1]
    KAI_ASM_INST(0xe0b0a864)  // st1w { za1v.s[x13] }, p2/Z, [x3, x16, LSL #2]
    whilelt p9.s, x11, x1
    incw x11
    KAI_ASM_INST(0xe0bca461)  // st1w { za0v.s[x13, #1] }, p1/Z, [x3, x28, LSL #2]
    add x9, x9, #0x10
    incw x22
    KAI_ASM_INST(0xe0baa065)  // st1w { za1v.s[x13, #1] }, p0/Z, [x3, x26, LSL #2]
    addvl x3, x3, #4
    whilelt p8.s, x11, x1
    cbz x14, label_7
KAI_ASM_LABEL(label_6)  // K loop: Main loop: Second: Loop
    KAI_ASM_INST(0x25306160)  // psel p0.s, p8.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25306142)  // psel p2.s, p8.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25706161)  // psel p1.s, p8.s/Z, p11.s[w12, #1]
    KAI_ASM_INST(0x25706143)  // psel p3.s, p8.s/Z, p10.s[w12, #1]
    KAI_ASM_INST(0xe0960360)  // ld1w { za0h.s[x12] }, p0/Z, [x27, x22, LSL #2]
    KAI_ASM_INST(0x25307120)  // psel p0.s, p12.s/Z, p9.s[w12]
    ldr x27, [x10, #0x0]
    KAI_ASM_INST(0xe0960b24)  // ld1w { za1h.s[x12] }, p2/Z, [x25, x22, LSL #2]
    KAI_ASM_INST(0x25307122)  // psel p2.s, p12.s/Z, p9.s[w12]
    ldr x25, [x9, #0x0]
    KAI_ASM_INST(0xe09606e1)  // ld1w { za0h.s[x12, #1] }, p1/Z, [x23, x22, LSL #2]
    KAI_ASM_INST(0x25707121)  // psel p1.s, p12.s/Z, p9.s[w12, #1]
    ldr x23, [x10, #0x8]
    add x10, x10, #0x10
    KAI_ASM_INST(0xe0960ea5)  // ld1w { za1h.s[x12, #1] }, p3/Z, [x21, x22, LSL #2]
    ldr x21, [x9, #0x8]
    KAI_ASM_INST(0xe0bf8068)  // st1w { za2v.s[x12] }, p0/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0x25707120)  // psel p0.s, p12.s/Z, p9.s[w12, #1]
    KAI_ASM_INST(0xe0b0886c)  // st1w { za3v.s[x12] }, p2/Z, [x3, x16, LSL #2]
    add x9, x9, #0x10
    KAI_ASM_INST(0xe0bc8469)  // st1w { za2v.s[x12, #1] }, p1/Z, [x3, x28, LSL #2]
    KAI_ASM_INST(0xe0ba806d)  // st1w { za3v.s[x12, #1] }, p0/Z, [x3, x26, LSL #2]
    add x12, x12, #0x2
    addvl x3, x3, #4
    cmp x12, x14
    blt label_6
KAI_ASM_LABEL(label_7)  // K loop: Main loop: Second: Tail
    KAI_ASM_INST(0x25306160)  // psel p0.s, p8.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25306142)  // psel p2.s, p8.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25706161)  // psel p1.s, p8.s/Z, p11.s[w12, #1]
    KAI_ASM_INST(0x25706143)  // psel p3.s, p8.s/Z, p10.s[w12, #1]
    mov x10, x2
    add x9, x2, x16, LSL #3
    KAI_ASM_INST(0xe0960360)  // ld1w { za0h.s[x12] }, p0/Z, [x27, x22, LSL #2]
    KAI_ASM_INST(0x25307120)  // psel p0.s, p12.s/Z, p9.s[w12]
    ldr x27, [x10, #0x0]
    KAI_ASM_INST(0xe0960b24)  // ld1w { za1h.s[x12] }, p2/Z, [x25, x22, LSL #2]
    KAI_ASM_INST(0x25307122)  // psel p2.s, p12.s/Z, p9.s[w12]
    ldr x25, [x9, #0x0]
    KAI_ASM_INST(0xe09606e1)  // ld1w { za0h.s[x12, #1] }, p1/Z, [x23, x22, LSL #2]
    KAI_ASM_INST(0x25707121)  // psel p1.s, p12.s/Z, p9.s[w12, #1]
    ldr x23, [x10, #0x8]
    add x10, x10, #0x10
    KAI_ASM_INST(0xe0960ea5)  // ld1w { za1h.s[x12, #1] }, p3/Z, [x21, x22, LSL #2]
    ldr x21, [x9, #0x8]
    KAI_ASM_INST(0xe0bf8068)  // st1w { za2v.s[x12] }, p0/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0x25707120)  // psel p0.s, p12.s/Z, p9.s[w12, #1]
    KAI_ASM_INST(0xe0b0886c)  // st1w { za3v.s[x12] }, p2/Z, [x3, x16, LSL #2]
    whilelt p9.s, x11, x1
    subs x20, x20, #0x1
    KAI_ASM_INST(0xe0bc8469)  // st1w { za2v.s[x12, #1] }, p1/Z, [x3, x28, LSL #2]
    add x9, x9, #0x10
    incw x11
    KAI_ASM_INST(0xe0ba806d)  // st1w { za3v.s[x12, #1] }, p0/Z, [x3, x26, LSL #2]
    addvl x3, x3, #4
    incw x22
    bgt label_3
KAI_ASM_LABEL(label_8)  // K loop: Tails
    cbnz x24, label_11
    mov x10, x2
    whilelt p8.s, x11, x1
    mov x12, #0x0
KAI_ASM_LABEL(label_9)  // K loop: Tails: Even: First
    KAI_ASM_INST(0x25307123)  // psel p3.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25307122)  // psel p2.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25306161)  // psel p1.s, p8.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25306140)  // psel p0.s, p8.s/Z, p10.s[w12]
    KAI_ASM_INST(0xe0bf8c60)  // st1w { za0v.s[x12] }, p3/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xe0b08864)  // st1w { za1v.s[x12] }, p2/Z, [x3, x16, LSL #2]
    addvl x3, x3, #2
    ldr x21, [x10, #0x0]
    ldr x20, [x10, x16, LSL #0x3]
    add x10, x10, #0x8
    KAI_ASM_INST(0xe09606a8)  // ld1w { za2h.s[x12] }, p1/Z, [x21, x22, LSL #2]
    KAI_ASM_INST(0xe096028c)  // ld1w { za3h.s[x12] }, p0/Z, [x20, x22, LSL #2]
    add x12, x12, #0x1
    cmp x12, x16
    blt label_9
    whilelt p9.s, x11, x1
    whilelt p8.s, x11, x1
    mov x12, #0x0
KAI_ASM_LABEL(label_10)  // K loop: Tails: Even: Second
    KAI_ASM_INST(0x25307121)  // psel p1.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25307120)  // psel p0.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0xe0bf8468)  // st1w { za2v.s[x12] }, p1/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xe0b0806c)  // st1w { za3v.s[x12] }, p0/Z, [x3, x16, LSL #2]
    add x12, x12, #0x1
    addvl x3, x3, #2
    cmp x12, x15
    blt label_10
    whilelt p8.s, x11, x1
    b label_13
KAI_ASM_LABEL(label_11)  // K loop: Tails: Odd
    mov x12, #0x0
KAI_ASM_LABEL(label_12)  // K loop: Tails: Odd: Loop
    KAI_ASM_INST(0x25307121)  // psel p1.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25307120)  // psel p0.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0xe0bf8460)  // st1w { za0v.s[x12] }, p1/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xe0b08064)  // st1w { za1v.s[x12] }, p0/Z, [x3, x16, LSL #2]
    add x12, x12, #0x1
    addvl x3, x3, #2
    cmp x12, x15
    blt label_12
KAI_ASM_LABEL(label_13)  // K loop: End
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp d8, d9, [sp, 72]
    ldp d10, d11, [sp, 88]
    ldp d12, d13, [sp, 104]
    ldp d14, d15, [sp, 120]
    ldp x20, x21, [sp], 144
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_lhs_pack_f32p2vlx1_f32_sme)

    KAI_ASM_END
