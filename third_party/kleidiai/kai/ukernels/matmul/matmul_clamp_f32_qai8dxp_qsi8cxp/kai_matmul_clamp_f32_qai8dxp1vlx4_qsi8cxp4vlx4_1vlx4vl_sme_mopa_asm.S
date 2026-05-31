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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa)
    stp     x19, x20, [sp, -112]!
    stp     x24, x25, [sp, 16]
    str     x28,      [sp, 32]
    stp     d8, d9,   [sp, 48]
    stp     d10, d11, [sp, 64]
    stp     d12, d13, [sp, 80]
    stp     d14, d15, [sp, 96]
    KAI_ASM_INST(0xd503477f)	//     	smstart
    ldr	x19, [x0]               //      dst
    ldr	x20, [x0, #0x8]         //      lhs_packed
    cntw	x7
    ptrue	p2.b
    KAI_ASM_INST(0x8556c81e) // ld1rw { z30.s }, p2/z, [x0, #0x58] // clamp_min
    KAI_ASM_INST(0x8557c81f) // ld1rw { z31.s }, p2/z, [x0, #0x5c] // clamp_max
    ldr	x12, [x0, #0x20]        //      m
    KAI_ASM_INST(0x25ac17e0)	//     	whilelt	p0.s, xzr, x12
KAI_ASM_LABEL(label_1)          //      Row Loop
    ldr	x8, [x0, #0x10]         //      rhs_packed
    mov	x9, x19
    ldr	x13, [x0, #0x28]        //      n
    cmp	x7, x12
    csel	x16, x7, x12, lt
    lsl	x16, x16, #2
    ldr	x24, [x0, #0x40]        //      rhs_row_bytes
    add	x24, x24, x8
    mov	x11, x8
    whilelt p3.h, x11, x24
    addvl	x11, x8, #0x2
    whilelt p4.h, x11, x24
KAI_ASM_LABEL(label_2)          //      Column Loop
    mov	x10, x20
    mov	x11, x8
    mov	x17, x9
    whilelt p5.s, xzr, x13
    KAI_ASM_INST(0xc00800ff)	//     	zero	{za}
    ldr	x24, [x0, #0x48]        //      m_blk
    add	x14, x10, x24
KAI_ASM_LABEL(label_3)          //      Block Loop
    ld1w	{ z4.s }, p0/z, [x10]
    addvl	x10, x10, #0x1
    ld1h z8.h, p3/z, [x11]
    ld1h z9.h, p3/z, [x11, #1, mul vl]
    KAI_ASM_INST(0xa0884880)	//     	smopa	za0.s, p2/m, p2/m, z4.b, z8.b
    KAI_ASM_INST(0xa0894881)	//     	smopa	za1.s, p2/m, p2/m, z4.b, z9.b
    ld1h z10.h, p4/z, [x11, #2, mul vl]
    ld1h z11.h, p4/z, [x11, #3, mul vl]
    KAI_ASM_INST(0xa08a4882)	//     	smopa	za2.s, p2/m, p2/m, z4.b, z10.b
    KAI_ASM_INST(0xa08b4883)	//     	smopa	za3.s, p2/m, p2/m, z4.b, z11.b
    addvl	x11, x11, #0x4
    cmp	x10, x14
    b.lt	label_3
    ld1w z0.s, p5/z, [x11]
    ld1w z1.s, p5/z, [x11, #1, mul vl]
    ld1w z2.s, p5/z, [x11, #2, mul vl]
    ld1w z3.s, p5/z, [x11, #3, mul vl]
    ld1w z4.s, p5/z, [x11, #4, mul vl]
    ld1w z5.s, p5/z, [x11, #5, mul vl]
    ld1w z6.s, p5/z, [x11, #6, mul vl]
    ld1w z7.s, p5/z, [x11, #7, mul vl]
    addvl x11, x11, #0x8
    ld1w z8.s, p5/z, [x11]
    ld1w z9.s, p5/z, [x11, #1, mul vl]
    ld1w z10.s, p5/z, [x11, #2, mul vl]
    ld1w z11.s, p5/z, [x11, #3, mul vl]
    addvl	x11, x11, #0x4
    mov	x14, #0x0               // =0
    addvl	x15, x10, #0x1
KAI_ASM_LABEL(label_4)
    KAI_ASM_INST(0x8540c950) // ld1rw { z16.s }, p2/z, [x10]
    KAI_ASM_INST(0x8540c9f1) // ld1rw { z17.s }, p2/z, [x15]
    add	x10, x10, #0x4
    add	x15, x15, #0x4
    KAI_ASM_INST(0xc002480c)	//    mov z12.b, p2/m, za0h.b[w14, 0]
    KAI_ASM_INST(0xc002482d)	//    mov z13.b, p2/m, za0h.b[w14, 1]
    KAI_ASM_INST(0xc002484e)	//    mov z14.b, p2/m, za0h.b[w14, 2]
    KAI_ASM_INST(0xc002486f)	//    mov z15.b, p2/m, za0h.b[w14, 3]
    fmul	z20.s, z17.s, z4.s
    fmul	z21.s, z17.s, z5.s
    fmul	z22.s, z17.s, z6.s
    fmul	z23.s, z17.s, z7.s
    mla	    z12.s, p2/m, z16.s, z0.s
    mla	    z13.s, p2/m, z16.s, z1.s
    mla	    z14.s, p2/m, z16.s, z2.s
    mla	    z15.s, p2/m, z16.s, z3.s
    scvtf z12.s, p2/m, z12.s
    scvtf z13.s, p2/m, z13.s
    scvtf z14.s, p2/m, z14.s
    scvtf z15.s, p2/m, z15.s
    fmul	z24.s, z20.s, z12.s
    fmul	z25.s, z21.s, z13.s
    fmul	z26.s, z22.s, z14.s
    fmul	z27.s, z23.s, z15.s
    fadd	z24.s, p2/m, z24.s, z8.s
    fadd	z25.s, p2/m, z25.s, z9.s
    fadd	z26.s, p2/m, z26.s, z10.s
    fadd	z27.s, p2/m, z27.s, z11.s
    fmax z24.s, p2/M, z24.s, z30.s
    fmin z24.s, p2/M, z24.s, z31.s
    fmax z25.s, p2/M, z25.s, z30.s
    fmin z25.s, p2/M, z25.s, z31.s
    fmax z26.s, p2/M, z26.s, z30.s
    fmin z26.s, p2/M, z26.s, z31.s
    fmax z27.s, p2/M, z27.s, z30.s
    fmin z27.s, p2/M, z27.s, z31.s
    mov     x28, xzr
    whilelt p5.s, x28, x13
    st1w    z24.s, p5, [x17]
    add     x28, x28, x7
    whilelt p5.s, x28, x13
    st1w    z25.s, p5, [x17, #1, mul vl]
    add     x28, x28, x7
    whilelt p5.s, x28, x13
    st1w    z26.s, p5, [x17, #2, mul vl]
    add     x28, x28, x7
    whilelt p5.s, x28, x13
    st1w    z27.s, p5, [x17, #3, mul vl]
    ldr	x24, [x0, #0x18]        //      dst_stride_row
    add	x17, x17, x24
    add	x14, x14, #0x4
    cmp	x14, x16
    b.lt label_4
    ldr	x24, [x0, #0x38]        //      rhs_stride
    add	x8, x8, x24
    addvl	x9, x9, #0x4
    ldr	x24, [x0, #0x40]        //      rhs_row_bytes
    add	x24, x24, x8
    mov	x11, x8
    whilelt p3.h, x11, x24
    addvl	x11, x8, #0x2
    whilelt p4.h, x11, x24
    cntw	x24, ALL, MUL #0x4
    sub	x13, x13, x24
    subs xzr, xzr, x13       // cmp xzr, x13
    b.mi	label_2
    ldr	x24, [x0, #0x30]
    add	x20, x20, x24
    ldr	x24, [x0, #0x50]
    add	x19, x19, x24
    cntw	x24
    sub	x12, x12, x24
    whilelt	p0.s, xzr, x12
    b.mi	label_1
    KAI_ASM_INST(0xd503467f)	//     	smstop
    ldp     d14, d15, [sp, 96]
    ldp     d12, d13, [sp, 80]
    ldp     d10, d11, [sp, 64]
    ldp     d8,  d9,  [sp, 48]
    ldr     x28,      [sp, 32]
    ldp     x24, x25, [sp, 16]
    ldp     x19, x20, [sp], 112
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa)

    KAI_ASM_END
